"""
RankFT (function-level): Ranking Fine-Tuning for function-level bug
localization reranking. Forked from train_rankft.py.

Key differences vs. file-level train_rankft.py:
  - Training data comes from a pre-mined function pair file
    (train_pairs.jsonl) whose schema is
      {query_id, query_text, repo, base_commit, pos_ids, neg_ids,
       gt_missing, shard_n_funcs}
    pos_ids / neg_ids are function ids of the form
      "<rel_path.py>/<Class>/<method>" or "<rel_path.py>/<func>".
  - Function bodies are read on-the-fly from per-shard corpus JSONL files at
    {corpus_dir}/<repo>__<commit[:12]>.jsonl whose records follow the
    SweRank byte-exact schema
      {"_id": "<func_id>", "title": "", "text": "<func_id>\\n<body>",
       "metadata": {}}
    i.e. body = text.split("\\n", 1)[1] (we strip the leading _id line).
  - Prompt is function-level, not file-level:
      "Given the bug report, is this function likely to need modification?
       Bug Report: {issue}
       Function: {func_id}
       Code:\\n{body}
       Answer:"
  - path_augment_fraction now shuffles the function id (entire path+Class+method
    string) across the pos+neg group while each body stays with its original
    function; this breaks any lexical shortcut via the id tokens.
  - No NegativeSampler / BM25 / graph / delex logic: negatives are already
    mined upstream into train_pairs.jsonl (200 per query). We randomly sample
    max_negatives of them per step to control memory.

Kept identical to train_rankft.py (minimal-risk regression):
  - LoRA config (rank, alpha, target_modules)
  - Optimizer (AdamW), cosine schedule with warmup
  - Listwise group-softmax loss (score = logit(Yes) - logit(No))
  - Gradient accumulation, checkpoint / best-tracking / resume
  - Diagnostics JSONL writer (pos/neg score gap per step + 50-step summary)

Usage (fresh LoRA):
    python src/train/train_rankft_function.py \\
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \\
        --train_pairs_path /data/chenlibin/codegrip_func/train_pairs.jsonl \\
        --corpus_dir /data/chenlibin/codegrip_func/corpus \\
        --output_dir /data/chenlibin/grepo_agent_experiments/func_codeaware \\
        --device cuda:0

Usage (resume from Run 2 file-level LoRA, optional):
    python src/train/train_rankft_function.py \\
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \\
        --lora_path experiments/rankft_runB_graph/best \\
        --train_pairs_path /data/chenlibin/codegrip_func/train_pairs.jsonl \\
        --corpus_dir /data/chenlibin/codegrip_func/corpus \\
        --output_dir /data/chenlibin/grepo_agent_experiments/func_codeaware_fromB \\
        --device cuda:0
"""

import os
import json
import argparse
import random
import math
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

try:
    from rankft_training_utils import save_final_and_best_adapters
except ImportError:
    from src.train.rankft_training_utils import save_final_and_best_adapters

# Deterministic seeds (re-seeded in train() from args.seed as well)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================
# Function-level prompt template
# ============================================================

PROMPT_TEMPLATE = (
    "Given the bug report, is this function likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "Function: {func_id}\n\n"
    "Code:\n{body}\n\n"
    "Answer:"
)


def build_function_prompt(issue_text: str, func_id: str, body: str,
                          max_issue_chars: int, max_body_chars: int) -> str:
    """Build the scoring prompt for a single (issue, function) pair.

    Hard character caps are applied BEFORE string interpolation so we know an
    upper bound on prompt size regardless of tokenizer behaviour; the
    tokenizer's max_length truncation still catches any tail overflow.
    """
    if max_issue_chars and max_issue_chars > 0:
        issue_text = issue_text[:max_issue_chars]
    if max_body_chars and max_body_chars > 0:
        body = body[:max_body_chars]
    return PROMPT_TEMPLATE.format(
        issue_text=issue_text,
        func_id=func_id,
        body=body,
    )


# ============================================================
# Corpus shard cache (lazy per-shard load)
# ============================================================

# Tripwire counters (mirrors train_rankft.py CODE_READ_STATS semantics).
CODE_READ_STATS = {"total": 0, "missing": 0, "missing_shards": 0}


class ShardBodyCache:
    """Lazily loads per-(repo, commit) corpus shards and returns function body
    strings by func_id.

    Shard path: {corpus_dir}/{repo}__{commit[:12]}.jsonl
    Each line's `text` field starts with the _id on its own line, followed by
    the body; we strip that first line to return only the body. If the shard
    file is missing we remember it so we don't retry repeatedly, and we count
    every lookup against it as a CodeMiss.
    """

    def __init__(self, corpus_dir: str):
        self.corpus_dir = corpus_dir
        # shard_key -> {func_id: body_str} (empty dict means shard missing / empty)
        self._cache: Dict[str, Dict[str, str]] = {}

    @staticmethod
    def _shard_key(repo: str, base_commit: str) -> str:
        return f"{repo}__{base_commit[:12]}"

    def _load_shard(self, shard_key: str) -> Dict[str, str]:
        shard_path = os.path.join(self.corpus_dir, f"{shard_key}.jsonl")
        if not os.path.isfile(shard_path):
            CODE_READ_STATS["missing_shards"] += 1
            self._cache[shard_key] = {}
            return self._cache[shard_key]
        shard: Dict[str, str] = {}
        with open(shard_path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fid = obj.get("_id")
                text = obj.get("text", "")
                if not fid or not text:
                    continue
                # text = "<func_id>\n<body>" per SweRank byte-exact schema.
                split = text.split("\n", 1)
                body = split[1] if len(split) == 2 else ""
                shard[fid] = body
        self._cache[shard_key] = shard
        return shard

    def get_body(self, repo: str, base_commit: str, func_id: str) -> str:
        shard_key = self._shard_key(repo, base_commit)
        shard = self._cache.get(shard_key)
        if shard is None:
            shard = self._load_shard(shard_key)
        CODE_READ_STATS["total"] += 1
        body = shard.get(func_id)
        if body is None:
            CODE_READ_STATS["missing"] += 1
            return "# (function body not available)"
        return body


# ============================================================
# Data loading
# ============================================================

def load_function_train_pairs(train_pairs_path: str) -> List[Dict]:
    """Load function-level training pairs (one record per query).

    Returns a list of dicts with keys:
      query_id, query_text, repo, base_commit, pos_ids, neg_ids.
    Drops records with empty pos_ids (can't form a positive) or empty neg_ids
    (can't form a contrastive group).
    """
    data: List[Dict] = []
    dropped_no_pos = 0
    dropped_no_neg = 0
    with open(train_pairs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not item.get("pos_ids"):
                dropped_no_pos += 1
                continue
            if not item.get("neg_ids"):
                dropped_no_neg += 1
                continue
            data.append({
                "query_id": item["query_id"],
                "query_text": item["query_text"],
                "repo": item["repo"],
                "base_commit": item["base_commit"],
                "pos_ids": list(item["pos_ids"]),
                "neg_ids": list(item["neg_ids"]),
            })
    print(f"  Loaded {len(data)} training pairs from {train_pairs_path} "
          f"(dropped {dropped_no_pos} empty-pos, {dropped_no_neg} empty-neg)")
    return data


# ============================================================
# Score extraction (identical semantics to train_rankft.py)
# ============================================================

def get_yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    yes_id = yes_ids[0]
    no_id = no_ids[0]
    print(f"  Yes token ID: {yes_id} (decoded: '{tokenizer.decode([yes_id])}')")
    print(f"  No token ID: {no_id} (decoded: '{tokenizer.decode([no_id])}')")
    return yes_id, no_id


def compute_scores(
    model,
    tokenizer,
    prompts: List[str],
    yes_id: int,
    no_id: int,
    max_seq_length: int,
    device: str,
    mini_batch_size: int = 4,
) -> torch.Tensor:
    """Forward pass: score = logit(Yes) - logit(No) for each prompt.

    Processes prompts in mini-batches to avoid OOM. Returns tensor of shape
    (len(prompts),) on the model's device.
    """
    all_scores = []
    for i in range(0, len(prompts), mini_batch_size):
        batch_prompts = prompts[i : i + mini_batch_size]
        encodings = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.size(0), device=device)
        last_logits = logits[batch_indices, seq_lengths]
        scores = last_logits[:, yes_id] - last_logits[:, no_id]
        all_scores.append(scores)
        del outputs, logits, input_ids, attention_mask
        torch.cuda.empty_cache()
    return torch.cat(all_scores, dim=0)


# ============================================================
# Training loop
# ============================================================

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Re-seed with args.seed so different seeds produce different runs
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    device = args.device

    # ---- Tokenizer ----
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # critical for batch scoring with causal LM

    yes_id, no_id = get_yes_no_token_ids(tokenizer)

    # ---- Base model ----
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    # ---- Attach or load LoRA ----
    if args.lora_path:
        print(f"Loading existing LoRA adapter from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=True)
    else:
        print(f"Creating fresh LoRA adapter (rank={args.lora_rank})...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.train()

    # ---- Training data ----
    print("Loading training pairs...")
    train_data = load_function_train_pairs(args.train_pairs_path)
    if len(train_data) == 0:
        print("ERROR: no training pairs after filtering. Exiting.")
        return

    print(f"Initializing corpus shard cache (lazy) from {args.corpus_dir}...")
    shard_cache = ShardBodyCache(args.corpus_dir)

    # ---- Optimizer and scheduler ----
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    total_examples = len(train_data)
    steps_per_epoch = math.ceil(total_examples / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * 0.05)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n{'='*60}")
    print(f"RankFT Training (FUNCTION-LEVEL)")
    print(f"  Examples: {total_examples}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Max negatives per positive: {args.max_negatives}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}")
    print(f"  LR: {args.learning_rate}, Warmup: {warmup_steps}")
    print(f"  Max seq length: {args.max_seq_length}")
    print(f"  Max issue chars: {args.max_issue_chars}, "
          f"max body chars: {args.max_body_chars}")
    print(f"  Path augment fraction: {args.path_augment_fraction}")
    if args.lora_path:
        print(f"  Initialized from: {args.lora_path}")
    print(f"{'='*60}\n")

    global_step = 0
    accumulated_loss = 0.0
    log_losses: List[float] = []
    best_loss = float("inf")
    resume_from_epoch = 0
    resume_from_example = 0
    start_time = time.time()

    # --- Resume from checkpoint ---
    if getattr(args, "resume", False):
        import glob as _glob
        ckpt_dirs = sorted(
            [d for d in _glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
             if os.path.isfile(os.path.join(d, "training_state.json"))],
            key=lambda d: int(os.path.basename(d).split("-")[1])
        )
        if ckpt_dirs:
            latest_ckpt = ckpt_dirs[-1]
            print(f"\n  Resuming from {latest_ckpt}")
            with open(os.path.join(latest_ckpt, "training_state.json")) as f:
                state = json.load(f)
            global_step = state["global_step"]
            log_losses = state.get("loss_history", [])
            best_loss = state.get("best_loss", float("inf"))
            resume_from_epoch = state["epoch"]
            resume_from_example = (global_step % steps_per_epoch) * args.gradient_accumulation_steps
            model = PeftModel.from_pretrained(
                model.base_model.model if hasattr(model, "base_model") else model,
                latest_ckpt, is_trainable=True,
            )
            model.to(args.device)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.learning_rate, weight_decay=0.01,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            opt_path = os.path.join(latest_ckpt, "optimizer_scheduler.pt")
            if os.path.exists(opt_path):
                opt_state = torch.load(opt_path, map_location=args.device)
                optimizer.load_state_dict(opt_state["optimizer"])
                scheduler.load_state_dict(opt_state["scheduler"])
                print(f"  Restored optimizer/scheduler state")
            print(f"  Resumed: step={global_step}, epoch={resume_from_epoch}, "
                  f"best_loss={best_loss:.4f}, {len(log_losses)} loss entries")
        else:
            print("  --resume specified but no checkpoints found, starting fresh")

    # --- Diagnostics ---
    diag_path = os.path.join(args.output_dir, "training_diagnostics.jsonl")
    diag_file = open(diag_path, "a" if global_step > 0 else "w")
    diag_pos_scores: List[float] = []
    diag_neg_scores: List[float] = []

    ex_idx = 0  # visible after loop for "remainder" handling if train_data empty-index-safe
    for epoch in range(args.num_epochs):
        if epoch < resume_from_epoch:
            continue

        indices = list(range(total_examples))
        random.shuffle(indices)

        optimizer.zero_grad()

        for ex_idx, data_idx in enumerate(indices):
            if epoch == resume_from_epoch and ex_idx < resume_from_example:
                continue

            example = train_data[data_idx]
            repo = example["repo"]
            base_commit = example["base_commit"]
            query_text = example["query_text"]
            pos_ids: List[str] = example["pos_ids"]
            neg_ids: List[str] = example["neg_ids"]

            if not pos_ids or not neg_ids:
                continue

            # Pick one positive; sample up to max_negatives from neg_ids.
            positive_id = random.choice(pos_ids)
            if len(neg_ids) > args.max_negatives:
                sampled_negs = random.sample(neg_ids, args.max_negatives)
            else:
                sampled_negs = list(neg_ids)

            candidates = [positive_id] + sampled_negs

            # Fetch bodies BEFORE any id shuffling so each body tracks its
            # original function.
            bodies = [
                shard_cache.get_body(repo, base_commit, fid)
                for fid in candidates
            ]

            # Consume one RNG draw for the augment decision regardless, so the
            # RNG stream is identical across augment=0 and augment=0.5 ablations
            # (same invariant as train_rankft.py).
            _aug_roll = random.random()
            shuffle_now = (args.path_augment_fraction > 0
                           and _aug_roll < args.path_augment_fraction)
            if shuffle_now:
                display_ids = list(candidates)
                random.shuffle(display_ids)
            else:
                display_ids = candidates

            prompts = [
                build_function_prompt(
                    query_text, display_ids[j], bodies[j],
                    args.max_issue_chars, args.max_body_chars,
                )
                for j in range(len(candidates))
            ]

            try:
                scores = compute_scores(
                    model, tokenizer, prompts,
                    yes_id, no_id,
                    args.max_seq_length, device,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at example {ex_idx}, skipping. "
                          f"Group size: {len(prompts)}")
                    torch.cuda.empty_cache()
                    continue
                raise

            # Listwise loss: -log_softmax(scores)[0]
            log_probs = F.log_softmax(scores, dim=0)
            loss = -log_probs[0]

            with torch.no_grad():
                s = scores.detach().float()
                diag_pos_scores.append(s[0].item())
                for ns in s[1:].tolist():
                    diag_neg_scores.append(ns)

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            accumulated_loss += loss.item()

            if (ex_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                step_loss = accumulated_loss
                log_losses.append(step_loss)
                accumulated_loss = 0.0

                if diag_pos_scores:
                    pos_arr = np.array(diag_pos_scores)
                    neg_arr = np.array(diag_neg_scores) if diag_neg_scores else np.array([0.0])
                    diag_record = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": step_loss,
                        "pos_score_mean": float(pos_arr.mean()),
                        "neg_score_mean": float(neg_arr.mean()),
                        "score_gap": float(pos_arr.mean() - neg_arr.mean()),
                        "pos_score_min": float(pos_arr.min()),
                        "pos_score_max": float(pos_arr.max()),
                        "neg_score_min": float(neg_arr.min()),
                        "neg_score_max": float(neg_arr.max()),
                        "code_miss_total": CODE_READ_STATS["total"],
                        "code_miss_count": CODE_READ_STATS["missing"],
                    }
                    diag_file.write(json.dumps(diag_record) + "\n")
                    diag_file.flush()
                diag_pos_scores.clear()
                diag_neg_scores.clear()

                # Per-50-step summary
                if global_step % 50 == 0:
                    recent_pos, recent_neg, recent_gap = [], [], []
                    diag_file.flush()
                    try:
                        with open(diag_path) as _df:
                            lines = _df.readlines()
                        for line in lines[-50:]:
                            rec = json.loads(line)
                            recent_pos.append(rec["pos_score_mean"])
                            recent_neg.append(rec["neg_score_mean"])
                            recent_gap.append(rec["score_gap"])
                    except Exception:
                        pass
                    if recent_pos:
                        print(f"  [Diagnostics @ step {global_step}]")
                        print(f"    Pos score:  mean={np.mean(recent_pos):.3f}  "
                              f"min={np.min(recent_pos):.3f}  max={np.max(recent_pos):.3f}")
                        print(f"    Neg score:  mean={np.mean(recent_neg):.3f}  "
                              f"min={np.min(recent_neg):.3f}  max={np.max(recent_neg):.3f}")
                        print(f"    Score gap:  mean={np.mean(recent_gap):.3f}")

                # Logging with CodeMiss tripwire
                if global_step % args.logging_steps == 0:
                    avg_recent = np.mean(log_losses[-args.logging_steps:])
                    elapsed = time.time() - start_time
                    current_lr = scheduler.get_last_lr()[0]
                    miss_frac = (CODE_READ_STATS["missing"] / max(1, CODE_READ_STATS["total"]))
                    print(
                        f"  [Epoch {epoch+1}/{args.num_epochs}] "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {step_loss:.4f} (avg: {avg_recent:.4f}) | "
                        f"LR: {current_lr:.2e} | "
                        f"Time: {elapsed:.0f}s | "
                        f"CodeMiss: {CODE_READ_STATS['missing']}/{CODE_READ_STATS['total']} "
                        f"({miss_frac*100:.1f}%) | "
                        f"MissingShards: {CODE_READ_STATS['missing_shards']}"
                    )
                    if (CODE_READ_STATS["total"] >= 200
                            and miss_frac > 0.05):
                        raise RuntimeError(
                            f"Missing-function rate {miss_frac*100:.1f}% > 5% "
                            f"threshold. Likely corpus_dir / shard naming mismatch "
                            f"(corpus_dir={args.corpus_dir}). Aborting to avoid "
                            f"training on placeholder code."
                        )

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    print(f"  Saving checkpoint to {ckpt_dir}...")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

                    state = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "loss_history": log_losses,
                        "best_loss": best_loss,
                    }
                    with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
                        json.dump(state, f, indent=2)
                    torch.save({
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }, os.path.join(ckpt_dir, "optimizer_scheduler.pt"))

                    recent_avg = np.mean(log_losses[-min(50, len(log_losses)):])
                    if recent_avg < best_loss:
                        best_loss = recent_avg
                        best_dir = os.path.join(args.output_dir, "best")
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
                        print(f"    New best (avg loss: {best_loss:.4f})")

        # Handle remaining accumulated gradients at end of epoch
        if (ex_idx + 1) % args.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        print(f"\n  Epoch {epoch+1} complete. "
              f"Avg loss: {np.mean(log_losses[-steps_per_epoch:]):.4f}")

    diag_file.close()
    print(f"\n  Training diagnostics saved to {diag_path}")

    recent_avg = np.mean(log_losses[-min(50, len(log_losses)):]) if log_losses else float("inf")
    if recent_avg < best_loss:
        best_loss = recent_avg
        best_dir = os.path.join(args.output_dir, "best")
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        print(f"  End-of-training best update (avg loss: {best_loss:.4f})")

    print(f"\nSaving final model under {args.output_dir}...")
    final_dir, best_dir, best_backfilled = save_final_and_best_adapters(
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
    )
    if best_backfilled:
        print(f"  Backfilled missing best adapter at {best_dir}")

    with open(os.path.join(args.output_dir, "loss_history.json"), "w") as f:
        json.dump({"losses": log_losses, "total_steps": global_step}, f, indent=2)

    total_time = time.time() - start_time
    final_miss_frac = CODE_READ_STATS["missing"] / max(1, CODE_READ_STATS["total"])
    print(f"\nTraining complete!")
    print(f"  Total steps: {global_step}")
    if log_losses:
        print(f"  Final avg loss: {np.mean(log_losses[-50:]):.4f}")
    print(f"  CodeMiss total: {CODE_READ_STATS['missing']}/{CODE_READ_STATS['total']} "
          f"({final_miss_frac*100:.2f}%), "
          f"missing shards: {CODE_READ_STATS['missing_shards']}")
    print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Output: {args.output_dir}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RankFT (function-level): ranking fine-tuning with "
                    "function bodies as context"
    )

    # Model
    parser.add_argument("--model_path", required=True,
                        help="Path to base model (e.g., Qwen2.5-7B-Instruct)")
    parser.add_argument("--lora_path", default=None,
                        help="Optional: start from existing LoRA checkpoint "
                             "(e.g. experiments/rankft_runB_graph/best to "
                             "resume from Run 2's file-level reranker)")

    # Data
    parser.add_argument("--train_pairs_path", required=True,
                        help="Function-level training pairs JSONL "
                             "(one record per query with pos_ids, neg_ids).")
    parser.add_argument("--corpus_dir", required=True,
                        help="Directory of per-(repo,commit) corpus shards, "
                             "one JSONL per shard at {repo}__{commit[:12]}.jsonl "
                             "with the SweRank byte-exact schema.")

    # Output
    parser.add_argument("--output_dir", required=True,
                        help="Where to save checkpoints")

    # Hardware
    parser.add_argument("--device", default="cuda:0")

    # Group size
    parser.add_argument("--max_negatives", type=int, default=8,
                        help="Max negatives sampled per positive per step "
                             "(from the 200 pre-mined negs in train_pairs).")

    # Prompt truncation
    parser.add_argument("--max_issue_chars", type=int, default=1500,
                        help="Truncate query_text to first N characters before "
                             "insertion into the prompt (~375 tokens @ 4 c/t).")
    parser.add_argument("--max_body_chars", type=int, default=3000,
                        help="Truncate function body to first N characters "
                             "before insertion into the prompt (~750 tokens).")

    # Code-aware / path augmentation
    parser.add_argument("--path_augment_fraction", type=float, default=0.5,
                        help="Fraction of steps where function ids are shuffled "
                             "across the pos+neg group while bodies stay with "
                             "their original function. 0 disables.")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Micro-batch size (groups per step, typically 1)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_seq_length", type=int, default=1536,
                        help="Max tokens per (issue, function) prompt. 1536 "
                             "recommended to fit issue+body without excessive "
                             "tail truncation.")

    # LoRA config
    parser.add_argument("--lora_rank", type=int, default=32)

    # Resume
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint in output_dir")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not 0.0 <= args.path_augment_fraction <= 1.0:
        parser.error(
            f"--path_augment_fraction must be in [0,1], got {args.path_augment_fraction}"
        )
    if args.max_negatives < 1:
        parser.error(f"--max_negatives must be >= 1, got {args.max_negatives}")

    train(args)


if __name__ == "__main__":
    main()
