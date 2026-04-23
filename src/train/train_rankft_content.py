"""
RankFT-Content: Content-aware Ranking Fine-Tuning.

Extension of RankFT that includes file content summaries in the prompt.
This is the "relaxed constraint" version — the model sees:
  - Issue text
  - File path
  - File summary (class names, function names, docstring excerpt)

The summary provides semantic signal about what each file contains,
allowing the model to match issue descriptions to file functionality.

Usage:
    python src/train/train_rankft_content.py \
        --model_path /path/to/Qwen2.5-7B \
        --train_data data/grepo_text/grepo_train.jsonl \
        --bm25_candidates data/rankft/grepo_train_bm25_top500.jsonl \
        --file_summaries data/file_summaries_all.json \
        --output_dir experiments/rankft_content_v1 \
        --device cuda:0
"""

import os
import json
import argparse
import random
import math
import time
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Deterministic seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================
# Prompt template (content-aware version)
# ============================================================

PROMPT_TEMPLATE_CONTENT = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n"
    "Content: {file_summary}\n\n"
    "Answer:"
)

PROMPT_TEMPLATE_PATH_ONLY = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)


def build_prompt(
    issue_text: str,
    candidate_path: str,
    file_summary: Optional[str] = None,
) -> str:
    """Build the scoring prompt for a single (issue, file) pair."""
    if file_summary:
        return PROMPT_TEMPLATE_CONTENT.format(
            issue_text=issue_text,
            candidate_path=candidate_path,
            file_summary=file_summary,
        )
    else:
        return PROMPT_TEMPLATE_PATH_ONLY.format(
            issue_text=issue_text,
            candidate_path=candidate_path,
        )


# ============================================================
# Negative mining (reused from train_rankft.py)
# ============================================================

def build_cochange_index(
    train_data_path: str,
    min_cochange: int = 1,
) -> Dict[str, Dict[str, Set[str]]]:
    repo_pairs: Dict[str, Counter] = defaultdict(Counter)
    with open(train_data_path) as f:
        for line in f:
            item = json.loads(line)
            repo = item["repo"]
            files = sorted(item.get("changed_py_files", []))
            for i, fa in enumerate(files):
                for fb in files[i + 1:]:
                    repo_pairs[repo][(fa, fb)] += 1

    index: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for repo, pairs in repo_pairs.items():
        for (fa, fb), count in pairs.items():
            if count >= min_cochange:
                index[repo][fa].add(fb)
                index[repo][fb].add(fa)
    return index


def build_import_adjacency(
    dep_graph_dir: str,
) -> Dict[str, Dict[str, Set[str]]]:
    result: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    if not os.path.isdir(dep_graph_dir):
        return result
    for fname in os.listdir(dep_graph_dir):
        if not fname.endswith("_rels.json"):
            continue
        repo = fname.replace("_rels.json", "")
        with open(os.path.join(dep_graph_dir, fname)) as f:
            rels = json.load(f)
        for importer, imported_list in rels.get("file_imports", {}).items():
            for imported in imported_list:
                if importer.endswith(".py") and imported.endswith(".py"):
                    result[repo][importer].add(imported)
                    result[repo][imported].add(importer)
    return result


def load_file_trees(file_tree_dir: str) -> Dict[str, List[str]]:
    repo_files: Dict[str, List[str]] = {}
    if not os.path.isdir(file_tree_dir):
        return repo_files
    for fname in os.listdir(file_tree_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(file_tree_dir, fname)) as f:
            tree = json.load(f)
        repo = tree.get("repo", fname.replace(".json", ""))
        repo_files[repo] = tree.get("py_files", [])
    return repo_files


class NegativeSampler:
    def __init__(
        self,
        bm25_candidates: Dict[str, Dict],
        cochange_index: Dict[str, Dict[str, Set[str]]],
        import_index: Dict[str, Dict[str, Set[str]]],
        repo_files: Dict[str, List[str]],
        neg_bm25_ratio: float = 0.5,
        neg_graph_ratio: float = 0.25,
        neg_random_ratio: float = 0.25,
    ):
        self.bm25_candidates = bm25_candidates
        self.cochange_index = cochange_index
        self.import_index = import_index
        self.repo_files = repo_files
        self.neg_bm25_ratio = neg_bm25_ratio
        self.neg_graph_ratio = neg_graph_ratio
        self.neg_random_ratio = neg_random_ratio

    def sample(self, repo, issue_id, gt_files, num_negatives):
        n_bm25 = int(round(num_negatives * self.neg_bm25_ratio))
        n_graph = int(round(num_negatives * self.neg_graph_ratio))
        n_random = num_negatives - n_bm25 - n_graph

        selected: Set[str] = set()
        result: List[str] = []

        bm25_key = f"{repo}_{issue_id}"
        bm25_cands = self.bm25_candidates.get(bm25_key, {}).get("candidates", [])
        bm25_negs = [c for c in bm25_cands if c not in gt_files]
        for f in bm25_negs[:n_bm25]:
            if f not in selected:
                selected.add(f)
                result.append(f)

        graph_pool: Set[str] = set()
        repo_cochange = self.cochange_index.get(repo, {})
        repo_imports = self.import_index.get(repo, {})
        for gt_f in gt_files:
            graph_pool.update(repo_cochange.get(gt_f, set()))
            graph_pool.update(repo_imports.get(gt_f, set()))
        graph_pool -= gt_files
        graph_pool -= selected
        graph_list = list(graph_pool)
        random.shuffle(graph_list)
        for f in graph_list[:n_graph]:
            selected.add(f)
            result.append(f)

        all_files = self.repo_files.get(repo, [])
        random_pool = [f for f in all_files if f not in gt_files and f not in selected]
        random.shuffle(random_pool)
        n_still_needed = num_negatives - len(result)
        for f in random_pool[:max(n_random, n_still_needed)]:
            if len(result) >= num_negatives:
                break
            selected.add(f)
            result.append(f)

        if len(result) < num_negatives:
            remaining = [f for f in all_files if f not in gt_files and f not in selected]
            random.shuffle(remaining)
            for f in remaining:
                if len(result) >= num_negatives:
                    break
                selected.add(f)
                result.append(f)

        while len(result) < num_negatives and len(result) > 0:
            result.append(result[random.randint(0, len(result) - 1)])

        return result[:num_negatives]


# ============================================================
# File summary loading
# ============================================================

def load_file_summaries(path: str) -> Dict[str, Dict[str, str]]:
    """Load file summaries: {repo: {file_path: summary_str}}.

    Supports two formats:
    1. Combined JSON: {"repo_name": {"path": "summary", ...}, ...}
    2. Directory of per-repo JSONs
    """
    if os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
        print(f"  Loaded file summaries for {len(data)} repos from {path}")
        return data

    if os.path.isdir(path):
        data = {}
        for fname in sorted(os.listdir(path)):
            if not fname.endswith(".json"):
                continue
            repo = fname.replace(".json", "")
            with open(os.path.join(path, fname)) as f:
                data[repo] = json.load(f)
        print(f"  Loaded file summaries for {len(data)} repos from {path}/")
        return data

    print(f"  Warning: file summaries not found at {path}")
    return {}


def get_summary(
    file_summaries: Dict[str, Dict[str, str]],
    repo: str,
    file_path: str,
) -> Optional[str]:
    """Look up file summary, return None if not available."""
    repo_summaries = file_summaries.get(repo, {})
    return repo_summaries.get(file_path)


# ============================================================
# Data loading
# ============================================================

def load_train_data(path: str) -> List[Dict]:
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if item.get("changed_py_files"):
                data.append(item)
    print(f"  Loaded {len(data)} training examples from {path}")
    return data


def load_bm25_candidates(path: str) -> Dict[str, Dict]:
    result = {}
    if not os.path.isfile(path):
        print(f"  Warning: BM25 candidates not found: {path}")
        return result
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            key = f"{item['repo']}_{item['issue_id']}"
            result[key] = item
    print(f"  Loaded BM25 candidates for {len(result)} examples from {path}")
    return result


# ============================================================
# Score extraction
# ============================================================

def get_yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    yes_id = yes_ids[0]
    no_id = no_ids[0]
    print(f"  Yes token ID: {yes_id} ('{tokenizer.decode([yes_id])}')")
    print(f"  No token ID: {no_id} ('{tokenizer.decode([no_id])}')")
    return yes_id, no_id


def compute_scores(
    model, tokenizer, prompts, yes_id, no_id,
    max_seq_length, device, mini_batch_size=4,
) -> torch.Tensor:
    all_scores = []
    for i in range(0, len(prompts), mini_batch_size):
        batch_prompts = prompts[i : i + mini_batch_size]
        encodings = tokenizer(
            batch_prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=max_seq_length,
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

    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    device = args.device

    # Load tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    yes_id, no_id = get_yes_no_token_ids(tokenizer)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True,
    )

    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=True)
    else:
        print(f"Creating fresh LoRA (rank={args.lora_rank})...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_rank,
            lora_alpha=args.lora_rank * 2, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.train()

    # Load data
    print("Loading training data...")
    train_data = load_train_data(args.train_data)

    print("Loading BM25 candidates...")
    bm25_candidates = load_bm25_candidates(args.bm25_candidates)

    print("Loading file summaries...")
    file_summaries = load_file_summaries(args.file_summaries)

    # Count coverage
    summary_hits = 0
    summary_misses = 0
    for item in train_data[:100]:
        repo = item["repo"]
        for f in item["changed_py_files"][:1]:
            if get_summary(file_summaries, repo, f):
                summary_hits += 1
            else:
                summary_misses += 1
    print(f"  Summary coverage (sampled 100): {summary_hits}/{summary_hits+summary_misses} "
          f"({100*summary_hits/max(1,summary_hits+summary_misses):.0f}%)")

    print("Building graph indexes...")
    cochange_index = {}
    if args.train_data_for_cochange:
        cochange_index = build_cochange_index(args.train_data_for_cochange)
        print(f"  Co-change index for {len(cochange_index)} repos")
    elif args.train_data:
        cochange_index = build_cochange_index(args.train_data)
        print(f"  Co-change index for {len(cochange_index)} repos")

    import_index = {}
    if args.dep_graph_dir:
        import_index = build_import_adjacency(args.dep_graph_dir)
        print(f"  Import index for {len(import_index)} repos")

    print("Loading file trees...")
    repo_files = load_file_trees(args.file_tree_dir or "data/file_trees")
    print(f"  File trees for {len(repo_files)} repos")

    neg_sampler = NegativeSampler(
        bm25_candidates=bm25_candidates,
        cochange_index=cochange_index,
        import_index=import_index,
        repo_files=repo_files,
        neg_bm25_ratio=args.neg_bm25_ratio,
        neg_graph_ratio=args.neg_graph_ratio,
        neg_random_ratio=args.neg_random_ratio,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=0.01,
    )

    total_examples = len(train_data)
    steps_per_epoch = math.ceil(total_examples / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * 0.05)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n{'='*60}")
    print(f"RankFT-Content Training")
    print(f"  Examples: {total_examples}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Negatives per positive: {args.num_negatives}")
    print(f"  Neg mix: BM25={args.neg_bm25_ratio} Graph={args.neg_graph_ratio} "
          f"Random={args.neg_random_ratio}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch: {args.gradient_accumulation_steps}")
    print(f"  Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}")
    print(f"  LR: {args.learning_rate}, Warmup: {warmup_steps}")
    print(f"  Max seq length: {args.max_seq_length}")
    print(f"  File summaries: {'YES' if file_summaries else 'NO'}")
    print(f"{'='*60}\n")

    global_step = 0
    accumulated_loss = 0.0
    log_losses = []
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(args.num_epochs):
        indices = list(range(total_examples))
        random.shuffle(indices)
        optimizer.zero_grad()

        for ex_idx, data_idx in enumerate(indices):
            example = train_data[data_idx]
            repo = example["repo"]
            issue_id = example["issue_id"]
            issue_text = example["issue_text"]
            gt_files = set(example["changed_py_files"])

            positive_file = random.choice(list(gt_files))
            negatives = neg_sampler.sample(repo, issue_id, gt_files, args.num_negatives)

            if len(negatives) == 0:
                continue

            candidates = [positive_file] + negatives
            prompts = []
            for cand in candidates:
                summary = get_summary(file_summaries, repo, cand)
                prompts.append(build_prompt(issue_text, cand, summary))

            try:
                scores = compute_scores(
                    model, tokenizer, prompts,
                    yes_id, no_id,
                    args.max_seq_length, device,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at example {ex_idx}, skipping.")
                    torch.cuda.empty_cache()
                    continue
                raise

            log_probs = F.log_softmax(scores, dim=0)
            loss = -log_probs[0]
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

                if global_step % args.logging_steps == 0:
                    avg_recent = np.mean(log_losses[-args.logging_steps:])
                    elapsed = time.time() - start_time
                    current_lr = scheduler.get_last_lr()[0]
                    print(
                        f"  [Epoch {epoch+1}/{args.num_epochs}] "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {step_loss:.4f} (avg: {avg_recent:.4f}) | "
                        f"LR: {current_lr:.2e} | "
                        f"Time: {elapsed:.0f}s"
                    )

                if global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    print(f"  Saving checkpoint to {ckpt_dir}...")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    state = {
                        "global_step": global_step, "epoch": epoch,
                        "loss_history": log_losses,
                    }
                    with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
                        json.dump(state, f, indent=2)

                    recent_avg = np.mean(log_losses[-min(50, len(log_losses)):])
                    if recent_avg < best_loss:
                        best_loss = recent_avg
                        best_dir = os.path.join(args.output_dir, "best")
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
                        print(f"    New best (avg loss: {best_loss:.4f})")

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

    final_dir = os.path.join(args.output_dir, "final")
    print(f"\nSaving final model to {final_dir}...")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(os.path.join(args.output_dir, "loss_history.json"), "w") as f:
        json.dump({"losses": log_losses, "total_steps": global_step}, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"  Total steps: {global_step}")
    print(f"  Final avg loss: {np.mean(log_losses[-50:]):.4f}")
    print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")


def main():
    parser = argparse.ArgumentParser(
        description="RankFT-Content: Content-aware reranker training"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--bm25_candidates", required=True)
    parser.add_argument("--file_summaries", required=True,
                        help="File summaries JSON (combined or directory)")
    parser.add_argument("--dep_graph_dir", default="data/dep_graphs")
    parser.add_argument("--train_data_for_cochange", default=None)
    parser.add_argument("--file_tree_dir", default="data/file_trees")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_negatives", type=int, default=16)
    parser.add_argument("--neg_bm25_ratio", type=float, default=0.5)
    parser.add_argument("--neg_graph_ratio", type=float, default=0.25)
    parser.add_argument("--neg_random_ratio", type=float, default=0.25)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_seq_length", type=int, default=768,
                        help="Longer than path-only due to summaries")
    parser.add_argument("--lora_rank", type=int, default=32)

    args = parser.parse_args()

    ratio_sum = args.neg_bm25_ratio + args.neg_graph_ratio + args.neg_random_ratio
    if abs(ratio_sum - 1.0) > 0.01:
        parser.error(f"Negative ratios must sum to 1.0, got {ratio_sum:.2f}")

    train(args)


if __name__ == "__main__":
    main()
