#!/usr/bin/env python3
"""
Train s_code: filename-masked code reranker for Scope-Aware Prior-Residual Localizer.

This reranker sees issue text + code content but with filenames MASKED
(replaced by anonymous tokens like "file_A.py"). This forces the model
to rank based on code semantics, not path identity.

Key design:
- Input: "Bug Report: {issue}\n\nFile: file_{hash}.py\n\nCode:\n{functions}\n\nAnswer:"
- Filename is anonymized (sha256 hash of original, consistent within example)
- Code content: top-5 AST-extracted functions, BM25-ranked by issue
- Loss: listwise CE (1 pos + 16 neg) + PathSwap augmentation (50%)
- Also mines path-collision negatives (same dir, different code)

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/train_code_reranker_masked.py \
        --gpu_id 0 \
        --output_dir /data/chenlibin/grepo_agent_experiments/code_reranker_masked \
        --epochs 2 --lr 5e-5
"""

import argparse
import hashlib
import json
import os
import random
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
TRAIN_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_train.jsonl"
BM25_TRAIN_PATH = "/home/chenlibin/grepo_agent/data/rankft/grepo_train_bm25_top500.jsonl"
REPO_DIR = "/home/chenlibin/grepo_agent/data/repos"
FUNC_CACHE_PATH = "/data/chenlibin/grepo_agent_experiments/function_cache_v2.json"

PROMPT_TEMPLATE = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {masked_path}\n\n"
    "Code:\n{code_content}\n\n"
    "Answer:"
)


def anonymize_path(path, salt=""):
    """Replace filename with deterministic anonymous token, keep extension."""
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    name, ext = os.path.splitext(basename)
    h = hashlib.sha256((name + salt).encode()).hexdigest()[:6]
    anon_name = f"mod_{h}{ext}"
    if dirname:
        # Also anonymize directory names
        parts = dirname.split("/")
        anon_parts = [f"pkg_{hashlib.sha256((p + salt).encode()).hexdigest()[:4]}"
                      for p in parts]
        return "/".join(anon_parts) + "/" + anon_name
    return anon_name


def get_functions_from_cache(func_cache, repo, file_path):
    """Get pre-extracted functions from cache."""
    repo_cache = func_cache.get(repo, {})
    return repo_cache.get(file_path, [])


def get_ranked_code(func_cache, repo, file_path, issue_text,
                    max_funcs=5):
    """Get BM25-ranked function text from cache."""
    functions = get_functions_from_cache(func_cache, repo, file_path)
    if not functions:
        # Fallback: read first lines
        full_path = os.path.join(REPO_DIR, repo, file_path)
        try:
            with open(full_path, "r", errors="replace") as f:
                return "".join(f.readlines()[:50])
        except Exception:
            return "# (file not available)"

    from rank_bm25 import BM25Okapi
    issue_tokens = issue_text.lower().split()
    func_tokens = [f["body"].lower().split() + f["name"].lower().split("_")
                   for f in functions]
    bm25 = BM25Okapi(func_tokens)
    scores = bm25.get_scores(issue_tokens)
    ranked = sorted(zip(functions, scores), key=lambda x: -x[1])

    texts = []
    for f, s in ranked[:max_funcs]:
        texts.append(f"# {f['name']}\n{f['body']}")
    return "\n\n".join(texts)


def build_prompt(issue_text, original_path, code_content, salt="",
                 tokenizer=None, max_seq_length=2048):
    """Build masked-path code prompt."""
    masked_path = anonymize_path(original_path, salt=salt)

    # Pre-truncate issue to leave room for code
    max_issue_tokens = max_seq_length - 800
    if tokenizer:
        issue_ids = tokenizer.encode(issue_text, add_special_tokens=False)
        if len(issue_ids) > max_issue_tokens:
            issue_text = tokenizer.decode(issue_ids[:max_issue_tokens],
                                           skip_special_tokens=True)

    # Truncate code content
    code_truncated = code_content[:2000]  # ~500 tokens

    return PROMPT_TEMPLATE.format(
        issue_text=issue_text,
        masked_path=masked_path,
        code_content=code_truncated,
    )


def compute_listwise_loss(model, tokenizer, prompts, yes_id, no_id,
                          device, max_seq_length):
    """Listwise CE loss. Index 0 is positive."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_seq_length,
                       padding_side="left").to(device)
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    scores = logits[:, yes_id].float() - logits[:, no_id].float()
    target = torch.zeros(1, dtype=torch.long, device=device)
    loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0), target)
    return loss


def mine_collision_negatives(gt_files, bm25_candidates, n=4):
    """Mine path-collision negatives: same directory as GT but wrong file."""
    gt_dirs = {os.path.dirname(g) for g in gt_files}
    collisions = []
    for c in bm25_candidates:
        if c not in gt_files and os.path.dirname(c) in gt_dirs:
            collisions.append(c)
    if len(collisions) >= n:
        return random.sample(collisions, n)
    return collisions


def load_data():
    train_data = []
    with open(TRAIN_PATH) as f:
        for line in f:
            train_data.append(json.loads(line))

    bm25_data = {}
    with open(BM25_TRAIN_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            bm25_data[key] = rec

    return train_data, bm25_data


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    print("Loading data...")
    train_data, bm25_data = load_data()
    print(f"  {len(train_data)} train, {len(bm25_data)} candidates")

    print("Loading function cache...")
    func_cache = json.load(open(FUNC_CACHE_PATH))
    n_cached = sum(len(v) for v in func_cache.values())
    print(f"  {len(func_cache)} repos, {n_cached} files cached")

    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": device},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.train()

    lora_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    num_neg = args.num_negatives
    accum_steps = args.gradient_accumulation_steps
    print(f"Training: {args.epochs} epochs, lr={args.lr}, neg={num_neg}, "
          f"accum={accum_steps}, pathswap_frac={args.pathswap_fraction}")

    log_file = open(os.path.join(args.output_dir, "training_log.jsonl"), "w")
    start_time = time.time()
    global_step = 0

    for epoch in range(args.epochs):
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        epoch_loss = 0
        epoch_examples = 0
        optimizer.zero_grad()
        accum_count = 0

        for ex_idx, data_idx in enumerate(indices):
            example = train_data[data_idx]
            repo = example["repo"]
            issue_text = example["issue_text"]
            gt_files = set(example.get("changed_py_files",
                                        example.get("changed_files", [])))
            if not gt_files:
                continue

            key = (repo, str(example["issue_id"]))
            if key not in bm25_data:
                continue

            positive_file = random.choice(sorted(gt_files))

            bm25_cands = bm25_data[key].get("candidates",
                                              bm25_data[key].get("bm25_candidates", []))

            # Mine negatives: regular + collision
            neg_pool = [c for c in bm25_cands if c not in gt_files]
            collision_negs = mine_collision_negatives(
                gt_files, bm25_cands, n=min(4, num_neg // 2))
            regular_neg_pool = [c for c in neg_pool if c not in collision_negs]

            n_regular = num_neg - len(collision_negs)
            if len(regular_neg_pool) < n_regular:
                continue
            regular_negs = random.sample(regular_neg_pool, n_regular)
            negatives = collision_negs + regular_negs

            all_files = [positive_file] + negatives

            # PathSwap augmentation: 50% of examples use shuffled paths
            # (but code stays with original file)
            use_pathswap = random.random() < args.pathswap_fraction

            # Build prompts with masked paths + code
            salt = f"{repo}_{example['issue_id']}"
            if use_pathswap:
                # Shuffle path assignments (code stays)
                display_paths = list(all_files)
                random.shuffle(display_paths)
            else:
                display_paths = list(all_files)

            prompts = []
            for i, original_file in enumerate(all_files):
                code = get_ranked_code(func_cache, repo, original_file,
                                       issue_text, max_funcs=5)
                prompt = build_prompt(
                    issue_text, display_paths[i], code, salt=salt,
                    tokenizer=tokenizer, max_seq_length=args.max_seq_length)
                prompts.append(prompt)

            try:
                loss = compute_listwise_loss(
                    model, tokenizer, prompts, yes_id, no_id,
                    device, args.max_seq_length)
                loss = loss / accum_steps
                loss.backward()
                epoch_loss += loss.item() * accum_steps
                epoch_examples += 1
                accum_count += 1
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    accum_count = 0
                    continue
                raise

            if accum_count >= accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                accum_count = 0

            if (ex_idx + 1) % 200 == 0:
                avg_loss = epoch_loss / max(1, epoch_examples)
                elapsed = time.time() - start_time
                print(f"  [{epoch+1}/{args.epochs}] [{ex_idx+1}] "
                      f"loss={avg_loss:.4f} ({elapsed:.0f}s)")
                log_file.write(json.dumps({
                    "epoch": epoch, "step": ex_idx + 1,
                    "avg_loss": avg_loss,
                }) + "\n")
                log_file.flush()

        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = epoch_loss / max(1, epoch_examples)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, examples={epoch_examples}")

    save_dir = os.path.join(args.output_dir, "best")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    log_file.close()

    print(f"\nSaved to {save_dir}")
    print(f"Total time: {time.time() - start_time:.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_negatives", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--pathswap_fraction", type=float, default=0.5,
                        help="Fraction of examples with PathSwap augmentation")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
