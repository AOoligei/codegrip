#!/usr/bin/env python3
"""
PathSwap contrastive training: forces the model to learn code-invariant features.

For each training example, the model sees TWO views:
  1. Original: real paths + code content
  2. PathSwap: shuffled paths + SAME code content (code stays with original file)

Loss = ranking_loss(original) + ranking_loss(pathswap) + lambda * consistency_loss

Consistency loss penalizes disagreement between original and pathswap predictions,
forcing the model to rely on code signals rather than path shortcuts.

If successful: a model that's robust to path shuffling = positive method contribution.
If failed: even stronger evidence that code cannot compensate for path loss.

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/train_contrastive_pathswap.py \
        --gpu_id 0 \
        --output_dir /data/chenlibin/grepo_agent_experiments/contrastive_pathswap \
        --epochs 2 --lr 5e-5 --consistency_lambda 0.5
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
TRAIN_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_train.jsonl"
BM25_TRAIN_PATH = "/home/chenlibin/grepo_agent/data/rankft/grepo_train_bm25_top500.jsonl"
REPO_DIR = "/home/chenlibin/grepo_agent/data/repos"

PROMPT_TEMPLATE = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n"
    "Code:\n{code_content}\n\n"
    "Answer:"
)


def read_code(repo, filepath, max_lines=50):
    """Read code content from repo file."""
    full_path = os.path.join(REPO_DIR, repo, filepath)
    try:
        with open(full_path, "r", errors="replace") as f:
            lines = f.readlines()[:max_lines]
            return "".join(lines).strip()
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return "# (file not available)"


def build_prompt(issue_text, path, code, max_issue_tokens, tokenizer):
    """Build prompt with truncated issue text."""
    issue_ids = tokenizer.encode(issue_text, add_special_tokens=False)
    if len(issue_ids) > max_issue_tokens:
        issue_text = tokenizer.decode(issue_ids[:max_issue_tokens],
                                       skip_special_tokens=True)
    return PROMPT_TEMPLATE.format(
        issue_text=issue_text,
        candidate_path=path,
        code_content=code[:800],
    )


def compute_scores(model, tokenizer, prompts, yes_id, no_id,
                   max_seq_length, device):
    """Forward pass, return per-candidate scores."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_seq_length,
                       padding_side="left").to(device)
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    return logits[:, yes_id].float() - logits[:, no_id].float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_negatives", type=int, default=4,
                        help="Reduced from 8 to fit 2x forward passes in 24GB")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--code_max_lines", type=int, default=50)
    parser.add_argument("--consistency_lambda", type=float, default=0.5,
                        help="Weight for consistency loss between original and PathSwap views")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    # Load data
    print("Loading data...")
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
    print(f"  {len(train_data)} train, {len(bm25_data)} candidates")

    # Load model
    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config,
        device_map={"": device}, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()
    model.print_trainable_parameters()

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    max_issue_tokens = args.max_seq_length - 600
    accum_steps = args.gradient_accumulation_steps

    print(f"Training: {args.epochs} epochs, lr={args.lr}, neg={args.num_negatives}, "
          f"lambda={args.consistency_lambda}, accum={accum_steps}")

    log_file = open(os.path.join(args.output_dir, "training_log.jsonl"), "w")
    start_time = time.time()

    for epoch in range(args.epochs):
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        epoch_rank_loss = 0
        epoch_cons_loss = 0
        epoch_examples = 0
        accum_count = 0
        optimizer.zero_grad()

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
            candidates = bm25_data[key].get("candidates",
                                             bm25_data[key].get("bm25_candidates", []))
            neg_pool = [c for c in candidates if c not in gt_files]
            if len(neg_pool) < args.num_negatives:
                continue
            negatives = random.sample(neg_pool, args.num_negatives)

            all_files = [positive_file] + negatives

            # Read code for each file (code stays with original file)
            codes = [read_code(repo, f, args.code_max_lines) for f in all_files]

            # --- View 1: Original paths + code ---
            orig_prompts = [
                build_prompt(issue_text, f, c, max_issue_tokens, tokenizer)
                for f, c in zip(all_files, codes)
            ]

            # --- View 2: Shuffled paths + SAME code ---
            shuffled_paths = list(all_files)
            random.shuffle(shuffled_paths)
            swap_prompts = [
                build_prompt(issue_text, sp, c, max_issue_tokens, tokenizer)
                for sp, c in zip(shuffled_paths, codes)
            ]

            try:
                # Forward passes sequentially to save VRAM
                # View 1: original
                orig_scores = compute_scores(
                    model, tokenizer, orig_prompts, yes_id, no_id,
                    args.max_seq_length, device)
                target = torch.zeros(1, dtype=torch.long, device=device)
                rank_loss_orig = F.cross_entropy(orig_scores.unsqueeze(0), target)

                # View 2: PathSwap (detach orig scores for consistency)
                orig_scores_detached = orig_scores.detach()
                swap_scores = compute_scores(
                    model, tokenizer, swap_prompts, yes_id, no_id,
                    args.max_seq_length, device)
                rank_loss_swap = F.cross_entropy(swap_scores.unsqueeze(0), target)

                # Consistency: swap view should match original (one-sided KL)
                orig_probs = F.softmax(orig_scores_detached, dim=0)
                swap_probs = F.log_softmax(swap_scores, dim=0)
                consistency_loss = F.kl_div(swap_probs, orig_probs, reduction="sum")

                total_loss = (rank_loss_orig + rank_loss_swap) / 2 + \
                             args.consistency_lambda * consistency_loss
                total_loss = total_loss / accum_steps
                total_loss.backward()

                # Free orig graph before next iteration
                del orig_scores
                torch.cuda.empty_cache()

                epoch_rank_loss += (rank_loss_orig.item() + rank_loss_swap.item()) / 2
                epoch_cons_loss += consistency_loss.item()
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
                accum_count = 0

            if (ex_idx + 1) % 200 == 0:
                avg_rl = epoch_rank_loss / max(1, epoch_examples)
                avg_cl = epoch_cons_loss / max(1, epoch_examples)
                elapsed = time.time() - start_time
                print(f"  [{epoch+1}/{args.epochs}] [{ex_idx+1}/{len(indices)}] "
                      f"rank={avg_rl:.4f} cons={avg_cl:.4f} ({elapsed:.0f}s)")
                log_file.write(json.dumps({
                    "epoch": epoch, "step": ex_idx + 1,
                    "rank_loss": avg_rl, "cons_loss": avg_cl,
                    "elapsed": elapsed,
                }) + "\n")
                log_file.flush()

        # Flush final accumulation
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_rl = epoch_rank_loss / max(1, epoch_examples)
        avg_cl = epoch_cons_loss / max(1, epoch_examples)
        print(f"  Epoch {epoch+1}: rank={avg_rl:.4f} cons={avg_cl:.4f} "
              f"examples={epoch_examples}")

    # Save
    save_dir = os.path.join(args.output_dir, "best")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    log_file.close()
    print(f"\nSaved to {save_dir}, total time: {time.time() - start_time:.0f}s")


if __name__ == "__main__":
    main()
