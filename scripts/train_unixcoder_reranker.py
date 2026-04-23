#!/usr/bin/env python3
"""
Train UniXcoder as a code-only cross-encoder reranker for bug localization.

This is an architecturally different baseline: a pretrained code understanding
model (UniXcoder, 126M, pretrained on AST + dataflow) fine-tuned as a
cross-encoder reranker using ONLY code content (NO file paths).

Input: (issue_text, code_content) -> relevance score
Architecture: RoBERTa cross-encoder with [CLS] classification head
Training: binary cross-entropy, 1 positive + N negatives per example

This baseline addresses reviewer W1: if even a model pretrained specifically
for code understanding cannot beat path-only, the "weak code baseline"
objection is substantially weakened.

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/train_unixcoder_reranker.py \
        --gpu_id 0 \
        --output_dir /data/chenlibin/grepo_agent_experiments/unixcoder_reranker \
        --epochs 3 --lr 2e-5 --num_negatives 7
"""

import argparse
import ast
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MODEL_NAME = "microsoft/unixcoder-base"
HF_CACHE = "/data/chenlibin/hf_cache"
TRAIN_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_train.jsonl"
BM25_TRAIN_PATH = "/home/chenlibin/grepo_agent/data/rankft/grepo_train_bm25_top500.jsonl"
REPO_DIR = "/home/chenlibin/grepo_agent/data/repos"


class UniXcoderReranker(nn.Module):
    """Cross-encoder reranker using UniXcoder backbone."""

    def __init__(self, model_name, cache_dir):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        hidden_size = self.encoder.config.hidden_size  # 768
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output).squeeze(-1)  # (batch_size,)
        return logits


def read_code_content(repo, file_path, max_lines=50):
    """Read code content from repo file."""
    full_path = os.path.join(REPO_DIR, repo, file_path)
    if not os.path.isfile(full_path):
        return ""
    try:
        with open(full_path, "r", errors="replace") as f:
            lines = f.readlines()[:max_lines]
        return "".join(lines)
    except Exception:
        return ""


def extract_functions_text(repo, file_path, max_funcs=5, max_lines_per_func=20):
    """Extract top functions as text for richer code representation."""
    full_path = os.path.join(REPO_DIR, repo, file_path)
    if not os.path.isfile(full_path):
        return ""
    try:
        with open(full_path, "r", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, Exception):
        return ""

    lines = source.splitlines()
    func_texts = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = min(start + max_lines_per_func, len(lines))
            func_texts.append("\n".join(lines[start:end]))
            if len(func_texts) >= max_funcs:
                break

    return "\n\n".join(func_texts)


def build_input(issue_text, code_content, tokenizer, max_length=512):
    """Build cross-encoder input: [CLS] issue [SEP] code [SEP].

    NO file path — code content only.
    """
    # Truncate issue and code to fit within max_length
    # Reserve ~60% for issue, ~40% for code
    issue_max = int(max_length * 0.4)
    code_max = max_length - issue_max - 3  # 3 special tokens

    issue_ids = tokenizer.encode(issue_text, add_special_tokens=False)[:issue_max]
    code_ids = tokenizer.encode(code_content, add_special_tokens=False)[:code_max]

    # [CLS] issue [SEP] code [SEP]
    input_ids = [tokenizer.cls_token_id] + issue_ids + [tokenizer.sep_token_id] + \
                code_ids + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)

    # Pad to max_length
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len

    return input_ids[:max_length], attention_mask[:max_length]


def load_train_data():
    """Load training data + BM25 candidates."""
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
    train_data, bm25_data = load_train_data()
    print(f"  {len(train_data)} train, {len(bm25_data)} candidates")

    print("Loading UniXcoder...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
    model = UniXcoderReranker(MODEL_NAME, HF_CACHE).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {total_params/1e6:.0f}M total, {train_params/1e6:.0f}M trainable")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Estimate total steps for scheduler
    usable_examples = sum(1 for ex in train_data
                          if (ex["repo"], str(ex["issue_id"])) in bm25_data)
    total_steps = usable_examples * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"Training: {args.epochs} epochs, lr={args.lr}, neg={args.num_negatives}, "
          f"code_mode={args.code_mode}, ~{total_steps} steps")

    log_file = open(os.path.join(args.output_dir, "training_log.jsonl"), "w")
    start_time = time.time()
    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
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

            candidates = bm25_data[key].get("candidates",
                                             bm25_data[key].get("bm25_candidates", []))
            neg_pool = [c for c in candidates if c not in gt_files]
            if len(neg_pool) < args.num_negatives:
                continue
            negatives = random.sample(neg_pool, args.num_negatives)

            all_files = [positive_file] + negatives

            # Build inputs: code content only, NO paths
            all_input_ids = []
            all_attention = []
            for cand in all_files:
                if args.code_mode == "raw":
                    code = read_code_content(repo, cand, max_lines=args.code_max_lines)
                elif args.code_mode == "functions":
                    code = extract_functions_text(repo, cand,
                                                   max_funcs=5,
                                                   max_lines_per_func=20)
                else:
                    code = read_code_content(repo, cand, max_lines=args.code_max_lines)

                if not code.strip():
                    code = "# empty file"

                ids, mask = build_input(issue_text, code, tokenizer,
                                         max_length=args.max_seq_length)
                all_input_ids.append(ids)
                all_attention.append(mask)

            input_ids = torch.tensor(all_input_ids, device=device)
            attention_mask = torch.tensor(all_attention, device=device)

            # Forward
            try:
                logits = model(input_ids, attention_mask)
                # Listwise loss: positive at index 0
                target = torch.zeros(1, dtype=torch.long, device=device)
                loss = torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0), target)
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                epoch_loss += loss.item() * args.gradient_accumulation_steps
                epoch_examples += 1
                accum_count += 1
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    accum_count = 0
                    continue
                raise

            if accum_count >= args.gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                accum_count = 0

            if (ex_idx + 1) % 500 == 0:
                avg_loss = epoch_loss / max(1, epoch_examples)
                elapsed = time.time() - start_time
                lr_now = scheduler.get_last_lr()[0]
                print(f"  [{epoch+1}/{args.epochs}] [{ex_idx+1}] "
                      f"loss={avg_loss:.4f} lr={lr_now:.2e} ({elapsed:.0f}s)")
                log_file.write(json.dumps({
                    "epoch": epoch, "step": ex_idx + 1,
                    "avg_loss": avg_loss, "lr": lr_now,
                }).encode().decode() + "\n")
                log_file.flush()

        # Flush remaining gradients
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = epoch_loss / max(1, epoch_examples)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, examples={epoch_examples}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_dir = os.path.join(args.output_dir, "best")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
            tokenizer.save_pretrained(save_dir)
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump({"model_name": MODEL_NAME, "code_mode": args.code_mode,
                           "max_seq_length": args.max_seq_length,
                           "code_max_lines": args.code_max_lines}, f)
            print(f"  Saved best model (loss={best_loss:.4f})")

    log_file.close()
    print(f"\nDone. Total time: {time.time() - start_time:.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_negatives", type=int, default=7)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--code_max_lines", type=int, default=50)
    parser.add_argument("--code_mode", choices=["raw", "functions"], default="functions",
                        help="raw=first N lines, functions=AST-extracted function bodies")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
