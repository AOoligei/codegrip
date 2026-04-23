#!/usr/bin/env python3
"""
Train a hierarchical path+code reranker using function snippets.

Wraps train_rankft.py with a modified prompt that includes top-3
BM25-ranked function snippets from each candidate file.

The key difference from path-invariant training: here code snippets are
genuinely relevant (BM25-selected by issue similarity), providing the
strongest possible code signal to the scorer.

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/train_hierarchical.py \
        --gpu_id 0 \
        --output_dir /data/chenlibin/grepo_agent_experiments/hierarchical_train \
        --epochs 2 --lr 5e-5
"""

import argparse
import ast
import json
import os
import random
import re
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rank_bm25 import BM25Okapi
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Paths
MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
TRAIN_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_train.jsonl"
BM25_TRAIN_PATH = "/home/chenlibin/grepo_agent/data/rankft/grepo_train_bm25_top500.jsonl"
REPO_DIR = "/home/chenlibin/grepo_agent/data/repos"

# Hierarchical prompt template
HIER_PROMPT = (
    "Given the bug report, is this file likely to need modification? "
    "Consider both the file path and the code snippets shown below.\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n"
    "Relevant functions:\n{function_snippets}\n\n"
    "Answer:"
)

PATH_ONLY_PROMPT = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)


def extract_functions(repo_name, file_path, max_lines=30):
    """Extract function definitions from a Python file using AST."""
    full_path = os.path.join(REPO_DIR, repo_name, file_path)
    if not os.path.isfile(full_path):
        return []
    try:
        with open(full_path, "r", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, Exception):
        return []

    lines = source.splitlines()
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = min(start + max_lines, len(lines))
            body = "\n".join(lines[start:end])
            name_tokens = re.split(r'[_]', node.name.lower())
            functions.append({
                "name": node.name,
                "body": body,
                "lineno": node.lineno,
                "name_tokens": name_tokens,
            })
    return functions


def get_function_snippets(repo, file_path, issue_text, top_m=3, max_lines=30):
    """Extract and BM25-rank functions, return formatted snippet string."""
    funcs = extract_functions(repo, file_path, max_lines)
    if not funcs:
        return "# (no functions extracted)"

    issue_tokens = issue_text.lower().split()
    func_tokens = [f["body"].lower().split() + f["name_tokens"] for f in funcs]

    bm25 = BM25Okapi(func_tokens)
    scores = bm25.get_scores(issue_tokens)
    ranked = sorted(zip(funcs, scores), key=lambda x: -x[1])

    snippet = ""
    for f, s in ranked[:top_m]:
        snippet += f"# {f['name']} (line {f['lineno']})\n"
        snippet += f["body"][:500] + "\n\n"
    return snippet.strip()


def build_hier_prompt(issue_text, candidate_path, function_snippets,
                      max_issue_tokens, tokenizer):
    """Build hierarchical prompt with truncated issue text."""
    # Truncate issue to leave room for snippets
    issue_ids = tokenizer.encode(issue_text, add_special_tokens=False)
    if len(issue_ids) > max_issue_tokens:
        issue_text = tokenizer.decode(issue_ids[:max_issue_tokens],
                                       skip_special_tokens=True)

    return HIER_PROMPT.format(
        issue_text=issue_text,
        candidate_path=candidate_path,
        function_snippets=function_snippets,
    )


def truncate_prompt_safely(prompt, tokenizer, max_seq_length):
    """Truncate prompt from the middle (issue text) to preserve Answer: suffix."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_seq_length:
        return prompt
    # Find "Answer:" suffix tokens (last ~3 tokens)
    suffix = "\n\nAnswer:"
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    # Keep prefix + suffix, truncate middle
    keep = max_seq_length - len(suffix_ids) - 1
    truncated_ids = ids[:keep] + suffix_ids
    return tokenizer.decode(truncated_ids, skip_special_tokens=True)


def compute_listwise_loss(model, tokenizer, prompts, yes_id, no_id,
                          device, max_seq_length):
    """Compute listwise ranking loss. First prompt is positive, rest negative."""
    # Pre-truncate to guarantee Answer: suffix survives
    prompts = [truncate_prompt_safely(p, tokenizer, max_seq_length) for p in prompts]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_seq_length,
                       padding_side="left").to(device)

    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    yes_logits = logits[:, yes_id].float()
    no_logits = logits[:, no_id].float()
    scores = yes_logits - no_logits  # (num_candidates,)

    # Listwise loss: positive should score highest
    # Cross-entropy over candidates: target = 0 (first is positive)
    target = torch.zeros(1, dtype=torch.long, device=device)
    loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0), target)

    return loss


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

    # Prepare for QLoRA training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # Add LoRA
    lora_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()
    model.print_trainable_parameters()

    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    yes_id, no_id = yes_ids[0], no_ids[0]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Training loop
    num_negatives = args.num_negatives
    max_issue_tokens = args.max_seq_length - 1000  # reserve for snippets
    accum_steps = args.gradient_accumulation_steps

    print(f"Training: {args.epochs} epochs, lr={args.lr}, neg={num_negatives}, "
          f"accum={accum_steps}, max_seq={args.max_seq_length}")

    global_step = 0
    log_file = open(os.path.join(args.output_dir, "training_log.jsonl"), "w")
    start_time = time.time()

    for epoch in range(args.epochs):
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        epoch_loss = 0
        epoch_examples = 0
        optimizer.zero_grad()
        accum_count = 0  # track actual backward passes in current window

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

            # Pick positive
            positive_file = random.choice(sorted(gt_files))

            # Sample negatives from BM25 candidates
            candidates = bm25_data[key].get("candidates",
                                             bm25_data[key].get("bm25_candidates", []))
            neg_pool = [c for c in candidates if c not in gt_files]
            if len(neg_pool) < num_negatives:
                continue
            negatives = random.sample(neg_pool, num_negatives)

            # Build prompts: positive first, then negatives
            all_files = [positive_file] + negatives
            prompts = []

            # Mix: 50% hierarchical prompts, 50% path-only
            use_hier = random.random() < args.hier_fraction

            # Pre-truncate issue text for path-only prompts too
            issue_ids = tokenizer.encode(issue_text, add_special_tokens=False)
            # Path-only needs ~50 tokens for template+path
            path_issue_limit = args.max_seq_length - 50
            if len(issue_ids) > path_issue_limit:
                issue_truncated_path = tokenizer.decode(
                    issue_ids[:path_issue_limit], skip_special_tokens=True)
            else:
                issue_truncated_path = issue_text

            for cand in all_files:
                if use_hier:
                    snippets = get_function_snippets(repo, cand, issue_text,
                                                     top_m=args.top_m_funcs,
                                                     max_lines=args.func_max_lines)
                    prompt = build_hier_prompt(issue_text, cand, snippets,
                                               max_issue_tokens, tokenizer)
                else:
                    prompt = PATH_ONLY_PROMPT.format(
                        issue_text=issue_truncated_path, candidate_path=cand)
                prompts.append(prompt)

            # Compute loss
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
                    optimizer.zero_grad()  # clear partial grads on OOM
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
                print(f"  [{epoch+1}/{args.epochs}] [{ex_idx+1}/{len(indices)}] "
                      f"loss={avg_loss:.4f} ({elapsed:.0f}s)")
                log_entry = {
                    "epoch": epoch, "step": ex_idx + 1,
                    "global_step": global_step,
                    "avg_loss": avg_loss, "elapsed": elapsed,
                }
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()

        # Handle final partial accumulation
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            accum_count = 0

        avg_loss = epoch_loss / max(1, epoch_examples)
        print(f"  Epoch {epoch+1} done: avg_loss={avg_loss:.4f}, "
              f"examples={epoch_examples}")

    # Save
    save_dir = os.path.join(args.output_dir, "best")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    log_file.close()

    print(f"\nModel saved to {save_dir}")
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
    parser.add_argument("--top_m_funcs", type=int, default=3)
    parser.add_argument("--func_max_lines", type=int, default=30)
    parser.add_argument("--hier_fraction", type=float, default=0.5,
                        help="Fraction of examples using hierarchical prompts (rest path-only)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
