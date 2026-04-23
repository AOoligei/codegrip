#!/usr/bin/env python3
"""
Train SPECTER code expert on hard examples.

Architecture: Qwen2.5-7B + LoRA, scoring (issue, path, top-3 function snippets).
Training data: hard examples mined from training set (path-confusable, low overlap).
Loss: listwise CE with 1 positive + 8 hard negatives + 8 easy negatives.

Key differences from standard training:
- Uses hierarchical prompt with function snippets (not path-only)
- Trains exclusively on hard examples (where code should matter)
- Hard negatives are same-directory / same-stem files (force reading code)
- No path-only mixing — this is a pure code expert

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/train_specter_expert.py \
        --gpu_id 0 \
        --output_dir /data/chenlibin/grepo_agent_experiments/specter/expert \
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

MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
HARD_DATA_PATH = "/data/chenlibin/grepo_agent_experiments/specter/data/hard_examples_train.jsonl"
REPO_DIR = "/home/chenlibin/grepo_agent/data/repos"

HIER_PROMPT = (
    "Given the bug report, is this file likely to need modification? "
    "Consider both the file path and the code snippets shown below.\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n"
    "Relevant functions:\n{function_snippets}\n\n"
    "Answer:"
)


def extract_functions(repo_name, file_path, max_lines=30):
    """Extract function definitions via AST."""
    full_path = os.path.join(REPO_DIR, repo_name, file_path)
    if not os.path.isfile(full_path):
        return []
    try:
        with open(full_path, "r", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception:
        return []
    lines = source.splitlines()
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = min(start + max_lines, len(lines))
            body = "\n".join(lines[start:end])
            functions.append({
                "name": node.name, "body": body, "lineno": node.lineno,
            })
    return functions


def get_snippets(repo, file_path, issue_text, top_m=3):
    """Return BM25-ranked function snippets for a candidate file."""
    funcs = extract_functions(repo, file_path)
    if not funcs:
        return "# (no functions extracted)"
    issue_tokens = issue_text.lower().split()
    func_tokens = [f["body"].lower().split() + re.split(r'[_]', f["name"].lower())
                   for f in funcs]
    bm25 = BM25Okapi(func_tokens)
    scores = bm25.get_scores(issue_tokens)
    ranked = sorted(zip(funcs, scores), key=lambda x: -x[1])
    snippet = ""
    for f, _ in ranked[:top_m]:
        snippet += f"# {f['name']} (line {f['lineno']})\n{f['body'][:500]}\n\n"
    return snippet.strip()


def build_prompt(issue_text, candidate_path, snippets, max_issue_tokens, tokenizer):
    """Build prompt with truncated issue to leave room for snippets."""
    issue_ids = tokenizer.encode(issue_text, add_special_tokens=False)
    if len(issue_ids) > max_issue_tokens:
        issue_text = tokenizer.decode(issue_ids[:max_issue_tokens],
                                       skip_special_tokens=True)
    return HIER_PROMPT.format(
        issue_text=issue_text,
        candidate_path=candidate_path,
        function_snippets=snippets,
    )


def truncate_prompt_safely(prompt, tokenizer, max_seq_length):
    """Truncate from middle to preserve Answer: suffix at the end."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_seq_length:
        return prompt
    suffix = "\n\nAnswer:"
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    keep = max_seq_length - len(suffix_ids) - 1
    truncated_ids = ids[:keep] + suffix_ids
    return tokenizer.decode(truncated_ids, skip_special_tokens=True)


def compute_listwise_loss(model, tokenizer, prompts, yes_id, no_id,
                          device, max_seq_length):
    """Listwise loss: positive at index 0, CE with target=0."""
    # Pre-truncate to guarantee Answer: suffix survives
    prompts = [truncate_prompt_safely(p, tokenizer, max_seq_length) for p in prompts]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_seq_length,
                       padding_side="left").to(device)
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    yes_logits = logits[:, yes_id].float()
    no_logits = logits[:, no_id].float()
    scores = yes_logits - no_logits
    target = torch.zeros(1, dtype=torch.long, device=device)
    loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0), target)
    return loss


def load_hard_data():
    """Load hard training examples."""
    data = []
    with open(HARD_DATA_PATH) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    print("Loading hard training examples...")
    train_data = load_hard_data()
    print(f"  {len(train_data)} hard examples")

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

    max_issue_tokens = max(200, args.max_seq_length - 1000)
    accum_steps = args.gradient_accumulation_steps

    print(f"Training: {args.epochs} epochs, lr={args.lr}, "
          f"hard_negs={args.num_hard_negs}, easy_negs={args.num_easy_negs}, "
          f"accum={accum_steps}, max_seq={args.max_seq_length}")

    global_step = 0
    log_file = open(os.path.join(args.output_dir, "training_log.jsonl"), "w")
    start_time = time.time()

    for epoch in range(args.epochs):
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        epoch_loss = 0
        epoch_examples = 0
        accum_count = 0
        optimizer.zero_grad()

        for ex_idx, data_idx in enumerate(indices):
            ex = train_data[data_idx]
            repo = ex["repo"]
            issue_text = ex["issue_text"]
            positive = ex["gt_file"]
            all_hard_negs = ex.get("hard_negs", [])
            all_easy_negs = ex.get("easy_negs", [])

            # Require full list structure for consistent listwise loss
            # (otherwise CE denominator varies, making short lists "easier")
            if len(all_hard_negs) < args.num_hard_negs:
                continue
            if len(all_easy_negs) < args.num_easy_negs:
                continue
            hard_negs = random.sample(all_hard_negs, args.num_hard_negs)
            easy_negs = random.sample(all_easy_negs, args.num_easy_negs)

            # Build prompts: positive first, then hard, then easy
            all_files = [positive] + hard_negs + easy_negs
            prompts = []
            for cand in all_files:
                snippets = get_snippets(repo, cand, issue_text,
                                         top_m=args.top_m_funcs)
                prompt = build_prompt(issue_text, cand, snippets,
                                       max_issue_tokens, tokenizer)
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

            if (ex_idx + 1) % 100 == 0:
                avg_loss = epoch_loss / max(1, epoch_examples)
                elapsed = time.time() - start_time
                print(f"  [{epoch+1}/{args.epochs}] [{ex_idx+1}/{len(indices)}] "
                      f"loss={avg_loss:.4f} ({elapsed:.0f}s)")
                log_file.write(json.dumps({
                    "epoch": epoch, "step": ex_idx + 1,
                    "global_step": global_step,
                    "avg_loss": avg_loss, "elapsed": elapsed,
                }) + "\n")
                log_file.flush()

        # Final partial accumulation
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = epoch_loss / max(1, epoch_examples)
        print(f"  Epoch {epoch+1} done: avg_loss={avg_loss:.4f}, "
              f"examples={epoch_examples}")

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
    parser.add_argument("--num_hard_negs", type=int, default=4)
    parser.add_argument("--num_easy_negs", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--top_m_funcs", type=int, default=3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
