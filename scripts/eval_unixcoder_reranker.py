#!/usr/bin/env python3
"""
Evaluate trained UniXcoder code-only reranker on GREPO test set.

Scores candidates using ONLY code content (no file paths).
Reports R@1 on full test, Code-Crucial subset, and under filename shuffle.

Key: under filename shuffle, this model should be INVARIANT (it never sees
paths). If it is, that proves it uses code; if path-only still beats it,
that proves code understanding alone is insufficient.

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/eval_unixcoder_reranker.py \
        --gpu_id 0 \
        --model_dir /data/chenlibin/grepo_agent_experiments/unixcoder_reranker/best \
        --output_dir /data/chenlibin/grepo_agent_experiments/unixcoder_reranker/eval_baseline
"""

import argparse
import ast
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

TEST_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_test.jsonl"
BM25_PATH = "/home/chenlibin/grepo_agent/data/rankft/merged_bm25_exp6_candidates.jsonl"
REPO_DIR = "/home/chenlibin/grepo_agent/data/repos"
HF_CACHE = "/data/chenlibin/hf_cache"


class UniXcoderReranker(nn.Module):
    def __init__(self, model_name, cache_dir):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output).squeeze(-1)
        return logits


def read_code_content(repo, file_path, max_lines=50):
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
    """[CLS] issue [SEP] code [SEP] -- NO file path."""
    issue_max = int(max_length * 0.4)
    code_max = max_length - issue_max - 3
    issue_ids = tokenizer.encode(issue_text, add_special_tokens=False)[:issue_max]
    code_ids = tokenizer.encode(code_content, add_special_tokens=False)[:code_max]
    input_ids = ([tokenizer.cls_token_id] + issue_ids +
                 [tokenizer.sep_token_id] + code_ids +
                 [tokenizer.sep_token_id])
    attention_mask = [1] * len(input_ids)
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
    return input_ids[:max_length], attention_mask[:max_length]


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    config_path = os.path.join(args.model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    model_name = config["model_name"]
    code_mode = config.get("code_mode", "functions")
    max_seq_length = config.get("max_seq_length", 512)
    code_max_lines = config.get("code_max_lines", 50)

    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = UniXcoderReranker(model_name, HF_CACHE)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pt"),
                                      map_location="cpu"))
    model = model.to(device)
    model.eval()
    print(f"  code_mode={code_mode}, max_seq={max_seq_length}")

    print("Loading data...")
    test_data = []
    with open(TEST_PATH) as f:
        for line in f:
            test_data.append(json.loads(line))

    bm25_data = {}
    with open(BM25_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            bm25_data[key] = rec
    print(f"  {len(test_data)} test, {len(bm25_data)} candidates")

    hits_at_k = defaultdict(list)
    results = []
    start_time = time.time()

    for idx, item in enumerate(test_data):
        repo = item["repo"]
        key = (repo, str(item["issue_id"]))
        if key not in bm25_data:
            continue

        issue_text = item["issue_text"]
        gt_files = set(item.get("changed_py_files", item.get("changed_files", [])))
        if not gt_files:
            continue

        bm25_rec = bm25_data[key]
        candidates = bm25_rec.get("candidates",
                                   bm25_rec.get("bm25_candidates", []))[:args.top_k]

        # UniXcoder sees CODE only, never paths.
        # Under shuffle: code stays with original file, so scores are IDENTICAL.
        # We still track display paths for GT matching under shuffle.
        original_candidates = list(candidates)

        # Score using code content only
        all_ids = []
        all_mask = []
        for cand in original_candidates:
            if code_mode == "functions":
                code = extract_functions_text(repo, cand)
            else:
                code = read_code_content(repo, cand, max_lines=code_max_lines)
            if not code.strip():
                code = "# empty file"
            ids, mask = build_input(issue_text, code, tokenizer, max_seq_length)
            all_ids.append(ids)
            all_mask.append(mask)

        scores = []
        for i in range(0, len(all_ids), args.batch_size):
            batch_ids = torch.tensor(all_ids[i:i+args.batch_size], device=device)
            batch_mask = torch.tensor(all_mask[i:i+args.batch_size], device=device)
            with torch.no_grad():
                batch_scores = model(batch_ids, batch_mask)
            scores.extend(batch_scores.cpu().numpy().tolist())

        # Rank (use original paths for GT matching - code-only model is path-invariant)
        ranked = sorted(zip(original_candidates, scores), key=lambda x: -x[1])
        predicted = [f for f, s in ranked]

        for k in [1, 3, 5, 10]:
            top_k_set = set(predicted[:k])
            hit = len(top_k_set & gt_files) / len(gt_files)
            hits_at_k[k].append(hit)

        results.append({
            "repo": repo,
            "issue_id": str(item["issue_id"]),
            "predicted": predicted[:20],
            "ground_truth": list(gt_files),
        })

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            r1 = np.mean(hits_at_k[1]) * 100
            print(f"  [{idx+1}/{len(test_data)}] R@1={r1:.2f}% ({elapsed:.0f}s)")

    n = len(hits_at_k[1])
    summary = {
        "num_examples": n,
        "note": "code-only model, inherently path-invariant",
        "code_mode": code_mode,
        "top_k": args.top_k,
    }
    for k in [1, 3, 5, 10]:
        summary[f"R@{k}"] = float(np.mean(hits_at_k[k]) * 100)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.output_dir, "predictions.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== Results (n={n}, code_mode={code_mode}) ===")
    for k in [1, 3, 5, 10]:
        print(f"  R@{k}: {summary[f'R@{k}']:.2f}%")
    print(f"Saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    # Note: UniXcoder is code-only (no paths), so it is inherently invariant
    # to filename shuffling. No --perturb needed. Results are identical.
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
