#!/usr/bin/env python3
"""
Pairwise code evaluation on SWE-bench: does code help disambiguate
GT vs hard-negative when both share similar paths?

Instead of pointwise scoring, we do PAIRWISE comparison:
Given (issue, file_A, file_B), ask "which one needs modification?"
Test: does adding code to both files improve pairwise accuracy?

This addresses Reviewer concern: "maybe code fails only under pointwise objective"

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/eval_swebench_pairwise.py \
        --gpu_id 0 --output_dir /data/chenlibin/grepo_agent_experiments/swebench_pairwise
"""

import argparse
import ast
import hashlib
import json
import os
import random
import re
import time

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
LORA_PATH = "/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/best"
TEST_PATH = "/home/chenlibin/grepo_agent/data/swebench_lite/swebench_lite_test.jsonl"
REPO_DIR = "/home/chenlibin/grepo_agent/data/swebench_lite/repos"

# Pairwise prompts (preserve the old-script formats verbatim)
PAIR_PATH_PROMPT = (
    "Given the bug report, which file is more likely to need modification? "
    "Answer A or B.\n\n"
    "Bug Report: {issue_text}\n\n"
    "File A: {file_a}\n"
    "File B: {file_b}\n\n"
    "Answer:"
)

PAIR_CODE_PROMPT = (
    "Given the bug report, which file is more likely to need modification? "
    "Consider both the file paths and code content. Answer A or B.\n\n"
    "Bug Report: {issue_text}\n\n"
    "File A: {file_a}\n"
    "Code A:\n{code_a}\n\n"
    "File B: {file_b}\n"
    "Code B:\n{code_b}\n\n"
    "Answer:"
)

PAIR_CODE_ONLY_PROMPT = (
    "Given the bug report, which code is more likely to need modification? "
    "Answer A or B.\n\n"
    "Bug Report: {issue_text}\n\n"
    "Code A:\n{code_a}\n\n"
    "Code B:\n{code_b}\n\n"
    "Answer:"
)


def hash_path(p):
    parts = []
    for x in p.split("/"):
        if not x:
            continue
        h = hashlib.sha256(x.encode()).hexdigest()[:8]
        if x.endswith(".py"):
            parts.append(f"m_{h}.py")
        else:
            parts.append(f"d_{h}")
    return "/".join(parts)


def read_file_head(repo, fpath, max_lines=50):
    full = os.path.join(REPO_DIR, repo, fpath)
    if not os.path.isfile(full):
        return "# (not available)"
    try:
        with open(full, "r", errors="replace") as f:
            return "".join(f.readlines()[:max_lines])
    except Exception:
        return "# (unreadable)"


def find_hard_negative(gt_file, candidates):
    """Find the most path-similar non-GT candidate."""
    gt_dir = os.path.dirname(gt_file)
    gt_stem = os.path.splitext(os.path.basename(gt_file))[0]

    # Priority: same dir > same stem > first candidate
    for c in candidates:
        if c == gt_file:
            continue
        if os.path.dirname(c) == gt_dir:
            return c
    for c in candidates:
        if c == gt_file:
            continue
        c_stem = os.path.splitext(os.path.basename(c))[0]
        if c_stem == gt_stem or gt_stem in c_stem or c_stem in gt_stem:
            return c
    # Fallback: first non-GT
    for c in candidates:
        if c != gt_file:
            return c
    return None


def load_model(device):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
                                              device_map={"": device},
                                              trust_remote_code=True,
                                              torch_dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, LORA_PATH)
    m.eval()
    return m, tok


def score_pair(model, tok, prompt, device):
    """Return P(A) - P(B) from the model."""
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=2048).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
    # Get logits for "A" and "B" tokens
    a_id = tok.encode("A", add_special_tokens=False)[0]
    b_id = tok.encode("B", add_special_tokens=False)[0]
    return (logits[0, a_id] - logits[0, b_id]).item()


def main():
    global MODEL_PATH, LORA_PATH, TEST_PATH, REPO_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--variant", choices=["path_code", "hash_code", "code_only"],
                        default="path_code",
                        help="path_code = original path+code (86% baseline); "
                             "hash_code = SHA-256 hashed paths + code; "
                             "code_only = no paths, only code")
    parser.add_argument("--code_lines", type=int, default=50,
                        help="Lines of code to read from file head")
    parser.add_argument("--code_chars", type=int, default=800,
                        help="Char cap per code block (matches old 86% config)")
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--lora_path", default=LORA_PATH)
    parser.add_argument("--test_path", default=TEST_PATH)
    parser.add_argument("--bm25_path",
                        default="/home/chenlibin/grepo_agent/data/swebench_lite/swebench_perturb_shuffle_filenames_candidates.jsonl")
    parser.add_argument("--repo_dir", default=REPO_DIR)
    args = parser.parse_args()
    MODEL_PATH = args.model_path
    LORA_PATH = args.lora_path
    TEST_PATH = args.test_path
    REPO_DIR = args.repo_dir

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    print("Loading data...")
    test_data = []
    with open(TEST_PATH) as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"  {len(test_data)} SWE-bench examples")

    # Load BM25 candidates
    bm25_data = {}
    if os.path.isfile(args.bm25_path):
        with open(args.bm25_path) as f:
            for line in f:
                r = json.loads(line)
                bm25_data[(r["repo"], str(r["issue_id"]))] = r
    print(f"  Loaded {len(bm25_data)} BM25 records from {args.bm25_path}")

    print("Loading model...")
    model, tok = load_model(device)

    path_correct = []
    code_correct = []
    results = []
    start = time.time()

    for i, rec in enumerate(test_data):
        repo = rec.get("repo_full", rec.get("repo", ""))
        issue = rec["issue_text"][:2000]
        gt_files = set(rec.get("changed_py_files", rec.get("changed_files", [])))
        if not gt_files:
            continue

        gt_file = list(gt_files)[0]

        # Find hard negative from BM25 or repo files
        key = (rec.get("repo", ""), str(rec.get("issue_id", "")))
        candidates = []
        if key in bm25_data:
            candidates = bm25_data[key].get("bm25_candidates",
                                             bm25_data[key].get("candidates", []))

        # If no BM25 candidates, list .py files from repo
        if not candidates:
            repo_path = os.path.join(REPO_DIR, repo)
            if os.path.isdir(repo_path):
                for root, dirs, files in os.walk(repo_path):
                    for f in files:
                        if f.endswith(".py"):
                            rel = os.path.relpath(os.path.join(root, f), repo_path)
                            candidates.append(rel)
                            if len(candidates) > 200:
                                break

        neg_file = find_hard_negative(gt_file, candidates)
        if neg_file is None:
            continue

        # Randomize order to avoid position bias
        if random.random() < 0.5:
            file_a, file_b = gt_file, neg_file
            gt_is_a = True
        else:
            file_a, file_b = neg_file, gt_file
            gt_is_a = False

        # Path-only pairwise
        prompt_path = PAIR_PATH_PROMPT.format(
            issue_text=issue, file_a=file_a, file_b=file_b)
        score_path = score_pair(model, tok, prompt_path, device)
        path_pred_a = score_path > 0
        path_hit = (path_pred_a == gt_is_a)
        path_correct.append(1.0 if path_hit else 0.0)

        # Code pairwise (three variants)
        code_a = read_file_head(repo, file_a if gt_is_a else neg_file,
                                max_lines=args.code_lines)
        code_b = read_file_head(repo, file_b if not gt_is_a else neg_file,
                                max_lines=args.code_lines)
        if args.variant == "path_code":
            prompt_code = PAIR_CODE_PROMPT.format(
                issue_text=issue[:1000], file_a=file_a, file_b=file_b,
                code_a=code_a[:args.code_chars], code_b=code_b[:args.code_chars])
        elif args.variant == "hash_code":
            prompt_code = PAIR_CODE_PROMPT.format(
                issue_text=issue[:1000],
                file_a=hash_path(file_a), file_b=hash_path(file_b),
                code_a=code_a[:args.code_chars], code_b=code_b[:args.code_chars])
        elif args.variant == "code_only":
            prompt_code = PAIR_CODE_ONLY_PROMPT.format(
                issue_text=issue[:1000],
                code_a=code_a[:args.code_chars], code_b=code_b[:args.code_chars])
        score_code = score_pair(model, tok, prompt_code, device)
        code_pred_a = score_code > 0
        code_hit = (code_pred_a == gt_is_a)
        code_correct.append(1.0 if code_hit else 0.0)

        results.append({
            "repo": repo, "gt_file": gt_file, "neg_file": neg_file,
            "path_hit": path_hit, "code_hit": code_hit,
        })

        if (i + 1) % 20 == 0:
            p_acc = np.mean(path_correct) * 100
            c_acc = np.mean(code_correct) * 100
            print(f"  [{i+1}] path_pair={p_acc:.1f}% code_pair={c_acc:.1f}% ({time.time()-start:.0f}s)")

    n = len(path_correct)
    summary = {
        "num_examples": n,
        "variant": args.variant,
        "code_lines": args.code_lines,
        "code_chars": args.code_chars,
        "path_pairwise_acc": float(np.mean(path_correct) * 100),
        "code_pairwise_acc": float(np.mean(code_correct) * 100),
        "delta": float((np.mean(code_correct) - np.mean(path_correct)) * 100),
        "benchmark": "swebench_lite",
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Pairwise Results (n={n}) ===")
    print(f"Path-only pairwise: {summary['path_pairwise_acc']:.2f}%")
    print(f"Code pairwise:      {summary['code_pairwise_acc']:.2f}%")
    print(f"Delta:              {summary['delta']:+.2f}pp")


if __name__ == "__main__":
    main()
