#!/usr/bin/env python3
"""
Structural code context eval on SWE-bench: does enriched code context
(signatures + imports + class names) help vs raw code?

Tests whether the problem is impoverished code representation.
If structural context helps → the issue is HOW code is presented.
If structural context doesn't help → the issue is deeper.

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/eval_swebench_structural.py \
        --gpu_id 0 --output_dir /data/chenlibin/grepo_agent_experiments/swebench_structural
"""

import argparse
import ast
import json
import os
import random
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

PATH_PROMPT = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)

STRUCTURAL_PROMPT = (
    "Given the bug report, is this file likely to need modification? "
    "Consider the file's structure carefully.\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n"
    "Imports: {imports}\n"
    "Classes: {classes}\n"
    "Functions: {functions}\n\n"
    "Answer:"
)


def extract_structure(repo, fpath):
    """Extract imports, class names, function signatures from a Python file."""
    full = os.path.join(REPO_DIR, repo, fpath)
    if not os.path.isfile(full):
        return {"imports": "N/A", "classes": "N/A", "functions": "N/A"}

    try:
        with open(full, "r", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception:
        return {"imports": "N/A", "classes": "N/A", "functions": "N/A"}

    imports = []
    classes = []
    functions = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                imports.append(f"{mod}.{alias.name}")
        elif isinstance(node, ast.ClassDef):
            bases = [getattr(b, "id", getattr(b, "attr", "?")) for b in node.bases]
            classes.append(f"{node.name}({', '.join(bases)})" if bases else node.name)
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(f"{node.name}.{item.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)

    return {
        "imports": ", ".join(imports[:15]) or "none",
        "classes": ", ".join(classes[:10]) or "none",
        "functions": ", ".join(functions[:15]) or "none",
    }


def truncate_prompt(prompt, tokenizer, max_len):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_len:
        return prompt
    suffix_ids = tokenizer.encode("\n\nAnswer:", add_special_tokens=False)
    keep = max_len - len(suffix_ids) - 1
    return tokenizer.decode(ids[:keep] + suffix_ids, skip_special_tokens=True)


def score_batch(model, tok, prompts, yes_id, no_id, device, max_len=1024, bs=8):
    prompts = [truncate_prompt(p, tok, max_len) for p in prompts]
    scores = []
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i+bs]
        inputs = tok(batch, return_tensors="pt", padding=True,
                     truncation=True, max_length=max_len,
                     padding_side="left").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
        s = (logits[:, yes_id].float() - logits[:, no_id].float()).cpu().numpy()
        scores.extend(s.tolist())
    return scores


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
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    return m, tok, yes_id, no_id


def run_eval(model, tok, yes_id, no_id, device, test_data, use_structure, label):
    hits = []
    start = time.time()

    for i, rec in enumerate(test_data):
        repo = rec.get("repo", "")
        issue = rec["issue_text"]
        gt = set(rec.get("changed_py_files", rec.get("changed_files", [])))
        if not gt:
            continue

        # List .py files from repo as candidates
        repo_path = os.path.join(REPO_DIR, repo)
        if not os.path.isdir(repo_path):
            continue
        candidates = []
        for root, dirs, files in os.walk(repo_path):
            for f in files:
                if f.endswith(".py"):
                    rel = os.path.relpath(os.path.join(root, f), repo_path)
                    candidates.append(rel)
        candidates = candidates[:100]  # cap for speed
        if not candidates:
            continue

        prompts = []
        for c in candidates:
            if use_structure:
                struct = extract_structure(repo, c)
                prompts.append(STRUCTURAL_PROMPT.format(
                    issue_text=issue, candidate_path=c, **struct))
            else:
                prompts.append(PATH_PROMPT.format(
                    issue_text=issue, candidate_path=c))

        scores = score_batch(model, tok, prompts, yes_id, no_id, device)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        top1 = ranked[0][0]
        hit = 1.0 if top1 in gt else 0.0
        hits.append(hit)

        if (i + 1) % 20 == 0:
            print(f"  [{label}] [{i+1}] Hit@1={np.mean(hits)*100:.1f}% ({time.time()-start:.0f}s)")

    return float(np.mean(hits) * 100) if hits else 0, len(hits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    print("Loading data...")
    test_data = []
    with open(TEST_PATH) as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"  {len(test_data)} SWE-bench examples")

    print("Loading model...")
    model, tok, yes_id, no_id = load_model(device)

    print("\n=== Path-only eval ===")
    path_r1, n1 = run_eval(model, tok, yes_id, no_id, device, test_data,
                            use_structure=False, label="path")
    print(f"\n  Path-only: Hit@1={path_r1:.2f}% (n={n1})")

    print("\n=== Structural context eval ===")
    struct_r1, n2 = run_eval(model, tok, yes_id, no_id, device, test_data,
                              use_structure=True, label="struct")
    print(f"\n  Structural: Hit@1={struct_r1:.2f}% (n={n2})")

    delta = struct_r1 - path_r1
    summary = {
        "path_only_hit1": path_r1,
        "structural_hit1": struct_r1,
        "delta": delta,
        "n": min(n1, n2),
        "benchmark": "swebench_lite",
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Result ===")
    print(f"Path: {path_r1:.2f}%  Structural: {struct_r1:.2f}%  Delta: {delta:+.2f}pp")


if __name__ == "__main__":
    main()
