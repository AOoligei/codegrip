#!/usr/bin/env python3
"""
Hierarchical path->code bug localization evaluation.

Stage A: path-only reranker selects top-k files
Stage B: extract functions from each file (AST), BM25 rank by issue text
Stage C: Qwen2.5-7B scores (issue, file_path, top-m function snippets) -- zero-shot

Reports decomposed metrics:
  - Stage A: top-k gold-file recall
  - Stage C: file-level R@1 conditioned on gold-in-top-k
  - Stage C: overall R@1 (file-level)
  - Under original and filename-shuffle conditions

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/eval_hierarchical.py \
        --gpu_id 0 --top_k_files 10 --top_m_funcs 3 \
        --perturb none \
        --output_dir /data/chenlibin/grepo_agent_experiments/hierarchical_eval
"""

import argparse
import ast
import json
import os
import random
import re
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from peft import PeftModel
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Paths
MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
LORA_PATH = "/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/best"
TEST_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_test.jsonl"
BM25_PATH = "/home/chenlibin/grepo_agent/data/rankft/merged_bm25_exp6_candidates.jsonl"
REPO_DIR = "/home/chenlibin/grepo_agent/data/repos"

# Prompt templates
PATH_ONLY_PROMPT = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)

HIERARCHICAL_PROMPT = (
    "Given the bug report, is this file likely to need modification? "
    "Consider both the file path and the code snippets shown below.\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n"
    "Relevant functions:\n{function_snippets}\n\n"
    "Answer:"
)


# ============================================================
# Function extraction via AST
# ============================================================

def extract_functions_from_file(repo_name: str, file_path: str,
                                 max_lines: int = 30) -> List[Dict]:
    """Extract function/method definitions from a Python file using AST.

    Returns list of {"name": str, "body": str, "lineno": int}.
    """
    full_path = os.path.join(REPO_DIR, repo_name, file_path)
    if not os.path.isfile(full_path):
        return []

    try:
        with open(full_path, "r", errors="replace") as f:
            source = f.read()
    except Exception:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = min(start + max_lines, len(lines))
            body = "\n".join(lines[start:end])
            functions.append({
                "name": node.name,
                "body": body,
                "lineno": node.lineno,
            })

    return functions


def bm25_rank_functions(issue_text: str, functions: List[Dict]) -> List[Dict]:
    """Rank functions by BM25 similarity to issue text."""
    if not functions:
        return []

    issue_tokens = issue_text.lower().split()
    func_tokens = []
    for f in functions:
        tokens = f["body"].lower().split()
        name_tokens = re.split(r'[_]', f["name"].lower())
        func_tokens.append(tokens + name_tokens)

    bm25 = BM25Okapi(func_tokens)
    scores = bm25.get_scores(issue_tokens)

    ranked = sorted(zip(functions, scores), key=lambda x: -x[1])
    return [f for f, s in ranked]


# ============================================================
# Model scoring
# ============================================================

def load_model(model_path, lora_path, gpu_id):
    """Load 4-bit quantized model with LoRA."""
    device = f"cuda:{gpu_id}"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # critical: ensures logits[:, -1, :] reads the right token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": device},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    yes_id, no_id = yes_ids[0], no_ids[0]

    return model, tokenizer, yes_id, no_id, device


def score_candidates(model, tokenizer, prompts, yes_id, no_id,
                     max_seq_length, device, batch_size=4):
    """Score a batch of prompts, return P(Yes) - P(No) for each."""
    scores = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_seq_length,
                           padding_side="left").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        yes_logits = logits[:, yes_id].float()
        no_logits = logits[:, no_id].float()
        batch_scores = (yes_logits - no_logits).cpu().numpy()
        scores.extend(batch_scores.tolist())
    return scores


# ============================================================
# Perturbation
# ============================================================

def shuffle_filenames(paths: List[str]) -> Dict[str, str]:
    """Shuffle filenames within each directory."""
    dir_to_files = defaultdict(list)
    for p in paths:
        parts = p.rsplit("/", 1)
        if len(parts) == 2:
            dir_to_files[parts[0]].append(parts[1])
        else:
            dir_to_files[""].append(parts[0])

    mapping = {}
    for dir_path, filenames in dir_to_files.items():
        shuffled = filenames.copy()
        random.shuffle(shuffled)
        for orig, new in zip(filenames, shuffled):
            orig_full = f"{dir_path}/{orig}" if dir_path else orig
            new_full = f"{dir_path}/{new}" if dir_path else new
            mapping[orig_full] = new_full
    return mapping


# ============================================================
# Main evaluation
# ============================================================

def load_data():
    """Load test + candidate data."""
    test_data = {}
    with open(TEST_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            test_data[key] = rec

    bm25_data = {}
    with open(BM25_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            bm25_data[key] = rec

    return test_data, bm25_data


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    test_data, bm25_data = load_data()
    print(f"  {len(test_data)} test, {len(bm25_data)} candidates")

    print(f"Loading model (LoRA: {args.lora_path})...")
    model, tokenizer, yes_id, no_id, device = load_model(
        MODEL_PATH, args.lora_path, args.gpu_id)
    print(f"  Yes={yes_id}, No={no_id}")

    # Metrics
    stage_a_recall = []
    overall_hits = []
    conditioned_hits = []
    path_only_hits = []
    func_extract_stats = {"found": 0, "empty": 0}

    results_per_example = []
    start_time = time.time()

    for idx, (key, test_rec) in enumerate(test_data.items()):
        if key not in bm25_data:
            continue

        repo = test_rec["repo"]
        issue_text = test_rec["issue_text"]
        gt_files = set(test_rec.get("changed_py_files",
                                     test_rec.get("changed_files", [])))
        if not gt_files:
            continue

        bm25_rec = bm25_data[key]
        candidates = bm25_rec.get("candidates",
                                   bm25_rec.get("bm25_candidates", []))[:200]

        # Apply perturbation to paths shown to model (not to repo access)
        display_candidates = list(candidates)
        display_gt = set(gt_files)
        if args.perturb == "shuffle_filenames":
            all_paths = list(candidates) + [g for g in gt_files if g not in candidates]
            path_mapping = shuffle_filenames(all_paths)
            display_candidates = [path_mapping.get(c, c) for c in candidates]
            display_gt = {path_mapping.get(g, g) for g in gt_files}

        # ---- Stage A: path-only scoring for top-k ----
        path_prompts = [
            PATH_ONLY_PROMPT.format(issue_text=issue_text, candidate_path=c)
            for c in display_candidates
        ]
        path_scores = score_candidates(
            model, tokenizer, path_prompts, yes_id, no_id,
            args.max_seq_length, device, batch_size=args.batch_size)

        ranked_by_path = sorted(zip(display_candidates, candidates, path_scores),
                                key=lambda x: -x[2])
        path_top1_display = ranked_by_path[0][0]
        path_only_hit = 1.0 if path_top1_display in display_gt else 0.0
        path_only_hits.append(path_only_hit)

        top_k_display = [r[0] for r in ranked_by_path[:args.top_k_files]]
        top_k_original = [r[1] for r in ranked_by_path[:args.top_k_files]]

        # Stage A recall
        gt_in_topk = bool(display_gt & set(top_k_display))
        stage_a_recall.append(1.0 if gt_in_topk else 0.0)

        # ---- Stage B: extract & rank functions for top-k files ----
        file_snippets = {}
        for disp_path, orig_path in zip(top_k_display, top_k_original):
            funcs = extract_functions_from_file(repo, orig_path,
                                                 max_lines=args.func_max_lines)
            if funcs:
                func_extract_stats["found"] += 1
                ranked_funcs = bm25_rank_functions(issue_text, funcs)
                top_funcs = ranked_funcs[:args.top_m_funcs]
                snippet_text = ""
                for f in top_funcs:
                    snippet_text += f"# {f['name']} (line {f['lineno']})\n"
                    snippet_text += f["body"][:500] + "\n\n"
                file_snippets[disp_path] = snippet_text.strip()
            else:
                func_extract_stats["empty"] += 1
                file_snippets[disp_path] = "# (no functions extracted)"

        # ---- Stage C: hierarchical scoring ----
        # Truncate issue_text to leave room for file path + function snippets + template
        # Budget: 3 funcs x 30 lines x ~10 tok/line = ~900 + template ~50 + path ~20
        # Reserve 1000 tokens for non-issue content
        max_issue_tokens = max(200, args.max_seq_length - 1000)
        issue_truncated = tokenizer.decode(
            tokenizer.encode(issue_text, add_special_tokens=False)[:max_issue_tokens],
            skip_special_tokens=True,
        )

        hier_prompts = []
        for fpath in top_k_display:
            prompt = HIERARCHICAL_PROMPT.format(
                issue_text=issue_truncated,
                candidate_path=fpath,
                function_snippets=file_snippets.get(fpath, "# (no functions)"),
            )
            hier_prompts.append(prompt)

        hier_scores = score_candidates(
            model, tokenizer, hier_prompts, yes_id, no_id,
            args.max_seq_length, device, batch_size=args.batch_size)

        ranked_by_hier = sorted(zip(top_k_display, hier_scores), key=lambda x: -x[1])
        hier_top1 = ranked_by_hier[0][0]

        hier_hit = 1.0 if hier_top1 in display_gt else 0.0
        overall_hits.append(hier_hit)

        if gt_in_topk:
            conditioned_hits.append(hier_hit)

        results_per_example.append({
            "repo": repo,
            "issue_id": str(test_rec["issue_id"]),
            "gt_in_topk": gt_in_topk,
            "path_only_hit": path_only_hit,
            "hier_hit": hier_hit,
        })

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            r1_path = np.mean(path_only_hits) * 100
            r1_hier = np.mean(overall_hits) * 100
            sa_rec = np.mean(stage_a_recall) * 100
            cond_r1 = np.mean(conditioned_hits) * 100 if conditioned_hits else 0
            print(f"  [{idx+1}] path={r1_path:.1f}% hier={r1_hier:.1f}% "
                  f"stgA={sa_rec:.1f}% cond={cond_r1:.1f}% ({elapsed:.0f}s)")

    # ---- Summary ----
    n = len(overall_hits)
    summary = {
        "num_examples": n,
        "top_k_files": args.top_k_files,
        "top_m_funcs": args.top_m_funcs,
        "func_max_lines": args.func_max_lines,
        "perturb": args.perturb,
        "path_only_R@1": float(np.mean(path_only_hits) * 100),
        "stage_a_hit_at_k": float(np.mean(stage_a_recall) * 100),
        "hierarchical_R@1": float(np.mean(overall_hits) * 100),
        "conditioned_R@1": float(np.mean(conditioned_hits) * 100) if conditioned_hits else 0,
        "conditioned_n": len(conditioned_hits),
        "func_extract_found": func_extract_stats["found"],
        "func_extract_empty": func_extract_stats["empty"],
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.output_dir, "per_example.jsonl"), "w") as f:
        for r in results_per_example:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== Results (n={n}, top_k={args.top_k_files}, top_m={args.top_m_funcs}, perturb={args.perturb}) ===")
    print(f"Path-only R@1:        {summary['path_only_R@1']:.2f}%")
    print(f"Stage A top-k recall: {summary['stage_a_hit_at_k']:.2f}%")
    print(f"Hierarchical R@1:     {summary['hierarchical_R@1']:.2f}%")
    print(f"Conditioned R@1:      {summary['conditioned_R@1']:.2f}% (n={summary['conditioned_n']})")
    print(f"Func extraction:      {func_extract_stats['found']} found, {func_extract_stats['empty']} empty")
    print(f"Saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--lora_path", type=str, default=LORA_PATH,
                        help="Path to LoRA adapter (default: path-only reranker)")
    parser.add_argument("--top_k_files", type=int, default=10)
    parser.add_argument("--top_m_funcs", type=int, default=3)
    parser.add_argument("--func_max_lines", type=int, default=30)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--perturb", choices=["none", "shuffle_filenames"],
                        default="none")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
