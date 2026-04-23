#!/usr/bin/env python3
"""
Hierarchical function-level localization using cross-encoder reranking.

Two-stage approach:
1. Take top-K file predictions from existing file-level predictions
2. For each predicted file, extract functions from file summaries
3. Score each (issue, file, function) triple with the RankFT cross-encoder
4. Rank functions by score, compute function-level Hit@K

This provides true function-level localization using the same cross-encoder
that was trained for file-level relevance judgments.

Usage:
    python scripts/eval_function_level_rerank.py \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path experiments/exp6_warmstart_cochange/stage2_sft/final \
        --predictions experiments/exp6_warmstart_cochange/eval_filetree/predictions.jsonl \
        --test_data data/grepo_text/grepo_test.jsonl \
        --function_index data/function_index_aligned.json \
        --output_dir experiments/exp6_warmstart_cochange/eval_func_rerank \
        --gpu_id 4
"""
import json
import os
import re
import argparse
import time
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# Prompt template — function-level variant
# ============================================================

PROMPT_TEMPLATE_FUNC = (
    "Given the bug report, is this function likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {file_path}\n"
    "Function: {function_name}\n\n"
    "Answer:"
)

PROMPT_TEMPLATE_FUNC_WITH_CONTEXT = (
    "Given the bug report, is this function likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {file_path}\n"
    "File Content: {file_summary}\n"
    "Function: {function_name}\n\n"
    "Answer:"
)


def build_func_prompt(issue_text, file_path, function_name, file_summary=None):
    if file_summary:
        return PROMPT_TEMPLATE_FUNC_WITH_CONTEXT.format(
            issue_text=issue_text,
            file_path=file_path,
            function_name=function_name,
            file_summary=file_summary,
        )
    return PROMPT_TEMPLATE_FUNC.format(
        issue_text=issue_text,
        file_path=file_path,
        function_name=function_name,
    )


# ============================================================
# Function index (AST-based, not truncated summaries)
# ============================================================

def build_file_function_map(func_index, repo):
    """Build file -> [(name, name), ...] for a repo from AST function index.

    Returns pairs of (name, name) for compatibility with scoring code.
    """
    repo_funcs = func_index.get(repo, {})
    result = {}
    for fpath, names in repo_funcs.items():
        if names:
            result[fpath] = [(n, n) for n in names]
    return result


# ============================================================
# Model scoring
# ============================================================

def get_yes_no_token_ids(tokenizer):
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    return yes_ids[0], no_ids[0]


@torch.no_grad()
def score_functions_batched(
    model, tokenizer, issue_text,
    file_func_pairs,  # list of (file_path, func_name, file_summary)
    yes_id, no_id, max_seq_length, device,
    batch_size=16,
):
    """Score (file, function) pairs using cross-encoder.

    Returns list of scores (logit_yes - logit_no).
    """
    if not file_func_pairs:
        return []

    prompts = []
    for file_path, func_name, file_summary in file_func_pairs:
        prompts.append(build_func_prompt(issue_text, file_path, func_name, file_summary))

    all_scores = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        encodings = tokenizer(
            batch_prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=max_seq_length,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                # Fallback: score one at a time
                for prompt in batch_prompts:
                    enc = tokenizer([prompt], return_tensors="pt",
                                    truncation=True, max_length=max_seq_length)
                    ids = enc["input_ids"].to(device)
                    mask = enc["attention_mask"].to(device)
                    out = model(input_ids=ids, attention_mask=mask)
                    score = (out.logits[0, -1, yes_id] - out.logits[0, -1, no_id]).item()
                    all_scores.append(score)
                continue
            raise

        logits = outputs.logits
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.size(0), device=device)
        last_logits = logits[batch_indices, seq_lengths]
        scores = (last_logits[:, yes_id] - last_logits[:, no_id]).cpu().tolist()
        all_scores.extend(scores)

    return all_scores


# ============================================================
# Evaluation
# ============================================================

def compute_func_hit_at_k(ranked_funcs, gt_functions, k):
    """Function-level Hit@K: fraction of GT functions in top-K predictions."""
    if not gt_functions:
        return None
    gt_set = set(gt_functions)
    pred_set = set(ranked_funcs[:k])
    return len(pred_set & gt_set) / len(gt_set) * 100


def compute_qualified_func_hit_at_k(ranked_file_func_pairs, gt_functions, gt_files,
                                      file_func_map, k):
    """File-qualified function Hit@K.

    A GT function is "hit" only if we predict a (file, function) pair where:
    1. The file is a GT file
    2. The function name matches a GT function name
    """
    if not gt_functions:
        return None

    # Build GT (file, func) pairs
    gt_pairs = set()
    for gt_file in gt_files:
        file_funcs = file_func_map.get(gt_file, [])
        file_bare_names = {bare for bare, _ in file_funcs}
        for func in gt_functions:
            if func in file_bare_names:
                gt_pairs.add((gt_file, func))

    if not gt_pairs:
        return None

    # Check top-K predictions
    pred_pairs = set()
    for file_path, func_name in ranked_file_func_pairs[:k]:
        pred_pairs.add((file_path, func_name))

    hits = len(pred_pairs & gt_pairs)
    return hits / len(gt_pairs) * 100


def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    # Load test data (for GT functions)
    print(f"Loading test data from {args.test_data}...")
    test_map = {}
    total_with_funcs = 0
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            key = f"{item['repo']}_{item['issue_id']}"
            test_map[key] = item
            if item.get("changed_functions"):
                total_with_funcs += 1
    print(f"  {len(test_map)} examples, {total_with_funcs} with function-level GT")

    # Load file-level predictions
    print(f"Loading file-level predictions from {args.predictions}...")
    predictions = []
    with open(args.predictions) as f:
        for line in f:
            predictions.append(json.loads(line))
    print(f"  {len(predictions)} predictions")

    # Load AST-based function index
    print(f"Loading function index from {args.function_index}...")
    with open(args.function_index) as f:
        func_index = json.load(f)
    print(f"  {len(func_index)} repos")

    # Load file summaries (optional, for prompt context)
    file_summaries = {}
    if args.file_summaries and os.path.exists(args.file_summaries):
        print(f"Loading file summaries from {args.file_summaries}...")
        with open(args.file_summaries) as f:
            file_summaries = json.load(f)
        print(f"  {len(file_summaries)} repos with summaries")

    # Load model
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    yes_id, no_id = get_yes_no_token_ids(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True,
    )
    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()
    print(f"  Model loaded on {device}")

    # Build per-repo function maps (cached)
    func_maps = {}

    k_values = [1, 3, 5, 10, 20]
    # Bare name matching
    bare_metrics = {f"func_hit@{k}": [] for k in k_values}
    # File-qualified matching
    qual_metrics = {f"qual_func_hit@{k}": [] for k in k_values}
    # File-level for comparison
    file_metrics = {f"file_hit@{k}": [] for k in k_values}

    results = []
    skipped = 0
    no_funcs_in_summary = 0
    total = len(predictions)
    start_time = time.time()

    for idx, pred in enumerate(predictions):
        repo = pred.get("repo", "")
        issue_id = pred.get("issue_id", "")
        key = f"{repo}_{issue_id}"

        test_item = test_map.get(key)
        if not test_item:
            skipped += 1
            continue

        gt_functions = test_item.get("changed_functions", [])
        if not gt_functions:
            skipped += 1
            continue

        gt_files = set(test_item.get("changed_py_files", []))
        issue_text = test_item.get("issue_text", "")
        predicted_files = pred.get("predicted", pred.get("predicted_files", []))

        # Build func map for this repo
        if repo not in func_maps:
            func_maps[repo] = build_file_function_map(func_index, repo)
        func_map = func_maps[repo]

        # Take top-N files for function expansion
        top_files = predicted_files[:args.top_n_files]

        # Expand to (file, function) pairs
        file_func_pairs = []  # (file_path, func_bare_name, file_summary)
        for fpath in top_files:
            funcs_in_file = func_map.get(fpath, [])
            file_summary = file_summaries.get(repo, {}).get(fpath, "")
            for bare_name, qualified_name in funcs_in_file:
                file_func_pairs.append((fpath, bare_name, file_summary))

        if not file_func_pairs:
            no_funcs_in_summary += 1
            # Fall back to file-level only
            ranked_funcs = []
            ranked_file_func_pairs = []
        else:
            # Score all (file, function) pairs
            scores = score_functions_batched(
                model, tokenizer, issue_text,
                file_func_pairs,
                yes_id, no_id, args.max_seq_length, device,
                batch_size=args.score_batch_size,
            )

            # Sort by score
            scored = sorted(
                zip(file_func_pairs, scores),
                key=lambda x: -x[1]
            )

            ranked_funcs = [bare_name for (_, bare_name, _), _ in scored]
            ranked_file_func_pairs = [(fpath, bare_name) for (fpath, bare_name, _), _ in scored]

        # Deduplicate bare names (keep first/highest scored)
        seen_bare = set()
        deduped_funcs = []
        for name in ranked_funcs:
            if name not in seen_bare:
                seen_bare.add(name)
                deduped_funcs.append(name)

        # Deduplicate (file, func) pairs
        seen_pairs = set()
        deduped_pairs = []
        for pair in ranked_file_func_pairs:
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                deduped_pairs.append(pair)

        # Bare name function Hit@K
        for k in k_values:
            hit = compute_func_hit_at_k(deduped_funcs, gt_functions, k)
            if hit is not None:
                bare_metrics[f"func_hit@{k}"].append(hit)

        # File-qualified function Hit@K
        for k in k_values:
            hit = compute_qualified_func_hit_at_k(
                deduped_pairs, gt_functions, gt_files, func_map, k)
            if hit is not None:
                qual_metrics[f"qual_func_hit@{k}"].append(hit)

        # File-level Hit@K
        for k in k_values:
            top_k = set(predicted_files[:k])
            fhit = len(top_k & gt_files) / len(gt_files) * 100 if gt_files else 0
            file_metrics[f"file_hit@{k}"].append(fhit)

        results.append({
            "repo": repo,
            "issue_id": issue_id,
            "ground_truth_files": list(gt_files),
            "ground_truth_functions": gt_functions,
            "predicted_files": predicted_files[:20],
            "predicted_functions": [
                {"file": f, "function": fn}
                for f, fn in deduped_pairs[:30]
            ],
            "n_candidates_scored": len(file_func_pairs),
        })

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / max(elapsed, 1)
            eta = (total - idx - 1) / max(rate, 0.001)
            qh1 = np.mean(qual_metrics["qual_func_hit@1"]) if qual_metrics["qual_func_hit@1"] else 0
            print(f"  [{idx+1}/{total}] Qual Func H@1: {qh1:.2f}% | "
                  f"ETA: {eta:.0f}s")

    elapsed_total = time.time() - start_time

    # Print results
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    print(f"\n{'='*65}")
    print(f"HIERARCHICAL FUNCTION-LEVEL LOCALIZATION")
    print(f"{'='*65}")
    print(f"Model: {args.model_path}")
    if args.lora_path:
        print(f"LoRA: {args.lora_path}")
    print(f"Top-N files expanded: {args.top_n_files}")
    print(f"Evaluated: {len(results)} (skipped {skipped}, no funcs in summary: {no_funcs_in_summary})")
    print(f"Time: {elapsed_total:.0f}s ({elapsed_total/3600:.2f}h)")

    print(f"\n{'Metric':<35} {'Score':>8} {'N':>6}")
    print("-" * 55)
    print("  File-qualified func Hit@K (primary):")
    for k in k_values:
        key = f"qual_func_hit@{k}"
        print(f"    Qual Func Hit@{k:<5}          {avg(qual_metrics[key]):>7.2f}%  {len(qual_metrics[key]):>5}")

    print()
    print("  Bare name func Hit@K (secondary):")
    for k in k_values:
        key = f"func_hit@{k}"
        print(f"    Func Hit@{k:<5}               {avg(bare_metrics[key]):>7.2f}%  {len(bare_metrics[key]):>5}")

    print()
    print("  File-level Hit@K (reference):")
    for k in k_values:
        key = f"file_hit@{k}"
        print(f"    File Hit@{k:<5}               {avg(file_metrics[key]):>7.2f}%  {len(file_metrics[key]):>5}")

    # Save
    summary = {
        "qualified_func_level": {
            f"func_hit@{k}": avg(qual_metrics[f"qual_func_hit@{k}"])
            for k in k_values
        },
        "bare_func_level": {
            f"func_hit@{k}": avg(bare_metrics[f"func_hit@{k}"])
            for k in k_values
        },
        "file_level": {
            f"file_hit@{k}": avg(file_metrics[f"file_hit@{k}"])
            for k in k_values
        },
        "config": {
            "model_path": args.model_path,
            "lora_path": args.lora_path,
            "predictions": args.predictions,
            "top_n_files": args.top_n_files,
            "max_seq_length": args.max_seq_length,
        },
        "n_evaluated": len(results),
        "n_skipped": skipped,
        "n_no_funcs_in_summary": no_funcs_in_summary,
        "wall_clock_seconds": round(elapsed_total, 2),
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    with open(pred_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data/shuyang/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_path", default=None,
                        help="Path to LoRA adapter (SFT or RankFT)")
    parser.add_argument("--predictions", required=True,
                        help="File-level predictions.jsonl to expand")
    parser.add_argument("--test_data", default="data/grepo_text/grepo_test.jsonl")
    parser.add_argument("--function_index", default="data/function_index_aligned.json",
                        help="AST-based function index (from build_function_index.py)")
    parser.add_argument("--file_summaries", default="data/file_summaries_aligned.json",
                        help="File summaries for prompt context (optional)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--top_n_files", type=int, default=10,
                        help="Number of top file predictions to expand to functions")
    parser.add_argument("--max_seq_length", type=int, default=768)
    parser.add_argument("--score_batch_size", type=int, default=16)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
