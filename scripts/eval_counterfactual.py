#!/usr/bin/env python3
"""
Evaluate models on counterfactual conflict test data.

Tests whether rerankers use code content or just file paths by swapping
code between correct and incorrect files across four conditions:
  - original: no swap
  - prcw: path=right, code=wrong (GT file gets wrong code)
  - pwcr: path=wrong, code=right (wrong file gets GT code)
  - crossed: both swaps applied simultaneously
"""
import os
import sys
import json
import hashlib
import argparse
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE_PATH_ONLY = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)

PROMPT_TEMPLATE_CODE_AWARE = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Code:\n{code_content}\n\n"
    "Answer:"
)


def anonymize_path(path: str) -> str:
    h = hashlib.md5(path.encode()).hexdigest()[:4]
    return f"file_{h}.py"


# ---------------------------------------------------------------------------
# Code reading
# ---------------------------------------------------------------------------

def find_repo_dir(repo_name: str, repo_base: str) -> str:
    candidates = [
        os.path.join(repo_base, repo_name),
        os.path.join(repo_base, repo_name.replace("/", "__")),
        os.path.join(repo_base, repo_name.replace("/", "_")),
        os.path.join(repo_base, repo_name.split("/")[-1]),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return ""


def read_file_content(repo_dir: str, file_path: str, max_lines: int = 30) -> str:
    full_path = os.path.join(repo_dir, file_path)
    try:
        with open(full_path, "r", errors="ignore") as f:
            lines = f.readlines()[:max_lines]
        return "".join(lines)
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return "# (file content unavailable)"


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(issue_text: str, candidate_path: str,
                 prompt_mode: str, code_content: Optional[str] = None,
                 anonymize: bool = False) -> str:
    display_path = anonymize_path(candidate_path) if anonymize else candidate_path
    if prompt_mode == "path_only":
        return PROMPT_TEMPLATE_PATH_ONLY.format(
            issue_text=issue_text, candidate_path=display_path)
    else:
        content = code_content if code_content else "# (file content unavailable)"
        return PROMPT_TEMPLATE_CODE_AWARE.format(
            issue_text=issue_text, candidate_path=display_path,
            code_content=content)


# ---------------------------------------------------------------------------
# Scoring (follows eval_rankft_4bit.py)
# ---------------------------------------------------------------------------

def get_yes_no_token_ids(tokenizer):
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    return yes_ids[0], no_ids[0]


@torch.no_grad()
def score_candidates_batched(model, tokenizer, prompts, yes_id, no_id,
                              max_seq_length, device, batch_size=4):
    all_scores = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        encodings = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                              truncation=True, max_length=max_seq_length)
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                for prompt in batch_prompts:
                    enc = tokenizer([prompt], return_tensors="pt", truncation=True,
                                    max_length=max_seq_length)
                    ids = enc["input_ids"].to(device)
                    mask = enc["attention_mask"].to(device)
                    out = model(input_ids=ids, attention_mask=mask)
                    logits = out.logits[0, -1]
                    score = (logits[yes_id] - logits[no_id]).item()
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


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def get_code_for_candidate(candidate: str, repo_dir: str, code_overrides: Dict[str, str],
                           code_max_lines: int) -> str:
    """Get code content for a candidate, using override if present."""
    if candidate in code_overrides:
        # Overrides are already full code strings; truncate to max_lines
        lines = code_overrides[candidate].split("\n")[:code_max_lines]
        return "\n".join(lines)
    return read_file_content(repo_dir, candidate, max_lines=code_max_lines)


def score_condition(model, tokenizer, issue_text, candidates,
                    repo_dir, code_overrides, prompt_mode, anonymize,
                    code_max_lines, yes_id, no_id, max_seq_length,
                    device, batch_size):
    """Build prompts and score all candidates for one condition."""
    prompts = []
    for cand in candidates:
        if prompt_mode == "code_aware":
            code = get_code_for_candidate(cand, repo_dir, code_overrides, code_max_lines)
        else:
            code = None
        prompts.append(build_prompt(issue_text, cand, prompt_mode, code, anonymize))

    scores = score_candidates_batched(
        model, tokenizer, prompts, yes_id, no_id,
        max_seq_length, device, batch_size)
    return scores


def compute_rank(candidates, scores, target_file):
    """Return 1-indexed rank of target_file. Returns len(candidates)+1 if absent."""
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    for rank_idx, (cand, _) in enumerate(ranked):
        if cand == target_file:
            return rank_idx + 1
    return len(candidates) + 1


def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    # Load counterfactual data
    print(f"Loading counterfactual data from {args.counterfactual_data}...")
    cf_data = []
    with open(args.counterfactual_data) as f:
        for line in f:
            cf_data.append(json.loads(line))
    print(f"  Loaded {len(cf_data)} examples")

    # Load model
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model in 4-bit from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
    )

    if args.lora_path:
        print(f"Loading LoRA adapter from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path)

    model.eval()
    yes_id, no_id = get_yes_no_token_ids(tokenizer)
    print(f"  Yes={yes_id}, No={no_id}")
    print(f"  GPU memory: {torch.cuda.memory_allocated(device) / 1024**3:.1f} GB")
    print(f"  Prompt mode: {args.prompt_mode}")
    print(f"  Anonymize paths: {args.anonymize_paths}")

    # Conditions to evaluate
    conditions = ["original", "prcw", "pwcr", "crossed"]

    # Per-condition accumulators
    cond_gt_scores = {c: [] for c in conditions}
    cond_gt_ranks = {c: [] for c in conditions}
    cond_wrong_scores = {c: [] for c in conditions}
    cond_wrong_ranks = {c: [] for c in conditions}
    cond_hits_1 = {c: [] for c in conditions}
    cond_hits_5 = {c: [] for c in conditions}

    per_example_results = []
    start_time = time.time()

    for idx, item in enumerate(cf_data):
        repo = item["repo"]
        issue_id = str(item["issue_id"])
        issue_text = item.get("issue_text", item.get("text", ""))
        candidates = item["candidates"][:args.top_k]
        gt_files = set(item.get("changed_py_files", item.get("ground_truth", [])))
        swap_gt_file = item["swap_gt_file"]
        swap_wrong_file = item["swap_wrong_file"]

        repo_dir = find_repo_dir(repo, args.repo_dir)

        example_result = {
            "repo": repo,
            "issue_id": issue_id,
            "swap_gt_file": swap_gt_file,
            "swap_wrong_file": swap_wrong_file,
            "num_candidates": len(candidates),
        }

        for condition in conditions:
            # Build code_overrides for this condition
            if condition == "original":
                code_overrides = {}
            elif condition == "prcw":
                # GT file path, but with wrong file's code
                code_overrides = item.get("condition_prcw_codes", {})
            elif condition == "pwcr":
                # Wrong file path, but with GT file's code
                code_overrides = item.get("condition_pwcr_codes", {})
            elif condition == "crossed":
                # Both swaps
                code_overrides = item.get("condition_crossed_codes", {})

            scores = score_condition(
                model, tokenizer, issue_text, candidates,
                repo_dir, code_overrides, args.prompt_mode, args.anonymize_paths,
                args.code_max_lines, yes_id, no_id, args.max_seq_length,
                device, args.score_batch_size)

            # Get score and rank for swap_gt_file and swap_wrong_file
            gt_score = None
            wrong_score = None
            for cand, sc in zip(candidates, scores):
                if cand == swap_gt_file:
                    gt_score = sc
                if cand == swap_wrong_file:
                    wrong_score = sc

            gt_rank = compute_rank(candidates, scores, swap_gt_file)
            wrong_rank = compute_rank(candidates, scores, swap_wrong_file)

            cond_gt_scores[condition].append(gt_score if gt_score is not None else 0.0)
            cond_gt_ranks[condition].append(gt_rank)
            cond_wrong_scores[condition].append(wrong_score if wrong_score is not None else 0.0)
            cond_wrong_ranks[condition].append(wrong_rank)

            # hit@1, hit@5 based on full GT set
            ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
            predicted = [c for c, _ in ranked]
            top1 = set(predicted[:1])
            top5 = set(predicted[:5])
            h1 = len(top1 & gt_files) / len(gt_files) if gt_files else 0.0
            h5 = len(top5 & gt_files) / len(gt_files) if gt_files else 0.0
            cond_hits_1[condition].append(h1)
            cond_hits_5[condition].append(h5)

            example_result[f"{condition}_gt_score"] = gt_score
            example_result[f"{condition}_gt_rank"] = gt_rank
            example_result[f"{condition}_wrong_score"] = wrong_score
            example_result[f"{condition}_wrong_rank"] = wrong_rank
            example_result[f"{condition}_hit@1"] = h1
            example_result[f"{condition}_hit@5"] = h5

        per_example_results.append(example_result)

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            r1 = np.mean(cond_hits_1["original"]) * 100
            print(f"  [{idx+1}/{len(cf_data)}] original R@1={r1:.2f}% ({elapsed:.0f}s)")

    # Build summary
    wall_clock = time.time() - start_time
    print(f"\nDone. {len(per_example_results)} examples in {wall_clock:.0f}s")

    summary = {}
    for condition in conditions:
        entry = {
            "hit@1": float(np.mean(cond_hits_1[condition]) * 100),
            "hit@5": float(np.mean(cond_hits_5[condition]) * 100),
            "mean_gt_score": float(np.mean(cond_gt_scores[condition])),
            "mean_gt_rank": float(np.mean(cond_gt_ranks[condition])),
            "mean_wrong_score": float(np.mean(cond_wrong_scores[condition])),
            "mean_wrong_rank": float(np.mean(cond_wrong_ranks[condition])),
        }
        if condition != "original":
            orig_scores = np.array(cond_gt_scores["original"])
            cond_scores = np.array(cond_gt_scores[condition])
            entry["delta_score_vs_original"] = float(np.mean(cond_scores - orig_scores))
            orig_ranks = np.array(cond_gt_ranks["original"], dtype=float)
            cond_ranks = np.array(cond_gt_ranks[condition], dtype=float)
            entry["delta_rank_vs_original"] = float(np.mean(cond_ranks - orig_ranks))
        summary[condition] = entry

    summary["num_examples"] = len(per_example_results)
    summary["prompt_mode"] = args.prompt_mode
    summary["anonymize_paths"] = args.anonymize_paths
    summary["wall_clock_seconds"] = wall_clock

    # Print
    print(f"\n=== Counterfactual Evaluation ({args.prompt_mode}) ===")
    for condition in conditions:
        s = summary[condition]
        delta_str = ""
        if condition != "original":
            delta_str = f"  delta_score={s['delta_score_vs_original']:+.4f}  delta_rank={s['delta_rank_vs_original']:+.1f}"
        print(f"  {condition:10s}  R@1={s['hit@1']:5.2f}%  R@5={s['hit@5']:5.2f}%"
              f"  gt_score={s['mean_gt_score']:.4f}  gt_rank={s['mean_gt_rank']:.1f}{delta_str}")

    # Save
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    results_path = os.path.join(args.output_dir, "per_example_results.jsonl")
    with open(results_path, "w") as f:
        for r in per_example_results:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved summary to {summary_path}")
    print(f"Saved per-example results to {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reranker on counterfactual conflict data")
    parser.add_argument("--model_path", required=True,
                        help="Base model path (e.g. Qwen2.5-7B-Instruct)")
    parser.add_argument("--lora_path", default=None,
                        help="LoRA adapter path")
    parser.add_argument("--counterfactual_data", required=True,
                        help="Path to counterfactual JSONL file")
    parser.add_argument("--repo_dir", default="data/repos",
                        help="Base directory for repo snapshots")
    parser.add_argument("--prompt_mode", choices=["path_only", "code_aware"],
                        default="path_only",
                        help="Prompt mode: path_only or code_aware")
    parser.add_argument("--anonymize_paths", action="store_true",
                        help="Replace paths with hash-based anonymous names")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory (use /data, not /home)")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=200,
                        help="Max candidates per example")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--code_max_lines", type=int, default=30,
                        help="Max lines of code to include in code_aware mode")
    parser.add_argument("--score_batch_size", type=int, default=4)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
