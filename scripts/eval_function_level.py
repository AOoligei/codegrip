#!/usr/bin/env python3
"""
Function-level bug localization evaluation for CodeGRIP.

Extends file-level localization to function-level by building qualified
candidates "file_path::function_name" and computing R@K metrics.

Workflow:
1. Load BM25 file-level candidates (top-N files per example)
2. Expand each file to its functions via function_index_aligned.json
3. Build "file_path::function_name" candidate list
4. Build ground-truth qualified pairs from changed_functions + changed_py_files
5. Compute function-level R@1, R@5, R@10

Perturbation mode:
  --perturb shuffle  : shuffle function names across files within each example,
                       keeping file paths fixed. Tests if model uses func names
                       or just file paths.

Usage (data prep only, no GPU):
    python scripts/eval_function_level.py \
        --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
        --top_n_files 50 --mode data_prep

Usage (reranking with GPU):
    python scripts/eval_function_level.py \
        --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path experiments/rankft_runB_graph/best \
        --mode rerank --gpu_id 4

Usage (perturbation):
    python scripts/eval_function_level.py \
        --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path experiments/rankft_runB_graph/best \
        --mode rerank --perturb shuffle --gpu_id 4
"""
import json
import os
import argparse
import random
import time

import numpy as np

random.seed(42)
np.random.seed(42)


# ============================================================
# Ground truth resolution
# ============================================================

def resolve_gt_function_pairs(changed_functions, changed_py_files, func_index_repo):
    """Resolve bare function names to qualified file_path::func_name pairs.

    Strategy: for each changed_function name, find which changed_py_file(s)
    contain that function in the function index. Returns set of
    "file_path::func_name" strings.
    """
    gt_pairs = set()
    unresolved = []

    for func_name in changed_functions:
        found = False
        for fpath in changed_py_files:
            file_funcs = func_index_repo.get(fpath, [])
            if func_name in file_funcs:
                gt_pairs.add(f"{fpath}::{func_name}")
                found = True
        if not found:
            unresolved.append(func_name)

    return gt_pairs, unresolved


# ============================================================
# Candidate expansion
# ============================================================

def expand_files_to_functions(candidate_files, func_index_repo, top_n_files=50):
    """Expand top-N candidate files to function-level candidates.

    Returns list of "file_path::func_name" strings, preserving file order.
    """
    candidates = []
    for fpath in candidate_files[:top_n_files]:
        funcs = func_index_repo.get(fpath, [])
        for func_name in funcs:
            candidates.append(f"{fpath}::{func_name}")
    return candidates


def perturb_shuffle_functions(candidate_files, func_index_repo, top_n_files=50):
    """Perturbation: shuffle function names in prompts, keep candidate identity for GT.

    Returns (original_candidates, display_name_map) where:
    - original_candidates: list of "file_path::func_name" with REAL identities (for GT matching)
    - display_name_map: dict mapping each candidate to a shuffled display name (for prompt)

    This isolates the effect of function names: the model sees shuffled names in the prompt,
    but we evaluate against the original GT pairs. Any performance drop is purely from
    the model losing function-name signal, not from GT coverage change.
    """
    files = candidate_files[:top_n_files]

    # Build original candidates
    original_candidates = []
    for fpath in files:
        funcs = func_index_repo.get(fpath, [])
        for func_name in funcs:
            original_candidates.append(f"{fpath}::{func_name}")

    # Collect all function names and shuffle
    all_funcs = [c.split("::", 1)[1] for c in original_candidates]
    shuffled = list(all_funcs)
    random.shuffle(shuffled)

    # Map each original candidate to a shuffled display name
    display_name_map = {}
    for i, cand in enumerate(original_candidates):
        fpath = cand.split("::", 1)[0]
        display_name_map[cand] = f"{fpath}::{shuffled[i]}"

    return original_candidates, display_name_map


# ============================================================
# Metrics
# ============================================================

def compute_recall_at_k(ranked_candidates, gt_set, k):
    """Compute Recall@K: fraction of GT items found in top-K predictions."""
    if not gt_set:
        return None
    top_k = set(ranked_candidates[:k])
    hits = len(top_k & gt_set)
    return hits / len(gt_set) * 100


def compute_hit_at_k(ranked_candidates, gt_set, k):
    """Compute Hit@K: 1 if any GT item is in top-K, else 0."""
    if not gt_set:
        return None
    top_k = set(ranked_candidates[:k])
    return 100.0 if top_k & gt_set else 0.0


# ============================================================
# Scoring (GPU mode)
# ============================================================

PROMPT_TEMPLATE = (
    "Given the bug report, is this function in {file_path} likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {file_path}\n"
    "Function: {function_name}\n\n"
    "Answer:"
)


def load_model(model_path, lora_path, gpu_id, load_in_4bit=True):
    """Load cross-encoder model for scoring."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(42)
    device = f"cuda:{gpu_id}"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = device
    else:
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()

    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    yes_id, no_id = yes_ids[0], no_ids[0]

    return model, tokenizer, yes_id, no_id, device


def score_candidates_batched(model, tokenizer, issue_text, candidates,
                              yes_id, no_id, device,
                              max_seq_length=768, batch_size=16,
                              display_name_map=None):
    """Score function-level candidates. Returns list of scores.

    If display_name_map is provided (perturbation mode), use shuffled names
    in the prompt but return scores aligned with original candidate order.
    """
    import torch

    if not candidates:
        return []

    prompts = []
    for cand in candidates:
        # Use display name if available (perturbation), else use real name
        display_cand = display_name_map.get(cand, cand) if display_name_map else cand
        fpath, func_name = display_cand.split("::", 1)
        prompts.append(PROMPT_TEMPLATE.format(
            issue_text=issue_text,
            file_path=fpath,
            function_name=func_name,
        ))

    all_scores = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt",
                        padding=True, truncation=True, max_length=max_seq_length)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        try:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                # Fallback: one at a time
                for p in batch:
                    e2 = tokenizer([p], return_tensors="pt",
                                   truncation=True, max_length=max_seq_length)
                    ids = e2["input_ids"].to(device)
                    mask = e2["attention_mask"].to(device)
                    with torch.no_grad():
                        out = model(input_ids=ids, attention_mask=mask)
                    s = (out.logits[0, -1, yes_id] - out.logits[0, -1, no_id]).item()
                    all_scores.append(s)
                continue
            raise

        logits = outputs.logits
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(logits.size(0), device=device)
        last_logits = logits[batch_idx, seq_lengths]
        scores = (last_logits[:, yes_id] - last_logits[:, no_id]).cpu().tolist()
        all_scores.extend(scores)

    return all_scores


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Function-level bug localization evaluation")
    parser.add_argument("--test_data", default="data/grepo_text/grepo_test.jsonl")
    parser.add_argument("--function_index",
                        default="data/function_index_aligned.json")
    parser.add_argument("--bm25_candidates",
                        default="data/rankft/merged_bm25_exp6_candidates.jsonl")
    parser.add_argument("--top_n_files", type=int, default=50,
                        help="Number of top BM25 files to expand to functions")
    parser.add_argument("--mode", choices=["data_prep", "rerank"],
                        default="data_prep",
                        help="data_prep: stats only; rerank: score with model")
    parser.add_argument("--perturb", choices=["none", "shuffle"],
                        default="none",
                        help="Perturbation mode for ablation")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N examples (0=all, for debugging)")
    parser.add_argument("--output_dir", default=None)

    # GPU/model args (only used in rerank mode)
    parser.add_argument("--model_path",
                        default="/data/shuyang/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--max_seq_length", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    # ---- Load data ----
    print(f"Loading test data from {args.test_data}...")
    test_map = {}
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            key = f"{item['repo']}_{item['issue_id']}"
            test_map[key] = item

    print(f"Loading function index from {args.function_index}...")
    with open(args.function_index) as f:
        func_index = json.load(f)

    print(f"Loading BM25 candidates from {args.bm25_candidates}...")
    bm25_map = {}
    with open(args.bm25_candidates) as f:
        for line in f:
            item = json.loads(line)
            key = f"{item['repo']}_{item['issue_id']}"
            bm25_map[key] = item["candidates"]

    print(f"  Test: {len(test_map)}, Func index repos: {len(func_index)}, "
          f"BM25: {len(bm25_map)}")

    # ---- Filter to examples with changed_functions ----
    examples = []
    for key, item in test_map.items():
        if not item.get("changed_functions"):
            continue
        if key not in bm25_map:
            continue
        examples.append((key, item, bm25_map[key]))

    if args.limit > 0:
        examples = examples[:args.limit]

    print(f"Examples with function GT and BM25 candidates: {len(examples)}")

    # ---- Load model if reranking ----
    model, tokenizer, yes_id, no_id, device = None, None, None, None, None
    if args.mode == "rerank":
        model, tokenizer, yes_id, no_id, device = load_model(
            args.model_path, args.lora_path, args.gpu_id, args.load_in_4bit)
        print(f"Model loaded on cuda:{args.gpu_id}")

    # ---- Process examples ----
    k_values = [1, 5, 10]

    # Metrics storage
    recall_metrics = {f"R@{k}": [] for k in k_values}
    hit_metrics = {f"Hit@{k}": [] for k in k_values}

    # File-level reference metrics (from BM25 order)
    file_recall = {f"file_R@{k}": [] for k in k_values}

    # Statistics
    stats = {
        "n_candidates": [],
        "n_gt_pairs": [],
        "n_gt_covered": [],
        "n_gt_uncovered": [],
        "n_unresolved_funcs": [],
    }
    per_example_results = []

    start_time = time.time()

    for idx, (key, item, candidate_files) in enumerate(examples):
        repo = item["repo"]
        issue_id = item["issue_id"]
        changed_functions = item["changed_functions"]
        changed_py_files = item.get("changed_py_files", [])
        issue_text = item.get("issue_text", "")

        repo_func_index = func_index.get(repo, {})

        # Step 1: Resolve GT to qualified pairs
        gt_pairs, unresolved = resolve_gt_function_pairs(
            changed_functions, changed_py_files, repo_func_index)

        if not gt_pairs:
            # Cannot evaluate if no GT pairs resolved
            stats["n_unresolved_funcs"].append(len(changed_functions))
            continue

        # Step 2: Expand files to function candidates
        display_name_map = None
        if args.perturb == "shuffle":
            func_candidates, display_name_map = perturb_shuffle_functions(
                candidate_files, repo_func_index, args.top_n_files)
        else:
            func_candidates = expand_files_to_functions(
                candidate_files, repo_func_index, args.top_n_files)

        # Step 3: Check GT coverage in candidate pool (always against real identities)
        cand_set = set(func_candidates)
        gt_covered = gt_pairs & cand_set
        gt_uncovered = gt_pairs - cand_set

        stats["n_candidates"].append(len(func_candidates))
        stats["n_gt_pairs"].append(len(gt_pairs))
        stats["n_gt_covered"].append(len(gt_covered))
        stats["n_gt_uncovered"].append(len(gt_uncovered))
        stats["n_unresolved_funcs"].append(len(unresolved))

        # Step 4: Rank candidates
        if args.mode == "rerank" and func_candidates:
            # Score with model (uses display_name_map for perturbation prompts)
            scores = score_candidates_batched(
                model, tokenizer, issue_text, func_candidates,
                yes_id, no_id, device,
                max_seq_length=args.max_seq_length,
                batch_size=args.batch_size,
                display_name_map=display_name_map)
            # Sort by score descending
            scored = sorted(zip(func_candidates, scores), key=lambda x: -x[1])
            ranked = [c for c, _ in scored]
        else:
            # In data_prep mode, use BM25 file order (functions within file unranked)
            ranked = func_candidates

        # Step 5: Compute metrics
        for k in k_values:
            r = compute_recall_at_k(ranked, gt_pairs, k)
            h = compute_hit_at_k(ranked, gt_pairs, k)
            if r is not None:
                recall_metrics[f"R@{k}"].append(r)
            if h is not None:
                hit_metrics[f"Hit@{k}"].append(h)

        # File-level reference
        gt_files = set(changed_py_files)
        for k in k_values:
            top_k_files = set(candidate_files[:k])
            fr = len(top_k_files & gt_files) / len(gt_files) * 100 if gt_files else 0
            file_recall[f"file_R@{k}"].append(fr)

        per_example_results.append({
            "repo": repo,
            "issue_id": issue_id,
            "n_func_candidates": len(func_candidates),
            "n_gt_pairs": len(gt_pairs),
            "n_gt_covered": len(gt_covered),
            "gt_pairs": sorted(gt_pairs),
            "gt_uncovered": sorted(gt_uncovered),
            "unresolved_funcs": unresolved,
            "top10_ranked": ranked[:10] if args.mode == "rerank" else ranked[:10],
        })

        if (idx + 1) % 100 == 0 or (args.limit > 0 and idx < args.limit):
            elapsed = time.time() - start_time
            n_eval = len(recall_metrics["R@1"])
            avg_r1 = np.mean(recall_metrics["R@1"]) if recall_metrics["R@1"] else 0
            print(f"  [{idx+1}/{len(examples)}] "
                  f"n_eval={n_eval}, avg_R@1={avg_r1:.2f}%, "
                  f"avg_candidates={np.mean(stats['n_candidates']):.0f}, "
                  f"elapsed={elapsed:.1f}s")

    elapsed_total = time.time() - start_time

    # ---- Print results ----
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print(f"\n{'='*65}")
    print(f"FUNCTION-LEVEL LOCALIZATION EVALUATION")
    print(f"{'='*65}")
    print(f"Mode: {args.mode} | Perturb: {args.perturb}")
    print(f"Top-N files: {args.top_n_files}")
    print(f"Evaluated: {len(recall_metrics['R@1'])} / {len(examples)} examples")
    print(f"Time: {elapsed_total:.1f}s")

    print(f"\n--- Candidate Statistics ---")
    print(f"  Avg function candidates per example: {avg(stats['n_candidates']):.1f}")
    print(f"  Avg GT pairs per example:            {avg(stats['n_gt_pairs']):.2f}")
    print(f"  Avg GT covered in candidate pool:    {avg(stats['n_gt_covered']):.2f}")
    print(f"  Avg GT uncovered:                    {avg(stats['n_gt_uncovered']):.2f}")
    cov_rate = (avg(stats['n_gt_covered']) /
                max(avg(stats['n_gt_pairs']), 1e-9)) * 100
    print(f"  GT coverage rate:                    {cov_rate:.1f}%")
    print(f"  Avg unresolved func names:           {avg(stats['n_unresolved_funcs']):.2f}")

    print(f"\n--- Function-Level Metrics ---")
    print(f"{'Metric':<20} {'Score':>8} {'N':>6}")
    print("-" * 38)
    for k in k_values:
        r = avg(recall_metrics[f"R@{k}"])
        n = len(recall_metrics[f"R@{k}"])
        print(f"  Func R@{k:<5}       {r:>7.2f}%  {n:>5}")
    print()
    for k in k_values:
        h = avg(hit_metrics[f"Hit@{k}"])
        n = len(hit_metrics[f"Hit@{k}"])
        print(f"  Func Hit@{k:<5}     {h:>7.2f}%  {n:>5}")

    print(f"\n--- File-Level Reference (BM25 order) ---")
    for k in k_values:
        r = avg(file_recall[f"file_R@{k}"])
        n = len(file_recall[f"file_R@{k}"])
        print(f"  File R@{k:<5}       {r:>7.2f}%  {n:>5}")

    # ---- Save results ----
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        summary = {
            "config": {
                "mode": args.mode,
                "perturb": args.perturb,
                "top_n_files": args.top_n_files,
                "bm25_candidates": args.bm25_candidates,
                "function_index": args.function_index,
                "model_path": args.model_path if args.mode == "rerank" else None,
                "lora_path": args.lora_path if args.mode == "rerank" else None,
            },
            "func_level": {
                f"R@{k}": avg(recall_metrics[f"R@{k}"]) for k in k_values
            },
            "func_hit": {
                f"Hit@{k}": avg(hit_metrics[f"Hit@{k}"]) for k in k_values
            },
            "file_level_ref": {
                f"R@{k}": avg(file_recall[f"file_R@{k}"]) for k in k_values
            },
            "stats": {
                "n_evaluated": len(recall_metrics["R@1"]),
                "n_total_with_gt": len(examples),
                "avg_candidates": avg(stats["n_candidates"]),
                "avg_gt_pairs": avg(stats["n_gt_pairs"]),
                "avg_gt_covered": avg(stats["n_gt_covered"]),
                "gt_coverage_rate_pct": cov_rate,
            },
            "wall_clock_seconds": round(elapsed_total, 2),
        }
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(args.output_dir, "per_example.jsonl"), "w") as f:
            for r in per_example_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
