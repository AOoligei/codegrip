"""
Evaluate RankFT-Function reranker on GREPO test set.

Given ground-truth files, ranks functions within each file using the
function-level reranker and computes function-level Hit@k metrics.

For each test example:
  - For each GT file that has function labels:
    - Score all functions in that file
    - Rank by score (descending)
    - Compute Hit@k: fraction of positive functions found in top-k

Metrics reported:
  - Function-level Hit@k (k=1,3,5): averaged over (example, file) pairs
  - Function-level Acc@1: whether the top-ranked function is a positive
  - Function-level MRR: mean reciprocal rank of first positive function
  - Per-repo breakdown

Requires function-level labels on test data. Generate them with:
    python scripts/extract_function_labels.py \
        --repos_dir data/repos \
        --train_data data/grepo_text/grepo_test.jsonl \
        --output data/grepo_text/grepo_test_function_labels.jsonl

Usage:
    python src/eval/eval_rankft_function.py \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path experiments/rankft_function_v1/best \
        --test_data data/grepo_text/grepo_test_function_labels.jsonl \
        --output_dir experiments/rankft_function_v1/eval \
        --gpu_id 0
"""

import os
import json
import argparse
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Deterministic
torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# Prompt (must match training — function-level template)
# ============================================================

PROMPT_TEMPLATE = (
    "Given the bug report, is this function in {file_path} likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {file_path}\n"
    "Function: {qualified_name}\n\n"
    "Answer:"
)


def build_prompt(issue_text: str, file_path: str, qualified_name: str) -> str:
    """Build the scoring prompt for a single (issue, file, function) triple."""
    return PROMPT_TEMPLATE.format(
        issue_text=issue_text,
        file_path=file_path,
        qualified_name=qualified_name,
    )


# ============================================================
# Scoring
# ============================================================

def get_yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    """Get token IDs for 'Yes' and 'No'."""
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    return yes_ids[0], no_ids[0]


@torch.no_grad()
def score_functions_batched(
    model,
    tokenizer,
    issue_text: str,
    file_path: str,
    functions: List[str],
    yes_id: int,
    no_id: int,
    max_seq_length: int,
    device: str,
    batch_size: int = 16,
) -> List[float]:
    """Score all functions for a single (issue, file), in batches.

    Returns list of scores (logit_yes - logit_no) for each function.
    """
    prompts = [
        build_prompt(issue_text, file_path, func_name)
        for func_name in functions
    ]
    all_scores = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        encodings = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Fall back to processing one at a time
                torch.cuda.empty_cache()
                for prompt in batch_prompts:
                    enc = tokenizer(
                        [prompt],
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_seq_length,
                    )
                    ids = enc["input_ids"].to(device)
                    mask = enc["attention_mask"].to(device)
                    out = model(input_ids=ids, attention_mask=mask)
                    logits = out.logits[0, -1]
                    score = (logits[yes_id] - logits[no_id]).item()
                    all_scores.append(score)
                continue
            raise

        logits = outputs.logits  # (batch, seq_len, vocab_size)

        # Get logits at last non-padding position for each sequence
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.size(0), device=device)
        last_logits = logits[batch_indices, seq_lengths]  # (batch, vocab_size)

        scores = (last_logits[:, yes_id] - last_logits[:, no_id]).cpu().tolist()
        all_scores.extend(scores)

    return all_scores


# ============================================================
# Metrics
# ============================================================

def compute_hit_at_k(predicted: List[str], gt: Set[str], k: int) -> float:
    """Hit@k: fraction of GT functions found in top-k predictions."""
    if not gt:
        return 0.0
    top_k = set(predicted[:k])
    return len(top_k & gt) / len(gt)


def compute_acc_at_1(predicted: List[str], gt: Set[str]) -> float:
    """Acc@1: 1.0 if the top-ranked function is a positive, else 0.0."""
    if not gt or not predicted:
        return 0.0
    return 1.0 if predicted[0] in gt else 0.0


def compute_mrr(predicted: List[str], gt: Set[str]) -> float:
    """MRR: reciprocal rank of the first positive function in the ranking."""
    if not gt:
        return 0.0
    for rank, func in enumerate(predicted, start=1):
        if func in gt:
            return 1.0 / rank
    return 0.0


# ============================================================
# Evaluation
# ============================================================

def evaluate(args):
    """Main evaluation routine."""
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    # ---- Load test data with function labels ----
    print(f"Loading test data from {args.test_data}...")
    test_data = []
    skipped = 0
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            cfd = item.get("changed_functions_detailed", {})
            # Must have at least one file with positive_functions and >=2 functions
            has_viable = any(
                len(v.get("positive_functions", [])) > 0
                and len(v.get("all_functions", [])) >= 2
                for v in cfd.values()
            )
            if has_viable:
                test_data.append(item)
            else:
                skipped += 1
    print(f"  {len(test_data)} test examples with function labels "
          f"(skipped {skipped} without viable files)")

    # ---- Load model ----
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    yes_id, no_id = get_yes_no_token_ids(tokenizer)
    print(f"  Yes ID: {yes_id}, No ID: {no_id}")

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    if args.lora_path:
        print(f"Loading LoRA adapter from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path)

    model.eval()
    print("  Model loaded and set to eval mode.")

    # ---- Evaluate ----
    k_values = [1, 3, 5]
    overall_metrics = {f"func_hit@{k}": [] for k in k_values}
    overall_metrics["func_acc@1"] = []
    overall_metrics["func_mrr"] = []

    per_repo_metrics = defaultdict(lambda: defaultdict(list))
    results = []

    total = len(test_data)
    total_files_evaluated = 0
    start_time = time.time()

    for idx, example in enumerate(test_data):
        repo = example["repo"]
        issue_id = example.get("issue_id", "")
        issue_text = example.get("issue_text", "")
        cfd = example.get("changed_functions_detailed", {})

        example_results = []

        for file_path, file_info in cfd.items():
            pos_funcs = file_info.get("positive_functions", [])
            all_funcs = file_info.get("all_functions", [])

            if len(pos_funcs) == 0 or len(all_funcs) < 2:
                continue

            gt_funcs = set(pos_funcs)

            # Score all functions in this file
            scores = score_functions_batched(
                model, tokenizer,
                issue_text, file_path, all_funcs,
                yes_id, no_id,
                args.max_seq_length, device,
                batch_size=args.score_batch_size,
            )

            # Rerank by score (descending)
            scored_funcs = sorted(
                zip(all_funcs, scores), key=lambda x: -x[1]
            )
            reranked = [f for f, _ in scored_funcs]

            # Compute metrics for this file
            file_metrics = {}
            for k in k_values:
                hit = compute_hit_at_k(reranked, gt_funcs, k)
                file_metrics[f"func_hit@{k}"] = hit
                overall_metrics[f"func_hit@{k}"].append(hit)
                per_repo_metrics[repo][f"func_hit@{k}"].append(hit)

            acc1 = compute_acc_at_1(reranked, gt_funcs)
            mrr = compute_mrr(reranked, gt_funcs)
            file_metrics["func_acc@1"] = acc1
            file_metrics["func_mrr"] = mrr
            overall_metrics["func_acc@1"].append(acc1)
            overall_metrics["func_mrr"].append(mrr)
            per_repo_metrics[repo]["func_acc@1"].append(acc1)
            per_repo_metrics[repo]["func_mrr"].append(mrr)

            example_results.append({
                "file_path": file_path,
                "ground_truth_functions": list(gt_funcs),
                "all_functions": all_funcs,
                "predicted_ranking": reranked,
                "scores": [s for _, s in scored_funcs],
                "metrics": file_metrics,
                "num_functions": len(all_funcs),
                "num_positive": len(gt_funcs),
            })
            total_files_evaluated += 1

        if example_results:
            results.append({
                "repo": repo,
                "issue_id": issue_id,
                "file_results": example_results,
            })

        if idx % 20 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / max(elapsed, 1)
            eta = (total - idx - 1) / max(rate, 0.001)
            print(
                f"  [{idx+1}/{total}] {repo}#{issue_id} | "
                f"{len(example_results)} files evaluated | "
                f"Files so far: {total_files_evaluated} | "
                f"ETA: {eta:.0f}s"
            )

    # ---- Aggregate metrics ----
    total_evaluated = len(results)
    elapsed_total = time.time() - start_time

    avg_overall = {}
    for metric_name, values in overall_metrics.items():
        if values:
            avg_overall[metric_name] = sum(values) / len(values) * 100
        else:
            avg_overall[metric_name] = 0.0

    avg_per_repo = {}
    for repo, repo_m in per_repo_metrics.items():
        avg_per_repo[repo] = {}
        for metric_name, values in repo_m.items():
            avg_per_repo[repo][metric_name] = sum(values) / len(values) * 100
        avg_per_repo[repo]["count"] = len(repo_m.get("func_hit@1", []))

    # ---- Print results ----
    print(f"\n{'='*70}")
    print(f"RANKFT-FUNCTION EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  Model: {args.model_path}")
    if args.lora_path:
        print(f"  LoRA: {args.lora_path}")
    print(f"  Examples evaluated: {total_evaluated}")
    print(f"  Files evaluated: {total_files_evaluated}")
    print(f"  Total time: {elapsed_total:.0f}s ({elapsed_total/3600:.2f}h)")
    if total_files_evaluated > 0:
        print(f"  Avg time/file: {elapsed_total/total_files_evaluated:.2f}s")

    print(f"\nOVERALL FUNCTION-LEVEL METRICS (averaged over files):")
    for k in k_values:
        metric_key = f"func_hit@{k}"
        print(f"  Hit@{k}: {avg_overall[metric_key]:.2f}%")
    print(f"  Acc@1: {avg_overall['func_acc@1']:.2f}%")
    print(f"  MRR:   {avg_overall['func_mrr']:.2f}%")

    # ---- Random baseline comparison ----
    # For each evaluated file, compute expected random baseline
    random_hit_at_k = {k: [] for k in k_values}
    random_acc1 = []
    random_mrr = []
    for result in results:
        for fr in result["file_results"]:
            n_total = fr["num_functions"]
            n_pos = fr["num_positive"]
            if n_total == 0 or n_pos == 0:
                continue
            for k in k_values:
                # Expected Hit@k for random ranking = min(k, n_total) * n_pos / n_total / n_pos
                # = min(k, n_total) / n_total
                # More precisely: E[|top_k intersect GT|] / |GT|
                # Using hypergeometric: E = k * n_pos / n_total (if k <= n_total)
                ek = min(k, n_total) * n_pos / n_total / n_pos
                random_hit_at_k[k].append(ek)
            # Random Acc@1 = n_pos / n_total
            random_acc1.append(n_pos / n_total)
            # Random MRR = sum_{r=1}^{n_total} P(first_pos_at_r) / r
            # Approximation: E[1/rank] ~ n_pos / n_total * H(n_total) is complex
            # Simple: for random permutation, expected rank of first positive = (n_total+1)/(n_pos+1)
            expected_rank = (n_total + 1) / (n_pos + 1)
            random_mrr.append(1.0 / expected_rank)

    print(f"\nCOMPARISON (Random baseline -> RankFT-Function):")
    for k in k_values:
        rand_val = np.mean(random_hit_at_k[k]) * 100 if random_hit_at_k[k] else 0
        model_val = avg_overall[f"func_hit@{k}"]
        delta = model_val - rand_val
        direction = "+" if delta >= 0 else ""
        print(f"  Hit@{k}: {rand_val:.2f}% -> {model_val:.2f}% ({direction}{delta:.2f}%)")
    rand_acc1_val = np.mean(random_acc1) * 100 if random_acc1 else 0
    model_acc1 = avg_overall["func_acc@1"]
    delta_acc1 = model_acc1 - rand_acc1_val
    print(f"  Acc@1: {rand_acc1_val:.2f}% -> {model_acc1:.2f}% "
          f"({'+'if delta_acc1>=0 else ''}{delta_acc1:.2f}%)")
    rand_mrr_val = np.mean(random_mrr) * 100 if random_mrr else 0
    model_mrr = avg_overall["func_mrr"]
    delta_mrr = model_mrr - rand_mrr_val
    print(f"  MRR:   {rand_mrr_val:.2f}% -> {model_mrr:.2f}% "
          f"({'+'if delta_mrr>=0 else ''}{delta_mrr:.2f}%)")

    print(f"\nPER-REPO RESULTS:")
    header = (
        "Repo".ljust(18)
        + "".join(f"H@{k}".rjust(8) for k in k_values)
        + " Acc@1".rjust(8)
        + "   MRR".rjust(8)
        + "  Count"
    )
    print(header)
    print("-" * len(header))
    for repo in sorted(avg_per_repo.keys()):
        m = avg_per_repo[repo]
        line = repo[:17].ljust(18)
        line += "".join(f"{m.get(f'func_hit@{k}', 0):7.2f}%" for k in k_values)
        line += f"{m.get('func_acc@1', 0):7.2f}%"
        line += f"{m.get('func_mrr', 0):7.2f}%"
        line += f"  {m['count']:5d}"
        print(line)

    # ---- Save results ----
    pred_path = os.path.join(args.output_dir, "function_predictions.jsonl")
    with open(pred_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nPredictions saved to {pred_path}")

    summary = {
        "overall": avg_overall,
        "per_repo": avg_per_repo,
        "config": {
            "model_path": args.model_path,
            "lora_path": args.lora_path,
            "max_seq_length": args.max_seq_length,
            "total_examples": total_evaluated,
            "total_files_evaluated": total_files_evaluated,
        },
        "random_baseline": {
            f"hit@{k}": float(np.mean(random_hit_at_k[k]) * 100) if random_hit_at_k[k] else 0
            for k in k_values
        },
        "wall_clock_seconds": round(elapsed_total, 2),
    }
    summary["random_baseline"]["acc@1"] = float(rand_acc1_val)
    summary["random_baseline"]["mrr"] = float(rand_mrr_val)

    summary_path = os.path.join(args.output_dir, "function_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RankFT-Function reranker on GREPO test set "
                    "(function-level ranking within GT files)"
    )

    # Model
    parser.add_argument("--model_path", type=str,
                        default="/data/shuyang/models/Qwen2.5-7B-Instruct",
                        help="Path to base model")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to RankFT-Function LoRA checkpoint")

    # Data
    parser.add_argument("--test_data", type=str,
                        default="data/grepo_text/grepo_test_function_labels.jsonl",
                        help="Test JSONL with function-level labels "
                             "(from extract_function_labels.py)")

    # Output
    parser.add_argument("--output_dir", type=str,
                        default="experiments/rankft_function_v1/eval",
                        help="Results directory")

    # Hardware
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU to use")

    # Evaluation params
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Max tokens per prompt")
    parser.add_argument("--score_batch_size", type=int, default=16,
                        help="Batch size for scoring functions")

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
