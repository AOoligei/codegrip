"""
Evaluate RankFT reranker on GREPO test set.

Loads a RankFT model (base + LoRA), scores BM25 top-K candidates
for each test example, reranks by score, and computes metrics.

Metrics:
  - Hit@k: fraction of GT files in top-k (averaged over examples)
  - Acc@k: whether ALL GT files are in top-k (exact match)
  - Recall@k: fraction of GT files found in top-k
  - Conditional Acc@1|GT in candidates: accuracy at rank 1 among
    examples where the GT file appears in the BM25 candidate list

Usage:
    python src/eval/eval_rankft.py \
        --model_path /path/to/Qwen2.5-7B \
        --lora_path experiments/rankft_v1/final \
        --test_data data/grepo_text/grepo_test.jsonl \
        --bm25_candidates data/bm25_candidates/test_bm25_top500.jsonl \
        --output_dir experiments/rankft_v1/eval \
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
# Prompt (must match training)
# ============================================================

PROMPT_TEMPLATE = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)


def build_prompt(issue_text: str, candidate_path: str) -> str:
    """Build the scoring prompt for a single (issue, file) pair."""
    return PROMPT_TEMPLATE.format(
        issue_text=issue_text,
        candidate_path=candidate_path,
    )


# ============================================================
# Scoring
# ============================================================

def get_yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    """Get token IDs for 'Yes' and 'No'."""
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    yes_id = yes_ids[0]
    no_id = no_ids[0]
    return yes_id, no_id


@torch.no_grad()
def score_candidates_batched(
    model,
    tokenizer,
    issue_text: str,
    candidates: List[str],
    yes_id: int,
    no_id: int,
    max_seq_length: int,
    device: str,
    batch_size: int = 16,
) -> List[float]:
    """Score all candidates for a single issue, in batches.

    Returns list of scores (logit_yes - logit_no) for each candidate.
    """
    prompts = [build_prompt(issue_text, cand) for cand in candidates]
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
    """Hit@k: fraction of GT files found in top-k predictions."""
    if not gt:
        return 0.0
    top_k = set(predicted[:k])
    return len(top_k & gt) / len(gt)


def compute_acc_at_k(predicted: List[str], gt: Set[str], k: int) -> float:
    """Acc@k: 1.0 if ALL GT files are in top-k, else 0.0."""
    if not gt:
        return 0.0
    top_k = set(predicted[:k])
    return 1.0 if gt.issubset(top_k) else 0.0


def compute_recall_at_k(predicted: List[str], gt: Set[str], k: int) -> float:
    """Recall@k: same as hit@k for this task (fraction of GT found)."""
    return compute_hit_at_k(predicted, gt, k)


# ============================================================
# Evaluation
# ============================================================

def evaluate(args):
    """Main evaluation routine."""
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    # ---- Load test data ----
    print(f"Loading test data from {args.test_data}...")
    test_data = []
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            if item.get("changed_py_files"):
                test_data.append(item)
    print(f"  {len(test_data)} test examples")

    # ---- Load BM25 candidates ----
    print(f"Loading BM25 candidates from {args.bm25_candidates}...")
    bm25_map: Dict[str, List[str]] = {}
    with open(args.bm25_candidates) as f:
        for line in f:
            item = json.loads(line)
            key = f"{item['repo']}_{item['issue_id']}"
            bm25_map[key] = item.get("candidates", item.get("bm25_candidates", []))
    print(f"  BM25 candidates for {len(bm25_map)} examples")

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
    k_values = [1, 3, 5, 10, 20]
    results = []
    overall_metrics = {f"hit@{k}": [] for k in k_values}
    overall_metrics.update({f"acc@{k}": [] for k in k_values})
    overall_metrics.update({f"recall@{k}": [] for k in k_values})

    # For conditional Acc@1
    cond_acc1_correct = 0
    cond_acc1_total = 0

    per_repo_metrics = defaultdict(lambda: defaultdict(list))

    total = len(test_data)
    start_time = time.time()

    for idx, example in enumerate(test_data):
        repo = example["repo"]
        issue_id = example["issue_id"]
        issue_text = example["issue_text"]
        gt_files = set(example["changed_py_files"])

        bm25_key = f"{repo}_{issue_id}"
        candidates = bm25_map.get(bm25_key, [])

        if not candidates:
            # No BM25 candidates available, skip
            continue

        # Truncate to top_k
        candidates = candidates[: args.top_k]

        # Check if any GT file is in the candidate list
        gt_in_candidates = bool(gt_files & set(candidates))

        if idx % 20 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / max(elapsed, 1)
            eta = (total - idx - 1) / max(rate, 0.001)
            print(
                f"  [{idx+1}/{total}] {repo}#{issue_id} | "
                f"{len(candidates)} candidates | "
                f"GT in candidates: {gt_in_candidates} | "
                f"ETA: {eta:.0f}s"
            )

        # Score all candidates
        t0 = time.time()
        scores = score_candidates_batched(
            model, tokenizer,
            issue_text, candidates,
            yes_id, no_id,
            args.max_seq_length, device,
            batch_size=args.score_batch_size,
        )
        scoring_time = time.time() - t0

        # Rerank by score (descending)
        scored_candidates = sorted(
            zip(candidates, scores), key=lambda x: -x[1]
        )
        reranked = [c for c, _ in scored_candidates]

        # Compute metrics
        metrics = {}
        for k in k_values:
            hit = compute_hit_at_k(reranked, gt_files, k)
            acc = compute_acc_at_k(reranked, gt_files, k)
            recall = compute_recall_at_k(reranked, gt_files, k)
            metrics[f"hit@{k}"] = hit
            metrics[f"acc@{k}"] = acc
            metrics[f"recall@{k}"] = recall
            overall_metrics[f"hit@{k}"].append(hit)
            overall_metrics[f"acc@{k}"].append(acc)
            overall_metrics[f"recall@{k}"].append(recall)
            per_repo_metrics[repo][f"hit@{k}"].append(hit)
            per_repo_metrics[repo][f"acc@{k}"].append(acc)
            per_repo_metrics[repo][f"recall@{k}"].append(recall)

        # Conditional Acc@1
        if gt_in_candidates:
            cond_acc1_total += 1
            if reranked[0] in gt_files:
                cond_acc1_correct += 1

        # Store result
        result = {
            "repo": repo,
            "issue_id": issue_id,
            "ground_truth": list(gt_files),
            "predicted": reranked[:50],  # Save top-50
            "bm25_original": candidates[:20],  # Original BM25 order (first 20)
            "scores": [s for _, s in scored_candidates[:50]],
            "metrics": metrics,
            "gt_in_candidates": gt_in_candidates,
            "num_candidates": len(candidates),
            "scoring_time": round(scoring_time, 3),
        }
        results.append(result)

    # ---- Aggregate metrics ----
    total_evaluated = len(results)
    elapsed_total = time.time() - start_time

    avg_overall = {}
    for metric_name, values in overall_metrics.items():
        if values:
            avg_overall[metric_name] = sum(values) / len(values) * 100
        else:
            avg_overall[metric_name] = 0.0

    cond_acc1 = (
        (cond_acc1_correct / cond_acc1_total * 100)
        if cond_acc1_total > 0
        else 0.0
    )
    avg_overall["cond_acc@1|gt_in_candidates"] = cond_acc1

    avg_per_repo = {}
    for repo, repo_m in per_repo_metrics.items():
        avg_per_repo[repo] = {}
        for metric_name, values in repo_m.items():
            avg_per_repo[repo][metric_name] = sum(values) / len(values) * 100
        avg_per_repo[repo]["count"] = len(repo_m.get("hit@1", []))

    # ---- Print results ----
    print(f"\n{'='*70}")
    print(f"RANKFT EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  Model: {args.model_path}")
    if args.lora_path:
        print(f"  LoRA: {args.lora_path}")
    print(f"  Examples evaluated: {total_evaluated}")
    print(f"  Total time: {elapsed_total:.0f}s ({elapsed_total/3600:.2f}h)")
    print(f"  Avg time/example: {elapsed_total/max(total_evaluated,1):.2f}s")

    print(f"\nOVERALL METRICS:")
    for k in k_values:
        print(
            f"  Hit@{k}: {avg_overall[f'hit@{k}']:.2f}%  |  "
            f"Acc@{k}: {avg_overall[f'acc@{k}']:.2f}%  |  "
            f"Recall@{k}: {avg_overall[f'recall@{k}']:.2f}%"
        )
    print(f"\n  Cond. Acc@1|GT in candidates: {cond_acc1:.2f}% "
          f"({cond_acc1_correct}/{cond_acc1_total})")

    print(f"\nPER-REPO RESULTS:")
    header = (
        "Repo".ljust(18)
        + "".join(f"H@{k}".rjust(8) for k in k_values)
        + "  Count"
    )
    print(header)
    print("-" * len(header))
    for repo in sorted(avg_per_repo.keys()):
        m = avg_per_repo[repo]
        line = repo[:17].ljust(18)
        line += "".join(f"{m.get(f'hit@{k}', 0):7.2f}%" for k in k_values)
        line += f"  {m['count']:5d}"
        print(line)

    # ---- Comparison with BM25 baseline ----
    # Compute BM25 original metrics for comparison
    bm25_metrics = {f"hit@{k}": [] for k in k_values}
    for result in results:
        gt = set(result["ground_truth"])
        bm25_orig = result.get("bm25_original", result["predicted"])
        for k in k_values:
            bm25_metrics[f"hit@{k}"].append(compute_hit_at_k(bm25_orig, gt, k))

    print(f"\nCOMPARISON (BM25 -> RankFT):")
    for k in k_values:
        bm25_val = sum(bm25_metrics[f"hit@{k}"]) / max(len(bm25_metrics[f"hit@{k}"]), 1) * 100
        rankft_val = avg_overall[f"hit@{k}"]
        delta = rankft_val - bm25_val
        direction = "+" if delta >= 0 else ""
        print(f"  Hit@{k}: {bm25_val:.2f}% -> {rankft_val:.2f}% ({direction}{delta:.2f}%)")

    # ---- Save results ----
    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
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
            "top_k": args.top_k,
            "max_seq_length": args.max_seq_length,
            "total_examples": total_evaluated,
            "cond_acc1_total": cond_acc1_total,
            "cond_acc1_correct": cond_acc1_correct,
        },
        "wall_clock_seconds": round(elapsed_total, 2),
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RankFT reranker on GREPO test set"
    )

    # Model
    parser.add_argument("--model_path", required=True,
                        help="Path to base model")
    parser.add_argument("--lora_path", default=None,
                        help="Path to RankFT LoRA checkpoint")

    # Data
    parser.add_argument("--test_data", required=True,
                        help="Test JSONL (GREPO format)")
    parser.add_argument("--bm25_candidates", required=True,
                        help="Precomputed BM25 top-K JSONL")

    # Output
    parser.add_argument("--output_dir", required=True,
                        help="Results directory")

    # Hardware
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU to use")

    # Evaluation params
    parser.add_argument("--top_k", type=int, default=200,
                        help="How many BM25 candidates to rerank")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Max tokens per prompt")
    parser.add_argument("--score_batch_size", type=int, default=16,
                        help="Batch size for scoring candidates")

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
