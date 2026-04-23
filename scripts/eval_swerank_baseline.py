#!/usr/bin/env python3
"""
Run SweRankEmbed-Large as a baseline on GREPO and SWE-bench.

Uses file summaries as document content (not full function bodies).
This gives us a dense retrieval baseline to compare against our RankFT.

Usage:
    python scripts/eval_swerank_baseline.py \
        --model_path /home/chenlibin/models/SweRankEmbed-Large \
        --dataset grepo \
        --device cuda:0
"""
import json
import os
import argparse
import time
from collections import defaultdict

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# Deterministic
np.random.seed(42)
torch.manual_seed(42)


QUERY_PREFIX = (
    "Instruct: Given a github issue, identify the code that needs to be changed "
    "to fix the issue.\nQuery"
)


def load_test_data(dataset):
    """Load test data with BM25 candidates."""
    if dataset == "grepo":
        test_path = "data/grepo_text/grepo_test.jsonl"
        bm25_path = "data/rankft/grepo_test_bm25_top500.jsonl"
        summary_path = "data/file_summaries_aligned.json"
    elif dataset == "swebench":
        test_path = None  # SWE-bench uses bm25 file directly
        bm25_path = "data/rankft/swebench_test_bm25_top500.jsonl"
        summary_path = "data/swebench_file_summaries/file_summaries_all.json"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Load BM25 candidates
    bm25_data = {}
    with open(bm25_path) as f:
        for line in f:
            item = json.loads(line)
            key = f"{item['repo']}_{item['issue_id']}"
            bm25_data[key] = item

    # Load file summaries
    with open(summary_path) as f:
        summaries = json.load(f)

    # Load GT if available
    gt_data = {}
    if test_path and os.path.exists(test_path):
        with open(test_path) as f:
            for line in f:
                item = json.loads(line)
                key = f"{item['repo']}_{item['issue_id']}"
                gt_data[key] = item

    return bm25_data, summaries, gt_data


def build_doc_text(file_path, summary):
    """Build document text from file path + summary."""
    if summary:
        return f"{file_path}\n{summary}"
    return file_path


def evaluate(model, bm25_data, summaries, gt_data, k_values, batch_size=64,
             max_candidates=200):
    """Run SweRankEmbed inference and compute Hit@K."""
    results = []
    metrics = {f"hit@{k}": [] for k in k_values}

    total = len(bm25_data)
    for idx, (key, item) in enumerate(bm25_data.items()):
        repo = item["repo"]
        issue_text = item["issue_text"]
        candidates = item["bm25_candidates"][:max_candidates]
        gt_files = set(item.get("ground_truth", []))

        # If GT from test data, use that
        if key in gt_data:
            gt_files = set(gt_data[key].get("changed_py_files", []))

        if not gt_files or not candidates:
            continue

        # Build query
        query_text = f"{QUERY_PREFIX}: {issue_text}"

        # Build document texts
        repo_summaries = summaries.get(repo, {})
        doc_texts = []
        for cand in candidates:
            summary = repo_summaries.get(cand, "")
            doc_texts.append(build_doc_text(cand, summary))

        # Encode
        with torch.no_grad():
            query_emb = model.encode([query_text], batch_size=1,
                                      show_progress_bar=False,
                                      convert_to_tensor=True)
            doc_embs = model.encode(doc_texts, batch_size=batch_size,
                                     show_progress_bar=False,
                                     convert_to_tensor=True)

            # Compute scores (dot product)
            scores = torch.matmul(query_emb, doc_embs.T).squeeze(0)
            ranked_indices = torch.argsort(scores, descending=True).cpu().numpy()

        # Get ranked file list
        ranked_files = [candidates[i] for i in ranked_indices]

        # Compute Hit@K
        for k in k_values:
            top_k = set(ranked_files[:k])
            hit = len(top_k & gt_files) / len(gt_files) * 100
            metrics[f"hit@{k}"].append(hit)

        results.append({
            "repo": repo,
            "issue_id": item["issue_id"],
            "predicted": ranked_files[:20],
            "ground_truth": list(gt_files),
        })

        if (idx + 1) % 50 == 0:
            h1 = np.mean(metrics["hit@1"]) if metrics["hit@1"] else 0
            print(f"  [{idx+1}/{total}] Hit@1: {h1:.2f}%")

    return metrics, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/home/chenlibin/models/SweRankEmbed-Large")
    parser.add_argument("--dataset", choices=["grepo", "swebench"], default="grepo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_candidates", type=int, default=200)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"experiments/baselines/swerank_{args.dataset}"

    print(f"Loading model from {args.model_path}...")
    model = SentenceTransformer(
        args.model_path, trust_remote_code=True,
        device="cpu",
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model.max_seq_length = 1024
    model = model.to(args.device)
    print(f"  Model loaded on {args.device} (bfloat16, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params)")

    print(f"Loading {args.dataset} data...")
    bm25_data, summaries, gt_data = load_test_data(args.dataset)
    print(f"  {len(bm25_data)} examples, {len(summaries)} repos with summaries")

    k_values = [1, 3, 5, 10, 20]
    print(f"Running inference (max {args.max_candidates} candidates per example)...")
    t0 = time.time()
    metrics, results = evaluate(model, bm25_data, summaries, gt_data, k_values,
                                 batch_size=args.batch_size,
                                 max_candidates=args.max_candidates)
    elapsed = time.time() - t0

    # Print results
    print(f"\n{'='*60}")
    print(f"SweRankEmbed-Large Baseline ({args.dataset})")
    print(f"{'='*60}")
    print(f"Evaluated: {len(metrics['hit@1'])} examples in {elapsed:.0f}s")
    print()
    for k in k_values:
        vals = metrics[f"hit@{k}"]
        avg = np.mean(vals) if vals else 0
        print(f"  Hit@{k:<5} {avg:>7.2f}%")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "overall": {f"hit@{k}": float(np.mean(metrics[f"hit@{k}"])) for k in k_values},
        "n_samples": len(metrics["hit@1"]),
        "model": args.model_path,
        "dataset": args.dataset,
        "max_candidates": args.max_candidates,
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.output_dir, "predictions.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
