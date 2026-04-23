#!/usr/bin/env python3
"""
Build hybrid (BM25 + E5-large dense) candidate pool for SWE-bench.
Merges BM25 top-500 with dense top-K, producing a unified candidate list
with higher oracle recall.

Usage:
    python scripts/swebench_build_hybrid_pool.py \
        --bm25_candidates data/rankft/swebench_test_bm25_top500.jsonl \
        --dense_candidates experiments/swebench/dense_e5large/dense_candidates.jsonl \
        --output data/rankft/swebench_hybrid_bm25_e5large.jsonl
"""
import json
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25_candidates", required=True)
    parser.add_argument("--dense_candidates", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load BM25 candidates (keyed by issue_id)
    bm25_data = {}
    with open(args.bm25_candidates) as f:
        for line in f:
            d = json.loads(line)
            bm25_data[d["issue_id"]] = d

    # Load dense candidates
    dense_data = {}
    with open(args.dense_candidates) as f:
        for line in f:
            d = json.loads(line)
            dense_data[d["issue_id"]] = d

    print(f"BM25: {len(bm25_data)} examples")
    print(f"Dense: {len(dense_data)} examples")

    # Merge
    results = []
    bm25_only_oracle = []
    hybrid_oracle = []
    n_added = []

    for issue_id in sorted(bm25_data.keys()):
        bm25 = bm25_data[issue_id]
        gt = set(bm25["ground_truth"])
        bm25_cands = list(bm25["bm25_candidates"])
        bm25_set = set(bm25_cands)

        # BM25 oracle recall
        bm25_hit = len(bm25_set & gt) / len(gt) * 100 if gt else 0
        bm25_only_oracle.append(bm25_hit)

        # Merge with dense candidates
        if issue_id in dense_data:
            dense_cands = dense_data[issue_id]["dense_candidates"]
            added = 0
            for fp in dense_cands:
                if fp not in bm25_set:
                    bm25_cands.append(fp)
                    bm25_set.add(fp)
                    added += 1
            n_added.append(added)
        else:
            n_added.append(0)

        # Hybrid oracle recall
        hybrid_hit = len(bm25_set & gt) / len(gt) * 100 if gt else 0
        hybrid_oracle.append(hybrid_hit)

        # Check GT coverage
        gt_in = all(g in bm25_set for g in gt)

        result = {
            "repo": bm25["repo"],
            "issue_id": issue_id,
            "issue_text": bm25.get("issue_text", ""),
            "ground_truth": bm25["ground_truth"],
            "bm25_candidates": bm25_cands,
            "gt_in_candidates": gt_in,
            "num_bm25_original": len(bm25["bm25_candidates"]),
            "num_dense_added": n_added[-1],
            "num_total_candidates": len(bm25_cands),
        }
        results.append(result)

    # Stats
    bm25_oracle = np.mean(bm25_only_oracle)
    hybrid_oracle_mean = np.mean(hybrid_oracle)
    avg_added = np.mean(n_added)
    avg_total = np.mean([r["num_total_candidates"] for r in results])

    print(f"\n=== Merge Statistics ===")
    print(f"BM25 oracle recall:   {bm25_oracle:.2f}%")
    print(f"Hybrid oracle recall: {hybrid_oracle_mean:.2f}%")
    print(f"Oracle recall gain:   +{hybrid_oracle_mean - bm25_oracle:.2f}pp")
    print(f"Avg dense files added: {avg_added:.1f}")
    print(f"Avg total candidates:  {avg_total:.1f}")
    print(f"Examples with oracle improvement: "
          f"{sum(1 for b, h in zip(bm25_only_oracle, hybrid_oracle) if h > b)}/{len(results)}")

    # Save
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(results)} examples to {args.output}")


if __name__ == "__main__":
    main()
