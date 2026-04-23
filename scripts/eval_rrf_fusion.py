#!/usr/bin/env python3
"""Training-free: Reciprocal Rank Fusion of our path reranker + SweRankEmbed.

Hypothesis: path-reliant reranker + code-reliant dense retriever have partially
complementary failure modes. RRF combines rank orders to reduce path-confound.

Usage:
  python scripts/eval_rrf_fusion.py \
      --path_ranks_file /data/chenlibin/grepo_agent_experiments/ranking/swe_point_code/predictions.jsonl \
      --swerank_candidates <swerankembed top-K json> \
      --test_data data/swebench_lite/swebench_lite_test.jsonl \
      --output_dir ...
"""
import argparse, json, os
import numpy as np


def rrf_score(rank, k=60):
    """Standard RRF formula."""
    return 1.0 / (k + rank)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_predictions", required=True,
                    help="jsonl with keys: repo, issue_id, predicted (list in rank order), ground_truth")
    ap.add_argument("--code_predictions", required=True,
                    help="same schema, predicted = SweRankEmbed-ranked list")
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--k", type=int, default=60, help="RRF constant")
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--alpha_sweep", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load predictions indexed by (repo, issue_id)
    path_preds = {}
    for l in open(args.path_predictions):
        r = json.loads(l)
        key = (r.get("repo", ""), str(r.get("issue_id", "")))
        path_preds[key] = r
    code_preds = {}
    for l in open(args.code_predictions):
        r = json.loads(l)
        key = (r.get("repo", ""), str(r.get("issue_id", "")))
        code_preds[key] = r

    test = [json.loads(l) for l in open(args.test_data)]
    alphas = [float(x) for x in args.alpha_sweep.split(",")]
    results = {a: [] for a in alphas}

    n = 0
    for rec in test:
        key = (rec.get("repo", ""), str(rec.get("issue_id", "")))
        if key not in path_preds or key not in code_preds: continue
        gt = set(rec.get("changed_py_files", rec.get("changed_files", [])))
        if not gt: continue

        p_list = path_preds[key].get("predicted", [])[:args.top_k]
        c_list = code_preds[key].get("predicted", [])[:args.top_k]
        union = list(dict.fromkeys(p_list + c_list))  # preserve order
        p_rank = {f: i+1 for i, f in enumerate(p_list)}
        c_rank = {f: i+1 for i, f in enumerate(c_list)}

        for a in alphas:
            scores = {}
            for f in union:
                pr = rrf_score(p_rank.get(f, args.top_k + 1), args.k)
                cr = rrf_score(c_rank.get(f, args.top_k + 1), args.k)
                scores[f] = a * pr + (1 - a) * cr
            ranked = sorted(union, key=lambda f: -scores[f])
            results[a].append(1.0 if ranked[0] in gt else 0.0)
        n += 1

    summary = {"n": n}
    for a in alphas:
        summary[f"alpha={a:.2f}"] = float(np.mean(results[a]) * 100) if results[a] else 0.0
    summary["best_alpha"] = max(alphas, key=lambda a: np.mean(results[a]) if results[a] else 0)
    summary["best_r1"] = max(summary[f"alpha={a:.2f}"] for a in alphas)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
