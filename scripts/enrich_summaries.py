"""Enrich existing summary.json files with per_repo, bootstrap CI, and strict Acc@k.
Uses predictions.jsonl (already saved by eval script) — no GPU needed.

Usage: python scripts/enrich_summaries.py experiments/*/eval_*/
"""
import json, os, sys, glob
import numpy as np

np.random.seed(42)
GREPO_9 = {'astropy', 'dvc', 'ipython', 'pylint', 'scipy', 'sphinx', 'streamlink', 'xarray', 'geopandas'}

def compute_hit_at_k(predicted, gt, k):
    if not gt:
        return 0.0
    top_k = set(predicted[:k])
    return len(top_k & gt) / len(gt)

def enrich(eval_dir):
    pred_path = os.path.join(eval_dir, "predictions.jsonl")
    summary_path = os.path.join(eval_dir, "summary.json")

    if not os.path.exists(pred_path):
        return False
    if not os.path.exists(summary_path):
        return False

    # Load predictions
    predictions = []
    for l in open(pred_path):
        try:
            predictions.append(json.loads(l))
        except json.JSONDecodeError:
            continue  # skip corrupted lines
    if not predictions:
        return False

    # Load existing summary
    summary = json.load(open(summary_path))

    # Skip if missing overall or already enriched
    if "overall" not in summary:
        return False
    if "per_repo" in summary and summary["per_repo"] and "bootstrap_ci" in summary \
       and "ndcg@1" in summary.get("overall", {}):
        return False

    k_values = [1, 3, 5, 10, 20]

    # Compute metrics
    hit_at_k = {k: [] for k in k_values}
    strict_acc = {k: [] for k in k_values}

    for p in predictions:
        gt = set(p["ground_truth"])
        pred = p["predicted"]
        for k in k_values:
            hit_at_k[k].append(compute_hit_at_k(pred, gt, k))
            # Empty GT → 0.0 (not vacuously True)
            strict_acc[k].append(1.0 if gt and gt <= set(pred[:k]) else 0.0)

    # Update overall with strict Acc@k
    for k in k_values:
        summary["overall"][f"acc@{k}"] = float(np.mean(strict_acc[k]) * 100)

    # NDCG@k (binary relevance, ideal DCG from full GT set, not just observed)
    def dcg(relevance_list, k):
        rel = relevance_list[:k]
        return sum(r / np.log2(i + 2) for i, r in enumerate(rel))

    ndcg_at_k = {k: [] for k in k_values}
    for p in predictions:
        gt = set(p["ground_truth"])
        if not gt:
            for k in k_values:
                ndcg_at_k[k].append(0.0)
            continue
        pred = p["predicted"]
        # binary relevance: 1 if file is GT, 0 otherwise
        rels = [1.0 if f in gt else 0.0 for f in pred]
        # Ideal: min(|GT|, k) relevant files at top positions
        for k in k_values:
            n_relevant = min(len(gt), k)
            ideal_rels_k = [1.0] * n_relevant + [0.0] * (k - n_relevant)
            idcg = dcg(ideal_rels_k, k)
            if idcg > 0:
                ndcg_at_k[k].append(dcg(rels, k) / idcg)
            else:
                ndcg_at_k[k].append(0.0)
    for k in k_values:
        summary["overall"][f"ndcg@{k}"] = float(np.mean(ndcg_at_k[k]) * 100)

    # Per-repo breakdown
    from collections import defaultdict
    per_repo_preds = defaultdict(list)
    for p in predictions:
        per_repo_preds[p["repo"]].append(p)

    per_repo = {}
    for repo, preds in sorted(per_repo_preds.items()):
        repo_metrics = {"n_examples": len(preds)}
        for k in k_values:
            repo_h = [compute_hit_at_k(p["predicted"], set(p["ground_truth"]), k) for p in preds]
            repo_metrics[f"hit@{k}"] = float(np.mean(repo_h) * 100)
            repo_sa = [1.0 if set(p["ground_truth"]) <= set(p["predicted"][:k]) else 0.0 for p in preds]
            repo_metrics[f"acc@{k}"] = float(np.mean(repo_sa) * 100)
        per_repo[repo] = repo_metrics
    summary["per_repo"] = per_repo

    # Bootstrap CI
    bootstrap_ci = {}
    for k in k_values:
        vals = np.array(hit_at_k[k])
        boot = [float(np.mean(np.random.choice(vals, len(vals), replace=True)) * 100) for _ in range(10000)]
        bootstrap_ci[f"recall@{k}"] = {
            "mean": float(np.mean(boot)),
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
        }
    summary["bootstrap_ci"] = bootstrap_ci

    # Add config fields if missing
    if "config" not in summary:
        summary["config"] = {}
    summary["config"]["total_examples"] = len(predictions)

    # Save
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    ci1 = bootstrap_ci["recall@1"]
    nine_repos = [r for r in per_repo if r.lower() in GREPO_9]
    if nine_repos:
        total_n = sum(per_repo[r]["n_examples"] for r in nine_repos)
        nine_h1 = sum(per_repo[r]["hit@1"] * per_repo[r]["n_examples"] for r in nine_repos) / total_n
    else:
        nine_h1 = float('nan')

    print(f"  {eval_dir}")
    print(f"    R@1={summary['overall']['hit@1']:.2f}% [{ci1['ci_lo']:.2f}, {ci1['ci_hi']:.2f}]")
    print(f"    Acc@1={summary['overall']['acc@1']:.2f}%  9-repo R@1={nine_h1:.2f}%")
    print(f"    {len(per_repo)} repos, {len(predictions)} examples")
    return True

if __name__ == "__main__":
    dirs = sys.argv[1:] if len(sys.argv) > 1 else []
    if not dirs:
        # Auto-find all eval dirs
        dirs = sorted(set(
            glob.glob("experiments/**/eval_*/", recursive=True) +
            glob.glob("/data/chenlibin/grepo_agent_experiments/**/eval_*/", recursive=True)
        ))

    enriched = 0
    for d in dirs:
        d = d.rstrip("/")
        if enrich(d):
            enriched += 1

    print(f"\nEnriched {enriched} summary files")
