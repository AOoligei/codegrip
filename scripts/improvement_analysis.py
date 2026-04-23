"""
Post-hoc analysis for improving CodeGRIP results.
Three analyses:
1. Selective reranking (gating) - only apply reranker when confident
2. Upper-bound decomposition - oracle bounds for the paper
3. Error taxonomy on candidate misses
"""

import json
import numpy as np
from collections import defaultdict, Counter

# ============================================================
# Load data
# ============================================================
PRED_FILE = "experiments/rankft_runB_graph/eval_merged_rerank/predictions.jsonl"

print("Loading predictions...")
predictions = []
with open(PRED_FILE) as f:
    for line in f:
        predictions.append(json.loads(line))
print(f"  {len(predictions)} examples loaded")


def hit_at_k(predicted, ground_truth, k):
    """Check if any GT file appears in top-k predictions."""
    gt_set = set(ground_truth)
    return any(p in gt_set for p in predicted[:k])


# ============================================================
# 1. Selective Reranking (Gating)
# ============================================================
print("\n" + "="*70)
print("1. SELECTIVE RERANKING (GATING)")
print("="*70)
print("Idea: only apply reranker when confident, else keep BM25 order")

# Compute margin for each example
results = []
for pred in predictions:
    scores = pred["scores"]
    predicted = pred["predicted"]
    bm25_order = pred["bm25_original"]
    gt = pred["ground_truth"]
    gt_in = pred["gt_in_candidates"]

    margin = scores[0] - scores[1] if len(scores) >= 2 else 0

    rerank_h1 = hit_at_k(predicted, gt, 1)
    bm25_h1 = hit_at_k(bm25_order, gt, 1)

    rerank_h5 = hit_at_k(predicted, gt, 5)
    bm25_h5 = hit_at_k(bm25_order, gt, 5)

    rerank_h10 = hit_at_k(predicted, gt, 10)
    bm25_h10 = hit_at_k(bm25_order, gt, 10)

    results.append({
        "margin": margin,
        "rerank_h1": rerank_h1,
        "bm25_h1": bm25_h1,
        "rerank_h5": rerank_h5,
        "bm25_h5": bm25_h5,
        "rerank_h10": rerank_h10,
        "bm25_h10": bm25_h10,
        "gt_in": gt_in,
        "top1_score": scores[0] if scores else 0,
    })

# Strategy 1: Margin-based gating
print("\nStrategy: Use reranker if margin > threshold, else keep BM25")
print(f"{'Threshold':>10} | {'H@1':>8} | {'H@5':>8} | {'H@10':>8} | {'%Reranked':>10}")
print("-" * 60)

best_h1 = 0
best_thresh = 0

for thresh in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
    h1_hits = 0
    h5_hits = 0
    h10_hits = 0
    n_reranked = 0
    for r in results:
        if r["margin"] >= thresh:
            # Use reranker
            h1_hits += r["rerank_h1"]
            h5_hits += r["rerank_h5"]
            h10_hits += r["rerank_h10"]
            n_reranked += 1
        else:
            # Keep BM25
            h1_hits += r["bm25_h1"]
            h5_hits += r["bm25_h5"]
            h10_hits += r["bm25_h10"]

    n = len(results)
    h1_pct = h1_hits / n * 100
    h5_pct = h5_hits / n * 100
    h10_pct = h10_hits / n * 100
    pct_reranked = n_reranked / n * 100

    if h1_pct > best_h1:
        best_h1 = h1_pct
        best_thresh = thresh

    marker = " <-- best" if h1_pct == best_h1 and thresh == best_thresh else ""
    print(f"{thresh:>10.1f} | {h1_pct:>7.2f}% | {h5_pct:>7.2f}% | {h10_pct:>7.2f}% | {pct_reranked:>9.1f}%{marker}")

# All reranked (baseline)
h1_all = sum(r["rerank_h1"] for r in results) / len(results) * 100
h5_all = sum(r["rerank_h5"] for r in results) / len(results) * 100
print(f"\nBaselines: Reranker-only H@1={h1_all:.2f}%, BM25-only H@1={sum(r['bm25_h1'] for r in results)/len(results)*100:.2f}%")

# Strategy 2: Score-based gating (use reranker if top-1 score > threshold)
print("\nStrategy 2: Use reranker if top-1 score > threshold, else keep BM25")
print(f"{'ScoreThresh':>12} | {'H@1':>8} | {'H@5':>8} | {'%Reranked':>10}")
print("-" * 50)

for thresh in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
    h1_hits = 0
    h5_hits = 0
    n_reranked = 0
    for r in results:
        if r["top1_score"] >= thresh:
            h1_hits += r["rerank_h1"]
            h5_hits += r["rerank_h5"]
            n_reranked += 1
        else:
            h1_hits += r["bm25_h1"]
            h5_hits += r["bm25_h5"]
    n = len(results)
    print(f"{thresh:>12.1f} | {h1_hits/n*100:>7.2f}% | {h5_hits/n*100:>7.2f}% | {n_reranked/n*100:>9.1f}%")

# Strategy 3: Score fusion (linear combination)
print("\nStrategy 3: Score fusion = alpha * reranker_rank + (1-alpha) * bm25_rank")
print(f"{'Alpha':>8} | {'H@1':>8} | {'H@5':>8} | {'H@10':>8}")
print("-" * 50)

for alpha in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]:
    h1_hits = 0
    h5_hits = 0
    h10_hits = 0
    for pred in predictions:
        gt = set(pred["ground_truth"])
        predicted = pred["predicted"]
        bm25_order = pred["bm25_original"]

        # Build rank maps
        rerank_rank = {f: i for i, f in enumerate(predicted)}
        bm25_rank = {f: i for i, f in enumerate(bm25_order)}

        # All candidates
        all_files = list(set(predicted) | set(bm25_order))

        # Fused rank score (lower is better)
        fused = []
        for f in all_files:
            r_rank = rerank_rank.get(f, len(predicted))
            b_rank = bm25_rank.get(f, len(bm25_order))
            score = alpha * r_rank + (1 - alpha) * b_rank
            fused.append((score, f))
        fused.sort()

        fused_order = [f for _, f in fused]
        h1_hits += hit_at_k(fused_order, pred["ground_truth"], 1)
        h5_hits += hit_at_k(fused_order, pred["ground_truth"], 5)
        h10_hits += hit_at_k(fused_order, pred["ground_truth"], 10)

    n = len(predictions)
    print(f"{alpha:>8.1f} | {h1_hits/n*100:>7.2f}% | {h5_hits/n*100:>7.2f}% | {h10_hits/n*100:>7.2f}%")


# ============================================================
# 2. Upper-Bound Decomposition
# ============================================================
print("\n" + "="*70)
print("2. UPPER-BOUND DECOMPOSITION")
print("="*70)

n = len(predictions)
gt_in_pool = sum(1 for p in predictions if p["gt_in_candidates"])
gt_not_in = n - gt_in_pool

print(f"Total examples: {n}")
print(f"GT in candidate pool: {gt_in_pool} ({gt_in_pool/n*100:.1f}%)")
print(f"GT NOT in pool: {gt_not_in} ({gt_not_in/n*100:.1f}%)")

# Oracle reranker: if GT is in pool, always rank it #1
oracle_h1 = gt_in_pool / n * 100
print(f"\nOracle reranker H@1 (upper bound): {oracle_h1:.1f}%")
print(f"  → Even perfect reranking caps at {oracle_h1:.1f}% H@1")

# Conditional metrics (only examples where GT is in pool)
cond_h1 = sum(1 for p in predictions if p["gt_in_candidates"] and hit_at_k(p["predicted"], p["ground_truth"], 1))
cond_h5 = sum(1 for p in predictions if p["gt_in_candidates"] and hit_at_k(p["predicted"], p["ground_truth"], 5))
cond_h10 = sum(1 for p in predictions if p["gt_in_candidates"] and hit_at_k(p["predicted"], p["ground_truth"], 10))

print(f"\nConditional metrics (given GT in pool, n={gt_in_pool}):")
print(f"  Cond H@1:  {cond_h1/gt_in_pool*100:.1f}% ({cond_h1}/{gt_in_pool})")
print(f"  Cond H@5:  {cond_h5/gt_in_pool*100:.1f}% ({cond_h5}/{gt_in_pool})")
print(f"  Cond H@10: {cond_h10/gt_in_pool*100:.1f}% ({cond_h10}/{gt_in_pool})")

# Decomposition: H@1 = P(GT in pool) * P(rank=1 | GT in pool)
actual_h1 = sum(1 for p in predictions if hit_at_k(p["predicted"], p["ground_truth"], 1))
print(f"\nDecomposition:")
print(f"  H@1 = P(GT in pool) × P(rank=1|GT in pool)")
print(f"  {actual_h1/n*100:.1f}% = {gt_in_pool/n*100:.1f}% × {cond_h1/gt_in_pool*100:.1f}%")
print(f"  → Improving candidate recall from {gt_in_pool/n*100:.1f}% to 85% would give:")
print(f"     0.85 × {cond_h1/gt_in_pool*100:.1f}% = {0.85 * cond_h1/gt_in_pool*100:.1f}% H@1")
print(f"  → Improving cond. accuracy from {cond_h1/gt_in_pool*100:.1f}% to 50% would give:")
print(f"     {gt_in_pool/n*100:.1f}% × 50% = {gt_in_pool/n*100 * 0.5:.1f}% H@1")

# BM25 conditional
bm25_cond_h1 = sum(1 for p in predictions if p["gt_in_candidates"] and hit_at_k(p["bm25_original"], p["ground_truth"], 1))
print(f"\nBM25 Cond H@1: {bm25_cond_h1/gt_in_pool*100:.1f}% ({bm25_cond_h1}/{gt_in_pool})")
print(f"Reranker lift over BM25: {(cond_h1 - bm25_cond_h1)/gt_in_pool*100:.1f}%")

# Pool size statistics
pool_sizes = [p["num_candidates"] for p in predictions]
print(f"\nCandidate pool stats:")
print(f"  Mean: {np.mean(pool_sizes):.1f}, Median: {np.median(pool_sizes):.0f}")
print(f"  Min: {min(pool_sizes)}, Max: {max(pool_sizes)}")


# ============================================================
# 3. Error Taxonomy on Candidate Misses
# ============================================================
print("\n" + "="*70)
print("3. ERROR TAXONOMY ON CANDIDATE MISSES (29.3%)")
print("="*70)

# Load test data for issue text and GT details
print("Loading test data for issue details...")
test_data = {}
with open("data/grepo_text/grepo_test.jsonl") as f:
    for line in f:
        item = json.loads(line)
        key = (item["repo"], item.get("issue_id", ""))
        test_data[key] = item

# Load candidate data
print("Loading candidate data...")
cand_data = {}
with open("data/rankft/merged_bm25_exp6_candidates.jsonl") as f:
    for line in f:
        item = json.loads(line)
        key = (item["repo"], item.get("issue_id", ""))
        cand_data[key] = item

# Analyze misses
misses = [p for p in predictions if not p["gt_in_candidates"]]
print(f"\nTotal misses: {len(misses)} ({len(misses)/len(predictions)*100:.1f}%)")

# Taxonomy
miss_by_repo = defaultdict(int)
total_by_repo = defaultdict(int)
miss_gt_count = []  # how many GT files per missed example
miss_pool_size = []
miss_reasons = defaultdict(int)

for pred in predictions:
    repo = pred["repo"]
    total_by_repo[repo] += 1
    if not pred["gt_in_candidates"]:
        miss_by_repo[repo] += 1
        miss_gt_count.append(len(pred["ground_truth"]))
        miss_pool_size.append(pred["num_candidates"])

        # Check if ANY GT file is in candidates (partial miss)
        gt_set = set(pred["ground_truth"])
        cand_set = set(pred["predicted"])  # all candidates
        overlap = gt_set & cand_set
        if len(overlap) > 0:
            miss_reasons["partial_miss"] += 1
        else:
            miss_reasons["complete_miss"] += 1

        # Check GT file characteristics
        key = (pred["repo"], pred.get("issue_id", ""))
        test_item = test_data.get(key, {})
        gt_files = pred["ground_truth"]

        # Is GT a deeply nested file?
        max_depth = max(f.count("/") for f in gt_files)
        if max_depth >= 4:
            miss_reasons["deep_nesting"] += 1

        # Is GT an __init__.py?
        if any("__init__" in f for f in gt_files):
            miss_reasons["init_file"] += 1

        # Is GT a test file?
        if any("test" in f.lower() for f in gt_files):
            miss_reasons["test_file"] += 1

        # Multiple GT files?
        if len(gt_files) > 1:
            miss_reasons["multi_gt"] += 1

        # Small pool?
        if pred["num_candidates"] < 10:
            miss_reasons["small_pool"] += 1

print(f"\nMiss breakdown:")
for reason, count in sorted(miss_reasons.items(), key=lambda x: -x[1]):
    print(f"  {reason}: {count} ({count/len(misses)*100:.1f}%)")

print(f"\nMiss stats:")
print(f"  Mean GT count: {np.mean(miss_gt_count):.1f}")
print(f"  Mean pool size: {np.mean(miss_pool_size):.1f}")

# Per-repo miss rates
print(f"\nPer-repo miss rates (repos with >10 examples):")
print(f"{'Repo':>25} | {'Misses':>7} | {'Total':>7} | {'MissRate':>9}")
print("-" * 60)
for repo in sorted(miss_by_repo, key=lambda r: miss_by_repo[r]/max(total_by_repo[r],1), reverse=True):
    if total_by_repo[repo] >= 10:
        rate = miss_by_repo[repo] / total_by_repo[repo] * 100
        print(f"{repo:>25} | {miss_by_repo[repo]:>7} | {total_by_repo[repo]:>7} | {rate:>8.1f}%")


# ============================================================
# 4. Reranker Help vs Hurt Analysis
# ============================================================
print("\n" + "="*70)
print("4. RERANKER HELP vs HURT ANALYSIS")
print("="*70)

helped = 0
hurt = 0
same = 0
helped_repos = defaultdict(int)
hurt_repos = defaultdict(int)

for pred in predictions:
    if not pred["gt_in_candidates"]:
        same += 1
        continue

    gt = set(pred["ground_truth"])
    predicted = pred["predicted"]
    bm25_order = pred["bm25_original"]

    # Find best GT rank in each
    def best_gt_rank(order, gt_set):
        for i, f in enumerate(order):
            if f in gt_set:
                return i
        return len(order)

    rerank_best = best_gt_rank(predicted, gt)
    bm25_best = best_gt_rank(bm25_order, gt)

    if rerank_best < bm25_best:
        helped += 1
        helped_repos[pred["repo"]] += 1
    elif rerank_best > bm25_best:
        hurt += 1
        hurt_repos[pred["repo"]] += 1
    else:
        same += 1

n = len(predictions)
print(f"Reranker helped: {helped} ({helped/n*100:.1f}%)")
print(f"Reranker hurt:   {hurt} ({hurt/n*100:.1f}%)")
print(f"Same / N/A:      {same} ({same/n*100:.1f}%)")

# Per-repo help/hurt
print(f"\nPer-repo reranker impact (repos with >10 examples):")
print(f"{'Repo':>25} | {'Helped':>8} | {'Hurt':>8} | {'Net':>8}")
print("-" * 60)
all_repos = set(list(helped_repos.keys()) + list(hurt_repos.keys()))
for repo in sorted(all_repos, key=lambda r: helped_repos.get(r,0) - hurt_repos.get(r,0), reverse=True):
    if total_by_repo[repo] >= 10:
        h = helped_repos.get(repo, 0)
        hu = hurt_repos.get(repo, 0)
        print(f"{repo:>25} | {h:>8} | {hu:>8} | {h-hu:>+8}")

print("\nDone!")
