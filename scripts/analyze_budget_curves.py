"""Compute oracle recall at different BM25 budget sizes and compare with
graph-expanded and random-expansion candidate pools.

Oracle recall = fraction of examples where ANY ground-truth file appears
in the candidate pool (at a given budget cutoff).
"""

import json
import random
from collections import OrderedDict
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/home/chenlibin/grepo_agent/data")
TEST_PATH = DATA_ROOT / "grepo_text" / "grepo_test.jsonl"
BM25_PATH = DATA_ROOT / "rankft" / "grepo_test_bm25_top500.jsonl"
GRAPH_PATH = DATA_ROOT / "rankft" / "merged_bm25_exp6_candidates.jsonl"
RANDOM_PATH = DATA_ROOT / "rankft" / "merged_random_expansion_candidates.jsonl"

BM25_BUDGETS = [50, 100, 150, 200, 250, 300, 400, 500]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def make_key(rec: dict) -> tuple:
    return (rec["repo"], rec["issue_id"])


def oracle_recall(ground_truth: list[str], candidates: set[str]) -> bool:
    """Return True if ANY ground-truth file is in the candidate set."""
    return any(f in candidates for f in ground_truth)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
test_data = load_jsonl(TEST_PATH)
bm25_data = load_jsonl(BM25_PATH)
graph_data = load_jsonl(GRAPH_PATH)
random_data = load_jsonl(RANDOM_PATH)

# Build ground-truth lookup from test data
gt_lookup: dict[tuple, list[str]] = {}
for rec in test_data:
    gt_lookup[make_key(rec)] = rec["changed_files"]

print(f"  Test examples:        {len(test_data)}")
print(f"  BM25 examples:        {len(bm25_data)}")
print(f"  Graph-expanded:       {len(graph_data)}")
print(f"  Random-expansion:     {len(random_data)}")
print()

# ---------------------------------------------------------------------------
# BM25 oracle recall at different budgets
# ---------------------------------------------------------------------------
bm25_results: dict[int, float] = OrderedDict()
for k in BM25_BUDGETS:
    hits = 0
    total = 0
    for rec in bm25_data:
        key = make_key(rec)
        gt = gt_lookup.get(key)
        if gt is None:
            continue
        pool = set(rec["bm25_candidates"][:k])
        if oracle_recall(gt, pool):
            hits += 1
        total += 1
    bm25_results[k] = hits / total if total > 0 else 0.0

# ---------------------------------------------------------------------------
# Graph-expanded oracle recall (fixed pool)
# ---------------------------------------------------------------------------
graph_hits = 0
graph_total = 0
graph_pool_sizes = []
for rec in graph_data:
    key = make_key(rec)
    gt = gt_lookup.get(key)
    if gt is None:
        continue
    pool = set(rec["candidates"])
    graph_pool_sizes.append(len(pool))
    if oracle_recall(gt, pool):
        graph_hits += 1
    graph_total += 1
graph_recall = graph_hits / graph_total if graph_total > 0 else 0.0
graph_avg_size = sum(graph_pool_sizes) / len(graph_pool_sizes) if graph_pool_sizes else 0

# ---------------------------------------------------------------------------
# Random-expansion oracle recall (fixed pool)
# ---------------------------------------------------------------------------
random_hits = 0
random_total = 0
random_pool_sizes = []
for rec in random_data:
    key = make_key(rec)
    gt = gt_lookup.get(key)
    if gt is None:
        continue
    pool = set(rec["candidates"])
    random_pool_sizes.append(len(pool))
    if oracle_recall(gt, pool):
        random_hits += 1
    random_total += 1
random_recall = random_hits / random_total if random_total > 0 else 0.0
random_avg_size = sum(random_pool_sizes) / len(random_pool_sizes) if random_pool_sizes else 0

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print("=" * 60)
print("Oracle Recall @ Top-K  (any GT file in candidate pool)")
print("=" * 60)

# BM25 table
print()
print("BM25 (ranked, variable budget):")
print(f"  {'Budget K':>10}  {'Oracle Recall':>14}  {'N examples':>10}")
print(f"  {'-'*10}  {'-'*14}  {'-'*10}")
for k, recall in bm25_results.items():
    # count how many examples have BM25 data
    n = sum(1 for rec in bm25_data if make_key(rec) in gt_lookup)
    print(f"  {k:>10}  {recall:>14.4f}  {n:>10}")

# Graph-expanded
print()
print("Graph-expanded (BM25 + 6-hop graph expansion):")
print(f"  {'Avg Pool':>10}  {'Oracle Recall':>14}  {'N examples':>10}")
print(f"  {'-'*10}  {'-'*14}  {'-'*10}")
print(f"  {graph_avg_size:>10.1f}  {graph_recall:>14.4f}  {graph_total:>10}")

# Random-expansion
print()
print("Random expansion (BM25 + random file expansion):")
print(f"  {'Avg Pool':>10}  {'Oracle Recall':>14}  {'N examples':>10}")
print(f"  {'-'*10}  {'-'*14}  {'-'*10}")
print(f"  {random_avg_size:>10.1f}  {random_recall:>14.4f}  {random_total:>10}")

# ---------------------------------------------------------------------------
# Compact comparison table
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Compact Comparison (on shared examples where all pools exist)")
print("=" * 60)

# Find shared keys across all three pools
bm25_keys = {make_key(r) for r in bm25_data}
graph_keys = {make_key(r) for r in graph_data}
random_keys = {make_key(r) for r in random_data}
shared_keys = bm25_keys & graph_keys & random_keys & set(gt_lookup.keys())

# Index pools by key for fast lookup
bm25_by_key = {make_key(r): r for r in bm25_data}
graph_by_key = {make_key(r): r for r in graph_data}
random_by_key = {make_key(r): r for r in random_data}

print(f"\nShared examples: {len(shared_keys)}")
print()
print(f"  {'Method':<30}  {'Avg Pool':>10}  {'Oracle Recall':>14}")
print(f"  {'-'*30}  {'-'*10}  {'-'*14}")

for k in BM25_BUDGETS:
    hits = sum(
        1 for key in shared_keys
        if oracle_recall(gt_lookup[key], set(bm25_by_key[key]["bm25_candidates"][:k]))
    )
    print(f"  {'BM25 top-' + str(k):<30}  {k:>10}  {hits / len(shared_keys):>14.4f}")

# Graph
g_sizes = [len(graph_by_key[key]["candidates"]) for key in shared_keys]
g_hits = sum(
    1 for key in shared_keys
    if oracle_recall(gt_lookup[key], set(graph_by_key[key]["candidates"]))
)
print(f"  {'Graph expansion':<30}  {sum(g_sizes)/len(g_sizes):>10.1f}  {g_hits / len(shared_keys):>14.4f}")

# Random
r_sizes = [len(random_by_key[key]["candidates"]) for key in shared_keys]
r_hits = sum(
    1 for key in shared_keys
    if oracle_recall(gt_lookup[key], set(random_by_key[key]["candidates"]))
)
print(f"  {'Random expansion':<30}  {sum(r_sizes)/len(r_sizes):>10.1f}  {r_hits / len(shared_keys):>14.4f}")

print()
