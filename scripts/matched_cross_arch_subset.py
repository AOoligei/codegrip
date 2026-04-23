"""
Select 200 examples for matched cross-architecture evaluation.

Finds examples present in all candidate pools and does stratified
sampling by repo (proportional to repo size in test set).

Usage:
    python scripts/matched_cross_arch_subset.py

Deterministic (seed 42), CPU-only, standard libraries + numpy.
"""

import json
import numpy as np
from collections import Counter


def load_jsonl(path):
    """Load JSONL, return list of dicts."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def make_key(rec):
    """Make a (repo, issue_id_str) key."""
    return (rec["repo"], str(rec["issue_id"]))


def main():
    np.random.seed(42)

    target_n = 200

    # Load test data
    test_path = "data/grepo_text/grepo_test.jsonl"
    test_data = load_jsonl(test_path)
    test_keys = {make_key(r): r for r in test_data}
    print(f"Test data: {len(test_data)} examples")

    # Load candidate pools
    pool_paths = [
        "data/rankft/merged_bm25_exp6_candidates.jsonl",
        "experiments/path_perturb_shuffle_filenames/bm25_candidates.jsonl",
    ]

    pool_key_sets = []
    for path in pool_paths:
        pool_data = load_jsonl(path)
        keys = set(make_key(r) for r in pool_data)
        pool_key_sets.append(keys)
        print(f"Pool {path}: {len(keys)} examples")

    # Find examples in ALL pools AND in test set
    common_keys = set(test_keys.keys())
    for pks in pool_key_sets:
        common_keys &= pks
    common_keys = sorted(common_keys)
    print(f"\nExamples in ALL pools: {len(common_keys)}")

    if len(common_keys) < target_n:
        print(f"WARNING: Only {len(common_keys)} common examples, less than target {target_n}")
        target_n = len(common_keys)

    # Group by repo
    repo_to_keys = {}
    for key in common_keys:
        repo = key[0]
        if repo not in repo_to_keys:
            repo_to_keys[repo] = []
        repo_to_keys[repo].append(key)

    # Compute test set repo distribution for proportional sampling
    test_repo_counts = Counter(r["repo"] for r in test_data)
    total_test = len(test_data)

    print(f"\nRepos in common pool: {len(repo_to_keys)}")

    # Stratified sampling: allocate proportionally, then sample
    # First compute allocation
    repo_allocation = {}
    remaining = target_n
    repos_sorted = sorted(repo_to_keys.keys())

    # Proportional allocation based on test set distribution
    raw_alloc = {}
    for repo in repos_sorted:
        test_frac = test_repo_counts.get(repo, 0) / total_test
        raw_alloc[repo] = test_frac * target_n

    # Round down, then distribute remainder by largest fractional part
    for repo in repos_sorted:
        repo_allocation[repo] = int(raw_alloc[repo])

    allocated = sum(repo_allocation.values())
    remainder = target_n - allocated

    # Sort by fractional part descending to distribute remainder
    frac_parts = [(repo, raw_alloc[repo] - int(raw_alloc[repo])) for repo in repos_sorted]
    frac_parts.sort(key=lambda x: -x[1])

    for i in range(remainder):
        repo = frac_parts[i][0]
        repo_allocation[repo] += 1

    # Cap allocation at available examples per repo
    for repo in repos_sorted:
        avail = len(repo_to_keys[repo])
        if repo_allocation[repo] > avail:
            repo_allocation[repo] = avail

    # If capping reduced total, redistribute
    actual_total = sum(repo_allocation.values())
    if actual_total < target_n:
        # Give extras to repos with remaining capacity
        deficit = target_n - actual_total
        for repo in repos_sorted:
            if deficit == 0:
                break
            avail = len(repo_to_keys[repo])
            can_add = avail - repo_allocation[repo]
            add = min(can_add, deficit)
            repo_allocation[repo] += add
            deficit -= add

    # Sample from each repo
    rng = np.random.RandomState(42)
    selected = []
    for repo in repos_sorted:
        n_select = repo_allocation[repo]
        if n_select == 0:
            continue
        keys = repo_to_keys[repo]
        if n_select >= len(keys):
            chosen = keys
        else:
            indices = rng.choice(len(keys), size=n_select, replace=False)
            indices.sort()
            chosen = [keys[i] for i in indices]
        selected.extend(chosen)

    print(f"\nSelected: {len(selected)} examples")

    # Print repo distribution
    sel_repo_counts = Counter(k[0] for k in selected)
    print(f"\n{'Repo':<30s} {'Test':>6s} {'Common':>8s} {'Selected':>10s}")
    print("-" * 56)
    for repo in sorted(sel_repo_counts.keys()):
        print(f"{repo:<30s} {test_repo_counts.get(repo,0):>6d} {len(repo_to_keys.get(repo,[])):>8d} {sel_repo_counts[repo]:>10d}")

    # Save output
    output = []
    for key in selected:
        output.append({"repo": key[0], "issue_id": key[1]})

    out_path = "data/matched_cross_arch_200.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
