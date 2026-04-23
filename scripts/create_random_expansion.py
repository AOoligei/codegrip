#!/usr/bin/env python3
"""
Create a random expansion control: replace graph neighbors with random
Python files from the same repo, keeping pool size identical.
This controls for pool size to show graph expansion is not just "more candidates".
"""
import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

random.seed(42)
np.random.seed(42)

DATA = Path("data")
MERGED = DATA / "rankft/merged_bm25_exp6_candidates.jsonl"
BM25 = DATA / "rankft/grepo_test_bm25_top500.jsonl"
FILE_TREES = DATA / "file_trees"
OUTPUT = DATA / "rankft/merged_random_expansion_candidates.jsonl"


def main():
    # Load file trees (all Python files per repo)
    repo_py_files = {}
    for ft_path in FILE_TREES.glob("*.json"):
        ft = json.loads(ft_path.read_text())
        repo_name = ft["repo"]
        repo_py_files[repo_name] = ft["py_files"]

    # Load BM25 candidates
    bm25_data = {}
    for line in open(BM25):
        d = json.loads(line)
        key = (d["repo"], d["issue_id"])
        bm25_data[key] = d

    # Load merged candidates to match pool composition
    merged_data = []
    for line in open(MERGED):
        merged_data.append(json.loads(line))

    # Create random expansion control
    output = []
    stats = {"matched": 0, "no_bm25": 0, "no_files": 0}

    for m in merged_data:
        key = (m["repo"], m["issue_id"])
        if key not in bm25_data:
            # Non-test example, skip
            stats["no_bm25"] += 1
            continue

        b = bm25_data[key]
        merged_set = set(m["candidates"])
        bm25_set = set(b["bm25_candidates"])

        # Identify which candidates came from BM25 vs graph expansion
        from_bm25 = [c for c in m["candidates"] if c in bm25_set]
        from_graph = [c for c in m["candidates"] if c not in bm25_set]
        n_graph = len(from_graph)

        # Sample same number of random Python files (not in BM25 pool)
        all_py = repo_py_files.get(m["repo"], [])
        available = [f for f in all_py if f not in bm25_set and f not in merged_set]

        if len(available) >= n_graph:
            random_files = random.sample(available, n_graph)
        else:
            random_files = available  # take all available
            stats["no_files"] += 1

        # Merge: same BM25 subset + random files instead of graph neighbors
        random_merged = from_bm25 + random_files

        output.append({
            "repo": m["repo"],
            "issue_id": m["issue_id"],
            "candidates": random_merged,
        })
        stats["matched"] += 1

    # Write output
    with open(OUTPUT, "w") as f:
        for item in output:
            f.write(json.dumps(item) + "\n")

    # Stats
    pool_sizes = [len(item["candidates"]) for item in output]
    orig_sizes = [len(m["candidates"]) for m in merged_data
                  if (m["repo"], m["issue_id"]) in bm25_data]

    print(f"Created {len(output)} examples")
    print(f"Random pool: mean={np.mean(pool_sizes):.1f}, median={np.median(pool_sizes):.0f}")
    print(f"Graph pool:  mean={np.mean(orig_sizes):.1f}, median={np.median(orig_sizes):.0f}")
    print(f"Stats: {stats}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
