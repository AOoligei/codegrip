#!/usr/bin/env python3
"""
Prepare repo-held-out training data.

Removes the top-15 repos (by test count) from training data.
The model trained on remaining repos is then evaluated on the held-out repos.

This tests whether path prior conclusions generalize to truly unseen repos.

Usage:
    python scripts/prepare_repo_held_out.py
"""

import json
import os
from collections import Counter

TRAIN_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_train.jsonl"
TEST_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_test.jsonl"
BM25_TRAIN = "/home/chenlibin/grepo_agent/data/rankft/grepo_train_bm25_top500.jsonl"
OUT_DIR = "/data/chenlibin/grepo_agent_experiments/repo_held_out"

# Top-15 repos by test count to hold out
HELD_OUT_REPOS = {
    'scipy', 'jax', 'sqlfluff', 'xarray', 'astropy',
    'networkx', 'datasets', 'litellm', 'ipython', 'Cirq',
    'dvc', 'pyvista', 'pylint', 'geopandas', 'sphinx',
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Filter training data
    kept = 0
    removed = 0
    train_out = os.path.join(OUT_DIR, "grepo_train_held_out.jsonl")
    with open(TRAIN_PATH) as fin, open(train_out, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            if rec["repo"] in HELD_OUT_REPOS:
                removed += 1
            else:
                fout.write(line)
                kept += 1
    print(f"Train: kept {kept}, removed {removed} (held-out repos)")

    # Filter BM25 candidates
    kept_bm25 = 0
    removed_bm25 = 0
    bm25_out = os.path.join(OUT_DIR, "grepo_train_bm25_held_out.jsonl")
    with open(BM25_TRAIN) as fin, open(bm25_out, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            if rec["repo"] in HELD_OUT_REPOS:
                removed_bm25 += 1
            else:
                fout.write(line)
                kept_bm25 += 1
    print(f"BM25: kept {kept_bm25}, removed {removed_bm25}")

    # Create held-out test split (only held-out repos)
    held_test = 0
    test_out = os.path.join(OUT_DIR, "grepo_test_held_out_repos.jsonl")
    with open(TEST_PATH) as fin, open(test_out, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            if rec["repo"] in HELD_OUT_REPOS:
                fout.write(line)
                held_test += 1
    print(f"Held-out test: {held_test} examples from {len(HELD_OUT_REPOS)} repos")

    print(f"\nOutput: {OUT_DIR}")
    print(f"  {train_out}")
    print(f"  {bm25_out}")
    print(f"  {test_out}")


if __name__ == "__main__":
    main()
