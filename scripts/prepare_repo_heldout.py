#!/usr/bin/env python3
"""
Prepare repo-held-out train/eval split for CodeGRIP reranker.

Tests whether path dependency holds when the model is evaluated on completely
unseen repositories (no repo overlap between train and test).

Steps:
  1. Load GREPO train and test JSONL
  2. Identify repos in test set, sorted by #test examples
  3. Select top-15 repos as "held-out" (removed from training entirely)
  4. Write filtered training data + held-out test data + filtered BM25 candidates
  5. Also create shuffle_filenames perturbation for held-out test

Usage:
    python scripts/prepare_repo_heldout.py
"""

import json
import os
import random
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "grepo_text" / "grepo_train.jsonl"
TEST_PATH = BASE_DIR / "data" / "grepo_text" / "grepo_test.jsonl"
BM25_TRAIN_PATH = BASE_DIR / "data" / "rankft" / "grepo_train_bm25_top500.jsonl"
BM25_TEST_PATH = BASE_DIR / "data" / "rankft" / "merged_bm25_exp6_candidates.jsonl"

OUT_DIR = Path("/data/chenlibin/grepo_agent_experiments/repo_heldout")

NUM_HELDOUT_REPOS = 15


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"  Wrote {len(data)} records to {path}")


# ---- Perturbation: shuffle_filenames ----
def perturb_shuffle_filenames(paths):
    """Keep dirs, randomly shuffle filenames within each dir."""
    if not paths:
        return {}

    dir_to_files = defaultdict(list)
    for p in paths:
        parts = p.rsplit("/", 1)
        if len(parts) == 2:
            dir_to_files[parts[0]].append(parts[1])
        else:
            dir_to_files[""].append(parts[0])

    mapping = {}
    for d, files in dir_to_files.items():
        original = list(files)
        shuffled = list(files)
        random.shuffle(shuffled)
        for orig, shuf in zip(original, shuffled):
            orig_path = f"{d}/{orig}" if d else orig
            new_path = f"{d}/{shuf}" if d else shuf
            mapping[orig_path] = new_path

    return mapping


def apply_mapping(paths, mapping):
    """Apply path mapping to a list of paths."""
    return [mapping.get(p, p) for p in paths]


def main():
    print("=" * 60)
    print("Repo-held-out split preparation")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_data = load_jsonl(TRAIN_PATH)
    test_data = load_jsonl(TEST_PATH)
    bm25_train = load_jsonl(BM25_TRAIN_PATH)
    bm25_test = load_jsonl(BM25_TEST_PATH)
    print(f"  Train: {len(train_data)} examples")
    print(f"  Test:  {len(test_data)} examples")
    print(f"  BM25 train candidates: {len(bm25_train)} entries")
    print(f"  BM25 test candidates:  {len(bm25_test)} entries")

    # Count test examples per repo
    test_repo_counts = Counter(item["repo"] for item in test_data)
    print(f"\n  Unique repos in test: {len(test_repo_counts)}")

    # Select top-15 repos by test count
    top_repos = [repo for repo, _ in test_repo_counts.most_common(NUM_HELDOUT_REPOS)]
    heldout_set = set(top_repos)

    print(f"\n  Top-{NUM_HELDOUT_REPOS} held-out repos:")
    for repo in top_repos:
        print(f"    {repo}: {test_repo_counts[repo]} test examples")

    heldout_test_count = sum(test_repo_counts[r] for r in top_repos)
    print(f"\n  Total held-out test examples: {heldout_test_count}")

    # Count how many train examples come from held-out repos
    train_repo_counts = Counter(item["repo"] for item in train_data)
    heldout_train_count = sum(train_repo_counts.get(r, 0) for r in top_repos)
    print(f"  Train examples from held-out repos (will be removed): {heldout_train_count}")

    # Filter training data: remove held-out repos
    train_filtered = [item for item in train_data if item["repo"] not in heldout_set]
    print(f"\n  Filtered training set: {len(train_filtered)} (removed {len(train_data) - len(train_filtered)})")

    # Filter test data: keep only held-out repos
    test_heldout = [item for item in test_data if item["repo"] in heldout_set]
    print(f"  Held-out test set: {len(test_heldout)}")

    # Filter BM25 train candidates: remove held-out repos
    bm25_train_filtered = [item for item in bm25_train if item["repo"] not in heldout_set]
    print(f"  Filtered BM25 train: {len(bm25_train_filtered)} (removed {len(bm25_train) - len(bm25_train_filtered)})")

    # Filter BM25 test candidates: keep only held-out repos
    bm25_test_heldout = [item for item in bm25_test if item["repo"] in heldout_set]
    print(f"  Held-out BM25 test: {len(bm25_test_heldout)}")

    # Write outputs
    print(f"\nWriting to {OUT_DIR}/")
    write_jsonl(train_filtered, OUT_DIR / "train_filtered.jsonl")
    write_jsonl(test_heldout, OUT_DIR / "test_heldout.jsonl")
    write_jsonl(bm25_train_filtered, OUT_DIR / "bm25_train_filtered.jsonl")
    write_jsonl(bm25_test_heldout, OUT_DIR / "bm25_test_heldout.jsonl")

    # Create shuffle_filenames perturbation for held-out test
    print("\nCreating shuffle_filenames perturbation for held-out repos...")

    # Collect all paths across held-out test + candidates for consistent mapping
    all_paths = set()
    for item in test_heldout:
        all_paths.update(item.get("changed_py_files", []))
        all_paths.update(item.get("changed_files", []))
    for item in bm25_test_heldout:
        all_paths.update(item.get("candidates", []))

    # Build per-repo mappings
    repo_paths = defaultdict(set)
    for item in test_heldout:
        repo = item["repo"]
        repo_paths[repo].update(item.get("changed_py_files", []))
        repo_paths[repo].update(item.get("changed_files", []))
    for item in bm25_test_heldout:
        repo = item["repo"]
        repo_paths[repo].update(item.get("candidates", []))

    repo_mappings = {}
    for repo, paths in repo_paths.items():
        repo_mappings[repo] = perturb_shuffle_filenames(sorted(paths))

    # Apply to test data
    test_shuffled = []
    for item in test_heldout:
        new_item = dict(item)
        mapping = repo_mappings.get(item["repo"], {})
        new_item["changed_py_files"] = apply_mapping(item.get("changed_py_files", []), mapping)
        new_item["changed_files"] = apply_mapping(item.get("changed_files", []), mapping)
        test_shuffled.append(new_item)

    # Apply to BM25 candidates
    bm25_shuffled = []
    for item in bm25_test_heldout:
        new_item = dict(item)
        mapping = repo_mappings.get(item["repo"], {})
        new_item["candidates"] = apply_mapping(item.get("candidates", []), mapping)
        bm25_shuffled.append(new_item)

    write_jsonl(test_shuffled, OUT_DIR / "test_heldout_shuffle_filenames.jsonl")
    write_jsonl(bm25_shuffled, OUT_DIR / "bm25_test_heldout_shuffle_filenames.jsonl")

    # Write summary stats
    stats = {
        "num_heldout_repos": NUM_HELDOUT_REPOS,
        "heldout_repos": top_repos,
        "heldout_test_examples": heldout_test_count,
        "train_original": len(train_data),
        "train_filtered": len(train_filtered),
        "train_removed": len(train_data) - len(train_filtered),
        "test_heldout": len(test_heldout),
        "test_remaining": len(test_data) - len(test_heldout),
        "bm25_train_filtered": len(bm25_train_filtered),
        "bm25_test_heldout": len(bm25_test_heldout),
    }
    stats_path = OUT_DIR / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats written to {stats_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
