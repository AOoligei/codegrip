#!/usr/bin/env python3
"""
Path perturbation for SWE-bench Lite evaluation.

Creates perturbed test and candidate files for 3 conditions:
  1. shuffle_filenames - keep dirs, shuffle filenames across dirs
  2. shuffle_dirs      - keep filenames, shuffle directory paths
  3. flatten_dirs      - remove all dirs, files at root level

Perturbation is applied CONSISTENTLY: same mapping for both test.jsonl
(ground truth) and candidates.jsonl within each (repo, issue_id) pair.

Output:
  data/swebench_lite/swebench_perturb_{condition}_test.jsonl
  data/swebench_lite/swebench_perturb_{condition}_candidates.jsonl

Usage:
    python scripts/swebench_path_perturbation.py
"""

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_PATH = BASE_DIR / "data" / "swebench_lite" / "swebench_lite_test.jsonl"
CAND_PATH = BASE_DIR / "data" / "rankft" / "swebench_bm25_tricked_top500.jsonl"
OUT_DIR = BASE_DIR / "data" / "swebench_lite"

CONDITIONS = ["shuffle_filenames", "shuffle_dirs", "flatten_dirs"]


# ============================================================
# Perturbation functions (per-example, deterministic via rng)
# ============================================================

def perturb_shuffle_filenames(paths: List[str], rng: random.Random) -> Dict[str, str]:
    """Keep directory structure, shuffle filenames across directories."""
    if not paths:
        return {}

    # Group by directory
    dir_to_files = defaultdict(list)
    for p in paths:
        parts = p.rsplit("/", 1)
        if len(parts) == 2:
            dir_to_files[parts[0]].append(parts[1])
        else:
            dir_to_files[""].append(parts[0])

    # Shuffle filenames within each directory
    dir_to_shuffled = {}
    for d in sorted(dir_to_files.keys()):  # sorted for determinism
        fnames = dir_to_files[d]
        shuffled = list(fnames)
        rng.shuffle(shuffled)
        dir_to_shuffled[d] = dict(zip(fnames, shuffled))

    mapping = {}
    for p in paths:
        parts = p.rsplit("/", 1)
        if len(parts) == 2:
            d, f = parts
            new_f = dir_to_shuffled[d][f]
            mapping[p] = f"{d}/{new_f}"
        else:
            new_f = dir_to_shuffled[""][parts[0]]
            mapping[p] = new_f
    return mapping


def perturb_shuffle_dirs(paths: List[str], rng: random.Random) -> Dict[str, str]:
    """Keep filenames, shuffle directory assignments."""
    if not paths:
        return {}

    # Collect unique dirs
    unique_dirs = set()
    for p in paths:
        parts = p.rsplit("/", 1)
        if len(parts) == 2:
            unique_dirs.add(parts[0])

    if not unique_dirs:
        return {p: p for p in paths}

    unique_dirs = sorted(unique_dirs)  # sorted for determinism
    dir_perm = list(unique_dirs)
    rng.shuffle(dir_perm)
    dir_map = dict(zip(unique_dirs, dir_perm))

    mapping = {}
    for p in paths:
        parts = p.rsplit("/", 1)
        if len(parts) == 2:
            old_dir, fname = parts
            new_dir = dir_map[old_dir]
            mapping[p] = f"{new_dir}/{fname}"
        else:
            mapping[p] = p
    return mapping


def perturb_flatten_dirs(paths: List[str], rng: random.Random) -> Dict[str, str]:
    """Remove all directory structure. Disambiguate collisions with index."""
    if not paths:
        return {}

    # Count filename collisions
    fname_count = defaultdict(int)
    for p in paths:
        fname = p.rsplit("/", 1)[-1]
        fname_count[fname] += 1

    fname_seen = defaultdict(int)
    mapping = {}
    for p in paths:
        fname = p.rsplit("/", 1)[-1]
        if fname_count[fname] > 1:
            idx = fname_seen[fname]
            fname_seen[fname] += 1
            stem = Path(fname).stem
            ext = Path(fname).suffix or ".py"
            mapping[p] = f"{stem}_{idx}{ext}"
        else:
            mapping[p] = fname
    return mapping


PERTURB_FN = {
    "shuffle_filenames": perturb_shuffle_filenames,
    "shuffle_dirs": perturb_shuffle_dirs,
    "flatten_dirs": perturb_flatten_dirs,
}


# ============================================================
# Main processing
# ============================================================

def load_data():
    """Load SWE-bench test data and BM25 candidates."""
    test_data = []
    with open(TEST_PATH) as f:
        for line in f:
            test_data.append(json.loads(line))

    cand_data = {}
    with open(CAND_PATH) as f:
        for line in f:
            item = json.loads(line)
            key = (item["repo"], item["issue_id"])
            cand_data[key] = item

    return test_data, cand_data


def create_perturbed_data(condition: str, test_data: List[dict], cand_data: dict):
    """Create perturbed test + candidate files for one condition."""
    out_test = OUT_DIR / f"swebench_perturb_{condition}_test.jsonl"
    out_cand = OUT_DIR / f"swebench_perturb_{condition}_candidates.jsonl"

    perturbed_test = []
    perturbed_cand = []
    n_skipped = 0
    n_changed = 0
    n_total = 0
    n_gt_changed = 0
    n_gt_total = 0

    # Use a master rng seeded at 42; each example gets a child seed
    # based on index for reproducibility regardless of skips
    master_rng = random.Random(42)

    for idx, item in enumerate(test_data):
        repo = item["repo"]
        issue_id = item["issue_id"]
        key = (repo, issue_id)

        if key not in cand_data:
            n_skipped += 1
            continue

        cand_item = cand_data[key]
        candidates = cand_item.get("bm25_candidates", [])
        gt_files = item.get("changed_py_files", [])

        # Per-example deterministic RNG
        example_rng = random.Random(42 + idx)

        # Build mapping from ALL unique paths in this example
        all_paths = sorted(set(candidates + gt_files))
        fn = PERTURB_FN[condition]
        path_map = fn(all_paths, example_rng)

        # Stats
        for p in all_paths:
            n_total += 1
            if path_map.get(p, p) != p:
                n_changed += 1
        for p in gt_files:
            n_gt_total += 1
            if path_map.get(p, p) != p:
                n_gt_changed += 1

        # Perturbed test item
        p_item = dict(item)
        p_item["changed_py_files"] = [path_map.get(f, f) for f in gt_files]
        if "changed_files" in p_item:
            p_item["changed_files"] = [path_map.get(f, f) for f in item["changed_files"]]
        perturbed_test.append(p_item)

        # Perturbed candidate item
        p_cand = dict(cand_item)
        p_cand["bm25_candidates"] = [path_map.get(c, c) for c in candidates]
        if "ground_truth" in cand_item:
            p_cand["ground_truth"] = [path_map.get(g, g) for g in cand_item["ground_truth"]]
        perturbed_cand.append(p_cand)

    # Write output
    with open(out_test, "w") as f:
        for item in perturbed_test:
            f.write(json.dumps(item) + "\n")

    with open(out_cand, "w") as f:
        for item in perturbed_cand:
            f.write(json.dumps(item) + "\n")

    pct = (n_changed / n_total * 100) if n_total else 0
    gt_pct = (n_gt_changed / n_gt_total * 100) if n_gt_total else 0
    print(f"\n=== {condition} ===")
    print(f"  Examples: {len(perturbed_test)} (skipped: {n_skipped})")
    print(f"  All paths changed: {n_changed}/{n_total} ({pct:.1f}%)")
    print(f"  GT paths changed:  {n_gt_changed}/{n_gt_total} ({gt_pct:.1f}%)")
    print(f"  -> {out_test}")
    print(f"  -> {out_cand}")


def verify_consistency(condition: str):
    """Verify that GT paths in test file match ground_truth in candidate file."""
    test_path = OUT_DIR / f"swebench_perturb_{condition}_test.jsonl"
    cand_path = OUT_DIR / f"swebench_perturb_{condition}_candidates.jsonl"

    test_data = []
    with open(test_path) as f:
        for line in f:
            test_data.append(json.loads(line))

    cand_data = {}
    with open(cand_path) as f:
        for line in f:
            item = json.loads(line)
            key = (item["repo"], item["issue_id"])
            cand_data[key] = item

    n_ok = 0
    n_fail = 0
    for item in test_data:
        key = (item["repo"], item["issue_id"])
        cand_item = cand_data[key]
        gt_test = set(item["changed_py_files"])
        gt_cand = set(cand_item.get("ground_truth", []))
        if gt_test == gt_cand:
            n_ok += 1
        else:
            n_fail += 1
            print(f"  MISMATCH {key}: test={gt_test} vs cand={gt_cand}")

    # Also verify GT files appear in candidates
    n_gt_in_cand = 0
    n_gt_not_in_cand = 0
    for item in test_data:
        key = (item["repo"], item["issue_id"])
        cand_item = cand_data[key]
        cand_set = set(cand_item["bm25_candidates"])
        for gt in item["changed_py_files"]:
            if gt in cand_set:
                n_gt_in_cand += 1
            else:
                n_gt_not_in_cand += 1

    print(f"  Consistency check [{condition}]: {n_ok} ok, {n_fail} mismatch")
    print(f"  GT in candidates: {n_gt_in_cand}, GT not in candidates: {n_gt_not_in_cand}")


def show_samples(condition: str, test_data: List[dict], cand_data: dict, n: int = 2):
    """Show sample perturbations for sanity check."""
    print(f"\n  Samples for '{condition}':")
    count = 0
    for idx, item in enumerate(test_data):
        key = (item["repo"], item["issue_id"])
        if key not in cand_data:
            continue
        cand_item = cand_data[key]
        candidates = cand_item.get("bm25_candidates", [])
        gt_files = item.get("changed_py_files", [])
        all_paths = sorted(set(candidates + gt_files))

        example_rng = random.Random(42 + idx)
        fn = PERTURB_FN[condition]
        path_map = fn(all_paths, example_rng)

        changed = [(old, new) for old, new in path_map.items() if old != new]
        if changed:
            print(f"    [{item['repo']}] {item['issue_id']}:")
            # Show GT mapping
            for gt in gt_files:
                print(f"      GT: {gt} -> {path_map.get(gt, gt)}")
            # Show a few candidate mappings
            for old, new in changed[:3]:
                if old not in gt_files:
                    print(f"      cand: {old} -> {new}")
            count += 1
            if count >= n:
                break


def main():
    print("SWE-bench Path Perturbation Data Generator")
    print(f"  Test:       {TEST_PATH}")
    print(f"  Candidates: {CAND_PATH}")
    print(f"  Output dir: {OUT_DIR}")

    test_data, cand_data = load_data()
    print(f"  Loaded {len(test_data)} test examples, {len(cand_data)} candidate entries")

    for cond in CONDITIONS:
        create_perturbed_data(cond, test_data, cand_data)
        show_samples(cond, test_data, cand_data)

    print("\n\n=== Verification ===")
    for cond in CONDITIONS:
        verify_consistency(cond)

    print("\nDone. Files created:")
    for cond in CONDITIONS:
        print(f"  data/swebench_lite/swebench_perturb_{cond}_test.jsonl")
        print(f"  data/swebench_lite/swebench_perturb_{cond}_candidates.jsonl")


if __name__ == "__main__":
    main()
