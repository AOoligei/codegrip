#!/usr/bin/env python3
"""
Generate filename-shuffled perturbation data for BeetleBox Java evaluation.

Applies the same shuffle_filenames perturbation as controlled_path_perturbation.py
but on BeetleBox Java data (*.java files instead of *.py).

Outputs:
  /data/chenlibin/beetlebox/perturb_shuffle_filenames/java_test.jsonl
  /data/chenlibin/beetlebox/perturb_shuffle_filenames/java_bm25_top500.jsonl

Usage:
    python scripts/perturb_beetlebox.py
"""

import json
import os
import random
from collections import defaultdict
from typing import Dict, List

random.seed(42)

INPUT_TEST = "/data/chenlibin/beetlebox/java_test.jsonl"
INPUT_BM25 = "/data/chenlibin/beetlebox/java_bm25_top500.jsonl"
OUTPUT_DIR = "/data/chenlibin/beetlebox/perturb_shuffle_filenames"


def shuffle_filenames(paths: List[str]) -> Dict[str, str]:
    """Keep directory structure, randomly shuffle filenames within each directory."""
    # Group files by directory
    dir_to_files: Dict[str, List[str]] = defaultdict(list)
    for p in paths:
        parts = p.rsplit("/", 1)
        if len(parts) == 2:
            dir_to_files[parts[0]].append(parts[1])
        else:
            dir_to_files[""].append(parts[0])

    # Shuffle filenames within each directory
    mapping = {}
    for dir_path, filenames in dir_to_files.items():
        shuffled = filenames.copy()
        random.shuffle(shuffled)
        for orig, new in zip(filenames, shuffled):
            orig_full = f"{dir_path}/{orig}" if dir_path else orig
            new_full = f"{dir_path}/{new}" if dir_path else new
            mapping[orig_full] = new_full

    return mapping


def apply_mapping(paths: List[str], mapping: Dict[str, str]) -> List[str]:
    """Apply path mapping, keeping unmapped paths unchanged."""
    return [mapping.get(p, p) for p in paths]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all candidate paths to build global mapping per repo×issue
    test_records = []
    with open(INPUT_TEST) as f:
        for line in f:
            test_records.append(json.loads(line))

    bm25_records = []
    with open(INPUT_BM25) as f:
        for line in f:
            bm25_records.append(json.loads(line))

    assert len(test_records) == len(bm25_records), \
        f"Mismatch: {len(test_records)} test vs {len(bm25_records)} bm25"

    out_test = os.path.join(OUTPUT_DIR, "java_test.jsonl")
    out_bm25 = os.path.join(OUTPUT_DIR, "java_bm25_top500.jsonl")

    with open(out_test, "w") as ft, open(out_bm25, "w") as fb:
        for test_rec, bm25_rec in zip(test_records, bm25_records):
            # Verify records match
            assert test_rec["repo"] == bm25_rec["repo"] and \
                   test_rec["issue_id"] == bm25_rec["issue_id"], \
                   f"Mismatch: {test_rec['repo']}#{test_rec['issue_id']} vs {bm25_rec['repo']}#{bm25_rec['issue_id']}"

            # Build mapping from all paths: candidates + GT + ground_truth
            all_paths = list(bm25_rec["bm25_candidates"])
            for gf in test_rec.get("changed_files", []):
                if gf not in all_paths:
                    all_paths.append(gf)
            for gf in bm25_rec.get("ground_truth", []):
                if gf not in all_paths:
                    all_paths.append(gf)

            mapping = shuffle_filenames(all_paths)

            # Apply to test record
            new_test = dict(test_rec)
            new_test["changed_files"] = apply_mapping(
                test_rec["changed_files"], mapping)
            if "changed_py_files" in test_rec:
                new_test["changed_py_files"] = apply_mapping(
                    test_rec["changed_py_files"], mapping)
            ft.write(json.dumps(new_test) + "\n")

            # Apply to bm25 record
            new_bm25 = dict(bm25_rec)
            new_bm25["bm25_candidates"] = apply_mapping(
                bm25_rec["bm25_candidates"], mapping)
            new_bm25["ground_truth"] = apply_mapping(
                bm25_rec["ground_truth"], mapping)
            fb.write(json.dumps(new_bm25) + "\n")

    print(f"Written {len(test_records)} examples to {OUTPUT_DIR}")
    print(f"  {out_test}")
    print(f"  {out_bm25}")


if __name__ == "__main__":
    main()
