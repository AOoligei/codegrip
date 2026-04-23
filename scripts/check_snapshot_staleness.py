"""Check how stale repo snapshots are vs bug-time code.

Sample 50 test examples and verify GT files exist and analyze their content.
"""
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

DATA_ROOT = Path("/home/chenlibin/grepo_agent/data")
TEST_FILE = DATA_ROOT / "grepo_text" / "grepo_test.jsonl"
REPOS_DIR = DATA_ROOT / "repos"

random.seed(42)

# Load all test examples
with open(TEST_FILE) as f:
    all_examples = [json.loads(line) for line in f]

# Sample 50
sample = random.sample(all_examples, 50)

# Stats
total_files = 0
existing_files = 0
missing_files = []
file_lengths = []
short_files = 0  # <=30 lines
first30_content_types = Counter()  # what's in first 30 lines

for ex in sample:
    repo = ex["repo"]
    for fpath in ex["changed_py_files"]:
        total_files += 1
        full_path = REPOS_DIR / repo / fpath

        if not full_path.exists():
            missing_files.append(f"{repo}/{fpath}")
            continue

        existing_files += 1
        try:
            lines = full_path.read_text(errors="replace").splitlines()
        except Exception as e:
            print(f"  ERROR reading {full_path}: {e}")
            continue

        file_lengths.append(len(lines))
        if len(lines) <= 30:
            short_files += 1

        # Analyze first 30 lines
        first30 = lines[:30]
        has_import = any(re.match(r"^\s*(import |from )", l) for l in first30)
        has_class = any(re.match(r"^\s*class ", l) for l in first30)
        has_func = any(re.match(r"^\s*def ", l) for l in first30)
        has_docstring = any('"""' in l or "'''" in l for l in first30)

        if has_import:
            first30_content_types["imports"] += 1
        if has_class:
            first30_content_types["class_def"] += 1
        if has_func:
            first30_content_types["func_def"] += 1
        if has_docstring:
            first30_content_types["docstring"] += 1

print("=" * 60)
print("SNAPSHOT STALENESS CHECK")
print("=" * 60)
print(f"\nSampled examples: {len(sample)}")
print(f"Total GT files across sample: {total_files}")
print(f"Existing in snapshot: {existing_files} ({100*existing_files/total_files:.1f}%)")
print(f"Missing from snapshot: {total_files - existing_files} ({100*(total_files-existing_files)/total_files:.1f}%)")

if missing_files:
    print(f"\nMissing files ({len(missing_files)}):")
    for mf in missing_files[:20]:
        print(f"  - {mf}")
    if len(missing_files) > 20:
        print(f"  ... and {len(missing_files)-20} more")

if file_lengths:
    import numpy as np
    lengths = np.array(file_lengths)
    print(f"\n--- File Length Stats (existing files) ---")
    print(f"Files with <=30 lines: {short_files} ({100*short_files/len(file_lengths):.1f}%) -- fully captured by code_max_lines=30")
    print(f"Mean length: {lengths.mean():.0f} lines")
    print(f"Median length: {np.median(lengths):.0f} lines")
    print(f"Min: {lengths.min()}, Max: {lengths.max()}")
    print(f"25th percentile: {np.percentile(lengths, 25):.0f}")
    print(f"75th percentile: {np.percentile(lengths, 75):.0f}")
    print(f"Files >100 lines: {(lengths > 100).sum()} ({100*(lengths > 100).sum()/len(lengths):.1f}%)")
    print(f"Files >500 lines: {(lengths > 500).sum()} ({100*(lengths > 500).sum()/len(lengths):.1f}%)")

    print(f"\n--- First 30 Lines Content (of {existing_files} existing files) ---")
    for kind, count in first30_content_types.most_common():
        print(f"  {kind}: {count} ({100*count/existing_files:.1f}%)")

# Per-repo breakdown of missing files
if missing_files:
    print(f"\n--- Missing Files by Repo ---")
    repo_missing = Counter()
    for mf in missing_files:
        repo_missing[mf.split("/")[0]] += 1
    for repo, cnt in repo_missing.most_common():
        print(f"  {repo}: {cnt} files missing")

# Check a few missing files: do similar-named files exist?
if missing_files:
    print(f"\n--- Investigating Missing Files (first 5) ---")
    for mf in missing_files[:5]:
        repo, *rest = mf.split("/")
        fname = rest[-1] if rest else ""
        repo_dir = REPOS_DIR / repo
        if repo_dir.exists():
            # Search for the filename anywhere in the repo
            matches = list(repo_dir.rglob(fname))
            if matches:
                print(f"  {mf} -> FOUND at different path(s):")
                for m in matches[:3]:
                    print(f"    {m.relative_to(REPOS_DIR)}")
            else:
                print(f"  {mf} -> NOT FOUND anywhere in {repo}/")
        else:
            print(f"  {mf} -> repo directory {repo}/ does not exist!")
