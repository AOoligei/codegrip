#!/usr/bin/env python3
"""
Find natural path-challenge cases in GREPO test set.

Three approaches:
1. Cross-version renames: files renamed/moved in git history of repo
2. Issue-path mismatch: issue text mentions paths different from actual changed_py_files
3. Same-stem competitors: multiple candidates share the same filename stem

Output: stats + JSONL challenge subsets compatible with eval pipeline.
"""

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import PurePosixPath

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PATH = os.path.join(ROOT, "data/grepo_text/grepo_test.jsonl")
CANDIDATES_PATH = os.path.join(ROOT, "data/rankft/merged_bm25_exp6_candidates.jsonl")
REPOS_DIR = os.path.join(ROOT, "data/repos")
OUTPUT_DIR = os.path.join(ROOT, "data/natural_path_challenge")


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def make_key(item):
    return (item["repo"], item["issue_id"])


# ─── Approach 1: Cross-version renames ───────────────────────────────────────

def get_repo_renames(repo_name):
    """Get all file renames from git history of a repo."""
    repo_path = os.path.join(REPOS_DIR, repo_name)
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        return {}

    try:
        result = subprocess.run(
            ["git", "log", "--all", "--diff-filter=R", "--name-status",
             "--format=", "-M50%"],
            cwd=repo_path, capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired:
        print(f"  [WARN] git log timed out for {repo_name}")
        return {}

    # Parse rename entries: R<pct>\told_path\tnew_path
    renames = {}  # old_path -> set of new_paths (and reverse)
    for line in result.stdout.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0].startswith("R"):
            old_path, new_path = parts[1], parts[2]
            if old_path.endswith(".py") or new_path.endswith(".py"):
                renames.setdefault(old_path, set()).add(new_path)
                renames.setdefault(new_path, set()).add(old_path)
    return renames


def find_rename_cases(test_data):
    """Find test examples whose changed_py_files have been renamed in git history."""
    # Group test examples by repo
    repo_examples = defaultdict(list)
    for item in test_data:
        repo_examples[item["repo"]].append(item)

    results = []
    for repo_name, examples in sorted(repo_examples.items()):
        renames = get_repo_renames(repo_name)
        if not renames:
            continue

        for item in examples:
            matched_renames = {}
            for f in item["changed_py_files"]:
                if f in renames:
                    matched_renames[f] = list(renames[f])
            if matched_renames:
                results.append({
                    **item,
                    "rename_info": matched_renames,
                    "challenge_type": "cross_version_rename",
                })

    return results


# ─── Approach 2: Issue-path mismatch ─────────────────────────────────────────

# Regex to find file-path-like mentions in issue text
PATH_RE = re.compile(
    r'(?:^|[\s`\'\"(,])('           # preceded by whitespace/quote/paren
    r'(?:[\w./-]+/)?'               # optional directory prefix
    r'[\w.-]+\.py'                  # filename.py
    r'(?::?\d+)?'                   # optional :lineno
    r')(?:[\s`\'\")\],.:;]|$)',     # followed by delimiter
    re.MULTILINE
)


def extract_paths_from_text(text):
    """Extract .py file path mentions from issue text."""
    matches = PATH_RE.findall(text)
    # Clean up
    paths = set()
    for m in matches:
        m = m.strip("`'\"(),: ")
        # Remove trailing :lineno
        m = re.sub(r':\d+$', '', m)
        if m.endswith('.py') and len(m) > 4:
            paths.add(m)
    return paths


def find_issue_mismatch_cases(test_data):
    """Find cases where issue mentions paths not in changed_py_files."""
    results = []
    for item in test_data:
        mentioned_paths = extract_paths_from_text(item["issue_text"])
        if not mentioned_paths:
            continue

        changed = set(item["changed_py_files"])
        # Check for mismatches: mentioned path doesn't match any changed file
        mismatched = set()
        for mp in mentioned_paths:
            # Skip bare filenames without directory (likely noise like "run-tests.py")
            if "/" not in mp:
                continue
            # Skip common non-localization mentions
            if any(x in mp for x in ["setup.py", "conftest.py", "__init__.py"]):
                continue
            # Check exact match
            if mp in changed:
                continue
            # Check suffix match (issue might mention partial path)
            suffix_match = any(c.endswith(mp) or mp.endswith(c) for c in changed)
            if suffix_match:
                continue
            # Check stem match (same filename, different directory)
            mp_stem = PurePosixPath(mp).name
            stem_match = any(PurePosixPath(c).name == mp_stem for c in changed)
            if not stem_match:
                mismatched.add(mp)
            else:
                # Same stem but different directory -- this IS a mismatch
                mismatched.add(mp)

        if mismatched:
            results.append({
                **item,
                "mentioned_paths": sorted(mentioned_paths),
                "mismatched_paths": sorted(mismatched),
                "challenge_type": "issue_path_mismatch",
            })

    return results


# ─── Approach 3: Same-stem competitors ───────────────────────────────────────

TRIVIAL_STEMS = {"__init__.py", "conftest.py", "setup.py", "__main__.py"}


def find_same_stem_cases(test_data, candidates_map):
    """Find cases where GT file's stem appears multiple times in candidate pool."""
    results = []
    results_nontrivial = []
    for item in test_data:
        key = make_key(item)
        cands = candidates_map.get(key, [])
        if not cands:
            continue

        gt_files = set(item["changed_py_files"])

        # Group candidates by stem
        stem_to_cands = defaultdict(list)
        for c in cands:
            stem = PurePosixPath(c).name
            stem_to_cands[stem].append(c)

        # Check if any GT file has same-stem competitors
        gt_stem_conflicts = {}
        gt_stem_conflicts_nontrivial = {}
        for gt_f in gt_files:
            gt_stem = PurePosixPath(gt_f).name
            same_stem = stem_to_cands.get(gt_stem, [])
            # Must have at least one non-GT file with same stem
            competitors = [s for s in same_stem if s not in gt_files]
            if competitors:
                gt_stem_conflicts[gt_f] = competitors
                if gt_stem not in TRIVIAL_STEMS:
                    gt_stem_conflicts_nontrivial[gt_f] = competitors

        if gt_stem_conflicts:
            results.append({
                **item,
                "stem_conflicts": {k: v for k, v in gt_stem_conflicts.items()},
                "num_stem_competitors": sum(len(v) for v in gt_stem_conflicts.values()),
                "challenge_type": "same_stem_competitor",
            })
        if gt_stem_conflicts_nontrivial:
            results_nontrivial.append({
                **item,
                "stem_conflicts": {k: v for k, v in gt_stem_conflicts_nontrivial.items()},
                "num_stem_competitors": sum(len(v) for v in gt_stem_conflicts_nontrivial.values()),
                "challenge_type": "same_stem_competitor",
            })

    return results, results_nontrivial


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading test data...")
    test_data = load_jsonl(TEST_PATH)
    print(f"  {len(test_data)} test examples")

    print("Loading candidate pools...")
    cand_data = load_jsonl(CANDIDATES_PATH)
    candidates_map = {make_key(c): c["candidates"] for c in cand_data}
    # Filter test_data to those with candidates
    test_with_cands = [t for t in test_data if make_key(t) in candidates_map]
    print(f"  {len(test_with_cands)} test examples have candidate pools")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Approach 1: Cross-version renames ──
    print("\n" + "=" * 70)
    print("APPROACH 1: Cross-version renames")
    print("=" * 70)
    rename_cases = find_rename_cases(test_data)
    print(f"\nFound {len(rename_cases)} examples with renamed files")

    if rename_cases:
        save_jsonl(rename_cases, os.path.join(OUTPUT_DIR, "rename_cases.jsonl"))
        print("\nTop examples:")
        for case in rename_cases[:5]:
            print(f"  [{case['repo']}] issue #{case['issue_id']}")
            for f, aliases in case["rename_info"].items():
                print(f"    {f} <-> {aliases[:3]}")

    # ── Approach 2: Issue-path mismatch ──
    print("\n" + "=" * 70)
    print("APPROACH 2: Issue-path mismatch")
    print("=" * 70)
    mismatch_cases = find_issue_mismatch_cases(test_data)
    print(f"\nFound {len(mismatch_cases)} examples with issue-path mismatch")

    if mismatch_cases:
        save_jsonl(mismatch_cases, os.path.join(OUTPUT_DIR, "issue_mismatch_cases.jsonl"))
        print("\nTop examples:")
        for case in mismatch_cases[:5]:
            print(f"  [{case['repo']}] issue #{case['issue_id']}")
            print(f"    Mentioned: {case['mentioned_paths'][:3]}")
            print(f"    Mismatched: {case['mismatched_paths'][:3]}")
            print(f"    Changed: {case['changed_py_files'][:3]}")

    # ── Approach 3: Same-stem competitors ──
    print("\n" + "=" * 70)
    print("APPROACH 3: Same-stem competitors in candidate pool")
    print("=" * 70)
    stem_cases_all, stem_cases_nontrivial = find_same_stem_cases(test_with_cands, candidates_map)
    stem_cases = stem_cases_nontrivial  # Use nontrivial for the benchmark
    print(f"\nFound {len(stem_cases_all)} examples with same-stem competitors (all)")
    print(f"Found {len(stem_cases)} examples with same-stem competitors (excl __init__.py etc)")

    if stem_cases:
        save_jsonl(stem_cases, os.path.join(OUTPUT_DIR, "same_stem_cases.jsonl"))
        save_jsonl(stem_cases_all, os.path.join(OUTPUT_DIR, "same_stem_cases_all.jsonl"))
        # Distribution of competitor counts
        comp_counts = [c["num_stem_competitors"] for c in stem_cases]
        print(f"  Mean competitors per example: {sum(comp_counts)/len(comp_counts):.1f}")
        print(f"  Max competitors: {max(comp_counts)}")
        # Most common conflicting stems
        stem_freq = defaultdict(int)
        for case in stem_cases:
            for gt_f in case["stem_conflicts"]:
                stem_freq[PurePosixPath(gt_f).name] += 1
        print("\n  Most common conflicting stems (non-trivial):")
        for stem, cnt in sorted(stem_freq.items(), key=lambda x: -x[1])[:15]:
            print(f"    {stem}: {cnt} examples")

        print("\nTop examples:")
        for case in sorted(stem_cases, key=lambda x: -x["num_stem_competitors"])[:5]:
            print(f"  [{case['repo']}] issue #{case['issue_id']} ({case['num_stem_competitors']} competitors)")
            for gt_f, comps in list(case["stem_conflicts"].items())[:2]:
                print(f"    GT: {gt_f}")
                print(f"    Competitors: {comps[:5]}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    # Union of all challenge cases
    all_keys = set()
    for cases, name in [(rename_cases, "rename"), (mismatch_cases, "mismatch"), (stem_cases, "stem")]:
        keys = {make_key(c) for c in cases}
        all_keys |= keys
        print(f"  {name}: {len(cases)} examples")

    print(f"  Union: {len(all_keys)} unique examples")
    print(f"  Total test set: {len(test_data)} examples")
    print(f"  Challenge fraction: {len(all_keys)/len(test_data)*100:.1f}%")

    # Overlap matrix
    rename_keys = {make_key(c) for c in rename_cases}
    mismatch_keys = {make_key(c) for c in mismatch_cases}
    stem_keys = {make_key(c) for c in stem_cases}
    print(f"\n  Overlap:")
    print(f"    rename & mismatch: {len(rename_keys & mismatch_keys)}")
    print(f"    rename & stem:     {len(rename_keys & stem_keys)}")
    print(f"    mismatch & stem:   {len(mismatch_keys & stem_keys)}")
    print(f"    all three:         {len(rename_keys & mismatch_keys & stem_keys)}")

    # Per-repo distribution
    print("\n  Per-repo distribution:")
    repo_counts = defaultdict(int)
    for k in all_keys:
        repo_counts[k[0]] += 1
    for repo, cnt in sorted(repo_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {repo}: {cnt}")

    # Save combined challenge set (only examples that have candidate pools)
    combined = []
    challenge_keys = all_keys
    for item in test_with_cands:
        key = make_key(item)
        if key in challenge_keys:
            challenge_types = []
            if key in rename_keys:
                challenge_types.append("rename")
            if key in mismatch_keys:
                challenge_types.append("issue_mismatch")
            if key in stem_keys:
                challenge_types.append("same_stem")
            combined.append({**item, "challenge_types": challenge_types})

    save_jsonl(combined, os.path.join(OUTPUT_DIR, "combined_challenge.jsonl"))
    print(f"\n  Combined challenge set (with candidate pools): {len(combined)} examples")
    print(f"  Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
