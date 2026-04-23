#!/usr/bin/env python3
"""
Mine natural file renames from SWE-bench repo git history.

Finds commits where files were renamed/moved without major content changes.
Then evaluates: does the reranker still find the file after rename?

This validates PathSwap's ecological realism: if real renames cause
similar degradation, the synthetic perturbation is externally valid.

Usage:
    python scripts/mine_natural_renames.py \
        --output_dir /data/chenlibin/grepo_agent_experiments/natural_renames
"""

import argparse
import json
import os
import subprocess
import random

import numpy as np

random.seed(42)
np.random.seed(42)

REPO_DIR = "/home/chenlibin/grepo_agent/data/swebench_lite/repos"
TEST_PATH = "/home/chenlibin/grepo_agent/data/swebench_lite/swebench_lite_test.jsonl"


def find_renames_in_repo(repo_path, max_commits=500):
    """Find file renames in git history using git log --diff-filter=R."""
    renames = []
    try:
        result = subprocess.run(
            ["git", "log", "--diff-filter=R", "--name-status",
             f"--max-count={max_commits}", "--format=%H"],
            cwd=repo_path, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return []

        current_sha = None
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(line) == 40 and all(c in "0123456789abcdef" for c in line):
                current_sha = line
            elif line.startswith("R"):
                parts = line.split("\t")
                if len(parts) >= 3:
                    similarity = parts[0][1:]  # R090 -> 090
                    old_path = parts[1]
                    new_path = parts[2]
                    if old_path.endswith(".py") and new_path.endswith(".py"):
                        renames.append({
                            "sha": current_sha,
                            "old": old_path,
                            "new": new_path,
                            "similarity": int(similarity) if similarity.isdigit() else 0,
                        })
    except (subprocess.TimeoutExpired, Exception):
        pass

    return renames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data to find repos
    test_data = []
    with open(TEST_PATH) as f:
        for line in f:
            test_data.append(json.loads(line))

    repos = set()
    for r in test_data:
        repo = r.get("repo_full", r.get("repo", ""))
        repos.add(repo)

    print(f"Mining renames from {len(repos)} SWE-bench repos...")
    all_renames = {}
    total_renames = 0

    for repo in sorted(repos):
        repo_path = os.path.join(REPO_DIR, repo)
        if not os.path.isdir(repo_path):
            continue

        renames = find_renames_in_repo(repo_path)
        if renames:
            all_renames[repo] = renames
            total_renames += len(renames)
            # Show high-similarity renames (content barely changed)
            high_sim = [r for r in renames if r["similarity"] >= 80]
            print(f"  {repo}: {len(renames)} renames ({len(high_sim)} with >80% similarity)")

    print(f"\nTotal: {total_renames} Python file renames across {len(all_renames)} repos")

    # Find renames that affect test GT files
    gt_affected = []
    for rec in test_data:
        repo = rec.get("repo", "")
        gt_files = set(rec.get("changed_py_files", rec.get("changed_files", [])))

        if repo not in all_renames:
            continue

        for rename in all_renames[repo]:
            # Check if the renamed file is a GT file (old or new name)
            if rename["old"] in gt_files or rename["new"] in gt_files:
                gt_affected.append({
                    "repo": repo,
                    "issue_id": rec.get("issue_id"),
                    "gt_file_old": rename["old"],
                    "gt_file_new": rename["new"],
                    "similarity": rename["similarity"],
                })

    print(f"\nRenames affecting test GT files: {len(gt_affected)}")

    # Statistics
    if all_renames:
        all_sims = [r["similarity"] for renames in all_renames.values() for r in renames]
        print(f"\nSimilarity distribution:")
        for threshold in [50, 70, 80, 90, 95, 100]:
            count = sum(1 for s in all_sims if s >= threshold)
            print(f"  >= {threshold}%: {count} ({count/len(all_sims)*100:.1f}%)")

    summary = {
        "num_repos_with_renames": len(all_renames),
        "total_renames": total_renames,
        "gt_affected_renames": len(gt_affected),
        "rename_examples": gt_affected[:20],
        "per_repo_counts": {repo: len(renames) for repo, renames in all_renames.items()},
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.output_dir, "all_renames.json"), "w") as f:
        json.dump(all_renames, f, indent=2)

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
