"""
Supplementary analysis: use real repo file listings to estimate realistic
same-directory expansion noise and precision.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

REPOS_DIR = "/home/chenlibin/grepo_agent/data/repos"


def get_repo_py_files(repo_name: str) -> Set[str]:
    """Get all .py files in a repo, with paths relative to repo root."""
    repo_path = Path(REPOS_DIR) / repo_name
    if not repo_path.exists():
        return set()
    files = set()
    for p in repo_path.rglob("*.py"):
        rel = str(p.relative_to(repo_path))
        files.add(rel)
    return files


def main():
    pred_path = "/home/chenlibin/grepo_agent/experiments/exp1_sft_only/eval_filetree/predictions.jsonl"

    preds = []
    with open(pred_path) as f:
        for line in f:
            preds.append(json.loads(line))

    # Cache repo file listings
    repo_files_cache: Dict[str, Set[str]] = {}
    repo_dir_index: Dict[str, Dict[str, Set[str]]] = {}

    repos_needed = set(p['repo'] for p in preds)
    print(f"Loading file listings for {len(repos_needed)} repos...")

    for repo in repos_needed:
        files = get_repo_py_files(repo)
        repo_files_cache[repo] = files
        dir_idx = defaultdict(set)
        for f in files:
            dir_idx[os.path.dirname(f)].add(f)
        repo_dir_index[repo] = dir_idx
        if files:
            print(f"  {repo}: {len(files)} .py files across {len(dir_idx)} dirs")
        else:
            print(f"  {repo}: NOT FOUND in repos dir")

    # Analysis with real file listings
    print("\n" + "=" * 72)
    print("REALISTIC SAME-DIRECTORY EXPANSION (using actual repo files)")
    print("=" * 72)

    metrics_before = defaultdict(list)
    metrics_after_samedir = defaultdict(list)
    metrics_after_samedir_limited = defaultdict(list)

    expansion_sizes = []
    expansion_sizes_limited = []
    precision_gains = []

    n_with_repo = 0
    n_helped = 0

    for p in preds:
        repo = p['repo']
        gt_set = set(p['ground_truth'])
        pred_list = list(p['predicted'])
        pred_set = set(pred_list)
        dir_idx = repo_dir_index.get(repo, {})

        if not dir_idx:
            # No repo listing available, skip
            continue
        n_with_repo += 1

        # Expansion: all .py in same dirs as predicted files
        expansion_full = []
        for pf in pred_list:
            d = os.path.dirname(pf)
            for neighbor in dir_idx.get(d, set()):
                if neighbor not in pred_set and neighbor not in expansion_full:
                    expansion_full.append(neighbor)

        # Limited expansion: max 5 per directory, sorted by name similarity
        expansion_limited = []
        for pf in pred_list:
            d = os.path.dirname(pf)
            candidates = [f for f in dir_idx.get(d, set())
                         if f not in pred_set and f not in expansion_limited]
            # Prioritize: test variants first, then alphabetically close
            pf_base = os.path.basename(pf).replace('.py', '')
            def sort_key(f):
                fb = os.path.basename(f).replace('.py', '')
                is_test = ('test' in fb and pf_base in fb) or ('test' in pf_base and fb in pf_base)
                name_overlap = len(set(pf_base) & set(fb)) / max(len(set(pf_base) | set(fb)), 1)
                return (-int(is_test), -name_overlap, fb)
            candidates.sort(key=sort_key)
            expansion_limited.extend(candidates[:5])

        expansion_sizes.append(len(expansion_full))
        expansion_sizes_limited.append(len(expansion_limited))

        new_pred_full = pred_list + expansion_full
        new_pred_limited = pred_list + expansion_limited

        helped = False
        for k in [1, 3, 5, 10, 20]:
            before = len(gt_set & set(pred_list[:k])) / len(gt_set) * 100 if gt_set else 0
            after_full = len(gt_set & set(new_pred_full[:k])) / len(gt_set) * 100 if gt_set else 0
            after_lim = len(gt_set & set(new_pred_limited[:k])) / len(gt_set) * 100 if gt_set else 0
            metrics_before[k].append(before)
            metrics_after_samedir[k].append(after_full)
            metrics_after_samedir_limited[k].append(after_lim)
            if after_full > before:
                helped = True

        if helped:
            n_helped += 1

    n = n_with_repo
    print(f"\nInstances with repo available: {n}")
    print(f"Instances helped by same-dir expansion: {n_helped}/{n} ({100*n_helped/n:.1f}%)")

    avg_full = sum(expansion_sizes) / n
    avg_lim = sum(expansion_sizes_limited) / n
    print(f"\nExpansion size (full same-dir):  avg={avg_full:.1f}, "
          f"median={sorted(expansion_sizes)[n//2]}, "
          f"max={max(expansion_sizes)}")
    print(f"Expansion size (limited 5/dir): avg={avg_lim:.1f}, "
          f"median={sorted(expansion_sizes_limited)[n//2]}, "
          f"max={max(expansion_sizes_limited)}")

    print(f"\nFull same-directory expansion:")
    for k in [1, 3, 5, 10, 20]:
        before = sum(metrics_before[k]) / n
        after = sum(metrics_after_samedir[k]) / n
        delta = after - before
        print(f"  Hit@{k:>2}: {before:.2f}% -> {after:.2f}%  (+{delta:.2f}%)")

    print(f"\nLimited same-directory expansion (max 5 per dir):")
    for k in [1, 3, 5, 10, 20]:
        before = sum(metrics_before[k]) / n
        after = sum(metrics_after_samedir_limited[k]) / n
        delta = after - before
        print(f"  Hit@{k:>2}: {before:.2f}% -> {after:.2f}%  (+{delta:.2f}%)")

    # Noise analysis: what fraction of expansion files are actual GT?
    print(f"\n" + "─" * 72)
    print("EXPANSION PRECISION (what fraction of added files are actual GT?)")
    tp_full, fp_full = 0, 0
    tp_lim, fp_lim = 0, 0
    for p in preds:
        repo = p['repo']
        gt_set = set(p['ground_truth'])
        pred_set = set(p['predicted'])
        dir_idx = repo_dir_index.get(repo, {})
        if not dir_idx:
            continue

        for pf in p['predicted']:
            d = os.path.dirname(pf)
            for neighbor in dir_idx.get(d, set()):
                if neighbor not in pred_set:
                    if neighbor in gt_set:
                        tp_full += 1
                    else:
                        fp_full += 1

    total_full = tp_full + fp_full
    print(f"  Full expansion: {tp_full} true positives, {fp_full} false positives")
    print(f"  Precision: {100*tp_full/max(total_full,1):.2f}%")
    print(f"  (i.e., {100*fp_full/max(total_full,1):.1f}% of expanded files are noise)")


if __name__ == '__main__':
    main()
