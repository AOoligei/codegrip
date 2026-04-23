"""
Deeper analysis: per-directory-size breakdown and smart filtering strategies.
"""

import json
import os
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Set

REPOS_DIR = "/home/chenlibin/grepo_agent/data/repos"
PRED_PATH = "/home/chenlibin/grepo_agent/experiments/exp1_sft_only/eval_filetree/predictions.jsonl"


def get_repo_py_files(repo_name: str) -> Set[str]:
    repo_path = Path(REPOS_DIR) / repo_name
    if not repo_path.exists():
        return set()
    return {str(p.relative_to(repo_path)) for p in repo_path.rglob("*.py")}


def main():
    preds = []
    with open(PRED_PATH) as f:
        for line in f:
            preds.append(json.loads(line))

    # Build repo indices (only for repos we have)
    repo_dir_index: Dict[str, Dict[str, Set[str]]] = {}
    repos_available = set()
    for repo in set(p['repo'] for p in preds):
        files = get_repo_py_files(repo)
        if files:
            repos_available.add(repo)
            dir_idx = defaultdict(set)
            for f in files:
                dir_idx[os.path.dirname(f)].add(f)
            repo_dir_index[repo] = dir_idx

    preds = [p for p in preds if p['repo'] in repos_available]
    n = len(preds)
    print(f"Instances with repo available: {n}")

    # ── Analysis 1: Directory size distribution for missed GT ────────────
    print("\n" + "=" * 72)
    print("DIRECTORY SIZE OF MISSED GT FILES (how many .py files in their dir?)")
    print("=" * 72)

    dir_size_of_missed_in_pred_dir = []  # missed GT file is in same dir as a pred
    dir_size_of_missed_not_in_pred_dir = []

    for p in preds:
        repo = p['repo']
        gt_set = set(p['ground_truth'])
        pred_set = set(p['predicted'])
        pred_dirs = set(os.path.dirname(f) for f in pred_set)
        dir_idx = repo_dir_index[repo]
        missed = gt_set - pred_set

        for mf in missed:
            d = os.path.dirname(mf)
            dir_size = len(dir_idx.get(d, set()))
            if d in pred_dirs:
                dir_size_of_missed_in_pred_dir.append(dir_size)
            else:
                dir_size_of_missed_not_in_pred_dir.append(dir_size)

    print(f"\nMissed GT in same dir as predicted ({len(dir_size_of_missed_in_pred_dir)} files):")
    if dir_size_of_missed_in_pred_dir:
        sizes = sorted(dir_size_of_missed_in_pred_dir)
        print(f"  Dir size: min={min(sizes)}, median={sizes[len(sizes)//2]}, "
              f"mean={sum(sizes)/len(sizes):.1f}, max={max(sizes)}")
        buckets = Counter()
        for s in sizes:
            if s <= 3:
                buckets['1-3'] += 1
            elif s <= 5:
                buckets['4-5'] += 1
            elif s <= 10:
                buckets['6-10'] += 1
            elif s <= 20:
                buckets['11-20'] += 1
            else:
                buckets['21+'] += 1
        for b in ['1-3', '4-5', '6-10', '11-20', '21+']:
            print(f"  Dir size {b:>5}: {buckets[b]:>4} files "
                  f"({100*buckets[b]/len(sizes):.1f}%)")

    # ── Analysis 2: Smart strategies ─────────────────────────────────────
    print("\n" + "=" * 72)
    print("SMART EXPANSION STRATEGIES")
    print("=" * 72)

    strategies = {
        'same_dir_small_only': {  # only expand in small dirs (<= 10 files)
            'desc': 'Same dir, only if dir has <= 10 .py files',
            'max_dir_size': 10,
            'sibling': False,
            'name_filter': False,
        },
        'same_dir_name_filter': {  # only expand to files with similar names
            'desc': 'Same dir, name similarity filter',
            'max_dir_size': 999,
            'sibling': False,
            'name_filter': True,
        },
        'same_dir_small_plus_name': {
            'desc': 'Same dir <= 10 files OR name-similar in any dir',
            'max_dir_size': 10,
            'sibling': False,
            'name_filter': True,  # name filter as fallback for large dirs
        },
        'small_dir_plus_sibling_small': {
            'desc': 'Same dir <= 10 files + sibling dirs <= 5 files',
            'max_dir_size': 10,
            'sibling': True,
            'sibling_max_size': 5,
            'name_filter': False,
        },
    }

    for strat_name, cfg in strategies.items():
        metrics_before = defaultdict(list)
        metrics_after = defaultdict(list)
        expansion_sizes = []
        tp, fp = 0, 0

        for p in preds:
            repo = p['repo']
            gt_set = set(p['ground_truth'])
            pred_list = list(p['predicted'])
            pred_set = set(pred_list)
            dir_idx = repo_dir_index[repo]

            expansion = []
            seen = set(pred_set)

            for pf in pred_list:
                pf_base = os.path.basename(pf).replace('.py', '')
                d = os.path.dirname(pf)
                dir_files = dir_idx.get(d, set())
                dir_size = len(dir_files)

                # Same-dir expansion
                if dir_size <= cfg['max_dir_size']:
                    for neighbor in dir_files:
                        if neighbor not in seen:
                            expansion.append(neighbor)
                            seen.add(neighbor)
                elif cfg.get('name_filter'):
                    # Only add name-similar files from large dirs
                    for neighbor in dir_files:
                        if neighbor not in seen:
                            nb = os.path.basename(neighbor).replace('.py', '')
                            if _name_similar(pf_base, nb):
                                expansion.append(neighbor)
                                seen.add(neighbor)

                # Sibling dir expansion
                if cfg.get('sibling'):
                    parent = os.path.dirname(d)
                    for sib_dir, sib_files in dir_idx.items():
                        if os.path.dirname(sib_dir) == parent and sib_dir != d:
                            if len(sib_files) <= cfg.get('sibling_max_size', 5):
                                for neighbor in sib_files:
                                    if neighbor not in seen:
                                        expansion.append(neighbor)
                                        seen.add(neighbor)

            expansion_sizes.append(len(expansion))
            new_pred = pred_list + expansion

            for f in expansion:
                if f in gt_set:
                    tp += 1
                else:
                    fp += 1

            for k in [1, 3, 5, 10, 20]:
                before = len(gt_set & set(pred_list[:k])) / len(gt_set) * 100 if gt_set else 0
                after = len(gt_set & set(new_pred[:k])) / len(gt_set) * 100 if gt_set else 0
                metrics_before[k].append(before)
                metrics_after[k].append(after)

        total = tp + fp
        avg_exp = sum(expansion_sizes) / n
        med_exp = sorted(expansion_sizes)[n // 2]
        prec = 100 * tp / max(total, 1)

        print(f"\n  Strategy: {strat_name}")
        print(f"  {cfg['desc']}")
        print(f"  Expansion: avg={avg_exp:.1f}, median={med_exp}, "
              f"TP={tp}, FP={fp}, precision={prec:.2f}%")
        for k in [1, 3, 5, 10, 20]:
            before = sum(metrics_before[k]) / n
            after = sum(metrics_after[k]) / n
            delta = after - before
            print(f"    Hit@{k:>2}: {before:.2f}% -> {after:.2f}%  (+{delta:.2f}%)")


def _name_similar(base_a: str, base_b: str) -> bool:
    """Check if two Python basenames (without .py) are related."""
    # test variants
    if base_a.startswith('test_') and base_a[5:] == base_b:
        return True
    if base_b.startswith('test_') and base_b[5:] == base_a:
        return True
    if base_a.endswith('_test') and base_a[:-5] == base_b:
        return True
    if base_b.endswith('_test') and base_b[:-5] == base_a:
        return True
    # conftest
    if base_a == 'conftest' or base_b == 'conftest':
        return True
    # Same prefix (e.g., parser.py and parser_utils.py)
    if len(base_a) > 3 and len(base_b) > 3:
        if base_b.startswith(base_a) or base_a.startswith(base_b):
            return True
    return False


if __name__ == '__main__':
    main()
