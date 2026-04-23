"""
Analyze directory-based co-location patterns in bug localization predictions.

Questions answered:
1. What fraction of GT files share a directory with at least one predicted file?
2. When we miss a GT file, is it often in the same dir as a predicted file?
3. Would same-directory or sibling-file expansion help?
4. What's the distribution of directory distances between predicted and GT files?
"""

import json
import os
from collections import defaultdict, Counter
from typing import List, Set, Tuple, Dict

# ── helpers ──────────────────────────────────────────────────────────────────

def get_dir(path: str) -> str:
    return os.path.dirname(path)


def get_parent_dir(path: str) -> str:
    """Parent of the directory containing the file."""
    return os.path.dirname(os.path.dirname(path))


def dir_distance(a: str, b: str) -> int:
    """
    Rough 'directory distance':
    0 = same directory
    1 = sibling directories (same parent)
    2 = cousin directories (same grandparent)
    -1 = unrelated (different root subtrees)
    """
    da, db = get_dir(a), get_dir(b)
    if da == db:
        return 0
    if get_parent_dir(a) == get_parent_dir(b):
        return 1
    # check grandparent
    gpa = os.path.dirname(get_parent_dir(a))
    gpb = os.path.dirname(get_parent_dir(b))
    if gpa and gpb and gpa == gpb:
        return 2
    return -1


def basename_similarity(a: str, b: str) -> str:
    """Check common basename patterns: same name, test variant, etc."""
    ba = os.path.basename(a).replace('.py', '')
    bb = os.path.basename(b).replace('.py', '')
    if ba == bb:
        return 'same_name'
    # test variant: foo.py <-> foo_test.py or test_foo.py
    if ba.startswith('test_') and ba[5:] == bb:
        return 'test_variant'
    if bb.startswith('test_') and bb[5:] == ba:
        return 'test_variant'
    if ba.endswith('_test') and ba[:-5] == bb:
        return 'test_variant'
    if bb.endswith('_test') and bb[:-5] == ba:
        return 'test_variant'
    return 'different'


# ── main analysis ────────────────────────────────────────────────────────────

def load_predictions(path: str) -> List[dict]:
    preds = []
    with open(path) as f:
        for line in f:
            preds.append(json.loads(line))
    return preds


def analyze(preds: List[dict]):
    n = len(preds)

    # Counters
    gt_total = 0
    gt_hit = 0  # already predicted correctly
    gt_missed = 0

    # For missed GT files
    missed_same_dir = 0       # missed GT in same dir as a predicted file
    missed_sibling_dir = 0    # missed GT in sibling dir
    missed_cousin_dir = 0     # missed GT in cousin dir
    missed_no_relation = 0
    missed_test_variant = 0   # missed GT is a test variant of predicted
    missed_same_dir_test = 0  # missed GT in same dir AND is test variant

    # For GT files in general
    gt_shares_dir_with_pred = 0  # GT file shares dir with >= 1 predicted
    gt_shares_dir_with_other_gt = 0  # GT file shares dir with another GT

    # Distance distribution for missed files
    dist_counter = Counter()
    basename_counter = Counter()

    # Per-instance: would expansion help?
    instances_helped_by_same_dir = 0
    instances_helped_by_sibling = 0
    instances_helped_by_test_variant = 0

    # Track potential new hits
    new_hits_same_dir = 0
    new_hits_sibling = 0

    # How many predicted files per instance
    pred_count_dist = Counter()

    for p in preds:
        gt_set = set(p['ground_truth'])
        pred_set = set(p['predicted'])
        pred_dirs = set(get_dir(f) for f in pred_set)
        pred_parent_dirs = set(get_parent_dir(f) for f in pred_set)
        gt_dirs = set(get_dir(f) for f in gt_set)

        pred_count_dist[len(pred_set)] += 1

        hit = gt_set & pred_set
        missed = gt_set - pred_set

        gt_total += len(gt_set)
        gt_hit += len(hit)
        gt_missed += len(missed)

        instance_helped_same_dir = False
        instance_helped_sibling = False
        instance_helped_test = False

        for gf in gt_set:
            gf_dir = get_dir(gf)
            # Does this GT share a dir with any predicted file?
            if gf_dir in pred_dirs:
                gt_shares_dir_with_pred += 1
            # Does it share a dir with another GT file?
            other_gt_dirs = set(get_dir(f) for f in gt_set if f != gf)
            if gf_dir in other_gt_dirs:
                gt_shares_dir_with_other_gt += 1

        for mf in missed:
            mf_dir = get_dir(mf)
            mf_parent = get_parent_dir(mf)

            # Check directory relationship to any predicted file
            in_same_dir = mf_dir in pred_dirs
            in_sibling = (not in_same_dir) and (mf_parent in pred_parent_dirs)

            # Check basename similarity
            best_basename = 'different'
            for pf in pred_set:
                bs = basename_similarity(mf, pf)
                if bs == 'same_name':
                    best_basename = 'same_name'
                    break
                elif bs == 'test_variant':
                    best_basename = 'test_variant'

            basename_counter[best_basename] += 1

            if in_same_dir:
                missed_same_dir += 1
                instance_helped_same_dir = True
                new_hits_same_dir += 1
                if best_basename == 'test_variant':
                    missed_same_dir_test += 1
            elif in_sibling:
                missed_sibling_dir += 1
                instance_helped_sibling = True
                new_hits_sibling += 1
            else:
                # Compute closest distance
                min_dist = -1
                for pf in pred_set:
                    d = dir_distance(mf, pf)
                    if d >= 0 and (min_dist < 0 or d < min_dist):
                        min_dist = d
                if min_dist == 2:
                    missed_cousin_dir += 1
                else:
                    missed_no_relation += 1

            if best_basename == 'test_variant':
                missed_test_variant += 1
                instance_helped_test = True

            # Closest distance
            for pf in pred_set:
                d = dir_distance(mf, pf)
                dist_counter[d] += 1

        if instance_helped_same_dir:
            instances_helped_by_same_dir += 1
        if instance_helped_sibling:
            instances_helped_by_sibling += 1
        if instance_helped_test:
            instances_helped_by_test_variant += 1

    # ── Report ───────────────────────────────────────────────────────────────
    print("=" * 72)
    print("DIRECTORY-BASED EXPANSION ANALYSIS")
    print("=" * 72)
    print(f"Total instances: {n}")
    print(f"Total GT files: {gt_total}  (avg {gt_total/n:.2f} per instance)")
    print(f"Already hit: {gt_hit}  ({100*gt_hit/gt_total:.1f}%)")
    print(f"Missed: {gt_missed}  ({100*gt_missed/gt_total:.1f}%)")
    print()

    print("─" * 72)
    print("Q1: What fraction of GT files share a directory with >= 1 predicted file?")
    print(f"  GT shares dir with predicted: {gt_shares_dir_with_pred}/{gt_total} "
          f"({100*gt_shares_dir_with_pred/gt_total:.1f}%)")
    print(f"  GT shares dir with other GT:  {gt_shares_dir_with_other_gt}/{gt_total} "
          f"({100*gt_shares_dir_with_other_gt/gt_total:.1f}%)")
    print()

    print("─" * 72)
    print("Q2: When we miss a GT file, what is its directory relationship to predictions?")
    print(f"  Same directory as a predicted file:    {missed_same_dir}/{gt_missed} "
          f"({100*missed_same_dir/max(gt_missed,1):.1f}%)")
    print(f"  Sibling directory (same parent):       {missed_sibling_dir}/{gt_missed} "
          f"({100*missed_sibling_dir/max(gt_missed,1):.1f}%)")
    print(f"  Cousin directory (same grandparent):   {missed_cousin_dir}/{gt_missed} "
          f"({100*missed_cousin_dir/max(gt_missed,1):.1f}%)")
    print(f"  No close relation:                     {missed_no_relation}/{gt_missed} "
          f"({100*missed_no_relation/max(gt_missed,1):.1f}%)")
    print()
    reachable = missed_same_dir + missed_sibling_dir
    print(f"  => Reachable by same-dir + sibling:    {reachable}/{gt_missed} "
          f"({100*reachable/max(gt_missed,1):.1f}%)")
    print()

    print("─" * 72)
    print("Q2b: Basename similarity for missed GT files")
    for k, v in basename_counter.most_common():
        print(f"  {k}: {v}/{gt_missed} ({100*v/max(gt_missed,1):.1f}%)")
    print(f"  Test variants in same dir: {missed_same_dir_test}")
    print()

    print("─" * 72)
    print("Q3: How many instances would be helped by directory expansion?")
    instances_currently_zero = sum(1 for p in preds if p['metrics'].get('hit@10', 0) == 0)
    instances_currently_partial = sum(1 for p in preds
                                     if 0 < p['metrics'].get('hit@10', 0) < 100)
    print(f"  Instances with hit@10=0 (total miss): {instances_currently_zero}/{n} "
          f"({100*instances_currently_zero/n:.1f}%)")
    print(f"  Instances with 0 < hit@10 < 100:      {instances_currently_partial}/{n} "
          f"({100*instances_currently_partial/n:.1f}%)")
    print()
    print(f"  Same-dir expansion would help:         {instances_helped_by_same_dir}/{n} "
          f"({100*instances_helped_by_same_dir/n:.1f}%)")
    print(f"  Sibling-dir expansion would help:      {instances_helped_by_sibling}/{n} "
          f"({100*instances_helped_by_sibling/n:.1f}%)")
    print(f"  Test-variant expansion would help:     {instances_helped_by_test_variant}/{n} "
          f"({100*instances_helped_by_test_variant/n:.1f}%)")
    print()

    print("─" * 72)
    print("Q4: Simulated impact on metrics (same-dir oracle expansion)")
    # Simulate: for each instance, add all GT files reachable by same-dir
    # This is an UPPER BOUND (oracle) - in practice we'd add all files in dir
    simulate_expansion(preds, mode='same_dir')
    simulate_expansion(preds, mode='sibling')
    simulate_expansion(preds, mode='test_variant')
    print()

    print("─" * 72)
    print("Q5: Realistic expansion (add ALL .py files in same dirs as predicted)")
    simulate_realistic_expansion(preds)


def simulate_expansion(preds: List[dict], mode: str = 'same_dir'):
    """Oracle simulation: if a missed GT is reachable, count it as hit."""
    metrics_before = defaultdict(list)
    metrics_after = defaultdict(list)

    for p in preds:
        gt_set = set(p['ground_truth'])
        pred_list = list(p['predicted'])
        pred_set = set(pred_list)
        pred_dirs = set(get_dir(f) for f in pred_set)
        pred_parent_dirs = set(get_parent_dir(f) for f in pred_set)

        # Find reachable missed files
        expansion = []
        for gf in gt_set:
            if gf in pred_set:
                continue
            gf_dir = get_dir(gf)

            if mode == 'same_dir' and gf_dir in pred_dirs:
                expansion.append(gf)
            elif mode == 'sibling' and (gf_dir in pred_dirs or get_parent_dir(gf) in pred_parent_dirs):
                expansion.append(gf)
            elif mode == 'test_variant':
                for pf in pred_set:
                    if basename_similarity(gf, pf) == 'test_variant' and get_dir(gf) == get_dir(pf):
                        expansion.append(gf)
                        break

        new_pred = pred_list + expansion

        for k in [1, 3, 5, 10, 20]:
            before = len(gt_set & set(pred_list[:k])) / len(gt_set) * 100 if gt_set else 0
            after = len(gt_set & set(new_pred[:k])) / len(gt_set) * 100 if gt_set else 0
            metrics_before[k].append(before)
            # For expansion, count original top-k PLUS expansion
            topk_expanded = set(pred_list[:k]) | set(expansion)
            after_expanded = len(gt_set & topk_expanded) / len(gt_set) * 100 if gt_set else 0
            metrics_after[k].append(after_expanded)

    n = len(preds)
    print(f"\n  [{mode}] Oracle expansion (upper bound):")
    for k in [1, 3, 5, 10, 20]:
        before = sum(metrics_before[k]) / n
        after = sum(metrics_after[k]) / n
        delta = after - before
        print(f"    Hit@{k:>2}: {before:.2f}% -> {after:.2f}%  (+{delta:.2f}%)")


def simulate_realistic_expansion(preds: List[dict]):
    """
    Realistic simulation: for each predicted file, add other .py files from
    the same directory that exist in the GT data (as a proxy for real files).

    We use the GT files across all instances as a rough file inventory per repo.
    """
    # Build file inventory per repo from all GT + predicted
    repo_files: Dict[str, Set[str]] = defaultdict(set)
    for p in preds:
        repo = p['repo']
        for f in p['ground_truth']:
            repo_files[repo].add(f)
        for f in p['predicted']:
            repo_files[repo].add(f)

    # Build dir -> files index
    repo_dir_files: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for repo, files in repo_files.items():
        for f in files:
            repo_dir_files[repo][get_dir(f)].add(f)

    # Simulate
    metrics_before = defaultdict(list)
    metrics_after = defaultdict(list)
    expansion_sizes = []

    for p in preds:
        repo = p['repo']
        gt_set = set(p['ground_truth'])
        pred_list = list(p['predicted'])
        pred_set = set(pred_list)

        # Expand: add all known files in same dirs as predicted
        expansion = []
        for pf in pred_list:
            d = get_dir(pf)
            for neighbor in repo_dir_files[repo].get(d, set()):
                if neighbor not in pred_set and neighbor not in expansion:
                    expansion.append(neighbor)

        expansion_sizes.append(len(expansion))
        new_pred = pred_list + expansion

        for k in [1, 5, 10, 20]:
            before = len(gt_set & set(pred_list[:k])) / len(gt_set) * 100 if gt_set else 0
            after = len(gt_set & set(new_pred[:k])) / len(gt_set) * 100 if gt_set else 0
            metrics_before[k].append(before)
            metrics_after[k].append(after)

    n = len(preds)
    avg_exp = sum(expansion_sizes) / n
    print(f"  Avg expansion size: {avg_exp:.1f} files")
    print(f"  (Using known file inventory as proxy - real expansion would need repo listing)")
    for k in [1, 5, 10, 20]:
        before = sum(metrics_before[k]) / n
        after = sum(metrics_after[k]) / n
        delta = after - before
        print(f"    Hit@{k:>2}: {before:.2f}% -> {after:.2f}%  (+{delta:.2f}%)")


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze directory patterns for expansion')
    parser.add_argument('--predictions', default='/home/chenlibin/grepo_agent/experiments/exp1_sft_only/eval_filetree/predictions.jsonl')
    args = parser.parse_args()

    print(f"Loading predictions from: {args.predictions}")
    preds = load_predictions(args.predictions)
    analyze(preds)


if __name__ == '__main__':
    main()
