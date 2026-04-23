"""
Final consolidated analysis: compare directory expansion vs co-change,
and analyze combined potential.

Also fix the name_filter strategy -- be more strict.
"""

import json
import os
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Set, List

REPOS_DIR = "/home/chenlibin/grepo_agent/data/repos"
PRED_BASE = "/home/chenlibin/grepo_agent/experiments/exp1_sft_only/eval_filetree/predictions.jsonl"
PRED_COCHANGE = "/home/chenlibin/grepo_agent/experiments/exp1_sft_only/eval_filetree_expanded/predictions.jsonl"


def get_dir(path: str) -> str:
    return os.path.dirname(path)


def get_parent_dir(path: str) -> str:
    return os.path.dirname(os.path.dirname(path))


def load_preds(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def compute_metrics(gt_set, pred_list):
    m = {}
    for k in [1, 3, 5, 10, 20]:
        topk = set(pred_list[:k])
        hits = len(gt_set & topk)
        m[k] = (hits / len(gt_set)) * 100 if gt_set else 0
    return m


def is_test_pair(a: str, b: str) -> bool:
    """Strict test-file pairing: foo.py <-> test_foo.py or foo_test.py."""
    ba = os.path.basename(a).replace('.py', '')
    bb = os.path.basename(b).replace('.py', '')
    if ba.startswith('test_') and ba[5:] == bb:
        return True
    if bb.startswith('test_') and bb[5:] == ba:
        return True
    if ba.endswith('_test') and ba[:-5] == bb:
        return True
    if bb.endswith('_test') and bb[:-5] == ba:
        return True
    return False


def main():
    base_preds = load_preds(PRED_BASE)
    cochange_preds = load_preds(PRED_COCHANGE)

    n = len(base_preds)
    print(f"Total instances: {n}")

    # ── Overlap analysis: co-change vs directory ─────────────────────────
    print("\n" + "=" * 72)
    print("ANALYSIS: Where do co-change and directory expansions overlap?")
    print("=" * 72)

    # Track which GT files each method recovers (oracle)
    cochange_recovers = 0  # GT files recovered by co-change (actual, not oracle)
    dir_recovers_oracle = 0  # GT files recoverable by same-dir (oracle)
    both_recover = 0
    cochange_only = 0
    dir_only = 0
    neither = 0

    for base_p, cc_p in zip(base_preds, cochange_preds):
        assert base_p['issue_id'] == cc_p['issue_id']
        gt_set = set(base_p['ground_truth'])
        base_pred_set = set(base_p['predicted'])
        cc_pred_set = set(cc_p['predicted'])
        base_pred_dirs = set(get_dir(f) for f in base_pred_set)

        already_hit = gt_set & base_pred_set
        missed = gt_set - base_pred_set

        for mf in missed:
            recovered_by_cc = mf in cc_pred_set
            recoverable_by_dir = get_dir(mf) in base_pred_dirs

            if recovered_by_cc and recoverable_by_dir:
                both_recover += 1
            elif recovered_by_cc:
                cochange_only += 1
            elif recoverable_by_dir:
                dir_only += 1
            else:
                neither += 1

    total_missed = cochange_only + dir_only + both_recover + neither
    print(f"\nOf {total_missed} missed GT files (from base predictions):")
    print(f"  Recovered by co-change AND reachable by same-dir: {both_recover} "
          f"({100*both_recover/total_missed:.1f}%)")
    print(f"  Recovered by co-change ONLY:                      {cochange_only} "
          f"({100*cochange_only/total_missed:.1f}%)")
    print(f"  Reachable by same-dir ONLY (not by co-change):    {dir_only} "
          f"({100*dir_only/total_missed:.1f}%)")
    print(f"  Neither:                                          {neither} "
          f"({100*neither/total_missed:.1f}%)")
    print(f"\n  => Directory expansion can recover {dir_only} GT files BEYOND what co-change gets")

    # ── Practical test-variant expansion ─────────────────────────────────
    print("\n" + "=" * 72)
    print("TEST-VARIANT EXPANSION (strict: foo.py <-> test_foo.py)")
    print("=" * 72)

    metrics_before = defaultdict(list)
    metrics_after = defaultdict(list)
    tp, fp = 0, 0

    for p in base_preds:
        gt_set = set(p['ground_truth'])
        pred_list = list(p['predicted'])
        pred_set = set(pred_list)

        # For each predicted file, also add its test variant (or implementation)
        expansion = []
        seen = set(pred_set)

        for pf in pred_list:
            pf_dir = get_dir(pf)
            pf_base = os.path.basename(pf).replace('.py', '')

            # Generate candidate test variants
            candidates = []
            if pf_base.startswith('test_'):
                # This is a test file, predict the implementation
                impl_name = pf_base[5:] + '.py'
                candidates.append(os.path.join(pf_dir, impl_name))
            elif pf_base.endswith('_test'):
                impl_name = pf_base[:-5] + '.py'
                candidates.append(os.path.join(pf_dir, impl_name))
            else:
                # This is implementation, predict the test file
                candidates.append(os.path.join(pf_dir, f'test_{pf_base}.py'))
                candidates.append(os.path.join(pf_dir, f'{pf_base}_test.py'))

            for c in candidates:
                if c not in seen:
                    expansion.append(c)
                    seen.add(c)

        new_pred = pred_list + expansion

        for f in expansion:
            if f in gt_set:
                tp += 1
            else:
                fp += 1

        for k in [1, 3, 5, 10, 20]:
            before = compute_metrics(gt_set, pred_list)[k]
            after = compute_metrics(gt_set, new_pred)[k]
            metrics_before[k].append(before)
            metrics_after[k].append(after)

    prec = 100 * tp / max(tp + fp, 1)
    print(f"\n  Test-variant expansion: TP={tp}, FP={fp}, precision={prec:.1f}%")
    for k in [1, 3, 5, 10, 20]:
        before = sum(metrics_before[k]) / n
        after = sum(metrics_after[k]) / n
        delta = after - before
        print(f"    Hit@{k:>2}: {before:.2f}% -> {after:.2f}%  ({'+' if delta >= 0 else ''}{delta:.2f}%)")

    # ── Same analysis on co-change-expanded predictions ──────────────────
    print("\n" + "=" * 72)
    print("TEST-VARIANT EXPANSION ON TOP OF CO-CHANGE EXPANDED PREDICTIONS")
    print("=" * 72)

    metrics_before2 = defaultdict(list)
    metrics_after2 = defaultdict(list)
    tp2, fp2 = 0, 0

    for p in cochange_preds:
        gt_set = set(p['ground_truth'])
        pred_list = list(p['predicted'])
        pred_set = set(pred_list)

        expansion = []
        seen = set(pred_set)

        for pf in pred_list:
            pf_dir = get_dir(pf)
            pf_base = os.path.basename(pf).replace('.py', '')

            candidates = []
            if pf_base.startswith('test_'):
                candidates.append(os.path.join(pf_dir, pf_base[5:] + '.py'))
            elif pf_base.endswith('_test'):
                candidates.append(os.path.join(pf_dir, pf_base[:-5] + '.py'))
            else:
                candidates.append(os.path.join(pf_dir, f'test_{pf_base}.py'))
                candidates.append(os.path.join(pf_dir, f'{pf_base}_test.py'))

            for c in candidates:
                if c not in seen:
                    expansion.append(c)
                    seen.add(c)

        new_pred = pred_list + expansion

        for f in expansion:
            if f in gt_set:
                tp2 += 1
            else:
                fp2 += 1

        for k in [1, 3, 5, 10, 20]:
            before = compute_metrics(gt_set, pred_list)[k]
            after = compute_metrics(gt_set, new_pred)[k]
            metrics_before2[k].append(before)
            metrics_after2[k].append(after)

    prec2 = 100 * tp2 / max(tp2 + fp2, 1)
    print(f"\n  Test-variant expansion: TP={tp2}, FP={fp2}, precision={prec2:.1f}%")
    for k in [1, 3, 5, 10, 20]:
        before = sum(metrics_before2[k]) / n
        after = sum(metrics_after2[k]) / n
        delta = after - before
        print(f"    Hit@{k:>2}: {before:.2f}% -> {after:.2f}%  ({'+' if delta >= 0 else ''}{delta:.2f}%)")

    # ── Summary recommendation ───────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY: DIRECTORY EXPANSION POTENTIAL")
    print("=" * 72)
    print("""
Key findings:
1. 22.4% of missed GT files are in the SAME directory as a predicted file.
   Another 22.8% are in a SIBLING directory. Total reachable: 45.2%.

2. However, most of these files are in LARGE directories (69% in dirs with 21+ files).
   Naive same-dir expansion adds avg 50 files with only 1.8% precision.

3. Co-change expansion and directory expansion have PARTIAL overlap:
   - Some GT files are recoverable by both methods
   - But directory expansion can recover additional files co-change misses

4. Test-variant expansion (foo.py <-> test_foo.py) is very cheap and precise
   but recovers only a small number of files.

Recommendation:
- Test-variant expansion: always apply (cheap, high precision, small gain)
- Small-dir expansion (<=10 files): apply selectively (low noise, modest gain)
- Same-dir for large dirs: only use as a RANKING signal, not expansion
  (e.g., boost score of co-change candidates that are also in same dir)
- Combine with co-change: use directory proximity as a secondary signal
  to re-rank co-change expansion candidates
""")


if __name__ == '__main__':
    main()
