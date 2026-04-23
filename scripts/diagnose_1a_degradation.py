"""
Diagnose why 1A (long-context code-centric) performs WORSE than path-only.

Hypothesis 1 (Noise): Long code introduces misleading tokens that confuse the model
  → Test: Compare 1A vs path-only on examples with long vs short GT files

Hypothesis 2 (Dilution): More code tokens dilute the path signal in attention
  → Test: Compare score margins (gap between top-1 score and next candidate)

Hypothesis 3 (OOD): 2048-token prompts are out-of-distribution for a model
  initialized from a 768-token LoRA
  → Test: Compare degradation on examples where code is short (fits in 768) vs long

Usage:
    python scripts/diagnose_1a_degradation.py \
        --pred_pathonly experiments/rankft_runB_graph/eval_graph_rerank/predictions.jsonl \
        --pred_longctx <1A_eval_dir>/predictions.jsonl \
        --test_data data/grepo_text/grepo_test.jsonl \
        --repo_dir data/repos
"""
import argparse
import json
import os
import re
import numpy as np

random_state = np.random.RandomState(42)

_TOKEN_RE = re.compile(r'[a-zA-Z0-9]+')


def tokenize(text):
    return set(_TOKEN_RE.findall(text.lower()))


def path_tokens(path):
    parts = path.replace('/', ' ').replace('_', ' ').replace('.', ' ')
    return set(_TOKEN_RE.findall(parts.lower()))


def load_predictions(path):
    preds = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            k = (d['repo'], str(d['issue_id']))
            preds[k] = d
    return preds


def get_file_length(repo_dir, repo, filepath):
    """Get file length in lines."""
    full_path = os.path.join(repo_dir, repo, filepath)
    try:
        with open(full_path, 'r', errors='replace') as f:
            return len(f.readlines())
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return -1  # unavailable


def compute_hit(pred_rec):
    gt = set(pred_rec['ground_truth'])
    predicted = pred_rec['predicted']
    if not predicted:
        return 0.0
    return 1.0 if predicted[0] in gt else 0.0


def compute_score_margin(pred_rec):
    """Score gap between rank-1 and rank-2 candidates."""
    scores = pred_rec.get('scores', {})
    if len(scores) < 2:
        return 0.0
    sorted_scores = sorted(scores.values(), reverse=True)
    return sorted_scores[0] - sorted_scores[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_pathonly', required=True)
    parser.add_argument('--pred_longctx', required=True)
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--repo_dir', default='data/repos')
    args = parser.parse_args()

    # Load data
    test = []
    with open(args.test_data) as f:
        for line in f:
            test.append(json.loads(line))
    test_map = {(ex['repo'], str(ex['issue_id'])): ex for ex in test}

    pred_po = load_predictions(args.pred_pathonly)
    pred_lc = load_predictions(args.pred_longctx)

    # Align
    common = sorted(set(pred_po.keys()) & set(pred_lc.keys()))
    print(f"Aligned examples: {len(common)}")

    # Per-example analysis
    results = []
    for k in common:
        ex = test_map.get(k, {})
        po = pred_po[k]
        lc = pred_lc[k]
        repo = k[0]

        hit_po = compute_hit(po)
        hit_lc = compute_hit(lc)

        # GT file lengths
        gt_files = po.get('ground_truth', [])
        gt_lengths = []
        for gf in gt_files:
            length = get_file_length(args.repo_dir, repo, gf)
            if length > 0:
                gt_lengths.append(length)
        avg_gt_length = np.mean(gt_lengths) if gt_lengths else -1

        # Issue-path overlap
        issue_toks = tokenize(ex.get('issue_text', ''))
        max_jaccard = 0
        for gf in gt_files:
            pt = path_tokens(gf)
            if issue_toks | pt:
                j = len(issue_toks & pt) / len(issue_toks | pt)
                max_jaccard = max(max_jaccard, j)

        # Score margins
        margin_po = compute_score_margin(po)
        margin_lc = compute_score_margin(lc)

        # Number of candidates
        n_cands = po.get('num_candidates', 0)

        results.append({
            'key': k,
            'hit_po': hit_po,
            'hit_lc': hit_lc,
            'delta': hit_lc - hit_po,
            'avg_gt_length': avg_gt_length,
            'max_jaccard': max_jaccard,
            'margin_po': margin_po,
            'margin_lc': margin_lc,
            'n_cands': n_cands,
            'repo': repo,
        })

    results_arr = np.array([(r['hit_po'], r['hit_lc'], r['delta'],
                             r['avg_gt_length'], r['max_jaccard'],
                             r['margin_po'], r['margin_lc'])
                            for r in results])

    # === Hypothesis 1: Noise from long files ===
    print("\n" + "="*70)
    print("Hypothesis 1: Long code files introduce noise")
    print("="*70)

    valid = [r for r in results if r['avg_gt_length'] > 0]
    lengths = [r['avg_gt_length'] for r in valid]
    median_len = np.median(lengths)

    short_files = [r for r in valid if r['avg_gt_length'] <= median_len]
    long_files = [r for r in valid if r['avg_gt_length'] > median_len]

    print(f"  Median GT file length: {median_len:.0f} lines")
    print(f"  Short files (≤{median_len:.0f}L): n={len(short_files)}")
    print(f"    Path-only R@1: {np.mean([r['hit_po'] for r in short_files]):.4f}")
    print(f"    Long-ctx R@1:  {np.mean([r['hit_lc'] for r in short_files]):.4f}")
    print(f"    Delta:         {np.mean([r['delta'] for r in short_files]):+.4f}")
    print(f"  Long files (>{median_len:.0f}L): n={len(long_files)}")
    print(f"    Path-only R@1: {np.mean([r['hit_po'] for r in long_files]):.4f}")
    print(f"    Long-ctx R@1:  {np.mean([r['hit_lc'] for r in long_files]):.4f}")
    print(f"    Delta:         {np.mean([r['delta'] for r in long_files]):+.4f}")

    # Quartile breakdown
    q25, q50, q75 = np.percentile(lengths, [25, 50, 75])
    bins = [(0, q25, 'Q1'), (q25, q50, 'Q2'), (q50, q75, 'Q3'), (q75, 1e9, 'Q4')]
    print(f"\n  By GT file length quartile:")
    print(f"  {'Quartile':<12} {'Range':<20} {'N':>5} {'PO R@1':>8} {'LC R@1':>8} {'Delta':>8}")
    print(f"  {'-'*65}")
    for lo, hi, label in bins:
        subset = [r for r in valid if lo <= r['avg_gt_length'] < hi]
        if subset:
            po_r1 = np.mean([r['hit_po'] for r in subset])
            lc_r1 = np.mean([r['hit_lc'] for r in subset])
            delta = lc_r1 - po_r1
            print(f"  {label:<12} [{lo:.0f}, {hi:.0f}){' ':<10} {len(subset):>5} {po_r1:>8.4f} {lc_r1:>8.4f} {delta:>+8.4f}")

    # === Hypothesis 2: Score margin dilution ===
    print("\n" + "="*70)
    print("Hypothesis 2: Code dilutes path signal (score margins)")
    print("="*70)

    margins_po = [r['margin_po'] for r in results if r['margin_po'] > 0]
    margins_lc = [r['margin_lc'] for r in results if r['margin_lc'] > 0]
    print(f"  Path-only score margin: mean={np.mean(margins_po):.4f}, median={np.median(margins_po):.4f}")
    print(f"  Long-ctx  score margin: mean={np.mean(margins_lc):.4f}, median={np.median(margins_lc):.4f}")
    print(f"  Ratio (LC/PO):          {np.mean(margins_lc)/np.mean(margins_po):.3f}")

    # === Hypothesis 3: OOD for short-code examples ===
    print("\n" + "="*70)
    print("Hypothesis 3: Degradation by code length (OOD effect)")
    print("="*70)

    # Split by whether code fits in 768 tokens (~200 lines)
    short_code = [r for r in valid if r['avg_gt_length'] <= 200]
    long_code = [r for r in valid if r['avg_gt_length'] > 200]

    print(f"  Short code (≤200L, fits in 768 ctx): n={len(short_code)}")
    if short_code:
        po_r1 = np.mean([r['hit_po'] for r in short_code])
        lc_r1 = np.mean([r['hit_lc'] for r in short_code])
        print(f"    Path-only: {po_r1:.4f}, Long-ctx: {lc_r1:.4f}, Delta: {lc_r1-po_r1:+.4f}")
    print(f"  Long code (>200L, needs truncation): n={len(long_code)}")
    if long_code:
        po_r1 = np.mean([r['hit_po'] for r in long_code])
        lc_r1 = np.mean([r['hit_lc'] for r in long_code])
        print(f"    Path-only: {po_r1:.4f}, Long-ctx: {lc_r1:.4f}, Delta: {lc_r1-po_r1:+.4f}")

    # === Win/loss by issue-path overlap ===
    print("\n" + "="*70)
    print("Win/loss by issue-path Jaccard quartile")
    print("="*70)

    jaccards = [r['max_jaccard'] for r in results]
    jq25, jq50, jq75 = np.percentile(jaccards, [25, 50, 75])
    jbins = [(0, jq25, 'Q1 low'), (jq25, jq50, 'Q2'), (jq50, jq75, 'Q3'), (jq75, 2, 'Q4 high')]
    print(f"  {'Quartile':<12} {'N':>5} {'Wins':>6} {'Losses':>8} {'Ties':>6} {'Net':>6} {'PO':>8} {'LC':>8}")
    print(f"  {'-'*70}")
    for lo, hi, label in jbins:
        subset = [r for r in results if lo <= r['max_jaccard'] < hi]
        wins = sum(1 for r in subset if r['delta'] > 0)
        losses = sum(1 for r in subset if r['delta'] < 0)
        ties = sum(1 for r in subset if r['delta'] == 0)
        po_r1 = np.mean([r['hit_po'] for r in subset]) if subset else 0
        lc_r1 = np.mean([r['hit_lc'] for r in subset]) if subset else 0
        print(f"  {label:<12} {len(subset):>5} {wins:>6} {losses:>8} {ties:>6} {wins-losses:>+6} {po_r1:>8.4f} {lc_r1:>8.4f}")

    # === Summary ===
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    total_wins = sum(1 for r in results if r['delta'] > 0)
    total_losses = sum(1 for r in results if r['delta'] < 0)
    total_ties = sum(1 for r in results if r['delta'] == 0)
    print(f"  Overall: wins={total_wins}, losses={total_losses}, ties={total_ties}")
    print(f"  Path-only R@1: {np.mean([r['hit_po'] for r in results]):.4f}")
    print(f"  Long-ctx R@1:  {np.mean([r['hit_lc'] for r in results]):.4f}")
    print(f"  Net delta:     {np.mean([r['delta'] for r in results]):+.4f}")


if __name__ == '__main__':
    main()
