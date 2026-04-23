#!/usr/bin/env python3
"""
Statistical analysis for CodeGRIP paper.
Computes bootstrap CIs, effect sizes, per-repo analysis, and McNemar's test.

Outputs LaTeX-ready statistics to experiments/analysis/statistics.tex
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
PRED_DIRS = {
    "graph_expanded":  BASE / "experiments/rankft_runB_graph/eval_merged_rerank",
    "bm25_expanded":   BASE / "experiments/rankft_runA_bm25only/eval_merged_rerank",
    "graph_bm25pool":  BASE / "experiments/rankft_runB_graph/eval_bm25pool",
    "bm25_bm25pool":   BASE / "experiments/rankft_runA_bm25only/eval_bm25pool",
}
OUT_DIR = BASE / "experiments" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TEX = OUT_DIR / "statistics.tex"

B = 10_000  # bootstrap iterations


# ---------------------------------------------------------------------------
# Load per-example data
# ---------------------------------------------------------------------------
def load_predictions(pred_dir: Path):
    """Return list of dicts with per-example data, preserving order."""
    path = pred_dir / "predictions.jsonl"
    examples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            examples.append(d)
    return examples


def extract_binary_hit1(examples):
    """hit@1 > 0 means at least one GT file found in top-1."""
    return np.array([1.0 if ex["metrics"]["hit@1"] > 0 else 0.0
                      for ex in examples])


def extract_recall1(examples):
    """Per-example recall@1 (fractional)."""
    return np.array([ex["metrics"]["recall@1"] for ex in examples])


def extract_repo_labels(examples):
    """Return list of repo names, aligned with examples."""
    return [ex["repo"] for ex in examples]


# ---------------------------------------------------------------------------
# Bootstrap CI for paired difference
# ---------------------------------------------------------------------------
def paired_bootstrap_ci(a, b, n_boot=B, alpha=0.05):
    """
    Paired bootstrap 95% CI for mean(a) - mean(b).
    Also returns the observed difference, p-value (two-sided), and bootstrap SE.
    """
    n = len(a)
    assert len(b) == n
    obs_diff = np.mean(a) - np.mean(b)
    diffs = np.empty(n_boot)
    rng = np.random.RandomState(42)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs[i] = np.mean(a[idx]) - np.mean(b[idx])
    lo = np.percentile(diffs, 100 * alpha / 2)
    hi = np.percentile(diffs, 100 * (1 - alpha / 2))
    se = np.std(diffs, ddof=1)
    # p-value: fraction of bootstrap samples where diff <= 0 (one-sided),
    # convert to two-sided
    p_one = np.mean(diffs <= 0)
    p_val = 2 * min(p_one, 1 - p_one)
    p_val = max(p_val, 1.0 / n_boot)  # floor at 1/B
    return {
        "obs_diff": obs_diff,
        "ci_lo": lo,
        "ci_hi": hi,
        "se": se,
        "p_value": p_val,
    }


# ---------------------------------------------------------------------------
# Bootstrap CI for a single mean
# ---------------------------------------------------------------------------
def bootstrap_mean_ci(a, n_boot=B, alpha=0.05):
    """Bootstrap 95% CI for mean(a)."""
    n = len(a)
    means = np.empty(n_boot)
    rng = np.random.RandomState(42)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        means[i] = np.mean(a[idx])
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return np.mean(a), lo, hi


# ---------------------------------------------------------------------------
# Cohen's d (paired)
# ---------------------------------------------------------------------------
def cohens_d_paired(a, b):
    """Cohen's d for paired samples: mean(a-b) / sd(a-b)."""
    diff = a - b
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return float("inf") if np.mean(diff) != 0 else 0.0
    return np.mean(diff) / sd


def interpret_d(d):
    """Interpret Cohen's d magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------
def mcnemar_test(a_binary, b_binary):
    """
    McNemar's test for paired binary outcomes.
    a_binary, b_binary: 1d arrays of 0/1.
    Returns (chi2, p_value, n01, n10) where:
      n01 = #examples where a=0, b=1 (b correct, a wrong)
      n10 = #examples where a=1, b=0 (a correct, b wrong)
    Uses exact binomial test when discordant pairs < 25, else chi-squared.
    """
    n = len(a_binary)
    assert len(b_binary) == n
    a = a_binary.astype(int)
    b = b_binary.astype(int)
    n01 = int(np.sum((a == 0) & (b == 1)))  # b wins
    n10 = int(np.sum((a == 1) & (b == 0)))  # a wins
    n11 = int(np.sum((a == 1) & (b == 1)))
    n00 = int(np.sum((a == 0) & (b == 0)))
    discordant = n01 + n10
    if discordant == 0:
        return 0.0, 1.0, n01, n10, n11, n00
    if discordant < 25:
        # Exact binomial test (two-sided)
        p_val = stats.binom_test(n01, discordant, 0.5)
    else:
        # McNemar chi-squared with continuity correction
        chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p_val = stats.chi2.sf(chi2, df=1)
    chi2_val = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if discordant > 0 else 0.0
    return chi2_val, p_val, n01, n10, n11, n00


# ---------------------------------------------------------------------------
# Per-repo analysis
# ---------------------------------------------------------------------------
def per_repo_analysis(examples_a, examples_b, label_a, label_b):
    """
    For each repo, compute mean recall@1 for conditions a and b.
    Returns dict with per-repo diffs, % repos where a > b, signed-rank test.
    """
    repos_a = defaultdict(list)
    repos_b = defaultdict(list)
    for ex in examples_a:
        repos_a[ex["repo"]].append(ex["metrics"]["recall@1"])
    for ex in examples_b:
        repos_b[ex["repo"]].append(ex["metrics"]["recall@1"])

    all_repos = sorted(set(repos_a.keys()) & set(repos_b.keys()))
    means_a = []
    means_b = []
    for repo in all_repos:
        means_a.append(np.mean(repos_a[repo]))
        means_b.append(np.mean(repos_b[repo]))
    means_a = np.array(means_a)
    means_b = np.array(means_b)
    diffs = means_a - means_b

    pct_a_wins = np.mean(diffs > 0) * 100
    pct_b_wins = np.mean(diffs < 0) * 100
    pct_ties = np.mean(diffs == 0) * 100

    # Wilcoxon signed-rank test (two-sided)
    # Filter out zero diffs for the test
    nonzero_mask = diffs != 0
    if np.sum(nonzero_mask) >= 10:
        stat, p_val = stats.wilcoxon(diffs[nonzero_mask])
    else:
        stat, p_val = float("nan"), float("nan")

    return {
        "repos": all_repos,
        "means_a": means_a,
        "means_b": means_b,
        "diffs": diffs,
        "n_repos": len(all_repos),
        "pct_a_wins": pct_a_wins,
        "pct_b_wins": pct_b_wins,
        "pct_ties": pct_ties,
        "wilcoxon_stat": stat,
        "wilcoxon_p": p_val,
        "label_a": label_a,
        "label_b": label_b,
    }


# ---------------------------------------------------------------------------
# LaTeX formatting helpers
# ---------------------------------------------------------------------------
def fmt_pct(x):
    """Format as percentage with 1 decimal."""
    return f"{x * 100:.1f}"


def fmt_ci(lo, hi):
    """Format CI as [lo, hi]."""
    return f"[{lo * 100:.1f}, {hi * 100:.1f}]"


def fmt_p(p):
    """Format p-value for LaTeX."""
    if p < 0.001:
        return f"$p < 0.001$"
    elif p < 0.01:
        return f"$p = {p:.3f}$"
    elif p < 0.05:
        return f"$p = {p:.2f}$"
    else:
        return f"$p = {p:.2f}$"


def fmt_d(d):
    return f"$d = {d:.3f}$"


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Loading predictions...")
    data = {}
    for key, path in PRED_DIRS.items():
        data[key] = load_predictions(path)
        print(f"  {key}: {len(data[key])} examples from {path}")

    # Verify alignment: all files should have same (repo, issue_id) in same order
    ref_ids = [(ex["repo"], ex["issue_id"]) for ex in data["graph_expanded"]]
    for key in ["bm25_expanded", "graph_bm25pool", "bm25_bm25pool"]:
        other_ids = [(ex["repo"], ex["issue_id"]) for ex in data[key]]
        assert ref_ids == other_ids, f"Mismatch in example ordering: {key}"
    print(f"All {len(ref_ids)} examples aligned across conditions.\n")

    # Extract binary hit@1 and recall@1
    hit1 = {k: extract_binary_hit1(v) for k, v in data.items()}
    rec1 = {k: extract_recall1(v) for k, v in data.items()}

    tex_lines = []
    tex_lines.append("% CodeGRIP Statistical Analysis")
    tex_lines.append("% Auto-generated by scripts/analyze_statistics.py")
    tex_lines.append("% B = {:,} bootstrap iterations, seed = 42".format(B))
    tex_lines.append("")

    # -----------------------------------------------------------------------
    # 1. Bootstrap CIs for key comparisons
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("1. PAIRED BOOTSTRAP CONFIDENCE INTERVALS (B={:,})".format(B))
    print("=" * 70)

    comparisons = [
        # (label, metric_name, array_a, array_b, description)
        ("Graph expansion (Recall@1)",
         "Recall@1", rec1["graph_expanded"], rec1["graph_bm25pool"],
         "Graph-hard reranker on expanded pool vs BM25-only pool"),
        ("Graph expansion (Hit@1)",
         "Hit@1", hit1["graph_expanded"], hit1["graph_bm25pool"],
         "Graph-hard reranker on expanded pool vs BM25-only pool"),
        ("Graph-hard training (Recall@1, expanded pool)",
         "Recall@1", rec1["graph_expanded"], rec1["bm25_expanded"],
         "Graph-hard vs BM25-only reranker, both on expanded pool"),
        ("Graph-hard training (Hit@1, expanded pool)",
         "Hit@1", hit1["graph_expanded"], hit1["bm25_expanded"],
         "Graph-hard vs BM25-only reranker, both on expanded pool"),
        ("Graph-hard training (Recall@1, BM25 pool)",
         "Recall@1", rec1["graph_bm25pool"], rec1["bm25_bm25pool"],
         "Graph-hard vs BM25-only reranker, both on BM25-only pool"),
        ("Graph-hard training (Hit@1, BM25 pool)",
         "Hit@1", hit1["graph_bm25pool"], hit1["bm25_bm25pool"],
         "Graph-hard vs BM25-only reranker, both on BM25-only pool"),
        ("Full system (Recall@1)",
         "Recall@1", rec1["graph_expanded"], rec1["bm25_bm25pool"],
         "Full CodeGRIP (graph-hard + expanded) vs baseline (BM25-only + BM25 pool)"),
        ("Full system (Hit@1)",
         "Hit@1", hit1["graph_expanded"], hit1["bm25_bm25pool"],
         "Full CodeGRIP (graph-hard + expanded) vs baseline (BM25-only + BM25 pool)"),
    ]

    tex_lines.append("% =============================================")
    tex_lines.append("% 1. Paired Bootstrap 95\\% CIs")
    tex_lines.append("% =============================================")
    tex_lines.append("")

    for label, metric, a, b, desc in comparisons:
        result = paired_bootstrap_ci(a, b, n_boot=B)
        mean_a, ci_a_lo, ci_a_hi = bootstrap_mean_ci(a, n_boot=B)
        mean_b, ci_b_lo, ci_b_hi = bootstrap_mean_ci(b, n_boot=B)

        print(f"\n--- {label} ---")
        print(f"  {desc}")
        print(f"  Condition A ({metric}): {mean_a*100:.2f}% {fmt_ci(ci_a_lo, ci_a_hi)}")
        print(f"  Condition B ({metric}): {mean_b*100:.2f}% {fmt_ci(ci_b_lo, ci_b_hi)}")
        print(f"  Diff (A-B): {result['obs_diff']*100:+.2f}pp "
              f"95% CI {fmt_ci(result['ci_lo'], result['ci_hi'])}")
        print(f"  Bootstrap SE: {result['se']*100:.2f}pp")
        print(f"  Bootstrap p: {result['p_value']:.4f}")

        # LaTeX macro names
        safe_label = label.replace(" ", "").replace("(", "").replace(")", "").replace(",", "").replace("@", "at").replace("-", "")
        tex_lines.append(f"% {label}: {desc}")
        tex_lines.append(f"\\newcommand{{\\stat{safe_label}MeanA}}{{{mean_a*100:.1f}\\%}}")
        tex_lines.append(f"\\newcommand{{\\stat{safe_label}CIA}}{{[{ci_a_lo*100:.1f}, {ci_a_hi*100:.1f}]}}")
        tex_lines.append(f"\\newcommand{{\\stat{safe_label}MeanB}}{{{mean_b*100:.1f}\\%}}")
        tex_lines.append(f"\\newcommand{{\\stat{safe_label}CIB}}{{[{ci_b_lo*100:.1f}, {ci_b_hi*100:.1f}]}}")
        tex_lines.append(f"\\newcommand{{\\stat{safe_label}Diff}}{{{result['obs_diff']*100:+.1f}}}")
        tex_lines.append(f"\\newcommand{{\\stat{safe_label}CIDiff}}{{[{result['ci_lo']*100:.1f}, {result['ci_hi']*100:.1f}]}}")
        tex_lines.append(f"\\newcommand{{\\stat{safe_label}Pval}}{{{result['p_value']:.4f}}}")
        tex_lines.append("")

    # -----------------------------------------------------------------------
    # 2. Effect sizes (Cohen's d)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. EFFECT SIZES (Cohen's d, paired)")
    print("=" * 70)

    tex_lines.append("% =============================================")
    tex_lines.append("% 2. Effect Sizes (Cohen's d)")
    tex_lines.append("% =============================================")
    tex_lines.append("")

    effect_comparisons = [
        ("Graph expansion (Recall@1)",
         rec1["graph_expanded"], rec1["graph_bm25pool"]),
        ("Graph expansion (Hit@1)",
         hit1["graph_expanded"], hit1["graph_bm25pool"]),
        ("Graph-hard training (Recall@1, expanded pool)",
         rec1["graph_expanded"], rec1["bm25_expanded"]),
        ("Graph-hard training (Hit@1, expanded pool)",
         hit1["graph_expanded"], hit1["bm25_expanded"]),
        ("Graph-hard training (Recall@1, BM25 pool)",
         rec1["graph_bm25pool"], rec1["bm25_bm25pool"]),
        ("Full system (Recall@1)",
         rec1["graph_expanded"], rec1["bm25_bm25pool"]),
        ("Full system (Hit@1)",
         hit1["graph_expanded"], hit1["bm25_bm25pool"]),
    ]

    for label, a, b in effect_comparisons:
        d = cohens_d_paired(a, b)
        interp = interpret_d(d)
        print(f"  {label}: d = {d:.4f} ({interp})")

        safe_label = label.replace(" ", "").replace("(", "").replace(")", "").replace(",", "").replace("@", "at").replace("-", "")
        tex_lines.append(f"\\newcommand{{\\stat{safe_label}CohenD}}{{{d:.3f}}}")
        tex_lines.append(f"\\newcommand{{\\stat{safe_label}CohenDInterp}}{{{interp}}}")

    tex_lines.append("")

    # -----------------------------------------------------------------------
    # 3. Per-repo analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. PER-REPO ANALYSIS")
    print("=" * 70)

    tex_lines.append("% =============================================")
    tex_lines.append("% 3. Per-Repo Analysis")
    tex_lines.append("% =============================================")
    tex_lines.append("")

    repo_comparisons = [
        ("Graph expansion effect",
         data["graph_expanded"], data["graph_bm25pool"],
         "graph+expanded", "graph+BM25pool"),
        ("Graph-hard training effect (expanded pool)",
         data["graph_expanded"], data["bm25_expanded"],
         "graph reranker", "BM25 reranker"),
        ("Full system effect",
         data["graph_expanded"], data["bm25_bm25pool"],
         "full CodeGRIP", "BM25 baseline"),
    ]

    for label, ex_a, ex_b, lab_a, lab_b in repo_comparisons:
        result = per_repo_analysis(ex_a, ex_b, lab_a, lab_b)
        print(f"\n--- {label} ---")
        print(f"  N repos: {result['n_repos']}")
        print(f"  % repos where {lab_a} wins: {result['pct_a_wins']:.1f}%")
        print(f"  % repos where {lab_b} wins: {result['pct_b_wins']:.1f}%")
        print(f"  % repos tied: {result['pct_ties']:.1f}%")
        print(f"  Wilcoxon signed-rank test: W = {result['wilcoxon_stat']:.1f}, "
              f"p = {result['wilcoxon_p']:.4f}")
        print(f"  Mean per-repo diff: {np.mean(result['diffs'])*100:+.2f}pp")
        print(f"  Median per-repo diff: {np.median(result['diffs'])*100:+.2f}pp")

        safe_label = label.replace(" ", "").replace("(", "").replace(")", "").replace(",", "").replace("-", "")
        tex_lines.append(f"% {label}")
        tex_lines.append(f"\\newcommand{{\\stat{safe_label}Nrepos}}{{{result['n_repos']}}}")
        tex_lines.append(f"\\newcommand{{\\stat{safe_label}PctAwins}}{{{result['pct_a_wins']:.1f}\\%}}")
        tex_lines.append(f"\\newcommand{{\\stat{safe_label}PctBwins}}{{{result['pct_b_wins']:.1f}\\%}}")
        tex_lines.append(f"\\newcommand{{\\stat{safe_label}WilcoxP}}{{{result['wilcoxon_p']:.4f}}}")
        tex_lines.append(f"\\newcommand{{\\stat{safe_label}MeanDiff}}{{{np.mean(result['diffs'])*100:+.1f}}}")
        tex_lines.append(f"\\newcommand{{\\stat{safe_label}MedianDiff}}{{{np.median(result['diffs'])*100:+.1f}}}")
        tex_lines.append("")

    # -----------------------------------------------------------------------
    # 4. McNemar's test
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. McNEMAR'S TEST (paired binary hit@1)")
    print("=" * 70)

    tex_lines.append("% =============================================")
    tex_lines.append("% 4. McNemar's Test")
    tex_lines.append("% =============================================")
    tex_lines.append("")

    mcnemar_comparisons = [
        ("Graph expansion (hit@1)",
         hit1["graph_expanded"], hit1["graph_bm25pool"],
         "expanded pool", "BM25-only pool"),
        ("Graph-hard training (hit@1, expanded pool)",
         hit1["graph_expanded"], hit1["bm25_expanded"],
         "graph-hard reranker", "BM25-only reranker"),
        ("Graph-hard training (hit@1, BM25 pool)",
         hit1["graph_bm25pool"], hit1["bm25_bm25pool"],
         "graph-hard reranker", "BM25-only reranker"),
        ("Full system (hit@1)",
         hit1["graph_expanded"], hit1["bm25_bm25pool"],
         "full CodeGRIP", "BM25 baseline"),
    ]

    for label, a, b, lab_a, lab_b in mcnemar_comparisons:
        chi2, p_val, n01, n10, n11, n00 = mcnemar_test(a, b)
        total = len(a)
        print(f"\n--- {label} ---")
        print(f"  Contingency table:")
        print(f"    Both correct (1,1): {n11}")
        print(f"    {lab_a} only (1,0):  {n10}")
        print(f"    {lab_b} only (0,1):  {n01}")
        print(f"    Both wrong (0,0):   {n00}")
        print(f"  Discordant pairs: {n01 + n10}")
        print(f"  Chi-squared: {chi2:.2f}")
        print(f"  p-value: {p_val:.4f}")
        print(f"  Net advantage ({lab_a}): {n10 - n01} examples "
              f"({(n10-n01)/total*100:+.1f}% of {total})")

        safe_label = label.replace(" ", "").replace("(", "").replace(")", "").replace(",", "").replace("@", "at").replace("-", "")
        tex_lines.append(f"% {label}")
        tex_lines.append(f"\\newcommand{{\\statMcNemar{safe_label}Chi}}{{{chi2:.2f}}}")
        tex_lines.append(f"\\newcommand{{\\statMcNemar{safe_label}Pval}}{{{p_val:.4f}}}")
        tex_lines.append(f"\\newcommand{{\\statMcNemar{safe_label}Concordant}}{{{n11 + n00}}}")
        tex_lines.append(f"\\newcommand{{\\statMcNemar{safe_label}Discordant}}{{{n01 + n10}}}")
        tex_lines.append(f"\\newcommand{{\\statMcNemar{safe_label}NetAdv}}{{{n10 - n01}}}")
        tex_lines.append(f"\\newcommand{{\\statMcNemar{safe_label}Awins}}{{{n10}}}")
        tex_lines.append(f"\\newcommand{{\\statMcNemar{safe_label}Bwins}}{{{n01}}}")
        tex_lines.append("")

    # -----------------------------------------------------------------------
    # Summary table (LaTeX)
    # -----------------------------------------------------------------------
    tex_lines.append("% =============================================")
    tex_lines.append("% Summary Table")
    tex_lines.append("% =============================================")
    tex_lines.append("")
    tex_lines.append("\\begin{table}[t]")
    tex_lines.append("\\centering")
    tex_lines.append("\\caption{Statistical analysis of CodeGRIP ablations. "
                      "All CIs are 95\\% paired bootstrap ($B=10{,}000$). "
                      "Effect sizes are paired Cohen's $d$.}")
    tex_lines.append("\\label{tab:statistical-analysis}")
    tex_lines.append("\\small")
    tex_lines.append("\\begin{tabular}{lcccc}")
    tex_lines.append("\\toprule")
    tex_lines.append("Comparison & $\\Delta$ R@1 (pp) & 95\\% CI & Cohen's $d$ & $p$-value \\\\")
    tex_lines.append("\\midrule")

    # Recompute for table rows
    table_rows = [
        ("Graph expansion",
         rec1["graph_expanded"], rec1["graph_bm25pool"]),
        ("Graph-hard training",
         rec1["graph_expanded"], rec1["bm25_expanded"]),
        ("Full system",
         rec1["graph_expanded"], rec1["bm25_bm25pool"]),
    ]
    for row_label, a, b in table_rows:
        r = paired_bootstrap_ci(a, b, n_boot=B)
        d = cohens_d_paired(a, b)
        p_str = f"${r['p_value']:.4f}$" if r['p_value'] >= 0.001 else "$<0.001$"
        tex_lines.append(
            f"{row_label} & ${r['obs_diff']*100:+.1f}$ & "
            f"$[{r['ci_lo']*100:.1f}, {r['ci_hi']*100:.1f}]$ & "
            f"${d:.3f}$ & {p_str} \\\\"
        )

    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    tex_lines.append("\\end{table}")
    tex_lines.append("")

    # McNemar summary table
    tex_lines.append("\\begin{table}[t]")
    tex_lines.append("\\centering")
    tex_lines.append("\\caption{McNemar's test for paired binary Hit@1 outcomes.}")
    tex_lines.append("\\label{tab:mcnemar}")
    tex_lines.append("\\small")
    tex_lines.append("\\begin{tabular}{lccccc}")
    tex_lines.append("\\toprule")
    tex_lines.append("Comparison & A wins & B wins & Net & $\\chi^2$ & $p$-value \\\\")
    tex_lines.append("\\midrule")

    for label, a, b, lab_a, lab_b in mcnemar_comparisons:
        chi2, p_val, n01, n10, n11, n00 = mcnemar_test(a, b)
        p_str = f"${p_val:.4f}$" if p_val >= 0.001 else "$<0.001$"
        short_label = label.split("(")[0].strip()
        tex_lines.append(
            f"{short_label} & {n10} & {n01} & ${n10-n01:+d}$ & "
            f"${chi2:.1f}$ & {p_str} \\\\"
        )

    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    tex_lines.append("\\end{table}")
    tex_lines.append("")

    # Per-repo summary table
    tex_lines.append("\\begin{table}[t]")
    tex_lines.append("\\centering")
    tex_lines.append("\\caption{Per-repository analysis. "
                      "Win rate = \\% of repos where condition A outperforms B "
                      "in mean Recall@1. Wilcoxon signed-rank test across repos.}")
    tex_lines.append("\\label{tab:per-repo}")
    tex_lines.append("\\small")
    tex_lines.append("\\begin{tabular}{lccc}")
    tex_lines.append("\\toprule")
    tex_lines.append("Comparison & Win rate & Median $\\Delta$ (pp) & Wilcoxon $p$ \\\\")
    tex_lines.append("\\midrule")

    for label, ex_a, ex_b, lab_a, lab_b in repo_comparisons:
        result = per_repo_analysis(ex_a, ex_b, lab_a, lab_b)
        p_str = f"${result['wilcoxon_p']:.4f}$" if result['wilcoxon_p'] >= 0.001 else "$<0.001$"
        short_label = label.replace(" effect", "")
        tex_lines.append(
            f"{short_label} & {result['pct_a_wins']:.0f}\\% & "
            f"${np.median(result['diffs'])*100:+.1f}$ & {p_str} \\\\"
        )

    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    tex_lines.append("\\end{table}")

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"\nLaTeX output written to: {OUT_TEX}")
    print("Done.")


if __name__ == "__main__":
    main()
