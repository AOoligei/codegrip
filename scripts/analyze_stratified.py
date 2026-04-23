"""
Stratified analysis: when/where does graph expansion help most?

Compares R@1 (= Hit@1) across strata for four experiment conditions:
  - runA_bm25only  on bm25pool        (BM25 model + BM25 pool)
  - runA_bm25only  on merged_rerank    (BM25 model + expanded pool)
  - runB_graph     on bm25pool        (Graph model + BM25 pool)
  - runB_graph     on merged_rerank    (Graph model + expanded pool)

Strata:
  1. Lexical overlap between issue_text and GT file paths
  2. Repo size (number of Python files per repo)
  3. Number of GT files (single vs multi-file)
  4. Graph expansion benefit (where GT appears)
  5. Co-change vs import contribution

Outputs LaTeX-friendly tables to stdout and saves CSV.
"""

import json
import os
import re
import random
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional

import numpy as np

random.seed(42)
np.random.seed(42)

# ============================================================
# Paths
# ============================================================
BASE = "/home/chenlibin/grepo_agent"

TEST_DATA = os.path.join(BASE, "data/grepo_text/grepo_test.jsonl")
BM25_CANDIDATES = os.path.join(BASE, "data/rankft/grepo_test_bm25_top500.jsonl")
MERGED_CANDIDATES = os.path.join(BASE, "data/rankft/merged_bm25_exp6_candidates.jsonl")
COCHANGE_CANDIDATES = os.path.join(BASE, "data/rankft/merged_bm25_cochange_only_candidates.jsonl")
IMPORT_CANDIDATES = os.path.join(BASE, "data/rankft/merged_bm25_import_only_candidates.jsonl")

PRED_DIRS = {
    "GraphModel+ExpandedPool": os.path.join(BASE, "experiments/rankft_runB_graph/eval_merged_rerank"),
    "GraphModel+BM25Pool":     os.path.join(BASE, "experiments/rankft_runB_graph/eval_bm25pool"),
    "BM25Model+ExpandedPool":  os.path.join(BASE, "experiments/rankft_runA_bm25only/eval_merged_rerank"),
    "BM25Model+BM25Pool":      os.path.join(BASE, "experiments/rankft_runA_bm25only/eval_bm25pool"),
}

OUTPUT_DIR = os.path.join(BASE, "experiments/analysis")


# ============================================================
# Data loading
# ============================================================

def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_predictions(pred_dir: str) -> Dict[str, dict]:
    """Load predictions.jsonl keyed by repo_issueId."""
    preds = {}
    path = os.path.join(pred_dir, "predictions.jsonl")
    if not os.path.exists(path):
        return preds
    for item in load_jsonl(path):
        key = f"{item['repo']}_{item['issue_id']}"
        preds[key] = item
    return preds


def compute_r_at_1(predicted: List[str], gt: Set[str]) -> float:
    """R@1 = Hit@1: fraction of GT files in top-1."""
    if not gt:
        return 0.0
    return len(set(predicted[:1]) & gt) / len(gt)


# ============================================================
# Helpers
# ============================================================

def tokenize_path(path: str) -> Set[str]:
    """Split file path into word tokens for overlap calculation."""
    # Split on / . _ - and lowercase
    tokens = re.split(r'[/._\-]', path.lower())
    tokens = [t for t in tokens if t and len(t) > 1]  # drop single chars
    return set(tokens)


def compute_lexical_overlap(issue_text: str, gt_files: List[str]) -> float:
    """Fraction of GT path tokens that appear in issue_text."""
    issue_lower = issue_text.lower()
    all_path_tokens = set()
    for f in gt_files:
        all_path_tokens |= tokenize_path(f)
    if not all_path_tokens:
        return 0.0
    matched = sum(1 for t in all_path_tokens if t in issue_lower)
    return matched / len(all_path_tokens)


def count_py_files(candidates: List[str]) -> int:
    """Estimate repo size from BM25 candidate pool size."""
    return len(candidates)


# ============================================================
# Main analysis
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load test data ---
    print("Loading test data...")
    test_raw = load_jsonl(TEST_DATA)
    test_data = {}
    for item in test_raw:
        gt = item.get("changed_py_files", [])
        if not gt:
            continue
        key = f"{item['repo']}_{item['issue_id']}"
        test_data[key] = item

    print(f"  {len(test_data)} test examples with changed_py_files")

    # --- Load candidate pools ---
    print("Loading candidate pools...")
    bm25_pool = {}
    repo_all_files = defaultdict(set)  # aggregate unique files per repo
    for item in load_jsonl(BM25_CANDIDATES):
        key = f"{item['repo']}_{item['issue_id']}"
        cands = item.get("bm25_candidates", [])
        bm25_pool[key] = set(cands)
        repo_all_files[item["repo"]].update(cands)

    merged_pool = {}
    for item in load_jsonl(MERGED_CANDIDATES):
        key = f"{item['repo']}_{item['issue_id']}"
        merged_pool[key] = set(item.get("candidates", []))

    cochange_pool = {}
    if os.path.exists(COCHANGE_CANDIDATES):
        for item in load_jsonl(COCHANGE_CANDIDATES):
            key = f"{item['repo']}_{item['issue_id']}"
            cochange_pool[key] = set(item.get("candidates", []))

    import_pool = {}
    if os.path.exists(IMPORT_CANDIDATES):
        for item in load_jsonl(IMPORT_CANDIDATES):
            key = f"{item['repo']}_{item['issue_id']}"
            import_pool[key] = set(item.get("candidates", []))

    print(f"  BM25 pool: {len(bm25_pool)} examples")
    print(f"  Merged pool: {len(merged_pool)} examples")
    print(f"  Cochange pool: {len(cochange_pool)} examples")
    print(f"  Import pool: {len(import_pool)} examples")

    # --- Load predictions ---
    print("Loading predictions...")
    preds = {}
    for name, d in PRED_DIRS.items():
        preds[name] = load_predictions(d)
        print(f"  {name}: {len(preds[name])} predictions")

    # --- Build per-example annotations ---
    keys = sorted(test_data.keys())
    # Filter to keys present in all prediction sets
    valid_keys = [k for k in keys if all(k in preds[n] for n in preds)]
    print(f"\n{len(valid_keys)} examples with predictions in all 4 conditions")

    annotations = {}
    for key in valid_keys:
        item = test_data[key]
        gt_files = item["changed_py_files"]
        gt_set = set(gt_files)
        issue_text = item["issue_text"]

        bm25_cands = bm25_pool.get(key, set())
        merged_cands = merged_pool.get(key, set())

        ann = {}
        ann["repo"] = item["repo"]
        ann["issue_id"] = item["issue_id"]
        ann["gt_set"] = gt_set
        ann["n_gt_files"] = len(gt_files)

        # 1. Lexical overlap
        ann["lexical_overlap"] = compute_lexical_overlap(issue_text, gt_files)

        # 2. Repo size: unique files seen across all BM25 entries for this repo
        ann["repo_size"] = len(repo_all_files.get(item["repo"], set()))

        # 3. Multi-file
        ann["is_multifile"] = len(gt_files) > 1

        # 4. Graph expansion benefit categories
        gt_in_bm25 = gt_set & bm25_cands
        gt_in_merged = gt_set & merged_cands
        expanded_only = merged_cands - bm25_cands  # files added by expansion

        gt_only_in_bm25 = gt_in_bm25 - gt_in_merged
        gt_only_in_expanded = gt_in_merged - gt_in_bm25
        gt_in_both = gt_in_bm25 & gt_in_merged
        gt_in_neither = gt_set - gt_in_bm25 - gt_in_merged

        if gt_only_in_expanded:
            ann["expansion_cat"] = "GT only in expanded"
        elif gt_only_in_bm25:
            ann["expansion_cat"] = "GT only in BM25"
        elif gt_in_both:
            ann["expansion_cat"] = "GT in both"
        else:
            ann["expansion_cat"] = "GT in neither"

        # Also track per-file for finer analysis
        ann["gt_only_in_expanded_files"] = gt_only_in_expanded
        ann["gt_in_expanded_only_count"] = len(gt_only_in_expanded)

        # 5. Co-change vs import attribution
        cochange_cands = cochange_pool.get(key, set())
        import_cands = import_pool.get(key, set())
        cochange_added = cochange_cands - bm25_cands
        import_added = import_cands - bm25_cands

        gt_from_cochange = gt_set & cochange_added
        gt_from_import = gt_set & import_added

        if gt_from_cochange and gt_from_import:
            ann["edge_source"] = "both"
        elif gt_from_cochange:
            ann["edge_source"] = "cochange"
        elif gt_from_import:
            ann["edge_source"] = "import"
        elif gt_only_in_expanded:
            ann["edge_source"] = "other_expansion"
        else:
            ann["edge_source"] = "none"

        ann["gt_from_cochange_count"] = len(gt_from_cochange)
        ann["gt_from_import_count"] = len(gt_from_import)

        # R@1 for each condition
        for name in preds:
            p = preds[name].get(key)
            if p:
                ann[f"r1_{name}"] = compute_r_at_1(p["predicted"], gt_set)
            else:
                ann[f"r1_{name}"] = np.nan

        annotations[key] = ann

    # ============================================================
    # Produce tables
    # ============================================================

    cond_names = list(PRED_DIRS.keys())
    cond_short = {
        "GraphModel+ExpandedPool": "G+Exp",
        "GraphModel+BM25Pool":     "G+BM25",
        "BM25Model+ExpandedPool":  "B+Exp",
        "BM25Model+BM25Pool":      "B+BM25",
    }

    def stratum_table(title: str, strata: Dict[str, List[str]]):
        """Print a table of R@1 per stratum per condition."""
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")

        # Header
        header = f"{'Stratum':<25} {'N':>5}"
        for c in cond_names:
            header += f"  {cond_short[c]:>7}"
        header += f"  {'Delta':>7}"
        print(header)
        print("-" * len(header))

        rows = []
        for sname, skeys in sorted(strata.items()):
            if not skeys:
                continue
            n = len(skeys)
            vals = {}
            for c in cond_names:
                vs = [annotations[k][f"r1_{c}"] for k in skeys
                      if not np.isnan(annotations[k][f"r1_{c}"])]
                vals[c] = np.mean(vs) * 100 if vs else float("nan")

            # Delta: (Graph+Expanded) - (BM25+BM25)
            delta = vals.get("GraphModel+ExpandedPool", 0) - vals.get("BM25Model+BM25Pool", 0)

            row = f"{sname:<25} {n:>5}"
            for c in cond_names:
                v = vals.get(c, float("nan"))
                row += f"  {v:>6.1f}%"
            row += f"  {delta:>+6.1f}%"
            print(row)
            rows.append((sname, n, vals, delta))

        return rows

    def latex_table(title: str, strata_rows, caption: str = ""):
        """Format strata_rows as LaTeX tabular."""
        n_cols = len(cond_names)
        cols = "l" + "r" * (n_cols + 2)  # stratum, N, conditions, delta
        lines = []
        lines.append(f"% {title}")
        lines.append(f"\\begin{{tabular}}{{{cols}}}")
        lines.append("\\toprule")
        header = "Stratum & N"
        for c in cond_names:
            header += f" & {cond_short[c]}"
        header += " & $\\Delta$ \\\\"
        lines.append(header)
        lines.append("\\midrule")
        for sname, n, vals, delta in strata_rows:
            row = f"{sname} & {n}"
            for c in cond_names:
                v = vals.get(c, float("nan"))
                if np.isnan(v):
                    row += " & --"
                else:
                    row += f" & {v:.1f}"
            row += f" & {delta:+.1f} \\\\"
            lines.append(row)
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        return "\n".join(lines)

    all_latex = []

    # ============================================================
    # 1. Lexical overlap stratification
    # ============================================================
    overlap_strata = {"high_overlap (>50%)": [], "low_overlap (<=50%)": []}
    for key in valid_keys:
        ann = annotations[key]
        if ann["lexical_overlap"] > 0.5:
            overlap_strata["high_overlap (>50%)"].append(key)
        else:
            overlap_strata["low_overlap (<=50%)"].append(key)

    rows = stratum_table("1. Lexical Overlap (issue_text vs GT paths)", overlap_strata)
    all_latex.append(latex_table("Lexical Overlap", rows))

    # ============================================================
    # 2. Repo size stratification
    # ============================================================
    size_strata = {"small (<200)": [], "medium (200-500)": [], "large (>500)": []}
    for key in valid_keys:
        ann = annotations[key]
        s = ann["repo_size"]
        if s < 200:
            size_strata["small (<200)"].append(key)
        elif s <= 500:
            size_strata["medium (200-500)"].append(key)
        else:
            size_strata["large (>500)"].append(key)

    rows = stratum_table("2. Repo Size (BM25 pool size as proxy)", size_strata)
    all_latex.append(latex_table("Repo Size", rows))

    # ============================================================
    # 3. Single vs multi-file
    # ============================================================
    gt_strata = {"single-file (1 GT)": [], "multi-file (2+ GT)": []}
    for key in valid_keys:
        ann = annotations[key]
        if ann["n_gt_files"] == 1:
            gt_strata["single-file (1 GT)"].append(key)
        else:
            gt_strata["multi-file (2+ GT)"].append(key)

    rows = stratum_table("3. Number of GT Files", gt_strata)
    all_latex.append(latex_table("GT File Count", rows))

    # ============================================================
    # 4. Graph expansion benefit
    # ============================================================
    exp_strata = defaultdict(list)
    for key in valid_keys:
        ann = annotations[key]
        exp_strata[ann["expansion_cat"]].append(key)

    rows = stratum_table("4. Graph Expansion Benefit (where GT appears)", dict(exp_strata))
    all_latex.append(latex_table("Expansion Benefit", rows))

    # ============================================================
    # 5. Co-change vs import contribution
    # ============================================================
    # Only among examples where expansion adds at least one GT file
    edge_strata = defaultdict(list)
    for key in valid_keys:
        ann = annotations[key]
        if ann["gt_in_expanded_only_count"] > 0 or ann["gt_from_cochange_count"] > 0 or ann["gt_from_import_count"] > 0:
            edge_strata[ann["edge_source"]].append(key)

    if edge_strata:
        rows = stratum_table("5. Edge Source (among examples where graph adds GT)", dict(edge_strata))
        all_latex.append(latex_table("Edge Source", rows))
    else:
        print("\n[5. Edge Source] No examples where graph expansion adds GT file.")

    # ============================================================
    # Summary statistics
    # ============================================================
    print(f"\n{'='*80}")
    print("  Summary Statistics")
    print(f"{'='*80}")

    # Expansion category distribution
    exp_counts = Counter(annotations[k]["expansion_cat"] for k in valid_keys)
    print("\nExpansion category distribution:")
    for cat, cnt in exp_counts.most_common():
        print(f"  {cat:<30} {cnt:>5} ({cnt/len(valid_keys)*100:.1f}%)")

    # Edge source distribution
    edge_counts = Counter(annotations[k]["edge_source"] for k in valid_keys)
    print("\nEdge source distribution (all examples):")
    for src, cnt in edge_counts.most_common():
        print(f"  {src:<30} {cnt:>5} ({cnt/len(valid_keys)*100:.1f}%)")

    # Overall R@1
    print("\nOverall R@1 (%):")
    for c in cond_names:
        vals = [annotations[k][f"r1_{c}"] for k in valid_keys
                if not np.isnan(annotations[k][f"r1_{c}"])]
        print(f"  {cond_short[c]:<10} {np.mean(vals)*100:.2f}% (N={len(vals)})")

    # Lexical overlap stats
    overlaps = [annotations[k]["lexical_overlap"] for k in valid_keys]
    print(f"\nLexical overlap: mean={np.mean(overlaps):.3f}, "
          f"median={np.median(overlaps):.3f}, "
          f"p25={np.percentile(overlaps, 25):.3f}, "
          f"p75={np.percentile(overlaps, 75):.3f}")

    # GT file count distribution
    gt_counts = Counter(annotations[k]["n_gt_files"] for k in valid_keys)
    print(f"\nGT file count distribution:")
    for n in sorted(gt_counts.keys()):
        print(f"  {n} files: {gt_counts[n]:>5} ({gt_counts[n]/len(valid_keys)*100:.1f}%)")

    # ============================================================
    # Save LaTeX
    # ============================================================
    latex_path = os.path.join(OUTPUT_DIR, "stratified_tables.tex")
    with open(latex_path, "w") as f:
        f.write("% Stratified analysis tables\n")
        f.write("% Generated by analyze_stratified.py\n\n")
        for i, tex in enumerate(all_latex):
            f.write(f"\n% Table {i+1}\n")
            f.write(tex)
            f.write("\n\n")
    print(f"\nLaTeX tables saved to {latex_path}")

    # ============================================================
    # Save CSV for further analysis
    # ============================================================
    csv_path = os.path.join(OUTPUT_DIR, "stratified_annotations.csv")
    with open(csv_path, "w") as f:
        header_fields = [
            "key", "repo", "issue_id", "n_gt_files",
            "lexical_overlap", "repo_size", "is_multifile",
            "expansion_cat", "edge_source",
            "gt_from_cochange_count", "gt_from_import_count",
        ]
        for c in cond_names:
            header_fields.append(f"r1_{cond_short[c]}")
        f.write(",".join(header_fields) + "\n")
        for key in valid_keys:
            ann = annotations[key]
            vals = [
                key, ann["repo"], str(ann["issue_id"]), str(ann["n_gt_files"]),
                f"{ann['lexical_overlap']:.4f}", str(ann["repo_size"]),
                str(int(ann["is_multifile"])),
                ann["expansion_cat"], ann["edge_source"],
                str(ann["gt_from_cochange_count"]),
                str(ann["gt_from_import_count"]),
            ]
            for c in cond_names:
                v = ann[f"r1_{c}"]
                vals.append(f"{v:.4f}" if not np.isnan(v) else "")
            f.write(",".join(vals) + "\n")
    print(f"CSV annotations saved to {csv_path}")


if __name__ == "__main__":
    main()
