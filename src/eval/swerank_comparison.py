"""
SweRank Comparison Evaluation: Fair comparison between CodeGRIP and SweRank.

SweRank (Salesforce, 2025) is a state-of-the-art file-level localization model
achieving 83.21% file Acc@1 on SWE-bench Lite with its 32B model.

Key metric distinction:
- SweRank Acc@1 = fraction of instances where at least ONE ground truth file
  appears in the top-1 prediction. This differs from Hit@1 when multiple GT
  files exist, because Hit@1 = (# GT files in top-1) / (# GT files total).
- For instances with a single GT file, Acc@1 == Hit@1.

This script computes both metric families so comparisons are fair.

Usage:
    # Compare against SweRank reference numbers (no SweRank predictions file)
    python src/eval/swerank_comparison.py \
        --codegrip_predictions experiments/exp1_sft_only/eval_swebench_lite/predictions.jsonl \
        --test_data data/swebench_lite/swebench_lite_test.jsonl \
        --output_dir experiments/swerank_comparison \
        --benchmark swebench_lite

    # Compare with actual SweRank predictions (if available)
    python src/eval/swerank_comparison.py \
        --codegrip_predictions experiments/exp1_sft_only/eval_swebench_lite/predictions.jsonl \
        --swerank_predictions path/to/swerank_preds.jsonl \
        --test_data data/swebench_lite/swebench_lite_test.jsonl \
        --output_dir experiments/swerank_comparison \
        --benchmark swebench_lite

    # Compare on GREPO benchmark
    python src/eval/swerank_comparison.py \
        --codegrip_predictions experiments/exp1_sft_only/eval_reranked/predictions.jsonl \
        --test_data data/grepo_text/grepo_test.jsonl \
        --output_dir experiments/swerank_comparison_grepo \
        --benchmark grepo
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Reference metrics from published papers (for display in comparison tables)
# ---------------------------------------------------------------------------

SWERANK_REFERENCE = {
    "swebench_lite": {
        "SweRank-32B": {"acc@1": 83.21, "source": "SweRank paper Table 2"},
        "SweRank-7B":  {"acc@1": 66.0,  "source": "SweRank paper Table 2 (approx)"},
        "SweRank-2B":  {"acc@1": 55.0,  "source": "SweRank paper Table 2 (approx)"},
        "Agentless":   {"acc@1": 53.3,  "source": "SweRank paper Table 2"},
        "AutoCodeRover": {"acc@1": 48.0, "source": "SweRank paper Table 2"},
    },
    "grepo": {
        # No published SweRank numbers on GREPO; we leave this empty.
    },
}


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_hit_at_k(predicted: List[str], gt_set: Set[str], k: int) -> float:
    """Hit@K = fraction of ground truth files found in top-K predictions.

    This is the GREPO primary metric: |top_k & GT| / |GT|.
    """
    if not gt_set:
        return 0.0
    topk = set(predicted[:k])
    return len(topk & gt_set) / len(gt_set)


def compute_acc_at_k(predicted: List[str], gt_set: Set[str], k: int) -> float:
    """Acc@K (SweRank metric) = 1 if at least ONE GT file is in top-K, else 0.

    Binary per-instance: does the top-K contain ANY correct file?
    """
    if not gt_set:
        return 0.0
    topk = set(predicted[:k])
    return 1.0 if len(topk & gt_set) > 0 else 0.0


def compute_mrr(predicted: List[str], gt_set: Set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first correct prediction."""
    for i, f in enumerate(predicted):
        if f in gt_set:
            return 1.0 / (i + 1)
    return 0.0


def compute_ap(predicted: List[str], gt_set: Set[str]) -> float:
    """Average Precision for a single query.

    AP = (1/|GT|) * sum_{k=1}^{n} P(k) * rel(k)
    where P(k) = precision at cutoff k, rel(k) = 1 if predicted[k] in GT.
    """
    if not gt_set:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, f in enumerate(predicted):
        if f in gt_set:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(gt_set)


def compute_all_metrics(
    predicted: List[str],
    gt_set: Set[str],
    k_values: List[int],
) -> Dict[str, float]:
    """Compute all metrics for a single instance."""
    metrics = {}
    for k in k_values:
        metrics[f"hit@{k}"] = compute_hit_at_k(predicted, gt_set, k)
        metrics[f"acc@{k}"] = compute_acc_at_k(predicted, gt_set, k)
    metrics["mrr"] = compute_mrr(predicted, gt_set)
    metrics["ap"] = compute_ap(predicted, gt_set)
    return metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions(path: str) -> Dict[Tuple[str, str], dict]:
    """Load predictions JSONL. Key = (repo, issue_id)."""
    preds = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            # issue_id may be int or string depending on benchmark
            key = (p["repo"], str(p["issue_id"]))
            preds[key] = p
    return preds


def load_test_data(path: str) -> Dict[Tuple[str, str], dict]:
    """Load test data JSONL for ground truth."""
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not item.get("changed_py_files"):
                continue
            key = (item["repo"], str(item["issue_id"]))
            data[key] = item
    return data


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate_metrics(
    per_instance: List[Dict[str, float]],
    k_values: List[int],
) -> Dict[str, float]:
    """Aggregate per-instance metrics into macro-averaged scores (as %)."""
    if not per_instance:
        return {}

    agg = {}
    for k in k_values:
        vals_hit = [m[f"hit@{k}"] for m in per_instance]
        vals_acc = [m[f"acc@{k}"] for m in per_instance]
        agg[f"hit@{k}"] = 100.0 * sum(vals_hit) / len(vals_hit)
        agg[f"acc@{k}"] = 100.0 * sum(vals_acc) / len(vals_acc)

    mrr_vals = [m["mrr"] for m in per_instance]
    ap_vals = [m["ap"] for m in per_instance]
    agg["mrr"] = 100.0 * sum(mrr_vals) / len(mrr_vals)
    agg["map"] = 100.0 * sum(ap_vals) / len(ap_vals)
    agg["n"] = len(per_instance)
    return agg


def per_repo_metrics(
    predictions: Dict[Tuple[str, str], dict],
    test_data: Dict[Tuple[str, str], dict],
    k_values: List[int],
) -> Dict[str, Dict[str, float]]:
    """Compute metrics broken down by repository."""
    repo_instances: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for key, test_item in test_data.items():
        pred_item = predictions.get(key)
        if pred_item is None:
            continue
        gt_set = set(test_item["changed_py_files"])
        predicted = pred_item.get("predicted", [])
        m = compute_all_metrics(predicted, gt_set, k_values)
        repo_instances[test_item["repo"]].append(m)

    repo_agg = {}
    for repo, instances in sorted(repo_instances.items()):
        repo_agg[repo] = aggregate_metrics(instances, k_values)
    return repo_agg


# ---------------------------------------------------------------------------
# Comparison analysis
# ---------------------------------------------------------------------------

def compare_systems(
    codegrip_preds: Dict[Tuple[str, str], dict],
    test_data: Dict[Tuple[str, str], dict],
    k_values: List[int],
    swerank_preds: Optional[Dict[Tuple[str, str], dict]] = None,
) -> Dict[str, Any]:
    """Run full comparison between CodeGRIP and SweRank.

    Returns structured results dict.
    """
    # --- CodeGRIP metrics ---
    codegrip_instances = []
    for key, test_item in test_data.items():
        pred_item = codegrip_preds.get(key)
        if pred_item is None:
            continue
        gt_set = set(test_item["changed_py_files"])
        predicted = pred_item.get("predicted", [])
        m = compute_all_metrics(predicted, gt_set, k_values)
        m["repo"] = test_item["repo"]
        m["issue_id"] = str(test_item["issue_id"])
        m["n_gt_files"] = len(gt_set)
        codegrip_instances.append(m)

    codegrip_overall = aggregate_metrics(codegrip_instances, k_values)

    # --- SweRank metrics (if predictions available) ---
    swerank_overall = None
    swerank_instances = []
    if swerank_preds is not None:
        for key, test_item in test_data.items():
            pred_item = swerank_preds.get(key)
            if pred_item is None:
                continue
            gt_set = set(test_item["changed_py_files"])
            predicted = pred_item.get("predicted", [])
            m = compute_all_metrics(predicted, gt_set, k_values)
            m["repo"] = test_item["repo"]
            m["issue_id"] = str(test_item["issue_id"])
            m["n_gt_files"] = len(gt_set)
            swerank_instances.append(m)
        swerank_overall = aggregate_metrics(swerank_instances, k_values)

    return {
        "codegrip_overall": codegrip_overall,
        "codegrip_instances": codegrip_instances,
        "swerank_overall": swerank_overall,
        "swerank_instances": swerank_instances,
    }


def identify_win_loss(
    codegrip_preds: Dict[Tuple[str, str], dict],
    swerank_preds: Dict[Tuple[str, str], dict],
    test_data: Dict[Tuple[str, str], dict],
    k_values: List[int],
    primary_k: int = 1,
) -> Dict[str, List[dict]]:
    """Identify instances where CodeGRIP wins, loses, or ties vs SweRank.

    Comparison based on acc@{primary_k} (the SweRank metric).
    """
    wins = []
    losses = []
    ties = []

    for key, test_item in test_data.items():
        cg = codegrip_preds.get(key)
        sw = swerank_preds.get(key)
        if cg is None or sw is None:
            continue

        gt_set = set(test_item["changed_py_files"])
        cg_acc = compute_acc_at_k(cg.get("predicted", []), gt_set, primary_k)
        sw_acc = compute_acc_at_k(sw.get("predicted", []), gt_set, primary_k)

        record = {
            "repo": test_item["repo"],
            "issue_id": str(test_item["issue_id"]),
            "codegrip_acc": cg_acc,
            "swerank_acc": sw_acc,
            "codegrip_top1": cg.get("predicted", [""])[0] if cg.get("predicted") else "",
            "swerank_top1": sw.get("predicted", [""])[0] if sw.get("predicted") else "",
            "ground_truth": list(gt_set),
        }

        if cg_acc > sw_acc:
            wins.append(record)
        elif cg_acc < sw_acc:
            losses.append(record)
        else:
            ties.append(record)

    return {"wins": wins, "losses": losses, "ties": ties}


def category_breakdown(
    codegrip_preds: Dict[Tuple[str, str], dict],
    test_data: Dict[Tuple[str, str], dict],
    k_values: List[int],
) -> Dict[str, Dict[str, float]]:
    """Break down CodeGRIP performance by instance categories.

    Categories:
    - single_file: instances with exactly 1 GT file
    - multi_file: instances with 2+ GT files
    - test_involved: GT includes test files
    - source_only: GT is exclusively source (non-test) files
    """
    categories: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for key, test_item in test_data.items():
        pred_item = codegrip_preds.get(key)
        if pred_item is None:
            continue

        gt_files = test_item["changed_py_files"]
        gt_set = set(gt_files)
        predicted = pred_item.get("predicted", [])
        m = compute_all_metrics(predicted, gt_set, k_values)

        # single vs multi file
        if len(gt_files) == 1:
            categories["single_file"].append(m)
        else:
            categories["multi_file"].append(m)

        # test involvement
        has_test = any(
            "test" in f.lower() or f.lower().endswith("_test.py")
            for f in gt_files
        )
        if has_test:
            categories["test_involved"].append(m)
        else:
            categories["source_only"].append(m)

    return {
        cat: aggregate_metrics(instances, k_values)
        for cat, instances in categories.items()
    }


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def generate_markdown_table(
    codegrip_overall: Dict[str, float],
    benchmark: str,
    k_values: List[int],
    swerank_overall: Optional[Dict[str, float]] = None,
    codegrip_repo_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    category_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    win_loss: Optional[Dict[str, List[dict]]] = None,
) -> str:
    """Generate comprehensive Markdown comparison report."""
    lines = []

    lines.append("# CodeGRIP vs SweRank: File-Level Localization Comparison")
    lines.append("")
    lines.append(f"Benchmark: **{benchmark}**")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ---- Section 1: Primary metric comparison ----
    lines.append("## 1. Primary Metrics Comparison")
    lines.append("")
    lines.append("**Acc@K** = SweRank metric (at least one GT file in top-K; binary per instance).")
    lines.append("**Hit@K** = GREPO metric (fraction of GT files found in top-K).")
    lines.append("When there is only one GT file per instance, Acc@1 == Hit@1.")
    lines.append("")

    # Build header
    metric_cols = []
    for k in k_values:
        metric_cols.append(f"Acc@{k}")
    for k in k_values:
        metric_cols.append(f"Hit@{k}")
    metric_cols.extend(["MRR", "MAP"])

    header = "| Method | " + " | ".join(metric_cols) + " | N |"
    separator = "|" + "|".join(["-" * max(len(c) + 2, 8) for c in ["Method"] + metric_cols + ["N"]]) + "|"
    lines.append(header)
    lines.append(separator)

    # CodeGRIP row
    cg = codegrip_overall
    cg_vals = []
    for k in k_values:
        cg_vals.append(f"{cg.get(f'acc@{k}', 0):.2f}")
    for k in k_values:
        cg_vals.append(f"{cg.get(f'hit@{k}', 0):.2f}")
    cg_vals.append(f"{cg.get('mrr', 0):.2f}")
    cg_vals.append(f"{cg.get('map', 0):.2f}")
    cg_vals.append(str(int(cg.get("n", 0))))
    lines.append("| **CodeGRIP (ours)** | " + " | ".join(cg_vals) + " |")

    # SweRank row (from actual predictions)
    if swerank_overall is not None:
        sw = swerank_overall
        sw_vals = []
        for k in k_values:
            sw_vals.append(f"{sw.get(f'acc@{k}', 0):.2f}")
        for k in k_values:
            sw_vals.append(f"{sw.get(f'hit@{k}', 0):.2f}")
        sw_vals.append(f"{sw.get('mrr', 0):.2f}")
        sw_vals.append(f"{sw.get('map', 0):.2f}")
        sw_vals.append(str(int(sw.get("n", 0))))
        lines.append("| SweRank (preds) | " + " | ".join(sw_vals) + " |")

    # Reference rows (published numbers, Acc@1 only)
    ref = SWERANK_REFERENCE.get(benchmark, {})
    for method, ref_metrics in ref.items():
        ref_vals = []
        for k in k_values:
            if k == 1 and "acc@1" in ref_metrics:
                ref_vals.append(f"{ref_metrics['acc@1']:.2f}")
            else:
                ref_vals.append("-")
        for k in k_values:
            ref_vals.append("-")  # no Hit@K published
        ref_vals.append("-")  # MRR
        ref_vals.append("-")  # MAP
        ref_vals.append("-")  # N
        lines.append(f"| {method} (ref) | " + " | ".join(ref_vals) + " |")

    lines.append("")

    # ---- Section 2: Acc@1 vs Hit@1 analysis ----
    lines.append("## 2. Acc@1 vs Hit@1 Analysis (metric gap)")
    lines.append("")
    lines.append("Acc@1 >= Hit@1 always holds. The gap increases when instances have multiple GT files.")
    lines.append("")
    acc1 = cg.get("acc@1", 0)
    hit1 = cg.get("hit@1", 0)
    lines.append(f"- CodeGRIP Acc@1: **{acc1:.2f}%**")
    lines.append(f"- CodeGRIP Hit@1: **{hit1:.2f}%**")
    lines.append(f"- Gap (Acc@1 - Hit@1): **{acc1 - hit1:.2f}pp**")
    lines.append("")

    # ---- Section 3: Category breakdown ----
    if category_metrics:
        lines.append("## 3. Performance by Instance Category")
        lines.append("")
        cat_header = "| Category | " + " | ".join([f"Acc@{k}" for k in k_values]) + " | " + " | ".join([f"Hit@{k}" for k in k_values]) + " | MRR | N |"
        lines.append(cat_header)
        cat_sep = "|" + "|".join(["-" * 14] * (1 + 2 * len(k_values) + 2)) + "|"
        lines.append(cat_sep)

        for cat_name, cat_m in category_metrics.items():
            vals = []
            for k in k_values:
                vals.append(f"{cat_m.get(f'acc@{k}', 0):.2f}")
            for k in k_values:
                vals.append(f"{cat_m.get(f'hit@{k}', 0):.2f}")
            vals.append(f"{cat_m.get('mrr', 0):.2f}")
            vals.append(str(int(cat_m.get("n", 0))))
            lines.append(f"| {cat_name} | " + " | ".join(vals) + " |")
        lines.append("")

    # ---- Section 4: Per-repo breakdown ----
    if codegrip_repo_metrics:
        lines.append("## 4. Per-Repository Breakdown (CodeGRIP)")
        lines.append("")
        repo_header = "| Repository | Acc@1 | Acc@5 | Hit@1 | Hit@5 | MRR | N |"
        lines.append(repo_header)
        lines.append("|" + "|".join(["-" * 14] * 7) + "|")

        # Sort by Acc@1 descending
        sorted_repos = sorted(
            codegrip_repo_metrics.items(),
            key=lambda x: x[1].get("acc@1", 0),
            reverse=True,
        )
        for repo, rm in sorted_repos:
            lines.append(
                f"| {repo} | "
                f"{rm.get('acc@1', 0):.2f} | "
                f"{rm.get('acc@5', 0):.2f} | "
                f"{rm.get('hit@1', 0):.2f} | "
                f"{rm.get('hit@5', 0):.2f} | "
                f"{rm.get('mrr', 0):.2f} | "
                f"{int(rm.get('n', 0))} |"
            )
        lines.append("")

    # ---- Section 5: Win/loss analysis ----
    if win_loss is not None:
        n_wins = len(win_loss["wins"])
        n_losses = len(win_loss["losses"])
        n_ties = len(win_loss["ties"])
        total = n_wins + n_losses + n_ties

        lines.append("## 5. Head-to-Head: CodeGRIP vs SweRank (Acc@1)")
        lines.append("")
        lines.append(f"- CodeGRIP wins: **{n_wins}** ({100 * n_wins / total:.1f}%)" if total > 0 else "- CodeGRIP wins: 0")
        lines.append(f"- SweRank wins:  **{n_losses}** ({100 * n_losses / total:.1f}%)" if total > 0 else "- SweRank wins: 0")
        lines.append(f"- Ties:          **{n_ties}** ({100 * n_ties / total:.1f}%)" if total > 0 else "- Ties: 0")
        lines.append("")

        # Repo-level win/loss aggregation
        if n_wins > 0 or n_losses > 0:
            win_repos = defaultdict(int)
            loss_repos = defaultdict(int)
            for rec in win_loss["wins"]:
                win_repos[rec["repo"]] += 1
            for rec in win_loss["losses"]:
                loss_repos[rec["repo"]] += 1

            lines.append("### Win/Loss by Repository")
            lines.append("")
            all_repos_wl = sorted(
                set(win_repos.keys()) | set(loss_repos.keys()),
                key=lambda r: win_repos.get(r, 0) - loss_repos.get(r, 0),
                reverse=True,
            )
            lines.append("| Repository | CodeGRIP wins | SweRank wins | Net |")
            lines.append("|" + "|".join(["-" * 18] * 4) + "|")
            for repo in all_repos_wl:
                w = win_repos.get(repo, 0)
                l = loss_repos.get(repo, 0)
                lines.append(f"| {repo} | {w} | {l} | {w - l:+d} |")
            lines.append("")

    # ---- Section 6: Strongest / weakest repos ----
    if codegrip_repo_metrics:
        sorted_by_acc1 = sorted(
            codegrip_repo_metrics.items(),
            key=lambda x: x[1].get("acc@1", 0),
            reverse=True,
        )
        n_repos = len(sorted_by_acc1)
        top_n = min(5, n_repos)
        bot_n = min(5, n_repos)

        lines.append("## 6. Strongest and Weakest Repositories")
        lines.append("")
        lines.append(f"### Top {top_n} (by Acc@1)")
        lines.append("")
        for repo, rm in sorted_by_acc1[:top_n]:
            lines.append(
                f"- **{repo}**: Acc@1={rm.get('acc@1', 0):.1f}%, "
                f"Hit@5={rm.get('hit@5', 0):.1f}%, N={int(rm.get('n', 0))}"
            )
        lines.append("")
        lines.append(f"### Bottom {bot_n} (by Acc@1)")
        lines.append("")
        for repo, rm in sorted_by_acc1[-bot_n:]:
            lines.append(
                f"- **{repo}**: Acc@1={rm.get('acc@1', 0):.1f}%, "
                f"Hit@5={rm.get('hit@5', 0):.1f}%, N={int(rm.get('n', 0))}"
            )
        lines.append("")

    return "\n".join(lines)


def build_structured_results(
    codegrip_overall: Dict[str, float],
    benchmark: str,
    k_values: List[int],
    codegrip_repo_metrics: Dict[str, Dict[str, float]],
    category_metrics: Dict[str, Dict[str, float]],
    swerank_overall: Optional[Dict[str, float]] = None,
    win_loss: Optional[Dict[str, List[dict]]] = None,
    n_test: int = 0,
    n_codegrip: int = 0,
    n_swerank: int = 0,
) -> Dict[str, Any]:
    """Build structured JSON results for programmatic consumption."""
    result: Dict[str, Any] = {
        "metadata": {
            "benchmark": benchmark,
            "timestamp": datetime.now().isoformat(),
            "n_test_instances": n_test,
            "n_codegrip_predictions": n_codegrip,
            "n_swerank_predictions": n_swerank,
            "k_values": k_values,
            "metric_definitions": {
                "acc@k": "Fraction of instances with at least one GT file in top-K (SweRank metric)",
                "hit@k": "Average fraction of GT files found in top-K (GREPO metric)",
                "mrr": "Mean Reciprocal Rank (1/rank of first correct file)",
                "map": "Mean Average Precision",
            },
        },
        "codegrip": {
            "overall": codegrip_overall,
            "per_repo": codegrip_repo_metrics,
            "by_category": category_metrics,
        },
        "reference_baselines": SWERANK_REFERENCE.get(benchmark, {}),
    }

    if swerank_overall is not None:
        result["swerank"] = {
            "overall": swerank_overall,
        }

    if win_loss is not None:
        result["head_to_head"] = {
            "codegrip_wins": len(win_loss["wins"]),
            "swerank_wins": len(win_loss["losses"]),
            "ties": len(win_loss["ties"]),
        }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare CodeGRIP file-level localization against SweRank."
    )
    parser.add_argument(
        "--codegrip_predictions", required=True,
        help="Path to CodeGRIP predictions JSONL",
    )
    parser.add_argument(
        "--swerank_predictions", default=None,
        help="Path to SweRank predictions JSONL (optional; if not provided, "
             "comparison uses published reference numbers only)",
    )
    parser.add_argument(
        "--test_data", required=True,
        help="Path to test data JSONL (ground truth)",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory for output files",
    )
    parser.add_argument(
        "--benchmark", default="grepo",
        choices=["grepo", "swebench_lite"],
        help="Which benchmark (affects reference numbers and field mapping)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    k_values = [1, 3, 5, 10, 20]

    # --- Load data ---
    print(f"Loading test data from {args.test_data}...")
    test_data = load_test_data(args.test_data)
    print(f"  {len(test_data)} instances with Python file changes")

    print(f"Loading CodeGRIP predictions from {args.codegrip_predictions}...")
    codegrip_preds = load_predictions(args.codegrip_predictions)
    print(f"  {len(codegrip_preds)} predictions loaded")

    swerank_preds = None
    if args.swerank_predictions:
        print(f"Loading SweRank predictions from {args.swerank_predictions}...")
        swerank_preds = load_predictions(args.swerank_predictions)
        print(f"  {len(swerank_preds)} predictions loaded")

    # Check overlap
    common_keys = set(test_data.keys()) & set(codegrip_preds.keys())
    print(f"  Instances with both GT and CodeGRIP predictions: {len(common_keys)}")
    if swerank_preds:
        common_sw = set(test_data.keys()) & set(swerank_preds.keys())
        print(f"  Instances with both GT and SweRank predictions: {len(common_sw)}")

    # --- Compute metrics ---
    print("\nComputing metrics...")
    comparison = compare_systems(
        codegrip_preds, test_data, k_values, swerank_preds
    )

    codegrip_overall = comparison["codegrip_overall"]
    swerank_overall = comparison["swerank_overall"]

    # Per-repo
    print("Computing per-repo breakdown...")
    codegrip_repo = per_repo_metrics(codegrip_preds, test_data, k_values)

    # Category breakdown
    print("Computing category breakdown...")
    cat_metrics = category_breakdown(codegrip_preds, test_data, k_values)

    # Win/loss (only if SweRank predictions available)
    win_loss = None
    if swerank_preds:
        print("Computing head-to-head win/loss...")
        win_loss = identify_win_loss(
            codegrip_preds, swerank_preds, test_data, k_values, primary_k=1
        )

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print(f"\nCodeGRIP ({args.benchmark}):")
    for k in k_values:
        acc = codegrip_overall.get(f"acc@{k}", 0)
        hit = codegrip_overall.get(f"hit@{k}", 0)
        print(f"  Acc@{k}: {acc:6.2f}%   Hit@{k}: {hit:6.2f}%")
    print(f"  MRR:   {codegrip_overall.get('mrr', 0):6.2f}%")
    print(f"  MAP:   {codegrip_overall.get('map', 0):6.2f}%")
    print(f"  N:     {int(codegrip_overall.get('n', 0))}")

    if swerank_overall:
        print(f"\nSweRank ({args.benchmark}):")
        for k in k_values:
            acc = swerank_overall.get(f"acc@{k}", 0)
            hit = swerank_overall.get(f"hit@{k}", 0)
            print(f"  Acc@{k}: {acc:6.2f}%   Hit@{k}: {hit:6.2f}%")
        print(f"  MRR:   {swerank_overall.get('mrr', 0):6.2f}%")
        print(f"  MAP:   {swerank_overall.get('map', 0):6.2f}%")

    # Published reference comparison
    ref = SWERANK_REFERENCE.get(args.benchmark, {})
    if ref:
        print("\nPublished reference numbers (Acc@1 on SWE-bench Lite):")
        our_acc1 = codegrip_overall.get("acc@1", 0)
        for method, ref_m in sorted(ref.items(), key=lambda x: -x[1].get("acc@1", 0)):
            ref_acc1 = ref_m["acc@1"]
            delta = our_acc1 - ref_acc1
            print(f"  {method:20s}: {ref_acc1:6.2f}%  (CodeGRIP delta: {delta:+.2f}pp)")

    if win_loss:
        print(f"\nHead-to-head (Acc@1):")
        print(f"  CodeGRIP wins: {len(win_loss['wins'])}")
        print(f"  SweRank wins:  {len(win_loss['losses'])}")
        print(f"  Ties:          {len(win_loss['ties'])}")

    # --- Generate outputs ---
    # Markdown report
    md_report = generate_markdown_table(
        codegrip_overall=codegrip_overall,
        benchmark=args.benchmark,
        k_values=k_values,
        swerank_overall=swerank_overall,
        codegrip_repo_metrics=codegrip_repo,
        category_metrics=cat_metrics,
        win_loss=win_loss,
    )
    md_path = os.path.join(args.output_dir, "comparison_report.md")
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"\nMarkdown report: {md_path}")

    # Structured JSON
    structured = build_structured_results(
        codegrip_overall=codegrip_overall,
        benchmark=args.benchmark,
        k_values=k_values,
        codegrip_repo_metrics=codegrip_repo,
        category_metrics=cat_metrics,
        swerank_overall=swerank_overall,
        win_loss=win_loss,
        n_test=len(test_data),
        n_codegrip=len(codegrip_preds),
        n_swerank=len(swerank_preds) if swerank_preds else 0,
    )
    json_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)
    print(f"JSON results:    {json_path}")

    # Per-instance details (for downstream analysis)
    details_path = os.path.join(args.output_dir, "per_instance_metrics.jsonl")
    with open(details_path, "w") as f:
        for m in comparison["codegrip_instances"]:
            record = {
                "system": "codegrip",
                "repo": m["repo"],
                "issue_id": m["issue_id"],
                "n_gt_files": m["n_gt_files"],
            }
            for k in k_values:
                record[f"acc@{k}"] = m[f"acc@{k}"]
                record[f"hit@{k}"] = m[f"hit@{k}"]
            record["mrr"] = m["mrr"]
            record["ap"] = m["ap"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        for m in comparison["swerank_instances"]:
            record = {
                "system": "swerank",
                "repo": m["repo"],
                "issue_id": m["issue_id"],
                "n_gt_files": m["n_gt_files"],
            }
            for k in k_values:
                record[f"acc@{k}"] = m[f"acc@{k}"]
                record[f"hit@{k}"] = m[f"hit@{k}"]
            record["mrr"] = m["mrr"]
            record["ap"] = m["ap"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Instance details: {details_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
