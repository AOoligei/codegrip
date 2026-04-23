"""
Qualitative failure/success analysis for CodeGRIP paper.

Compares graph-expanded reranker (runB) vs BM25-only reranker (runA)
to categorize wins, losses, and ties with explanations.

Usage:
    python scripts/analyze_qualitative.py
"""

import json
import os
import random
from collections import Counter, defaultdict

random.seed(42)

# ---------- paths ----------
BASE = "/home/chenlibin/grepo_agent"
GRAPH_PRED = os.path.join(BASE, "experiments/rankft_runB_graph/eval_merged_rerank/predictions.jsonl")
BM25_PRED = os.path.join(BASE, "experiments/rankft_runA_bm25only/eval_bm25pool/predictions.jsonl")
TEST_DATA = os.path.join(BASE, "data/grepo_text/grepo_test.jsonl")
OUT_FILE = os.path.join(BASE, "experiments/analysis/qualitative_analysis.txt")

# ---------- evaluation cutoff ----------
# We compare accuracy at a fixed K for win/loss determination
K = 10  # use recall@10 as primary metric for win/loss


def load_jsonl(path):
    """Load JSONL file into list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def make_key(rec):
    return (rec["repo"], rec["issue_id"])


def recall_at_k(predicted, ground_truth, k):
    """Fraction of ground truth files found in top-k predictions."""
    if not ground_truth:
        return 0.0
    top_k = set(predicted[:k])
    hits = sum(1 for gt in ground_truth if gt in top_k)
    return hits / len(ground_truth)


def acc_at_k(predicted, ground_truth, k):
    """1.0 if ALL ground truth files are in top-k, else 0.0."""
    top_k = set(predicted[:k])
    return 1.0 if all(gt in top_k for gt in ground_truth) else 0.0


def any_hit_at_k(predicted, ground_truth, k):
    """1.0 if ANY ground truth file is in top-k, else 0.0."""
    top_k = set(predicted[:k])
    return 1.0 if any(gt in top_k for gt in ground_truth) else 0.0


def gt_rank(predicted, gt_file):
    """Return 1-indexed rank of gt_file in predicted list, or None if absent."""
    try:
        return predicted.index(gt_file) + 1
    except ValueError:
        return None


def _path_components(filepath):
    """Split a filepath into directory and filename."""
    parts = filepath.rsplit("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", parts[0]


def categorize_win(graph_rec, bm25_rec, gt_files):
    """
    Categorize why the graph pipeline won for this instance.
    Returns (category, detail_string).
    """
    graph_pred_set = set(graph_rec["predicted"])  # full top-50 from expanded pool
    bm25_pred_set = set(bm25_rec["predicted"])    # full top-50 from BM25 pool

    # bm25_original in the BM25 pipeline = the BM25-only retrieval pool top-K
    bm25_pool = set(bm25_rec["predicted"])  # entire BM25 reranked output as proxy for pool

    coverage_wins = []
    reranking_wins = []

    for gt in gt_files:
        g_rank = gt_rank(graph_rec["predicted"], gt)
        b_rank = gt_rank(bm25_rec["predicted"], gt)

        if g_rank is not None and g_rank <= K:
            # Graph got it in top-K
            if b_rank is None or b_rank > K:
                # BM25 didn't get it in top-K
                if gt not in bm25_pred_set:
                    # GT not even in BM25's top-50 reranked output -> coverage win
                    coverage_wins.append((gt, g_rank, b_rank))
                else:
                    # GT is in BM25's pool but ranked lower -> reranking win
                    reranking_wins.append((gt, g_rank, b_rank))

    # Subcategorize
    subcategory = _subcategorize_win(graph_rec, bm25_rec, gt_files, coverage_wins, reranking_wins)

    if coverage_wins and reranking_wins:
        category = "Coverage + Reranking win"
    elif coverage_wins:
        category = "Coverage win"
    elif reranking_wins:
        category = "Reranking win"
    else:
        category = "Other win"

    details = []
    for gt, gr, br in coverage_wins:
        details.append(f"  [Coverage] {gt}: graph_rank={gr}, bm25_rank={'absent' if br is None else br}")
    for gt, gr, br in reranking_wins:
        details.append(f"  [Reranking] {gt}: graph_rank={gr}, bm25_rank={br}")

    return category, subcategory, "\n".join(details)


def _subcategorize_win(graph_rec, bm25_rec, gt_files, coverage_wins, reranking_wins):
    """Further subcategorize wins."""
    subcats = []

    # Check for cross-directory patterns
    if len(gt_files) >= 1:
        gt_dirs = set(_path_components(f)[0] for f in gt_files)
        pred_dirs = set(_path_components(f)[0] for f in graph_rec["predicted"][:K])
        # If GT spans multiple directories and graph pipeline covers them
        if len(gt_dirs) > 1:
            subcats.append("cross-directory")

    # Check for same-filename disambiguation
    for gt in gt_files:
        _, gt_name = _path_components(gt)
        # Count how many files in the predicted list share the same filename
        same_name = [p for p in graph_rec["predicted"][:K] if _path_components(p)[1] == gt_name]
        if len(same_name) > 1:
            subcats.append("same-filename-disambiguation")
            break

    # Check keyword match -- if issue text has clear keyword matching GT path
    # (heuristic: any component of the GT path appears in the issue text of the test data)
    # We'll check this at the caller level since we need issue_text

    if not subcats:
        subcats.append("general")

    return ", ".join(subcats)


def categorize_loss(graph_rec, bm25_rec, gt_files):
    """
    Categorize why the graph pipeline lost for this instance.
    Returns (category, detail_string).
    """
    graph_pred_set = set(graph_rec["predicted"])
    bm25_pred_set = set(bm25_rec["predicted"])

    expansion_noise = []
    reranking_losses = []

    for gt in gt_files:
        g_rank = gt_rank(graph_rec["predicted"], gt)
        b_rank = gt_rank(bm25_rec["predicted"], gt)

        if b_rank is not None and b_rank <= K:
            # BM25 got it in top-K
            if g_rank is None or g_rank > K:
                # Graph didn't get it in top-K
                if gt not in graph_pred_set:
                    # GT not even in graph's top-50 output despite being in the 200-candidate pool
                    # This suggests expansion noise pushed it out
                    expansion_noise.append((gt, g_rank, b_rank))
                else:
                    # GT is in graph's output but ranked too low -> reranking failure
                    reranking_losses.append((gt, g_rank, b_rank))

    if expansion_noise and reranking_losses:
        category = "Expansion noise + Reranking loss"
    elif expansion_noise:
        category = "Expansion noise"
    elif reranking_losses:
        category = "Reranking loss"
    else:
        category = "Other loss"

    details = []
    for gt, gr, br in expansion_noise:
        details.append(f"  [Expansion noise] {gt}: graph_rank={'absent' if gr is None else gr}, bm25_rank={br}")
    for gt, gr, br in reranking_losses:
        details.append(f"  [Reranking loss] {gt}: graph_rank={gr}, bm25_rank={br}")

    return category, details


def compute_win_metric(graph_rec, bm25_rec):
    """
    Return a numeric 'margin' for sorting wins/losses.
    Positive = graph better, negative = BM25 better.
    """
    gt = graph_rec["ground_truth"]
    g_recall = recall_at_k(graph_rec["predicted"], gt, K)
    b_recall = recall_at_k(bm25_rec["predicted"], gt, K)
    return g_recall - b_recall


def main():
    # Load data
    graph_preds = load_jsonl(GRAPH_PRED)
    bm25_preds = load_jsonl(BM25_PRED)
    test_data = load_jsonl(TEST_DATA)

    # Index by (repo, issue_id)
    graph_by_key = {make_key(r): r for r in graph_preds}
    bm25_by_key = {make_key(r): r for r in bm25_preds}
    test_by_key = {make_key(r): r for r in test_data}

    # Find common keys (both pipelines evaluated on same instances)
    common_keys = sorted(set(graph_by_key.keys()) & set(bm25_by_key.keys()))
    print(f"Total common instances: {len(common_keys)}")

    wins = []     # graph better
    losses = []   # BM25 better
    ties = []     # same recall@K

    for key in common_keys:
        g = graph_by_key[key]
        b = bm25_by_key[key]
        gt = g["ground_truth"]

        g_recall = recall_at_k(g["predicted"], gt, K)
        b_recall = recall_at_k(b["predicted"], gt, K)

        margin = g_recall - b_recall

        if margin > 0:
            wins.append((key, margin))
        elif margin < 0:
            losses.append((key, margin))
        else:
            ties.append((key, 0.0))

    # Sort wins by margin descending, losses by margin ascending (most negative first)
    wins.sort(key=lambda x: -x[1])
    losses.sort(key=lambda x: x[1])

    # ---------- Categorize ----------
    win_categories = Counter()
    win_subcategories = Counter()
    loss_categories = Counter()

    win_details = []
    for key, margin in wins:
        g = graph_by_key[key]
        b = bm25_by_key[key]
        cat, subcat, detail = categorize_win(g, b, g["ground_truth"])
        win_categories[cat] += 1
        for sc in subcat.split(", "):
            win_subcategories[sc] += 1
        win_details.append((key, margin, cat, subcat, detail))

    loss_details = []
    for key, margin in losses:
        g = graph_by_key[key]
        b = bm25_by_key[key]
        cat, detail_list = categorize_loss(g, b, g["ground_truth"])
        loss_categories[cat] += 1
        loss_details.append((key, margin, cat, "\n".join(detail_list)))

    # ---------- Build keyword-match subcategory for wins ----------
    # Check if GT path components appear in issue text
    keyword_match_count = 0
    for key, margin, cat, subcat, detail in win_details:
        if key in test_by_key:
            issue_text = test_by_key[key].get("issue_text", "").lower()
            gt_files = graph_by_key[key]["ground_truth"]
            for gt in gt_files:
                parts = gt.replace("/", " ").replace("_", " ").replace(".py", "").lower().split()
                # Check if any meaningful path component (len > 3) appears in issue text
                if any(p in issue_text for p in parts if len(p) > 3):
                    keyword_match_count += 1
                    break

    # ---------- Per-repo breakdown ----------
    repo_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})
    for key, _ in wins:
        repo_stats[key[0]]["wins"] += 1
    for key, _ in losses:
        repo_stats[key[0]]["losses"] += 1
    for key, _ in ties:
        repo_stats[key[0]]["ties"] += 1

    # ---------- Output ----------
    lines = []
    lines.append("=" * 80)
    lines.append("QUALITATIVE ANALYSIS: CodeGRIP Graph Pipeline vs BM25-Only Pipeline")
    lines.append(f"Evaluation cutoff: recall@{K}")
    lines.append(f"Total instances compared: {len(common_keys)}")
    lines.append("=" * 80)

    # Summary statistics
    lines.append("")
    lines.append("-" * 60)
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 60)
    lines.append(f"  WINS  (graph > BM25):  {len(wins):4d}  ({100*len(wins)/len(common_keys):.1f}%)")
    lines.append(f"  LOSSES (graph < BM25): {len(losses):4d}  ({100*len(losses)/len(common_keys):.1f}%)")
    lines.append(f"  TIES   (graph = BM25): {len(ties):4d}  ({100*len(ties)/len(common_keys):.1f}%)")
    lines.append(f"  Net wins:              {len(wins) - len(losses):+4d}")
    lines.append("")

    # Win category breakdown
    lines.append("-" * 60)
    lines.append("WIN CATEGORY BREAKDOWN")
    lines.append("-" * 60)
    for cat, cnt in win_categories.most_common():
        lines.append(f"  {cat:40s}  {cnt:4d}  ({100*cnt/len(wins):.1f}%)")
    lines.append("")
    lines.append("  Win subcategories:")
    for sc, cnt in win_subcategories.most_common():
        lines.append(f"    {sc:38s}  {cnt:4d}  ({100*cnt/len(wins):.1f}%)")
    lines.append(f"    {'keyword-match (GT path in issue text)':38s}  {keyword_match_count:4d}  ({100*keyword_match_count/len(wins):.1f}%)")

    # Loss category breakdown
    lines.append("")
    lines.append("-" * 60)
    lines.append("LOSS CATEGORY BREAKDOWN")
    lines.append("-" * 60)
    for cat, cnt in loss_categories.most_common():
        lines.append(f"  {cat:40s}  {cnt:4d}  ({100*cnt/len(losses):.1f}%)")

    # Per-repo breakdown
    lines.append("")
    lines.append("-" * 60)
    lines.append("PER-REPO BREAKDOWN")
    lines.append("-" * 60)
    lines.append(f"  {'Repo':25s}  {'Wins':>5s}  {'Losses':>6s}  {'Ties':>5s}  {'Net':>5s}")
    for repo in sorted(repo_stats.keys()):
        s = repo_stats[repo]
        net = s["wins"] - s["losses"]
        lines.append(f"  {repo:25s}  {s['wins']:5d}  {s['losses']:6d}  {s['ties']:5d}  {net:+5d}")

    # ---------- Top 10 Wins ----------
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"TOP 10 WINS (Graph pipeline correct, BM25 pipeline wrong at recall@{K})")
    lines.append("=" * 80)

    for i, (key, margin, cat, subcat, detail) in enumerate(win_details[:10]):
        repo, issue_id = key
        g = graph_by_key[key]
        b = bm25_by_key[key]
        issue_text = test_by_key.get(key, {}).get("issue_text", "N/A")[:100]

        g_recall = recall_at_k(g["predicted"], g["ground_truth"], K)
        b_recall = recall_at_k(b["predicted"], g["ground_truth"], K)

        lines.append("")
        lines.append(f"--- Win #{i+1} (margin={margin:.2f}) ---")
        lines.append(f"  Repo:       {repo}")
        lines.append(f"  Issue ID:   {issue_id}")
        lines.append(f"  Issue text: {issue_text}...")
        lines.append(f"  GT files:   {g['ground_truth']}")
        lines.append(f"  Graph top-5 predictions: {g['predicted'][:5]}")
        lines.append(f"  BM25  top-5 predictions: {b['predicted'][:5]}")
        lines.append(f"  Graph recall@{K}: {g_recall:.2f}  |  BM25 recall@{K}: {b_recall:.2f}")
        lines.append(f"  Category:   {cat}")
        lines.append(f"  Subcategory: {subcat}")
        if detail:
            lines.append(f"  Details:")
            lines.append(detail)

    # ---------- Top 10 Losses ----------
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"TOP 10 LOSSES (BM25 pipeline correct, Graph pipeline wrong at recall@{K})")
    lines.append("=" * 80)

    for i, (key, margin, cat, detail_str) in enumerate(loss_details[:10]):
        repo, issue_id = key
        g = graph_by_key[key]
        b = bm25_by_key[key]
        issue_text = test_by_key.get(key, {}).get("issue_text", "N/A")[:100]

        g_recall = recall_at_k(g["predicted"], g["ground_truth"], K)
        b_recall = recall_at_k(b["predicted"], g["ground_truth"], K)

        lines.append("")
        lines.append(f"--- Loss #{i+1} (margin={margin:.2f}) ---")
        lines.append(f"  Repo:       {repo}")
        lines.append(f"  Issue ID:   {issue_id}")
        lines.append(f"  Issue text: {issue_text}...")
        lines.append(f"  GT files:   {g['ground_truth']}")
        lines.append(f"  Graph top-5 predictions: {g['predicted'][:5]}")
        lines.append(f"  BM25  top-5 predictions: {b['predicted'][:5]}")
        lines.append(f"  Graph recall@{K}: {g_recall:.2f}  |  BM25 recall@{K}: {b_recall:.2f}")
        lines.append(f"  Category:   {cat}")
        if detail_str:
            lines.append(f"  Details:")
            lines.append(detail_str)

    # ---------- Additional analysis: margin distribution ----------
    lines.append("")
    lines.append("=" * 80)
    lines.append("ADDITIONAL ANALYSIS")
    lines.append("=" * 80)

    # Average margin among wins/losses
    if wins:
        avg_win_margin = sum(m for _, m in wins) / len(wins)
        lines.append(f"  Average win margin:  {avg_win_margin:.4f}")
    if losses:
        avg_loss_margin = sum(abs(m) for _, m in losses) / len(losses)
        lines.append(f"  Average loss margin: {avg_loss_margin:.4f}")

    # GT-in-candidates analysis
    graph_gt_in_cand = sum(1 for k in common_keys if graph_by_key[k].get("gt_in_candidates", False))
    bm25_gt_in_cand = sum(1 for k in common_keys if bm25_by_key[k].get("gt_in_candidates", False))
    lines.append(f"  Graph GT-in-candidates: {graph_gt_in_cand}/{len(common_keys)} ({100*graph_gt_in_cand/len(common_keys):.1f}%)")
    lines.append(f"  BM25  GT-in-candidates: {bm25_gt_in_cand}/{len(common_keys)} ({100*bm25_gt_in_cand/len(common_keys):.1f}%)")

    # Coverage analysis: among wins, how many are pure coverage gains?
    pure_coverage = sum(1 for _, _, c, _, _ in win_details if c == "Coverage win")
    lines.append(f"  Pure coverage wins (GT not in BM25 top-50): {pure_coverage}/{len(wins)} ({100*pure_coverage/len(wins):.1f}%)")

    # Tie analysis: how many ties have both 0 recall (both wrong) vs both correct
    both_correct = 0
    both_wrong = 0
    for key, _ in ties:
        g = graph_by_key[key]
        b = bm25_by_key[key]
        gr = recall_at_k(g["predicted"], g["ground_truth"], K)
        if gr == 0.0:
            both_wrong += 1
        elif gr == 1.0:
            both_correct += 1

    lines.append(f"  Ties where both fully correct: {both_correct}/{len(ties)} ({100*both_correct/len(ties):.1f}%)")
    lines.append(f"  Ties where both fully wrong:   {both_wrong}/{len(ties)} ({100*both_wrong/len(ties):.1f}%)")

    # Multi-file analysis: win/loss rate by number of GT files
    lines.append("")
    lines.append("  Win/Loss rate by number of GT files:")
    gt_size_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0, "total": 0})
    for key, _ in wins:
        n = len(graph_by_key[key]["ground_truth"])
        bucket = "1" if n == 1 else ("2" if n == 2 else ("3-5" if n <= 5 else "6+"))
        gt_size_stats[bucket]["wins"] += 1
        gt_size_stats[bucket]["total"] += 1
    for key, _ in losses:
        n = len(graph_by_key[key]["ground_truth"])
        bucket = "1" if n == 1 else ("2" if n == 2 else ("3-5" if n <= 5 else "6+"))
        gt_size_stats[bucket]["losses"] += 1
        gt_size_stats[bucket]["total"] += 1
    for key, _ in ties:
        n = len(graph_by_key[key]["ground_truth"])
        bucket = "1" if n == 1 else ("2" if n == 2 else ("3-5" if n <= 5 else "6+"))
        gt_size_stats[bucket]["ties"] += 1
        gt_size_stats[bucket]["total"] += 1

    for bucket in ["1", "2", "3-5", "6+"]:
        if bucket in gt_size_stats:
            s = gt_size_stats[bucket]
            lines.append(f"    GT={bucket:4s}: {s['total']:4d} total | wins {s['wins']:4d} ({100*s['wins']/s['total']:.1f}%) | losses {s['losses']:4d} ({100*s['losses']/s['total']:.1f}%) | ties {s['ties']:4d}")

    # Write output
    output = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        f.write(output)

    print(output)
    print(f"\nWritten to: {OUT_FILE}")


if __name__ == "__main__":
    main()
