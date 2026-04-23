#!/usr/bin/env python3
"""
Build path-controlled challenge set for CodeGRIP.

Three confounder types:
  1. Same-directory confounders: GT and distractors share directory
  2. Issue text de-leaked: mask filename mentions from issue text
  3. Path-misleading hard cases: wrong predictions with higher path overlap than GT

Usage:
    python scripts/build_path_controlled_challenge.py
"""

import json
import os
import re
from collections import defaultdict
from pathlib import PurePosixPath

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PATH = os.path.join(ROOT, "data/grepo_text/grepo_test.jsonl")
CANDIDATES_PATH = os.path.join(ROOT, "data/rankft/merged_bm25_exp6_candidates.jsonl")
PREDICTIONS_PATH = os.path.join(
    ROOT, "experiments/rankft_runB_graph/eval_merged_rerank/predictions.jsonl"
)
OUTPUT_DIR = os.path.join(ROOT, "data/grepo_text")
CHALLENGE_DIR = os.path.join(ROOT, "data/path_controlled_challenge")


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def make_key(item):
    return (item["repo"], item["issue_id"])


# ---------------------------------------------------------------------------
# Type 1: Same-directory confounders
# ---------------------------------------------------------------------------
def build_same_directory_confounders(test_data, candidates_map, predictions_map):
    """Find examples where GT file has >= 3 same-directory candidates in top-200.

    Also identifies the refined subset: wrong predictions where the top-1
    predicted file is in the SAME directory as a GT file (true same-dir confusion).
    """
    results = []
    for item in test_data:
        key = make_key(item)
        if key not in candidates_map:
            continue
        cands = candidates_map[key]
        gt_files = set(item["changed_files"])

        # For each GT file, count candidates in the same directory
        for gt_file in gt_files:
            gt_dir = str(PurePosixPath(gt_file).parent)
            same_dir_cands = [
                c for c in cands if str(PurePosixPath(c).parent) == gt_dir and c != gt_file
            ]
            if len(same_dir_cands) >= 3:
                results.append(
                    {
                        "repo": item["repo"],
                        "issue_id": item["issue_id"],
                        "gt_file": gt_file,
                        "gt_dir": gt_dir,
                        "same_dir_candidates": same_dir_cands[:20],
                        "num_same_dir": len(same_dir_cands),
                    }
                )
                break  # one match per example is enough

    # Compute R@1 on this subset vs full set using predictions
    subset_keys = {(r["repo"], r["issue_id"]) for r in results}
    r1_subset = _compute_r1(predictions_map, subset_keys)
    r1_full = _compute_r1(predictions_map, set(predictions_map.keys()))

    # Refined: wrong predictions where top-1 is in same dir as GT
    same_dir_confused = []
    for pred in predictions_map.values():
        gt_set = set(pred["ground_truth"])
        predicted = pred["predicted"]
        if not predicted:
            continue
        top1 = predicted[0]
        if top1 in gt_set:
            continue
        pred_dir = str(PurePosixPath(top1).parent)
        for gt_file in gt_set:
            gt_dir = str(PurePosixPath(gt_file).parent)
            if gt_dir == pred_dir:
                same_dir_confused.append(
                    {
                        "repo": pred["repo"],
                        "issue_id": pred["issue_id"],
                        "gt_file": gt_file,
                        "predicted_top1": top1,
                        "shared_dir": gt_dir,
                    }
                )
                break

    return results, r1_subset, r1_full, same_dir_confused


def _compute_r1(predictions_map, key_set):
    """Compute Recall@1 for a set of (repo, issue_id) keys."""
    hits = 0
    total = 0
    for key in key_set:
        if key not in predictions_map:
            continue
        pred = predictions_map[key]
        gt = set(pred["ground_truth"])
        if pred["predicted"] and pred["predicted"][0] in gt:
            hits += 1
        total += 1
    return hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Type 2: Issue text de-leaked
# ---------------------------------------------------------------------------
def build_deleaked(test_data):
    """Mask filename mentions in issue_text."""
    deleaked = []
    stats = {"total": 0, "leaked_examples": 0, "total_replacements": 0}

    for item in test_data:
        stats["total"] += 1
        issue = item["issue_text"]
        gt_files = item["changed_files"]
        num_replacements = 0
        leaked = False

        # Build patterns from GT filenames: full path, filename, stem
        patterns = set()
        for f in gt_files:
            p = PurePosixPath(f)
            patterns.add(f)  # full path
            patterns.add(p.name)  # filename with extension
            stem = p.stem
            if len(stem) >= 3:  # avoid very short stems like "a" matching everywhere
                patterns.add(stem)

        # Sort by length descending so we replace longer patterns first
        sorted_patterns = sorted(patterns, key=len, reverse=True)

        new_issue = issue
        for pat in sorted_patterns:
            # Use word-boundary-aware replacement to avoid partial matches
            # Escape for regex
            escaped = re.escape(pat)
            regex = re.compile(escaped, re.IGNORECASE)
            matches = regex.findall(new_issue)
            if matches:
                leaked = True
                num_replacements += len(matches)
                new_issue = regex.sub("[FILE]", new_issue)

        if leaked:
            stats["leaked_examples"] += 1
            stats["total_replacements"] += num_replacements

        new_item = dict(item)
        new_item["issue_text"] = new_issue
        new_item["_original_issue_text"] = issue
        new_item["_filename_leaked"] = leaked
        new_item["_num_replacements"] = num_replacements
        deleaked.append(new_item)

    return deleaked, stats


# ---------------------------------------------------------------------------
# Type 3: Path-misleading hard cases
# ---------------------------------------------------------------------------
def path_tokens(filepath):
    """Tokenize a file path into meaningful tokens."""
    # Split by / and then by _ and .
    parts = filepath.replace("\\", "/").split("/")
    tokens = set()
    for part in parts:
        tokens.add(part.lower())
        # Also split by _ and .
        for sub in re.split(r"[_.\-]", part):
            if len(sub) >= 2:
                tokens.add(sub.lower())
    return tokens


def issue_path_tokens(issue_text):
    """Extract path-like tokens from issue text."""
    # Tokenize: split on whitespace, then extract path-like components
    words = re.split(r"[\s,;:\"\'`\(\)\[\]\{\}]+", issue_text.lower())
    tokens = set()
    for w in words:
        for sub in re.split(r"[_.\-/\\]", w):
            if len(sub) >= 2:
                tokens.add(sub)
    return tokens


def build_path_misleading(predictions_data):
    """Find wrong predictions where predicted has more path overlap with issue than GT."""
    # We need issue text - load test data keyed
    test_data = load_jsonl(TEST_PATH)
    test_map = {make_key(t): t for t in test_data}

    results = []
    total_wrong = 0

    for pred in predictions_data:
        key = make_key(pred)
        gt_set = set(pred["ground_truth"])
        predicted = pred["predicted"]

        if not predicted:
            continue

        top1 = predicted[0]
        if top1 in gt_set:
            continue  # correct prediction, skip

        total_wrong += 1

        # Get issue text
        test_item = test_map.get(key)
        if test_item is None:
            continue

        issue_tokens = issue_path_tokens(test_item["issue_text"])

        # Path overlap for top-1 prediction
        pred_tokens = path_tokens(top1)
        pred_overlap = len(pred_tokens & issue_tokens)

        # Path overlap for GT files (take the max across GT files)
        max_gt_overlap = 0
        best_gt_file = None
        for gt_file in gt_set:
            gt_tokens = path_tokens(gt_file)
            overlap = len(gt_tokens & issue_tokens)
            if overlap > max_gt_overlap:
                max_gt_overlap = overlap
                best_gt_file = gt_file

        if pred_overlap > max_gt_overlap:
            results.append(
                {
                    "repo": pred["repo"],
                    "issue_id": pred["issue_id"],
                    "predicted_top1": top1,
                    "pred_path_overlap": pred_overlap,
                    "pred_overlap_tokens": sorted(pred_tokens & issue_tokens),
                    "gt_file": best_gt_file,
                    "gt_path_overlap": max_gt_overlap,
                    "gt_overlap_tokens": sorted(
                        path_tokens(best_gt_file) & issue_tokens
                    )
                    if best_gt_file
                    else [],
                    "ground_truth": list(gt_set),
                }
            )

    return results, total_wrong


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Path-Controlled Challenge Set Builder")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    test_data = load_jsonl(TEST_PATH)
    candidates_data = load_jsonl(CANDIDATES_PATH)
    predictions_data = load_jsonl(PREDICTIONS_PATH)

    candidates_map = {make_key(c): c["candidates"] for c in candidates_data}
    predictions_map = {make_key(p): p for p in predictions_data}

    print(f"  Test examples: {len(test_data)}")
    print(f"  Candidate sets: {len(candidates_map)}")
    print(f"  Predictions: {len(predictions_map)}")

    os.makedirs(CHALLENGE_DIR, exist_ok=True)

    # --- Type 1: Same-directory confounders ---
    print("\n" + "-" * 70)
    print("Type 1: Same-Directory Confounders")
    print("-" * 70)
    same_dir_results, r1_subset, r1_full, same_dir_confused = (
        build_same_directory_confounders(test_data, candidates_map, predictions_map)
    )
    total_wrong = sum(
        1 for p in predictions_map.values()
        if p["predicted"] and p["predicted"][0] not in set(p["ground_truth"])
    )
    print(f"  Examples with >= 3 same-dir candidates: {len(same_dir_results)}")
    print(f"  R@1 on same-dir subset: {r1_subset:.4f}")
    print(f"  R@1 on full test set:   {r1_full:.4f}")
    if r1_full > 0:
        print(f"  Relative drop:          {(r1_full - r1_subset) / r1_full * 100:.1f}%")
    print(f"\n  REFINED -- Same-dir confusion errors (wrong top-1 in same dir as GT):")
    print(f"    Count: {len(same_dir_confused)}/{total_wrong} errors ({len(same_dir_confused)/max(total_wrong,1)*100:.1f}%)")

    # Save subset keys for downstream eval
    same_dir_keys = [(r["repo"], r["issue_id"]) for r in same_dir_results]
    # Save as test subset (same format as grepo_test.jsonl)
    test_map = {make_key(t): t for t in test_data}
    same_dir_test = [test_map[k] for k in same_dir_keys if k in test_map]
    save_jsonl(same_dir_test, os.path.join(CHALLENGE_DIR, "type1_same_dir_test.jsonl"))
    save_jsonl(
        same_dir_results, os.path.join(CHALLENGE_DIR, "type1_same_dir_analysis.jsonl")
    )
    # Save refined same-dir confusion errors
    confused_keys = {(r["repo"], r["issue_id"]) for r in same_dir_confused}
    confused_test = [test_map[k] for k in confused_keys if k in test_map]
    save_jsonl(confused_test, os.path.join(CHALLENGE_DIR, "type1_same_dir_confused_test.jsonl"))
    save_jsonl(same_dir_confused, os.path.join(CHALLENGE_DIR, "type1_same_dir_confused_analysis.jsonl"))
    print(f"  Saved: {CHALLENGE_DIR}/type1_same_dir_test.jsonl ({len(same_dir_test)} examples)")
    print(f"  Saved: {CHALLENGE_DIR}/type1_same_dir_confused_test.jsonl ({len(confused_test)} examples)")

    # Show samples
    print("\n  Sample same-dir confusions:")
    for ex in same_dir_confused[:5]:
        print(f"    [{ex['repo']}#{ex['issue_id']}] dir={ex['shared_dir']}")
        print(f"      GT:   {ex['gt_file']}")
        print(f"      Pred: {ex['predicted_top1']}")

    # --- Type 2: Issue text de-leaked ---
    print("\n" + "-" * 70)
    print("Type 2: Issue Text De-Leaked")
    print("-" * 70)
    deleaked, leak_stats = build_deleaked(test_data)
    print(f"  Total examples: {leak_stats['total']}")
    print(
        f"  Examples with filename leakage: {leak_stats['leaked_examples']} "
        f"({leak_stats['leaked_examples'] / leak_stats['total'] * 100:.1f}%)"
    )
    print(f"  Total replacements made: {leak_stats['total_replacements']}")

    # Save de-leaked test set (compatible format)
    deleaked_clean = []
    for item in deleaked:
        clean = {k: v for k, v in item.items() if not k.startswith("_")}
        deleaked_clean.append(clean)
    save_jsonl(deleaked_clean, os.path.join(OUTPUT_DIR, "grepo_test_deleaked.jsonl"))

    # Save full version with metadata for analysis
    save_jsonl(deleaked, os.path.join(CHALLENGE_DIR, "type2_deleaked_analysis.jsonl"))

    # Also save the leaked-only subset
    leaked_only = [item for item in deleaked if item["_filename_leaked"]]
    leaked_only_clean = [{k: v for k, v in item.items() if not k.startswith("_")} for item in leaked_only]
    save_jsonl(leaked_only_clean, os.path.join(CHALLENGE_DIR, "type2_leaked_subset_test.jsonl"))
    print(f"  Saved: {OUTPUT_DIR}/grepo_test_deleaked.jsonl ({len(deleaked_clean)} examples)")
    print(f"  Saved: {CHALLENGE_DIR}/type2_leaked_subset_test.jsonl ({len(leaked_only_clean)} examples)")

    # --- Type 3: Path-misleading hard cases ---
    print("\n" + "-" * 70)
    print("Type 3: Path-Misleading Hard Cases")
    print("-" * 70)
    path_misled, total_wrong_t3 = build_path_misleading(predictions_data)
    print(f"  Total wrong predictions: {total_wrong_t3}")
    print(f"  Path-misled errors (pred has more path overlap than GT): {len(path_misled)}")
    if total_wrong_t3 > 0:
        print(
            f"  Fraction of errors that are path-misled: "
            f"{len(path_misled) / total_wrong_t3 * 100:.1f}%"
        )

    # Save
    save_jsonl(path_misled, os.path.join(CHALLENGE_DIR, "type3_path_misled_analysis.jsonl"))
    # Also save as eval-compatible test subset
    path_misled_keys = {(r["repo"], r["issue_id"]) for r in path_misled}
    path_misled_test = [test_map[k] for k in path_misled_keys if k in test_map]
    save_jsonl(path_misled_test, os.path.join(CHALLENGE_DIR, "type3_path_misled_test.jsonl"))
    print(f"  Saved: {CHALLENGE_DIR}/type3_path_misled_test.jsonl ({len(path_misled_test)} examples)")

    # Show a few examples
    print("\n  Sample path-misled errors:")
    for ex in path_misled[:5]:
        print(f"    [{ex['repo']}#{ex['issue_id']}]")
        print(f"      Predicted: {ex['predicted_top1']} (overlap={ex['pred_path_overlap']}: {ex['pred_overlap_tokens'][:5]})")
        print(f"      GT:        {ex['gt_file']} (overlap={ex['gt_path_overlap']}: {ex['gt_overlap_tokens'][:5]})")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Type 1a (same-dir candidates):   {len(same_dir_results)} examples (>= 3 same-dir candidates)")
    print(f"  Type 1b (same-dir confusion):    {len(same_dir_confused)}/{total_wrong} errors ({len(same_dir_confused)/max(total_wrong,1)*100:.1f}%)")
    print(f"  Type 2  (filename leaked):       {leak_stats['leaked_examples']}/{leak_stats['total']} examples ({leak_stats['leaked_examples']/leak_stats['total']*100:.1f}%)")
    print(f"  Type 3  (path-misled errors):    {len(path_misled)}/{total_wrong_t3} wrong predictions ({len(path_misled)/max(total_wrong_t3,1)*100:.1f}%)")
    print(f"\nOutput files:")
    print(f"  {CHALLENGE_DIR}/type1_same_dir_test.jsonl")
    print(f"  {CHALLENGE_DIR}/type1_same_dir_confused_test.jsonl")
    print(f"  {CHALLENGE_DIR}/type1_same_dir_analysis.jsonl")
    print(f"  {CHALLENGE_DIR}/type1_same_dir_confused_analysis.jsonl")
    print(f"  {OUTPUT_DIR}/grepo_test_deleaked.jsonl")
    print(f"  {CHALLENGE_DIR}/type2_deleaked_analysis.jsonl")
    print(f"  {CHALLENGE_DIR}/type2_leaked_subset_test.jsonl")
    print(f"  {CHALLENGE_DIR}/type3_path_misled_test.jsonl")
    print(f"  {CHALLENGE_DIR}/type3_path_misled_analysis.jsonl")


if __name__ == "__main__":
    main()
