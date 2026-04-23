"""Generate counterfactual conflict test data for CodeGRIP.

Swaps code between ground-truth and wrong-candidate files to create
counterfactual conditions for evaluating file localization models.
"""

import argparse
import json
import os
import random
from collections import defaultdict


def find_repo_dir(repo_name, base="data/repos"):
    candidates = [
        os.path.join(base, repo_name),
        os.path.join(base, repo_name.replace("/", "__")),
        os.path.join(base, repo_name.replace("/", "_")),
        os.path.join(base, repo_name.split("/")[-1]),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def read_file_content(repo_dir, file_path, max_lines=100):
    full_path = os.path.join(repo_dir, file_path)
    try:
        with open(full_path, "r", errors="ignore") as f:
            lines = f.readlines()[:max_lines]
        return "".join(lines)
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return "# (file content unavailable)"


def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main(args):
    random.seed(42)

    project_root = args.project_root
    test_path = os.path.join(project_root, args.test_file)
    cand_path = os.path.join(project_root, args.candidates_file)
    repos_base = os.path.join(project_root, args.repos_dir)
    out_dir = args.output_dir

    os.makedirs(out_dir, exist_ok=True)

    # Load data
    test_data = load_jsonl(test_path)
    cand_data = load_jsonl(cand_path)

    # Index candidates by (repo, issue_id)
    cand_index = {}
    for rec in cand_data:
        key = (rec["repo"], rec["issue_id"])
        cand_index[key] = rec["candidates"]

    # Statistics
    skip_reasons = defaultdict(int)
    results = []
    swap_manifest = []

    for ex in test_data:
        repo = ex["repo"]
        issue_id = ex["issue_id"]
        issue_text = ex["issue_text"]
        gt_files = set(ex["changed_py_files"])
        key = (repo, issue_id)

        # Get candidates
        if key not in cand_index:
            skip_reasons["no_candidates_entry"] += 1
            continue
        candidates = cand_index[key]

        # Find GT files in candidates
        gt_in_cand = [(i, f) for i, f in enumerate(candidates) if f in gt_files]
        if not gt_in_cand:
            skip_reasons["no_gt_in_candidates"] += 1
            continue

        # Pick f_gt: GT file with highest BM25 rank (lowest index)
        gt_in_cand.sort(key=lambda x: x[0])
        gt_idx, f_gt = gt_in_cand[0]

        # Pick f_wrong: closest non-GT candidate to f_gt in BM25 ranking
        non_gt = [(i, f) for i, f in enumerate(candidates) if f not in gt_files]
        if not non_gt:
            skip_reasons["no_non_gt_candidates"] += 1
            continue

        # Find the non-GT candidate ranked just after f_gt, or just before if f_gt is last
        after = [(i, f) for i, f in non_gt if i > gt_idx]
        before = [(i, f) for i, f in non_gt if i < gt_idx]

        if after:
            after.sort(key=lambda x: x[0])
            wrong_idx, f_wrong = after[0]
        elif before:
            before.sort(key=lambda x: x[0], reverse=True)
            wrong_idx, f_wrong = before[0]
        else:
            skip_reasons["no_adjacent_non_gt"] += 1
            continue

        # Find repo directory
        repo_dir = find_repo_dir(repo, base=repos_base)
        if repo_dir is None:
            skip_reasons["repo_snapshot_missing"] += 1
            continue

        # Read file contents
        gt_code = read_file_content(repo_dir, f_gt)
        wrong_code = read_file_content(repo_dir, f_wrong)

        record = {
            "repo": repo,
            "issue_id": issue_id,
            "issue_text": issue_text,
            "changed_py_files": list(gt_files),
            "candidates": candidates,
            "swap_gt_file": f_gt,
            "swap_wrong_file": f_wrong,
            "gt_code_original": gt_code,
            "wrong_code_original": wrong_code,
            # Condition: put wrong code at GT position, right code at wrong position
            "condition_prcw_codes": {f_gt: wrong_code},
            "condition_pwcr_codes": {f_wrong: gt_code},
            "condition_crossed_codes": {
                f_gt: wrong_code,
                f_wrong: gt_code,
            },
        }
        results.append(record)

        swap_manifest.append({
            "repo": repo,
            "issue_id": issue_id,
            "gt_file": f_gt,
            "gt_rank": gt_idx,
            "wrong_file": f_wrong,
            "wrong_rank": wrong_idx,
        })

    # Write outputs
    out_jsonl = os.path.join(out_dir, "counterfactual_crossed.jsonl")
    with open(out_jsonl, "w") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out_manifest = os.path.join(out_dir, "swap_manifest.json")
    with open(out_manifest, "w") as f:
        json.dump(swap_manifest, f, indent=2, ensure_ascii=False)

    # Print statistics
    print(f"Total test examples: {len(test_data)}")
    print(f"Valid counterfactual examples: {len(results)}")
    print(f"Skipped: {sum(skip_reasons.values())}")
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    print(f"\nOutput written to: {out_dir}")
    print(f"  {out_jsonl}")
    print(f"  {out_manifest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate counterfactual conflict test data for CodeGRIP"
    )
    parser.add_argument(
        "--project-root",
        default="/home/chenlibin/grepo_agent",
        help="Project root directory",
    )
    parser.add_argument(
        "--test-file",
        default="data/grepo_text/grepo_test.jsonl",
        help="Path to test JSONL (relative to project root)",
    )
    parser.add_argument(
        "--candidates-file",
        default="data/rankft/merged_bm25_exp6_candidates.jsonl",
        help="Path to BM25 candidates JSONL (relative to project root)",
    )
    parser.add_argument(
        "--repos-dir",
        default="data/repos",
        help="Path to repo snapshots directory (relative to project root)",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/chenlibin/grepo_agent_experiments/counterfactual",
        help="Output directory for counterfactual data",
    )
    args = parser.parse_args()
    main(args)
