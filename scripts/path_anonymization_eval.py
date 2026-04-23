#!/usr/bin/env python3
"""
Path anonymization control: evaluate the reranker with anonymized file paths.
If the model is just exploiting lexical path cues, anonymization should destroy performance.
If it has learned deeper patterns, some signal should remain.

Three conditions:
1. Normal (baseline) - already have this
2. Directory-anonymized: replace directory names with generic labels (dir1/dir2/file.py)
3. Fully-anonymized: replace entire path with hash-like labels (file_001, file_002, ...)

Usage: python scripts/path_anonymization_eval.py --gpu_id 4
"""
import argparse
import json
import hashlib
import re
import sys
import os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


def anonymize_directory(path: str, dir_map: dict) -> str:
    """Replace directory components with generic labels, keep filename."""
    parts = path.split("/")
    if len(parts) <= 1:
        return path
    filename = parts[-1]
    anon_dirs = []
    for d in parts[:-1]:
        if d not in dir_map:
            dir_map[d] = f"dir{len(dir_map):03d}"
        anon_dirs.append(dir_map[d])
    return "/".join(anon_dirs) + "/" + filename


def anonymize_full(path: str, path_map: dict) -> str:
    """Replace entire path with a generic label."""
    if path not in path_map:
        path_map[path] = f"file_{len(path_map):04d}.py"
    return path_map[path]


def anonymize_filename(path: str) -> str:
    """Keep directory structure, anonymize filename to hash."""
    parts = path.split("/")
    if len(parts) <= 1:
        h = hashlib.md5(path.encode()).hexdigest()[:8]
        return f"{h}.py"
    dirs = "/".join(parts[:-1])
    h = hashlib.md5(parts[-1].encode()).hexdigest()[:8]
    ext = Path(parts[-1]).suffix or ".py"
    return f"{dirs}/{h}{ext}"


def create_anonymized_candidates(
    test_path: str,
    bm25_path: str,
    output_test: str,
    output_bm25: str,
    mode: str,  # "dir", "full", "filename"
):
    """Create anonymized versions of test and BM25 candidate files."""
    # Load test data
    test_data = []
    with open(test_path) as f:
        for line in f:
            test_data.append(json.loads(line))

    # Load BM25 candidates
    bm25_data = {}
    with open(bm25_path) as f:
        for line in f:
            item = json.loads(line)
            key = (item["repo"], str(item["issue_id"]))
            bm25_data[key] = item

    # Per-example anonymization (maps reset per example to avoid cross-example leakage)
    anon_test = []
    anon_bm25 = []

    for item in test_data:
        repo = item["repo"]
        issue_id = str(item["issue_id"])
        key = (repo, issue_id)

        if key not in bm25_data:
            continue

        bm25_item = bm25_data[key]
        candidates = bm25_item.get("bm25_candidates", [])
        gt_files = item.get("changed_py_files", [])

        # Build anonymization map for this example
        # Include all candidates + gt files
        all_paths = list(set(candidates + gt_files))

        if mode == "dir":
            dir_map = {}
            path_map = {p: anonymize_directory(p, dir_map) for p in all_paths}
        elif mode == "full":
            path_map_dict = {}
            path_map = {p: anonymize_full(p, path_map_dict) for p in all_paths}
        elif mode == "filename":
            path_map = {p: anonymize_filename(p) for p in all_paths}
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Create anonymized test item
        anon_item = dict(item)
        anon_item["changed_py_files"] = [path_map.get(f, f) for f in gt_files]
        anon_test.append(anon_item)

        # Create anonymized BM25 item
        anon_bm25_item = dict(bm25_item)
        anon_bm25_item["bm25_candidates"] = [path_map.get(c, c) for c in candidates]
        anon_bm25.append(anon_bm25_item)

    # Write output
    with open(output_test, "w") as f:
        for item in anon_test:
            f.write(json.dumps(item) + "\n")

    with open(output_bm25, "w") as f:
        for item in anon_bm25:
            f.write(json.dumps(item) + "\n")

    print(f"[{mode}] Wrote {len(anon_test)} test, {len(anon_bm25)} bm25 items")
    return len(anon_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=4)
    parser.add_argument("--modes", nargs="+", default=["dir", "full"])
    args = parser.parse_args()

    BASE = Path(".")
    PYTHON = sys.executable
    test_path = "data/grepo_text/grepo_test.jsonl"
    bm25_path = "data/rankft/merged_bm25_exp6_candidates.jsonl"
    model_path = "/data/shuyang/models/Qwen2.5-7B-Instruct"
    lora_path = "experiments/rankft_runB_graph/best"

    for mode in args.modes:
        out_dir = f"experiments/path_anon_{mode}"
        os.makedirs(out_dir, exist_ok=True)

        out_test = f"{out_dir}/test.jsonl"
        out_bm25 = f"{out_dir}/bm25_candidates.jsonl"

        print(f"\n=== Creating {mode}-anonymized data ===")
        n = create_anonymized_candidates(
            test_path, bm25_path, out_test, out_bm25, mode
        )

        print(f"=== Evaluating {mode}-anonymized ({n} examples) ===")
        eval_cmd = (
            f"{PYTHON} src/eval/eval_rankft.py "
            f"--model_path {model_path} "
            f"--lora_path {lora_path} "
            f"--test_data {out_test} "
            f"--bm25_candidates {out_bm25} "
            f"--output_dir {out_dir}/eval "
            f"--gpu_id {args.gpu_id} "
            f"--top_k 200 "
            f"--max_seq_length 512"
        )
        print(f"Running: {eval_cmd}")
        os.system(eval_cmd)

        # Print results
        summary_path = f"{out_dir}/eval/summary.json"
        if os.path.exists(summary_path):
            d = json.load(open(summary_path))["overall"]
            r1 = d.get("recall@1", d.get("hit@1", 0))
            r5 = d.get("recall@5", d.get("hit@5", 0))
            print(f"\n[{mode}] R@1={r1:.2f}, R@5={r5:.2f}")
        else:
            print(f"\n[{mode}] No results found")

    print("\n=== Path Anonymization Summary ===")
    print(f"{'Mode':<15} {'R@1':>8} {'R@5':>8}")
    print("-" * 35)
    print(f"{'Normal':<15} {'27.01':>8} {'49.17':>8}")
    for mode in args.modes:
        sp = f"experiments/path_anon_{mode}/eval/summary.json"
        if os.path.exists(sp):
            d = json.load(open(sp))["overall"]
            r1 = d.get("recall@1", d.get("hit@1", 0))
            r5 = d.get("recall@5", d.get("hit@5", 0))
            print(f"{mode:<15} {r1:>8.2f} {r5:>8.2f}")


if __name__ == "__main__":
    main()
