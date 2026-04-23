#!/usr/bin/env python3
"""
Frontier long-context evaluation via DeepSeek API.

Tests whether a frontier model with full code context shows path dependency.
Uses listwise ranking: given issue + 20 candidates with code, rank them.

Usage:
    python scripts/eval_frontier_longctx.py \
        --condition none --max_examples 200 \
        --output_dir /data/chenlibin/grepo_agent_experiments/frontier_longctx
"""

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict

import numpy as np
from openai import OpenAI

random.seed(42)
np.random.seed(42)

API_KEY = "sk-d3778e26020f48e68bd6aad694f9f962"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3-max"

TEST_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_test.jsonl"
BM25_PATH = "/home/chenlibin/grepo_agent/data/rankft/merged_bm25_exp6_candidates.jsonl"
REPO_DIR = "/home/chenlibin/grepo_agent/data/repos"


def read_code(repo, file_path, max_lines=100):
    """Read first max_lines of a file from repo."""
    full = os.path.join(REPO_DIR, repo, file_path)
    if not os.path.isfile(full):
        return "# (file not available)"
    try:
        with open(full, "r", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip())
        return "\n".join(lines)
    except Exception:
        return "# (file not available)"


def shuffle_filenames(paths):
    """Shuffle filenames within directories."""
    dir_files = defaultdict(list)
    for p in paths:
        parts = p.rsplit("/", 1)
        d = parts[0] if len(parts) == 2 else ""
        dir_files[d].append(parts[-1])

    mapping = {}
    for d, files in dir_files.items():
        shuffled = files.copy()
        random.shuffle(shuffled)
        for orig, new in zip(files, shuffled):
            o = f"{d}/{orig}" if d else orig
            n = f"{d}/{new}" if d else new
            mapping[o] = n
    return mapping


def build_prompt(issue_text, candidates_with_code):
    """Build listwise ranking prompt with code."""
    prompt = (
        "Given this bug report, rank the following files by likelihood "
        "of needing modification. Consider both the file path and code content.\n\n"
        f"Bug Report:\n{issue_text[:3000]}\n\n"
        "Candidates:\n"
    )
    for i, (path, code) in enumerate(candidates_with_code):
        prompt += f"\n{i+1}. {path}\n```\n{code[:1500]}\n```\n"

    prompt += (
        "\nOutput ONLY a comma-separated list of candidate numbers, "
        "most likely first. Example: 3,1,5,2,4\n"
        "Ranking:"
    )
    return prompt


def parse_ranking(response, n_candidates):
    """Parse comma-separated ranking from model response."""
    text = response.strip()
    # Extract numbers
    numbers = re.findall(r'\d+', text)
    ranking = []
    seen = set()
    for n in numbers:
        idx = int(n) - 1  # 1-indexed to 0-indexed
        if 0 <= idx < n_candidates and idx not in seen:
            ranking.append(idx)
            seen.add(idx)
    # Pad with remaining indices in original order
    for i in range(n_candidates):
        if i not in seen:
            ranking.append(i)
    return ranking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["none", "shuffle_filenames"],
                        default="none")
    parser.add_argument("--max_examples", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--code_lines", type=int, default=100)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Load data
    print("Loading data...")
    test_data = {}
    with open(TEST_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            test_data[key] = rec

    bm25_data = {}
    with open(BM25_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            bm25_data[key] = rec

    # Sample examples
    keys = [k for k in test_data if k in bm25_data]
    random.shuffle(keys)
    keys = keys[:args.max_examples]
    print(f"  {len(keys)} examples sampled")

    hits = 0
    total = 0
    parse_failures = 0
    results = []
    start = time.time()

    for idx, key in enumerate(keys):
        test_rec = test_data[key]
        bm25_rec = bm25_data[key]
        repo = test_rec["repo"]
        issue_text = test_rec["issue_text"]
        gt_files = set(test_rec.get("changed_py_files",
                                     test_rec.get("changed_files", [])))
        candidates = bm25_rec.get("candidates",
                                   bm25_rec.get("bm25_candidates", []))[:args.top_k]

        if not gt_files or not candidates:
            continue

        # Apply perturbation
        display_candidates = list(candidates)
        display_gt = set(gt_files)
        if args.condition == "shuffle_filenames":
            all_paths = list(candidates) + [g for g in gt_files if g not in candidates]
            mapping = shuffle_filenames(all_paths)
            display_candidates = [mapping.get(c, c) for c in candidates]
            display_gt = {mapping.get(g, g) for g in gt_files}

        # Build prompt with code (use original paths for code access)
        cands_with_code = []
        for disp, orig in zip(display_candidates, candidates):
            code = read_code(repo, orig, args.code_lines)
            cands_with_code.append((disp, code))

        prompt = build_prompt(issue_text, cands_with_code)

        # Call API with retry
        response_text = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0,
                )
                response_text = response.choices[0].message.content
                break
            except Exception as e:
                print(f"  API error (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)

        if response_text is None:
            continue

        # Parse ranking
        ranking = parse_ranking(response_text, len(display_candidates))
        top1_idx = ranking[0]
        top1_file = display_candidates[top1_idx]

        hit = 1.0 if top1_file in display_gt else 0.0
        hits += hit
        total += 1

        # Check if ranking was identity (parse failure proxy)
        if ranking == list(range(len(display_candidates))):
            parse_failures += 1

        results.append({
            "repo": repo, "issue_id": str(test_rec["issue_id"]),
            "hit": hit, "top1": top1_file,
        })

        if (idx + 1) % 20 == 0:
            r1 = hits / total * 100
            elapsed = time.time() - start
            print(f"  [{idx+1}/{len(keys)}] {args.condition}: "
                  f"R@1={r1:.2f}% parse_fail={parse_failures} ({elapsed:.0f}s)")

        time.sleep(0.5)  # rate limit

    # Summary
    r1 = hits / max(1, total) * 100
    summary = {
        "model": MODEL,
        "condition": args.condition,
        "R@1": r1,
        "num_examples": total,
        "top_k": args.top_k,
        "code_lines": args.code_lines,
        "parse_failures": parse_failures,
    }

    fname = f"results_{args.condition}.json"
    with open(os.path.join(args.output_dir, fname), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {MODEL} | {args.condition} ===")
    print(f"R@1: {r1:.2f}% (N={total}, parse_fail={parse_failures})")
    print(f"Saved to {args.output_dir}/{fname}")


if __name__ == "__main__":
    main()
