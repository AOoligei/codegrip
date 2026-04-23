#!/usr/bin/env python3
"""
Pre-extract all function bodies from GREPO repos and cache as JSON.

Output: {repo: {file_path: [{name, body, lineno}, ...], ...}, ...}

This avoids repeated AST parsing during training, speeding up
train_code_reranker_masked.py by ~10x.

Usage:
    python scripts/preextract_functions.py \
        --output /data/chenlibin/grepo_agent_experiments/function_cache.json
"""

import ast
import json
import os
import sys
import time

REPO_DIR = "/home/chenlibin/grepo_agent/data/repos"
BM25_TRAIN_PATH = "/home/chenlibin/grepo_agent/data/rankft/grepo_train_bm25_top500.jsonl"
BM25_TEST_PATH = "/home/chenlibin/grepo_agent/data/rankft/merged_bm25_exp6_candidates.jsonl"
MAX_LINES_PER_FUNC = 30


def extract_functions(file_path):
    """Extract all function defs from a Python file."""
    if not os.path.isfile(file_path):
        return []
    try:
        with open(file_path, "r", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, Exception):
        return []

    lines = source.splitlines()
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = min(start + MAX_LINES_PER_FUNC, len(lines))
            body = "\n".join(lines[start:end])
            functions.append({
                "name": node.name,
                "body": body,
                "lineno": node.lineno,
            })
    return functions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,
                        default="/data/chenlibin/grepo_agent_experiments/function_cache.json")
    args = parser.parse_args()

    # Collect all unique (repo, file_path) pairs from train + test candidates
    all_files = set()
    for path in [BM25_TRAIN_PATH, BM25_TEST_PATH]:
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                repo = rec["repo"]
                candidates = rec.get("candidates", rec.get("bm25_candidates", []))
                for c in candidates[:200]:
                    all_files.add((repo, c))

    print(f"Total unique (repo, file) pairs: {len(all_files)}")

    # Extract functions
    cache = {}
    start = time.time()
    for i, (repo, fpath) in enumerate(sorted(all_files)):
        full_path = os.path.join(REPO_DIR, repo, fpath)
        funcs = extract_functions(full_path)
        if repo not in cache:
            cache[repo] = {}
        if funcs:
            cache[repo][fpath] = funcs

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(all_files)}] {elapsed:.0f}s")

    # Save
    with open(args.output, "w") as f:
        json.dump(cache, f)

    total_funcs = sum(len(funcs) for repo_files in cache.values()
                      for funcs in repo_files.values())
    total_files = sum(len(repo_files) for repo_files in cache.values())
    print(f"\nDone: {len(cache)} repos, {total_files} files, {total_funcs} functions")
    print(f"Saved to {args.output}")
    print(f"Time: {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
