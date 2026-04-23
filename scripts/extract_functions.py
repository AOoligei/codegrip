#!/usr/bin/env python3
"""
Extract function bodies from GREPO repos using Python AST.

For each repo, extracts all top-level and class-method function definitions
with their source code. Outputs a JSON index:
  {repo: {file_path: [{name, start_line, end_line, body}]}}

Usage:
    python scripts/extract_functions.py \
        --repos_dir data/repos \
        --output /data/chenlibin/grepo_agent_experiments/hierarchical/function_bodies.json
"""

import argparse
import ast
import json
import os
import sys
from pathlib import Path

MAX_FUNCTION_LINES = 200  # truncate very long functions


def extract_functions_from_file(filepath: str) -> list:
    """Extract function definitions from a Python file using AST."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
            lines = source.split("\n")
    except Exception:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    functions = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1  # 0-indexed
            end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 1
            body_lines = lines[start:end]

            # Truncate very long functions
            if len(body_lines) > MAX_FUNCTION_LINES:
                body_lines = body_lines[:MAX_FUNCTION_LINES] + ["    # ... (truncated)"]

            functions.append({
                "name": node.name,
                "start_line": node.lineno,
                "end_line": end,
                "body": "\n".join(body_lines),
                "num_lines": min(end - start, MAX_FUNCTION_LINES),
            })

    return functions


def process_repo(repo_dir: str, repo_name: str) -> dict:
    """Extract all functions from a repo."""
    result = {}
    repo_path = Path(repo_dir)

    for py_file in repo_path.rglob("*.py"):
        # Get relative path
        rel_path = str(py_file.relative_to(repo_path))

        # Skip common non-source directories
        if any(part in rel_path for part in [
            "__pycache__", ".git", "node_modules", ".tox",
            "dist/", "build/", ".eggs", "site-packages"
        ]):
            continue

        functions = extract_functions_from_file(str(py_file))
        if functions:
            result[rel_path] = functions

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos_dir", default="data/repos")
    parser.add_argument("--output", default="/data/chenlibin/grepo_agent_experiments/hierarchical/function_bodies.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    repos_dir = args.repos_dir
    repo_names = sorted(os.listdir(repos_dir))
    print(f"Processing {len(repo_names)} repos from {repos_dir}")

    all_data = {}
    total_functions = 0
    total_files = 0

    for i, repo_name in enumerate(repo_names):
        repo_path = os.path.join(repos_dir, repo_name)
        if not os.path.isdir(repo_path):
            continue

        functions = process_repo(repo_path, repo_name)
        all_data[repo_name] = functions

        n_files = len(functions)
        n_funcs = sum(len(v) for v in functions.values())
        total_files += n_files
        total_functions += n_funcs

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(repo_names)}] {repo_name}: {n_files} files, {n_funcs} functions")

    print(f"\nTotal: {total_files} files, {total_functions} functions across {len(all_data)} repos")

    with open(args.output, "w") as f:
        json.dump(all_data, f)

    print(f"Saved to {args.output}")

    # Also save a lightweight version (just names + line numbers, no bodies)
    lightweight_output = args.output.replace(".json", "_index.json")
    lightweight = {}
    for repo, files in all_data.items():
        lightweight[repo] = {}
        for fpath, funcs in files.items():
            lightweight[repo][fpath] = [
                {"name": fn["name"], "start_line": fn["start_line"],
                 "end_line": fn["end_line"], "num_lines": fn["num_lines"]}
                for fn in funcs
            ]

    with open(lightweight_output, "w") as f:
        json.dump(lightweight, f)
    print(f"Lightweight index saved to {lightweight_output}")


if __name__ == "__main__":
    main()
