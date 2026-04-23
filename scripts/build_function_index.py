#!/usr/bin/env python3
"""
Build a complete function index from source repos using AST parsing.

Extracts all function/method/class names per file, no truncation.
Output format: {repo: {file_path: [func_name, ...]}}

Usage:
    python scripts/build_function_index.py \
        --repos_dir data/repos \
        --output data/function_index.json
"""
import ast
import json
import os
import argparse
from collections import defaultdict


def extract_names_from_file(filepath):
    """Extract function and class names from a Python file using AST."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception:
        return [], []

    functions = []
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)

    return functions, classes


def build_index_for_repo(repo_dir, repo_name):
    """Build function index for a single repo."""
    index = {}
    for root, dirs, files in os.walk(repo_dir):
        # Skip hidden dirs, __pycache__, etc.
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, repo_dir)

            functions, classes = extract_names_from_file(full_path)
            if functions or classes:
                # Deduplicate while preserving order
                seen = set()
                all_names = []
                for name in functions + classes:
                    if name not in seen:
                        seen.add(name)
                        all_names.append(name)
                index[rel_path] = all_names
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos_dir", default="data/repos")
    parser.add_argument("--output", default="data/function_index.json")
    parser.add_argument("--test_data", default="data/grepo_text/grepo_test.jsonl",
                        help="Test data to check which repos are needed")
    args = parser.parse_args()

    # Find which repos we need
    needed_repos = set()
    if os.path.exists(args.test_data):
        with open(args.test_data) as f:
            for line in f:
                item = json.loads(line)
                needed_repos.add(item["repo"])
        print(f"Need {len(needed_repos)} repos from test data")

    # Build index
    all_index = {}
    available = sorted(os.listdir(args.repos_dir))

    for repo_name in available:
        repo_dir = os.path.join(args.repos_dir, repo_name)
        if not os.path.isdir(repo_dir):
            continue
        if needed_repos and repo_name not in needed_repos:
            continue

        index = build_index_for_repo(repo_dir, repo_name)
        all_index[repo_name] = index
        print(f"  {repo_name}: {len(index)} files with functions/classes")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_index, f)
    print(f"\nSaved index for {len(all_index)} repos to {args.output}")
    total_files = sum(len(v) for v in all_index.values())
    total_funcs = sum(len(fns) for v in all_index.values() for fns in v.values())
    print(f"Total: {total_files} files, {total_funcs} function/class names")


if __name__ == "__main__":
    main()
