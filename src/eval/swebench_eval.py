"""
SWE-bench Lite file-level localization evaluation.
Converts SWE-bench Lite data to GREPO format and evaluates CodeGRIP models.

Usage:
    # Step 1: Prepare SWE-bench Lite data
    python src/eval/swebench_eval.py prepare \
        --output_dir data/swebench_lite

    # Step 2: Run evaluation (with trained model)
    python src/eval/swebench_eval.py eval \
        --model_path /path/to/Qwen2.5-7B-Instruct \
        --lora_path experiments/exp1_sft_only/stage2_sft/final \
        --data_dir data/swebench_lite \
        --output_dir experiments/exp1_sft_only/eval_swebench
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def extract_changed_files_from_patch(patch: str) -> List[str]:
    """Extract changed file paths from a unified diff patch string."""
    files = []
    for line in patch.split('\n'):
        # Match 'diff --git a/path b/path' or '--- a/path' or '+++ b/path'
        m = re.match(r'^diff --git a/(.+?) b/(.+?)$', line)
        if m:
            files.append(m.group(2))
            continue
        # Also catch +++ b/path for robustness
        m = re.match(r'^\+\+\+ b/(.+?)$', line)
        if m and m.group(1) not in files:
            files.append(m.group(1))
    return files


def get_repo_files_at_commit(repo_url: str, commit: str, clone_dir: str) -> List[str]:
    """Clone repo at specific commit and list all Python files."""
    repo_name = repo_url.replace('/', '__')
    repo_path = os.path.join(clone_dir, repo_name)

    if not os.path.exists(repo_path):
        print(f"  Cloning {repo_url}...")
        subprocess.run(
            ['git', 'clone', '--depth', '1', f'https://github.com/{repo_url}.git', repo_path],
            capture_output=True, timeout=120
        )

    # Fetch specific commit
    try:
        subprocess.run(
            ['git', 'fetch', '--depth', '1', 'origin', commit],
            cwd=repo_path, capture_output=True, timeout=60
        )
        subprocess.run(
            ['git', 'checkout', commit],
            cwd=repo_path, capture_output=True, timeout=30
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        pass

    # List all Python files
    py_files = []
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden dirs and common non-source dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in
                   {'node_modules', '__pycache__', '.git', 'venv', '.venv', 'env'}]
        for f in files:
            if f.endswith('.py'):
                rel = os.path.relpath(os.path.join(root, f), repo_path)
                py_files.append(rel)

    return sorted(py_files)


def prepare_swebench_lite(output_dir: str, clone_repos: bool = False):
    """Convert SWE-bench Lite to GREPO-compatible format."""
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    print("Loading SWE-bench Lite from HuggingFace...")
    ds = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
    print(f"  Loaded {len(ds)} examples")

    # Convert to GREPO format
    converted = []
    repo_files = defaultdict(set)  # Track all files per repo
    skipped = 0

    for i, ex in enumerate(ds):
        # Extract changed files from patch
        changed_files = extract_changed_files_from_patch(ex['patch'])
        changed_py_files = [f for f in changed_files if f.endswith('.py')]

        if not changed_py_files:
            skipped += 1
            continue

        # Map SWE-bench fields to GREPO format
        instance = {
            'repo': ex['repo'].replace('/', '__'),  # e.g., django__django
            'repo_full': ex['repo'],  # e.g., django/django
            'issue_id': ex['instance_id'],
            'issue_text': f"Title: {ex['problem_statement'][:200]}\n\nDescription: {ex['problem_statement']}",
            'changed_files': changed_files,
            'changed_py_files': changed_py_files,
            'split': 'test',
            'base_commit': ex['base_commit'],
            'instance_id': ex['instance_id'],
        }
        converted.append(instance)

        # Collect all changed files for file tree
        for f in changed_py_files:
            repo_files[instance['repo']].add(f)

    print(f"  Converted: {len(converted)} examples ({skipped} skipped, no .py files)")

    # Save converted test data
    test_path = os.path.join(output_dir, 'swebench_lite_test.jsonl')
    with open(test_path, 'w') as f:
        for inst in converted:
            f.write(json.dumps(inst) + '\n')
    print(f"  Saved: {test_path}")

    # Build file trees (from patch info only - lightweight approach)
    # For proper eval, we'd clone repos and list all files
    if clone_repos:
        print("\nCloning repos for file trees...")
        clone_dir = os.path.join(output_dir, 'repos')
        file_tree_dir = os.path.join(output_dir, 'file_trees')
        os.makedirs(file_tree_dir, exist_ok=True)

        # Group by repo to avoid redundant clones
        repo_commits = defaultdict(list)
        for inst in converted:
            repo_commits[inst['repo_full']].append(inst['base_commit'])

        for repo_full, commits in repo_commits.items():
            repo_name = repo_full.replace('/', '__')
            commit = commits[0]  # Use first commit as representative

            py_files = get_repo_files_at_commit(repo_full, commit, clone_dir)

            tree = {
                'repo': repo_name,
                'all_files': py_files,
                'py_files': py_files,
                'num_all_files': len(py_files),
                'num_py_files': len(py_files),
            }

            tree_path = os.path.join(file_tree_dir, f'{repo_name}.json')
            with open(tree_path, 'w') as f:
                json.dump(tree, f, indent=2)
            print(f"  {repo_name}: {len(py_files)} Python files")
    else:
        print("\n  Skipping repo cloning (use --clone_repos to enable)")
        print("  For evaluation, you need file trees. Options:")
        print("  1. Run with --clone_repos (requires disk space + network)")
        print("  2. Use zeroshot mode (no file tree needed)")

    # Print statistics
    print(f"\n=== SWE-bench Lite Statistics ===")
    repo_counts = defaultdict(int)
    for inst in converted:
        repo_counts[inst['repo']] += 1
    for repo, count in sorted(repo_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {repo}: {count} examples")
    print(f"  Total repos: {len(repo_counts)}")

    avg_files = np.mean([len(inst['changed_py_files']) for inst in converted])
    print(f"  Avg changed .py files per instance: {avg_files:.1f}")


def run_swebench_eval_zeroshot(data_dir: str, output_dir: str,
                                model_path: str, lora_path: str = None,
                                gpu_id: int = 0):
    """Run CodeGRIP evaluation on SWE-bench Lite (zeroshot mode - no file tree)."""
    os.makedirs(output_dir, exist_ok=True)

    test_path = os.path.join(data_dir, 'swebench_lite_test.jsonl')
    file_tree_dir = os.path.join(data_dir, 'file_trees')

    # Check if file trees exist
    has_trees = os.path.isdir(file_tree_dir) and len(os.listdir(file_tree_dir)) > 0

    cmd = [
        sys.executable, 'src/eval/eval_grepo_file_level.py',
        '--model_path', model_path,
        '--test_data', test_path,
        '--output_dir', output_dir,
    ]

    if lora_path:
        cmd.extend(['--lora_path', lora_path])

    if has_trees:
        cmd.extend(['--file_tree_dir', file_tree_dir, '--prompt_mode', 'filetree'])
    else:
        cmd.extend(['--prompt_mode', 'zeroshot'])

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"Running eval: {' '.join(cmd)}")
    print(f"  GPU: {gpu_id}, Mode: {'filetree' if has_trees else 'zeroshot'}")

    subprocess.run(cmd, env=env)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Prepare command
    prep = subparsers.add_parser('prepare', help='Convert SWE-bench Lite to GREPO format')
    prep.add_argument('--output_dir', default='data/swebench_lite')
    prep.add_argument('--clone_repos', action='store_true',
                      help='Clone repos to build full file trees')

    # Eval command
    ev = subparsers.add_parser('eval', help='Run evaluation on SWE-bench Lite')
    ev.add_argument('--model_path', required=True)
    ev.add_argument('--lora_path', default=None)
    ev.add_argument('--data_dir', default='data/swebench_lite')
    ev.add_argument('--output_dir', required=True)
    ev.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()

    if args.command == 'prepare':
        prepare_swebench_lite(args.output_dir, args.clone_repos)
    elif args.command == 'eval':
        run_swebench_eval_zeroshot(
            args.data_dir, args.output_dir,
            args.model_path, args.lora_path, args.gpu_id
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
