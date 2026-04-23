"""
Content-aware BM25 baseline for file-level localization.
Instead of matching against file paths, matches issue text against file contents.
This provides a much stronger retrieval signal for code localization.

Usage:
    python src/eval/bm25_content_baseline.py \
        --test_data data/swebench_lite/swebench_lite_test.jsonl \
        --repo_dir data/swebench_lite/repos \
        --file_tree_dir data/swebench_lite/file_trees \
        --output_dir experiments/baselines/swebench_content_bm25
"""
import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize_code(code: str) -> List[str]:
    """Tokenize Python code into meaningful tokens."""
    # Remove comments and docstrings (simplified)
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    
    # Extract identifiers
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', code)
    
    # Split camelCase
    expanded = []
    for t in tokens:
        parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', t).lower().split()
        expanded.extend(parts)
    
    # Filter
    stop = {'self', 'def', 'class', 'return', 'import', 'from', 'if', 'else',
            'elif', 'for', 'while', 'try', 'except', 'with', 'as', 'in',
            'not', 'and', 'or', 'is', 'none', 'true', 'false', 'pass',
            'break', 'continue', 'raise', 'yield', 'lambda', 'global',
            'nonlocal', 'assert', 'del', 'print'}
    return [t for t in expanded if len(t) > 1 and t not in stop]


def tokenize_issue(text: str) -> List[str]:
    """Tokenize issue text."""
    text = text.lower()
    text = re.sub(r'^title:\s*', '', text)
    tokens = re.findall(r'[a-z_][a-z0-9_]*', text)
    return [t for t in tokens if len(t) > 1]


def read_file_contents(repo_dir: str, file_path: str, max_lines: int = 200) -> str:
    """Read file contents with line limit."""
    full_path = os.path.join(repo_dir, file_path)
    try:
        with open(full_path, 'r', errors='ignore') as f:
            lines = f.readlines()[:max_lines]
        return ''.join(lines)
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return ''


def compute_hit_at_k(predicted: List[str], ground_truth: Set[str], k: int) -> float:
    if not ground_truth:
        return 0.0
    top_k = set(predicted[:k])
    hits = len(top_k & ground_truth)
    return hits / len(ground_truth)


def run_content_bm25(test_data: List[dict], repo_dir: str, file_trees: dict,
                     output_dir: str, max_lines: int = 200) -> dict:
    """Run BM25 on file contents instead of paths."""
    os.makedirs(output_dir, exist_ok=True)
    predictions = []
    all_metrics = {f'hit@{k}': [] for k in [1, 3, 5, 10, 20]}
    
    # Cache file contents per repo
    repo_contents_cache = {}
    
    start = time.time()
    for i, ex in enumerate(test_data):
        repo = ex['repo']
        gt = set(ex['changed_py_files'])
        
        if repo not in file_trees:
            continue
        
        candidates = file_trees[repo].get('py_files', [])
        if not candidates:
            continue
        
        # Find repo directory
        repo_path = os.path.join(repo_dir, repo)
        if not os.path.isdir(repo_path):
            # Try other naming conventions
            for name in os.listdir(repo_dir):
                if name.replace('__', '/') == repo or name == repo:
                    repo_path = os.path.join(repo_dir, name)
                    break
        
        if not os.path.isdir(repo_path):
            continue
        
        # Get or cache file contents
        if repo not in repo_contents_cache:
            contents = {}
            for f in candidates:
                content = read_file_contents(repo_path, f, max_lines)
                if content:
                    contents[f] = tokenize_code(content)
                else:
                    contents[f] = []
            repo_contents_cache[repo] = contents
        
        contents = repo_contents_cache[repo]
        
        # Build BM25 index on file contents
        valid_candidates = [c for c in candidates if contents.get(c)]
        if not valid_candidates:
            continue
        
        tokenized_docs = [contents[c] for c in valid_candidates]
        
        bm25 = BM25Okapi(tokenized_docs)
        query_tokens = tokenize_issue(ex['issue_text'])
        
        if not query_tokens:
            ranked = valid_candidates[:20]
        else:
            scores = bm25.get_scores(query_tokens)
            ranked_idx = np.argsort(scores)[::-1]
            ranked = [valid_candidates[j] for j in ranked_idx[:20]]
        
        metrics = {}
        for k in [1, 3, 5, 10, 20]:
            h = compute_hit_at_k(ranked, gt, k)
            metrics[f'hit@{k}'] = h
            all_metrics[f'hit@{k}'].append(h)
        
        predictions.append({
            'repo': repo,
            'issue_id': ex['issue_id'],
            'ground_truth': list(gt),
            'predicted': ranked,
            'metrics': metrics,
            'method': 'bm25_content',
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(test_data)}")
    
    elapsed = time.time() - start
    
    pred_path = os.path.join(output_dir, 'predictions.jsonl')
    with open(pred_path, 'w') as f:
        for p in predictions:
            f.write(json.dumps(p) + '\n')
    
    summary = {
        'method': 'bm25_content',
        'num_examples': len(predictions),
        'wall_clock_seconds': round(elapsed, 2),
        'metrics': {k: round(np.mean(v) * 100, 2) for k, v in all_metrics.items()},
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--repo_dir', required=True)
    parser.add_argument('--file_tree_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_lines', type=int, default=200,
                        help='Max lines to read per file')
    args = parser.parse_args()
    
    # Load test data
    test_data = []
    with open(args.test_data) as f:
        for line in f:
            ex = json.loads(line)
            if ex.get('changed_py_files'):
                test_data.append(ex)
    
    # Load file trees
    file_trees = {}
    for fp in Path(args.file_tree_dir).glob('*.json'):
        with open(fp) as f:
            tree = json.load(f)
            file_trees[tree['repo']] = tree
    
    print(f"Loaded {len(test_data)} test examples, {len(file_trees)} file trees")
    
    summary = run_content_bm25(test_data, args.repo_dir, file_trees,
                                args.output_dir, args.max_lines)
    
    print(f"\nContent BM25 results ({summary['num_examples']} examples):")
    for k, v in summary['metrics'].items():
        print(f"  {k}: {v}")
    print(f"Time: {summary['wall_clock_seconds']}s")


if __name__ == '__main__':
    main()
