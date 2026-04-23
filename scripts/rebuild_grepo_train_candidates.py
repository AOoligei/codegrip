#!/usr/bin/env python3
"""
Rebuild GREPO train candidate pool using BM25 over HEAD-version .py files.

Why:
    The existing BM25 / graph candidate files reference old-commit paths that
    no longer exist on the current HEAD of each repo (~95% miss).  For every
    issue in grepo_train we (re)index the current HEAD .py files of the repo
    and retrieve top-K candidates with a content+path BM25, so every
    candidate path actually exists on disk.

Input:
    data/grepo_text/grepo_train.jsonl           (7883 train issues)
    data/repos/<repo>/                          (HEAD checkouts, 75 repos)

Output (JSONL, one line per issue):
    {
        "repo":          str,
        "issue_id":      str/int,
        "issue_text":    str,
        "candidates":    [rel_path, ...]   # top-K files that exist on HEAD
        "ground_truth":  [rel_path, ...]   # GT files that EXIST on HEAD
        "ground_truth_all":    [rel_path, ...]   # all GT files (may include stale)
        "gt_in_candidates":    bool              # any surviving GT in candidates
        "gt_coverage_rate":    float             # |GT_surviving n cands| / |GT_surviving|
        "n_candidates":        int
    }

Design notes:
    - Seed 42 for any shuffling / tie-breaking.
    - BM25 tokenizer and indexing logic borrowed from
      scripts/swebench_bm25_content.py (path tokens 3x + first-N-line content).
    - One index per repo, reused across that repo's issues (fast).
    - 100% of emitted candidates are verified to exist as files on HEAD.
    - Samples with zero surviving GT files are still emitted (candidates are
      still produced) but flagged with gt_in_candidates=False - downstream
      code can decide whether to keep or drop them.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError as exc:
    print(f"[FATAL] rank_bm25 not available: {exc}", file=sys.stderr)
    sys.exit(2)


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# -----------------------------------------------------------------------------
# Tokenization  (same definitions as scripts/swebench_bm25_content.py)
# -----------------------------------------------------------------------------
_STOPWORDS = frozenset({
    'the', 'and', 'for', 'not', 'but', 'are', 'was', 'has', 'had',
    'can', 'may', 'use', 'def', 'class', 'self', 'return', 'import',
    'from', 'if', 'else', 'elif', 'try', 'except', 'with', 'as',
    'in', 'is', 'or', 'none', 'true', 'false', 'pass', 'raise',
    'this', 'that', 'will', 'would', 'should', 'could',
})

_CAMEL_RE_1 = re.compile(r'([a-z])([A-Z])')
_CAMEL_RE_2 = re.compile(r'([A-Z]+)([A-Z][a-z])')
_SEP_RE = re.compile(r'[_/\-.]')
_TOKEN_RE = re.compile(r'[a-zA-Z][a-zA-Z0-9]*')
_PY_REF_RE = re.compile(r'[\w/]+\.py\b')
_QUOTED_RE = re.compile(r'[`\'"](\w+)[`\'"]')
_DOTTED_RE = re.compile(r'\b\w+(?:\.\w+){2,}\b')


def tokenize_code(text: str) -> List[str]:
    text = _CAMEL_RE_1.sub(r'\1 \2', text)
    text = _CAMEL_RE_2.sub(r'\1 \2', text)
    text = _SEP_RE.sub(' ', text)
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


def tokenize_path(path: str) -> List[str]:
    clean = re.sub(r'\.py$', '', path)
    parts = re.split(r'[/_\-.]', clean)
    tokens: List[str] = []
    for part in parts:
        sub = _CAMEL_RE_1.sub(r'\1 \2', part)
        tokens.extend(sub.lower().split())
    return [t for t in tokens if len(t) > 1]


def tokenize_document(path: str, content: str) -> List[str]:
    # Path tokens repeated 3x to emphasise file-path matches.
    path_tokens = tokenize_path(path) * 3
    return path_tokens + tokenize_code(content)


def tokenize_query(text: str) -> List[str]:
    tokens = tokenize_code(text)
    for ref in _PY_REF_RE.findall(text):
        tokens.extend(tokenize_path(ref) * 3)
    tokens.extend(q.lower() for q in _QUOTED_RE.findall(text) if len(q) > 1)
    for dotted in _DOTTED_RE.findall(text):
        tokens.extend(dotted.lower().split('.'))
    return tokens


# -----------------------------------------------------------------------------
# File IO helpers
# -----------------------------------------------------------------------------
def read_head_py_file(abs_path: str, max_lines: int) -> str:
    try:
        with open(abs_path, 'r', errors='replace') as f:
            lines: List[str] = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip())
        return '\n'.join(lines)
    except (FileNotFoundError, PermissionError, IsADirectoryError, OSError):
        return ''


_SKIP_DIRS = frozenset({
    '.git', '.tox', '.mypy_cache', '.pytest_cache', '.venv', '.idea',
    '.vscode', '.eggs', '.nox', '.ruff_cache', '.hypothesis',
    '__pycache__', 'node_modules',
})


def list_repo_py_files(repo_dir: str) -> List[str]:
    out: List[str] = []
    for root, dirs, files in os.walk(repo_dir, followlinks=False):
        # Skip only well-known noise dirs -- keep `.ci`, `.github/scripts` etc
        # because some training GTs live under dotdirs.
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for name in sorted(files):
            if name.endswith('.py'):
                full = os.path.join(root, name)
                # Guarantee the emitted path is a real regular file on HEAD.
                if not os.path.isfile(full):
                    continue
                rel = os.path.relpath(full, repo_dir)
                out.append(rel)
    return out


# -----------------------------------------------------------------------------
# Repo index
# -----------------------------------------------------------------------------
def build_repo_index(
    repo_dir: str, max_lines: int
) -> Tuple[Optional[BM25Okapi], List[str]]:
    if not os.path.isdir(repo_dir):
        return None, []

    py_files = list_repo_py_files(repo_dir)
    if not py_files:
        return None, []

    tokenized_docs: List[List[str]] = []
    valid_files: List[str] = []
    for rel in py_files:
        abs_path = os.path.join(repo_dir, rel)
        content = read_head_py_file(abs_path, max_lines)
        tokens = tokenize_document(rel, content)
        if tokens:
            tokenized_docs.append(tokens)
            valid_files.append(rel)

    if not tokenized_docs:
        return None, []

    bm25 = BM25Okapi(tokenized_docs)
    return bm25, valid_files


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def process_repo(
    repo_name: str,
    issues: Sequence[dict],
    repos_dir: str,
    top_k: int,
    max_lines: int,
    log_fh,
) -> List[dict]:
    repo_dir = os.path.join(repos_dir, repo_name)
    t0 = time.time()
    bm25, valid_files = build_repo_index(repo_dir, max_lines)
    build_t = time.time() - t0
    if bm25 is None:
        # Emit placeholder records so we preserve the 1:1 input->output
        # contract.  Candidates are empty; downstream can filter if desired.
        msg = (
            f"  [WARN] {repo_name}: no indexable .py files; emitting "
            f"{len(issues)} empty-candidate records"
        )
        print(msg, flush=True)
        log_fh.write(msg + '\n')
        log_fh.flush()
        placeholders: List[dict] = []
        for ex in issues:
            gt_all = list(ex.get('changed_py_files') or [])
            placeholders.append({
                'repo': repo_name,
                'issue_id': ex.get('issue_id', ex.get('instance_id', '')),
                'issue_text': ex.get('issue_text', '') or '',
                'candidates': [],
                'ground_truth': [],
                'ground_truth_all': gt_all,
                'gt_in_candidates': False,
                'gt_coverage_rate': 0.0,
                'n_candidates': 0,
            })
        return placeholders

    valid_set = set(valid_files)
    msg = (
        f"  {repo_name}: {len(valid_files)} .py files, "
        f"index built in {build_t:.1f}s, {len(issues)} issues"
    )
    print(msg, flush=True)
    log_fh.write(msg + '\n')
    log_fh.flush()

    out: List[dict] = []
    for ex in issues:
        issue_text = ex.get('issue_text', '') or ''
        gt_all = list(ex.get('changed_py_files') or [])
        gt_surviving = [g for g in gt_all if g in valid_set]

        query_tokens = tokenize_query(issue_text)
        if not query_tokens:
            # Fall back to repeating repo name so we still emit something
            query_tokens = tokenize_path(repo_name) or ['the']

        scores = bm25.get_scores(query_tokens)
        # Deterministic tie break: stable sort by (-score, index)
        order = np.argsort(-scores, kind='stable')
        top_idx = order[:top_k]
        candidates = [valid_files[i] for i in top_idx]

        cand_set = set(candidates)
        gt_hit = [g for g in gt_surviving if g in cand_set]
        coverage = (len(gt_hit) / len(gt_surviving)) if gt_surviving else 0.0

        out.append({
            'repo': repo_name,
            'issue_id': ex.get('issue_id', ex.get('instance_id', '')),
            'issue_text': issue_text,
            'candidates': candidates,
            'ground_truth': gt_surviving,
            'ground_truth_all': gt_all,
            'gt_in_candidates': bool(gt_hit),
            'gt_coverage_rate': coverage,
            'n_candidates': len(candidates),
        })
    return out


def report_metrics(results: List[dict], tag: str, repos_dir: str) -> Dict[str, float]:
    if not results:
        print(f"[{tag}] no results to report")
        return {}
    n = len(results)

    # All samples: ceiling-bounded by how many GT survived
    per_sample_coverage = np.asarray([r['gt_coverage_rate'] for r in results])
    any_hit = np.asarray([float(r['gt_in_candidates']) for r in results])

    # Samples that have at least one surviving GT (upper-bound subset)
    has_gt = [r for r in results if r['ground_truth']]
    any_hit_g = np.asarray([float(r['gt_in_candidates']) for r in has_gt])
    cov_g = np.asarray([r['gt_coverage_rate'] for r in has_gt])

    # Candidate count distribution
    ncs = np.asarray([r['n_candidates'] for r in results])

    # Sanity: 100% candidate existence (we trust index but verify sample)
    sample_size = min(200, n)
    rng = random.Random(SEED)
    sample_idxs = rng.sample(range(n), sample_size)
    miss = 0
    total = 0
    for i in sample_idxs:
        r = results[i]
        repo_dir = os.path.join(repos_dir, r['repo'])
        for c in r['candidates'][:50]:
            total += 1
            if not os.path.isfile(os.path.join(repo_dir, c)):
                miss += 1
    exist_rate = (1 - miss / total) if total else 1.0

    print(f"\n===== [{tag}] metrics over {n} samples =====")
    print(
        f"  any-GT-in-cands (all samples):          "
        f"{any_hit.mean():.3%}"
    )
    print(
        f"  any-GT-in-cands (samples with surviving GT, n={len(has_gt)}): "
        f"{any_hit_g.mean():.3%}"
    )
    print(
        f"  partial-recall mean  (all):             "
        f"{per_sample_coverage.mean():.3%}"
    )
    print(
        f"  partial-recall mean  (with GT):         "
        f"{cov_g.mean():.3%}"
    )
    print(
        f"  candidates/sample:  min={ncs.min()}, "
        f"median={int(np.median(ncs))}, max={ncs.max()}"
    )
    print(
        f"  candidate-exists rate (sampled {total}): "
        f"{exist_rate:.3%} "
        f"(miss={miss})"
    )
    return {
        'n': float(n),
        'any_hit_all': float(any_hit.mean()),
        'any_hit_with_gt': float(any_hit_g.mean() if len(has_gt) else 0.0),
        'partial_recall_all': float(per_sample_coverage.mean()),
        'partial_recall_with_gt': float(cov_g.mean() if len(has_gt) else 0.0),
        'candidate_exists_rate': float(exist_rate),
        'min_cands': float(ncs.min()),
        'median_cands': float(np.median(ncs)),
        'max_cands': float(ncs.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--train_jsonl',
        default='/home/chenlibin/grepo_agent/data/grepo_text/grepo_train.jsonl',
    )
    ap.add_argument(
        '--repos_dir',
        default='/home/chenlibin/grepo_agent/data/repos',
    )
    ap.add_argument(
        '--output',
        default=(
            '/home/chenlibin/grepo_agent/data/rankft/'
            'grepo_train_head_candidates.jsonl'
        ),
    )
    ap.add_argument('--top_k', type=int, default=100)
    ap.add_argument('--max_lines', type=int, default=200)
    ap.add_argument(
        '--smoke_n',
        type=int,
        default=0,
        help='If >0, only process first N samples (preserving repo order) and'
             ' write output to <output>.smoke<N>.jsonl.',
    )
    ap.add_argument(
        '--smoke_mode',
        default='head',
        choices=['head', 'stratified'],
        help='head: first N samples; stratified: uniform random N sampled '
             'across repos AND restricted to samples with >=1 surviving GT '
             '(so we can actually measure recall).',
    )
    args = ap.parse_args()

    print('=== Rebuild GREPO train candidates (HEAD .py, BM25) ===', flush=True)
    print(f'  train_jsonl : {args.train_jsonl}')
    print(f'  repos_dir   : {args.repos_dir}')
    print(f'  output      : {args.output}')
    print(f'  top_k       : {args.top_k}')
    print(f'  max_lines   : {args.max_lines}')
    print(f'  smoke_n     : {args.smoke_n}')

    # Load train data (preserving original order)
    examples: List[dict] = []
    with open(args.train_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f'  loaded {len(examples)} train examples', flush=True)

    if args.smoke_n and args.smoke_n > 0:
        if args.smoke_mode == 'stratified':
            # Keep only samples whose at least one GT file survives on HEAD.
            # Then uniformly-random sample N of them with seed 42.  This lets
            # us measure recall even though ~33% of all samples have no
            # surviving GT (unrecoverable).
            survivors = []
            for ex in examples:
                base = os.path.join(args.repos_dir, ex['repo'])
                gt = ex.get('changed_py_files') or []
                if any(os.path.isfile(os.path.join(base, g)) for g in gt):
                    survivors.append(ex)
            rng = random.Random(SEED)
            rng.shuffle(survivors)
            examples = survivors[: args.smoke_n]
            print(
                f'  stratified smoke: {len(survivors)} samples with surviving'
                f' GT -> picked first {len(examples)} after seeded shuffle'
            )
        else:
            examples = examples[: args.smoke_n]
        out_path = args.output + f'.smoke{args.smoke_n}.jsonl'
        tag = f'smoke{args.smoke_n}'
    else:
        out_path = args.output
        tag = 'full'

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log_path = out_path + '.log'
    log_fh = open(log_path, 'w')

    # Preserve issue order per repo, group for cache reuse
    per_repo: Dict[str, List[dict]] = defaultdict(list)
    order_idx: Dict[str, int] = {}
    for i, ex in enumerate(examples):
        per_repo[ex['repo']].append(ex)
        order_idx.setdefault(ex['repo'], i)

    # Iterate repos in deterministic order (order of first appearance,
    # then alphabetical within tie)
    repo_order = sorted(per_repo.keys(), key=lambda r: (order_idx[r], r))
    print(f'  {len(repo_order)} distinct repos', flush=True)
    log_fh.write(f'[START] {len(examples)} examples across {len(repo_order)} repos\n')
    log_fh.flush()

    t_start = time.time()
    id_to_result: Dict[int, dict] = {}
    for ri, repo_name in enumerate(repo_order):
        issues = per_repo[repo_name]
        print(
            f'\n[{ri+1}/{len(repo_order)}] repo={repo_name} '
            f'issues={len(issues)} elapsed={time.time()-t_start:.0f}s',
            flush=True,
        )
        repo_results = process_repo(
            repo_name=repo_name,
            issues=issues,
            repos_dir=args.repos_dir,
            top_k=args.top_k,
            max_lines=args.max_lines,
            log_fh=log_fh,
        )
        # Map back to original examples order via (repo, issue_id) pair
        for r in repo_results:
            # Preserve input order: find position in examples
            pass
        # Simpler: store results in a dict keyed by id(exsource)
        for ex, r in zip(issues, repo_results):
            id_to_result[id(ex)] = r

    # Produce results in original example order (stable).  Every input
    # example MUST map to exactly one output row (possibly with empty
    # candidates); drop-silently would break dataset alignment.
    ordered_results: List[dict] = []
    for ex in examples:
        r = id_to_result.get(id(ex))
        if r is None:
            raise RuntimeError(
                f"internal: example repo={ex.get('repo')!r} "
                f"issue_id={ex.get('issue_id')!r} produced no result row"
            )
        ordered_results.append(r)

    if len(ordered_results) != len(examples):
        raise RuntimeError(
            f"row count mismatch: got {len(ordered_results)} results for "
            f"{len(examples)} input examples"
        )

    # Write output
    with open(out_path, 'w') as f:
        for r in ordered_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    elapsed = time.time() - t_start
    print(f'\nWrote {len(ordered_results)} records to {out_path} in {elapsed:.1f}s')
    log_fh.write(f'[DONE] wrote {len(ordered_results)} records in {elapsed:.1f}s\n')

    metrics = report_metrics(ordered_results, tag, args.repos_dir)
    metrics_path = out_path + '.metrics.json'
    with open(metrics_path, 'w') as mf:
        json.dump(metrics, mf, indent=2)
    print(f'Metrics -> {metrics_path}')

    log_fh.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
