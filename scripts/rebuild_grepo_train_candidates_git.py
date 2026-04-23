#!/usr/bin/env python3
"""
Rebuild GREPO train candidate pool using BM25 over each issue's TRAIN-ERA
commit of the repo (git archeology route).

Alternative to rebuild_grepo_train_candidates.py (which uses HEAD).  For each
training sample we:
    1.  Parse `timestamp` -> UTC datetime.
    2.  `git rev-list -1 --before=<timestamp> HEAD` inside the repo to find
        the nearest pre-issue commit SHA.
    3.  Enumerate .py files at that commit with `git ls-tree -r --name-only`.
    4.  Read file content via `git cat-file -p <sha>:<path>` (first N lines).
    5.  Build BM25 in memory, retrieve top-K candidate paths.
    6.  Record candidates as paths that exist AT THAT COMMIT (not HEAD).

Repos with shallow clones (< MIN_COMMITS, e.g. depth 1) cannot support git
archeology.  Samples from such repos are SKIPPED with a `skipped` flag so the
caller can decide what to do with them (typical fallback: use HEAD-BM25 from
the other script).

Input:
    data/grepo_text/grepo_train.jsonl
    data/repos/<repo>/.git

Output JSONL schema per line (for processed samples):
    {
        "repo": str,
        "issue_id": int/str,
        "commit_sha": str,           # nearest pre-issue commit
        "commit_date": str,          # commit committer ISO date
        "issue_timestamp": str,      # from train data
        "candidates": [path, ...],   # top-K at commit_sha
        "ground_truth": [path, ...], # GT files that EXIST at commit_sha
        "ground_truth_all": [path, ...],
        "gt_in_candidates": bool,
        "gt_coverage_rate": float,
        "n_candidates": int,
        "corpus_size": int,          # total .py files at that commit
    }

Skipped samples get an extra record to aid the consumer:
    {"repo", "issue_id", "skipped": True, "reason": "shallow_clone" | "no_commit_before_ts" | ...}

Determinism: seed 42.  Tokenizer identical to rebuild_grepo_train_candidates.py
so candidates can be compared head-to-head with the HEAD route.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
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
# Tokenization  (IDENTICAL to scripts/rebuild_grepo_train_candidates.py)
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
# Git helpers
# -----------------------------------------------------------------------------
def _git(repo_dir: str, *args: str, check: bool = True, timeout: int = 60) -> str:
    res = subprocess.run(
        ["git", "-C", repo_dir, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if check and res.returncode != 0:
        raise RuntimeError(
            f"git {args!r} failed in {repo_dir}: "
            f"{res.stderr.decode('utf-8', 'replace').strip()}"
        )
    return res.stdout.decode("utf-8", "replace")


class GitError(RuntimeError):
    """Raised to distinguish transient git failures from legitimate empty output."""


def repo_commit_depth(repo_dir: str) -> int:
    """
    Return the commit count from HEAD.  Raises GitError on failure (so a
    healthy repo is not misclassified as shallow due to a transient error).
    """
    try:
        res = subprocess.run(
            ["git", "-C", repo_dir, "rev-list", "--count", "HEAD"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise GitError(f"rev-list --count HEAD failed in {repo_dir}: {exc}")
    if res.returncode != 0:
        raise GitError(
            f"rev-list --count HEAD rc={res.returncode} in {repo_dir}: "
            f"{res.stderr.decode('utf-8', 'replace').strip()[:200]}"
        )
    out = res.stdout.decode("utf-8", "replace").strip()
    if not out.isdigit():
        raise GitError(f"rev-list --count HEAD unexpected output: {out!r}")
    return int(out)


_SENTINEL_NO_COMMIT = object()


def find_commit_before(repo_dir: str, ts_iso: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the newest commit whose COMMITTER DATE is <= ts_iso.

    Returns:
        (sha, commit_iso_date) when a commit is found.
        (None, None) ONLY when git legitimately reported no such commit
            (issue pre-dates first commit in the clone).
    Raises:
        GitError on any transient failure (timeout, non-zero rc with stderr).
    """
    try:
        res = subprocess.run(
            ["git", "-C", repo_dir, "rev-list", "-1",
             f"--before={ts_iso}", "HEAD"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise GitError(f"rev-list --before failed in {repo_dir}: {exc}")
    if res.returncode != 0:
        raise GitError(
            f"rev-list --before rc={res.returncode} in {repo_dir}: "
            f"{res.stderr.decode('utf-8', 'replace').strip()[:200]}"
        )
    sha = res.stdout.decode("utf-8", "replace").strip()
    if not sha:
        # Legitimate: no commit exists before ts_iso
        return None, None
    # Fetch commit date; raise GitError on transient failure so callers can
    # distinguish "no such commit" from "git misbehaving".
    try:
        date_res = subprocess.run(
            ["git", "-C", repo_dir, "log", "-1", "--pretty=%cI", sha],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise GitError(f"log --pretty=%cI failed in {repo_dir} @ {sha[:8]}: {exc}")
    if date_res.returncode != 0:
        raise GitError(
            f"log --pretty=%cI rc={date_res.returncode} in {repo_dir} @ {sha[:8]}: "
            f"{date_res.stderr.decode('utf-8', 'replace').strip()[:200]}"
        )
    commit_date = date_res.stdout.decode("utf-8", "replace").strip()
    return sha, commit_date


def ls_py_files_at(repo_dir: str, sha: str) -> List[str]:
    """List .py files at commit sha.  Raises GitError on transient failure."""
    try:
        res = subprocess.run(
            ["git", "-C", repo_dir, "ls-tree", "-r", "--name-only", sha],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise GitError(f"ls-tree failed in {repo_dir} @ {sha[:8]}: {exc}")
    if res.returncode != 0:
        raise GitError(
            f"ls-tree rc={res.returncode} in {repo_dir} @ {sha[:8]}: "
            f"{res.stderr.decode('utf-8', 'replace').strip()[:200]}"
        )
    out = res.stdout.decode("utf-8", "replace")
    files: List[str] = []
    for line in out.splitlines():
        if line.endswith(".py"):
            files.append(line)
    return sorted(files)


def read_blob_lines(
    repo_dir: str, sha: str, path: str, max_lines: int
) -> Tuple[str, bool]:
    """
    Read first max_lines of a file at commit sha.

    Returns (text, ok).  ok=False signals the blob could not be read
    (transient error or genuinely missing/non-blob entry).  Callers
    that need to distinguish 'file exists but unreadable' from
    'file indexed but empty' should use the flag.
    """
    try:
        res = subprocess.run(
            ["git", "-C", repo_dir, "cat-file", "-p", f"{sha}:{path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return "", False
    if res.returncode != 0:
        return "", False
    try:
        text = res.stdout.decode("utf-8", "replace")
    except Exception:
        return "", False
    lines = text.splitlines()[:max_lines]
    return "\n".join(line.rstrip() for line in lines), True


# -----------------------------------------------------------------------------
# Per-commit index
# -----------------------------------------------------------------------------
def build_commit_index(
    repo_dir: str,
    sha: str,
    max_lines: int,
) -> Tuple[Optional[BM25Okapi], List[str], List[str], int]:
    """
    Returns (bm25, all_files, indexed_files, n_blob_read_failures).

    all_files     : full .py universe at the commit (from git ls-tree).
                    Used for GT-survival checks and corpus_size reporting.
    indexed_files : subset aligned 1:1 with the BM25 corpus rows; only files
                    whose blob read + tokenization produced >=1 token.
    """
    py_files = ls_py_files_at(repo_dir, sha)  # may raise GitError
    if not py_files:
        return None, [], [], 0

    tokenized_docs: List[List[str]] = []
    indexed_files: List[str] = []
    n_blob_fail = 0
    for rel in py_files:
        content, ok = read_blob_lines(repo_dir, sha, rel, max_lines)
        if not ok:
            n_blob_fail += 1
        tokens = tokenize_document(rel, content)
        if tokens:
            tokenized_docs.append(tokens)
            indexed_files.append(rel)

    if not tokenized_docs:
        return None, py_files, [], n_blob_fail
    return BM25Okapi(tokenized_docs), py_files, indexed_files, n_blob_fail


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def process_sample(
    ex: dict,
    repo_dir: str,
    top_k: int,
    max_lines: int,
    index_cache: Dict[str, Tuple["BM25Okapi", List[str], List[str], int]],
) -> dict:
    """Process one sample.  Cache is keyed by commit_sha to reuse BM25 index.
    Raises GitError on transient git failures (caller decides policy)."""
    ts = ex.get("timestamp")
    if not ts:
        return {
            "repo": ex["repo"],
            "issue_id": ex.get("issue_id"),
            "skipped": True,
            "reason": "no_timestamp",
        }

    # git accepts ISO 8601; '2018-01-10 17:57:28+00:00' form is fine.
    ts_norm = ts.strip()

    sha, commit_date = find_commit_before(repo_dir, ts_norm)  # may raise GitError
    if sha is None:
        return {
            "repo": ex["repo"],
            "issue_id": ex.get("issue_id"),
            "skipped": True,
            "reason": "no_commit_before_ts",
            "issue_timestamp": ts_norm,
        }

    # Build or reuse index for this commit
    if sha not in index_cache:
        bm25, all_files, indexed_files, n_blob_fail = build_commit_index(
            repo_dir, sha, max_lines
        )
        if bm25 is None:
            return {
                "repo": ex["repo"],
                "issue_id": ex.get("issue_id"),
                "skipped": True,
                "reason": "empty_commit_corpus",
                "commit_sha": sha,
                "commit_date": commit_date,
                "issue_timestamp": ts_norm,
                "corpus_size": len(all_files),
                "blob_read_failures": n_blob_fail,
            }
        index_cache[sha] = (bm25, all_files, indexed_files, n_blob_fail)
    bm25, all_files, indexed_files, n_blob_fail = index_cache[sha]

    all_files_set = set(all_files)

    gt_all = list(ex.get("changed_py_files") or [])
    # A GT file "survives" if it exists at that commit (in full ls-tree),
    # not only if we managed to tokenise it.
    gt_surviving = [g for g in gt_all if g in all_files_set]

    issue_text = ex.get("issue_text", "") or ""
    query_tokens = tokenize_query(issue_text)
    if not query_tokens:
        query_tokens = tokenize_path(ex["repo"]) or ["the"]

    scores = bm25.get_scores(query_tokens)
    order = np.argsort(-scores, kind="stable")
    top_idx = order[:top_k]
    candidates = [indexed_files[i] for i in top_idx]

    cand_set = set(candidates)
    gt_hit = [g for g in gt_surviving if g in cand_set]
    coverage = (len(gt_hit) / len(gt_surviving)) if gt_surviving else 0.0

    return {
        "repo": ex["repo"],
        "issue_id": ex.get("issue_id"),
        "commit_sha": sha,
        "commit_date": commit_date,
        "issue_timestamp": ts_norm,
        "candidates": candidates,
        "ground_truth": gt_surviving,
        "ground_truth_all": gt_all,
        "gt_in_candidates": bool(gt_hit),
        "gt_coverage_rate": coverage,
        "n_candidates": len(candidates),
        "corpus_size": len(all_files),           # full .py universe at commit
        "indexed_size": len(indexed_files),      # files that entered BM25
        "blob_read_failures": n_blob_fail,
    }


def process_repo(
    repo_name: str,
    issues: Sequence[dict],
    repos_dir: str,
    top_k: int,
    max_lines: int,
    min_commits: int,
    log_fh,
) -> List[dict]:
    repo_dir = os.path.join(repos_dir, repo_name)
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        msg = f"  [SKIP] {repo_name}: no .git ({len(issues)} issues skipped)"
        print(msg, flush=True); log_fh.write(msg + "\n"); log_fh.flush()
        return [{
            "repo": repo_name, "issue_id": ex.get("issue_id"),
            "skipped": True, "reason": "no_git",
        } for ex in issues]

    try:
        depth = repo_commit_depth(repo_dir)
    except GitError as exc:
        msg = f"  [SKIP-GITERR] {repo_name}: depth-probe failed: {exc}"
        print(msg, flush=True); log_fh.write(msg + "\n"); log_fh.flush()
        return [{
            "repo": repo_name, "issue_id": ex.get("issue_id"),
            "skipped": True, "reason": "git_error_depth_probe",
            "error": str(exc),
        } for ex in issues]

    if depth < min_commits:
        msg = f"  [SKIP-SHALLOW] {repo_name}: depth={depth} < {min_commits} ({len(issues)} issues skipped)"
        print(msg, flush=True); log_fh.write(msg + "\n"); log_fh.flush()
        return [{
            "repo": repo_name, "issue_id": ex.get("issue_id"),
            "skipped": True, "reason": "shallow_clone", "repo_depth": depth,
        } for ex in issues]

    t0 = time.time()
    out: List[dict] = []
    index_cache: Dict[str, Tuple["BM25Okapi", List[str], List[str], int]] = {}
    n_processed = 0
    n_skipped = 0
    n_git_err = 0
    for ex in issues:
        try:
            r = process_sample(ex, repo_dir, top_k, max_lines, index_cache)
        except GitError as exc:
            r = {
                "repo": repo_name, "issue_id": ex.get("issue_id"),
                "skipped": True, "reason": "git_error_during_processing",
                "error": str(exc),
            }
            n_git_err += 1
        out.append(r)
        if r.get("skipped"):
            n_skipped += 1
        else:
            n_processed += 1
    dt = time.time() - t0
    # Explicitly drop the BM25 cache for this repo before returning so we
    # don't grow memory unboundedly across the 30 deep repos.
    index_cache.clear()
    msg = (
        f"  {repo_name}: depth={depth} issues={len(issues)} "
        f"processed={n_processed} skipped={n_skipped} (git_err={n_git_err}) "
        f"elapsed={dt:.1f}s"
    )
    print(msg, flush=True); log_fh.write(msg + "\n"); log_fh.flush()
    return out


def report_metrics(results: List[dict], tag: str, repos_dir: str) -> Dict[str, float]:
    processed = [r for r in results if not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]
    if not processed:
        print(f"[{tag}] no processed samples to report")
        return {
            "n": float(len(results)),
            "n_processed": 0.0,
            "n_skipped": float(len(skipped)),
        }

    per_sample_cov = np.asarray([r["gt_coverage_rate"] for r in processed])
    any_hit = np.asarray([float(r["gt_in_candidates"]) for r in processed])
    has_gt = [r for r in processed if r["ground_truth"]]
    any_hit_g = np.asarray([float(r["gt_in_candidates"]) for r in has_gt])
    cov_g = np.asarray([r["gt_coverage_rate"] for r in has_gt])

    ncs = np.asarray([r["n_candidates"] for r in processed])
    corp = np.asarray([r["corpus_size"] for r in processed])
    idx_sz = np.asarray([r.get("indexed_size", r["corpus_size"]) for r in processed])
    blob_fail = np.asarray([r.get("blob_read_failures", 0) for r in processed])

    # Sanity: verify a random sample of candidates are readable via git cat-file
    rng = random.Random(SEED)
    sample = rng.sample(processed, min(50, len(processed)))
    miss = 0; total = 0
    for r in sample:
        repo_dir = os.path.join(repos_dir, r["repo"])
        for p in r["candidates"][:20]:
            total += 1
            res = subprocess.run(
                ["git", "-C", repo_dir, "cat-file", "-e", f"{r['commit_sha']}:{p}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=10,
            )
            if res.returncode != 0:
                miss += 1
    exist_rate = (1 - miss / total) if total else 1.0

    print(f"\n===== [{tag}] metrics =====")
    print(f"  total={len(results)}  processed={len(processed)}  skipped={len(skipped)}")
    reasons = defaultdict(int)
    for r in skipped:
        reasons[r.get("reason", "?")] += 1
    for k, v in sorted(reasons.items()):
        print(f"    skip.{k}: {v}")
    print(f"  any-GT-in-cands (processed):                  {any_hit.mean():.3%}")
    print(f"  any-GT-in-cands (processed & has_gt n={len(has_gt)}): "
          f"{(any_hit_g.mean() if len(has_gt) else 0.0):.3%}")
    print(f"  partial-recall mean (processed):              {per_sample_cov.mean():.3%}")
    print(f"  partial-recall mean (processed & has_gt):     "
          f"{(cov_g.mean() if len(has_gt) else 0.0):.3%}")
    print(f"  candidates/sample:   min={ncs.min()}, median={int(np.median(ncs))}, max={ncs.max()}")
    print(f"  corpus_size:         min={corp.min()}, median={int(np.median(corp))}, max={corp.max()}")
    print(f"  indexed_size:        min={idx_sz.min()}, median={int(np.median(idx_sz))}, max={idx_sz.max()}")
    print(f"  blob_read_failures:  sum={int(blob_fail.sum())}, max_per_sample={int(blob_fail.max())}")
    print(f"  candidate-readable rate (git cat-file -e on {total}): {exist_rate:.3%} (miss={miss})")

    return {
        "n": float(len(results)),
        "n_processed": float(len(processed)),
        "n_skipped": float(len(skipped)),
        "skipped_reasons": {k: int(v) for k, v in reasons.items()},
        "any_hit_processed": float(any_hit.mean()),
        "any_hit_with_gt": float(any_hit_g.mean() if len(has_gt) else 0.0),
        "partial_recall_processed": float(per_sample_cov.mean()),
        "partial_recall_with_gt": float(cov_g.mean() if len(has_gt) else 0.0),
        "candidate_readable_rate": float(exist_rate),
        "median_corpus_size": float(np.median(corp)),
        "median_indexed_size": float(np.median(idx_sz)),
        "median_candidates": float(np.median(ncs)),
        "total_blob_read_failures": int(blob_fail.sum()),
        "max_blob_read_failures_per_sample": int(blob_fail.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl",
        default="/home/chenlibin/grepo_agent/data/grepo_text/grepo_train.jsonl")
    ap.add_argument("--repos_dir",
        default="/home/chenlibin/grepo_agent/data/repos")
    ap.add_argument("--output",
        default=("/home/chenlibin/grepo_agent/data/rankft/"
                 "grepo_train_git_historical_candidates.jsonl"))
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--max_lines", type=int, default=200)
    ap.add_argument("--min_commits", type=int, default=50,
        help="Minimum git depth to attempt archeology; shallower repos are skipped.")
    ap.add_argument("--smoke_n", type=int, default=0,
        help="If >0, only process this many deep-repo samples (stratified).")
    ap.add_argument("--smoke_mode", default="stratified",
        choices=["head", "stratified"])
    args = ap.parse_args()

    print("=== Rebuild GREPO train candidates (GIT HISTORICAL, BM25) ===", flush=True)
    print(f"  train_jsonl : {args.train_jsonl}")
    print(f"  repos_dir   : {args.repos_dir}")
    print(f"  output      : {args.output}")
    print(f"  top_k       : {args.top_k}")
    print(f"  max_lines   : {args.max_lines}")
    print(f"  min_commits : {args.min_commits}")
    print(f"  smoke_n     : {args.smoke_n}")

    examples: List[dict] = []
    with open(args.train_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"  loaded {len(examples)} train examples", flush=True)

    # Pre-compute repo depths so we can filter smoke samples to deep repos only
    repos_needed = sorted({ex["repo"] for ex in examples})
    depth: Dict[str, int] = {}
    for r in repos_needed:
        p = os.path.join(args.repos_dir, r)
        if not os.path.isdir(os.path.join(p, ".git")):
            depth[r] = 0
            continue
        try:
            depth[r] = repo_commit_depth(p)
        except GitError as exc:
            print(f"  [WARN] depth-probe failed for {r}: {exc}", flush=True)
            depth[r] = 0
    deep_repos = {r for r, d in depth.items() if d >= args.min_commits}
    print(f"  deep repos (depth>={args.min_commits}): {len(deep_repos)}/{len(repos_needed)}",
          flush=True)

    if args.smoke_n and args.smoke_n > 0:
        if args.smoke_mode == "stratified":
            # Only deep repos, only samples with at least one GT .py file
            pool = [ex for ex in examples
                    if ex["repo"] in deep_repos and (ex.get("changed_py_files") or [])]
            rng = random.Random(SEED)
            rng.shuffle(pool)
            # Try to spread across repos: round-robin sample
            by_repo: Dict[str, List[dict]] = defaultdict(list)
            for ex in pool:
                by_repo[ex["repo"]].append(ex)
            rotation = sorted(by_repo.keys())
            chosen: List[dict] = []
            i = 0
            while len(chosen) < args.smoke_n and any(by_repo[r] for r in rotation):
                r = rotation[i % len(rotation)]
                if by_repo[r]:
                    chosen.append(by_repo[r].pop())
                i += 1
            examples = chosen
            print(f"  stratified smoke: {len(examples)} samples across "
                  f"{len(set(e['repo'] for e in examples))} repos", flush=True)
        else:
            examples = [ex for ex in examples if ex["repo"] in deep_repos][: args.smoke_n]
        out_path = args.output + f".smoke{args.smoke_n}.jsonl"
        tag = f"smoke{args.smoke_n}"
    else:
        out_path = args.output
        tag = "full"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log_path = out_path + ".log"
    log_fh = open(log_path, "w")

    per_repo: Dict[str, List[dict]] = defaultdict(list)
    order_idx: Dict[str, int] = {}
    for i, ex in enumerate(examples):
        per_repo[ex["repo"]].append(ex)
        order_idx.setdefault(ex["repo"], i)
    repo_order = sorted(per_repo.keys(), key=lambda r: (order_idx[r], r))
    print(f"  {len(repo_order)} distinct repos", flush=True)
    log_fh.write(f"[START] {len(examples)} examples across {len(repo_order)} repos\n")
    log_fh.flush()

    t_start = time.time()
    id_to_result: Dict[int, dict] = {}
    for ri, repo_name in enumerate(repo_order):
        issues = per_repo[repo_name]
        print(f"\n[{ri+1}/{len(repo_order)}] repo={repo_name} "
              f"issues={len(issues)} elapsed={time.time()-t_start:.0f}s",
              flush=True)
        repo_results = process_repo(
            repo_name=repo_name, issues=issues,
            repos_dir=args.repos_dir, top_k=args.top_k,
            max_lines=args.max_lines, min_commits=args.min_commits,
            log_fh=log_fh,
        )
        for ex, r in zip(issues, repo_results):
            id_to_result[id(ex)] = r

    ordered_results: List[dict] = []
    for ex in examples:
        r = id_to_result.get(id(ex))
        if r is not None:
            ordered_results.append(r)

    with open(out_path, "w") as f:
        for r in ordered_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - t_start
    print(f"\nWrote {len(ordered_results)} records to {out_path} in {elapsed:.1f}s")
    log_fh.write(f"[DONE] wrote {len(ordered_results)} records in {elapsed:.1f}s\n")

    metrics = report_metrics(ordered_results, tag, args.repos_dir)
    metrics_path = out_path + ".metrics.json"
    with open(metrics_path, "w") as mf:
        json.dump(metrics, mf, indent=2)
    print(f"Metrics -> {metrics_path}")

    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
