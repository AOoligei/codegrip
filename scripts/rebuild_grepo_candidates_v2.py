#!/usr/bin/env python3
"""
Rebuild GREPO training candidate pool — Strategy D (Hybrid with Path-Alias Resolution).

Implements GREPO_REBUILD_SPEC.md §4 phases 1-5:
  Phase 1: Path-alias map per repo (GT old-path -> HEAD path by basename + edit distance).
  Phase 2: Per-repo BM25 over HEAD .py files (rank_bm25). Top-150 per issue.
  Phase 3: Graph expansion (1-hop import neighbors from data/dep_graphs/<repo>_rels.json).
  Phase 4: Assemble 100-candidate list: gt + bm25(50%) + graph(20%) + samedir(10%) + random(20%).
  Phase 5: Existence check; purge missing; refill from random pool.

Output schema (spec §1):
  {repo, issue_id, candidates, candidate_sources, gt_files, gt_in_candidates,
   gt_coverage_rate, n_candidates, build_seed}

Determinism: seed 42, sorted iteration, stable sort ties.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError as exc:  # pragma: no cover
    print(f"[FATAL] rank_bm25 not available: {exc}", file=sys.stderr)
    sys.exit(2)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def stable_seed(*parts: object) -> int:
    """Deterministic 32-bit seed derived from a sha256 digest of the parts.
    Python's built-in hash() is salted per interpreter process; this is not.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(b'\0')
        h.update(str(p).encode('utf-8'))
    return int.from_bytes(h.digest()[:4], 'big')


def is_readable_file(abs_path: str) -> bool:
    """Spec §2 gate: 'readable on disk'. os.path.isfile is insufficient
    because it passes on files without read permission. This opens and
    reads one byte to verify."""
    try:
        if not os.path.isfile(abs_path):
            return False
        with open(abs_path, 'rb') as f:
            f.read(1)
        return True
    except (OSError, PermissionError):
        return False


# ---------------------------------------------------------------------------
# Tokenization (spec §4 Phase 2: identifiers of length >=3, per train_rankft
# pattern). We keep the broader code/path tokenizer for documents to maintain
# recall; issue-text queries use the >=3 identifier filter.
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset({
    'the', 'and', 'for', 'not', 'but', 'are', 'was', 'has', 'had',
    'can', 'may', 'use', 'def', 'class', 'self', 'return', 'import',
    'from', 'if', 'else', 'elif', 'try', 'except', 'with', 'as',
    'in', 'is', 'or', 'none', 'true', 'false', 'pass', 'raise',
    'this', 'that', 'will', 'would', 'should', 'could', 'title',
})

_CAMEL_RE_1 = re.compile(r'([a-z])([A-Z])')
_CAMEL_RE_2 = re.compile(r'([A-Z]+)([A-Z][a-z])')
_SEP_RE = re.compile(r'[_/\-.]')
_TOKEN_RE = re.compile(r'[a-zA-Z][a-zA-Z0-9]*')
_PY_REF_RE = re.compile(r'[\w/]+\.py\b')
_QUOTED_RE = re.compile(r'[`\'"](\w+)[`\'"]')
_DOTTED_RE = re.compile(r'\b\w+(?:\.\w+){2,}\b')
_IDENT_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def _split_camel(text: str) -> str:
    text = _CAMEL_RE_1.sub(r'\1 \2', text)
    text = _CAMEL_RE_2.sub(r'\1 \2', text)
    return text


def tokenize_code(text: str) -> List[str]:
    text = _split_camel(text)
    text = _SEP_RE.sub(' ', text)
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


def tokenize_path(path: str) -> List[str]:
    clean = re.sub(r'\.py$', '', path)
    parts = re.split(r'[/_\-.]', clean)
    tokens: List[str] = []
    for part in parts:
        sub = _split_camel(part)
        tokens.extend(sub.lower().split())
    return [t for t in tokens if len(t) > 1]


def tokenize_document(path: str, content: str) -> List[str]:
    path_tokens = tokenize_path(path) * 3
    return path_tokens + tokenize_code(content)


def tokenize_issue(issue_text: str) -> List[str]:
    """Spec §4 Phase 2: identifiers (len>=3) extracted from issue_text."""
    if not issue_text:
        return []
    # Identifier-style extraction on the raw text (preserves snake/Pascal case),
    # then camel-split and lowercase, then length >= 3 filter.
    idents = _IDENT_RE.findall(issue_text)
    expanded: List[str] = []
    for ident in idents:
        for tok in _split_camel(ident).lower().split():
            if len(tok) >= 3 and tok not in _STOPWORDS:
                expanded.append(tok)
    # Also include .py references (path tokens emphasised) for better recall
    for ref in _PY_REF_RE.findall(issue_text):
        for tok in tokenize_path(ref):
            if len(tok) >= 3:
                expanded.append(tok)
    # Quoted identifiers
    for q in _QUOTED_RE.findall(issue_text):
        q = q.lower()
        if len(q) >= 3 and q not in _STOPWORDS:
            expanded.append(q)
    return expanded


# ---------------------------------------------------------------------------
# Phase 1: Path-alias resolution
# ---------------------------------------------------------------------------
def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance. Pure-Python; O(len(a)*len(b)). Paths are short."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j, cb in enumerate(b, 1):
        curr = [j] + [0] * len(a)
        for i, ca in enumerate(a, 1):
            cost = 0 if ca == cb else 1
            curr[i] = min(prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost)
        prev = curr
    return prev[-1]


def resolve_path_aliases(
    gt_paths: Sequence[str],
    head_files_by_basename: Dict[str, List[str]],
    head_file_set: Set[str],
    repo_dir: str,
) -> Dict[str, Optional[str]]:
    """For each gt path find best HEAD equivalent. Spec §4 Phase 1.

    Acceptance: edit_distance(old, new) / len(old) < 0.5 AND file readable.
    If old path itself exists on HEAD (no reorg), return it directly.
    Returns {gt_path: resolved_path_or_None}.
    """
    result: Dict[str, Optional[str]] = {}
    for gt in gt_paths:
        # Direct hit: no reorganization for this file. Still require readability.
        if gt in head_file_set and is_readable_file(os.path.join(repo_dir, gt)):
            result[gt] = gt
            continue
        base = os.path.basename(gt)
        candidates = head_files_by_basename.get(base, [])
        if not candidates:
            result[gt] = None
            continue
        # 1. Longest suffix match wins (deterministic). Walks the gt path
        # right-to-left and finds HEAD files whose relative path ends with
        # "<suffix>"; picks the one with the longest suffix match. This is
        # the right fix for monorepo reorgs like cirq/foo/bar.py ->
        # cirq-core/cirq/foo/bar.py.
        parts = gt.split('/')
        best_suffix: Optional[str] = None
        best_suffix_len = 0
        for i in range(len(parts)):
            suffix = '/'.join(parts[i:])
            # Need a leading '/' to avoid matching 'bar/foo.py' -> 'ar/foo.py'.
            match_suffix = '/' + suffix if i > 0 else suffix
            matches = [
                c for c in candidates
                if c == suffix or c.endswith(match_suffix)
            ]
            if matches:
                # Tie-break deterministically: shortest then lex-sorted.
                matches.sort(key=lambda c: (len(c), c))
                if len(suffix) > best_suffix_len:
                    best_suffix = matches[0]
                    best_suffix_len = len(suffix)
                    break  # longest suffix search ran from shortest i first
        if best_suffix is not None and is_readable_file(
            os.path.join(repo_dir, best_suffix)
        ):
            result[gt] = best_suffix
            continue
        # 2. Fall back to basename-only edit-distance matching. Stricter
        # threshold (0.40) because this is much more likely to mismatch on
        # common basenames like __init__.py.
        scored = [(_edit_distance(gt, c), c) for c in sorted(candidates)]
        scored.sort(key=lambda x: (x[0], x[1]))
        best_d, best_c = scored[0]
        denom = max(len(gt), 1)
        if best_d / denom < 0.40 and is_readable_file(
            os.path.join(repo_dir, best_c)
        ):
            result[gt] = best_c
        else:
            result[gt] = None
    return result


# ---------------------------------------------------------------------------
# Phase 2: Per-repo BM25 index
# ---------------------------------------------------------------------------
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


def list_repo_py_files(repo_dir: str) -> List[str]:
    out: List[str] = []
    for root, dirs, files in os.walk(repo_dir, followlinks=False):
        dirs[:] = sorted(
            d for d in dirs if not d.startswith('.') and d != '__pycache__'
        )
        for name in sorted(files):
            if name.endswith('.py'):
                full = os.path.join(root, name)
                rel = os.path.relpath(full, repo_dir)
                out.append(rel)
    return out


def build_bm25_index(
    repo_dir: str, max_lines: int
) -> Tuple[Optional[BM25Okapi], List[str]]:
    py_files = list_repo_py_files(repo_dir)
    if not py_files:
        return None, []
    tokenized: List[List[str]] = []
    valid: List[str] = []
    for rel in py_files:
        content = read_head_py_file(os.path.join(repo_dir, rel), max_lines)
        toks = tokenize_document(rel, content)
        if toks:
            tokenized.append(toks)
            valid.append(rel)
    if not tokenized:
        return None, []
    return BM25Okapi(tokenized), valid


def bm25_query(
    bm25: BM25Okapi, files: List[str], query_tokens: List[str], top_k: int
) -> List[str]:
    if not query_tokens:
        return []
    scores = bm25.get_scores(query_tokens)
    order = np.argsort(-scores, kind='stable')  # stable tiebreak by index
    top = order[:top_k]
    return [files[int(i)] for i in top]


# ---------------------------------------------------------------------------
# Phase 3: Graph expansion
# ---------------------------------------------------------------------------
def load_graph(repo: str, graphs_dir: str) -> Dict[str, List[str]]:
    """Returns undirected 1-hop import neighbors: {file: [neighbors]}."""
    path = os.path.join(graphs_dir, f'{repo}_rels.json')
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    fi = obj.get('file_imports') or {}
    # Build undirected adjacency
    adj: Dict[str, Set[str]] = defaultdict(set)
    for src, dsts in fi.items():
        if not isinstance(dsts, list):
            continue
        for d in dsts:
            if not isinstance(d, str):
                continue
            if d == src:
                continue
            adj[src].add(d)
            adj[d].add(src)
    return {k: sorted(v) for k, v in adj.items()}


def graph_neighbors(
    seeds: Sequence[str], graph: Dict[str, List[str]], head_file_set: Set[str]
) -> List[str]:
    seen: List[str] = []
    seen_set: Set[str] = set()
    for s in seeds:
        for nb in graph.get(s, []):
            if nb in head_file_set and nb not in seen_set:
                seen.append(nb)
                seen_set.add(nb)
    return seen


# ---------------------------------------------------------------------------
# Phase 4: Assemble candidates
# ---------------------------------------------------------------------------
def assemble_candidates(
    resolved_gt: List[str],
    bm25_hits: List[str],
    graph_hits: List[str],
    samedir_pool: List[str],
    random_pool: List[str],
    target: int,
) -> Tuple[List[str], List[str]]:
    """Return (candidates, sources) preserving ordering: GT first, then mixed
    negatives in fixed-ratio slots, truncated to `target`.
    """
    cands: List[str] = []
    sources: List[str] = []
    seen: Set[str] = set()

    def _add(p: str, src: str) -> None:
        if p in seen:
            return
        cands.append(p)
        sources.append(src)
        seen.add(p)

    for g in resolved_gt:
        _add(g, 'gt')

    remaining = target - len(cands)
    if remaining <= 0:
        return cands[:target], sources[:target]

    # Slot budgets (fractions of the remaining budget).
    slot_plan = [
        ('bm25', 0.50, bm25_hits),
        ('graph', 0.20, graph_hits),
        ('samedir', 0.10, samedir_pool),
        ('random', 0.20, random_pool),
    ]
    # Compute integer budgets that sum exactly to `remaining`.
    raw = [int(math.floor(frac * remaining)) for _, frac, _ in slot_plan]
    assigned = sum(raw)
    # Distribute leftover 1-by-1 in listed order (bm25 gets extras first).
    leftover = remaining - assigned
    for i in range(leftover):
        raw[i % len(raw)] += 1
    budgets = {name: b for (name, _, _), b in zip(slot_plan, raw)}

    # Fill in order; unused budget cascades to random at the end.
    unused = 0
    for name, _, pool in slot_plan:
        want = budgets[name]
        got = 0
        for p in pool:
            if got >= want:
                break
            if p in seen:
                continue
            _add(p, name)
            got += 1
        unused += (want - got)

    # Cascade unused slots to random pool.
    if unused > 0:
        for p in random_pool:
            if unused <= 0:
                break
            if p in seen:
                continue
            _add(p, 'random')
            unused -= 1

    return cands[:target], sources[:target]


# ---------------------------------------------------------------------------
# Phase 5: Existence filter + random refill
# ---------------------------------------------------------------------------
def existence_filter(
    cands: List[str],
    sources: List[str],
    head_file_set: Set[str],
    repo_dir: str,
    random_pool: List[str],
    target: int,
) -> Tuple[List[str], List[str], int]:
    """Drop any candidate not on disk; refill from random_pool. Returns
    (filtered_cands, filtered_sources, n_purged)."""
    kept: List[str] = []
    kept_sources: List[str] = []
    seen: Set[str] = set()
    purged = 0
    for p, s in zip(cands, sources):
        if p in seen:
            continue
        # Readability check (spec §2 existence gate: "readable on disk").
        if p in head_file_set and is_readable_file(os.path.join(repo_dir, p)):
            kept.append(p)
            kept_sources.append(s)
            seen.add(p)
        else:
            purged += 1
    # Refill up to target from random_pool; require readability for fills too.
    if len(kept) < target:
        for p in random_pool:
            if len(kept) >= target:
                break
            if p in seen:
                continue
            if p in head_file_set and is_readable_file(os.path.join(repo_dir, p)):
                kept.append(p)
                kept_sources.append('random')
                seen.add(p)
    return kept[:target], kept_sources[:target], purged


# ---------------------------------------------------------------------------
# Per-repo processing
# ---------------------------------------------------------------------------
def process_repo(
    repo: str,
    issues: List[dict],
    repos_dir: str,
    graphs_dir: str,
    max_lines: int,
    target_cands: int,
    bm25_topn: int,
    write_alias_sidecar_dir: Optional[str],
    log_fh,
) -> Tuple[List[dict], Dict[str, float]]:
    repo_dir = os.path.join(repos_dir, repo)
    t0 = time.time()
    bm25, head_files = build_bm25_index(repo_dir, max_lines)
    if bm25 is None:
        msg = (
            f"  [WARN] {repo}: no .py files / no tokens; emitting "
            f"{len(issues)} placeholder records with n_candidates=0"
        )
        print(msg, flush=True)
        log_fh.write(msg + '\n'); log_fh.flush()
        placeholder: List[dict] = []
        for ex in issues:
            placeholder.append({
                'repo': repo,
                'issue_id': ex.get('issue_id', ex.get('instance_id', '')),
                'candidates': [],
                'candidate_sources': [],
                'gt_files': [],
                'gt_in_candidates': False,
                'gt_coverage_rate': 0.0,
                'n_candidates': 0,
                'build_seed': SEED,
            })
        return placeholder, {
            'repo': repo,
            'n_py_files': 0,
            'n_issues': len(issues),
            'n_gt_total': 0,
            'n_gt_resolved': 0,
            'alias_resolve_rate': 0.0,
            'any_gt_hit_rate': 0.0,
        }

    head_file_set = set(head_files)
    # Basename index for alias resolution.
    basename_index: Dict[str, List[str]] = defaultdict(list)
    for f in head_files:
        basename_index[os.path.basename(f)].append(f)
    # Directory index for samedir pool.
    dir_index: Dict[str, List[str]] = defaultdict(list)
    for f in head_files:
        dir_index[os.path.dirname(f)].append(f)

    graph = load_graph(repo, graphs_dir)
    idx_time = time.time() - t0
    msg = (
        f"  {repo}: {len(head_files)} .py files, index built in {idx_time:.1f}s, "
        f"graph={len(graph)} nodes, {len(issues)} issues"
    )
    print(msg, flush=True)
    log_fh.write(msg + '\n'); log_fh.flush()

    # Pre-compute pooled alias resolutions once per repo (for audit side-car).
    alias_audit: Dict[str, Optional[str]] = {}

    out: List[dict] = []
    # Track stats per repo
    n_gt_total = 0
    n_gt_resolved = 0
    n_samples_with_gt_hit = 0

    for ex in issues:
        gt_all: List[str] = list(ex.get('changed_py_files') or [])
        issue_text = ex.get('issue_text', '') or ''
        issue_id = ex.get('issue_id', ex.get('instance_id', ''))

        # Phase 1: resolve aliases
        alias_map = resolve_path_aliases(
            gt_all, basename_index, head_file_set, repo_dir
        )
        for k, v in alias_map.items():
            alias_audit.setdefault(k, v)
        resolved_gt: List[str] = []
        seen_resolved: Set[str] = set()
        for p in gt_all:  # preserve original order
            r = alias_map.get(p)
            if r and r not in seen_resolved:
                resolved_gt.append(r)
                seen_resolved.add(r)
        n_gt_total += len(gt_all)
        n_gt_resolved += len(resolved_gt)

        # Phase 2: BM25
        query_toks = tokenize_issue(issue_text)
        bm25_hits = bm25_query(bm25, head_files, query_toks, bm25_topn)

        # Phase 3: graph neighbors
        graph_hits = graph_neighbors(resolved_gt, graph, head_file_set)

        # Samedir pool: other .py files in resolved GT's directories.
        samedir_pool: List[str] = []
        samedir_seen: Set[str] = set()
        for g in resolved_gt:
            d = os.path.dirname(g)
            for f in dir_index.get(d, []):
                if f == g or f in samedir_seen:
                    continue
                samedir_pool.append(f)
                samedir_seen.add(f)

        # Random pool: deterministic per-sample rng seeded by a stable digest
        # so output is byte-identical across interpreter processes.
        rng_sample = random.Random(stable_seed(SEED, repo, issue_id))
        random_pool = list(head_files)
        rng_sample.shuffle(random_pool)

        # Phase 4: assemble
        cands, sources = assemble_candidates(
            resolved_gt=resolved_gt,
            bm25_hits=bm25_hits,
            graph_hits=graph_hits,
            samedir_pool=samedir_pool,
            random_pool=random_pool,
            target=target_cands,
        )

        # Phase 5: existence filter (head_file_set + disk check)
        cands, sources, purged = existence_filter(
            cands, sources, head_file_set, repo_dir, random_pool, target_cands
        )

        cand_set = set(cands)
        gt_hits = [g for g in resolved_gt if g in cand_set]
        if resolved_gt:
            coverage = len(gt_hits) / len(resolved_gt)
        else:
            coverage = 0.0
        any_hit = bool(gt_hits)
        if any_hit:
            n_samples_with_gt_hit += 1

        out.append({
            'repo': repo,
            'issue_id': issue_id,
            'candidates': cands,
            'candidate_sources': sources,
            'gt_files': resolved_gt,
            'gt_in_candidates': any_hit,
            'gt_coverage_rate': float(coverage),
            'n_candidates': len(cands),
            'build_seed': SEED,
        })

    # Side-car: path-alias audit for this repo.
    if write_alias_sidecar_dir:
        os.makedirs(write_alias_sidecar_dir, exist_ok=True)
        side = os.path.join(write_alias_sidecar_dir, f'path_alias_{repo}.json')
        try:
            with open(side, 'w') as f:
                json.dump(alias_audit, f, indent=2, sort_keys=True)
        except OSError as e:
            log_fh.write(f"  [WARN] side-car write failed for {repo}: {e}\n")

    stats = {
        'repo': repo,
        'n_py_files': len(head_files),
        'n_issues': len(issues),
        'n_gt_total': n_gt_total,
        'n_gt_resolved': n_gt_resolved,
        'alias_resolve_rate': (n_gt_resolved / n_gt_total) if n_gt_total else 1.0,
        'any_gt_hit_rate': (n_samples_with_gt_hit / len(issues)) if issues else 0.0,
    }
    return out, stats


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------
def _candidates_digest(results: List[dict]) -> str:
    """SHA-256 of a canonical serialization of (repo, issue_id, candidates).
    Exposed in the gate report so external reruns can byte-compare."""
    h = hashlib.sha256()
    for r in results:
        payload = json.dumps(
            {
                'repo': r.get('repo'),
                'issue_id': r.get('issue_id'),
                'candidates': r.get('candidates'),
            },
            ensure_ascii=False, sort_keys=True,
        )
        h.update(payload.encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()


def _verify_determinism(
    results: List[dict], expected_digest: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """Compute the canonical digest of this run. If `expected_digest` is
    provided (e.g. from a prior run via --expected_digest), compare and gate
    on equality. Otherwise the gate passes and the digest is published for an
    external rerun to check.

    Returns (ok, detail_str, computed_digest).
    """
    digest = _candidates_digest(results)
    if expected_digest is None:
        return True, f'digest_published={digest} (no expected baseline provided)', digest
    if digest == expected_digest:
        return True, f'digest_match={digest}', digest
    return (
        False,
        f'digest_mismatch computed={digest} expected={expected_digest}',
        digest,
    )


def evaluate_gates(
    results: List[dict], repos_dir: str, is_smoke: bool,
    per_repo_stats: Optional[List[Dict[str, object]]] = None,
    expected_digest: Optional[str] = None,
) -> Dict[str, object]:
    n = len(results)
    if n == 0:
        return {'all_gates_passed': False, 'reason': 'no_results'}

    cov = np.array([r['gt_coverage_rate'] for r in results])
    any_hit = np.array([float(r['gt_in_candidates']) for r in results])
    ncs = np.array([r['n_candidates'] for r in results])

    # Source mix aggregate (over all emitted candidates).
    src_counts: Dict[str, int] = defaultdict(int)
    src_total = 0
    for r in results:
        for s in r.get('candidate_sources', []):
            src_counts[s] += 1
            src_total += 1
    src_frac = {k: (v / src_total) for k, v in src_counts.items()} if src_total else {}

    # Existence check: full scan of every emitted candidate (spec §2 says
    # 100% of listed paths readable on disk — sampled audits cannot satisfy
    # that). Uses readability helper, not bare isfile.
    miss = 0
    total = 0
    for r in results:
        repo_dir = os.path.join(repos_dir, r['repo'])
        for c in r['candidates']:
            total += 1
            if not is_readable_file(os.path.join(repo_dir, c)):
                miss += 1

    n_zero_cands = int((ncs == 0).sum())
    n_below_10 = int((ncs < 10).sum())
    has_gt = [r for r in results if r.get('gt_files')]

    # Determinism gate (spec §2 "Re-run with seed=42 produces byte-identical
    # output"). A single in-process run cannot prove that; we instead publish
    # a canonical SHA-256 digest of (repo, issue_id, candidates) tuples. If
    # the caller supplies `expected_digest` (e.g. from a prior run), this
    # gate hard-fails on mismatch. Otherwise it passes and publishes the
    # digest so an external rerun can diff.
    determinism_ok, determinism_detail, computed_digest = _verify_determinism(
        results, expected_digest=expected_digest
    )

    # Per-repo alias audit (spec §5 smoke acceptance): each repo that has
    # any GT at all must achieve alias_resolve_rate >= 0.60. Repos with
    # n_gt_total == 0 are skipped from this check (no denominator).
    alias_audit_ok = True
    alias_audit_failing: List[Dict[str, object]] = []
    if per_repo_stats:
        for s in per_repo_stats:
            n_gt = int(s.get('n_gt_total') or 0)
            if n_gt == 0:
                continue
            rate = float(s.get('alias_resolve_rate') or 0.0)
            if rate < 0.60:
                alias_audit_ok = False
                alias_audit_failing.append({
                    'repo': s.get('repo'),
                    'alias_resolve_rate': rate,
                    'n_gt_total': n_gt,
                })

    if is_smoke:
        # Spec §5 smoke gates.
        gates = {
            # "70/100 samples have gt_in_candidates=True"
            'smoke_any_gt_hit_>=70': {
                'pass': bool(any_hit.sum() >= 70),
                'value': int(any_hit.sum()),
                'threshold': 70,
            },
            # "Mean n_candidates in [60,100]"
            'smoke_mean_n_candidates_in_[60,100]': {
                'pass': bool(60 <= ncs.mean() <= 100),
                'value': float(ncs.mean()),
                'threshold': [60, 100],
            },
            # "0 existence failures"
            'existence_100pct_readable': {
                'pass': bool(miss == 0),
                'value_miss': int(miss),
                'value_total': int(total),
            },
            'zero_candidate_samples_==0': {
                'pass': bool(n_zero_cands == 0),
                'value': int(n_zero_cands),
            },
            # Spec §5: "Path-alias audit: for each repo, >=60% of GT files
            # in the smoke set have a valid alias".
            'smoke_per_repo_alias_resolve_rate_>=60pct': {
                'pass': bool(alias_audit_ok),
                'value_failing_repos': alias_audit_failing,
                'threshold': 0.60,
            },
            'Determinism_seed42_stable': {
                'pass': bool(determinism_ok),
                'value': determinism_detail,
                'computed_digest': computed_digest,
                'expected_digest': expected_digest,
            },
        }
    else:
        # Full gates per spec §2 table.
        gates = {
            'GT_coverage_sample_level_>=80pct': {
                'pass': bool((cov > 0).mean() >= 0.80),
                'value': float((cov > 0).mean()),
                'threshold': 0.80,
            },
            'GT_file_recall_mean_>=0.70': {
                'pass': bool(cov.mean() >= 0.70) if len(cov) else False,
                'value': float(cov.mean()),
                'threshold': 0.70,
            },
            'Candidate_existence_100pct_readable': {
                'pass': bool(miss == 0),
                'value_miss': int(miss),
                'value_total': int(total),
            },
            'Candidates_per_sample_min_>=10': {
                'pass': bool(int(ncs.min()) >= 10),
                'value': int(ncs.min()),
                'n_below_10': int(n_below_10),
                'threshold': 10,
            },
            'Candidates_per_sample_mean_in_[80,100]': {
                'pass': bool(80 <= ncs.mean() <= 100),
                'value': float(ncs.mean()),
                'threshold': [80, 100],
            },
            'Candidates_per_sample_max_<=100': {
                'pass': bool(int(ncs.max()) <= 100),
                'value': int(ncs.max()),
                'threshold': 100,
            },
            'Zero_candidate_samples_==0': {
                'pass': bool(n_zero_cands == 0),
                'value': int(n_zero_cands),
            },
            'Hard_negative_source_mix': {
                'pass': bool(
                    0.40 <= src_frac.get('bm25', 0) <= 0.60 and
                    0.15 <= src_frac.get('graph', 0) <= 0.25 and
                    0.05 <= src_frac.get('samedir', 0) <= 0.15 and
                    0.10 <= src_frac.get('random', 0) <= 0.25
                ),
                'value': src_frac,
                'threshold': {
                    'bm25': [0.40, 0.60], 'graph': [0.15, 0.25],
                    'samedir': [0.05, 0.15], 'random': [0.10, 0.25],
                },
            },
            'Determinism_seed42_stable': {
                'pass': bool(determinism_ok),
                'value': determinism_detail,
                'computed_digest': computed_digest,
                'expected_digest': expected_digest,
                'note': (
                    'Spec §2 "byte-identical re-run" check: pass --expected_digest '
                    'with the value of computed_digest from a prior run; this '
                    'gate hard-fails on mismatch. Without an expected baseline, '
                    'the gate publishes the digest only.'
                ),
            },
            'Per_repo_alias_resolve_rate_>=60pct': {
                'pass': bool(alias_audit_ok),
                'value_failing_repos': alias_audit_failing,
                'threshold': 0.60,
            },
        }

    all_ok = all(g['pass'] for g in gates.values())
    return {
        'all_gates_passed': bool(all_ok),
        'n_samples': n,
        'is_smoke': is_smoke,
        'gates': gates,
        'summary': {
            'mean_coverage': float(cov.mean()),
            'any_hit_rate': float(any_hit.mean()),
            'n_any_hit': int(any_hit.sum()),
            'mean_n_candidates': float(ncs.mean()),
            'min_n_candidates': int(ncs.min()),
            'max_n_candidates': int(ncs.max()),
            'n_samples_with_gt': len(has_gt),
            'n_zero_cand_samples': n_zero_cands,
            'n_below_10_cand_samples': n_below_10,
            'source_mix': src_frac,
            'existence_miss': int(miss),
            'existence_total': int(total),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_jsonl',
                    default='/home/chenlibin/grepo_agent/data/grepo_text/grepo_train.jsonl')
    ap.add_argument('--repos_dir',
                    default='/home/chenlibin/grepo_agent/data/repos')
    ap.add_argument('--graphs_dir',
                    default='/home/chenlibin/grepo_agent/data/dep_graphs')
    ap.add_argument('--output',
                    default='/home/chenlibin/grepo_agent/data/rankft/grepo_train_head_candidates_v2.jsonl')
    ap.add_argument('--alias_dir',
                    default='/home/chenlibin/grepo_agent/data/rankft')
    ap.add_argument('--gate_report',
                    default='/home/chenlibin/grepo_agent/data/rankft/grepo_train_head_candidates_v2.gate_report.json')
    ap.add_argument('--target_cands', type=int, default=100)
    ap.add_argument('--bm25_topn', type=int, default=150)
    ap.add_argument('--max_lines', type=int, default=200)
    ap.add_argument('--smoke_n', type=int, default=0,
                    help='If >0, only process first N samples of grepo_train.')
    ap.add_argument('--expected_digest', default=None,
                    help='If set, the determinism gate hard-fails unless the '
                         'computed canonical digest matches this baseline. '
                         'Use the computed_digest emitted by a prior run.')
    args = ap.parse_args()

    print('=== GREPO Candidate Rebuild v2 (Strategy D) ===', flush=True)
    print(f'  train_jsonl : {args.train_jsonl}')
    print(f'  repos_dir   : {args.repos_dir}')
    print(f'  graphs_dir  : {args.graphs_dir}')
    print(f'  output      : {args.output}')
    print(f'  target_cands: {args.target_cands}')
    print(f'  smoke_n     : {args.smoke_n}')

    examples: List[dict] = []
    with open(args.train_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f'  loaded {len(examples)} train examples', flush=True)

    is_smoke = args.smoke_n > 0
    if is_smoke:
        examples = examples[: args.smoke_n]
        out_path = args.output + f'.smoke{args.smoke_n}.jsonl'
        gate_path = args.gate_report + f'.smoke{args.smoke_n}.json'
    else:
        out_path = args.output
        gate_path = args.gate_report

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log_path = out_path + '.log'
    log_fh = open(log_path, 'w')

    # Group by repo; process repos in deterministic sorted order (spec:
    # determinism requires sorted iteration over repos).
    per_repo: Dict[str, List[dict]] = defaultdict(list)
    example_order: Dict[int, int] = {}
    for i, ex in enumerate(examples):
        per_repo[ex['repo']].append(ex)
        example_order[id(ex)] = i

    repo_order = sorted(per_repo.keys())
    print(f'  {len(repo_order)} distinct repos (sorted)', flush=True)
    log_fh.write(f'[START] {len(examples)} examples across {len(repo_order)} repos\n')
    log_fh.flush()

    t_start = time.time()
    id_to_result: Dict[int, dict] = {}
    repo_stats: List[Dict[str, float]] = []
    for ri, repo in enumerate(repo_order):
        issues = per_repo[repo]
        print(
            f'\n[{ri+1}/{len(repo_order)}] repo={repo} '
            f'issues={len(issues)} elapsed={time.time()-t_start:.0f}s',
            flush=True,
        )
        log_fh.write(f'[{ri+1}/{len(repo_order)}] {repo} n={len(issues)}\n')
        log_fh.flush()
        results, stats = process_repo(
            repo=repo, issues=issues,
            repos_dir=args.repos_dir, graphs_dir=args.graphs_dir,
            max_lines=args.max_lines,
            target_cands=args.target_cands, bm25_topn=args.bm25_topn,
            write_alias_sidecar_dir=args.alias_dir, log_fh=log_fh,
        )
        repo_stats.append(stats)
        for ex, r in zip(issues, results):
            id_to_result[id(ex)] = r

    # Emit in original example order (stable).
    ordered: List[dict] = []
    for ex in examples:
        r = id_to_result.get(id(ex))
        if r is not None:
            ordered.append(r)

    with open(out_path, 'w') as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    elapsed = time.time() - t_start
    print(f'\nWrote {len(ordered)} records to {out_path} in {elapsed:.1f}s')
    log_fh.write(f'[DONE] {len(ordered)} records in {elapsed:.1f}s\n')

    # Gate evaluation & report.
    report = evaluate_gates(
        ordered, args.repos_dir, is_smoke=is_smoke,
        per_repo_stats=repo_stats, expected_digest=args.expected_digest,
    )
    report['per_repo_stats'] = repo_stats
    report['output_file'] = out_path
    report['elapsed_sec'] = float(elapsed)
    with open(gate_path, 'w') as gf:
        json.dump(report, gf, indent=2)
    print(f'Gate report -> {gate_path}')
    print(f'all_gates_passed={report.get("all_gates_passed")}')
    print(json.dumps(report.get('summary', {}), indent=2))

    log_fh.close()
    return 0 if report.get('all_gates_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
