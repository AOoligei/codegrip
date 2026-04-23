#!/usr/bin/env python3
"""
Build the feature-defined Code-Crucial v2 subset for the CodeGRIP paper.

Two-tier output:

Strict (main text): Code-Crucial if BOTH hold:
  1. Path-Hard-Strict: low_jaccard AND (same_stem OR path_misled_strict)
     - low_jaccard: issue-path Jaccard in bottom quartile
     - same_stem: another .py file IN THE FULL REPO shares the GT filename stem
     - path_misled_strict: under full-repo path Jaccard heuristic,
       best GT file is NOT in top-10 ranked files
  2. Code-Available-Strict:
     - ALL GT files resolvable on disk
     - >=50% of candidate files resolved
     - Code-only BM25 places at least one GT file in top-20

Broad (appendix): original looser definition
  1. Path-Hard: low_jaccard OR same_stem OR path_misled (old)
  2. Code-Available: GT in BM25 top-50 (old)

Inputs:
  - grepo_test.jsonl
  - merged_bm25_exp6_candidates.jsonl
  - data/file_trees/{repo}.json     (full repo file trees)
  - data/repos/{repo}/...           (source files)

Outputs:
  - data/code_crucial_v2_strict.jsonl
  - data/code_crucial_v2_broad.jsonl
"""

import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import random
random.seed(42)

# ─── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/chenlibin/grepo_agent")
TEST_PATH = BASE_DIR / "data" / "grepo_text" / "grepo_test.jsonl"
CAND_PATH = BASE_DIR / "data" / "rankft" / "merged_bm25_exp6_candidates.jsonl"
REPOS_DIR = BASE_DIR / "data" / "repos"
FILE_TREES_DIR = BASE_DIR / "data" / "file_trees"
OLD_CC_PATH = BASE_DIR / "data" / "code_crucial_subset.jsonl"
OUT_STRICT = BASE_DIR / "data" / "code_crucial_v2_strict.jsonl"
OUT_BROAD = BASE_DIR / "data" / "code_crucial_v2_broad.jsonl"


# ─── Tokenizer ───────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric token split."""
    return _TOKEN_RE.findall(text.lower())


def tokenize_path(path: str) -> list[str]:
    """Tokenize a file path: split on / . _ and camelCase."""
    s = path.replace("/", " ").replace("\\", " ").replace(".", " ").replace("_", " ")
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return tokenize(s)


def jaccard(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Jaccard similarity between two token lists."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ─── File tree loading ───────────────────────────────────────────────────
def load_file_tree(repo: str) -> list[str]:
    """Load the list of all .py files for a repo from the file_trees dir."""
    tree_path = FILE_TREES_DIR / f"{repo}.json"
    if not tree_path.exists():
        return []
    with open(tree_path) as f:
        data = json.load(f)
    return data.get("py_files", [])


# ─── File resolution ─────────────────────────────────────────────────────
def build_file_index(repo_dir: Path) -> dict[str, str]:
    """Build a mapping from relative suffix -> full path for a repo."""
    index = {}
    repo_str = str(repo_dir)
    for root, dirs, files in os.walk(repo_str):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for f in files:
            if not f.endswith(".py"):
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, repo_str)
            index[rel] = full
    return index


def resolve_file(filepath: str, repo_dir: Path, file_index: dict[str, str]) -> str | None:
    """Resolve a filepath to a full path on disk."""
    direct = repo_dir / filepath
    if direct.is_file():
        return str(direct)
    for rel, full in file_index.items():
        if rel.endswith(filepath) or rel == filepath:
            return full
    return None


def read_file_content(full_path: str, max_chars: int = 50000) -> str:
    """Read file content, with size limit."""
    try:
        with open(full_path, "r", errors="replace") as f:
            return f.read(max_chars)
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return ""


# ─── TF-IDF BM25 retriever ───────────────────────────────────────────────
class SimpleBM25:
    """Minimal BM25 (Okapi) scorer over a set of documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score_documents(
        self,
        query_tokens: list[str],
        doc_token_lists: list[list[str]],
    ) -> list[float]:
        """Return BM25 scores for each document given query tokens."""
        n_docs = len(doc_token_lists)
        if n_docs == 0:
            return []

        doc_lens = [len(d) for d in doc_token_lists]
        avgdl = sum(doc_lens) / n_docs if n_docs > 0 else 1.0

        query_terms = set(query_tokens)
        df = Counter()
        for dtokens in doc_token_lists:
            present = query_terms & set(dtokens)
            for t in present:
                df[t] += 1

        doc_tfs = []
        for dtokens in doc_token_lists:
            tf = Counter(dtokens)
            doc_tfs.append(tf)

        scores = []
        for i in range(n_docs):
            s = 0.0
            dl = doc_lens[i]
            for t in query_tokens:
                if df[t] == 0:
                    continue
                idf = math.log((n_docs - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
                tf_val = doc_tfs[i].get(t, 0)
                numerator = tf_val * (self.k1 + 1)
                denominator = tf_val + self.k1 * (1 - self.b + self.b * dl / avgdl)
                s += idf * numerator / denominator
            scores.append(s)
        return scores


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    print("Loading test data...")
    test_data = []
    with open(TEST_PATH) as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"  {len(test_data)} test examples")

    print("Loading candidates...")
    cand_map = {}  # (repo, issue_id) -> candidates list
    with open(CAND_PATH) as f:
        for line in f:
            d = json.loads(line)
            key = (d["repo"], d["issue_id"])
            cand_map[key] = d["candidates"]
    print(f"  {len(cand_map)} candidate entries")

    # Load old Code-Crucial for overlap comparison
    old_cc_keys = set()
    if OLD_CC_PATH.exists():
        with open(OLD_CC_PATH) as f:
            for line in f:
                d = json.loads(line)
                old_cc_keys.add((d["repo"], str(d["issue_id"])))
        print(f"  Old Code-Crucial: {len(old_cc_keys)} examples")

    # Load full-repo file trees (cached per repo)
    print("Loading file trees...")
    repo_py_files: dict[str, list[str]] = {}

    def get_repo_py_files(repo: str) -> list[str]:
        if repo not in repo_py_files:
            repo_py_files[repo] = load_file_tree(repo)
        return repo_py_files[repo]

    # Build per-repo file indexes (lazy, cached)
    print("Building file indexes per repo...")
    repo_file_indexes: dict[str, dict[str, str]] = {}

    def get_file_index(repo: str) -> dict[str, str]:
        if repo not in repo_file_indexes:
            repo_dir = REPOS_DIR / repo
            if repo_dir.is_dir():
                repo_file_indexes[repo] = build_file_index(repo_dir)
            else:
                repo_file_indexes[repo] = {}
        return repo_file_indexes[repo]

    # ─── Compute per-example features ────────────────────────────────────
    print("Computing features for each example...")
    bm25 = SimpleBM25()
    issue_path_jaccards = []

    # First pass: compute issue_path_jaccard for all examples to find Q1
    print("  Pass 1: computing issue-path Jaccard for all examples...")
    example_records = []
    for ex in test_data:
        repo = ex["repo"]
        issue_id = ex["issue_id"]
        issue_text = ex["issue_text"]
        gt_files = ex["changed_py_files"]
        key = (repo, issue_id)
        candidates = cand_map.get(key, [])

        if not candidates or not gt_files:
            continue

        issue_tokens = tokenize(issue_text)

        gt_jaccards = []
        for gt_f in gt_files:
            gt_path_tokens = tokenize_path(gt_f)
            j = jaccard(issue_tokens, gt_path_tokens)
            gt_jaccards.append(j)
        max_gt_jaccard = max(gt_jaccards) if gt_jaccards else 0.0
        issue_path_jaccards.append(max_gt_jaccard)

        example_records.append({
            "repo": repo,
            "issue_id": issue_id,
            "issue_text": issue_text,
            "gt_files": gt_files,
            "candidates": candidates,
            "issue_tokens": issue_tokens,
            "max_gt_jaccard": max_gt_jaccard,
        })

    # Compute Q1 threshold
    sorted_jaccards = sorted(issue_path_jaccards)
    n = len(sorted_jaccards)
    q1_idx = n // 4
    q1_threshold = sorted_jaccards[q1_idx] if n > 0 else 0.0
    print(f"  Jaccard Q1 threshold: {q1_threshold:.4f} (idx {q1_idx}/{n})")
    print(f"  Jaccard stats: min={sorted_jaccards[0]:.4f}, median={sorted_jaccards[n//2]:.4f}, "
          f"max={sorted_jaccards[-1]:.4f}")

    # Second pass: compute all features
    print("  Pass 2: computing all features (including code BM25)...")
    n_total = len(example_records)

    # Feature counters
    cnt_low_jaccard = 0
    cnt_same_stem_cand = 0      # old: same stem among candidates
    cnt_same_stem_repo = 0      # new: same stem among full repo
    cnt_path_misled_old = 0     # old: any non-GT candidate has higher Jaccard
    cnt_path_misled_strict = 0  # new: best GT not in top-10 of full-repo Jaccard ranking
    cnt_path_hard_broad = 0
    cnt_path_hard_strict = 0
    cnt_code_avail_broad = 0
    cnt_code_avail_strict = 0
    cnt_code_crucial_broad = 0
    cnt_code_crucial_strict = 0
    cnt_all_gt_resolved = 0
    cnt_file_not_found = 0
    resolved_fractions = []     # fraction of candidates resolved per example

    strict_records = []
    broad_records = []

    for idx, rec in enumerate(example_records):
        if (idx + 1) % 200 == 0:
            print(f"    [{idx+1}/{n_total}]")

        repo = rec["repo"]
        issue_id = rec["issue_id"]
        gt_files = rec["gt_files"]
        candidates = rec["candidates"]
        issue_tokens = rec["issue_tokens"]
        max_gt_jaccard = rec["max_gt_jaccard"]

        file_index = get_file_index(repo)
        repo_dir = REPOS_DIR / repo
        all_repo_py = get_repo_py_files(repo)
        gt_set = set(gt_files)

        # ── Condition 1a: Low Jaccard (bottom quartile) ──
        low_jaccard = max_gt_jaccard <= q1_threshold

        # ── Condition 1b: Same-stem ambiguity ──
        # Old: check against candidates only
        same_stem_cand = False
        cand_stems = defaultdict(list)
        for c in candidates:
            cand_stems[Path(c).stem].append(c)
        for gt_f in gt_files:
            stem = Path(gt_f).stem
            others = [c for c in cand_stems.get(stem, []) if c != gt_f]
            if others:
                same_stem_cand = True
                break

        # New: check against ALL .py files in the full repo
        same_stem_repo = False
        if all_repo_py:
            repo_stems = defaultdict(list)
            for py_f in all_repo_py:
                repo_stems[Path(py_f).stem].append(py_f)
            for gt_f in gt_files:
                stem = Path(gt_f).stem
                others = [f for f in repo_stems.get(stem, []) if f != gt_f]
                if others:
                    same_stem_repo = True
                    break

        # ── Condition 1c: Path-misled ──
        # Old: any non-GT candidate has higher Jaccard than best GT (among candidates)
        max_gt_path_score = 0.0
        max_non_gt_path_score = 0.0
        for c in candidates:
            c_tokens = tokenize_path(c)
            score = jaccard(issue_tokens, c_tokens)
            if c in gt_set:
                max_gt_path_score = max(max_gt_path_score, score)
            else:
                max_non_gt_path_score = max(max_non_gt_path_score, score)
        path_misled_old = max_non_gt_path_score > max_gt_path_score

        # New (path_misled_strict): under full-repo path Jaccard ranking,
        # best GT file is NOT in top-10
        path_misled_strict = False
        if all_repo_py:
            # Compute Jaccard(issue, path) for ALL repo .py files
            repo_path_scores = []
            for py_f in all_repo_py:
                py_tokens = tokenize_path(py_f)
                score = jaccard(issue_tokens, py_tokens)
                repo_path_scores.append((score, py_f))
            # Sort descending by score, then by path for determinism
            repo_path_scores.sort(key=lambda x: (-x[0], x[1]))
            top10_files = {f for _, f in repo_path_scores[:10]}
            # Check if ANY GT file appears in top-10
            best_gt_in_top10 = bool(gt_set & top10_files)
            path_misled_strict = not best_gt_in_top10

        # ── Path-Hard definitions ──
        # Broad (old): any of 3 conditions
        path_hard_broad = low_jaccard or same_stem_cand or path_misled_old
        # Strict (new): low_jaccard AND (same_stem_repo OR path_misled_strict)
        path_hard_strict = low_jaccard and (same_stem_repo or path_misled_strict)

        # ── Condition 2: Code-Available ──
        # Resolve files and compute BM25
        doc_tokens_list = []
        valid_candidates = []
        n_resolved = 0
        for c in candidates:
            full_path = resolve_file(c, repo_dir, file_index)
            if full_path is None:
                doc_tokens_list.append([])
                valid_candidates.append(c)
                cnt_file_not_found += 1
            else:
                content = read_file_content(full_path)
                doc_tokens_list.append(tokenize(content))
                valid_candidates.append(c)
                n_resolved += 1

        n_cand_total = len(candidates)
        resolved_frac = n_resolved / n_cand_total if n_cand_total > 0 else 0.0
        resolved_fractions.append(resolved_frac)

        # Check if ALL GT files are resolvable
        all_gt_resolved = True
        for gt_f in gt_files:
            if resolve_file(gt_f, repo_dir, file_index) is None:
                all_gt_resolved = False
                break
        if all_gt_resolved:
            cnt_all_gt_resolved += 1

        # BM25 scoring (code-only, over resolved candidates)
        # For strict: only use candidates that were actually resolved
        # Build resolved-only doc list for strict BM25
        resolved_doc_tokens = []
        resolved_candidates = []
        for i, c in enumerate(valid_candidates):
            if doc_tokens_list[i]:  # non-empty means resolved
                resolved_doc_tokens.append(doc_tokens_list[i])
                resolved_candidates.append(c)

        # Broad code-available: BM25 over ALL candidates (including unresolved), top-50
        scores_broad = bm25.score_documents(issue_tokens, doc_tokens_list)
        scored_broad = list(zip(scores_broad, valid_candidates))
        scored_broad.sort(key=lambda x: (-x[0], x[1]))
        top50_files = {c for _, c in scored_broad[:50]}
        code_avail_broad = bool(gt_set & top50_files)

        # Strict code-available:
        #   - ALL GT files resolvable
        #   - >=50% of candidates resolved
        #   - BM25 over resolved-only candidates, GT in top-20
        code_avail_strict = False
        if all_gt_resolved and resolved_frac >= 0.5 and resolved_candidates:
            scores_strict = bm25.score_documents(issue_tokens, resolved_doc_tokens)
            scored_strict = list(zip(scores_strict, resolved_candidates))
            scored_strict.sort(key=lambda x: (-x[0], x[1]))
            top20_files = {c for _, c in scored_strict[:20]}
            code_avail_strict = bool(gt_set & top20_files)

        # ── Final definitions ──
        code_crucial_broad = path_hard_broad and code_avail_broad
        code_crucial_strict = path_hard_strict and code_avail_strict

        # Count features
        if low_jaccard:
            cnt_low_jaccard += 1
        if same_stem_cand:
            cnt_same_stem_cand += 1
        if same_stem_repo:
            cnt_same_stem_repo += 1
        if path_misled_old:
            cnt_path_misled_old += 1
        if path_misled_strict:
            cnt_path_misled_strict += 1
        if path_hard_broad:
            cnt_path_hard_broad += 1
        if path_hard_strict:
            cnt_path_hard_strict += 1
        if code_avail_broad:
            cnt_code_avail_broad += 1
        if code_avail_strict:
            cnt_code_avail_strict += 1
        if code_crucial_broad:
            cnt_code_crucial_broad += 1
        if code_crucial_strict:
            cnt_code_crucial_strict += 1

        record = {
            "repo": repo,
            "issue_id": issue_id,
            "low_jaccard": low_jaccard,
            "same_stem_cand": same_stem_cand,
            "same_stem_repo": same_stem_repo,
            "path_misled_old": path_misled_old,
            "path_misled_strict": path_misled_strict,
            "path_hard_broad": path_hard_broad,
            "path_hard_strict": path_hard_strict,
            "code_avail_broad": code_avail_broad,
            "code_avail_strict": code_avail_strict,
            "max_gt_jaccard": round(max_gt_jaccard, 4),
            "resolved_frac": round(resolved_frac, 4),
            "all_gt_resolved": all_gt_resolved,
        }

        if code_crucial_strict:
            strict_records.append(record)
        if code_crucial_broad:
            broad_records.append(record)

    # ─── Output ──────────────────────────────────────────────────────────
    # Sort for determinism
    strict_records.sort(key=lambda x: (x["repo"], x["issue_id"]))
    broad_records.sort(key=lambda x: (x["repo"], x["issue_id"]))

    with open(OUT_STRICT, "w") as f:
        for rec in strict_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(OUT_BROAD, "w") as f:
        for rec in broad_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ─── Statistics ──────────────────────────────────────────────────────
    pct = lambda x: f"{100*x/max(n_total,1):.1f}%"

    print(f"\n{'='*70}")
    print(f"CODE-CRUCIAL v2 STATISTICS")
    print(f"{'='*70}")
    print(f"Total test examples (with candidates): {n_total}")

    print(f"\n--- Individual Feature Counts ---")
    print(f"  low_jaccard (Q1 <= {q1_threshold:.4f}):  {cnt_low_jaccard:5d}  ({pct(cnt_low_jaccard)})")
    print(f"  same_stem (candidates only):             {cnt_same_stem_cand:5d}  ({pct(cnt_same_stem_cand)})")
    print(f"  same_stem (full repo):                   {cnt_same_stem_repo:5d}  ({pct(cnt_same_stem_repo)})")
    print(f"  path_misled (old, candidate-based):      {cnt_path_misled_old:5d}  ({pct(cnt_path_misled_old)})")
    print(f"  path_misled_strict (full repo, top-10):  {cnt_path_misled_strict:5d}  ({pct(cnt_path_misled_strict)})")

    print(f"\n--- Path-Hard ---")
    print(f"  path_hard_broad  (OR of 3):              {cnt_path_hard_broad:5d}  ({pct(cnt_path_hard_broad)})")
    print(f"  path_hard_strict (AND logic):            {cnt_path_hard_strict:5d}  ({pct(cnt_path_hard_strict)})")

    print(f"\n--- Code-Available ---")
    print(f"  code_avail_broad  (top-50, any resolve): {cnt_code_avail_broad:5d}  ({pct(cnt_code_avail_broad)})")
    print(f"  code_avail_strict (top-20, all GT res):  {cnt_code_avail_strict:5d}  ({pct(cnt_code_avail_strict)})")
    print(f"  all GT files resolved on disk:           {cnt_all_gt_resolved:5d}  ({pct(cnt_all_gt_resolved)})")

    print(f"\n--- Code-Crucial (final) ---")
    print(f"  STRICT (main text):  {cnt_code_crucial_strict:5d}  ({pct(cnt_code_crucial_strict)})")
    print(f"  BROAD  (appendix):   {cnt_code_crucial_broad:5d}  ({pct(cnt_code_crucial_broad)})")

    # Coverage stats
    if resolved_fractions:
        avg_res = sum(resolved_fractions) / len(resolved_fractions)
        sorted_res = sorted(resolved_fractions)
        med_res = sorted_res[len(sorted_res) // 2]
        low_res = sum(1 for r in resolved_fractions if r < 0.5)
        print(f"\n--- Coverage Stats ---")
        print(f"  Candidate resolve fraction: mean={avg_res:.3f}, median={med_res:.3f}")
        print(f"  Examples with <50% resolved: {low_res} ({pct(low_res)})")
        print(f"  Total file-not-found events: {cnt_file_not_found}")

    # Overlap with old model-defined Code-Crucial
    if old_cc_keys:
        strict_keys = {(r["repo"], str(r["issue_id"])) for r in strict_records}
        broad_keys = {(r["repo"], str(r["issue_id"])) for r in broad_records}

        print(f"\n--- Overlap with Old Code-Crucial ({len(old_cc_keys)} examples) ---")
        for label, new_keys in [("strict", strict_keys), ("broad", broad_keys)]:
            overlap = old_cc_keys & new_keys
            only_old = old_cc_keys - new_keys
            only_new = new_keys - old_cc_keys
            union = old_cc_keys | new_keys
            print(f"  {label}:")
            print(f"    Intersection:  {len(overlap)}")
            print(f"    Only old:      {len(only_old)}")
            print(f"    Only new (v2): {len(only_new)}")
            print(f"    Jaccard(old, new): {len(overlap)/len(union):.3f}" if union else "    Jaccard: N/A")

    # Overlap between strict and broad
    strict_keys = {(r["repo"], str(r["issue_id"])) for r in strict_records}
    broad_keys = {(r["repo"], str(r["issue_id"])) for r in broad_records}
    print(f"\n--- Strict vs Broad ---")
    print(f"  strict subset of broad: {strict_keys <= broad_keys}")
    print(f"  broad - strict: {len(broad_keys - strict_keys)}")

    # Breakdown within strict
    if strict_records:
        n_s = len(strict_records)
        n_lj = sum(1 for r in strict_records if r["low_jaccard"])
        n_ss = sum(1 for r in strict_records if r["same_stem_repo"])
        n_pm = sum(1 for r in strict_records if r["path_misled_strict"])
        n_ss_only = sum(1 for r in strict_records if r["same_stem_repo"] and not r["path_misled_strict"])
        n_pm_only = sum(1 for r in strict_records if r["path_misled_strict"] and not r["same_stem_repo"])
        n_both = sum(1 for r in strict_records if r["same_stem_repo"] and r["path_misled_strict"])
        print(f"\n--- Path-Hard Breakdown within Strict ({n_s}) ---")
        print(f"  low_jaccard:        {n_lj:5d}  (all, by definition)")
        print(f"  same_stem_repo:     {n_ss:5d}  ({100*n_ss/n_s:.1f}%)")
        print(f"  path_misled_strict: {n_pm:5d}  ({100*n_pm/n_s:.1f}%)")
        print(f"  same_stem only:     {n_ss_only:5d}")
        print(f"  misled only:        {n_pm_only:5d}")
        print(f"  both:               {n_both:5d}")

    print(f"\nSaved strict to: {OUT_STRICT}")
    print(f"Saved broad  to: {OUT_BROAD}")


if __name__ == "__main__":
    main()
