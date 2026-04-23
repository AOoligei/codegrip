"""
Analyze correlation between file path tokens and code content tokens in GREPO dataset.
Supports the narrative: "path dominance is expected because paths encode code structure."

Three analyses:
1. Identifier overlap: fraction of code identifiers recoverable from path
2. Path predicts code content: Spearman correlation of BM25 rankings
3. Code-signal conditioned on path overlap: does code help more when path fails?
"""

import json
import os
import re
import ast
import sys
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Set, Tuple, Dict, Optional
import random
import numpy as np
from scipy import stats

random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path("/home/chenlibin/grepo_agent")
TEST_FILE = BASE_DIR / "data/grepo_text/grepo_test.jsonl"
CANDIDATES_FILE = BASE_DIR / "data/rankft/merged_bm25_exp6_candidates.jsonl"
REPOS_DIR = BASE_DIR / "data/repos"
PREDICTIONS_FILE = Path("/data/chenlibin/grepo_agent_experiments/multiseed_seed42/eval_graph_rerank/predictions.jsonl")
OUTPUT_FILE = BASE_DIR / "data/analysis/path_code_correlation.json"


# ──────────────────────────────────────────────
# Token extraction utilities
# ──────────────────────────────────────────────
def tokenize_path(filepath: str) -> Set[str]:
    """Extract tokens from file path (directories + file stem), splitting on separators and camelCase."""
    parts = Path(filepath).parts
    stem = Path(filepath).stem
    raw_tokens = []
    for part in parts:
        raw_tokens.extend(re.split(r'[_./\-]', part))
    raw_tokens.extend(re.split(r'[_./\-]', stem))
    # Also split camelCase
    expanded = []
    for tok in raw_tokens:
        expanded.extend(split_camel(tok))
    return {t.lower() for t in expanded if len(t) >= 2}


def split_camel(s: str) -> List[str]:
    """Split camelCase and PascalCase strings."""
    tokens = re.sub(r'([A-Z][a-z]+)', r' \1', re.sub(r'([A-Z]+)', r' \1', s)).split()
    return [t for t in tokens if t]


def extract_code_identifiers(source: str) -> Set[str]:
    """Extract class/function/variable names from Python source using AST, with regex fallback."""
    identifiers = set()
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                identifiers.add(node.name)
            elif isinstance(node, ast.ClassDef):
                identifiers.add(node.name)
            elif isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
    except SyntaxError:
        # Regex fallback
        identifiers.update(re.findall(r'(?:def|class)\s+(\w+)', source))
        identifiers.update(re.findall(r'\b([a-zA-Z_]\w{2,})\b', source))

    # Expand identifiers by splitting camelCase/underscore
    expanded = set()
    for ident in identifiers:
        parts = re.split(r'_', ident)
        for part in parts:
            for sub in split_camel(part):
                if len(sub) >= 2:
                    expanded.add(sub.lower())
    return expanded


def tokenize_text(text: str) -> List[str]:
    """Tokenize text for BM25."""
    text = text.lower()
    tokens = re.findall(r'[a-z][a-z0-9]{1,}', text)
    # Also split camelCase from original
    camel_tokens = re.findall(r'[A-Za-z][a-zA-Z0-9]+', text)
    for tok in camel_tokens:
        for sub in split_camel(tok):
            if len(sub) >= 2:
                tokens.append(sub.lower())
    return tokens


def find_file(repo: str, relpath: str) -> Optional[str]:
    """Find actual file path given repo name and relative path from candidates."""
    repo_dir = REPOS_DIR / repo
    direct = repo_dir / relpath
    if direct.exists():
        return str(direct)
    # Search for file ending with relpath
    for root, dirs, files in os.walk(repo_dir):
        full = os.path.join(root, relpath)
        if os.path.isfile(full):
            return full
        # Try matching the end of path
        rp = os.path.relpath(root, repo_dir)
        if rp.endswith(os.path.dirname(relpath)):
            candidate = os.path.join(root, os.path.basename(relpath))
            if os.path.isfile(candidate):
                return candidate
    # Last resort: find by filename
    basename = os.path.basename(relpath)
    for root, dirs, files in os.walk(repo_dir):
        if basename in files:
            full = os.path.join(root, basename)
            if full.endswith(relpath):
                return full
    return None


# ──────────────────────────────────────────────
# BM25 implementation
# ──────────────────────────────────────────────
class BM25:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / max(self.N, 1)
        self.df = Counter()
        for doc in corpus:
            unique = set(doc)
            for term in unique:
                self.df[term] += 1

    def score(self, query: List[str], doc_idx: int) -> float:
        doc = self.corpus[doc_idx]
        doc_len = len(doc)
        tf = Counter(doc)
        score = 0.0
        for term in query:
            if term not in tf:
                continue
            n = self.df.get(term, 0)
            idf = math.log((self.N - n + 0.5) / (n + 0.5) + 1)
            term_tf = tf[term]
            tf_norm = (term_tf * (self.k1 + 1)) / (term_tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            score += idf * tf_norm
        return score

    def rank(self, query: List[str]) -> List[Tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: -x[1])
        return scores


# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


print("Loading data...")
test_data = load_jsonl(TEST_FILE)
candidates_data = load_jsonl(CANDIDATES_FILE)
predictions_data = load_jsonl(PREDICTIONS_FILE)

# Build lookup dicts
candidates_by_key = {}
for c in candidates_data:
    key = (c["repo"], c["issue_id"])
    candidates_by_key[key] = c["candidates"]

predictions_by_key = {}
for p in predictions_data:
    key = (p["repo"], p.get("issue_id", str(p.get("issue_id"))))
    predictions_by_key[key] = p

# Also try string key for predictions
pred_by_key2 = {}
for p in predictions_data:
    key = (p["repo"], int(p["issue_id"]) if isinstance(p["issue_id"], str) else p["issue_id"])
    pred_by_key2[key] = p

print(f"Test examples: {len(test_data)}")
print(f"Candidates entries: {len(candidates_data)}")
print(f"Predictions entries: {len(predictions_data)}")


# ──────────────────────────────────────────────
# Analysis 1: Identifier overlap
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("Analysis 1: Identifier Overlap (Path tokens vs Code identifiers)")
print("="*60)

overlap_fractions = []  # fraction of code identifiers in path tokens
reverse_overlap = []     # fraction of path tokens in code identifiers
file_count = 0
files_checked = 0
files_not_found = 0

# Sample to keep it tractable
sampled_test = random.sample(test_data, min(500, len(test_data)))

for example in sampled_test:
    repo = example["repo"]
    key = (repo, example["issue_id"])
    cands = candidates_by_key.get(key, [])

    # Use GT files + sample of candidates
    gt_files = set(example.get("changed_py_files", example.get("changed_files", [])))
    sample_cands = list(gt_files) + random.sample(cands, min(20, len(cands)))
    sample_cands = list(set(sample_cands))

    for filepath in sample_cands:
        if not filepath.endswith(".py"):
            continue
        files_checked += 1
        actual_path = find_file(repo, filepath)
        if actual_path is None:
            files_not_found += 1
            continue

        try:
            with open(actual_path, 'r', errors='ignore') as f:
                source = f.read()
        except Exception:
            continue

        path_tokens = tokenize_path(filepath)
        code_idents = extract_code_identifiers(source)

        if len(code_idents) == 0:
            continue

        # What fraction of code identifiers are in path tokens?
        overlap = len(code_idents & path_tokens) / len(code_idents)
        overlap_fractions.append(overlap)

        # What fraction of path tokens are in code identifiers?
        if len(path_tokens) > 0:
            rev = len(path_tokens & code_idents) / len(path_tokens)
            reverse_overlap.append(rev)

        file_count += 1

print(f"Files analyzed: {file_count} (not found: {files_not_found}/{files_checked})")
print(f"\nFraction of CODE identifiers recoverable from PATH:")
print(f"  Mean:   {np.mean(overlap_fractions):.4f}")
print(f"  Median: {np.median(overlap_fractions):.4f}")
print(f"  Std:    {np.std(overlap_fractions):.4f}")
print(f"  P25:    {np.percentile(overlap_fractions, 25):.4f}")
print(f"  P75:    {np.percentile(overlap_fractions, 75):.4f}")

print(f"\nFraction of PATH tokens found in CODE identifiers:")
print(f"  Mean:   {np.mean(reverse_overlap):.4f}")
print(f"  Median: {np.median(reverse_overlap):.4f}")
print(f"  Std:    {np.std(reverse_overlap):.4f}")


# ──────────────────────────────────────────────
# Analysis 2: Path predicts Code (BM25 ranking correlation)
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("Analysis 2: BM25 Ranking Correlation (Path vs Code)")
print("="*60)

spearman_correlations = []
kendall_correlations = []
top10_overlaps = []
examples_processed = 0

# Sample for tractability
sampled_test_2 = random.sample(test_data, min(300, len(test_data)))

for example in sampled_test_2:
    repo = example["repo"]
    key = (repo, example["issue_id"])
    cands = candidates_by_key.get(key, [])

    if len(cands) < 10:
        continue

    # Limit candidates for speed
    cands = cands[:50]

    issue_tokens = tokenize_text(example["issue_text"])
    if len(issue_tokens) < 3:
        continue

    # Build path-token corpus and code-token corpus
    path_corpus = []
    code_corpus = []
    valid_cands = []

    for filepath in cands:
        path_tokens = list(tokenize_path(filepath))

        # Try to read file for code tokens
        actual_path = find_file(repo, filepath)
        if actual_path is None:
            continue
        try:
            with open(actual_path, 'r', errors='ignore') as f:
                source = f.read()[:5000]  # first 5000 chars for speed
            code_tokens = tokenize_text(source)
        except Exception:
            continue

        if len(path_tokens) == 0 or len(code_tokens) == 0:
            continue

        path_corpus.append(path_tokens)
        code_corpus.append(code_tokens)
        valid_cands.append(filepath)

    if len(valid_cands) < 10:
        continue

    # BM25 rankings
    bm25_path = BM25(path_corpus)
    bm25_code = BM25(code_corpus)

    path_scores = [bm25_path.score(issue_tokens, i) for i in range(len(valid_cands))]
    code_scores = [bm25_code.score(issue_tokens, i) for i in range(len(valid_cands))]

    # Spearman correlation
    if len(set(path_scores)) > 1 and len(set(code_scores)) > 1:
        rho, pval = stats.spearmanr(path_scores, code_scores)
        if not np.isnan(rho):
            spearman_correlations.append(rho)

            tau, _ = stats.kendalltau(path_scores, code_scores)
            if not np.isnan(tau):
                kendall_correlations.append(tau)

    # Top-10 overlap
    path_ranking = sorted(range(len(valid_cands)), key=lambda i: -path_scores[i])[:10]
    code_ranking = sorted(range(len(valid_cands)), key=lambda i: -code_scores[i])[:10]
    overlap = len(set(path_ranking) & set(code_ranking)) / 10
    top10_overlaps.append(overlap)

    examples_processed += 1

print(f"Examples processed: {examples_processed}")
print(f"\nSpearman correlation between PATH-BM25 and CODE-BM25 rankings:")
print(f"  Mean:   {np.mean(spearman_correlations):.4f}")
print(f"  Median: {np.median(spearman_correlations):.4f}")
print(f"  Std:    {np.std(spearman_correlations):.4f}")
print(f"  Fraction > 0.5: {np.mean([r > 0.5 for r in spearman_correlations]):.4f}")
print(f"  Fraction > 0.7: {np.mean([r > 0.7 for r in spearman_correlations]):.4f}")

print(f"\nKendall tau:")
print(f"  Mean:   {np.mean(kendall_correlations):.4f}")
print(f"  Median: {np.median(kendall_correlations):.4f}")

print(f"\nTop-10 candidate overlap (PATH-BM25 vs CODE-BM25):")
print(f"  Mean:   {np.mean(top10_overlaps):.4f}")
print(f"  Median: {np.median(top10_overlaps):.4f}")


# ──────────────────────────────────────────────
# Analysis 3: Code-signal conditioned on path overlap
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("Analysis 3: Code Signal Conditioned on Path-Issue Overlap")
print("="*60)

# For each test example, compute path-token overlap between issue and GT files
# Then bin by overlap level and compare path-only vs graph-rerank R@1

bins = {"high": [], "medium": [], "low": []}
bin_path_r1 = {"high": [], "medium": [], "low": []}
bin_rerank_r1 = {"high": [], "medium": [], "low": []}

for example in test_data:
    repo = example["repo"]
    key = (repo, example["issue_id"])

    gt_files = set(example.get("changed_py_files", example.get("changed_files", [])))
    if not gt_files:
        continue

    issue_tokens = set(tokenize_text(example["issue_text"]))

    # Compute path token overlap between issue and GT files
    gt_path_tokens = set()
    for f in gt_files:
        gt_path_tokens.update(tokenize_path(f))

    if len(gt_path_tokens) == 0:
        continue

    overlap_ratio = len(issue_tokens & gt_path_tokens) / len(gt_path_tokens)

    # Classify into bins
    if overlap_ratio >= 0.5:
        bin_name = "high"
    elif overlap_ratio >= 0.2:
        bin_name = "medium"
    else:
        bin_name = "low"

    bins[bin_name].append(overlap_ratio)

    # Check BM25 path-only R@1 (using original BM25 ranking from candidates data)
    cands = candidates_by_key.get(key, [])
    if len(cands) > 0:
        # BM25 original is the candidate list order (already BM25-ranked)
        bm25_top1 = set(cands[:1])
        bm25_hit = 1 if len(bm25_top1 & gt_files) > 0 else 0
        bin_path_r1[bin_name].append(bm25_hit)

    # Check graph-rerank R@1
    pred = pred_by_key2.get(key)
    if pred is not None:
        pred_top1 = set(pred["predicted"][:1])
        rerank_hit = 1 if len(pred_top1 & gt_files) > 0 else 0
        bin_rerank_r1[bin_name].append(rerank_hit)

print("\nPath-Issue token overlap distribution:")
for bin_name in ["high", "medium", "low"]:
    n = len(bins[bin_name])
    mean_overlap = np.mean(bins[bin_name]) if n > 0 else 0
    print(f"  {bin_name:8s}: N={n:5d}, mean overlap={mean_overlap:.4f}")

print("\nBM25 (path-dominant) R@1 by bin:")
for bin_name in ["high", "medium", "low"]:
    vals = bin_path_r1[bin_name]
    r1 = np.mean(vals) if len(vals) > 0 else 0
    print(f"  {bin_name:8s}: R@1={r1:.4f} (N={len(vals)})")

print("\nGraph-Rerank R@1 by bin:")
for bin_name in ["high", "medium", "low"]:
    vals = bin_rerank_r1[bin_name]
    r1 = np.mean(vals) if len(vals) > 0 else 0
    print(f"  {bin_name:8s}: R@1={r1:.4f} (N={len(vals)})")

print("\nDelta (Rerank - BM25) by bin:")
for bin_name in ["high", "medium", "low"]:
    bm25_r1 = np.mean(bin_path_r1[bin_name]) if len(bin_path_r1[bin_name]) > 0 else 0
    rerank_r1 = np.mean(bin_rerank_r1[bin_name]) if len(bin_rerank_r1[bin_name]) > 0 else 0
    delta = rerank_r1 - bm25_r1
    print(f"  {bin_name:8s}: Delta={delta:+.4f}")


# ──────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────
results = {
    "analysis_1_identifier_overlap": {
        "n_files": file_count,
        "code_from_path": {
            "mean": round(float(np.mean(overlap_fractions)), 4),
            "median": round(float(np.median(overlap_fractions)), 4),
            "std": round(float(np.std(overlap_fractions)), 4),
            "p25": round(float(np.percentile(overlap_fractions, 25)), 4),
            "p75": round(float(np.percentile(overlap_fractions, 75)), 4),
        },
        "path_in_code": {
            "mean": round(float(np.mean(reverse_overlap)), 4),
            "median": round(float(np.median(reverse_overlap)), 4),
            "std": round(float(np.std(reverse_overlap)), 4),
        },
    },
    "analysis_2_bm25_ranking_correlation": {
        "n_examples": examples_processed,
        "spearman": {
            "mean": round(float(np.mean(spearman_correlations)), 4),
            "median": round(float(np.median(spearman_correlations)), 4),
            "std": round(float(np.std(spearman_correlations)), 4),
            "frac_gt_0.5": round(float(np.mean([r > 0.5 for r in spearman_correlations])), 4),
            "frac_gt_0.7": round(float(np.mean([r > 0.7 for r in spearman_correlations])), 4),
        },
        "kendall_tau": {
            "mean": round(float(np.mean(kendall_correlations)), 4),
            "median": round(float(np.median(kendall_correlations)), 4),
        },
        "top10_overlap": {
            "mean": round(float(np.mean(top10_overlaps)), 4),
            "median": round(float(np.median(top10_overlaps)), 4),
        },
    },
    "analysis_3_conditioned_on_path_overlap": {
        bin_name: {
            "n_examples": len(bins[bin_name]),
            "mean_overlap": round(float(np.mean(bins[bin_name])), 4) if bins[bin_name] else 0,
            "bm25_r1": round(float(np.mean(bin_path_r1[bin_name])), 4) if bin_path_r1[bin_name] else 0,
            "rerank_r1": round(float(np.mean(bin_rerank_r1[bin_name])), 4) if bin_rerank_r1[bin_name] else 0,
            "delta": round(float(np.mean(bin_rerank_r1[bin_name]) - np.mean(bin_path_r1[bin_name])), 4) if bin_path_r1[bin_name] and bin_rerank_r1[bin_name] else 0,
        }
        for bin_name in ["high", "medium", "low"]
    },
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUTPUT_FILE}")


# ──────────────────────────────────────────────
# LaTeX summary
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("LaTeX Summary")
print("="*60)

a1 = results["analysis_1_identifier_overlap"]
a2 = results["analysis_2_bm25_ranking_correlation"]
a3 = results["analysis_3_conditioned_on_path_overlap"]

latex = r"""
\begin{table}[t]
\centering
\caption{Path--code token correlation in GREPO. File paths encode substantial code structure information, explaining why path-based features dominate.}
\label{tab:path-code-correlation}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
\multicolumn{2}{l}{\textit{Analysis 1: Identifier Overlap}} \\
Code identifiers recoverable from path (mean) & """ + f"{a1['code_from_path']['mean']:.1%}" + r""" \\
Path tokens found in code identifiers (mean) & """ + f"{a1['path_in_code']['mean']:.1%}" + r""" \\
\midrule
\multicolumn{2}{l}{\textit{Analysis 2: BM25 Ranking Correlation}} \\
Spearman $\rho$ (path vs.\ code BM25) & """ + f"{a2['spearman']['mean']:.3f}" + r""" \\
Fraction with $\rho > 0.5$ & """ + f"{a2['spearman']['frac_gt_0.5']:.1%}" + r""" \\
Top-10 candidate overlap & """ + f"{a2['top10_overlap']['mean']:.1%}" + r""" \\
\midrule
\multicolumn{2}{l}{\textit{Analysis 3: R@1 by Path--Issue Overlap}} \\
""" + "\n".join([
    f"\\quad {bn.capitalize()} overlap ($n={a3[bn]['n_examples']}$): BM25 / Rerank & "
    f"{a3[bn]['bm25_r1']:.1%} / {a3[bn]['rerank_r1']:.1%} ($\\Delta={a3[bn]['delta']:+.1%}$) \\\\"
    for bn in ["high", "medium", "low"]
]) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

print(latex)
print("\nDone.")
