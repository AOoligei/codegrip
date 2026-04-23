#!/usr/bin/env python3
"""
Code-chunk retrieval baselines for bug localization.

Two modes:
  bm25_code   - BM25 on file content (no paths), re-rank top-200 candidates
  embedding_code - E5-large cosine similarity on file content (no paths)

These are code-only baselines that use ZERO path information.

Usage:
    python scripts/eval_code_retrieval_baseline.py --mode bm25_code
    python scripts/eval_code_retrieval_baseline.py --mode embedding_code --device cuda:7
"""
import json
import os
import argparse
import time
import math
from collections import defaultdict

import numpy as np

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_file_content(repo_dir, repo_name, file_path, max_chars=None):
    """Read file content. Returns empty string if file not found."""
    full_path = os.path.join(repo_dir, repo_name, file_path)
    try:
        with open(full_path, "r", errors="replace") as f:
            content = f.read()
        if max_chars and len(content) > max_chars:
            content = content[:max_chars]
        return content
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return ""


def load_data():
    """Load test examples and candidate pools."""
    # Load test examples
    test_path = os.path.join(BASE_DIR, "data/grepo_text/grepo_test.jsonl")
    examples = {}
    with open(test_path) as f:
        for line in f:
            ex = json.loads(line)
            key = (ex["repo"], ex["issue_id"])
            examples[key] = ex
    print(f"  Loaded {len(examples)} test examples")

    # Load BM25 candidates (top-200 from graph-expanded pool)
    cand_path = os.path.join(BASE_DIR, "data/rankft/merged_bm25_exp6_candidates.jsonl")
    candidates = {}
    with open(cand_path) as f:
        for line in f:
            d = json.loads(line)
            key = (d["repo"], d["issue_id"])
            # Take top-200 candidates
            candidates[key] = d["candidates"][:200]
    print(f"  Loaded candidates for {len(candidates)} examples")

    return examples, candidates


def compute_bm25_scores(query, documents):
    """
    Simple BM25 scoring of query against documents.
    Returns list of scores, one per document.
    """
    from collections import Counter

    # Tokenize
    def tokenize(text):
        # Simple whitespace + punctuation tokenization
        import re
        return re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower())

    query_tokens = tokenize(query)
    if not query_tokens:
        return [0.0] * len(documents)

    doc_token_lists = [tokenize(d) for d in documents]
    doc_lengths = [len(t) for t in doc_token_lists]
    avg_dl = sum(doc_lengths) / max(len(doc_lengths), 1)

    # IDF: compute document frequency for each query term
    n_docs = len(documents)
    query_terms = set(query_tokens)
    df = Counter()
    for dt in doc_token_lists:
        seen = set(dt)
        for qt in query_terms:
            if qt in seen:
                df[qt] += 1

    # BM25 parameters
    k1 = 1.5
    b = 0.75

    scores = []
    for i, dt in enumerate(doc_token_lists):
        tf_map = Counter(dt)
        dl = doc_lengths[i]
        score = 0.0
        for qt in query_tokens:
            if qt not in tf_map:
                continue
            tf = tf_map[qt]
            idf = math.log((n_docs - df[qt] + 0.5) / (df[qt] + 0.5) + 1.0)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
            score += idf * tf_norm
        scores.append(score)

    return scores


def eval_bm25_code(examples, candidates, repo_dir, max_chars=50000):
    """BM25 on code content baseline."""
    print("\n=== BM25 on Code Content (no paths) ===")

    k_values = [1, 3, 5, 10, 20]
    recalls = {k: [] for k in k_values}
    n_with_content = 0
    n_total_files = 0
    n_files_found = 0

    matched_keys = set(examples.keys()) & set(candidates.keys())
    print(f"  Matched examples with candidates: {len(matched_keys)}")

    t0 = time.time()
    for idx, key in enumerate(sorted(matched_keys)):
        ex = examples[key]
        cands = candidates[key]
        repo = ex["repo"]
        gt_files = set(ex.get("changed_py_files", []))

        if not gt_files:
            continue

        issue_text = ex["issue_text"]

        # Read file contents for all candidates
        contents = []
        for fp in cands:
            content = read_file_content(repo_dir, repo, fp, max_chars=max_chars)
            contents.append(content)
            n_total_files += 1
            if content:
                n_files_found += 1

        has_content = sum(1 for c in contents if c)
        if has_content > 0:
            n_with_content += 1

        # BM25 score issue_text against each file's CODE CONTENT (not path!)
        scores = compute_bm25_scores(issue_text, contents)

        # Rank by score (descending)
        ranked_indices = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
        ranked_files = [cands[i] for i in ranked_indices]

        # Compute recall@k
        for k in k_values:
            top_k = set(ranked_files[:k])
            hits = len(gt_files & top_k)
            recall = hits / len(gt_files)
            recalls[k].append(recall)

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {idx+1}/{len(matched_keys)} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Files with content: {n_files_found}/{n_total_files} ({100*n_files_found/max(n_total_files,1):.1f}%)")
    print(f"  Examples with >=1 file content: {n_with_content}/{len(matched_keys)}")

    print(f"\n  Results (BM25 on code content, {len(recalls[1])} examples):")
    for k in k_values:
        r = 100 * np.mean(recalls[k])
        print(f"    Recall@{k}: {r:.2f}%")

    return {f"recall@{k}": float(np.mean(recalls[k])) for k in k_values}


def eval_embedding_code(examples, candidates, repo_dir, device="cuda:0",
                        max_chars=2000, batch_size=64):
    """E5-large embedding on code content baseline."""
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(42)

    print("\n=== E5-large Embedding on Code Content (no paths) ===")

    model_name = "intfloat/e5-large-v2"
    print(f"  Loading model: {model_name}...")
    model = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        device="cpu",
        model_kwargs={"torch_dtype": torch.float16},
    )
    model.max_seq_length = 512
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded ({n_params:.0f}M params) on {device}")

    k_values = [1, 3, 5, 10, 20]
    recalls = {k: [] for k in k_values}
    n_files_found = 0
    n_total_files = 0

    matched_keys = set(examples.keys()) & set(candidates.keys())
    print(f"  Matched examples with candidates: {len(matched_keys)}")

    t0 = time.time()
    for idx, key in enumerate(sorted(matched_keys)):
        ex = examples[key]
        cands = candidates[key]
        repo = ex["repo"]
        gt_files = set(ex.get("changed_py_files", []))

        if not gt_files:
            continue

        issue_text = ex["issue_text"]

        # Read file contents -- CODE ONLY, no paths
        doc_texts = []
        for fp in cands:
            content = read_file_content(repo_dir, repo, fp, max_chars=max_chars)
            n_total_files += 1
            if content:
                n_files_found += 1
                # Use passage prefix for E5, but content only (no path)
                doc_texts.append(f"passage: {content}")
            else:
                doc_texts.append("passage: ")

        # Encode query (issue text only)
        query_text = f"query: {issue_text}"

        with torch.no_grad():
            q_emb = model.encode(
                [query_text],
                batch_size=1,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            d_embs = model.encode(
                doc_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            # Cosine similarity
            scores = (q_emb @ d_embs.T).squeeze(0).cpu().numpy()

        # Rank by score (descending)
        ranked_indices = np.argsort(-scores)
        ranked_files = [cands[i] for i in ranked_indices]

        # Compute recall@k
        for k in k_values:
            top_k = set(ranked_files[:k])
            hits = len(gt_files & top_k)
            recall = hits / len(gt_files)
            recalls[k].append(recall)

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {idx+1}/{len(matched_keys)} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Files with content: {n_files_found}/{n_total_files}")

    print(f"\n  Results (E5-large on code content, {len(recalls[1])} examples):")
    for k in k_values:
        r = 100 * np.mean(recalls[k])
        print(f"    Recall@{k}: {r:.2f}%")

    return {f"recall@{k}": float(np.mean(recalls[k])) for k in k_values}


def main():
    parser = argparse.ArgumentParser(description="Code-chunk retrieval baselines")
    parser.add_argument("--mode", choices=["bm25_code", "embedding_code"],
                        required=True)
    parser.add_argument("--device", default="cuda:7",
                        help="Device for embedding mode")
    parser.add_argument("--repo_dir", default=os.path.join(BASE_DIR, "data/repos"),
                        help="Path to cloned repos")
    parser.add_argument("--max_chars_bm25", type=int, default=50000,
                        help="Max chars per file for BM25")
    parser.add_argument("--max_chars_emb", type=int, default=2000,
                        help="Max chars per file for embedding")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(BASE_DIR, f"experiments/baselines/{args.mode}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Code Retrieval Baseline: {args.mode} ===")
    print(f"Repo dir: {args.repo_dir}")

    # Load data
    print("Loading data...")
    examples, candidates = load_data()

    if args.mode == "bm25_code":
        results = eval_bm25_code(examples, candidates, args.repo_dir,
                                  max_chars=args.max_chars_bm25)
    elif args.mode == "embedding_code":
        results = eval_embedding_code(examples, candidates, args.repo_dir,
                                       device=args.device,
                                       max_chars=args.max_chars_emb,
                                       batch_size=args.batch_size)

    # Save results
    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print comparison
    print("\n=== Comparison with Path-based Baselines ===")
    print(f"  BM25-path (prior):     R@1=22.51%  R@5=44.76%")
    print(f"  E5-path (prior):       R@1= 9.50%")
    r1 = results.get("recall@1", 0) * 100
    r5 = results.get("recall@5", 0) * 100
    print(f"  {args.mode} (this):    R@1={r1:.2f}%  R@5={r5:.2f}%")


if __name__ == "__main__":
    main()
