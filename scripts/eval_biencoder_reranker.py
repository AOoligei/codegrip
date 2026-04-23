#!/usr/bin/env python3
"""
Bi-encoder (embedding-based) reranker evaluation for CodeGRIP.

Zero-shot experiment: load a pretrained embedding model (e5-large-v2),
encode bug reports and candidate file paths, rank by cosine similarity,
and compute R@1, R@5. Supports path perturbation modes.

Usage:
    CUDA_VISIBLE_DEVICES=6 python scripts/eval_biencoder_reranker.py --perturb none
    CUDA_VISIBLE_DEVICES=6 python scripts/eval_biencoder_reranker.py --perturb shuffle_filenames
    CUDA_VISIBLE_DEVICES=6 python scripts/eval_biencoder_reranker.py --perturb shuffle_dirs
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Deterministic
np.random.seed(42)
torch.manual_seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
TEST_PATH = BASE_DIR / "data" / "grepo_text" / "grepo_test.jsonl"
CANDIDATES_PATH = BASE_DIR / "data" / "rankft" / "merged_bm25_exp6_candidates.jsonl"

PERTURB_TEST_PATHS = {
    "shuffle_filenames": BASE_DIR / "experiments" / "path_perturb_shuffle_filenames" / "test.jsonl",
    "shuffle_dirs": BASE_DIR / "experiments" / "path_perturb_shuffle_dirs" / "test.jsonl",
}
PERTURB_CAND_PATHS = {
    "shuffle_filenames": BASE_DIR / "experiments" / "path_perturb_shuffle_filenames" / "bm25_candidates.jsonl",
    "shuffle_dirs": BASE_DIR / "experiments" / "path_perturb_shuffle_dirs" / "bm25_candidates.jsonl",
}

MODEL_NAME = "intfloat/e5-large-v2"
MODEL_CACHE_DIR = "/data/chenlibin/models"


def load_data(perturb: str) -> List[Dict]:
    """Load test data and candidates, merge them."""
    if perturb == "none":
        test_path = TEST_PATH
        cand_path = CANDIDATES_PATH
    else:
        test_path = PERTURB_TEST_PATHS[perturb]
        cand_path = PERTURB_CAND_PATHS[perturb]

    # Load test examples
    test_data = []
    with open(test_path) as f:
        for line in f:
            test_data.append(json.loads(line))

    # Load candidates (keyed by repo+issue_id)
    cand_map = {}
    with open(cand_path) as f:
        for line in f:
            d = json.loads(line)
            key = (d["repo"], d["issue_id"])
            cand_map[key] = d["candidates"]

    # Merge
    merged = []
    for ex in test_data:
        key = (ex["repo"], ex["issue_id"])
        if key not in cand_map:
            continue
        merged.append({
            "repo": ex["repo"],
            "issue_id": ex["issue_id"],
            "issue_text": ex["issue_text"],
            "changed_py_files": ex["changed_py_files"],
            "candidates": cand_map[key],
        })

    return merged


def compute_metrics(data: List[Dict], model: SentenceTransformer, batch_size: int = 256) -> Dict[str, float]:
    """
    For each example:
    - Encode query: "query: {issue_text}"
    - Encode each candidate: "passage: File: {path}"
    - Rank by cosine similarity
    - Compute partial recall@K
    """
    recalls = {1: [], 5: []}

    total = len(data)
    t0 = time.time()

    for idx, ex in enumerate(data):
        issue_text = ex["issue_text"]
        candidates = ex["candidates"]
        gt_files = set(ex["changed_py_files"])

        if not candidates or not gt_files:
            for k in recalls:
                recalls[k].append(0.0)
            continue

        # e5-large-v2 format: prefix with "query:" and "passage:"
        query_text = f"query: {issue_text}"
        doc_texts = [f"passage: File: {c}" for c in candidates]

        # Encode
        q_emb = model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)
        d_embs = model.encode(doc_texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)

        # Cosine similarity (already normalized)
        scores = (q_emb @ d_embs.T).squeeze(0)

        # Rank
        ranked_indices = np.argsort(-scores)
        ranked_candidates = [candidates[i] for i in ranked_indices]

        # Partial recall@K
        n_gt = len(gt_files)
        for k in recalls:
            top_k = set(ranked_candidates[:k])
            hit = len(top_k & gt_files)
            recalls[k].append(hit / n_gt)

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{total}] elapsed={elapsed:.1f}s", flush=True)

    results = {}
    for k in recalls:
        results[f"R@{k}"] = np.mean(recalls[k]) * 100
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturb", type=str, default="none",
                        choices=["none", "shuffle_filenames", "shuffle_dirs"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Perturbation: {args.perturb}")

    # Load model
    print(f"Loading model: {MODEL_NAME} (cache: {MODEL_CACHE_DIR})")
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE_DIR, device=device)
    model.half()  # fp16 to save GPU memory
    print(f"Model loaded (fp16). Embedding dim: {model.get_sentence_embedding_dimension()}")

    # Load data
    print("Loading data...")
    data = load_data(args.perturb)
    print(f"Loaded {len(data)} examples")

    # Compute metrics
    print("Computing metrics...")
    results = compute_metrics(data, model, batch_size=args.batch_size)

    print(f"\n=== Bi-encoder ({MODEL_NAME}) | perturb={args.perturb} ===")
    for k, v in results.items():
        print(f"  {k} = {v:.2f}%")

    return results


if __name__ == "__main__":
    main()
