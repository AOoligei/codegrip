#!/usr/bin/env python3
"""
Build E5-large dense retrieval candidates for the GREPO TRAIN split.

This is needed for the mixed-pool reranker training experiment:
the reranker is trained on negatives from BOTH BM25 and dense retriever pools,
so it generalizes across retriever distributions at test time.

Adapts the same logic from scripts/build_hybrid_retrieval.py (which does test split)
but processes train examples instead.

Output: data/rankft/grepo_train_e5large_top500.jsonl
Format per line:
    {"repo": ..., "issue_id": ..., "candidates": [...], "gt_in_candidates": bool}
    (note: uses "candidates" key to match load_bm25_candidates() in train_rankft.py)

Usage:
    python scripts/build_dense_train_candidates.py --device cuda:2
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA = os.path.join(PROJECT_ROOT, "data/grepo_text/grepo_train.jsonl")
FILE_TREE_DIR = os.path.join(PROJECT_ROOT, "data/file_trees")
REPOS_DIR = os.path.join(PROJECT_ROOT, "data/repos")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/rankft/grepo_train_e5large_top500.jsonl")

E5_CONFIG = {
    "hf_name": "intfloat/e5-large-v2",
    "query_prefix": "query: ",
    "doc_prefix": "passage: ",
    "max_seq_length": 512,
}


def read_file_content(repo_dir, file_path, max_lines=200):
    """Read first max_lines of a file."""
    full_path = os.path.join(repo_dir, file_path)
    try:
        with open(full_path, "r", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip())
        return "\n".join(lines)
    except (FileNotFoundError, PermissionError):
        return ""


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build E5-large dense candidates for GREPO train split"
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_lines", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=500)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    config = E5_CONFIG

    print("=" * 60)
    print("Building E5-large dense candidates for TRAIN split")
    print("=" * 60)

    # Load model
    print(f"Loading {config['hf_name']} on {args.device}...")
    model = SentenceTransformer(
        config["hf_name"],
        trust_remote_code=True,
        device="cpu",
        model_kwargs={"torch_dtype": torch.float16},
    )
    model.max_seq_length = config["max_seq_length"]
    model = model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded ({n_params:.0f}M params)")

    # Load train data
    print("Loading train data and file trees...")
    examples = []
    with open(TRAIN_DATA) as f:
        for line in f:
            examples.append(json.loads(line))

    file_trees = {}
    for fname in os.listdir(FILE_TREE_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(FILE_TREE_DIR, fname)) as f:
                tree = json.load(f)
            file_trees[tree["repo"]] = tree

    # Group by repo
    repo_examples = defaultdict(list)
    for ex in examples:
        repo = ex["repo"]
        gt_files = set(ex.get("changed_py_files", []))
        if gt_files and repo in file_trees:
            repo_examples[repo].append(ex)

    total = sum(len(v) for v in repo_examples.values())
    print(f"  {total} train examples across {len(repo_examples)} repos")

    # Process per repo (encode files once, query all examples)
    results = []
    processed = 0
    t0 = time.time()

    for repo_idx, (repo, exs) in enumerate(sorted(repo_examples.items())):
        tree = file_trees[repo]
        py_files = tree["py_files"]

        # Read file contents and encode
        repo_dir = os.path.join(REPOS_DIR, repo)
        doc_texts = []
        valid_files = []

        for fp in py_files:
            content = read_file_content(repo_dir, fp, args.max_lines)
            doc_text = f"{fp}\n{content}" if content else fp
            doc_texts.append(f"{config['doc_prefix']}{doc_text}")
            valid_files.append(fp)

        if not doc_texts:
            continue

        repo_t0 = time.time()
        with torch.no_grad():
            file_embs = model.encode(
                doc_texts,
                batch_size=args.batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

        for ex in exs:
            query_text = f"{config['query_prefix']}{ex['issue_text']}"
            with torch.no_grad():
                query_emb = model.encode(
                    [query_text],
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )
                scores = torch.matmul(query_emb, file_embs.T).squeeze(0)
                top_k = min(args.top_k, len(valid_files))
                top_indices = torch.argsort(scores, descending=True)[:top_k].cpu().numpy()

            ranked_files = [valid_files[i] for i in top_indices]
            gt_files = set(ex.get("changed_py_files", []))

            results.append({
                "repo": ex["repo"],
                "issue_id": ex["issue_id"],
                "candidates": ranked_files,  # use "candidates" key for compatibility
                "gt_in_candidates": bool(gt_files & set(ranked_files)),
            })
            processed += 1

        repo_time = time.time() - repo_t0
        if (repo_idx + 1) % 5 == 0 or repo_idx == len(repo_examples) - 1:
            elapsed = time.time() - t0
            print(f"  [{processed}/{total}] {repo} ({len(py_files)} files, "
                  f"{repo_time:.1f}s) | {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"  Done: {len(results)} examples in {elapsed:.0f}s")

    # Save
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved: {args.output}")

    # Report oracle recall
    gt_in = sum(1 for r in results if r["gt_in_candidates"])
    print(f"\n  Oracle recall (GT in E5 top-{args.top_k}): "
          f"{gt_in}/{len(results)} = {gt_in/len(results)*100:.1f}%")

    # Clean up
    del model, file_embs
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
