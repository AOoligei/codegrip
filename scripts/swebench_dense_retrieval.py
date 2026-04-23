#!/usr/bin/env python3
"""
Dense retrieval on SWE-bench Lite test set using E5-large.
Produces per-example top-K dense candidates for merging with BM25.

Usage:
    python scripts/swebench_dense_retrieval.py \
        --model e5-large \
        --device cuda:0 \
        --output_dir experiments/swebench/dense_e5large
"""
import json
import os
import argparse
import time
from collections import defaultdict

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

np.random.seed(42)
torch.manual_seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_CONFIGS = {
    "e5-large": {
        "hf_name": "intfloat/e5-large-v2",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "max_seq_length": 512,
    },
    "e5-base": {
        "hf_name": "intfloat/e5-base-v2",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "max_seq_length": 512,
    },
}


def read_file_content(repo_dir, file_path, max_lines=200):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="e5-large")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_lines", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=200,
                        help="Number of dense candidates to retrieve per example")
    parser.add_argument("--test_data",
                        default=os.path.join(BASE_DIR, "data/rankft/swebench_test_bm25_top500.jsonl"))
    parser.add_argument("--repos_dir",
                        default=os.path.join(BASE_DIR, "data/swebench_lite/repos"))
    parser.add_argument("--file_trees_dir",
                        default=os.path.join(BASE_DIR, "data/swebench_lite/file_trees"))
    parser.add_argument("--output_dir",
                        default=os.path.join(BASE_DIR, "experiments/swebench/dense_e5large"))
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    print(f"=== SWE-bench Dense Retrieval: {args.model} ===")
    print(f"Top-K: {args.top_k}, Max lines: {args.max_lines}")

    # Load model
    print(f"Loading model: {config['hf_name']}...")
    model = SentenceTransformer(
        config["hf_name"],
        trust_remote_code=True,
        device="cpu",
        model_kwargs={"torch_dtype": torch.float16},
    )
    model.max_seq_length = config["max_seq_length"]
    model = model.to(args.device)
    print(f"  Loaded on {args.device}")

    # Load test data
    print("Loading test data...")
    examples = []
    with open(args.test_data) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"  {len(examples)} test examples")

    # Load file trees
    file_trees = {}
    for fname in os.listdir(args.file_trees_dir):
        if fname.endswith(".json"):
            with open(os.path.join(args.file_trees_dir, fname)) as f:
                tree = json.load(f)
            file_trees[tree["repo"]] = tree
    print(f"  {len(file_trees)} file trees")

    # Group examples by repo
    repo_examples = defaultdict(list)
    for ex in examples:
        repo = ex["repo"]
        repo_examples[repo].append(ex)
    print(f"  {len(repo_examples)} unique repos")

    # Process per repo (encode files once, reuse across examples)
    all_results = []
    k_values = [1, 3, 5, 10, 20, 50, 100, 200]
    all_metrics = {f"recall@{k}": [] for k in k_values}
    processed = 0
    t0 = time.time()

    for repo_idx, (repo, exs) in enumerate(sorted(repo_examples.items())):
        if repo not in file_trees:
            print(f"  WARNING: no file tree for {repo}, skipping {len(exs)} examples")
            continue

        tree = file_trees[repo]
        py_files = tree["py_files"]
        repo_dir = os.path.join(args.repos_dir, repo)

        if not os.path.isdir(repo_dir):
            print(f"  WARNING: repo dir not found for {repo}, skipping")
            continue

        # Encode all py files
        print(f"  [{repo_idx+1}/{len(repo_examples)}] {repo}: {len(py_files)} py files, {len(exs)} examples")
        repo_t0 = time.time()

        doc_texts = []
        valid_files = []
        for fp in py_files:
            content = read_file_content(repo_dir, fp, args.max_lines)
            doc_text = f"{config['doc_prefix']}{fp}\n{content}" if content else f"{config['doc_prefix']}{fp}"
            doc_texts.append(doc_text)
            valid_files.append(fp)

        if not doc_texts:
            continue

        with torch.no_grad():
            file_embs = model.encode(
                doc_texts,
                batch_size=args.batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

        repo_encode_time = time.time() - repo_t0
        print(f"    Encoded {len(valid_files)} files in {repo_encode_time:.1f}s")

        # Score each example
        for ex in exs:
            issue_text = ex.get("issue_text", "")
            gt_files = set(ex["ground_truth"])

            query_text = f"{config['query_prefix']}{issue_text}"
            with torch.no_grad():
                query_emb = model.encode(
                    [query_text],
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )
                scores = torch.matmul(query_emb, file_embs.T).squeeze(0)
                ranked_indices = torch.argsort(scores, descending=True).cpu().numpy()

            ranked_files = [valid_files[i] for i in ranked_indices]
            ranked_scores = scores[ranked_indices].cpu().numpy().tolist()

            # Save top-K dense candidates
            top_k_files = ranked_files[:args.top_k]
            top_k_scores = ranked_scores[:args.top_k]

            result = {
                "repo": repo,
                "issue_id": ex["issue_id"],
                "ground_truth": list(gt_files),
                "dense_candidates": top_k_files,
                "dense_scores": top_k_scores,
            }
            all_results.append(result)

            # Compute recall@K
            for k in k_values:
                if k > len(ranked_files):
                    continue
                top_k_set = set(ranked_files[:k])
                recall = len(top_k_set & gt_files) / len(gt_files) * 100 if gt_files else 0
                all_metrics[f"recall@{k}"].append(recall)

            processed += 1

        if (repo_idx + 1) % 3 == 0 or repo_idx == len(repo_examples) - 1:
            r1 = np.mean(all_metrics["recall@1"]) if all_metrics["recall@1"] else 0
            r5 = np.mean(all_metrics["recall@5"]) if all_metrics["recall@5"] else 0
            elapsed = time.time() - t0
            print(f"    Progress: {processed}/{len(examples)} | R@1={r1:.2f}% R@5={r5:.2f}% | {elapsed:.0f}s")

    elapsed = time.time() - t0

    # Print results
    print(f"\n{'='*60}")
    print(f"SWE-bench Dense Retrieval: {args.model}")
    print(f"{'='*60}")
    print(f"Evaluated: {processed} examples in {elapsed:.0f}s")
    for k in k_values:
        vals = all_metrics[f"recall@{k}"]
        if vals:
            print(f"  Recall@{k:<5} {np.mean(vals):>7.2f}%")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "overall": {f"recall@{k}": float(np.mean(all_metrics[f"recall@{k}"]))
                     for k in k_values if all_metrics[f"recall@{k}"]},
        "n_samples": processed,
        "model": config["hf_name"],
        "top_k": args.top_k,
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.output_dir, "dense_candidates.jsonl"), "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
