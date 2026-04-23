#!/usr/bin/env python3
"""
Dense retrieval baselines (Contriever, E5, SweRank) on GREPO.

Retrieves from ALL py_files per repo (not BM25-filtered), giving a fair
full-retrieval comparison. Pre-computes file embeddings per repo, then
reuses across examples.

Usage:
    python scripts/eval_dense_retrieval.py \
        --model contriever \
        --device cuda:0 \
        --output_dir experiments/baselines/contriever_grepo
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

MODEL_CONFIGS = {
    "contriever": {
        "hf_name": "facebook/contriever-msmarco",
        "query_prefix": "",
        "doc_prefix": "",
        "max_seq_length": 512,
    },
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
    "swerank": {
        "hf_name": "/home/chenlibin/models/SweRankEmbed-Large",
        "query_prefix": "Instruct: Given a github issue, identify the code that needs to be changed to fix the issue.\nQuery: ",
        "doc_prefix": "",
        "max_seq_length": 1024,
    },
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


def build_doc_text(file_path, content, use_content=True):
    """Build document text from file path + optional content."""
    if use_content and content:
        return f"{file_path}\n{content}"
    return file_path


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data():
    """Load test data and file trees."""
    # Load test examples
    test_path = os.path.join(BASE_DIR, "data/grepo_text/grepo_test.jsonl")
    examples = []
    with open(test_path) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"  Loaded {len(examples)} test examples")

    # Load file trees
    file_trees = {}
    tree_dir = os.path.join(BASE_DIR, "data/file_trees")
    for fname in os.listdir(tree_dir):
        if fname.endswith(".json"):
            with open(os.path.join(tree_dir, fname)) as f:
                tree = json.load(f)
            file_trees[tree["repo"]] = tree
    print(f"  Loaded {len(file_trees)} file trees")

    return examples, file_trees


def encode_repo_files(model, repo_name, py_files, repos_dir, doc_prefix,
                      use_content, max_lines, batch_size):
    """Encode all py_files for a repo. Returns embeddings tensor and file list."""
    repo_dir = os.path.join(repos_dir, repo_name)
    doc_texts = []
    valid_files = []

    for fp in py_files:
        if use_content:
            content = read_file_content(repo_dir, fp, max_lines)
        else:
            content = ""
        doc_text = build_doc_text(fp, content, use_content)
        doc_texts.append(f"{doc_prefix}{doc_text}")
        valid_files.append(fp)

    if not doc_texts:
        return None, []

    with torch.no_grad():
        embeddings = model.encode(
            doc_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    return embeddings, valid_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()),
                        default="contriever")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_lines", type=int, default=200,
                        help="Max lines of file content to include")
    parser.add_argument("--no_content", action="store_true",
                        help="Path-only (no file content)")
    parser.add_argument("--repos_dir", default=os.path.join(BASE_DIR, "data/repos"))
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    use_content = not args.no_content

    if args.output_dir is None:
        suffix = "_pathonly" if not use_content else ""
        args.output_dir = os.path.join(BASE_DIR, f"experiments/baselines/{args.model}_grepo_allfiles{suffix}")

    print(f"=== Dense Retrieval Baseline: {args.model} ===")
    content_desc = f"path+content (first {args.max_lines} lines)" if use_content else "path-only"
    print(f"Content: {content_desc}")

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
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded ({n_params:.0f}M params) on {args.device}")

    # Load data
    print("Loading data...")
    examples, file_trees = load_data()

    # Group examples by repo
    repo_examples = defaultdict(list)
    for ex in examples:
        repo = ex["repo"]
        gt_files = set(ex.get("changed_py_files", []))
        if gt_files and repo in file_trees:
            repo_examples[repo].append(ex)

    print(f"  {sum(len(v) for v in repo_examples.values())} examples across {len(repo_examples)} repos")

    # Evaluate per repo (encode files once, reuse)
    k_values = [1, 3, 5, 10, 20]
    all_metrics = {f"recall@{k}": [] for k in k_values}
    all_results = []
    total_examples = sum(len(v) for v in repo_examples.values())
    processed = 0

    t0 = time.time()
    for repo_idx, (repo, exs) in enumerate(sorted(repo_examples.items())):
        tree = file_trees[repo]
        py_files = tree["py_files"]

        # Encode all files for this repo
        repo_t0 = time.time()
        file_embs, valid_files = encode_repo_files(
            model, repo, py_files, args.repos_dir,
            config["doc_prefix"], use_content, args.max_lines, args.batch_size
        )
        if file_embs is None:
            continue
        repo_encode_time = time.time() - repo_t0

        # Score each example
        for ex in exs:
            issue_text = ex["issue_text"]
            gt_files = set(ex["changed_py_files"])

            # Encode query
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

            # Compute recall@K
            for k in k_values:
                top_k = set(ranked_files[:k])
                recall = len(top_k & gt_files) / len(gt_files) * 100
                all_metrics[f"recall@{k}"].append(recall)

            all_results.append({
                "repo": repo,
                "issue_id": ex["issue_id"],
                "predicted": ranked_files[:20],
                "ground_truth": list(gt_files),
            })

            processed += 1

        if (repo_idx + 1) % 5 == 0 or repo_idx == len(repo_examples) - 1:
            r1 = np.mean(all_metrics["recall@1"]) if all_metrics["recall@1"] else 0
            r5 = np.mean(all_metrics["recall@5"]) if all_metrics["recall@5"] else 0
            elapsed = time.time() - t0
            print(f"  [{processed}/{total_examples}] {repo} ({len(py_files)} files, "
                  f"encode: {repo_encode_time:.1f}s) | R@1: {r1:.2f}% R@5: {r5:.2f}% "
                  f"| {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0

    # Print final results
    print(f"\n{'='*60}")
    print(f"{args.model} Dense Retrieval (GREPO, all files)")
    print(f"Content: {'path+content' if use_content else 'path-only'}")
    print(f"{'='*60}")
    print(f"Evaluated: {len(all_metrics['recall@1'])} examples in {elapsed:.0f}s")
    print()
    for k in k_values:
        vals = all_metrics[f"recall@{k}"]
        avg = np.mean(vals) if vals else 0
        print(f"  Recall@{k:<5} {avg:>7.2f}%")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "overall": {f"recall@{k}": float(np.mean(all_metrics[f"recall@{k}"])) for k in k_values},
        "n_samples": len(all_metrics["recall@1"]),
        "model": config["hf_name"],
        "model_key": args.model,
        "use_content": use_content,
        "max_lines": args.max_lines if use_content else 0,
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.output_dir, "predictions.jsonl"), "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
