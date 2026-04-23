#!/usr/bin/env python3
"""
SweRank path perturbation evaluation.

Runs SweRankEmbed (Small or Large) on our SWE-bench perturbation data
to test whether Salesforce's SOTA bug localization model also exhibits
path-prior bias.

This is analogous to eval_biencoder_reranker.py but adapted for SweRank's
query format and candidate structure.

Usage:
    # SweRankEmbed-Small, no perturbation (baseline)
    CUDA_VISIBLE_DEVICES=5 python scripts/eval_swerank_perturb.py \
        --model small --perturb none --gpu_id 0

    # SweRankEmbed-Large, shuffle_dirs perturbation
    CUDA_VISIBLE_DEVICES=5 python scripts/eval_swerank_perturb.py \
        --model large --perturb shuffle_dirs --gpu_id 0

    # Full sweep
    bash scripts/run_swerank_perturb.sh
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Model configs ──────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "small": {
        "hf_name": "Salesforce/SweRankEmbed-Small",
        "query_prefix": "Represent this query for searching relevant code: ",
        "doc_prefix": "",  # no prefix for documents
        "use_prompt_name": True,  # use prompt_name="query" in encode()
    },
    "large": {
        "hf_name": "Salesforce/SweRankEmbed-Large",
        "query_prefix": "Instruct: Given a github issue, identify the code that needs to be changed to fix the issue.\nQuery: ",
        "doc_prefix": "",
        "use_prompt_name": True,
    },
}

MODEL_CACHE_DIR = "/data/chenlibin/models"

# ── Data paths ─────────────────────────────────────────────────────────────
# SWE-bench Lite perturbation data
SWEBENCH_DIR = BASE_DIR / "data" / "swebench_lite"

TEST_PATHS = {
    "none": SWEBENCH_DIR / "swebench_lite_test.jsonl",
    "shuffle_dirs": SWEBENCH_DIR / "swebench_perturb_shuffle_dirs_test.jsonl",
    "shuffle_filenames": SWEBENCH_DIR / "swebench_perturb_shuffle_filenames_test.jsonl",
    "flatten_dirs": SWEBENCH_DIR / "swebench_perturb_flatten_dirs_test.jsonl",
}

CAND_PATHS = {
    "none": None,  # will use original candidates
    "shuffle_dirs": SWEBENCH_DIR / "swebench_perturb_shuffle_dirs_candidates.jsonl",
    "shuffle_filenames": SWEBENCH_DIR / "swebench_perturb_shuffle_filenames_candidates.jsonl",
    "flatten_dirs": SWEBENCH_DIR / "swebench_perturb_flatten_dirs_candidates.jsonl",
}


def parse_string_list(s):
    """Parse a string like "['a.py', 'b.py']" into a Python list."""
    if isinstance(s, list):
        return s
    return json.loads(s.replace("'", '"'))


def load_data(perturb: str) -> List[Dict]:
    """Load test data and candidates for the given perturbation mode."""
    test_path = TEST_PATHS[perturb]
    cand_path = CAND_PATHS[perturb]

    # Load test examples
    test_data = []
    with open(test_path) as f:
        for line in f:
            d = json.loads(line)
            test_data.append(d)

    if cand_path is None:
        # For "none" perturbation, we need a candidates file.
        # Use shuffle_dirs candidates as reference for the original paths.
        # Actually, we should have an original candidates file.
        # Fall back: use any available candidates file to get the structure,
        # but the candidate paths should match the original (unperturbed) data.
        # For now, try to find an original candidates file.
        # Try multiple known locations for unperturbed candidates
        for orig_path in [
            SWEBENCH_DIR / "swebench_lite_candidates.jsonl",
            BASE_DIR / "data" / "rankft" / "swebench_bm25_tricked_top500.jsonl",
            BASE_DIR / "data" / "rankft" / "swebench_test_bm25_top500.jsonl",
        ]:
            if orig_path.exists():
                cand_path = orig_path
                break
        if cand_path is None:
            raise FileNotFoundError("No unperturbed SWE-bench candidates file found.")

    # Load candidates
    cand_map = {}
    with open(cand_path) as f:
        for line in f:
            d = json.loads(line)
            issue_id = d["issue_id"]
            candidates = parse_string_list(d["bm25_candidates"])
            gt = parse_string_list(d["ground_truth"])
            cand_map[issue_id] = {"candidates": candidates, "ground_truth": gt}

    # Merge test data with candidates
    merged = []
    skipped = 0
    for ex in test_data:
        issue_id = ex["issue_id"]
        if issue_id not in cand_map:
            skipped += 1
            continue

        changed_files = parse_string_list(ex["changed_py_files"])
        merged.append({
            "repo": ex["repo"],
            "issue_id": issue_id,
            "issue_text": ex["issue_text"],
            "changed_py_files": changed_files,
            "candidates": cand_map[issue_id]["candidates"],
            "ground_truth": cand_map[issue_id]["ground_truth"],
        })

    if skipped > 0:
        print(f"  Skipped {skipped} examples (no candidates)")
    return merged


def compute_metrics(
    data: List[Dict],
    model,
    model_config: Dict,
    batch_size: int = 128,
) -> Dict[str, float]:
    """
    For each example:
    - Encode query (issue text) with appropriate prefix
    - Encode each candidate file path
    - Rank by dot-product similarity
    - Compute partial recall@K
    """
    ks = [1, 3, 5, 10, 20]
    recalls = {k: [] for k in ks}

    total = len(data)
    t0 = time.time()

    for idx, ex in enumerate(data):
        issue_text = ex["issue_text"]
        candidates = ex["candidates"]
        gt_files = set(ex["ground_truth"])

        if not candidates or not gt_files:
            for k in ks:
                recalls[k].append(0.0)
            continue

        # Encode query
        # SweRankEmbed models support prompt_name="query" via SentenceTransformer
        if model_config["use_prompt_name"]:
            q_emb = model.encode(
                [issue_text],
                prompt_name="query",
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            query_text = f"{model_config['query_prefix']}{issue_text}"
            q_emb = model.encode(
                [query_text],
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        # Encode candidate file paths as documents
        # SweRank treats code as documents; we use file paths as a proxy
        # (same setup as our e5-large-v2 experiment)
        doc_texts = [f"File: {c}" for c in candidates]
        d_embs = model.encode(
            doc_texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        # Dot product (embeddings are normalized -> cosine similarity)
        scores = (q_emb @ d_embs.T).squeeze(0)

        # Rank
        ranked_indices = np.argsort(-scores)
        ranked_candidates = [candidates[i] for i in ranked_indices]

        # Partial recall@K
        n_gt = len(gt_files)
        for k in ks:
            top_k = set(ranked_candidates[:k])
            hit = len(top_k & gt_files)
            recalls[k].append(hit / n_gt)

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (total - idx - 1) / rate
            print(f"  [{idx+1}/{total}] elapsed={elapsed:.1f}s, ETA={eta:.1f}s", flush=True)

    results = {}
    for k in ks:
        results[f"R@{k}"] = np.mean(recalls[k]) * 100
    return results


def main():
    parser = argparse.ArgumentParser(description="SweRank path perturbation eval")
    parser.add_argument("--model", type=str, default="small",
                        choices=["small", "large"],
                        help="SweRankEmbed variant: small (137M) or large (7B)")
    parser.add_argument("--perturb", type=str, default="none",
                        choices=["none", "shuffle_filenames", "shuffle_dirs", "flatten_dirs"],
                        help="Path perturbation type")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device index (after CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Encoding batch size")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results JSON")
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    print(f"{'='*60}")
    print(f"SweRank Perturbation Eval")
    print(f"  Model: {config['hf_name']}")
    print(f"  Perturbation: {args.perturb}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Load model
    print(f"\nLoading model: {config['hf_name']}")
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    # Import here to avoid slow startup if just checking args
    from sentence_transformers import SentenceTransformer

    local_path = os.path.join(MODEL_CACHE_DIR, config["hf_name"].split("/")[-1])
    model_src = local_path if os.path.isdir(local_path) else config["hf_name"]
    print(f"  Source: {model_src}")
    model_kwargs = {"torch_dtype": torch.bfloat16} if args.model == "large" else {}
    model = SentenceTransformer(
        model_src,
        cache_folder=MODEL_CACHE_DIR,
        trust_remote_code=True,
        device=device,
        model_kwargs=model_kwargs,
    )

    # For Large model (7B), use bfloat16 to save memory
    if args.model == "large":
        model = model.to(torch.bfloat16)
        print(f"  Cast to bfloat16")

    emb_dim = model.get_sentence_embedding_dimension()
    print(f"  Embedding dim: {emb_dim}")

    # Load data
    print("\nLoading data...")
    data = load_data(args.perturb)
    print(f"  Loaded {len(data)} examples")

    # Compute metrics
    print("\nComputing metrics...")
    results = compute_metrics(data, model, config, batch_size=args.batch_size)

    # Print results
    model_tag = f"SweRankEmbed-{'Small' if args.model == 'small' else 'Large'}"
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_tag} | perturb={args.perturb}")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k} = {v:.2f}%")

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(
            args.output_dir,
            f"swerank_{args.model}_{args.perturb}.json"
        )
        with open(out_path, "w") as f:
            json.dump({
                "model": config["hf_name"],
                "model_tag": model_tag,
                "perturb": args.perturb,
                "num_examples": len(data),
                "metrics": results,
            }, f, indent=2)
        print(f"\nSaved to {out_path}")

    return results


if __name__ == "__main__":
    main()
