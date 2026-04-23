#!/usr/bin/env python3
"""
All-files reranking evaluation for CodeGRIP.

Addresses reviewer concern: BM25 candidate pool is path-biased, confounding
the "code adds no value" conclusion. By scoring ALL .py files in each repo
(not just BM25 top-200), we test code signal without retrieval bias.

For each test example:
  1. Load ALL .py files for that repo from data/file_trees/{repo}.json
  2. Score each file using the reranker (path-only or code-centric prompt)
  3. Rank all files by score
  4. Compute R@1, R@5, R@10

Large-repo handling: repos with >2000 .py files are sampled to 1000 random
files + all GT files, with oracle recall reported.

Usage (path-only):
  python scripts/eval_all_files_reranker.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data data/grepo_text/grepo_test.jsonl \
    --file_tree_dir data/file_trees \
    --output_dir experiments/all_files_rerank_path_only \
    --gpu_id 0

Usage (code-centric):
  python scripts/eval_all_files_reranker.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path /data/chenlibin/grepo_agent_experiments/code_centric_scorer/best \
    --test_data data/grepo_text/grepo_test.jsonl \
    --file_tree_dir data/file_trees \
    --output_dir experiments/all_files_rerank_code_centric \
    --prompt_mode code_centric \
    --repo_dir /data/chenlibin/grepo_repos \
    --gpu_id 0

Estimated runtime: ~8-12h per model on 1 GPU (1704 examples, ~541 files/repo avg).
"""
import ast
import os
import sys
import json
import argparse
import random
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ============================================================
# Prompt templates
# ============================================================

PATH_ONLY_TEMPLATE = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)


def build_path_only_prompt(issue_text: str, candidate_path: str, **kwargs) -> str:
    return PATH_ONLY_TEMPLATE.format(issue_text=issue_text, candidate_path=candidate_path)


def build_code_centric_prompt(issue_text: str, candidate_path: str,
                               code_content: str = "",
                               tokenizer=None, max_seq_length: int = 1024,
                               **kwargs) -> str:
    """Build code-centric scoring prompt (same as training)."""
    suffix = "Based on the code content and structure, is this file likely to need modification?\nAnswer:"
    prefix = (
        f"Given the bug report, analyze the code and determine if this file "
        f"likely needs modification.\n\n"
        f"Bug Report: {issue_text}\n\n"
        f"File: {candidate_path}\n\n"
        f"Code (key sections):\n"
    )
    if tokenizer is not None:
        prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
        suffix_tokens = len(tokenizer.encode(suffix, add_special_tokens=False))
        code_budget = max_seq_length - prefix_tokens - suffix_tokens - 10
        if code_budget > 0:
            code_tokens = tokenizer.encode(code_content, add_special_tokens=False)
            if len(code_tokens) > code_budget:
                code_content = tokenizer.decode(code_tokens[:code_budget])

    return f"{prefix}{code_content}\n\n{suffix}"


# ============================================================
# Code extraction (for code-centric mode)
# ============================================================

def _extract_signatures(source: str) -> List[str]:
    sigs = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return sigs
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            args_str = ast.get_source_segment(source, node.args)
            if args_str is None:
                lines = source.split('\n')
                if node.lineno <= len(lines):
                    sig_line = lines[node.lineno - 1].strip()
                    sigs.append(sig_line)
            else:
                sigs.append(f"{prefix} {node.name}({args_str}):")
        elif isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                seg = ast.get_source_segment(source, base)
                if seg:
                    bases.append(seg)
            base_str = f"({', '.join(bases)})" if bases else ""
            sigs.append(f"class {node.name}{base_str}:")
    return sigs


def extract_code_content(repo_dir: str, repo: str, filepath: str,
                          head_lines: int = 50, max_chars: int = 1500) -> str:
    full_path = os.path.join(repo_dir, repo, filepath)
    try:
        with open(full_path, 'r', errors='replace') as f:
            full_source = f.read()
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return "# (file not available)"

    lines = full_source.split('\n')
    head = '\n'.join(lines[:head_lines])
    sigs = _extract_signatures(full_source)
    extra_sigs = [s.strip() for s in sigs if s.strip() not in head]
    if extra_sigs:
        sig_block = "\n# ... (signatures from rest of file)\n" + '\n'.join(extra_sigs)
        content = head + sig_block
    else:
        content = head
    if len(content) > max_chars:
        content = content[:max_chars] + "\n# ... (truncated)"
    return content


# ============================================================
# Scoring
# ============================================================

def get_yes_no_token_ids(tokenizer):
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    return yes_ids[0], no_ids[0]


@torch.no_grad()
def score_files_batched(model, tokenizer, prompts: List[str],
                         yes_id: int, no_id: int,
                         max_seq_length: int, device: str,
                         batch_size: int = 16) -> List[float]:
    """Score a list of pre-built prompts. Returns list of yes-no logit diffs."""
    all_scores = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        encodings = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                              truncation=True, max_length=max_seq_length)
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                # Fall back to single-item scoring
                for prompt in batch_prompts:
                    enc = tokenizer([prompt], return_tensors="pt", truncation=True,
                                    max_length=max_seq_length)
                    ids = enc["input_ids"].to(device)
                    mask = enc["attention_mask"].to(device)
                    out = model(input_ids=ids, attention_mask=mask)
                    logits = out.logits[0, -1]
                    score = (logits[yes_id] - logits[no_id]).item()
                    all_scores.append(score)
                continue
            raise

        logits = outputs.logits
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.size(0), device=device)
        last_logits = logits[batch_indices, seq_lengths]
        scores = (last_logits[:, yes_id] - last_logits[:, no_id]).cpu().tolist()
        all_scores.extend(scores)

    return all_scores


# ============================================================
# File list management
# ============================================================

MAX_FILES_PER_REPO = 2000
SAMPLE_SIZE = 1000


def get_scored_file_list(all_py_files: List[str], gt_files: Set[str]) -> Tuple[List[str], float]:
    """
    Get the file list to score. If >MAX_FILES_PER_REPO, sample SAMPLE_SIZE
    random files + all GT files. Returns (file_list, oracle_recall).

    oracle_recall = fraction of GT files present in the scored set.
    """
    if len(all_py_files) <= MAX_FILES_PER_REPO:
        # Score all files
        oracle_hit = len(gt_files & set(all_py_files))
        oracle_recall = oracle_hit / len(gt_files) if gt_files else 1.0
        return all_py_files, oracle_recall

    # Large repo: sample + GT
    all_set = set(all_py_files)
    non_gt = [f for f in all_py_files if f not in gt_files]
    gt_in_repo = [f for f in gt_files if f in all_set]

    rng = random.Random(42)
    if len(non_gt) > SAMPLE_SIZE:
        sampled = rng.sample(non_gt, SAMPLE_SIZE)
    else:
        sampled = non_gt

    scored = list(set(sampled + gt_in_repo))
    oracle_recall = len(gt_in_repo) / len(gt_files) if gt_files else 1.0
    return scored, oracle_recall


# ============================================================
# Metrics
# ============================================================

def compute_hit_at_k(predicted: List[str], gt: Set[str], k: int) -> float:
    if not gt:
        return 0.0
    top_k = set(predicted[:k])
    return len(top_k & gt) / len(gt)


def _dcg(rels, k):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels[:k]))


# ============================================================
# Main evaluation
# ============================================================

def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    # --- Load test data ---
    print(f"Loading test data from {args.test_data}...")
    test_data = []
    with open(args.test_data) as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"  {len(test_data)} examples")

    # --- Load file trees ---
    print(f"Loading file trees from {args.file_tree_dir}...")
    file_trees: Dict[str, List[str]] = {}
    for fname in os.listdir(args.file_tree_dir):
        if fname.endswith('.json'):
            repo = fname.replace('.json', '')
            with open(os.path.join(args.file_tree_dir, fname)) as f:
                data = json.load(f)
            file_trees[repo] = data['py_files']
    print(f"  {len(file_trees)} repos loaded")

    # --- Group test examples by repo for efficiency ---
    repo_examples: Dict[str, List[Tuple[int, dict]]] = defaultdict(list)
    for idx, item in enumerate(test_data):
        repo_examples[item['repo']].append((idx, item))

    # --- Load model ---
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model in 4-bit from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
    )

    if args.lora_path:
        print(f"Loading LoRA adapter from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path)

    model.eval()
    yes_id, no_id = get_yes_no_token_ids(tokenizer)
    print(f"  Yes={yes_id}, No={no_id}")
    print(f"  GPU memory: {torch.cuda.memory_allocated(device) / 1024**3:.1f} GB")
    print(f"  Prompt mode: {args.prompt_mode}")
    print(f"  Batch size: {args.score_batch_size}")

    # Select prompt builder
    if args.prompt_mode == "path_only":
        prompt_builder = build_path_only_prompt
    elif args.prompt_mode == "code_centric":
        if not args.repo_dir:
            raise ValueError("--repo_dir is required for code_centric prompt mode")
        prompt_builder = build_code_centric_prompt
    else:
        raise ValueError(f"Unknown prompt_mode: {args.prompt_mode}")

    # --- Evaluate ---
    k_values = [1, 5, 10]
    hit_at_k = defaultdict(list)
    oracle_recalls = []
    predictions = []
    total_scored = 0
    total_examples_done = 0
    start_time = time.time()

    repos_sorted = sorted(repo_examples.keys())
    print(f"\nProcessing {len(repos_sorted)} repos, {len(test_data)} examples total...")

    for repo_idx, repo in enumerate(repos_sorted):
        examples = repo_examples[repo]

        if repo not in file_trees:
            print(f"  WARNING: no file tree for {repo}, skipping {len(examples)} examples")
            continue

        all_py_files = file_trees[repo]
        print(f"\n[Repo {repo_idx+1}/{len(repos_sorted)}] {repo}: "
              f"{len(all_py_files)} py files, {len(examples)} examples")

        for ex_idx, (global_idx, item) in enumerate(examples):
            issue_text = item.get("issue_text", item.get("text", ""))
            gt_files = set(item.get("changed_py_files", []))

            if not gt_files:
                continue

            # Get file list (with sampling for large repos)
            scored_files, oracle_recall = get_scored_file_list(all_py_files, gt_files)
            oracle_recalls.append(oracle_recall)
            total_scored += len(scored_files)

            # Build prompts
            prompts = []
            for fpath in scored_files:
                if args.prompt_mode == "code_centric":
                    code_content = extract_code_content(
                        args.repo_dir, repo, fpath,
                        head_lines=args.code_head_lines,
                        max_chars=args.code_max_chars)
                    prompt = prompt_builder(
                        issue_text=issue_text,
                        candidate_path=fpath,
                        code_content=code_content,
                        tokenizer=tokenizer,
                        max_seq_length=args.max_seq_length)
                else:
                    prompt = prompt_builder(
                        issue_text=issue_text,
                        candidate_path=fpath)
                prompts.append(prompt)

            # Score
            scores = score_files_batched(
                model, tokenizer, prompts,
                yes_id, no_id, args.max_seq_length, device,
                batch_size=args.score_batch_size)

            # Rank
            ranked = sorted(zip(scored_files, scores), key=lambda x: -x[1])
            predicted = [c for c, s in ranked]

            # Metrics
            for k in k_values:
                h = compute_hit_at_k(predicted, gt_files, k)
                hit_at_k[k].append(h)

            pred_entry = {
                "repo": repo,
                "issue_id": str(item.get("issue_id", "")),
                "ground_truth": list(gt_files),
                "predicted_top20": predicted[:20],
                "scores_top20": [s for _, s in ranked[:20]],
                "num_scored_files": len(scored_files),
                "num_total_py_files": len(all_py_files),
                "oracle_recall": oracle_recall,
                "was_sampled": len(all_py_files) > MAX_FILES_PER_REPO,
            }
            predictions.append(pred_entry)
            total_examples_done += 1

            if total_examples_done % 50 == 0:
                elapsed = time.time() - start_time
                r1 = np.mean(hit_at_k[1]) * 100
                r5 = np.mean(hit_at_k[5]) * 100
                avg_files = total_scored / total_examples_done
                eta_h = (elapsed / total_examples_done) * (len(test_data) - total_examples_done) / 3600
                print(f"  [{total_examples_done}/{len(test_data)}] "
                      f"R@1={r1:.2f}% R@5={r5:.2f}% "
                      f"avg_files={avg_files:.0f} "
                      f"elapsed={elapsed/3600:.1f}h ETA={eta_h:.1f}h")

    # ============================================================
    # Summary
    # ============================================================
    if not predictions:
        raise RuntimeError("No predictions generated. Check file trees and test data.")

    wall_clock = time.time() - start_time

    overall = {}
    for k in k_values:
        overall[f"recall@{k}"] = np.mean(hit_at_k[k]) * 100

    # Strict Acc@k
    strict_acc = {k: [] for k in k_values}
    for p in predictions:
        gt = set(p["ground_truth"])
        pred = p["predicted_top20"]
        for k in k_values:
            strict_acc[k].append(1.0 if gt and gt <= set(pred[:k]) else 0.0)
    for k in k_values:
        overall[f"acc@{k}"] = np.mean(strict_acc[k]) * 100

    # Oracle recall
    overall["oracle_recall"] = np.mean(oracle_recalls) * 100

    # NDCG@k
    ndcg_at_k = {k: [] for k in k_values}
    for p in predictions:
        gt = set(p["ground_truth"])
        pred = p["predicted_top20"]
        if not gt:
            for k in k_values:
                ndcg_at_k[k].append(0.0)
            continue
        rels = [1.0 if f in gt else 0.0 for f in pred]
        for k in k_values:
            n_rel = min(len(gt), k)
            ideal = [1.0] * n_rel + [0.0] * (k - n_rel)
            idcg = _dcg(ideal, k)
            ndcg_at_k[k].append(_dcg(rels, k) / idcg if idcg > 0 else 0.0)
    for k in k_values:
        overall[f"ndcg@{k}"] = np.mean(ndcg_at_k[k]) * 100

    # Bootstrap CI
    np.random.seed(42)
    n_boot = 10000
    bootstrap_ci = {}
    for k in k_values:
        vals = np.array(hit_at_k[k])
        boot = [np.mean(np.random.choice(vals, len(vals), replace=True)) * 100
                for _ in range(n_boot)]
        bootstrap_ci[f"recall@{k}"] = {
            "mean": float(np.mean(boot)),
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
        }

    # Per-repo breakdown
    per_repo_preds = defaultdict(list)
    for p in predictions:
        per_repo_preds[p["repo"]].append(p)

    per_repo = {}
    for repo, preds in sorted(per_repo_preds.items()):
        repo_metrics = {"n_examples": len(preds)}
        for k in k_values:
            repo_h = [compute_hit_at_k(p["predicted_top20"], set(p["ground_truth"]), k)
                       for p in preds]
            repo_metrics[f"recall@{k}"] = np.mean(repo_h) * 100
        per_repo[repo] = repo_metrics

    # Print results
    print(f"\n{'='*60}")
    print(f"ALL-FILES RERANKING RESULTS ({len(predictions)} examples, {wall_clock/3600:.1f}h)")
    print(f"{'='*60}")
    print(f"  Prompt mode: {args.prompt_mode}")
    print(f"  LoRA: {args.lora_path}")
    print(f"  Oracle recall: {overall['oracle_recall']:.2f}%")
    print(f"  Avg files scored per example: {total_scored / len(predictions):.0f}")
    for k in k_values:
        ci = bootstrap_ci[f"recall@{k}"]
        print(f"  R@{k}: {overall[f'recall@{k}']:.2f}% "
              f"[{ci['ci_lo']:.2f}, {ci['ci_hi']:.2f}]  "
              f"Acc@{k}: {overall[f'acc@{k}']:.2f}%")
    print(f"  Repos: {len(per_repo)}")

    # Sampled repos info
    sampled_count = sum(1 for p in predictions if p["was_sampled"])
    if sampled_count > 0:
        print(f"  Examples from sampled repos (>2000 files): {sampled_count}")

    # Save predictions
    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    with open(pred_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    # Save summary
    summary = {
        "overall": overall,
        "per_repo": per_repo,
        "bootstrap_ci": bootstrap_ci,
        "config": {
            "model_path": args.model_path,
            "lora_path": args.lora_path,
            "prompt_mode": args.prompt_mode,
            "repo_dir": args.repo_dir,
            "file_tree_dir": args.file_tree_dir,
            "test_data": args.test_data,
            "quantization": "4bit-nf4",
            "max_seq_length": args.max_seq_length,
            "score_batch_size": args.score_batch_size,
            "max_files_per_repo": MAX_FILES_PER_REPO,
            "sample_size": SAMPLE_SIZE,
            "total_examples": len(predictions),
            "total_files_scored": total_scored,
            "avg_files_per_example": total_scored / len(predictions),
        },
        "wall_clock_seconds": wall_clock,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {args.output_dir}")
    return overall


def main():
    parser = argparse.ArgumentParser(
        description="All-files reranking eval (no BM25 candidate pool)")
    parser.add_argument("--model_path", required=True,
                        help="Base model path (e.g. Qwen2.5-7B-Instruct)")
    parser.add_argument("--lora_path", default=None,
                        help="LoRA adapter path")
    parser.add_argument("--test_data", required=True,
                        help="Test JSONL file")
    parser.add_argument("--file_tree_dir", required=True,
                        help="Directory with {repo}.json file trees")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for predictions and summary")
    parser.add_argument("--prompt_mode", default="path_only",
                        choices=["path_only", "code_centric"],
                        help="Prompt format to use")
    parser.add_argument("--repo_dir", default=None,
                        help="Directory with repo source code (required for code_centric)")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--score_batch_size", type=int, default=16)
    parser.add_argument("--code_head_lines", type=int, default=50,
                        help="Lines of code to include (code_centric mode)")
    parser.add_argument("--code_max_chars", type=int, default=1500,
                        help="Max chars of code content (code_centric mode)")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
