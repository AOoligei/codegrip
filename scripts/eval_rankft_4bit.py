#!/usr/bin/env python3
"""
4-bit quantized version of eval_rankft.py for running on GPUs with limited VRAM.
Qwen2.5-7B in 4-bit needs ~5GB VRAM vs ~14GB in bf16.
"""
import os
import sys
import json
import argparse
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

try:
    from experiment_automation import resolve_candidate_path
except ImportError:
    from scripts.experiment_automation import resolve_candidate_path

torch.manual_seed(42)
np.random.seed(42)

PROMPT_TEMPLATE = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)


def build_prompt(issue_text: str, candidate_path: str) -> str:
    return PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=candidate_path)


def get_yes_no_token_ids(tokenizer):
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    return yes_ids[0], no_ids[0]


@torch.no_grad()
def score_candidates_batched(model, tokenizer, issue_text, candidates, yes_id, no_id,
                              max_seq_length, device, batch_size=4):
    prompts = [build_prompt(issue_text, cand) for cand in candidates]
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


def compute_hit_at_k(predicted, gt, k):
    if not gt:
        return 0.0
    top_k = set(predicted[:k])
    return len(top_k & gt) / len(gt)


def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"
    candidate_path, candidate_pool = resolve_candidate_path(
        bm25_candidates=args.bm25_candidates,
        graph_candidates=args.graph_candidates,
        hybrid_candidates=args.hybrid_candidates,
    )

    # Load data
    print(f"Loading test data from {args.test_data}...")
    test_data = []
    with open(args.test_data) as f:
        for line in f:
            test_data.append(json.loads(line))

    print(f"Loading {candidate_pool} candidates from {candidate_path}...")
    bm25_data = {}
    with open(candidate_path) as f:
        for line in f:
            item = json.loads(line)
            key = (item["repo"], str(item["issue_id"]))
            bm25_data[key] = item

    # Load model in 4-bit
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

    # Evaluate
    predictions = []
    k_values = [1, 3, 5, 10, 20]
    hit_at_k = defaultdict(list)
    recall_at_k = defaultdict(list)
    cond_acc_at_1 = []
    start_time = time.time()

    for idx, item in enumerate(test_data):
        repo = item["repo"]
        issue_id = str(item["issue_id"])
        key = (repo, issue_id)

        if key not in bm25_data:
            continue

        bm25_item = bm25_data[key]
        candidates = bm25_item.get("candidates", bm25_item.get("bm25_candidates", []))
        gt_files = set(item.get("changed_py_files", []))

        if not gt_files or not candidates:
            continue

        candidates = candidates[:args.top_k]
        issue_text = item.get("issue_text", item.get("text", ""))

        scores = score_candidates_batched(
            model, tokenizer, issue_text, candidates,
            yes_id, no_id, args.max_seq_length, device,
            batch_size=args.score_batch_size,
        )

        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        predicted = [c for c, s in ranked]

        gt_in_candidates = bool(gt_files & set(candidates))

        for k in k_values:
            h = compute_hit_at_k(predicted, gt_files, k)
            hit_at_k[k].append(h)
            recall_at_k[k].append(h)

        if gt_in_candidates:
            cond_acc_at_1.append(1.0 if predicted[0] in gt_files else 0.0)

        pred_entry = {
            "repo": repo,
            "issue_id": issue_id,
            "ground_truth": list(gt_files),
            "predicted": predicted[:50],
            "bm25_original": candidates[:20],
            "scores": [s for _, s in ranked[:50]],
            "gt_in_candidates": gt_in_candidates,
            "num_candidates": len(candidates),
        }
        predictions.append(pred_entry)

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            r1 = np.mean(recall_at_k[1]) * 100
            print(f"  [{idx+1}/{len(test_data)}] R@1={r1:.2f}% ({elapsed:.0f}s)")

    # Summary
    if not predictions:
        raise RuntimeError(
            "No predictions were generated. Check that the candidate file matches "
            "the test set and includes candidate paths."
        )

    # --- Compute metrics ---
    overall = {}
    for k in k_values:
        overall[f"hit@{k}"] = np.mean(hit_at_k[k]) * 100
        overall[f"recall@{k}"] = np.mean(recall_at_k[k]) * 100

    # Strict Acc@k: all GT files must be in top-k (1.0 or 0.0)
    strict_acc = {k: [] for k in k_values}
    for p in predictions:
        gt = set(p["ground_truth"])
        pred = p["predicted"]
        for k in k_values:
            strict_acc[k].append(1.0 if gt and gt <= set(pred[:k]) else 0.0)
    for k in k_values:
        overall[f"acc@{k}"] = np.mean(strict_acc[k]) * 100

    overall["cond_acc@1|gt_in_candidates"] = np.mean(cond_acc_at_1) * 100 if cond_acc_at_1 else 0

    # NDCG@k (binary relevance, ideal DCG from full GT set)
    def _dcg(rels, k):
        return sum(r / np.log2(i + 2) for i, r in enumerate(rels[:k]))

    ndcg_at_k = {k: [] for k in k_values}
    for p in predictions:
        gt = set(p["ground_truth"])
        if not gt:
            for k in k_values:
                ndcg_at_k[k].append(0.0)
            continue
        pred = p["predicted"]
        rels = [1.0 if f in gt else 0.0 for f in pred]
        for k in k_values:
            n_rel = min(len(gt), k)
            ideal = [1.0] * n_rel + [0.0] * (k - n_rel)
            idcg = _dcg(ideal, k)
            ndcg_at_k[k].append(_dcg(rels, k) / idcg if idcg > 0 else 0.0)
    for k in k_values:
        overall[f"ndcg@{k}"] = np.mean(ndcg_at_k[k]) * 100

    # --- Per-repo breakdown ---
    from collections import defaultdict as _defaultdict
    per_repo_preds = _defaultdict(list)
    for p in predictions:
        per_repo_preds[p["repo"]].append(p)

    per_repo = {}
    for repo, preds in sorted(per_repo_preds.items()):
        repo_metrics = {"n_examples": len(preds)}
        for k in k_values:
            repo_h = [compute_hit_at_k(p["predicted"], set(p["ground_truth"]), k) for p in preds]
            repo_metrics[f"hit@{k}"] = np.mean(repo_h) * 100
            repo_sa = [1.0 if set(p["ground_truth"]) <= set(p["predicted"][:k]) else 0.0 for p in preds]
            repo_metrics[f"acc@{k}"] = np.mean(repo_sa) * 100
        per_repo[repo] = repo_metrics

    # --- Bootstrap CI (instance-level) ---
    np.random.seed(42)
    n_boot = 10000
    bootstrap_ci = {}
    for k in k_values:
        vals = np.array(hit_at_k[k])
        boot = [np.mean(np.random.choice(vals, len(vals), replace=True)) * 100 for _ in range(n_boot)]
        bootstrap_ci[f"recall@{k}"] = {
            "mean": float(np.mean(boot)),
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
        }

    wall_clock = time.time() - start_time

    print(f"\n=== Results ({len(predictions)} examples, {wall_clock:.0f}s) ===")
    for k in k_values:
        ci = bootstrap_ci[f"recall@{k}"]
        print(f"  R@{k}: {overall[f'recall@{k}']:.2f}% [{ci['ci_lo']:.2f}, {ci['ci_hi']:.2f}]  Acc@{k}: {overall[f'acc@{k}']:.2f}%")
    print(f"  Cond Acc@1: {overall['cond_acc@1|gt_in_candidates']:.2f}%")
    print(f"  Repos: {len(per_repo)}")

    # Save
    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    with open(pred_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    summary = {
        "overall": overall,
        "per_repo": per_repo,
        "bootstrap_ci": bootstrap_ci,
        "config": {
            "model_path": args.model_path,
            "lora_path": args.lora_path,
            "candidate_pool": candidate_pool,
            "candidate_path": candidate_path,
            "test_data": getattr(args, 'test_data', ''),
            "bm25_candidates": getattr(args, 'bm25_candidates', ''),
            "quantization": "4bit-nf4",
            "top_k": args.top_k,
            "max_seq_length": args.max_seq_length,
            "total_examples": len(predictions),
        },
        "wall_clock_seconds": wall_clock,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--bm25_candidates", default=None)
    parser.add_argument("--graph_candidates", default=None)
    parser.add_argument("--hybrid_candidates", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu_id", type=int, default=7)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--score_batch_size", type=int, default=2)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
