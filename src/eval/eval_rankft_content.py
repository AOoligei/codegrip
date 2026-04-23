"""
Evaluate RankFT-Content (content-aware reranker) on GREPO/SWE-bench.

Same as eval_rankft.py but includes file summaries in the prompt.
Must match the training prompt format.

Usage:
    python src/eval/eval_rankft_content.py \
        --model_path /path/to/Qwen2.5-7B \
        --lora_path experiments/rankft_content_v1/final \
        --test_data data/grepo_text/grepo_test.jsonl \
        --bm25_candidates data/rankft/grepo_test_bm25_top500.jsonl \
        --file_summaries data/file_summaries_all.json \
        --output_dir experiments/rankft_content_v1/eval \
        --gpu_id 0
"""

import os
import json
import argparse
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# Prompt (content-aware version — must match training)
# ============================================================

PROMPT_TEMPLATE_CONTENT = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n"
    "Content: {file_summary}\n\n"
    "Answer:"
)

PROMPT_TEMPLATE_PATH_ONLY = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)


def build_prompt(issue_text, candidate_path, file_summary=None):
    if file_summary:
        return PROMPT_TEMPLATE_CONTENT.format(
            issue_text=issue_text,
            candidate_path=candidate_path,
            file_summary=file_summary,
        )
    return PROMPT_TEMPLATE_PATH_ONLY.format(
        issue_text=issue_text,
        candidate_path=candidate_path,
    )


def load_file_summaries(path):
    if os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
        print(f"  Loaded file summaries for {len(data)} repos from {path}")
        return data
    if os.path.isdir(path):
        data = {}
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".json") and fname != "file_summaries_all.json":
                repo = fname.replace(".json", "")
                with open(os.path.join(path, fname)) as f:
                    data[repo] = json.load(f)
        print(f"  Loaded file summaries for {len(data)} repos from {path}/")
        return data
    print(f"  Warning: no file summaries at {path}")
    return {}


def get_summary(file_summaries, repo, file_path):
    repo_data = file_summaries.get(repo, {})
    # Try exact match
    s = repo_data.get(file_path)
    if s:
        return s
    # For SWE-bench repos, the repo key might be "org__repo" format
    # Try matching by repo name suffix
    for key in repo_data:
        if key == repo or key.endswith(f"__{repo}") or key.endswith(f"/{repo}"):
            return repo_data.get(key, {}).get(file_path)
    return None


def get_yes_no_token_ids(tokenizer):
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    return yes_ids[0], no_ids[0]


@torch.no_grad()
def score_candidates_batched(
    model, tokenizer, issue_text, candidates,
    yes_id, no_id, max_seq_length, device,
    file_summaries=None, repo=None,
    batch_size=16,
):
    prompts = []
    for cand in candidates:
        summary = None
        if file_summaries and repo:
            summary = get_summary(file_summaries, repo, cand)
        prompts.append(build_prompt(issue_text, cand, summary))

    all_scores = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        encodings = tokenizer(
            batch_prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=max_seq_length,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                for prompt in batch_prompts:
                    enc = tokenizer([prompt], return_tensors="pt",
                                    truncation=True, max_length=max_seq_length)
                    ids = enc["input_ids"].to(device)
                    mask = enc["attention_mask"].to(device)
                    out = model(input_ids=ids, attention_mask=mask)
                    score = (out.logits[0, -1, yes_id] - out.logits[0, -1, no_id]).item()
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
    return len(set(predicted[:k]) & gt) / len(gt)


def compute_acc_at_k(predicted, gt, k):
    if not gt:
        return 0.0
    return 1.0 if gt.issubset(set(predicted[:k])) else 0.0


def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    print(f"Loading test data from {args.test_data}...")
    test_data = []
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            if item.get("changed_py_files"):
                test_data.append(item)
    print(f"  {len(test_data)} test examples")

    print(f"Loading BM25 candidates from {args.bm25_candidates}...")
    bm25_map = {}
    with open(args.bm25_candidates) as f:
        for line in f:
            item = json.loads(line)
            key = f"{item['repo']}_{item['issue_id']}"
            bm25_map[key] = item.get("candidates", item.get("bm25_candidates", []))
    print(f"  BM25 candidates for {len(bm25_map)} examples")

    print(f"Loading file summaries...")
    file_summaries = load_file_summaries(args.file_summaries)

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    yes_id, no_id = get_yes_no_token_ids(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True,
    )
    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    k_values = [1, 3, 5, 10, 20]
    overall_metrics = {f"hit@{k}": [] for k in k_values}
    overall_metrics.update({f"acc@{k}": [] for k in k_values})
    cond_acc1_correct = 0
    cond_acc1_total = 0
    results = []
    summary_coverage = {"with": 0, "without": 0}

    total = len(test_data)
    start_time = time.time()

    for idx, example in enumerate(test_data):
        repo = example["repo"]
        issue_id = example["issue_id"]
        issue_text = example["issue_text"]
        gt_files = set(example["changed_py_files"])

        bm25_key = f"{repo}_{issue_id}"
        candidates = bm25_map.get(bm25_key, [])
        if not candidates:
            continue

        candidates = candidates[:args.top_k]
        gt_in_candidates = bool(gt_files & set(candidates))

        # Track summary coverage
        for c in candidates[:5]:
            if get_summary(file_summaries, repo, c):
                summary_coverage["with"] += 1
            else:
                summary_coverage["without"] += 1

        if idx % 20 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / max(elapsed, 1)
            eta = (total - idx - 1) / max(rate, 0.001)
            print(f"  [{idx+1}/{total}] {repo}#{issue_id} | "
                  f"{len(candidates)} cands | GT∈cands: {gt_in_candidates} | "
                  f"ETA: {eta:.0f}s")

        scores = score_candidates_batched(
            model, tokenizer, issue_text, candidates,
            yes_id, no_id, args.max_seq_length, device,
            file_summaries=file_summaries, repo=repo,
            batch_size=args.score_batch_size,
        )

        scored = sorted(zip(candidates, scores), key=lambda x: -x[1])
        reranked = [c for c, _ in scored]

        metrics = {}
        for k in k_values:
            hit = compute_hit_at_k(reranked, gt_files, k)
            acc = compute_acc_at_k(reranked, gt_files, k)
            metrics[f"hit@{k}"] = hit
            metrics[f"acc@{k}"] = acc
            overall_metrics[f"hit@{k}"].append(hit)
            overall_metrics[f"acc@{k}"].append(acc)

        if gt_in_candidates:
            cond_acc1_total += 1
            if reranked[0] in gt_files:
                cond_acc1_correct += 1

        results.append({
            "repo": repo, "issue_id": issue_id,
            "ground_truth": list(gt_files),
            "predicted": reranked[:50],
            "metrics": metrics,
            "gt_in_candidates": gt_in_candidates,
        })

    total_evaluated = len(results)
    elapsed_total = time.time() - start_time

    avg = {}
    for name, vals in overall_metrics.items():
        avg[name] = sum(vals) / len(vals) * 100 if vals else 0.0

    cond_acc1 = cond_acc1_correct / cond_acc1_total * 100 if cond_acc1_total else 0

    print(f"\n{'='*70}")
    print(f"RANKFT-CONTENT EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  Model: {args.model_path}")
    if args.lora_path:
        print(f"  LoRA: {args.lora_path}")
    print(f"  Examples: {total_evaluated}")
    print(f"  Time: {elapsed_total:.0f}s ({elapsed_total/3600:.2f}h)")
    print(f"  Summary coverage: {summary_coverage}")

    print(f"\nOVERALL:")
    for k in k_values:
        print(f"  Hit@{k}: {avg[f'hit@{k}']:.2f}%  |  Acc@{k}: {avg[f'acc@{k}']:.2f}%")
    print(f"  Cond. Acc@1|GT∈cands: {cond_acc1:.2f}% ({cond_acc1_correct}/{cond_acc1_total})")

    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    with open(pred_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "overall": avg,
        "cond_acc1": cond_acc1,
        "config": vars(args),
        "summary_coverage": summary_coverage,
        "wall_clock_seconds": round(elapsed_total, 2),
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--bm25_candidates", required=True)
    parser.add_argument("--file_summaries", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--max_seq_length", type=int, default=768)
    parser.add_argument("--score_batch_size", type=int, default=8)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
