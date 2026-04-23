#!/usr/bin/env python3
"""
Two-stage reranking experiment.

Stage A: Path-only scoring (standard) → top-K candidates
Stage B: Re-score top-K with augmented prompts (path + file content snippet)
Final: Combine Stage A and Stage B scores

Tests whether adding file content at inference time helps a path-trained model.
Also tests content-only second pass and summary-based second pass.
"""
import os
import sys
import json
import argparse
import time
import random
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROMPT_PATH_ONLY = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)

PROMPT_WITH_CONTENT = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n"
    "Content (first 30 lines):\n{content}\n\n"
    "Answer:"
)

PROMPT_WITH_SUMMARY = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n"
    "Summary: {summary}\n\n"
    "Answer:"
)


def get_yes_no_token_ids(tokenizer):
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    return yes_ids[0], no_ids[0]


@torch.no_grad()
def score_batch(model, tokenizer, prompts, yes_id, no_id, max_seq_length, device):
    encodings = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=max_seq_length,
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            scores = []
            for prompt in prompts:
                enc = tokenizer([prompt], return_tensors="pt",
                               truncation=True, max_length=max_seq_length)
                ids = enc["input_ids"].to(device)
                mask = enc["attention_mask"].to(device)
                out = model(input_ids=ids, attention_mask=mask)
                logits = out.logits[0, -1]
                scores.append((logits[yes_id] - logits[no_id]).item())
            return scores
        raise

    logits = outputs.logits
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(logits.size(0), device=device)
    last_logits = logits[batch_indices, seq_lengths]
    return (last_logits[:, yes_id] - last_logits[:, no_id]).cpu().tolist()


def score_candidates(model, tokenizer, prompts, yes_id, no_id, max_seq_length,
                     device, batch_size=16):
    all_scores = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        scores = score_batch(model, tokenizer, batch, yes_id, no_id,
                            max_seq_length, device)
        all_scores.extend(scores)
    return all_scores


def read_file_content(repos_dir, repo, filepath, max_lines=30):
    """Read first max_lines of a file from repo."""
    full_path = os.path.join(repos_dir, repo, filepath)
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


def compute_recall_at_k(predicted, gt_set, k):
    if not gt_set:
        return 0.0
    return len(set(predicted[:k]) & gt_set) / len(gt_set)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data/shuyang/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_path",
                        default=os.path.join(BASE_DIR, "experiments/rankft_runB_graph/best"))
    parser.add_argument("--test_data",
                        default=os.path.join(BASE_DIR, "data/grepo_text/grepo_test.jsonl"))
    parser.add_argument("--candidates_file",
                        default=os.path.join(BASE_DIR, "data/rankft/merged_bm25_exp6_candidates.jsonl"))
    parser.add_argument("--repos_dir",
                        default=os.path.join(BASE_DIR, "data/repos"))
    parser.add_argument("--summaries_dir",
                        default=os.path.join(BASE_DIR, "data/file_summaries"))
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir",
                        default=os.path.join(BASE_DIR, "experiments/twostage_rerank"))
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--second_stage_seq_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--topk_rerank", type=int, default=50,
                        help="How many top candidates to re-score in stage 2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    print("=== Two-Stage Reranking Experiment ===")

    # Load test data
    test_data = {}
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            if item.get("changed_py_files"):
                key = f"{item['repo']}_{item['issue_id']}"
                test_data[key] = item
    print(f"  {len(test_data)} test examples")

    # Load candidates
    cand_map = {}
    with open(args.candidates_file) as f:
        for line in f:
            item = json.loads(line)
            key = f"{item['repo']}_{item['issue_id']}"
            cand_map[key] = item.get("candidates", [])

    # Load file summaries
    file_summaries = {}
    if os.path.isdir(args.summaries_dir):
        for fname in os.listdir(args.summaries_dir):
            if fname.endswith(".json"):
                with open(os.path.join(args.summaries_dir, fname)) as f:
                    data = json.load(f)
                repo = data.get("repo", fname.replace(".json", "").replace("_", "/"))
                file_summaries[repo] = data.get("summaries", {})
    print(f"  Summaries for {len(file_summaries)} repos")

    # Match
    examples = []
    for key, td in test_data.items():
        if key in cand_map and cand_map[key]:
            examples.append({
                "key": key,
                "repo": td["repo"],
                "issue_id": td["issue_id"],
                "issue_text": td["issue_text"],
                "gt_files": td["changed_py_files"],
                "candidates": cand_map[key],
            })
    print(f"  {len(examples)} matched examples")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    yes_id, no_id = get_yes_no_token_ids(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()
    print("  Model loaded.")

    # Evaluate
    k_values = [1, 3, 5, 10, 20]
    # Track results for multiple strategies
    strategy_names = [
        "path_only",           # baseline: just stage 1
        "twostage_content",    # stage1 + stage2 with content (combined)
        "twostage_summary",    # stage1 + stage2 with summary (combined)
        "content_replace",     # stage2 content score replaces stage1 for top-K
        "summary_replace",     # stage2 summary score replaces stage1 for top-K
    ]
    results = {s: {f"recall@{k}": [] for k in k_values} for s in strategy_names}
    cond_acc = {s: [0, 0] for s in strategy_names}  # [correct, total]

    t0 = time.time()
    for idx, ex in enumerate(examples):
        candidates = ex["candidates"]
        issue_text = ex["issue_text"]
        repo = ex["repo"]
        gt_set = set(ex["gt_files"])
        gt_in_pool = bool(gt_set & set(candidates))

        # Stage 1: Path-only scoring
        path_prompts = [PROMPT_PATH_ONLY.format(issue_text=issue_text, candidate_path=c)
                        for c in candidates]
        path_scores = score_candidates(
            model, tokenizer, path_prompts, yes_id, no_id,
            args.max_seq_length, device, args.batch_size
        )

        # Get top-K for stage 2
        scored = sorted(zip(candidates, path_scores), key=lambda x: -x[1])
        top_k_files = [f for f, _ in scored[:args.topk_rerank]]
        top_k_scores = [s for _, s in scored[:args.topk_rerank]]
        rest_files = [f for f, _ in scored[args.topk_rerank:]]
        rest_scores = [s for _, s in scored[args.topk_rerank:]]

        # Stage 2a: Re-score top-K with content
        repo_summaries = file_summaries.get(repo, {})
        content_prompts = []
        summary_prompts = []
        for f in top_k_files:
            content = read_file_content(args.repos_dir, repo, f, max_lines=30)
            content_prompts.append(PROMPT_WITH_CONTENT.format(
                issue_text=issue_text, candidate_path=f,
                content=content if content else "(empty)"
            ))
            summary = repo_summaries.get(f, "")
            summary_prompts.append(PROMPT_WITH_SUMMARY.format(
                issue_text=issue_text, candidate_path=f,
                summary=summary if summary else "N/A"
            ))

        content_scores = score_candidates(
            model, tokenizer, content_prompts, yes_id, no_id,
            args.second_stage_seq_length, device, args.batch_size
        )
        summary_scores = score_candidates(
            model, tokenizer, summary_prompts, yes_id, no_id,
            args.second_stage_seq_length, device, args.batch_size
        )

        # Strategy 1: Path only (baseline)
        path_ranked = [f for f, _ in scored]

        # Strategy 2: Two-stage content (combine stage1 + stage2 scores)
        # Normalize and combine: 0.6 * path + 0.4 * content for top-K
        ps = np.array(top_k_scores)
        cs = np.array(content_scores)
        ps_norm = (ps - ps.min()) / (ps.max() - ps.min() + 1e-8)
        cs_norm = (cs - cs.min()) / (cs.max() - cs.min() + 1e-8) if cs.max() > cs.min() else np.zeros_like(cs)
        combined_content = 0.6 * ps_norm + 0.4 * cs_norm
        reranked_content = sorted(zip(top_k_files, combined_content), key=lambda x: -x[1])
        twostage_content_ranked = [f for f, _ in reranked_content] + rest_files

        # Strategy 3: Two-stage summary
        ss = np.array(summary_scores)
        ss_norm = (ss - ss.min()) / (ss.max() - ss.min() + 1e-8) if ss.max() > ss.min() else np.zeros_like(ss)
        combined_summary = 0.6 * ps_norm + 0.4 * ss_norm
        reranked_summary = sorted(zip(top_k_files, combined_summary), key=lambda x: -x[1])
        twostage_summary_ranked = [f for f, _ in reranked_summary] + rest_files

        # Strategy 4: Content replaces (just use content scores for top-K)
        content_replace_ranked = sorted(zip(top_k_files, content_scores), key=lambda x: -x[1])
        content_replace_ranked = [f for f, _ in content_replace_ranked] + rest_files

        # Strategy 5: Summary replaces
        summary_replace_ranked = sorted(zip(top_k_files, summary_scores), key=lambda x: -x[1])
        summary_replace_ranked = [f for f, _ in summary_replace_ranked] + rest_files

        # Compute metrics
        all_rankings = {
            "path_only": path_ranked,
            "twostage_content": twostage_content_ranked,
            "twostage_summary": twostage_summary_ranked,
            "content_replace": content_replace_ranked,
            "summary_replace": summary_replace_ranked,
        }

        for sname, ranked in all_rankings.items():
            for k in k_values:
                results[sname][f"recall@{k}"].append(compute_recall_at_k(ranked, gt_set, k))
            if gt_in_pool:
                cond_acc[sname][1] += 1
                if ranked[0] in gt_set:
                    cond_acc[sname][0] += 1

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            r1_path = np.mean(results["path_only"]["recall@1"]) * 100
            r1_content = np.mean(results["twostage_content"]["recall@1"]) * 100
            r1_summary = np.mean(results["twostage_summary"]["recall@1"]) * 100
            print(f"  [{idx+1}/{len(examples)}] path={r1_path:.2f}% "
                  f"2stage_content={r1_content:.2f}% 2stage_summary={r1_summary:.2f}% "
                  f"({elapsed:.0f}s)")

    elapsed = time.time() - t0

    # Report
    print(f"\n{'='*70}")
    print("TWO-STAGE RERANKING RESULTS")
    print(f"{'='*70}")
    print(f"Examples: {len(examples)}, top-K rerank: {args.topk_rerank}")
    print(f"Time: {elapsed:.0f}s\n")

    print(f"{'Strategy':<25} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'R@20':>7} {'C.Acc@1':>8}")
    print("-" * 70)
    for sname in strategy_names:
        r = {k: np.mean(v) * 100 for k, v in results[sname].items()}
        ca = cond_acc[sname][0] / max(cond_acc[sname][1], 1) * 100
        print(f"{sname:<25} {r['recall@1']:>7.2f} {r['recall@5']:>7.2f} "
              f"{r['recall@10']:>7.2f} {r['recall@20']:>7.2f} {ca:>8.2f}")

    # Save
    summary = {}
    for sname in strategy_names:
        summary[sname] = {
            k: np.mean(v) * 100 for k, v in results[sname].items()
        }
        summary[sname]["cond_acc1"] = cond_acc[sname][0] / max(cond_acc[sname][1], 1) * 100
    summary["config"] = {
        "topk_rerank": args.topk_rerank,
        "max_seq_length_stage1": args.max_seq_length,
        "max_seq_length_stage2": args.second_stage_seq_length,
        "n_examples": len(examples),
        "elapsed": elapsed,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
