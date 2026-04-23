#!/usr/bin/env python3
"""
Score combination experiment: try multiple strategies to improve R@1.

Re-scores all candidates with neural model, computes BM25 scores,
then tries multiple combination strategies:
1. Pure neural (baseline)
2. BM25 + neural (weighted)
3. RRF (reciprocal rank fusion)
4. Directory co-location boost
5. Temperature scaling

Saves all raw scores for offline analysis.
"""
import os
import sys
import json
import re
import math
import argparse
import time
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROMPT_TEMPLATE = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)


def get_yes_no_token_ids(tokenizer):
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    return yes_ids[0], no_ids[0]


@torch.no_grad()
def score_candidates_batched(model, tokenizer, issue_text, candidates,
                              yes_id, no_id, max_seq_length, device,
                              batch_size=16):
    """Score ALL candidates and return ALL scores."""
    prompts = [PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=c)
               for c in candidates]
    all_scores = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        encodings = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_seq_length,
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


def compute_bm25_path_score(issue_text: str, filepath: str) -> float:
    """Simple BM25-like score based on path token overlap with issue text."""
    # Tokenize issue text
    issue_tokens = set(re.findall(r'[a-z][a-z0-9]*', issue_text.lower()))
    # Tokenize file path
    path_tokens = set(re.findall(r'[a-z][a-z0-9]*', filepath.lower()))
    # Remove very common tokens
    stopwords = {'py', 'test', 'tests', 'src', 'lib', 'init', 'the', 'a', 'an',
                 'is', 'in', 'to', 'of', 'and', 'or', 'for', 'with', 'on', 'at',
                 'by', 'from', 'that', 'this', 'it', 'be', 'as', 'not', 'but',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                 'could', 'should', 'may', 'might', 'can', 'if', 'when', 'file',
                 'error', 'bug', 'issue', 'fix', 'add', 'new', 'change', 'update'}
    issue_tokens -= stopwords
    path_tokens -= stopwords
    if not path_tokens:
        return 0.0
    overlap = issue_tokens & path_tokens
    # Score: overlap count weighted by inverse path token frequency
    return len(overlap) / (1 + math.log(1 + len(path_tokens)))


def compute_recall_at_k(predicted, gt_set, k):
    if not gt_set:
        return 0.0
    top_k = set(predicted[:k])
    return len(top_k & gt_set) / len(gt_set)


def rrf_score(ranks_dict, k=60):
    """Reciprocal Rank Fusion. ranks_dict: {signal_name: {file: rank}}"""
    all_files = set()
    for r in ranks_dict.values():
        all_files.update(r.keys())
    scores = {}
    for f in all_files:
        s = 0.0
        for signal_name, rank_map in ranks_dict.items():
            rank = rank_map.get(f, len(rank_map) + 100)  # penalty for missing
            s += 1.0 / (k + rank)
        scores[f] = s
    return scores


def directory_boost(candidates, scores, boost_factor=0.15):
    """Boost files in the same directory as high-scoring files."""
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: -x[1])

    # Find top-scoring directories
    dir_scores = defaultdict(list)
    for f, s in scored[:10]:  # top 10 files
        d = os.path.dirname(f)
        dir_scores[d].append(s)

    # Compute directory boost
    dir_boost = {}
    for d, ss in dir_scores.items():
        dir_boost[d] = np.mean(ss) * boost_factor

    # Apply boost
    boosted_scores = []
    for f, s in zip(candidates, scores):
        d = os.path.dirname(f)
        boost = dir_boost.get(d, 0.0)
        boosted_scores.append(s + boost)

    return boosted_scores


def test_pair_boost(candidates, scores, boost_factor=0.5):
    """If test_foo.py scores high, boost foo.py and vice versa."""
    # Build test<->source mapping
    test_to_source = {}
    source_to_test = {}
    cand_set = set(candidates)
    for c in candidates:
        basename = os.path.basename(c)
        dirname = os.path.dirname(c)
        if '_test.py' in basename:
            source_name = basename.replace('_test.py', '.py')
            source_path = os.path.join(dirname, source_name)
            if source_path in cand_set:
                test_to_source[c] = source_path
                source_to_test[source_path] = c
        elif basename.startswith('test_'):
            source_name = basename.replace('test_', '', 1)
            # source might be in parent dir
            parent_dir = os.path.dirname(dirname)
            for candidate_source in [
                os.path.join(dirname, source_name),
                os.path.join(parent_dir, source_name),
            ]:
                if candidate_source in cand_set:
                    test_to_source[c] = candidate_source
                    source_to_test[candidate_source] = c
                    break

    # Build score lookup
    score_map = dict(zip(candidates, scores))

    # Apply mutual boosting
    boosted_scores = list(scores)
    for i, c in enumerate(candidates):
        pair = test_to_source.get(c) or source_to_test.get(c)
        if pair and pair in score_map:
            pair_score = score_map[pair]
            if pair_score > 0:  # only boost if pair is positive
                boosted_scores[i] += pair_score * boost_factor

    return boosted_scores


def evaluate_strategy(candidates_list, scores_list, gt_list, strategy_fn, **kwargs):
    """Evaluate a scoring strategy across all examples."""
    k_values = [1, 3, 5, 10, 20]
    metrics = {f"recall@{k}": [] for k in k_values}
    cond_acc1_correct = 0
    cond_acc1_total = 0

    for candidates, scores, gt_files in zip(candidates_list, scores_list, gt_list):
        gt_set = set(gt_files)
        modified_scores = strategy_fn(candidates, scores, **kwargs)
        scored = sorted(zip(candidates, modified_scores), key=lambda x: -x[1])
        ranked = [f for f, _ in scored]

        for k in k_values:
            metrics[f"recall@{k}"].append(compute_recall_at_k(ranked, gt_set, k))

        gt_in_pool = bool(gt_set & set(candidates))
        if gt_in_pool:
            cond_acc1_total += 1
            if ranked[0] in gt_set:
                cond_acc1_correct += 1

    result = {}
    for k in k_values:
        result[f"recall@{k}"] = np.mean(metrics[f"recall@{k}"]) * 100
    result["cond_acc1"] = (cond_acc1_correct / max(cond_acc1_total, 1)) * 100
    result["n"] = len(candidates_list)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data/shuyang/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_path",
                        default=os.path.join(BASE_DIR, "experiments/rankft_runB_graph/best"))
    parser.add_argument("--test_data",
                        default=os.path.join(BASE_DIR, "data/grepo_text/grepo_test.jsonl"))
    parser.add_argument("--candidates_file",
                        default=os.path.join(BASE_DIR, "data/rankft/merged_bm25_exp6_candidates.jsonl"))
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir",
                        default=os.path.join(BASE_DIR, "experiments/score_combination"))
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    print("=== Score Combination Experiment ===")

    # Load test data
    test_data = {}
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            if item.get("changed_py_files"):
                key = f"{item['repo']}_{item['issue_id']}"
                test_data[key] = item

    # Load candidates
    cand_map = {}
    with open(args.candidates_file) as f:
        for line in f:
            item = json.loads(line)
            key = f"{item['repo']}_{item['issue_id']}"
            cand_map[key] = item.get("candidates", [])

    # Match test data with candidates
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
    print(f"  {len(examples)} examples with candidates")

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

    # Score all candidates for all examples
    all_candidates = []
    all_neural_scores = []
    all_bm25_scores = []
    all_gt = []

    t0 = time.time()
    for idx, ex in enumerate(examples):
        candidates = ex["candidates"]
        issue_text = ex["issue_text"]

        # Neural scores
        neural_scores = score_candidates_batched(
            model, tokenizer, issue_text, candidates,
            yes_id, no_id, args.max_seq_length, device, args.batch_size,
        )

        # BM25-like path scores
        bm25_scores = [compute_bm25_path_score(issue_text, c) for c in candidates]

        all_candidates.append(candidates)
        all_neural_scores.append(neural_scores)
        all_bm25_scores.append(bm25_scores)
        all_gt.append(ex["gt_files"])

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            # Quick baseline check
            scored = sorted(zip(candidates, neural_scores), key=lambda x: -x[1])
            ranked = [f for f, _ in scored]
            gt_set = set(ex["gt_files"])
            r1_so_far = np.mean([
                compute_recall_at_k(
                    [f for f, _ in sorted(zip(c, s), key=lambda x: -x[1])],
                    set(g), 1
                )
                for c, s, g in zip(all_candidates, all_neural_scores, all_gt)
            ]) * 100
            print(f"  [{idx+1}/{len(examples)}] R@1(neural): {r1_so_far:.2f}% ({elapsed:.0f}s)")

    scoring_time = time.time() - t0
    print(f"\nScoring complete: {len(examples)} examples in {scoring_time:.0f}s")

    # Save raw scores
    raw_scores_path = os.path.join(args.output_dir, "raw_scores.jsonl")
    with open(raw_scores_path, "w") as f:
        for i, ex in enumerate(examples):
            f.write(json.dumps({
                "repo": ex["repo"],
                "issue_id": ex["issue_id"],
                "candidates": all_candidates[i],
                "neural_scores": all_neural_scores[i],
                "bm25_scores": all_bm25_scores[i],
                "ground_truth": all_gt[i],
            }) + "\n")
    print(f"Saved raw scores to {raw_scores_path}")

    # ==========================================
    # Try multiple combination strategies
    # ==========================================
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)

    strategies = {}

    # 1. Pure neural (baseline)
    strategies["neural_only"] = evaluate_strategy(
        all_candidates, all_neural_scores, all_gt,
        lambda c, s: s
    )

    # 2. Temperature scaling
    for temp in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        name = f"temp_{temp}"
        strategies[name] = evaluate_strategy(
            all_candidates, all_neural_scores, all_gt,
            lambda c, s, t=temp: [x / t for x in s]
        )

    # 3. BM25 + neural (weighted combination)
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        name = f"hybrid_a{alpha}"

        def hybrid_fn(candidates, neural_scores, bm25_scores_list=all_bm25_scores,
                      idx_ref=[0], a=alpha):
            # Get corresponding BM25 scores
            bm25_s = bm25_scores_list[idx_ref[0]]
            idx_ref[0] += 1

            # Min-max normalize both
            ns = np.array(neural_scores)
            bs = np.array(bm25_s)
            ns_norm = (ns - ns.min()) / (ns.max() - ns.min() + 1e-8)
            bs_norm = (bs - bs.min()) / (bs.max() - bs.min() + 1e-8) if bs.max() > bs.min() else np.zeros_like(bs)
            combined = a * ns_norm + (1 - a) * bs_norm
            return combined.tolist()

        # Reset index counter for each alpha
        idx_counter = [0]
        result = {"recall@1": [], "recall@3": [], "recall@5": [], "recall@10": [], "recall@20": []}
        cond_correct = 0
        cond_total = 0
        for i in range(len(all_candidates)):
            candidates = all_candidates[i]
            neural_s = all_neural_scores[i]
            bm25_s = all_bm25_scores[i]
            gt_set = set(all_gt[i])

            ns = np.array(neural_s)
            bs = np.array(bm25_s)
            ns_norm = (ns - ns.min()) / (ns.max() - ns.min() + 1e-8)
            bs_norm = (bs - bs.min()) / (bs.max() - bs.min() + 1e-8) if bs.max() > bs.min() else np.zeros_like(bs)
            combined = alpha * ns_norm + (1 - alpha) * bs_norm

            scored = sorted(zip(candidates, combined), key=lambda x: -x[1])
            ranked = [f for f, _ in scored]

            for k in [1, 3, 5, 10, 20]:
                result[f"recall@{k}"].append(compute_recall_at_k(ranked, gt_set, k))

            if gt_set & set(candidates):
                cond_total += 1
                if ranked[0] in gt_set:
                    cond_correct += 1

        strategies[name] = {
            f"recall@{k}": np.mean(result[f"recall@{k}"]) * 100 for k in [1, 3, 5, 10, 20]
        }
        strategies[name]["cond_acc1"] = cond_correct / max(cond_total, 1) * 100
        strategies[name]["n"] = len(all_candidates)

    # 4. RRF (neural rank + BM25 rank)
    for rrf_k in [30, 60, 100]:
        name = f"rrf_k{rrf_k}"
        result = {"recall@1": [], "recall@3": [], "recall@5": [], "recall@10": [], "recall@20": []}
        cond_correct = 0
        cond_total = 0
        for i in range(len(all_candidates)):
            candidates = all_candidates[i]
            neural_s = all_neural_scores[i]
            bm25_s = all_bm25_scores[i]
            gt_set = set(all_gt[i])

            # Neural ranking
            neural_ranked = sorted(range(len(candidates)), key=lambda j: -neural_s[j])
            neural_rank = {candidates[j]: r + 1 for r, j in enumerate(neural_ranked)}

            # BM25 ranking
            bm25_ranked = sorted(range(len(candidates)), key=lambda j: -bm25_s[j])
            bm25_rank = {candidates[j]: r + 1 for r, j in enumerate(bm25_ranked)}

            # Pool ordering (original position = implicit BM25 rank)
            pool_rank = {candidates[j]: j + 1 for j in range(len(candidates))}

            # RRF
            rrf_scores = {}
            for f in candidates:
                rrf_scores[f] = (
                    1.0 / (rrf_k + neural_rank[f]) +
                    0.3 / (rrf_k + bm25_rank[f]) +
                    0.3 / (rrf_k + pool_rank[f])
                )

            scored = sorted(rrf_scores.items(), key=lambda x: -x[1])
            ranked = [f for f, _ in scored]

            for k in [1, 3, 5, 10, 20]:
                result[f"recall@{k}"].append(compute_recall_at_k(ranked, gt_set, k))

            if gt_set & set(candidates):
                cond_total += 1
                if ranked[0] in gt_set:
                    cond_correct += 1

        strategies[name] = {
            f"recall@{k}": np.mean(result[f"recall@{k}"]) * 100 for k in [1, 3, 5, 10, 20]
        }
        strategies[name]["cond_acc1"] = cond_correct / max(cond_total, 1) * 100
        strategies[name]["n"] = len(all_candidates)

    # 5. Directory boost
    for boost in [0.05, 0.1, 0.15, 0.2, 0.3]:
        name = f"dir_boost_{boost}"
        strategies[name] = evaluate_strategy(
            all_candidates, all_neural_scores, all_gt,
            directory_boost, boost_factor=boost
        )

    # 6. Test-source pair boost
    for boost in [0.2, 0.3, 0.5, 0.7]:
        name = f"test_pair_{boost}"
        strategies[name] = evaluate_strategy(
            all_candidates, all_neural_scores, all_gt,
            test_pair_boost, boost_factor=boost
        )

    # 7. Combined: directory + test-pair + BM25 hybrid
    for alpha in [0.7, 0.8, 0.9]:
        for dir_b in [0.1, 0.15]:
            for tp_b in [0.3, 0.5]:
                name = f"combined_a{alpha}_d{dir_b}_t{tp_b}"
                result = {"recall@1": [], "recall@3": [], "recall@5": [], "recall@10": [], "recall@20": []}
                cond_correct = 0
                cond_total = 0
                for i in range(len(all_candidates)):
                    candidates = all_candidates[i]
                    neural_s = all_neural_scores[i]
                    bm25_s = all_bm25_scores[i]
                    gt_set = set(all_gt[i])

                    # Step 1: BM25 hybrid
                    ns = np.array(neural_s)
                    bs = np.array(bm25_s)
                    ns_norm = (ns - ns.min()) / (ns.max() - ns.min() + 1e-8)
                    bs_norm = (bs - bs.min()) / (bs.max() - bs.min() + 1e-8) if bs.max() > bs.min() else np.zeros_like(bs)
                    combined = alpha * ns_norm + (1 - alpha) * bs_norm

                    # Step 2: Directory boost
                    combined = directory_boost(candidates, combined.tolist(), boost_factor=dir_b)

                    # Step 3: Test-source pair boost
                    combined = test_pair_boost(candidates, combined, boost_factor=tp_b)

                    scored = sorted(zip(candidates, combined), key=lambda x: -x[1])
                    ranked = [f for f, _ in scored]

                    for k in [1, 3, 5, 10, 20]:
                        result[f"recall@{k}"].append(compute_recall_at_k(ranked, gt_set, k))

                    if gt_set & set(candidates):
                        cond_total += 1
                        if ranked[0] in gt_set:
                            cond_correct += 1

                strategies[name] = {
                    f"recall@{k}": np.mean(result[f"recall@{k}"]) * 100 for k in [1, 3, 5, 10, 20]
                }
                strategies[name]["cond_acc1"] = cond_correct / max(cond_total, 1) * 100
                strategies[name]["n"] = len(all_candidates)

    # Print results sorted by R@1
    print(f"\n{'Strategy':<35} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'R@20':>7} {'C.Acc@1':>8}")
    print("-" * 80)
    sorted_strategies = sorted(strategies.items(), key=lambda x: -x[1]["recall@1"])
    for name, m in sorted_strategies:
        print(f"{name:<35} {m['recall@1']:>7.2f} {m['recall@5']:>7.2f} "
              f"{m['recall@10']:>7.2f} {m['recall@20']:>7.2f} {m['cond_acc1']:>8.2f}")

    # Save
    with open(os.path.join(args.output_dir, "strategies.json"), "w") as f:
        json.dump(strategies, f, indent=2)

    print(f"\nBest strategy: {sorted_strategies[0][0]} with R@1={sorted_strategies[0][1]['recall@1']:.2f}%")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
