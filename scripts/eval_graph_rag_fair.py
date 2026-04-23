"""
Fair Graph-RAG comparison: trained model WITH vs WITHOUT graph context.

Tests whether adding graph edges to the prompt at inference helps a model
that has already internalized graph structure via training.

Three conditions:
1. Trained model + path only (our method, baseline)
2. Trained model + path + graph edges (Graph-RAG with trained model)
3. Trained model + path + file summary (content-aware at inference)

This is the proper ablation: same trained model, vary inference context.
"""

import os
import json
import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# Prompt templates for the three conditions
# ============================================================

PROMPT_PATH_ONLY = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)

PROMPT_WITH_GRAPH = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n"
    "Co-changed files: {cochange_neighbors}\n"
    "Import-related files: {import_neighbors}\n\n"
    "Answer:"
)

PROMPT_WITH_SUMMARY = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n"
    "Content: {file_summary}\n\n"
    "Answer:"
)


def build_prompt(issue_text, candidate_path, condition,
                 cochange_neighbors=None, import_neighbors=None,
                 file_summary=None, max_neighbors=10):
    if condition == "path_only":
        return PROMPT_PATH_ONLY.format(
            issue_text=issue_text, candidate_path=candidate_path
        )
    elif condition in ("graph_rag", "shuffled_graph"):
        cc = ", ".join((cochange_neighbors or [])[:max_neighbors]) or "none"
        imp = ", ".join((import_neighbors or [])[:max_neighbors]) or "none"
        return PROMPT_WITH_GRAPH.format(
            issue_text=issue_text, candidate_path=candidate_path,
            cochange_neighbors=cc, import_neighbors=imp
        )
    elif condition == "content":
        summary = file_summary or "N/A"
        return PROMPT_WITH_SUMMARY.format(
            issue_text=issue_text, candidate_path=candidate_path,
            file_summary=summary
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")


# ============================================================
# Graph loading (same as before)
# ============================================================

def load_dep_graphs(dep_graph_dir):
    import_graph = {}
    rels_dir = Path(dep_graph_dir)
    for fpath in sorted(rels_dir.glob("*_rels.json")):
        with open(fpath) as f:
            data = json.load(f)
        repo = data.get("repo", fpath.stem.replace("_rels", ""))
        fi = data.get("file_imports", {})
        graph = defaultdict(set)
        for src, targets in fi.items():
            for tgt in targets:
                graph[src].add(tgt)
                graph[tgt].add(src)
        import_graph[repo] = {k: sorted(v) for k, v in graph.items()}
    return import_graph


def build_cochange_index(train_data_path):
    repo_cochanges = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    with open(train_data_path) as f:
        for line in f:
            item = json.loads(line)
            if item.get("split") != "train":
                continue
            repo = item["repo"]
            files = item.get("changed_py_files", [])
            for i, fa in enumerate(files):
                for fb in files[i + 1:]:
                    repo_cochanges[repo][fa][fb] += 1
                    repo_cochanges[repo][fb][fa] += 1
    return dict(repo_cochanges)


def load_file_summaries(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# Scoring
# ============================================================

def score_batch(model, tokenizer, prompts, yes_id, no_id, max_length, device):
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(logits.size(0), device=device)
    last_logits = logits[batch_idx, seq_lengths]

    scores = (last_logits[:, yes_id] - last_logits[:, no_id]).cpu().float().tolist()
    del outputs, logits, input_ids, attention_mask
    torch.cuda.empty_cache()
    return scores


def recall_at_k(predicted, gt, k):
    gt_set = set(gt)
    found = sum(1 for p in predicted[:k] if p in gt_set)
    return found / len(gt_set)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data/shuyang/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_path", default="experiments/rankft_runB_graph/best")
    parser.add_argument("--test_data", default="data/grepo_text/grepo_test.jsonl")
    parser.add_argument("--candidates", default="data/rankft/merged_bm25_exp6_candidates.jsonl")
    parser.add_argument("--dep_graph_dir", default="data/dep_graphs")
    parser.add_argument("--train_data", default="data/grepo_text/grepo_train.jsonl")
    parser.add_argument("--file_summaries", default="data/file_summaries/file_summaries_all.json")
    parser.add_argument("--output_dir", default="experiments/graph_rag_fair")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_seq_length", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_examples", type=int, default=200,
                        help="Max examples to evaluate (for speed). -1 for all.")
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    print("Loading test data...")
    test_data = []
    with open(args.test_data) as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"  {len(test_data)} test examples")

    # Load candidates
    print("Loading candidates...")
    cand_data = {}
    with open(args.candidates) as f:
        for line in f:
            item = json.loads(line)
            key = (item["repo"], item.get("issue_id", ""))
            cand_data[key] = item
    print(f"  Candidates for {len(cand_data)} examples")

    # Load graphs
    print("Loading graphs...")
    import_graph = load_dep_graphs(args.dep_graph_dir)
    cochange_graph = build_cochange_index(args.train_data)
    print(f"  Import: {len(import_graph)} repos, Co-change: {len(cochange_graph)} repos")

    # Load file summaries
    print("Loading file summaries...")
    file_summaries = load_file_summaries(args.file_summaries)
    print(f"  Summaries for {len(file_summaries)} repos")

    # Load model
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    yes_id, no_id = yes_ids[0], no_ids[0]
    print(f"  Yes={yes_id}, No={no_id}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True,
    )
    print(f"Loading LoRA from {args.lora_path}...")
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()
    print("  Model loaded")

    # Match test data to candidates
    examples = []
    for td in test_data:
        key = (td["repo"], td.get("issue_id", ""))
        if key in cand_data:
            cd = cand_data[key]
            candidates = cd.get("candidates", cd.get("candidate_files", []))
            if not candidates:
                continue
            if isinstance(candidates[0], dict):
                candidates = [c["file"] for c in candidates]
            examples.append({
                "repo": td["repo"],
                "issue_id": td.get("issue_id", ""),
                "issue_text": td.get("issue_text", ""),
                "ground_truth": td.get("changed_py_files", []),
                "candidates": candidates,
            })

    if args.max_examples > 0:
        examples = examples[:args.max_examples]
    print(f"  Evaluating {len(examples)} examples")

    # Evaluate four conditions (added shuffled_graph as control)
    conditions = ["path_only", "graph_rag", "shuffled_graph", "content"]
    results = {c: [] for c in conditions}

    # Build per-repo file lists for shuffled graph control
    repo_all_files = defaultdict(list)
    for ex in examples:
        repo = ex["repo"]
        for c in ex["candidates"]:
            repo_all_files[repo].append(c)
    repo_all_files = {r: sorted(set(fs)) for r, fs in repo_all_files.items()}

    start_time = time.time()
    for ex_idx, ex in enumerate(examples):
        repo = ex["repo"]
        issue_text = ex["issue_text"]
        candidates = ex["candidates"]
        gt = ex["ground_truth"]

        # Get graph info for this repo
        repo_cc = cochange_graph.get(repo, {})
        repo_imp = import_graph.get(repo, {})
        repo_summaries = file_summaries.get(repo, {})
        all_files = repo_all_files.get(repo, candidates)

        for condition in conditions:
            # Build prompts
            prompts = []
            for cand in candidates:
                if condition == "shuffled_graph":
                    # Shuffled control: random files instead of real neighbors
                    # Preserve same count as real neighbors would have
                    n_cc = min(10, len(repo_cc.get(cand, {})))
                    n_imp = min(10, len(repo_imp.get(cand, [])))
                    others = [f for f in all_files if f != cand]
                    cc_files = random.sample(others, min(max(n_cc, 3), len(others)))
                    imp_files = random.sample(others, min(max(n_imp, 3), len(others)))
                else:
                    cc_neighbors = sorted(
                        repo_cc.get(cand, {}).items(),
                        key=lambda x: -x[1]
                    )[:10]
                    cc_files = [f for f, _ in cc_neighbors]
                    imp_files = repo_imp.get(cand, [])[:10]

                summary = repo_summaries.get(cand, "")

                prompt = build_prompt(
                    issue_text, cand, condition,
                    cochange_neighbors=cc_files,
                    import_neighbors=imp_files,
                    file_summary=summary,
                )
                prompts.append(prompt)

            # Score all candidates
            all_scores = []
            for i in range(0, len(prompts), args.batch_size):
                batch = prompts[i:i + args.batch_size]
                scores = score_batch(
                    model, tokenizer, batch,
                    yes_id, no_id, args.max_seq_length, device
                )
                all_scores.extend(scores)

            # Rank by score
            scored = list(zip(candidates, all_scores))
            scored.sort(key=lambda x: -x[1])
            ranked = [f for f, _ in scored]

            # Compute metrics
            r1 = recall_at_k(ranked, gt, 1)
            r5 = recall_at_k(ranked, gt, 5)
            r10 = recall_at_k(ranked, gt, 10)
            results[condition].append({
                "r1": r1, "r5": r5, "r10": r10,
                "gt_in_pool": any(f in set(candidates) for f in gt),
            })

        if (ex_idx + 1) % 20 == 0 or ex_idx == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (ex_idx + 1) * (len(examples) - ex_idx - 1)
            # Quick summary
            summaries = []
            for c in conditions:
                r1_avg = np.mean([r["r1"] for r in results[c]]) * 100
                summaries.append(f"{c}={r1_avg:.1f}%")
            print(f"  [{ex_idx+1}/{len(examples)}] R@1: {', '.join(summaries)} | ETA: {eta:.0f}s")

    # ============================================================
    # Report
    # ============================================================
    print("\n" + "=" * 70)
    print("FAIR GRAPH-RAG COMPARISON (trained model, same candidates)")
    print("=" * 70)
    print(f"Examples: {len(examples)}")
    print(f"Model: {args.lora_path}")
    print()

    print(f"{'Condition':>15} | {'R@1':>8} | {'R@5':>8} | {'R@10':>8}")
    print("-" * 50)
    for c in conditions:
        r1 = np.mean([r["r1"] for r in results[c]]) * 100
        r5 = np.mean([r["r5"] for r in results[c]]) * 100
        r10 = np.mean([r["r10"] for r in results[c]]) * 100
        print(f"{c:>15} | {r1:>7.2f}% | {r5:>7.2f}% | {r10:>7.2f}%")

    # Conditional (GT in pool)
    print(f"\nConditional R@1 (GT in pool):")
    for c in conditions:
        cond = [r for r in results[c] if r["gt_in_pool"]]
        if cond:
            r1 = np.mean([r["r1"] for r in cond]) * 100
            print(f"  {c}: {r1:.2f}% (n={len(cond)})")

    # Save summary
    summary = {}
    for c in conditions:
        summary[c] = {
            "recall@1": np.mean([r["r1"] for r in results[c]]) * 100,
            "recall@5": np.mean([r["r5"] for r in results[c]]) * 100,
            "recall@10": np.mean([r["r10"] for r in results[c]]) * 100,
        }
    with open(os.path.join(args.output_dir, "fair_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-example results for bootstrap analysis
    per_example = []
    for i in range(len(examples)):
        row = {"repo": examples[i]["repo"], "issue_id": examples[i].get("issue_id", "")}
        for c in conditions:
            row[f"{c}_r1"] = results[c][i]["r1"]
            row[f"{c}_r5"] = results[c][i]["r5"]
            row[f"{c}_r10"] = results[c][i]["r10"]
        per_example.append(row)
    with open(os.path.join(args.output_dir, "per_example.jsonl"), "w") as f:
        for row in per_example:
            f.write(json.dumps(row) + "\n")

    print(f"\nSaved to {args.output_dir}/fair_comparison.json")
    print(f"Per-example data: {args.output_dir}/per_example.jsonl")


if __name__ == "__main__":
    main()
