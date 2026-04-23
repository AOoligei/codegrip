#!/usr/bin/env python3
"""
Eval code-centric scorer with OBFUSCATED code content.

Purpose: Causal test — if replacing all user-defined identifiers with random
tokens (v_001, f_002, c_003) barely changes performance, the "code signal"
is identifier naming (which correlates with paths), not semantic understanding.

Comparison points:
  - Code-centric normal: 28.74% R@1
  - Path-only baseline:  29.27% R@1

This script is eval_rankft_4bit.py + code-centric prompt + obfuscated repo_dir.
"""
import ast
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

# ============================================================
# Code-centric prompt (same as training)
# ============================================================

def build_prompt(issue_text: str, candidate_path: str, code_content: str,
                 tokenizer=None, max_seq_length: int = 1024) -> str:
    """Build code-centric scoring prompt."""
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
# Code extraction (same as train_rankft_code_centric.py)
# ============================================================

def _extract_signatures(source: str) -> List[str]:
    sigs = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return sigs
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
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


def extract_code_content(repo_dir, repo, filepath, head_lines=50, max_chars=1500):
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
def score_candidates_batched(model, tokenizer, issue_text, candidates,
                              repo_dir, repo, yes_id, no_id,
                              max_seq_length, device, batch_size=2,
                              code_head_lines=50, code_max_chars=1500,
                              anonymize_paths=False):
    """Score candidates using code-centric prompt with (optionally obfuscated) code."""
    import hashlib
    prompts = []
    for cand in candidates:
        code_content = extract_code_content(
            repo_dir, repo, cand,
            head_lines=code_head_lines, max_chars=code_max_chars)
        if anonymize_paths:
            display_path = f"file_{hashlib.md5(cand.encode()).hexdigest()[:8]}.py"
        else:
            display_path = cand
        prompt = build_prompt(issue_text, display_path, code_content,
                              tokenizer=tokenizer, max_seq_length=max_seq_length)
        prompts.append(prompt)

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
    print(f"  Repo dir (obfuscated): {args.repo_dir}")

    # Count how many files are available
    available = 0
    missing = 0
    for item in test_data[:50]:
        repo = item["repo"]
        key = (repo, str(item["issue_id"]))
        if key in bm25_data:
            cands = bm25_data[key].get("candidates", bm25_data[key].get("bm25_candidates", []))
            for c in cands[:5]:
                path = os.path.join(args.repo_dir, repo, c)
                if os.path.exists(path):
                    available += 1
                else:
                    missing += 1
    print(f"  File availability check (sample): {available} found, {missing} missing")

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
            args.repo_dir, repo, yes_id, no_id,
            args.max_seq_length, device,
            batch_size=args.score_batch_size,
            code_head_lines=args.code_head_lines,
            code_max_chars=args.code_max_chars,
            anonymize_paths=args.anonymize_paths,
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
        raise RuntimeError("No predictions generated.")

    overall = {}
    for k in k_values:
        overall[f"hit@{k}"] = np.mean(hit_at_k[k]) * 100
        overall[f"recall@{k}"] = np.mean(recall_at_k[k]) * 100

    # Strict Acc@k
    strict_acc = {k: [] for k in k_values}
    for p in predictions:
        gt = set(p["ground_truth"])
        pred = p["predicted"]
        for k in k_values:
            strict_acc[k].append(1.0 if gt and gt <= set(pred[:k]) else 0.0)
    for k in k_values:
        overall[f"acc@{k}"] = np.mean(strict_acc[k]) * 100

    overall["cond_acc@1|gt_in_candidates"] = np.mean(cond_acc_at_1) * 100 if cond_acc_at_1 else 0

    # NDCG@k
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

    # Bootstrap CI
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

    # Save
    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    with open(pred_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    summary = {
        "overall": overall,
        "bootstrap_ci": bootstrap_ci,
        "config": {
            "model_path": args.model_path,
            "lora_path": args.lora_path,
            "repo_dir": args.repo_dir,
            "candidate_pool": candidate_pool,
            "candidate_path": candidate_path,
            "test_data": args.test_data,
            "quantization": "4bit-nf4",
            "top_k": args.top_k,
            "max_seq_length": args.max_seq_length,
            "code_head_lines": args.code_head_lines,
            "code_max_chars": args.code_max_chars,
            "total_examples": len(predictions),
            "experiment": "identifier_obfuscation",
        },
        "wall_clock_seconds": wall_clock,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate code-centric scorer with obfuscated code")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--bm25_candidates", default=None)
    parser.add_argument("--graph_candidates", default=None)
    parser.add_argument("--hybrid_candidates", default=None)
    parser.add_argument("--repo_dir", required=True,
                        help="Path to obfuscated repos (e.g. /data/.../repos_obfuscated)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--score_batch_size", type=int, default=2)
    parser.add_argument("--code_head_lines", type=int, default=50)
    parser.add_argument("--code_max_chars", type=int, default=1500)
    parser.add_argument("--anonymize_paths", action="store_true",
                        help="Replace real paths with hashed anonymized paths (file_XXXX.py)")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
