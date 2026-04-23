"""
Evaluate pairwise code-contrastive reranker.

Inference strategy: For each candidate file, compute pairwise win rate
against K random opponents from the same candidate pool. Rank by win rate.

This produces a code-based ranking WITHOUT any path information.

Usage:
    python scripts/eval_pairwise_code.py \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path experiments/pairwise_code/best \
        --test_data data/grepo_text/grepo_test.jsonl \
        --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
        --output_dir experiments/pairwise_code/eval_graph \
        --gpu_id 0 --num_opponents 5 --top_k 30
"""

import os
import json
import argparse
import random
import time
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

PAIRWISE_PROMPT = (
    "Given the bug report, which file is more likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File A:\n{code_a}\n\n"
    "File B:\n{code_b}\n\n"
    "Answer: File"
)

_REPO_BASE_DIR = "data/repos"


def _find_repo_dir(repo_name):
    candidates = [
        os.path.join(_REPO_BASE_DIR, repo_name),
        os.path.join(_REPO_BASE_DIR, repo_name.replace('/', '__')),
        os.path.join(_REPO_BASE_DIR, repo_name.replace('/', '_')),
        os.path.join(_REPO_BASE_DIR, repo_name.split('/')[-1]),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return ""


def read_file_content(repo_dir, file_path, max_lines=50):
    full_path = os.path.join(repo_dir, file_path)
    try:
        with open(full_path, 'r', errors='ignore') as f:
            lines = f.readlines()[:max_lines]
        return ''.join(lines)[:2000]
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return '# (file content unavailable)'


def truncate_code_to_fit(issue_text, code_a, code_b, tokenizer, max_seq_length):
    """Truncate code blocks to ensure 'Answer: File' survives tokenization."""
    frame_and_issue = tokenizer(
        f"Given the bug report, which file is more likely to need modification?\n\nBug Report: {issue_text[:800]}\n\nFile A:\n\n\nFile B:\n\n\nAnswer: File",
        add_special_tokens=True
    )
    frame_tokens = len(frame_and_issue['input_ids'])
    budget_per_code = max(50, (max_seq_length - frame_tokens) // 2)
    code_a_tok = tokenizer.encode(code_a, add_special_tokens=False)[:budget_per_code]
    code_b_tok = tokenizer.encode(code_b, add_special_tokens=False)[:budget_per_code]
    return tokenizer.decode(code_a_tok), tokenizer.decode(code_b_tok)


def pairwise_score(model, tokenizer, issue_text, code_a, code_b, a_id, b_id, device, max_seq_length):
    """Return P(A wins) from the pairwise comparison."""
    code_a_trunc, code_b_trunc = truncate_code_to_fit(issue_text, code_a, code_b, tokenizer, max_seq_length)
    prompt = PAIRWISE_PROMPT.format(
        issue_text=issue_text[:800],
        code_a=code_a_trunc,
        code_b=code_b_trunc,
    )
    encoding = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=False,
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits[0, -1]
        probs = torch.softmax(logits[torch.tensor([a_id, b_id])], dim=0)
        return probs[0].item()  # P(A)


def compute_hit_at_k(predicted, gt, k):
    if not gt:
        return 0.0
    top_k = set(predicted[:k])
    return len(top_k & gt) / len(gt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--bm25_candidates", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_opponents", type=int, default=5,
                        help="Number of random opponents per candidate")
    parser.add_argument("--top_k", type=int, default=30,
                        help="Only rank top-K BM25 candidates (for speed)")
    parser.add_argument("--max_seq_length", type=int, default=768)
    parser.add_argument("--code_max_lines", type=int, default=50)
    parser.add_argument("--repo_dir", default="data/repos")
    args = parser.parse_args()

    global _REPO_BASE_DIR
    _REPO_BASE_DIR = args.repo_dir

    device = f"cuda:{args.gpu_id}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model in 4-bit
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb_config,
        device_map=device, trust_remote_code=True,
    )
    print(f"Loading LoRA from {args.lora_path}...")
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    # Get A/B token IDs — match the exact continuation after "Answer: File"
    a_full = tokenizer.encode("Answer: File A", add_special_tokens=False)
    b_full = tokenizer.encode("Answer: File B", add_special_tokens=False)
    a_id = a_full[-1]
    b_id = b_full[-1]
    assert a_id != b_id, f"A and B tokens are the same: {a_id}"
    print(f"Token IDs: A={a_id} ('{tokenizer.decode([a_id])}'), B={b_id} ('{tokenizer.decode([b_id])}')")

    # Load data
    print("Loading data...")
    test_data = [json.loads(l) for l in open(args.test_data)]
    bm25_index = {}
    for l in open(args.bm25_candidates):
        d = json.loads(l)
        key = (d['repo'], d['issue_id'])
        bm25_index[key] = d.get('candidates', d.get('bm25_candidates', []))

    # Evaluate
    print(f"Evaluating {len(test_data)} examples (top-{args.top_k}, {args.num_opponents} opponents)...")
    hit_at_k = {1: [], 3: [], 5: [], 10: []}
    predictions = []
    t0 = time.time()

    for idx, ex in enumerate(test_data):
        key = (ex['repo'], ex['issue_id'])
        if key not in bm25_index:
            continue
        gt_files = set(ex.get('changed_py_files', []))
        if not gt_files:
            continue

        candidates = bm25_index[key][:args.top_k]
        if len(candidates) < 2:
            continue

        repo_dir = _find_repo_dir(ex['repo'])
        if not repo_dir:
            continue

        # Read code for all candidates
        code_cache = {}
        for c in candidates:
            code_cache[c] = read_file_content(repo_dir, c, args.code_max_lines)

        # Compute pairwise win rates
        win_counts = defaultdict(float)
        match_counts = defaultdict(int)

        for c in candidates:
            opponents = random.sample(
                [o for o in candidates if o != c],
                min(args.num_opponents, len(candidates) - 1)
            )
            for opp in opponents:
                p_a = pairwise_score(
                    model, tokenizer, ex['issue_text'],
                    code_cache[c], code_cache[opp],
                    a_id, b_id, device, args.max_seq_length
                )
                win_counts[c] += p_a
                win_counts[opp] += (1.0 - p_a)
                match_counts[c] += 1
                match_counts[opp] += 1

        # Rank by average win rate
        scores = {}
        for c in candidates:
            if match_counts[c] > 0:
                scores[c] = win_counts[c] / match_counts[c]
            else:
                scores[c] = 0.0

        ranked = sorted(candidates, key=lambda c: -scores.get(c, 0))

        # Compute metrics
        for k in hit_at_k:
            h = compute_hit_at_k(ranked, gt_files, k)
            hit_at_k[k].append(h)

        predictions.append({
            'repo': ex['repo'],
            'issue_id': ex['issue_id'],
            'predicted': ranked[:20],
            'scores': [scores.get(c, 0) for c in ranked[:20]],
            'ground_truth': list(gt_files),
        })

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            r1 = np.mean(hit_at_k[1]) * 100
            print(f"  [{idx+1}/{len(test_data)}] R@1={r1:.2f}% ({elapsed:.0f}s)", flush=True)

    # Summary
    overall = {}
    for k in hit_at_k:
        overall[f"hit@{k}"] = float(np.mean(hit_at_k[k]) * 100)

    wall_clock = time.time() - t0

    print(f"\n=== Pairwise Code-Contrastive Reranker ===")
    print(f"Examples: {len(predictions)}, Wall clock: {wall_clock:.0f}s")
    for k in [1, 3, 5, 10]:
        print(f"  R@{k} = {overall[f'hit@{k}']:.2f}%")

    # Save
    summary = {"overall": overall, "wall_clock_seconds": wall_clock,
               "config": vars(args)}
    with open(os.path.join(args.output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "predictions.jsonl"), 'w') as f:
        for p in predictions:
            f.write(json.dumps(p) + '\n')
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
