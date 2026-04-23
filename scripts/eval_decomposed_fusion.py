"""
Decomposed Reranker Evaluation: Path Prior + Code Residual Fusion
Scores each candidate with both models and combines:
  s_total = s_path + alpha * s_code

Usage:
    python scripts/eval_decomposed_fusion.py \
        --path_model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --path_lora experiments/rankft_runB_graph_v2/best \
        --code_model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --code_lora experiments/code_residual_v1/best \
        --test_data data/grepo_text/grepo_test.jsonl \
        --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
        --repo_dir data/repos \
        --output_dir experiments/decomposed_fusion \
        --gpu_id 0 --alpha 1.0
"""

import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# --- Prompt templates ---
PATH_PROMPT = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)

CODE_PROMPT = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Code:\n{code_content}\n\n"
    "Answer:"
)


def read_file_content(repo_dir, file_path, max_lines=100):
    full_path = os.path.join(repo_dir, file_path)
    try:
        with open(full_path, 'r', errors='ignore') as f:
            return ''.join(f.readlines()[:max_lines])
    except:
        return '# (unavailable)'


def find_repo_dir(repo_name, repo_base="data/repos"):
    for suffix in [repo_name, repo_name.replace('/', '__'),
                   repo_name.replace('/', '_'), repo_name.split('/')[-1]]:
        p = os.path.join(repo_base, suffix)
        if os.path.isdir(p):
            return p
    return ""


def load_model(model_path, lora_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map=device, trust_remote_code=True
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)

    return model, tokenizer, yes_ids[0], no_ids[0]


@torch.no_grad()
def score_batch(model, tokenizer, prompts, yes_id, no_id, max_length, device):
    scores = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length,
                          truncation=True).to(device)
        outputs = model(**inputs)
        logits = outputs.logits[0, -1]
        score = logits[yes_id].item() - logits[no_id].item()
        scores.append(score)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_model_path", required=True)
    parser.add_argument("--path_lora", required=True)
    parser.add_argument("--code_model_path", required=True)
    parser.add_argument("--code_lora", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--bm25_candidates", required=True)
    parser.add_argument("--repo_dir", default="data/repos")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for code score: s = s_path + alpha * s_code")
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--code_max_lines", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    # Load test data
    test_data = [json.loads(l) for l in open(args.test_data)]
    bm25_data = {}
    for line in open(args.bm25_candidates):
        d = json.loads(line)
        key = (d['repo'], str(d['issue_id']))
        bm25_data[key] = d

    print(f"Test: {len(test_data)}, BM25: {len(bm25_data)}")

    # Load path model
    print("Loading path model...")
    path_model, path_tok, path_yes, path_no = load_model(
        args.path_model_path, args.path_lora, device)

    # Evaluate with path model first, then free memory
    print("Scoring with path model...")
    path_scores_all = {}
    start = time.time()

    for idx, item in enumerate(test_data):
        key = (item['repo'], str(item['issue_id']))
        if key not in bm25_data:
            continue
        candidates = bm25_data[key].get('candidates', bm25_data[key].get('bm25_candidates', []))[:args.top_k]
        issue_text = item['issue_text'][:500]

        prompts = [PATH_PROMPT.format(issue_text=issue_text, candidate_path=c) for c in candidates]
        scores = score_batch(path_model, path_tok, prompts, path_yes, path_no, 512, device)
        path_scores_all[key] = dict(zip(candidates, scores))

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(test_data)}] path scoring ({time.time()-start:.0f}s)")

    # Free path model
    del path_model, path_tok
    torch.cuda.empty_cache()

    # Load code model
    print("Loading code model...")
    code_model, code_tok, code_yes, code_no = load_model(
        args.code_model_path, args.code_lora, device)

    # Score with code model
    print("Scoring with code model...")
    predictions = []
    k_values = [1, 3, 5, 10, 20]
    hit_at_k = defaultdict(list)

    for idx, item in enumerate(test_data):
        key = (item['repo'], str(item['issue_id']))
        if key not in bm25_data or key not in path_scores_all:
            continue

        candidates = list(path_scores_all[key].keys())
        gt_files = set(item.get('changed_py_files', []))
        issue_text = item['issue_text'][:500]
        repo_dir = find_repo_dir(item['repo'], args.repo_dir)

        # Build shuffled anonymization map for this example.
        # Uses stable seed from (repo, issue_id) via hashlib (not hash(),
        # which is salted per-process and non-reproducible across runs).
        shuffled = list(candidates)
        example_seed = int.from_bytes(
            hashlib.blake2s(f"{item['repo']}\0{item['issue_id']}".encode(), digest_size=4).digest(),
            "big",
        )
        random.Random(example_seed).shuffle(shuffled)
        anon_ids = {path: f"file_{j % 10000:04d}.py" for j, path in enumerate(shuffled)}

        # Code prompts with anonymized paths
        code_prompts = []
        for i, c in enumerate(candidates):
            code = read_file_content(repo_dir, c, args.code_max_lines) if repo_dir else '# (unavailable)'
            prompt = CODE_PROMPT.format(
                issue_text=issue_text,
                candidate_path=anon_ids[c],
                code_content=code[:2000],
            )
            code_prompts.append(prompt)

        code_scores = score_batch(code_model, code_tok, code_prompts, code_yes, code_no,
                                  args.max_seq_length, device)

        # Fusion: s_total = s_path + alpha * s_code
        fused = []
        for c, cs in zip(candidates, code_scores):
            ps = path_scores_all[key][c]
            fused.append((c, ps + args.alpha * cs, ps, cs))

        fused.sort(key=lambda x: x[1], reverse=True)
        predicted = [c for c, _, _, _ in fused]

        # Metrics
        for k in k_values:
            top_k_set = set(predicted[:k])
            hit = len(top_k_set & gt_files) / len(gt_files) if gt_files else 0
            hit_at_k[k].append(hit)

        predictions.append({
            'repo': item['repo'],
            'issue_id': str(item['issue_id']),
            'ground_truth': list(gt_files),
            'predicted': predicted[:20],
            'path_score_top1': fused[0][2],
            'code_score_top1': fused[0][3],
            'fused_score_top1': fused[0][1],
        })

        if (idx + 1) % 100 == 0:
            r1 = np.mean(hit_at_k[1]) * 100
            print(f"  [{idx+1}/{len(test_data)}] R@1={r1:.2f}% ({time.time()-start:.0f}s)")

    # Summary
    overall = {}
    for k in k_values:
        overall[f'hit@{k}'] = np.mean(hit_at_k[k]) * 100

    print(f"\n=== Decomposed Fusion Results (alpha={args.alpha}) ===")
    for k in k_values:
        print(f"  R@{k}: {overall[f'hit@{k}']:.2f}%")

    # Save
    with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as f:
        for p in predictions:
            f.write(json.dumps(p) + '\n')

    summary = {'overall': overall, 'alpha': args.alpha, 'n_examples': len(predictions)}
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved to {args.output_dir}")


if __name__ == '__main__':
    main()
