#!/usr/bin/env python3
"""
Symmetric corruption control: isolate path vs code contribution.

Runs the code-centric reranker (path + code prompt) under 3 conditions:
  1. true path + true code (baseline)
  2. true path + shuffled code (code corrupted)
  3. shuffled path + true code (path corrupted)

If condition 2 approx condition 1 and condition 3 collapses,
this directly proves path dominates code.

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/eval_symmetric_corruption.py \
        --gpu_id 0 --condition true_true \
        --output_dir /data/chenlibin/grepo_agent_experiments/symmetric_corruption/true_true
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
TEST_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_test.jsonl"
BM25_PATH = "/home/chenlibin/grepo_agent/data/rankft/merged_bm25_exp6_candidates.jsonl"
REPO_DIR = "/home/chenlibin/grepo_agent/data/repos"

CODE_PROMPT = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Code:\n{code_content}\n\n"
    "Answer:"
)


def read_code(repo, file_path, max_lines=50):
    full_path = os.path.join(REPO_DIR, repo, file_path)
    try:
        with open(full_path, "r", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip())
            return "\n".join(lines)
    except (FileNotFoundError, IsADirectoryError):
        return "# (file not available)"


def shuffle_filenames(paths):
    dir_to_files = defaultdict(list)
    for p in paths:
        parts = p.rsplit("/", 1)
        if len(parts) == 2:
            dir_to_files[parts[0]].append(parts[1])
        else:
            dir_to_files[""].append(parts[0])
    mapping = {}
    for dir_path, filenames in dir_to_files.items():
        shuffled = filenames.copy()
        random.shuffle(shuffled)
        for orig, new in zip(filenames, shuffled):
            orig_full = f"{dir_path}/{orig}" if dir_path else orig
            new_full = f"{dir_path}/{new}" if dir_path else new
            mapping[orig_full] = new_full
    return mapping


def load_model(model_path, lora_path, gpu_id):
    device = f"cuda:{gpu_id}"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config,
        device_map={"": device}, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    return model, tokenizer, yes_id, no_id, device


def score_batch(model, tokenizer, prompts, yes_id, no_id, max_len, device, bs=4):
    scores = []
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i+bs]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_len,
                           padding_side="left").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
        s = (logits[:, yes_id] - logits[:, no_id]).float().cpu().numpy()
        scores.extend(s.tolist())
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--lora_path", type=str,
                        default="/home/chenlibin/grepo_agent/experiments/rankft_code_centric/best")
    parser.add_argument("--condition", choices=["true_true", "true_shuffled_code", "shuffled_path_true"],
                        required=True)
    parser.add_argument("--max_seq_length", type=int, default=768)
    parser.add_argument("--code_max_lines", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    test_data = {}
    with open(TEST_PATH) as f:
        for line in f:
            rec = json.loads(line)
            test_data[(rec["repo"], str(rec["issue_id"]))] = rec

    bm25_data = {}
    with open(BM25_PATH) as f:
        for line in f:
            rec = json.loads(line)
            bm25_data[(rec["repo"], str(rec["issue_id"]))] = rec

    print(f"Loading model (LoRA: {args.lora_path})...")
    model, tokenizer, yes_id, no_id, device = load_model(
        MODEL_PATH, args.lora_path, args.gpu_id)

    max_issue_tokens = args.max_seq_length - 200
    hits = []
    start_time = time.time()

    for idx, (key, test_rec) in enumerate(test_data.items()):
        if key not in bm25_data:
            continue
        repo = test_rec["repo"]
        issue_text = test_rec["issue_text"]
        gt_files = set(test_rec.get("changed_py_files",
                                     test_rec.get("changed_files", [])))
        if not gt_files:
            continue

        candidates = bm25_data[key].get("candidates",
                                         bm25_data[key].get("bm25_candidates", []))[:args.top_k]

        issue_ids = tokenizer.encode(issue_text, add_special_tokens=False)
        if len(issue_ids) > max_issue_tokens:
            issue_text_trunc = tokenizer.decode(issue_ids[:max_issue_tokens],
                                                 skip_special_tokens=True)
        else:
            issue_text_trunc = issue_text

        if args.condition == "true_true":
            display_paths = candidates
            codes = [read_code(repo, c, args.code_max_lines) for c in candidates]

        elif args.condition == "true_shuffled_code":
            display_paths = candidates
            # Shuffle code: each file gets another file's code
            codes = [read_code(repo, c, args.code_max_lines) for c in candidates]
            shuffled_codes = codes.copy()
            random.shuffle(shuffled_codes)
            codes = shuffled_codes

        elif args.condition == "shuffled_path_true":
            mapping = shuffle_filenames(candidates)
            display_paths = [mapping.get(c, c) for c in candidates]
            codes = [read_code(repo, c, args.code_max_lines) for c in candidates]
            gt_files = {mapping.get(g, g) for g in gt_files}

        prompts = []
        for path, code in zip(display_paths, codes):
            prompt = CODE_PROMPT.format(
                issue_text=issue_text_trunc,
                candidate_path=path,
                code_content=code[:600],
            )
            prompts.append(prompt)

        scores = score_batch(model, tokenizer, prompts, yes_id, no_id,
                             args.max_seq_length, device, args.batch_size)

        ranked = sorted(zip(display_paths, scores), key=lambda x: -x[1])
        top1 = ranked[0][0]
        hit = len({top1} & gt_files) / max(1, len(gt_files))
        hits.append(hit)

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  [{idx+1}] R@1={np.mean(hits)*100:.2f}% ({elapsed:.0f}s)")

    r1 = float(np.mean(hits) * 100)
    summary = {
        "condition": args.condition,
        "R@1": r1,
        "num_examples": len(hits),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {args.condition}: R@1={r1:.2f}% (n={len(hits)}) ===")


if __name__ == "__main__":
    main()
