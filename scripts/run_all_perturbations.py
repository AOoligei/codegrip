#!/usr/bin/env python3
"""
Run all path perturbation evaluations with a SINGLE model load.
Uses 4-bit quantization to fit on a GPU with ~9GB free.
Unbuffered output for real-time logging.
"""
import os
import sys
import json
import time
from collections import defaultdict
from typing import List, Set

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

torch.manual_seed(42)
np.random.seed(42)

GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 7
MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
LORA_PATH = "experiments/rankft_runB_graph/best"
BASE_DIR = "/home/chenlibin/grepo_agent"

CONDITIONS = [
    "shuffle_dirs",
    "shuffle_filenames",
    "remove_module_names",
    "flatten_dirs",
    "swap_leaf_dirs",
]

PROMPT_TEMPLATE = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)


@torch.no_grad()
def score_batch(model, tokenizer, issue_text, candidates, yes_id, no_id, device, max_len=512):
    """Score candidates, auto-fallback to batch=1 on OOM."""
    prompts = [PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=c) for c in candidates]
    all_scores = []

    # Process one-by-one to avoid OOM on shared GPU
    for p in prompts:
        e = tokenizer([p], return_tensors="pt", truncation=True, max_length=max_len)
        o = model(input_ids=e["input_ids"].to(device), attention_mask=e["attention_mask"].to(device))
        s = (o.logits[0, -1, yes_id] - o.logits[0, -1, no_id]).item()
        all_scores.append(s)
        del o, e
    return all_scores


def evaluate_condition(model, tokenizer, yes_id, no_id, device, cond_name):
    """Evaluate one perturbation condition."""
    test_path = os.path.join(BASE_DIR, f"experiments/path_perturb_{cond_name}/test.jsonl")
    cand_path = os.path.join(BASE_DIR, f"experiments/path_perturb_{cond_name}/bm25_candidates.jsonl")
    out_dir = os.path.join(BASE_DIR, f"experiments/path_perturb_{cond_name}/eval_4bit")

    if not os.path.exists(test_path):
        print(f"  SKIP {cond_name}: no data")
        return None

    if os.path.exists(os.path.join(out_dir, "summary.json")):
        d = json.load(open(os.path.join(out_dir, "summary.json")))["overall"]
        print(f"  SKIP {cond_name}: already done (R@1={d['recall@1']:.2f})")
        return d

    os.makedirs(out_dir, exist_ok=True)

    # Load data
    with open(test_path) as f:
        test_data = [json.loads(l) for l in f]
    cand_data = {}
    with open(cand_path) as f:
        for l in f:
            item = json.loads(l)
            cand_data[(item["repo"], str(item["issue_id"]))] = item

    print(f"  {cond_name}: {len(test_data)} test, {len(cand_data)} candidates")

    # Evaluate
    predictions = []
    k_values = [1, 3, 5, 10, 20]
    recall_at_k = defaultdict(list)
    cond_acc = []
    t0 = time.time()

    for idx, item in enumerate(test_data):
        key = (item["repo"], str(item["issue_id"]))
        if key not in cand_data:
            continue

        gt = set(item.get("changed_py_files", []))
        candidates = cand_data[key].get("candidates", cand_data[key].get("bm25_candidates", []))[:200]
        issue_text = item.get("issue_text", item.get("text", ""))

        if not gt or not candidates:
            continue

        scores = score_batch(model, tokenizer, issue_text, candidates, yes_id, no_id, device)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        predicted = [c for c, _ in ranked]

        gt_in = bool(gt & set(candidates))
        for k in k_values:
            top_k = set(predicted[:k])
            recall_at_k[k].append(len(top_k & gt) / len(gt))
        if gt_in:
            cond_acc.append(1.0 if predicted[0] in gt else 0.0)

        predictions.append({
            "repo": item["repo"], "issue_id": item["issue_id"],
            "ground_truth": list(gt), "predicted": predicted[:50],
            "gt_in_candidates": gt_in, "num_candidates": len(candidates),
        })

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            r1 = np.mean(recall_at_k[1]) * 100
            speed = (idx + 1) / elapsed
            eta = (len(test_data) - idx - 1) / speed / 60
            print(f"    [{idx+1}/{len(test_data)}] R@1={r1:.2f}% ({elapsed:.0f}s, ETA {eta:.0f}min)")

    elapsed = time.time() - t0

    overall = {}
    for k in k_values:
        overall[f"recall@{k}"] = np.mean(recall_at_k[k]) * 100
    overall["cond_acc@1"] = np.mean(cond_acc) * 100 if cond_acc else 0

    print(f"  {cond_name} DONE: R@1={overall['recall@1']:.2f}%, R@5={overall['recall@5']:.2f}% ({elapsed:.0f}s, {len(predictions)} examples)")

    # Save
    with open(os.path.join(out_dir, "predictions.jsonl"), "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    summary = {"overall": overall, "config": {"condition": cond_name, "quantization": "4bit-nf4",
                "total_examples": len(predictions)}, "wall_clock_seconds": elapsed}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return overall


def main():
    device = f"cuda:{GPU_ID}"
    print(f"=== Path Perturbation Eval (4-bit, GPU {GPU_ID}) ===")
    print(f"Loading model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config, device_map=device, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, os.path.join(BASE_DIR, LORA_PATH))
    model.eval()

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    print(f"Model loaded. GPU mem: {torch.cuda.memory_allocated(GPU_ID)/1024**3:.1f}GB")

    results = {}
    for cond in CONDITIONS:
        print(f"\n>>> {cond}")
        r = evaluate_condition(model, tokenizer, yes_id, no_id, device, cond)
        if r:
            results[cond] = r

    print(f"\n{'='*60}")
    print(f"{'Condition':<25} {'R@1':>8} {'R@5':>8} {'Cond Acc@1':>10}")
    print(f"{'-'*55}")
    print(f"{'Normal (baseline)':<25} {'27.01':>8} {'49.17':>8} {'51.47':>10}")
    for cond, r in results.items():
        print(f"{cond:<25} {r['recall@1']:>8.2f} {r['recall@5']:>8.2f} {r.get('cond_acc@1', 0):>10.2f}")
    print()


if __name__ == "__main__":
    main()
