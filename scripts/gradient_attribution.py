#!/usr/bin/env python3
"""
Gradient attribution analysis: measure how much the reranker attends to
path tokens vs code/issue tokens.

For each test example, computes input-gradient attribution (|grad * embedding|)
and partitions total attribution mass into:
  - path_tokens: file path components
  - issue_tokens: bug report text
  - template_tokens: prompt template / separators

Reports mean attribution fraction across examples.

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/gradient_attribution.py \
        --gpu_id 0 --max_examples 300 --output_dir /data/chenlibin/grepo_agent_experiments/gradient_attribution
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Paths
MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
LORA_PATH = "/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/best"
TEST_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_test.jsonl"
BM25_PATH = "/home/chenlibin/grepo_agent/data/rankft/merged_bm25_exp6_candidates.jsonl"


def load_data(max_examples: int = 300):
    """Load and merge test + candidate data."""
    test_data = {}
    with open(TEST_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], rec["issue_id"])
            test_data[key] = rec

    merged = []
    with open(BM25_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], rec["issue_id"])
            if key in test_data:
                test_rec = test_data[key]
                gt_files = set(test_rec.get("changed_py_files",
                                            test_rec.get("changed_files", [])))
                candidates = rec.get("bm25_candidates", rec.get("candidates", []))[:200]
                gt_in_pool = [c for c in candidates if c in gt_files]
                non_gt = [c for c in candidates if c not in gt_files]
                if gt_in_pool and non_gt:
                    merged.append({
                        "issue_text": test_rec["issue_text"],
                        "gt_file": gt_in_pool[0],
                        "wrong_file": non_gt[0],
                        "repo": test_rec["repo"],
                    })
            if len(merged) >= max_examples:
                break

    return merged


def build_prompt(issue_text: str, file_path: str) -> str:
    """Build the reranker prompt (must match eval_rankft_4bit.py exactly)."""
    return (
        f"Given the bug report, is this file likely to need modification?\n\n"
        f"Bug Report: {issue_text}\n\n"
        f"File: {file_path}\n\n"
        f"Answer:"
    )


def compute_attribution(model, tokenizer, prompt: str, device: str):
    """Compute input-gradient attribution for each token.

    Returns attribution scores and tokens.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=512).to(device)
    input_ids = inputs["input_ids"]

    # Get embeddings with gradient (stay in eval mode for faithful attribution)
    embeddings = model.get_input_embeddings()
    embed = embeddings(input_ids)
    embed.requires_grad_(True)
    embed.retain_grad()

    # Forward pass
    outputs = model(inputs_embeds=embed, attention_mask=inputs["attention_mask"])
    logits = outputs.logits[:, -1, :]

    # Find "Yes" and "No" token IDs
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    # Score = logit(Yes) - logit(No)
    score = logits[0, yes_id] - logits[0, no_id]
    score.backward()

    # Attribution = |grad * embedding| summed over hidden dim
    grad = embed.grad[0]  # (seq_len, hidden_dim)
    attr = (grad * embed.detach()[0]).abs().sum(dim=-1)  # (seq_len,)
    attr = attr.cpu().float().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())

    model.zero_grad()

    return attr, tokens


def classify_tokens_by_offset(tokenizer, prompt):
    """Classify tokens using offset mapping."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=512, return_offsets_mapping=True)

    offsets = enc["offset_mapping"][0].numpy()  # (seq_len, 2)

    # Find character boundaries matching the real prompt template.
    # Use rfind for "File:" and "Answer:" to handle edge cases where
    # bug report text contains these substrings.
    issue_start = prompt.find("Bug Report: ") + len("Bug Report: ")
    issue_end = prompt.rfind("\n\nFile: ")
    path_start = prompt.rfind("File: ") + len("File: ")
    path_end = prompt.rfind("\n\nAnswer:")

    labels = []
    for start, end in offsets:
        if start == 0 and end == 0:
            labels.append("template")  # special tokens
        elif start >= path_start and end <= path_end:
            labels.append("path")
        elif start >= issue_start and end <= issue_end:
            labels.append("issue")
        else:
            labels.append("template")

    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=300)
    parser.add_argument("--output_dir", type=str,
                        default="/data/chenlibin/grepo_agent_experiments/gradient_attribution")
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    data = load_data(args.max_examples)
    print(f"Loaded {len(data)} examples")

    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": device},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    print("Running gradient attribution...")
    results = []
    path_fracs = []
    issue_fracs = []
    template_fracs = []

    for i, ex in enumerate(data):
        prompt = build_prompt(ex["issue_text"], ex["gt_file"])

        try:
            attr, tokens = compute_attribution(model, tokenizer, prompt, device)
            labels = classify_tokens_by_offset(tokenizer, prompt)

            # Ensure lengths match (offset_mapping may differ from model input)
            min_len = min(len(attr), len(labels))
            attr = attr[:min_len]
            labels = labels[:min_len]

            # Aggregate by category
            total = attr.sum() + 1e-10
            path_attr = sum(attr[j] for j in range(len(labels)) if labels[j] == "path")
            issue_attr = sum(attr[j] for j in range(len(labels)) if labels[j] == "issue")
            template_attr = sum(attr[j] for j in range(len(labels)) if labels[j] == "template")

            path_frac = float(path_attr / total)
            issue_frac = float(issue_attr / total)
            template_frac = float(template_attr / total)

            path_fracs.append(path_frac)
            issue_fracs.append(issue_frac)
            template_fracs.append(template_frac)

            n_path = sum(1 for l in labels if l == "path")
            n_issue = sum(1 for l in labels if l == "issue")
            n_template = sum(1 for l in labels if l == "template")

            results.append({
                "repo": ex["repo"],
                "file": ex["gt_file"],
                "path_frac": path_frac,
                "issue_frac": issue_frac,
                "template_frac": template_frac,
                "n_path_tokens": n_path,
                "n_issue_tokens": n_issue,
                "n_template_tokens": n_template,
            })

        except Exception as e:
            print(f"  [{i}] Error: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(data)}] path={np.mean(path_fracs):.3f} "
                  f"issue={np.mean(issue_fracs):.3f} "
                  f"template={np.mean(template_fracs):.3f}")

    # Save results
    summary = {
        "num_examples": len(results),
        "mean_path_attribution": float(np.mean(path_fracs)),
        "mean_issue_attribution": float(np.mean(issue_fracs)),
        "mean_template_attribution": float(np.mean(template_fracs)),
        "std_path_attribution": float(np.std(path_fracs)),
        "std_issue_attribution": float(np.std(issue_fracs)),
        "std_template_attribution": float(np.std(template_fracs)),
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.output_dir, "per_example.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== Results ({len(results)} examples) ===")
    print(f"Path attribution:     {summary['mean_path_attribution']:.3f} +/- {summary['std_path_attribution']:.3f}")
    print(f"Issue attribution:    {summary['mean_issue_attribution']:.3f} +/- {summary['std_issue_attribution']:.3f}")
    print(f"Template attribution: {summary['mean_template_attribution']:.3f} +/- {summary['std_template_attribution']:.3f}")


if __name__ == "__main__":
    main()
