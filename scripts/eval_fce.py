#!/usr/bin/env python3
"""
Evaluate Function-level Code Expert (FCE) on GREPO function-level task.

Compares path-only function scoring vs FCE (with code body).
If FCE > path-only, code understanding works at function granularity.

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/eval_fce.py \
        --gpu_id 0 \
        --fce_lora /data/chenlibin/grepo_agent_experiments/fce/expert_v2/best \
        --path_lora experiments/rankft_runB_graph/best \
        --output_dir /data/chenlibin/grepo_agent_experiments/fce/fce_eval
"""

import argparse
import ast
import json
import os
import random
import time

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
FUNC_INDEX_PATH = "/home/chenlibin/grepo_agent/data/function_index_aligned.json"
REPO_DIR = "/home/chenlibin/grepo_agent/data/repos"

PATH_FUNC_PROMPT = (
    "Given the bug report, is this function in {file_path} likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {file_path}\n"
    "Function: {function_name}\n\n"
    "Answer:"
)

FCE_PROMPT = (
    "Given the bug report, is this function likely to need modification? "
    "Read the function body carefully.\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {file_path}\n"
    "Function: {function_name}\n"
    "Code:\n{function_body}\n\n"
    "Answer:"
)


def extract_functions(repo, file_path, max_lines=40):
    full_path = os.path.join(REPO_DIR, repo, file_path)
    if not os.path.isfile(full_path):
        return []
    try:
        with open(full_path, "r", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception:
        return []
    lines = source.splitlines()
    funcs = []

    def walk(node, scope=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, f"{scope}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{scope}{child.name}"
                start = child.lineno - 1
                end = min(start + max_lines, len(lines))
                funcs.append({
                    "name": child.name, "qual_name": qual,
                    "body": "\n".join(lines[start:end]), "lineno": child.lineno,
                })
                walk(child, f"{qual}.")
            else:
                walk(child, scope)
    walk(tree)
    return funcs


def truncate_prompt(prompt, tokenizer, max_len):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_len:
        return prompt
    suffix_ids = tokenizer.encode("\n\nAnswer:", add_special_tokens=False)
    keep = max_len - len(suffix_ids) - 1
    return tokenizer.decode(ids[:keep] + suffix_ids, skip_special_tokens=True)


def score_batch(model, tokenizer, prompts, yes_id, no_id, device, max_len, batch_size=4):
    prompts = [truncate_prompt(p, tokenizer, max_len) for p in prompts]
    scores = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_len,
                           padding_side="left").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
        s = (logits[:, yes_id].float() - logits[:, no_id].float()).cpu().numpy()
        scores.extend(s.tolist())
    return scores


def load_model(lora_path, device):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
                                              device_map={"": device},
                                              trust_remote_code=True,
                                              torch_dtype=torch.bfloat16)
    if lora_path:
        m = PeftModel.from_pretrained(m, lora_path)
    m.eval()
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    return m, tok, yes_id, no_id


def run_eval(model, tokenizer, yes_id, no_id, device, test_data, bm25_data,
             use_code, max_len, label):
    hits1, hits5, n = [], [], 0
    start = time.time()

    for idx, (key, rec) in enumerate(test_data.items()):
        if key not in bm25_data:
            continue
        repo = rec["repo"]
        issue = rec["issue_text"]
        gt_files = rec.get("changed_py_files", rec.get("changed_files", []))
        gt_funcs = set(rec.get("changed_functions", []))
        if not gt_funcs or not gt_files:
            continue

        bm25_cands = bm25_data[key].get("candidates",
                                          bm25_data[key].get("bm25_candidates", []))

        # Build function candidates from GT files + BM25 top files
        candidates = []
        gt_pairs = set()
        for gf in gt_files:
            for func in extract_functions(repo, gf):
                cid = f"{gf}::{func['qual_name']}"
                is_gt = func["name"] in gt_funcs or func["qual_name"] in gt_funcs
                candidates.append({"file": gf, "func": func, "id": cid})
                if is_gt:
                    gt_pairs.add(cid)

        for bf in bm25_cands[:15]:
            if bf in gt_files:
                continue
            for func in extract_functions(repo, bf)[:5]:
                cid = f"{bf}::{func['qual_name']}"
                candidates.append({"file": bf, "func": func, "id": cid})

        if not gt_pairs or len(candidates) < 3:
            continue

        candidates = candidates[:50]

        prompts = []
        for c in candidates:
            if use_code:
                prompts.append(FCE_PROMPT.format(
                    issue_text=issue, file_path=c["file"],
                    function_name=c["func"].get("qual_name", c["func"]["name"]),
                    function_body=c["func"]["body"][:800]))
            else:
                prompts.append(PATH_FUNC_PROMPT.format(
                    issue_text=issue, file_path=c["file"],
                    function_name=c["func"].get("qual_name", c["func"]["name"])))

        scores = score_batch(model, tokenizer, prompts, yes_id, no_id, device, max_len)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        h1 = 1.0 if ranked[0][0]["id"] in gt_pairs else 0.0
        h5 = 1.0 if any(r[0]["id"] in gt_pairs for r in ranked[:5]) else 0.0
        hits1.append(h1)
        hits5.append(h5)
        n += 1

        if (idx + 1) % 50 == 0:
            print(f"  [{label}] [{idx+1}] Hit@1={np.mean(hits1)*100:.1f}% "
                  f"Hit@5={np.mean(hits5)*100:.1f}% n={n} ({time.time()-start:.0f}s)")

    return {
        "Hit@1": float(np.mean(hits1) * 100) if hits1 else 0,
        "Hit@5": float(np.mean(hits5) * 100) if hits5 else 0,
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--path_lora", type=str, required=True)
    parser.add_argument("--fce_lora", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    print("Loading data...")
    test_data = {}
    with open(TEST_PATH) as f:
        for line in f:
            r = json.loads(line)
            test_data[(r["repo"], str(r["issue_id"]))] = r
    bm25_data = {}
    with open(BM25_PATH) as f:
        for line in f:
            r = json.loads(line)
            bm25_data[(r["repo"], str(r["issue_id"]))] = r
    print(f"  {len(test_data)} test")

    print("\n=== Path-only function scoring ===")
    pm, pt, py, pn = load_model(args.path_lora, device)
    path_r = run_eval(pm, pt, py, pn, device, test_data, bm25_data,
                       use_code=False, max_len=512, label="path")
    print(f"\n  Path: Hit@1={path_r['Hit@1']:.2f}% Hit@5={path_r['Hit@5']:.2f}%")
    del pm; torch.cuda.empty_cache()

    print("\n=== FCE function scoring (with code) ===")
    fm, ft, fy, fn_ = load_model(args.fce_lora, device)
    fce_r = run_eval(fm, ft, fy, fn_, device, test_data, bm25_data,
                      use_code=True, max_len=1536, label="FCE")
    print(f"\n  FCE: Hit@1={fce_r['Hit@1']:.2f}% Hit@5={fce_r['Hit@5']:.2f}%")
    del fm; torch.cuda.empty_cache()

    d1 = fce_r["Hit@1"] - path_r["Hit@1"]
    d5 = fce_r["Hit@5"] - path_r["Hit@5"]
    print(f"\n=== RESULT ===")
    print(f"Path: Hit@1={path_r['Hit@1']:.2f}%  FCE: Hit@1={fce_r['Hit@1']:.2f}%  Delta: {d1:+.2f}pp")
    if d1 > 0:
        print(f">>> CODE HELPS AT FUNCTION LEVEL!")
    else:
        print(f">>> Code does NOT help at function level")

    summary = {"path_only": path_r, "fce": fce_r, "delta_hit1": d1, "delta_hit5": d5}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
