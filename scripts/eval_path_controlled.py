#!/usr/bin/env python3
"""
Path-controlled contrastive evaluation: build candidate pools where path
signal is neutralized, then test whether code helps.

For each test example, constructs a "path-confusable" pool:
- GT file + N candidates from same directory or with same filename stem
- On these pools, path-only should struggle (similar paths)
- If code-aware also struggles, code doesn't help even when paths can't distinguish

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/eval_path_controlled.py \
        --gpu_id 0 --pool_size 20 \
        --output_dir /data/chenlibin/grepo_agent_experiments/path_controlled_eval
"""

import argparse
import ast
import json
import os
import random
import re
import time

import numpy as np
import torch
from peft import PeftModel
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
LORA_PATH = "/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/best"
TEST_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_test.jsonl"
BM25_PATH = "/home/chenlibin/grepo_agent/data/rankft/merged_bm25_exp6_candidates.jsonl"
REPO_DIR = "/home/chenlibin/grepo_agent/data/repos"

PATH_ONLY_PROMPT = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)

HIER_PROMPT = (
    "Given the bug report, is this file likely to need modification? "
    "Consider both the file path and the code snippets shown below.\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n"
    "Relevant functions:\n{function_snippets}\n\n"
    "Answer:"
)


def extract_functions(repo_name, file_path, max_lines=30):
    full_path = os.path.join(REPO_DIR, repo_name, file_path)
    if not os.path.isfile(full_path):
        return []
    try:
        with open(full_path, "r", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception:
        return []
    lines = source.splitlines()
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = min(start + max_lines, len(lines))
            body = "\n".join(lines[start:end])
            functions.append({"name": node.name, "body": body, "lineno": node.lineno})
    return functions


def get_snippets(repo, file_path, issue_text, top_m=3):
    funcs = extract_functions(repo, file_path)
    if not funcs:
        return "# (no functions extracted)"
    issue_tokens = issue_text.lower().split()
    func_tokens = [f["body"].lower().split() + re.split(r'[_]', f["name"].lower()) for f in funcs]
    bm25 = BM25Okapi(func_tokens)
    scores = bm25.get_scores(issue_tokens)
    ranked = sorted(zip(funcs, scores), key=lambda x: -x[1])
    snippet = ""
    for f, s in ranked[:top_m]:
        snippet += f"# {f['name']} (line {f['lineno']})\n{f['body'][:500]}\n\n"
    return snippet.strip()


def path_similarity(p1, p2):
    """Jaccard similarity between path token sets."""
    t1 = set(re.split(r'[/._\-]', p1.lower()))
    t2 = set(re.split(r'[/._\-]', p2.lower()))
    t1.discard('')
    t2.discard('')
    if not t1 or not t2:
        return 0
    return len(t1 & t2) / len(t1 | t2)


def build_confusable_pool(gt_file, all_candidates, pool_size=20):
    """Build a pool of genuinely path-confusable candidates.

    Priority: same-directory files first, then same-stem files,
    then highest Jaccard similarity. Must have at least 3 candidates.
    """
    gt_dir = os.path.dirname(gt_file)
    gt_stem = os.path.splitext(os.path.basename(gt_file))[0]
    # Strip test_ / _test for stem matching
    gt_stem_clean = gt_stem.replace("test_", "").replace("_test", "")

    same_dir = []
    same_stem = []
    other = []

    for c in all_candidates:
        if c == gt_file:
            continue
        c_dir = os.path.dirname(c)
        c_stem = os.path.splitext(os.path.basename(c))[0]
        c_stem_clean = c_stem.replace("test_", "").replace("_test", "")

        if c_dir == gt_dir:
            same_dir.append(c)
        elif c_stem_clean == gt_stem_clean:
            same_stem.append(c)
        else:
            other.append((c, path_similarity(gt_file, c)))

    other.sort(key=lambda x: -x[1])

    pool = [gt_file]
    # Add same-dir first, then same-stem, then by similarity
    for c in same_dir[:pool_size]:
        if len(pool) >= pool_size:
            break
        pool.append(c)
    for c in same_stem[:pool_size]:
        if len(pool) >= pool_size:
            break
        pool.append(c)
    for c, sim in other:
        if len(pool) >= pool_size:
            break
        pool.append(c)

    if len(pool) < 3:
        return None
    random.shuffle(pool)
    return pool


def load_model(model_path, lora_path, gpu_id):
    device = f"cuda:{gpu_id}"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config,
        device_map={"": device}, trust_remote_code=True,
        torch_dtype=torch.bfloat16)
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    return model, tokenizer, yes_id, no_id, device


def score_batch(model, tokenizer, prompts, yes_id, no_id, device, max_len=1024):
    scores = []
    for i in range(0, len(prompts), 4):
        batch = prompts[i:i+4]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
        s = (logits[:, yes_id].float() - logits[:, no_id].float()).cpu().numpy()
        scores.extend(s.tolist())
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--pool_size", type=int, default=20)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
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

    print(f"  {len(test_data)} test, {len(bm25_data)} candidates")

    print("Loading model...")
    model, tokenizer, yes_id, no_id, device = load_model(MODEL_PATH, LORA_PATH, args.gpu_id)

    path_hits = []
    hier_hits = []
    pool_sizes = []
    avg_sims = []
    results = []
    start = time.time()

    for idx, (key, test_rec) in enumerate(test_data.items()):
        if key not in bm25_data:
            continue
        repo = test_rec["repo"]
        issue_text = test_rec["issue_text"]
        gt_files = set(test_rec.get("changed_py_files", test_rec.get("changed_files", [])))
        if not gt_files:
            continue

        candidates = bm25_data[key].get("candidates", bm25_data[key].get("bm25_candidates", []))[:200]
        gt_in_cands = [g for g in gt_files if g in candidates]
        if not gt_in_cands:
            continue

        gt_file = gt_in_cands[0]
        pool = build_confusable_pool(gt_file, candidates, args.pool_size)
        if pool is None:
            continue

        pool_sizes.append(len(pool))
        sims = [path_similarity(gt_file, c) for c in pool if c != gt_file]
        avg_sims.append(np.mean(sims) if sims else 0)

        # Truncate issue for hierarchical
        issue_ids = tokenizer.encode(issue_text, add_special_tokens=False)
        issue_trunc = tokenizer.decode(issue_ids[:1024], skip_special_tokens=True)

        # Path-only scoring (same truncation budget as hierarchical for fair comparison)
        path_prompts = [PATH_ONLY_PROMPT.format(issue_text=issue_trunc, candidate_path=c) for c in pool]
        path_scores = score_batch(model, tokenizer, path_prompts, yes_id, no_id, device)
        path_ranked = sorted(zip(pool, path_scores), key=lambda x: -x[1])
        path_hit = 1.0 if path_ranked[0][0] in gt_files else 0.0
        path_hits.append(path_hit)

        # Hierarchical scoring
        hier_prompts = []
        for c in pool:
            snip = get_snippets(repo, c, issue_text)
            hier_prompts.append(HIER_PROMPT.format(
                issue_text=issue_trunc, candidate_path=c, function_snippets=snip))
        hier_scores = score_batch(model, tokenizer, hier_prompts, yes_id, no_id, device, max_len=2048)
        hier_ranked = sorted(zip(pool, hier_scores), key=lambda x: -x[1])
        hier_hit = 1.0 if hier_ranked[0][0] in gt_files else 0.0
        hier_hits.append(hier_hit)

        results.append({
            "repo": repo, "issue_id": str(test_rec["issue_id"]),
            "pool_size": len(pool), "avg_sim": float(np.mean(sims) if sims else 0),
            "path_hit": path_hit, "hier_hit": hier_hit,
        })

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}] path={np.mean(path_hits)*100:.1f}% "
                  f"hier={np.mean(hier_hits)*100:.1f}% "
                  f"pool={np.mean(pool_sizes):.0f} "
                  f"sim={np.mean(avg_sims):.3f} ({time.time()-start:.0f}s)")

    n = len(path_hits)
    summary = {
        "num_examples": n,
        "pool_size": args.pool_size,
        "path_controlled_R@1": float(np.mean(path_hits) * 100),
        "hier_controlled_R@1": float(np.mean(hier_hits) * 100),
        "delta_pp": float((np.mean(hier_hits) - np.mean(path_hits)) * 100),
        "avg_pool_size": float(np.mean(pool_sizes)),
        "avg_path_similarity": float(np.mean(avg_sims)),
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "per_example.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== Path-Controlled Results (n={n}, pool={args.pool_size}) ===")
    print(f"Path-only R@1:      {summary['path_controlled_R@1']:.2f}%")
    print(f"Hierarchical R@1:   {summary['hier_controlled_R@1']:.2f}%")
    print(f"Delta:              {summary['delta_pp']:+.2f}pp")
    print(f"Avg path similarity: {summary['avg_path_similarity']:.3f}")


if __name__ == "__main__":
    main()
