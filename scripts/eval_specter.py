#!/usr/bin/env python3
"""
SPECTER end-to-end evaluation: path expert + router + code expert.

Pipeline:
1. Path expert scores all BM25 candidates (top-200)
2. Router decides per-example whether to invoke code expert based on path uncertainty
3. If routed: code expert re-scores top-10 using function snippets
4. Final R@1 aggregates path-only and code-routed predictions

Supports multiple routing budgets and reports cost-accuracy tradeoff.

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/eval_specter.py \
        --gpu_id 0 \
        --code_expert_lora /data/chenlibin/grepo_agent_experiments/specter/expert/best \
        --path_lora experiments/rankft_runB_graph/best \
        --output_dir /data/chenlibin/grepo_agent_experiments/specter/specter_eval
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
TEST_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_test.jsonl"
BM25_PATH = "/home/chenlibin/grepo_agent/data/rankft/merged_bm25_exp6_candidates.jsonl"
REPO_DIR = "/home/chenlibin/grepo_agent/data/repos"

PATH_PROMPT = (
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


def extract_functions(repo, file_path, max_lines=30):
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
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = min(start + max_lines, len(lines))
            funcs.append({
                "name": node.name,
                "body": "\n".join(lines[start:end]),
                "lineno": node.lineno,
            })
    return funcs


def get_snippets(repo, file_path, issue_text, top_m=3):
    funcs = extract_functions(repo, file_path)
    if not funcs:
        return "# (no functions extracted)"
    issue_tokens = issue_text.lower().split()
    func_tokens = [f["body"].lower().split() + re.split(r'[_]', f["name"].lower())
                   for f in funcs]
    bm25 = BM25Okapi(func_tokens)
    scores = bm25.get_scores(issue_tokens)
    ranked = sorted(zip(funcs, scores), key=lambda x: -x[1])
    snippet = ""
    for f, _ in ranked[:top_m]:
        snippet += f"# {f['name']} (line {f['lineno']})\n{f['body'][:500]}\n\n"
    return snippet.strip()


def truncate_prompt_safely(prompt, tokenizer, max_seq_length):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_seq_length:
        return prompt
    suffix_ids = tokenizer.encode("\n\nAnswer:", add_special_tokens=False)
    keep = max_seq_length - len(suffix_ids) - 1
    return tokenizer.decode(ids[:keep] + suffix_ids, skip_special_tokens=True)


def score_batch(model, tokenizer, prompts, yes_id, no_id, device,
                max_len, batch_size=8):
    scores = []
    for i in range(0, len(prompts), batch_size):
        batch = [truncate_prompt_safely(p, tokenizer, max_len)
                 for p in prompts[i:i + batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_len,
                           padding_side="left").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
        s = (logits[:, yes_id].float() - logits[:, no_id].float()).cpu().numpy()
        scores.extend(s.tolist())
    return scores


def path_coverage(issue_text, path):
    issue_tokens = set(t for t in re.split(r'[/._\-\s,;:!?()\[\]{}"\'`<>]+',
                                            issue_text.lower()) if t)
    path_tokens = set(t for t in re.split(r'[/._\-]', path.lower()) if t)
    if not path_tokens:
        return 0.0
    return len(issue_tokens & path_tokens) / len(path_tokens)


def has_dup_stem(candidates, top_k=10):
    stems = []
    for c in candidates[:top_k]:
        stem = os.path.splitext(os.path.basename(c))[0]
        stem = re.sub(r'^test_|_test$|^tests_|_tests$', '', stem)
        stems.append(stem)
    return len(stems) != len(set(stems))


def extract_router_features(issue_text, candidates, scores):
    """Extract features for routing decision."""
    if len(scores) < 2:
        score_gap = 0
        score_std = 0
        top1_score = 0
    else:
        score_gap = scores[0] - scores[1]
        score_std = float(np.std(scores[:10]))
        top1_score = scores[0]
    top1_path = candidates[0] if candidates else ""
    max_cov = max(path_coverage(issue_text, p) for p in candidates[:5]) if candidates else 0
    top1_cov = path_coverage(issue_text, top1_path)
    dup = 1 if has_dup_stem(candidates) else 0
    issue_len = len(issue_text.split())
    return [score_gap, score_std, top1_score, max_cov, top1_cov, dup, issue_len, 0]


def load_model(lora_path, device):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb,
        device_map={"": device}, trust_remote_code=True,
        torch_dtype=torch.bfloat16)
    if lora_path:
        m = PeftModel.from_pretrained(m, lora_path)
    m.eval()
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    return m, tok, yes_id, no_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--path_lora", type=str, required=True)
    parser.add_argument("--code_expert_lora", type=str, required=True)
    parser.add_argument("--top_k_files", type=int, default=10)
    parser.add_argument("--top_m_funcs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--routing_budgets", type=str,
                        default="0,0.1,0.2,0.3,0.4,0.5,1.0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

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

    # Pass 1: Path expert scores all candidates
    print("\n=== Pass 1: Path expert scoring ===")
    path_model, path_tok, yes_id, no_id = load_model(args.path_lora, device)

    path_results = []
    start = time.time()
    for idx, (key, test_rec) in enumerate(test_data.items()):
        if key not in bm25_data:
            continue
        gt = set(test_rec.get("changed_py_files",
                               test_rec.get("changed_files", [])))
        if not gt:
            continue
        raw_candidates = bm25_data[key].get("candidates",
                                              bm25_data[key].get("bm25_candidates", []))
        # Deduplicate while preserving order
        seen = set()
        candidates = []
        for c in raw_candidates:
            if c not in seen:
                seen.add(c)
                candidates.append(c)
            if len(candidates) >= 200:
                break
        if not candidates:
            continue

        issue = test_rec["issue_text"]
        prompts = [PATH_PROMPT.format(issue_text=issue, candidate_path=c)
                   for c in candidates]
        scores = score_batch(path_model, path_tok, prompts, yes_id, no_id,
                              device, max_len=512)

        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        ranked_cands = [c for c, _ in ranked]
        ranked_scores = [s for _, s in ranked]
        top_k = ranked_cands[:args.top_k_files]

        features = extract_router_features(issue, ranked_cands, ranked_scores)
        path_top1_hit = 1.0 if ranked_cands[0] in gt else 0.0

        path_results.append({
            "key": key,
            "repo": test_rec["repo"],
            "issue_text": issue,
            "gt_files": list(gt),
            "top_k": top_k,
            "path_top1": ranked_cands[0],
            "path_top1_hit": path_top1_hit,
            "features": features,
        })

        if (idx + 1) % 100 == 0:
            r1 = np.mean([r["path_top1_hit"] for r in path_results]) * 100
            print(f"  [{idx+1}] path R@1={r1:.2f}% ({time.time()-start:.0f}s)")

    print(f"\n  Path-only final R@1: {np.mean([r['path_top1_hit'] for r in path_results])*100:.2f}%")

    del path_model
    torch.cuda.empty_cache()

    # Pass 2: Code expert scores top-k for all examples
    print("\n=== Pass 2: Code expert scoring (top-10 per example) ===")
    code_model, code_tok, c_yes, c_no = load_model(args.code_expert_lora, device)

    start = time.time()
    for i, r in enumerate(path_results):
        repo = r["repo"]
        issue = r["issue_text"]
        top_k = r["top_k"]

        issue_ids = code_tok.encode(issue, add_special_tokens=False)
        issue_trunc = code_tok.decode(issue_ids[:max(200, 1024 - 600)],
                                        skip_special_tokens=True)

        prompts = []
        for c in top_k:
            snip = get_snippets(repo, c, issue, top_m=args.top_m_funcs)
            prompts.append(HIER_PROMPT.format(
                issue_text=issue_trunc, candidate_path=c, function_snippets=snip))

        scores = score_batch(code_model, code_tok, prompts, c_yes, c_no,
                              device, max_len=1024, batch_size=4)
        ranked = sorted(zip(top_k, scores), key=lambda x: -x[1])
        code_top1 = ranked[0][0]
        code_top1_hit = 1.0 if code_top1 in set(r["gt_files"]) else 0.0

        r["code_top1"] = code_top1
        r["code_top1_hit"] = code_top1_hit

        if (i + 1) % 100 == 0:
            r1 = np.mean([x.get("code_top1_hit", 0) for x in path_results[:i+1]]) * 100
            print(f"  [{i+1}] code R@1 (top-10)={r1:.2f}% ({time.time()-start:.0f}s)")

    del code_model
    torch.cuda.empty_cache()

    # Pass 3: Router + budget sweep
    print("\n=== Pass 3: Routing evaluation (repo-grouped OOF CV) ===")
    print("  NOTE: router is trained via repo-grouped 5-fold OOF CV on test features")
    print("  (each example's routing decision comes from a fold that never saw its repo)")

    X = np.array([r["features"] for r in path_results])
    y_path_wrong = np.array([1 - r["path_top1_hit"] for r in path_results]).astype(int)
    # Stable group ids (use repo string directly so order is deterministic)
    unique_repos = sorted(set(r["key"][0] for r in path_results))
    repo_to_id = {repo: i for i, repo in enumerate(unique_repos)}
    groups = np.array([repo_to_id[r["key"][0]] for r in path_results])

    gkf = GroupKFold(n_splits=5)
    router_prob = np.zeros(len(y_path_wrong))
    for train_idx, val_idx in gkf.split(X, y_path_wrong, groups):
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                          learning_rate=0.1, random_state=42)
        clf.fit(X[train_idx], y_path_wrong[train_idx])
        router_prob[val_idx] = clf.predict_proba(X[val_idx])[:, 1]

    budgets = [float(b) for b in args.routing_budgets.split(",")]
    results = []
    for budget in budgets:
        if budget == 0:
            routed = [False] * len(path_results)
        elif budget >= 1.0:
            routed = [True] * len(path_results)
        else:
            threshold = np.quantile(router_prob, 1 - budget)
            routed = [p >= threshold for p in router_prob]

        hits = 0
        for i, r in enumerate(path_results):
            if routed[i]:
                hits += r.get("code_top1_hit", 0)
            else:
                hits += r["path_top1_hit"]

        overall = hits / len(path_results) * 100
        n_routed = sum(routed)
        results.append({
            "budget": budget,
            "n_routed": int(n_routed),
            "route_fraction": float(n_routed / len(path_results)),
            "overall_R@1": float(overall),
        })
        print(f"  budget={budget:.2f}: route={n_routed/len(path_results):.0%} "
              f"R@1={overall:.2f}%")

    oracle_hits = 0
    for r in path_results:
        oracle_hits += max(r["path_top1_hit"], r.get("code_top1_hit", 0))
    oracle_r1 = oracle_hits / len(path_results) * 100
    print(f"  Oracle routing: R@1={oracle_r1:.2f}%")

    summary = {
        "num_examples": len(path_results),
        "path_only_R@1": float(np.mean([r["path_top1_hit"] for r in path_results]) * 100),
        "code_only_R@1_top10": float(np.mean([r.get("code_top1_hit", 0) for r in path_results]) * 100),
        "oracle_R@1": oracle_r1,
        "budget_sweep": results,
        "router_protocol": "repo_grouped_5fold_OOF_CV_on_test",
        "protocol_caveat": "NOT a held-out router evaluation; router features and "
                            "labels come from the test set via repo-grouped cross-validation",
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== SPECTER Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
