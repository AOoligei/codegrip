#!/usr/bin/env python3
"""SWE-bench Lite pointwise R@1 on BM25 top-100 hard pool: path vs path+code.

Matches GREPO pointwise protocol (hard BM25 pool) for fair comparison.
"""
import argparse, json, os, random, time
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42); np.random.seed(42); torch.manual_seed(42)

MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
LORA_PATH = "/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/best"
TEST_PATH = "/home/chenlibin/grepo_agent/data/swebench_lite/swebench_lite_test.jsonl"
BM25_PATH = "/home/chenlibin/grepo_agent/data/rankft/swebench_bm25_final_top500.jsonl"
REPO_DIR = "/home/chenlibin/grepo_agent/data/swebench_lite/repos"
TOPK = 100

PATH_PROMPT = ("Given the bug report, is this file likely to need modification?\n\n"
               "Bug Report: {issue_text}\n\nFile: {candidate_path}\n\nAnswer:")
CODE_PROMPT = ("Given the bug report, is this file likely to need modification?\n\n"
               "Bug Report: {issue_text}\n\nFile: {candidate_path}\n"
               "Code:\n{code}\n\nAnswer:")


def read_head(repo, fpath, n=50):
    full = os.path.join(REPO_DIR, repo, fpath)
    if not os.path.isfile(full):
        return "# (unavailable)"
    try:
        with open(full, "r", errors="replace") as fh:
            return "".join(fh.readlines()[:n])
    except Exception:
        return "# (unreadable)"


def truncate_prompt(p, tok, max_len):
    ids = tok.encode(p, add_special_tokens=False)
    if len(ids) <= max_len:
        return p
    suf = tok.encode("\n\nAnswer:", add_special_tokens=False)
    keep = max_len - len(suf) - 1
    return tok.decode(ids[:keep] + suf, skip_special_tokens=True)


def score_batch(m, tok, prompts, yes_id, no_id, device, max_len=1536, bs=4):
    prompts = [truncate_prompt(p, tok, max_len) for p in prompts]
    out = []
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i+bs]
        inp = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len, padding_side="left").to(device)
        with torch.no_grad():
            logits = m(**inp).logits[:, -1, :]
        s = (logits[:, yes_id].float() - logits[:, no_id].float()).cpu().numpy()
        out.extend(s.tolist())
    return out


def load_model(device):
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
    m = PeftModel.from_pretrained(m, LORA_PATH); m.eval()
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    return m, tok, yes_id, no_id


def run_one(m, tok, yes_id, no_id, device, data, bm25, use_code, label):
    hits = []; start = time.time()
    for i, rec in enumerate(data):
        repo = rec.get("repo", "")
        issue_id = str(rec.get("issue_id", ""))
        key = (repo, issue_id)
        if key not in bm25: continue
        cand_rec = bm25[key]
        cands = list(cand_rec.get("bm25_candidates", []))[:TOPK]
        gt = set(cand_rec.get("ground_truth", []) or
                 rec.get("changed_py_files", rec.get("changed_files", [])))
        if not gt or not cands: continue
        # ensure GT present so oracle=1.0 (fair hit@1 over same pool)
        for g in gt:
            if g not in cands: cands.append(g)
        issue = rec["issue_text"][:1500]
        if use_code:
            prompts = [CODE_PROMPT.format(issue_text=issue, candidate_path=c,
                                           code=read_head(repo, c)[:1500]) for c in cands]
        else:
            prompts = [PATH_PROMPT.format(issue_text=issue, candidate_path=c) for c in cands]
        scores = score_batch(m, tok, prompts, yes_id, no_id, device)
        ranked = sorted(zip(cands, scores), key=lambda x: -x[1])
        hits.append(1.0 if ranked[0][0] in gt else 0.0)
        if (i+1) % 20 == 0:
            print(f"  [{label}] [{i+1}] R@1={np.mean(hits)*100:.2f}% ({time.time()-start:.0f}s)", flush=True)
    return float(np.mean(hits)*100) if hits else 0.0, len(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--output_dir", type=str, required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    data = []
    with open(TEST_PATH) as f:
        for line in f: data.append(json.loads(line))
    bm25 = {}
    with open(BM25_PATH) as f:
        for line in f:
            r = json.loads(line)
            bm25[(r["repo"], str(r["issue_id"]))] = r
    print(f"Loaded {len(data)} test, {len(bm25)} bm25 candidate records", flush=True)

    m, tok, yes_id, no_id = load_model(device)

    print(f"\n=== Path-only (BM25 top-{TOPK}) ===", flush=True)
    p_r1, n1 = run_one(m, tok, yes_id, no_id, device, data, bm25, False, "path")
    print(f"  path R@1={p_r1:.2f}% (n={n1})", flush=True)

    print(f"\n=== Path+Code (BM25 top-{TOPK}) ===", flush=True)
    c_r1, n2 = run_one(m, tok, yes_id, no_id, device, data, bm25, True, "code")
    print(f"  code R@1={c_r1:.2f}% (n={n2})", flush=True)

    summary = {"path_pointwise_r1": p_r1, "code_pointwise_r1": c_r1,
               "delta": c_r1 - p_r1, "n": min(n1, n2),
               "benchmark": "swebench_lite", "pool": f"bm25_top{TOPK}"}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Result ===\nPath: {p_r1:.2f}%  Code: {c_r1:.2f}%  Delta: {c_r1-p_r1:+.2f}pp", flush=True)


if __name__ == "__main__":
    main()
