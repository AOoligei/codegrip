#!/usr/bin/env python3
"""
Training-free self-consistency scoring:
  s_full(c) = path-only reranker score with REAL path
  s_hashed(c) = same reranker with HASHED path (inference-only)
  path_reliance(c) = s_full(c) - s_hashed(c)  ; larger = more path-dependent
  s_final(c) = s_full(c) - lambda * max(0, path_reliance(c))

Evaluate on SWE-bench Lite BM25 top-100, sweep lambda.
"""
import argparse, hashlib, json, os, random, time
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42); np.random.seed(42); torch.manual_seed(42)

MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
LORA_PATH = "/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/best"
TEST_PATH = "/home/chenlibin/grepo_agent/data/swebench_lite/swebench_lite_test.jsonl"
BM25_PATH = "/home/chenlibin/grepo_agent/data/rankft/swebench_bm25_final_top500.jsonl"

def hash_path(p):
    parts = []
    for x in p.split("/"):
        if not x: continue
        h = hashlib.sha256(x.encode()).hexdigest()[:8]
        parts.append(f"m_{h}.py" if x.endswith(".py") else f"d_{h}")
    return "/".join(parts)

PROMPT = ("Given the bug report, is this file likely to need modification?\n\n"
          "Bug Report: {issue_text}\n\nFile: {file}\n\nAnswer:")

def load_model(dev):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
            device_map={"": dev}, trust_remote_code=True, torch_dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, LORA_PATH); m.eval()
    return m, tok, tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]

def score_batch(m, tok, prompts, yid, nid, dev, bs=8):
    out = []
    for i in range(0, len(prompts), bs):
        b = prompts[i:i+bs]
        inp = tok(b, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024, padding_side="left").to(dev)
        with torch.no_grad():
            logits = m(**inp).logits[:, -1, :]
        s = (logits[:, yid].float() - logits[:, nid].float()).cpu().numpy()
        out.extend(s.tolist())
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--K", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dev = f"cuda:{args.gpu_id}"

    test = [json.loads(l) for l in open(TEST_PATH)]
    bm25 = {}
    with open(BM25_PATH) as f:
        for l in f:
            r = json.loads(l); bm25[(r["repo"], str(r["issue_id"]))] = r

    m, tok, yid, nid = load_model(dev)

    records = []
    start = time.time()
    for i, rec in enumerate(test):
        repo = rec.get("repo", "")
        issue = rec["issue_text"][:1500]
        gt = set(rec.get("changed_py_files", rec.get("changed_files", [])))
        if not gt: continue
        key = (repo, str(rec.get("issue_id", "")))
        if key not in bm25: continue
        cands = bm25[key].get("bm25_candidates", [])[:args.K]
        if not cands: continue
        for g in gt:
            if g not in cands: cands.append(g)

        # Score with real paths
        prompts_real = [PROMPT.format(issue_text=issue, file=c) for c in cands]
        s_full = score_batch(m, tok, prompts_real, yid, nid, dev)

        # Score with hashed paths
        prompts_hash = [PROMPT.format(issue_text=issue, file=hash_path(c)) for c in cands]
        s_hashed = score_batch(m, tok, prompts_hash, yid, nid, dev)

        records.append({"repo": repo, "gt": list(gt), "cands": cands,
                        "s_full": s_full, "s_hashed": s_hashed})

        if (i+1) % 20 == 0:
            print(f"  [{i+1}] ({time.time()-start:.0f}s)", flush=True)

    # Sweep lambda and compute R@1 for each strategy
    print(f"\n=== Scoring strategies (n={len(records)}) ===")
    strategies = {
        "full_only":        lambda sf, sh: sf,
        "hashed_only":      lambda sf, sh: sh,
        "sum":              lambda sf, sh: [a+b for a,b in zip(sf, sh)],
        "min":              lambda sf, sh: [min(a,b) for a,b in zip(sf, sh)],
        "max":              lambda sf, sh: [max(a,b) for a,b in zip(sf, sh)],
    }
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
        strategies[f"penalty_λ={lam}"] = lambda sf, sh, L=lam: [a - L*max(0,a-b) for a,b in zip(sf, sh)]
    for alpha in [0.3, 0.5, 0.7]:
        strategies[f"blend_α={alpha}"] = lambda sf, sh, A=alpha: [A*a + (1-A)*b for a,b in zip(sf, sh)]

    results = {}
    for name, fn in strategies.items():
        hits = []
        for r in records:
            scored = fn(r["s_full"], r["s_hashed"])
            top_idx = int(np.argmax(scored))
            top1 = r["cands"][top_idx]
            hits.append(1.0 if top1 in set(r["gt"]) else 0.0)
        acc = float(np.mean(hits)*100)
        results[name] = acc
        print(f"  {name:25s}: R@1={acc:.2f}%")

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"n": len(records), "strategies": results}, f, indent=2)

if __name__ == "__main__":
    main()
