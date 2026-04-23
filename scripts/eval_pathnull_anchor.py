#!/usr/bin/env python3
"""Path-Null Anchor: pointwise reranker with path-swap counterfactual.

Method (training-free):
  For each (issue, candidate), score twice:
    s_real = reranker(issue, real_path, code)
    s_hash = reranker(issue, SHA256(path), code)
  code_residual = s_real - s_hash
  fused(alpha) = alpha * s_real + (1-alpha) * code_residual

Report R@1 at each alpha. Also report path-only (alpha=1), residual-only (alpha=0),
and sweep.

Usage:
  CUDA_VISIBLE_DEVICES=5 python scripts/eval_pathnull_anchor.py \
      --test_data data/swebench_lite/swebench_lite_test.jsonl \
      --bm25_candidates data/rankft/swebench_bm25_final_top500.jsonl \
      --repo_dir data/swebench_lite/repos \
      --top_k 100 \
      --output_dir /data/chenlibin/grepo_agent_experiments/pathnull_anchor_swebench
"""
import argparse, json, os, random, hashlib, time
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42); np.random.seed(42); torch.manual_seed(42)

YESNO_PROMPT = ("Given the bug report, is this file likely to need modification?\n\n"
                "Bug Report: {issue_text}\n\nFile: {candidate_path}\n\nAnswer:")


def hash_path(p):
    parts = []
    for x in p.split("/"):
        if not x: continue
        h = hashlib.sha256(x.encode()).hexdigest()[:8]
        if x.endswith(".py"): parts.append(f"m_{h}.py")
        else: parts.append(f"d_{h}")
    return "/".join(parts)


def truncate_prompt(p, tok, max_len):
    ids = tok.encode(p, add_special_tokens=False)
    if len(ids) <= max_len: return p
    suf = tok.encode("\n\nAnswer:", add_special_tokens=False)
    keep = max_len - len(suf) - 1
    return tok.decode(ids[:keep] + suf, skip_special_tokens=True)


def score_batch(m, tok, prompts, yes_id, no_id, device, max_len=1024, bs=8):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/data/shuyang/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--lora_path", default="/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/best")
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--bm25_candidates", required=True)
    ap.add_argument("--repo_dir", required=True)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(args.model_path, quantization_config=bnb,
        device_map={"": device}, trust_remote_code=True, torch_dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, args.lora_path); m.eval()
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    print(f"Loaded model; Yes={yes_id} No={no_id}", flush=True)

    data = [json.loads(l) for l in open(args.test_data)]
    bm25 = {}
    for l in open(args.bm25_candidates):
        r = json.loads(l)
        bm25[(r["repo"], str(r["issue_id"]))] = r
    print(f"Loaded {len(data)} test, {len(bm25)} bm25 records", flush=True)

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hits = {a: [] for a in alphas}
    per_example = []
    start = time.time()

    for i, rec in enumerate(data):
        repo = rec.get("repo", "")
        issue = rec.get("issue_text", "")[:1500]
        gt = set(rec.get("changed_py_files", rec.get("changed_files", [])))
        key = (repo, str(rec.get("issue_id", "")))
        if key not in bm25: continue
        cands = bm25[key].get("bm25_candidates", bm25[key].get("candidates", []))[:args.top_k]
        if not gt or not cands: continue
        # ensure GT present
        for g in gt:
            if g not in cands: cands.append(g)

        prompts_real = [YESNO_PROMPT.format(issue_text=issue, candidate_path=c) for c in cands]
        prompts_hash = [YESNO_PROMPT.format(issue_text=issue, candidate_path=hash_path(c)) for c in cands]
        s_real = score_batch(m, tok, prompts_real, yes_id, no_id, device, bs=args.batch_size)
        s_hash = score_batch(m, tok, prompts_hash, yes_id, no_id, device, bs=args.batch_size)
        s_real = np.array(s_real); s_hash = np.array(s_hash)
        residual = s_real - s_hash

        for a in alphas:
            fused = a * s_real + (1 - a) * residual
            ranked = sorted(zip(cands, fused.tolist()), key=lambda x: -x[1])
            hits[a].append(1.0 if ranked[0][0] in gt else 0.0)

        per_example.append({"repo": repo, "issue_id": rec.get("issue_id"),
                            "gt": list(gt), "top1_real": max(zip(cands, s_real), key=lambda x: x[1])[0],
                            "top1_residual": max(zip(cands, residual), key=lambda x: x[1])[0]})
        if (i+1) % 20 == 0:
            msg = " ".join([f"a={a:.1f}:{np.mean(hits[a])*100:.1f}" for a in [0.0, 0.5, 1.0]])
            print(f"  [{i+1}] {msg} ({time.time()-start:.0f}s)", flush=True)

    summary = {}
    for a in alphas:
        summary[f"alpha={a:.1f}"] = float(np.mean(hits[a])*100) if hits[a] else 0.0
    summary["n"] = len(hits[alphas[0]])
    summary["best_alpha"] = max(alphas, key=lambda a: np.mean(hits[a]) if hits[a] else 0)
    summary["best_r1"] = max(float(np.mean(hits[a])*100) for a in alphas if hits[a])
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "per_example.jsonl"), "w") as f:
        for p in per_example: f.write(json.dumps(p) + "\n")
    print("\n=== Results ===")
    for a in alphas: print(f"  alpha={a:.1f}: R@1={summary[f'alpha={a:.1f}']:.2f}%")
    print(f"Best: alpha={summary['best_alpha']:.1f} R@1={summary['best_r1']:.2f}%")


if __name__ == "__main__":
    main()
