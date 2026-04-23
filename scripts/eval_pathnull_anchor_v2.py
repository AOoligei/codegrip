#!/usr/bin/env python3
"""Path-Null Anchor v2: add CODE to prompt so hash residual isolates code signal."""
import argparse, json, os, random, hashlib, time
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42); np.random.seed(42); torch.manual_seed(42)

PROMPT = ("Given the bug report, is this file likely to need modification?\n\n"
          "Bug Report: {issue_text}\n\nFile: {candidate_path}\nCode:\n{code}\n\nAnswer:")


def hash_path(p):
    parts = []
    for x in p.split("/"):
        if not x: continue
        h = hashlib.sha256(x.encode()).hexdigest()[:8]
        parts.append(f"m_{h}.py" if x.endswith(".py") else f"d_{h}")
    return "/".join(parts)


def read_head(repo_dir, repo, fpath, n=50):
    full = os.path.join(repo_dir, repo, fpath)
    if not os.path.isfile(full): return "# (not available)"
    try:
        with open(full, "r", errors="replace") as fh:
            return "".join(fh.readlines()[:n])[:800]
    except Exception:
        return "# (unreadable)"


def truncate(p, tok, max_len):
    ids = tok.encode(p, add_special_tokens=False)
    if len(ids) <= max_len: return p
    suf = tok.encode("\n\nAnswer:", add_special_tokens=False)
    return tok.decode(ids[:max_len-len(suf)-1] + suf, skip_special_tokens=True)


def score_batch(m, tok, prompts, yes_id, no_id, device, max_len=1536, bs=4):
    prompts = [truncate(p, tok, max_len) for p in prompts]
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
    ap.add_argument("--batch_size", type=int, default=4)
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
    print(f"Loaded; Yes={yes_id} No={no_id}", flush=True)

    data = [json.loads(l) for l in open(args.test_data)]
    bm25 = {}
    for l in open(args.bm25_candidates):
        r = json.loads(l); bm25[(r["repo"], str(r["issue_id"]))] = r
    print(f"Loaded {len(data)} test, {len(bm25)} bm25", flush=True)

    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    hits = {a: [] for a in alphas}
    hits_hash_only = []  # alpha doesn't make sense here; separately track s_hash (code-only)
    start = time.time()

    for i, rec in enumerate(data):
        repo = rec.get("repo", "")
        issue = rec.get("issue_text", "")[:1500]
        gt = set(rec.get("changed_py_files", rec.get("changed_files", [])))
        key = (repo, str(rec.get("issue_id", "")))
        if key not in bm25: continue
        cands = bm25[key].get("bm25_candidates", bm25[key].get("candidates", []))[:args.top_k]
        if not gt or not cands: continue
        for g in gt:
            if g not in cands: cands.append(g)

        codes = [read_head(args.repo_dir, repo, c) for c in cands]
        prompts_real = [PROMPT.format(issue_text=issue, candidate_path=c, code=code)
                        for c, code in zip(cands, codes)]
        prompts_hash = [PROMPT.format(issue_text=issue, candidate_path=hash_path(c), code=code)
                        for c, code in zip(cands, codes)]
        s_real = np.array(score_batch(m, tok, prompts_real, yes_id, no_id, device, bs=args.batch_size))
        s_hash = np.array(score_batch(m, tok, prompts_hash, yes_id, no_id, device, bs=args.batch_size))
        residual = s_real - s_hash

        for a in alphas:
            fused = a * s_real + (1 - a) * residual
            ranked = sorted(zip(cands, fused.tolist()), key=lambda x: -x[1])
            hits[a].append(1.0 if ranked[0][0] in gt else 0.0)
        # s_hash alone (code-dominant): rank by s_hash
        ranked_h = sorted(zip(cands, s_hash.tolist()), key=lambda x: -x[1])
        hits_hash_only.append(1.0 if ranked_h[0][0] in gt else 0.0)

        if (i+1) % 20 == 0:
            print(f"  [{i+1}] real={np.mean(hits[1.0])*100:.1f} "
                  f"resid={np.mean(hits[0.0])*100:.1f} "
                  f"mid={np.mean(hits[0.5])*100:.1f} "
                  f"hash={np.mean(hits_hash_only)*100:.1f} "
                  f"({time.time()-start:.0f}s)", flush=True)

    summary = {f"alpha={a:.1f}": float(np.mean(hits[a])*100) for a in alphas}
    summary["hash_only_r1"] = float(np.mean(hits_hash_only)*100)
    summary["n"] = len(hits_hash_only)
    summary["best_alpha"] = max(alphas, key=lambda a: np.mean(hits[a]))
    summary["best_r1"] = max(summary[f"alpha={a:.1f}"] for a in alphas)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Summary ===\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
