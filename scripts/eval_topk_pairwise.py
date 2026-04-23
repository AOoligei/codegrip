#!/usr/bin/env python3
"""Top-K pairwise reranking + latency analysis.

Method: take BM25 top-K, do K(K-1)/2 pairwise comparisons per example,
rank by win count. Compare R@1 vs pointwise. Report avg latency.
"""
import argparse, json, os, random, time
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42); np.random.seed(42); torch.manual_seed(42)

PROMPT = ("Given the bug report, which file is more likely to need modification? Answer A or B.\n\n"
          "Bug Report: {issue_text}\n\nFile A: {fa}\nCode A:\n{ca}\n\n"
          "File B: {fb}\nCode B:\n{cb}\n\nAnswer:")


def read_head(repo_dir, repo, fpath, n=50):
    full = os.path.join(repo_dir, repo, fpath)
    if not os.path.isfile(full):
        return "# (unavailable)"
    try:
        with open(full, "r", errors="replace") as fh:
            return "".join(fh.readlines()[:n])[:1500]
    except Exception:
        return "# (unreadable)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/data/shuyang/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--lora_path", default="/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/best")
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--bm25_candidates", required=True)
    ap.add_argument("--repo_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--K_list", default="5,10,20")
    ap.add_argument("--max_examples", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"
    K_list = [int(x) for x in args.K_list.split(",")]

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(args.model_path, quantization_config=bnb,
                                              device_map={"": device}, trust_remote_code=True,
                                              torch_dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, args.lora_path); m.eval()
    a_id = tok.encode("A", add_special_tokens=False)[0]
    b_id = tok.encode("B", add_special_tokens=False)[0]

    data = [json.loads(l) for l in open(args.test_data)][:args.max_examples]
    bm25 = {}
    for l in open(args.bm25_candidates):
        r = json.loads(l); bm25[(r["repo"], str(r["issue_id"]))] = r
    print(f"Loaded {len(data)} examples; K_list={K_list}", flush=True)

    results = {K: {"hits": [], "latencies": [], "n_pairs": []} for K in K_list}
    K_max = max(K_list)

    for i, rec in enumerate(data):
        repo = rec.get("repo", "")
        key = (repo, str(rec.get("issue_id", "")))
        if key not in bm25: continue
        cands = bm25[key].get("bm25_candidates", [])[:K_max]
        if len(cands) < 2: continue
        gt = set(rec.get("changed_py_files", rec.get("changed_files", []))) or set(bm25[key].get("ground_truth", []))
        if not gt: continue
        issue = rec.get("issue_text", "")[:1000]

        codes = {c: read_head(args.repo_dir, repo, c) for c in cands}

        for K in K_list:
            cs = cands[:K]
            wins = {c: 0 for c in cs}
            t0 = time.time()
            n_p = 0
            for ii in range(len(cs)):
                for jj in range(ii+1, len(cs)):
                    a, b = cs[ii], cs[jj]
                    prompt = PROMPT.format(issue_text=issue, fa=a, fb=b,
                                            ca=codes[a], cb=codes[b])
                    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=1800).to(device)
                    with torch.no_grad():
                        logits = m(**inp).logits[:, -1, :]
                    s = (logits[0, a_id] - logits[0, b_id]).item()
                    if s > 0: wins[a] += 1
                    else: wins[b] += 1
                    n_p += 1
            elapsed = time.time() - t0
            ranked = sorted(cs, key=lambda c: -wins[c])
            hit = 1.0 if ranked[0] in gt else 0.0
            results[K]["hits"].append(hit)
            results[K]["latencies"].append(elapsed)
            results[K]["n_pairs"].append(n_p)
        if (i+1) % 10 == 0:
            for K in K_list:
                hh = results[K]["hits"]; lat = results[K]["latencies"]
                print(f"  [{i+1}] K={K}: R@1={np.mean(hh)*100:.1f}% lat={np.mean(lat):.2f}s/ex", flush=True)

    summary = {}
    for K in K_list:
        summary[f"K={K}"] = {
            "n": len(results[K]["hits"]),
            "R@1": float(np.mean(results[K]["hits"])*100) if results[K]["hits"] else 0.0,
            "avg_latency_s": float(np.mean(results[K]["latencies"])) if results[K]["latencies"] else 0.0,
            "avg_pairs": float(np.mean(results[K]["n_pairs"])) if results[K]["n_pairs"] else 0.0,
        }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Summary ===\n{json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
