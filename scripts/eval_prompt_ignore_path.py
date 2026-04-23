#!/usr/bin/env python3
"""Test prompt-engineering: tell model to ignore path, focus on code.
Compare to baseline path-only and code-aware prompts. Training-free."""
import argparse, json, os, time
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random; random.seed(42); np.random.seed(42); torch.manual_seed(42)

MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
LORA_PATH = "/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/best"
TEST_PATH = "/home/chenlibin/grepo_agent/data/swebench_lite/swebench_lite_test.jsonl"
BM25_PATH = "/home/chenlibin/grepo_agent/data/rankft/swebench_bm25_final_top500.jsonl"
REPO_DIR = "/home/chenlibin/grepo_agent/data/swebench_lite/repos"

PROMPTS = {
    "baseline_path":
        "Given the bug report, is this file likely to need modification?\n\n"
        "Bug Report: {issue_text}\n\nFile: {file}\n\nAnswer:",
    "baseline_code":
        "Given the bug report, is this file likely to need modification? "
        "Consider both the file path and code content.\n\n"
        "Bug Report: {issue_text}\n\nFile: {file}\nCode:\n{code}\n\nAnswer:",
    "ignore_path":
        "IGNORE the file path. Look ONLY at the code below.\n"
        "Does this code implement the functionality described in the bug report?\n\n"
        "Bug Report: {issue_text}\n\nFile: {file}\nCode:\n{code}\n\nAnswer:",
    "code_first":
        "Read the code carefully. The file path is just a label.\n"
        "Based primarily on the code, judge whether modification is needed.\n\n"
        "Bug Report: {issue_text}\n\nCode:\n{code}\n\n(Path: {file})\n\nAnswer:",
}


def read_head(repo, fpath, n=50):
    full = os.path.join(REPO_DIR, repo, fpath)
    if not os.path.isfile(full):
        return "# (not available)"
    try:
        with open(full, "r", errors="replace") as f:
            return "".join(f.readlines()[:n])[:800]
    except Exception:
        return "# (unreadable)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--K", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dev = f"cuda:{args.gpu_id}"

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
            device_map={"": dev}, trust_remote_code=True, torch_dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, LORA_PATH); m.eval()
    yid = tok.encode("Yes", add_special_tokens=False)[0]
    nid = tok.encode("No", add_special_tokens=False)[0]

    def score(prompts, bs=8):
        out = []
        for i in range(0, len(prompts), bs):
            inp = tok(prompts[i:i+bs], return_tensors="pt", padding=True, truncation=True,
                      max_length=1024, padding_side="left").to(dev)
            with torch.no_grad():
                logits = m(**inp).logits[:, -1, :]
            s = (logits[:, yid].float() - logits[:, nid].float()).cpu().numpy()
            out.extend(s.tolist())
        return out

    test = [json.loads(l) for l in open(TEST_PATH)]
    bm25 = {}
    with open(BM25_PATH) as f:
        for l in f:
            r = json.loads(l); bm25[(r["repo"], str(r["issue_id"]))] = r

    hits = {k: [] for k in PROMPTS}
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
        codes = {c: read_head(repo, c, 50) for c in cands}

        for name, tpl in PROMPTS.items():
            prompts = [tpl.format(issue_text=issue if "code" not in tpl else issue[:1000],
                                   file=c, code=codes[c]) for c in cands]
            scores = score(prompts)
            top = cands[int(np.argmax(scores))]
            hits[name].append(1.0 if top in gt else 0.0)

        if (i+1) % 20 == 0:
            line = " ".join(f"{k}={np.mean(v)*100:.1f}" for k, v in hits.items() if v)
            print(f"  [{i+1}] {line} ({time.time()-start:.0f}s)", flush=True)

    print(f"\n=== Prompt variants R@1 (n={len(hits['baseline_path'])}) ===")
    summary = {k: float(np.mean(v)*100) for k, v in hits.items() if v}
    for k, v in summary.items():
        print(f"  {k:20s}: {v:.2f}%")
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"n": len(hits["baseline_path"]), "strategies": summary}, f, indent=2)

if __name__ == "__main__":
    main()
