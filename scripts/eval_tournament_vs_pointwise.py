#!/usr/bin/env python3
"""
Pairwise tournament ranking vs pointwise ranking on the same BM25 top-K pool.

For each test example:
  - Take BM25 top-K candidates
  - Pointwise: score each independently, rank, check if top-1 is GT
  - Tournament: for all K(K-1)/2 unordered pairs, randomized A/B,
    score pairwise, accumulate wins per candidate, rank by wins

Compare R@1 between the two scoring strategies on the SAME pool.

If tournament >> pointwise, the objective-mismatch story can be
reframed as an inference-time property:
  "pointwise ranking cannot fully leverage code; pairwise
   aggregation extracts additional signal."
"""
import argparse, json, os, random, time, itertools
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

PW_PATH_PROMPT = ("Given the bug report, is this file likely to need modification?\n\n"
                  "Bug Report: {issue_text}\n\nFile: {file}\n\nAnswer:")
PW_CODE_PROMPT = ("Given the bug report, is this file likely to need modification? "
                  "Consider both the file path and code content.\n\n"
                  "Bug Report: {issue_text}\n\nFile: {file}\nCode:\n{code}\n\nAnswer:")
PAIR_PATH_PROMPT = ("Given the bug report, which file is more likely to need modification? "
                    "Answer A or B.\n\nBug Report: {issue_text}\n\n"
                    "File A: {file_a}\nFile B: {file_b}\n\nAnswer:")
PAIR_CODE_PROMPT = ("Given the bug report, which file is more likely to need modification? "
                    "Consider both the file paths and code content. Answer A or B.\n\n"
                    "Bug Report: {issue_text}\n\n"
                    "File A: {file_a}\nCode A:\n{code_a}\n\n"
                    "File B: {file_b}\nCode B:\n{code_b}\n\nAnswer:")


def read_head(repo, fpath, n=50):
    full = os.path.join(REPO_DIR, repo, fpath)
    if not os.path.isfile(full):
        return "# (not available)"
    try:
        with open(full, "r", errors="replace") as f:
            return "".join(f.readlines()[:n])[:800]
    except Exception:
        return "# (unreadable)"


def load_model(device):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
                                              device_map={"": device},
                                              trust_remote_code=True,
                                              torch_dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, LORA_PATH); m.eval()
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    a_id = tok.encode("A", add_special_tokens=False)[0]
    b_id = tok.encode("B", add_special_tokens=False)[0]
    return m, tok, yes_id, no_id, a_id, b_id


def pw_score(m, tok, prompt, yes_id, no_id, device, max_len=1800):
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        logits = m(**inp).logits[:, -1, :]
    return (logits[0, yes_id] - logits[0, no_id]).item()


def pair_score(m, tok, prompt, a_id, b_id, device, max_len=2048):
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        logits = m(**inp).logits[:, -1, :]
    return (logits[0, a_id] - logits[0, b_id]).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--variant", choices=["path", "code", "both"], default="both")
    ap.add_argument("--max_examples", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"
    K = args.K

    test_data = [json.loads(l) for l in open(TEST_PATH)][:args.max_examples]
    bm25 = {}
    with open(BM25_PATH) as f:
        for line in f:
            r = json.loads(line); bm25[(r["repo"], str(r["issue_id"]))] = r
    print(f"Loaded {len(test_data)} test, K={K}, variant={args.variant}", flush=True)

    m, tok, yes_id, no_id, a_id, b_id = load_model(device)

    pw_path_hits, pw_code_hits, t_path_hits, t_code_hits = [], [], [], []
    start = time.time()

    for idx, rec in enumerate(test_data):
        repo = rec.get("repo", "")
        issue = rec["issue_text"][:1500]
        gt_files = set(rec.get("changed_py_files", rec.get("changed_files", [])))
        if not gt_files: continue

        key = (repo, str(rec.get("issue_id", "")))
        if key not in bm25: continue
        cands_all = bm25[key].get("bm25_candidates", [])
        if len(cands_all) < K: continue
        cands = cands_all[:K]
        # Ensure at least one GT in the K candidates (oracle recall control)
        gt_in = [g for g in gt_files if g in cands]
        if not gt_in: continue

        codes = {c: read_head(repo, c, 50) for c in cands}

        if args.variant in ("path", "both"):
            # Pointwise PATH
            scores = [pw_score(m, tok, PW_PATH_PROMPT.format(issue_text=issue, file=c),
                               yes_id, no_id, device) for c in cands]
            pw_top = cands[int(np.argmax(scores))]
            pw_path_hits.append(1.0 if pw_top in gt_files else 0.0)

            # Tournament PATH
            wins = {c: 0 for c in cands}
            for i, j in itertools.combinations(range(K), 2):
                a_idx, b_idx = (i, j) if random.random() < 0.5 else (j, i)
                s = pair_score(m, tok, PAIR_PATH_PROMPT.format(
                    issue_text=issue, file_a=cands[a_idx], file_b=cands[b_idx]),
                    a_id, b_id, device)
                winner = cands[a_idx] if s > 0 else cands[b_idx]
                wins[winner] += 1
            t_top = max(cands, key=lambda c: wins[c])
            t_path_hits.append(1.0 if t_top in gt_files else 0.0)

        if args.variant in ("code", "both"):
            # Pointwise CODE
            scores = [pw_score(m, tok, PW_CODE_PROMPT.format(
                issue_text=issue[:1000], file=c, code=codes[c]),
                yes_id, no_id, device) for c in cands]
            pw_top = cands[int(np.argmax(scores))]
            pw_code_hits.append(1.0 if pw_top in gt_files else 0.0)

            # Tournament CODE
            wins = {c: 0 for c in cands}
            for i, j in itertools.combinations(range(K), 2):
                a_idx, b_idx = (i, j) if random.random() < 0.5 else (j, i)
                s = pair_score(m, tok, PAIR_CODE_PROMPT.format(
                    issue_text=issue[:1000],
                    file_a=cands[a_idx], file_b=cands[b_idx],
                    code_a=codes[cands[a_idx]], code_b=codes[cands[b_idx]]),
                    a_id, b_id, device)
                winner = cands[a_idx] if s > 0 else cands[b_idx]
                wins[winner] += 1
            t_top = max(cands, key=lambda c: wins[c])
            t_code_hits.append(1.0 if t_top in gt_files else 0.0)

        if (idx+1) % 10 == 0:
            msg = f"  [{idx+1}] "
            if pw_path_hits: msg += f"pw_path={np.mean(pw_path_hits)*100:.1f}% "
            if t_path_hits: msg += f"tour_path={np.mean(t_path_hits)*100:.1f}% "
            if pw_code_hits: msg += f"pw_code={np.mean(pw_code_hits)*100:.1f}% "
            if t_code_hits: msg += f"tour_code={np.mean(t_code_hits)*100:.1f}% "
            msg += f"({time.time()-start:.0f}s)"
            print(msg, flush=True)

    def pct(x): return float(np.mean(x)*100) if x else 0.0

    summary = {
        "K": K, "n": len(pw_path_hits) or len(pw_code_hits),
        "pointwise_path_r1":   pct(pw_path_hits),
        "tournament_path_r1":  pct(t_path_hits),
        "pointwise_code_r1":   pct(pw_code_hits),
        "tournament_code_r1":  pct(t_code_hits),
        "delta_path": pct(t_path_hits) - pct(pw_path_hits),
        "delta_code": pct(t_code_hits) - pct(pw_code_hits),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Tournament vs Pointwise (K={K}, n={summary['n']}) ===")
    print(f"  Pointwise  path: {summary['pointwise_path_r1']:.2f}%")
    print(f"  Tournament path: {summary['tournament_path_r1']:.2f}%  Δ={summary['delta_path']:+.2f}pp")
    print(f"  Pointwise  code: {summary['pointwise_code_r1']:.2f}%")
    print(f"  Tournament code: {summary['tournament_code_r1']:.2f}%  Δ={summary['delta_code']:+.2f}pp")


if __name__ == "__main__":
    main()
