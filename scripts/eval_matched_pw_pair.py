#!/usr/bin/env python3
"""
P4: Matched pointwise vs pairwise on the SAME (GT, hard_neg) pair.

For each test example:
  1. Pick GT and hard_neg (same logic as pairwise script)
  2. Score each independently with pointwise reranker (Yes/No logits)
  3. Pick higher score -> compare to GT
  4. Parallel: run the same pair through pairwise scorer

Output: matched pointwise acc vs pairwise acc. If they converge,
objective-mismatch claim loses. If pairwise >> matched pointwise,
objective-mismatch has legs independent of task difficulty.
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
BM25_PATH = "/home/chenlibin/grepo_agent/data/swebench_lite/swebench_perturb_shuffle_filenames_candidates.jsonl"
REPO_DIR = "/home/chenlibin/grepo_agent/data/swebench_lite/repos"

# Pointwise prompts (exact match to what the reranker was trained on style)
PW_PATH_PROMPT = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {file}\n\n"
    "Answer:"
)
PW_CODE_PROMPT = (
    "Given the bug report, is this file likely to need modification? "
    "Consider both the file path and code content.\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {file}\n"
    "Code:\n{code}\n\n"
    "Answer:"
)

# Pairwise prompts (already proven to give 80.3/86.0)
PAIR_PATH_PROMPT = (
    "Given the bug report, which file is more likely to need modification? "
    "Answer A or B.\n\n"
    "Bug Report: {issue_text}\n\n"
    "File A: {file_a}\n"
    "File B: {file_b}\n\n"
    "Answer:"
)
PAIR_CODE_PROMPT = (
    "Given the bug report, which file is more likely to need modification? "
    "Consider both the file paths and code content. Answer A or B.\n\n"
    "Bug Report: {issue_text}\n\n"
    "File A: {file_a}\n"
    "Code A:\n{code_a}\n\n"
    "File B: {file_b}\n"
    "Code B:\n{code_b}\n\n"
    "Answer:"
)


def read_file_head(repo, fpath, max_lines=50):
    full = os.path.join(REPO_DIR, repo, fpath)
    if not os.path.isfile(full):
        return "# (not available)"
    try:
        with open(full, "r", errors="replace") as f:
            return "".join(f.readlines()[:max_lines])
    except Exception:
        return "# (unreadable)"


def find_hard_negative(gt_file, candidates):
    gt_dir = os.path.dirname(gt_file)
    gt_stem = os.path.splitext(os.path.basename(gt_file))[0]
    for c in candidates:
        if c == gt_file: continue
        if os.path.dirname(c) == gt_dir: return c
    for c in candidates:
        if c == gt_file: continue
        cs = os.path.splitext(os.path.basename(c))[0]
        if cs == gt_stem or gt_stem in cs or cs in gt_stem: return c
    for c in candidates:
        if c != gt_file: return c
    return None


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


def pw_score(m, tok, prompt, yes_id, no_id, device):
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        logits = m(**inp).logits[:, -1, :]
    return (logits[0, yes_id] - logits[0, no_id]).item()


def pair_score(m, tok, prompt, a_id, b_id, device):
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        logits = m(**inp).logits[:, -1, :]
    return (logits[0, a_id] - logits[0, b_id]).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    test_data = [json.loads(l) for l in open(TEST_PATH)]
    bm25 = {}
    with open(BM25_PATH) as f:
        for line in f:
            r = json.loads(line); bm25[(r["repo"], str(r["issue_id"]))] = r
    print(f"Loaded {len(test_data)} test, {len(bm25)} bm25", flush=True)

    m, tok, yes_id, no_id, a_id, b_id = load_model(device)

    mpw_path, mpw_code, pair_path, pair_code = [], [], [], []
    start = time.time()

    for i, rec in enumerate(test_data):
        repo = rec.get("repo", "")
        issue = rec["issue_text"][:2000]
        gt_files = set(rec.get("changed_py_files", rec.get("changed_files", [])))
        if not gt_files: continue
        gt = list(gt_files)[0]

        key = (repo, str(rec.get("issue_id", "")))
        cands = []
        if key in bm25:
            cands = bm25[key].get("bm25_candidates", bm25[key].get("candidates", []))
        if not cands:
            rp = os.path.join(REPO_DIR, repo)
            if os.path.isdir(rp):
                for root, _, files in os.walk(rp):
                    for f in files:
                        if f.endswith(".py"):
                            cands.append(os.path.relpath(os.path.join(root, f), rp))
                            if len(cands) > 200: break

        neg = find_hard_negative(gt, cands)
        if neg is None: continue

        # Read code
        code_gt = read_file_head(repo, gt, 50)[:800]
        code_neg = read_file_head(repo, neg, 50)[:800]

        # Randomize A/B (must match pairwise script to stay comparable)
        if random.random() < 0.5:
            fa, fb, ca, cb = gt, neg, code_gt, code_neg; gt_is_a = True
        else:
            fa, fb, ca, cb = neg, gt, code_neg, code_gt; gt_is_a = False

        # --- Matched Pointwise PATH: score each independently, pick higher ---
        s_gt_p = pw_score(m, tok, PW_PATH_PROMPT.format(issue_text=issue, file=gt),
                          yes_id, no_id, device)
        s_neg_p = pw_score(m, tok, PW_PATH_PROMPT.format(issue_text=issue, file=neg),
                           yes_id, no_id, device)
        mpw_path.append(1.0 if s_gt_p > s_neg_p else 0.0)

        # --- Matched Pointwise CODE: path + 50L code ---
        s_gt_c = pw_score(m, tok, PW_CODE_PROMPT.format(issue_text=issue[:1000], file=gt, code=code_gt),
                          yes_id, no_id, device)
        s_neg_c = pw_score(m, tok, PW_CODE_PROMPT.format(issue_text=issue[:1000], file=neg, code=code_neg),
                           yes_id, no_id, device)
        mpw_code.append(1.0 if s_gt_c > s_neg_c else 0.0)

        # --- Pairwise PATH ---
        p_path = pair_score(m, tok, PAIR_PATH_PROMPT.format(issue_text=issue, file_a=fa, file_b=fb),
                            a_id, b_id, device)
        pair_path.append(1.0 if ((p_path > 0) == gt_is_a) else 0.0)

        # --- Pairwise CODE ---
        p_code = pair_score(m, tok, PAIR_CODE_PROMPT.format(issue_text=issue[:1000],
                                                              file_a=fa, file_b=fb,
                                                              code_a=ca, code_b=cb),
                            a_id, b_id, device)
        pair_code.append(1.0 if ((p_code > 0) == gt_is_a) else 0.0)

        if (i+1) % 20 == 0:
            print(f"  [{i+1}] mPw_path={np.mean(mpw_path)*100:.1f}% "
                  f"mPw_code={np.mean(mpw_code)*100:.1f}% "
                  f"Pair_path={np.mean(pair_path)*100:.1f}% "
                  f"Pair_code={np.mean(pair_code)*100:.1f}% "
                  f"({time.time()-start:.0f}s)", flush=True)

    n = len(mpw_path)
    summary = {
        "n": n,
        "matched_pw_path": float(np.mean(mpw_path)*100),
        "matched_pw_code": float(np.mean(mpw_code)*100),
        "pairwise_path": float(np.mean(pair_path)*100),
        "pairwise_code": float(np.mean(pair_code)*100),
        "delta_pair_vs_mpw_path": float((np.mean(pair_path) - np.mean(mpw_path))*100),
        "delta_pair_vs_mpw_code": float((np.mean(pair_code) - np.mean(mpw_code))*100),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Matched Pw vs Pair (n={n}) ===")
    print(f"  matched Pw path: {summary['matched_pw_path']:.2f}%")
    print(f"  matched Pw code: {summary['matched_pw_code']:.2f}%")
    print(f"  pairwise   path: {summary['pairwise_path']:.2f}%")
    print(f"  pairwise   code: {summary['pairwise_code']:.2f}%")
    print(f"  Δ path (pair − matched-pw) = {summary['delta_pair_vs_mpw_path']:+.2f}pp")
    print(f"  Δ code (pair − matched-pw) = {summary['delta_pair_vs_mpw_code']:+.2f}pp")


if __name__ == "__main__":
    main()
