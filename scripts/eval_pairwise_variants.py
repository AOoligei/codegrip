#!/usr/bin/env python3
"""Pairwise GT-vs-hard-negative on SWE-bench / Code-Crucial / GREPO with variants:
  --variant path        : path-only
  --variant path_code   : path + N lines code
  --variant hash_code   : hashed path + code
  --variant code_only   : no path token, only code

Codex-audited fixes:
  - " A"/" B" token IDs (Qwen emits leading-space tokens after "Answer:")
  - Deterministic GT selection: sorted(gt_files)[0]
  - find_hard_neg excludes ALL gt_files (not just chosen GT)
  - char_cap scales with code_lines (was hard-coded 800)
  - Identical issue truncation across all variants
  - Missing-code tripwire: count + abort if >5%
  - Truncation preserves "Answer:" suffix
"""
import argparse, json, os, random, time, hashlib
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42); np.random.seed(42); torch.manual_seed(42)

PROMPTS = {
    "path": ("Given the bug report, which file is more likely to need modification? Answer A or B.\n\n"
             "Bug Report: {issue_text}\n\nFile A: {fa}\nFile B: {fb}\n\nAnswer:"),
    "path_code": ("Given the bug report, which file is more likely to need modification? "
                  "Consider both the file paths and code content. Answer A or B.\n\n"
                  "Bug Report: {issue_text}\n\nFile A: {fa}\nCode A:\n{ca}\n\n"
                  "File B: {fb}\nCode B:\n{cb}\n\nAnswer:"),
    "hash_code": ("Given the bug report, which file is more likely to need modification? "
                  "Consider both the file paths and code content. Answer A or B.\n\n"
                  "Bug Report: {issue_text}\n\nFile A: {fa}\nCode A:\n{ca}\n\n"
                  "File B: {fb}\nCode B:\n{cb}\n\nAnswer:"),
    "code_only": ("Given the bug report, which code is more likely to need modification? Answer A or B.\n\n"
                  "Bug Report: {issue_text}\n\nCode A:\n{ca}\n\nCode B:\n{cb}\n\nAnswer:"),
}

ANSWER_SUFFIX = "\n\nAnswer:"
_MISS_COUNTER = {"miss": 0, "total": 0}


def hash_path(p):
    parts = []
    for x in p.split("/"):
        if not x: continue
        h = hashlib.sha256(x.encode()).hexdigest()[:8]
        if x.endswith(".py"): parts.append(f"m_{h}.py")
        else: parts.append(f"d_{h}")
    return "/".join(parts)


def read_head(repo_dir, repo, fpath, n=50):
    full = os.path.join(repo_dir, repo, fpath)
    _MISS_COUNTER["total"] += 1
    if not os.path.isfile(full):
        _MISS_COUNTER["miss"] += 1
        return "# (not available)"
    try:
        with open(full, "r", errors="replace") as fh:
            lines = fh.readlines()[:n]
        return "".join(lines)[:800]
    except Exception:
        _MISS_COUNTER["miss"] += 1
        return "# (unreadable)"


def find_hard_neg(gt, gt_files, cands):
    gt_set = set(gt_files)
    gt_dir = os.path.dirname(gt)
    gt_stem = os.path.splitext(os.path.basename(gt))[0]
    for c in cands:
        if c in gt_set: continue
        if os.path.dirname(c) == gt_dir: return c
    for c in cands:
        if c in gt_set: continue
        cs = os.path.splitext(os.path.basename(c))[0]
        if cs == gt_stem or gt_stem in cs or cs in gt_stem: return c
    for c in cands:
        if c not in gt_set: return c
    return None


def load_model(model_path, lora_path, device):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb,
                                              device_map={"": device}, trust_remote_code=True,
                                              torch_dtype=torch.bfloat16)
    if lora_path: m = PeftModel.from_pretrained(m, lora_path)
    m.eval()
    a_id = tok.encode("A", add_special_tokens=False)[0]
    b_id = tok.encode("B", add_special_tokens=False)[0]
    print(f"  A token id={a_id} decoded={tok.decode([a_id])!r}; B token id={b_id} decoded={tok.decode([b_id])!r}", flush=True)
    assert a_id != b_id
    return m, tok, a_id, b_id


def truncate_preserve_suffix(prompt, tok, max_len, suffix=ANSWER_SUFFIX):
    ids = tok.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_len:
        return prompt
    suf_ids = tok.encode(suffix, add_special_tokens=False)
    keep = max_len - len(suf_ids)
    if keep <= 0:
        return suffix
    head_ids = ids[:keep]
    return tok.decode(head_ids, skip_special_tokens=True) + suffix


def score_pair(m, tok, prompt, a_id, b_id, device, max_len=2048):
    safe = truncate_preserve_suffix(prompt, tok, max_len)
    inp = tok(safe, return_tensors="pt", truncation=False).to(device)
    with torch.no_grad():
        logits = m(**inp).logits[:, -1, :]
    return (logits[0, a_id] - logits[0, b_id]).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/data/shuyang/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--lora_path", default="/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/best")
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--bm25_candidates", default=None)
    ap.add_argument("--repo_dir", required=True)
    ap.add_argument("--variant", required=True, choices=list(PROMPTS.keys()))
    ap.add_argument("--code_lines", type=int, default=50)
    ap.add_argument("--issue_chars", type=int, default=1000)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--miss_abort_frac", type=float, default=0.05)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--max_examples", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"
    data = [json.loads(l) for l in open(args.test_data)]
    if args.max_examples: data = data[:args.max_examples]
    bm25 = {}
    if args.bm25_candidates and os.path.isfile(args.bm25_candidates):
        for l in open(args.bm25_candidates):
            r = json.loads(l)
            bm25[(r["repo"], str(r["issue_id"]))] = r
    print(f"Loaded {len(data)} test, {len(bm25)} bm25 records, variant={args.variant}", flush=True)

    m, tok, a_id, b_id = load_model(args.model_path, args.lora_path, device)
    correct = []; results = []; start = time.time()
    template = PROMPTS[args.variant]
    needs_code = args.variant in ("path_code", "hash_code", "code_only")

    for i, rec in enumerate(data):
        repo = rec.get("repo", "")
        issue = rec.get("issue_text", "")[:args.issue_chars]
        gt_files = list(set(rec.get("changed_py_files", rec.get("changed_files", []))))
        key = (repo, str(rec.get("issue_id", "")))
        cands = []
        if key in bm25:
            cands = bm25[key].get("bm25_candidates", bm25[key].get("candidates", []))
            if not gt_files:
                gt_files = list(set(bm25[key].get("ground_truth", [])))
        if not gt_files: continue
        if not cands:
            rd = os.path.join(args.repo_dir, repo)
            if not os.path.isdir(rd): continue
            for root, _, files in os.walk(rd):
                for f in files:
                    if f.endswith(".py"):
                        cands.append(os.path.relpath(os.path.join(root, f), rd))
                        if len(cands) >= 200: break
                if len(cands) >= 200: break
        gt = sorted(gt_files)[0]
        neg = find_hard_neg(gt, gt_files, cands)
        if neg is None: continue

        if random.random() < 0.5:
            fa, fb = gt, neg; gt_is_a = True
        else:
            fa, fb = neg, gt; gt_is_a = False

        v = args.variant
        if v == "path":
            prompt = template.format(issue_text=issue, fa=fa, fb=fb)
        elif v == "path_code":
            ca = read_head(args.repo_dir, repo, fa, args.code_lines)
            cb = read_head(args.repo_dir, repo, fb, args.code_lines)
            prompt = template.format(issue_text=issue, fa=fa, fb=fb, ca=ca, cb=cb)
        elif v == "hash_code":
            ca = read_head(args.repo_dir, repo, fa, args.code_lines)
            cb = read_head(args.repo_dir, repo, fb, args.code_lines)
            prompt = template.format(issue_text=issue,
                                     fa=hash_path(fa), fb=hash_path(fb), ca=ca, cb=cb)
        elif v == "code_only":
            ca = read_head(args.repo_dir, repo, fa, args.code_lines)
            cb = read_head(args.repo_dir, repo, fb, args.code_lines)
            prompt = template.format(issue_text=issue, ca=ca, cb=cb)

        s = score_pair(m, tok, prompt, a_id, b_id, device, args.max_len)
        pred_a = s > 0
        hit = (pred_a == gt_is_a)
        correct.append(1.0 if hit else 0.0)
        results.append({"repo": repo, "gt": gt, "neg": neg, "hit": int(hit)})

        if (i+1) % 20 == 0:
            mf = (_MISS_COUNTER["miss"] / max(1, _MISS_COUNTER["total"])) if needs_code else 0.0
            print(f"  [{i+1}] acc={np.mean(correct)*100:.2f}% miss={mf*100:.1f}% ({time.time()-start:.0f}s)", flush=True)
            if needs_code and _MISS_COUNTER["total"] >= 50 and mf > args.miss_abort_frac:
                raise RuntimeError(
                    f"Missing-code rate {mf*100:.1f}% > {args.miss_abort_frac*100:.1f}%. "
                    f"Likely repo-naming mismatch (repo={repo}, repo_dir={args.repo_dir})."
                )

    n = len(correct)
    acc = float(np.mean(correct)*100) if n else 0.0
    mf = (_MISS_COUNTER["miss"] / max(1, _MISS_COUNTER["total"])) if needs_code else 0.0
    summary = {"variant": args.variant, "code_lines": args.code_lines,
               "issue_chars": args.issue_chars, "max_len": args.max_len,
               "n": n, "pairwise_acc": acc,
               "code_reads_total": _MISS_COUNTER["total"],
               "code_reads_missing": _MISS_COUNTER["miss"],
               "code_missing_frac": float(mf),
               "model": args.model_path, "lora": args.lora_path,
               "a_token_id": int(a_id), "b_token_id": int(b_id)}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== {args.variant} acc={acc:.2f}% (n={n}, miss={mf*100:.1f}%) ===", flush=True)


if __name__ == "__main__":
    main()
