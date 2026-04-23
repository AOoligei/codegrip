#!/usr/bin/env python3
"""Full reranking R@1/5/10/MRR on BM25 top-K. Pointwise (' Yes'/' No') OR pairwise.

Codex audit v2 fixes:
  - Stable seed via hashlib.sha256 (was: hash() process-randomized)
  - Per-side token budget for ca/cb so File B never truncated
  - --hash_paths now applied in pointwise mode too
  - Assert each scoring token is exactly 1 token
  - Order-bias randomization in pairwise via stable per-pair hash
  - Skipped-example counters reported
  - Oracle Hit@K reported (BM25 ceiling)
  - Tie-breaking by sum-of-margins (not BM25 order)
  - Truncation rate logged
"""
import argparse, json, os, random, time, hashlib
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42); np.random.seed(42); torch.manual_seed(42)

POINT_PROMPT = ("Given the bug report, is this file likely to need modification?\n\n"
                "Bug Report: {issue_text}\n\nFile: {fa}\n\nAnswer:")
POINT_CODE_PROMPT = ("Given the bug report, is this file likely to need modification? "
                     "Consider both the file path and code content.\n\n"
                     "Bug Report: {issue_text}\n\nFile: {fa}\nCode:\n{ca}\n\nAnswer:")
PAIR_PROMPT = ("Given the bug report, which file is more likely to need modification? Answer A or B.\n\n"
               "Bug Report: {issue_text}\n\nFile A: {fa}\nFile B: {fb}\n\nAnswer:")
PAIR_CODE_PROMPT = ("Given the bug report, which file is more likely to need modification? "
                    "Consider both the file paths and code content. Answer A or B.\n\n"
                    "Bug Report: {issue_text}\n\nFile A: {fa}\nCode A:\n{ca}\n\n"
                    "File B: {fb}\nCode B:\n{cb}\n\nAnswer:")

ANSWER_SUFFIX = "\n\nAnswer:"
_MISS = {"miss": 0, "total": 0}
_TRUNC = {"truncated": 0, "total": 0}
_SKIP = {"no_bm25": 0, "no_gt": 0, "few_cands": 0, "no_cand_after_dedup": 0}


def stable_seed(*parts):
    s = "\n".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)


def hash_path(p):
    parts = []
    for x in p.split("/"):
        if not x: continue
        h = hashlib.sha256(x.encode()).hexdigest()[:16]
        if x.endswith(".py"): parts.append(f"m_{h}.py")
        else: parts.append(f"d_{h}")
    return "/".join(parts)


def read_head(repo_dir, repo, fpath, n=50):
    full = os.path.join(repo_dir, repo, fpath)
    _MISS["total"] += 1
    if not os.path.isfile(full):
        _MISS["miss"] += 1
        return "# (not available)"
    try:
        with open(full, "r", errors="replace") as fh:
            lines = fh.readlines()[:n]
        char_cap = max(800, n * 80)
        return "".join(lines)[:char_cap]
    except Exception:
        _MISS["miss"] += 1
        return "# (unreadable)"


def truncate_safe(prompt, tok, max_len, suffix=ANSWER_SUFFIX):
    """Truncate prompt to max_len, ALWAYS preserving the suffix."""
    ids = tok.encode(prompt, add_special_tokens=False)
    _TRUNC["total"] += 1
    if len(ids) <= max_len:
        return prompt
    _TRUNC["truncated"] += 1
    suf_ids = tok.encode(suffix, add_special_tokens=False)
    keep = max_len - len(suf_ids)
    if keep <= 0:
        return suffix
    return tok.decode(ids[:keep], skip_special_tokens=False) + suffix


def truncate_per_side_pair_code(tok, issue, fa, fb, ca, cb, max_len,
                                 template, suffix=ANSWER_SUFFIX):
    """Build PAIR_CODE_PROMPT but token-budget ca/cb individually so BOTH survive."""
    # Compute frame size with empty code
    frame_prompt = template.format(issue_text=issue, fa=fa, fb=fb, ca="", cb="")
    suf_ids = tok.encode(suffix, add_special_tokens=False)
    frame_ids = tok.encode(frame_prompt, add_special_tokens=False)
    # Budget: total - frame - safety margin
    budget = max_len - len(frame_ids) - 32  # safety margin for whitespace/joins
    if budget < 200:
        # frame alone is too big; fall back to standard truncation (rare)
        full = template.format(issue_text=issue, fa=fa, fb=fb, ca=ca, cb=cb)
        return truncate_safe(full, tok, max_len, suffix)
    per_side = budget // 2
    ca_ids = tok.encode(ca, add_special_tokens=False)[:per_side]
    cb_ids = tok.encode(cb, add_special_tokens=False)[:per_side]
    ca_t = tok.decode(ca_ids, skip_special_tokens=False)
    cb_t = tok.decode(cb_ids, skip_special_tokens=False)
    out = template.format(issue_text=issue, fa=fa, fb=fb, ca=ca_t, cb=cb_t)
    _TRUNC["total"] += 1
    if len(ca_ids) < len(tok.encode(ca, add_special_tokens=False)) or \
       len(cb_ids) < len(tok.encode(cb, add_special_tokens=False)):
        _TRUNC["truncated"] += 1
    # Final enforce: re-encode and tail-truncate if still over budget
    out_ids = tok.encode(out, add_special_tokens=False)
    if len(out_ids) > max_len:
        keep = max_len - len(suf_ids)
        if keep <= 0:
            return suffix
        out = tok.decode(out_ids[:keep], skip_special_tokens=False) + suffix
    return out


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
    # Strict: each scoring word must be exactly 1 token
    def must_one(s, label):
        ids = tok.encode(s, add_special_tokens=False)
        assert len(ids) == 1, f"{label}: {s!r} tokenizes to {len(ids)} tokens: {ids}"
        return ids[0]
    # Training convention (train_rankft.py get_yes_no_token_ids): standalone "Yes"/"No"
    # WITHOUT leading space. The LoRA pushes up logit[yes_id] where yes_id = 9454 for
    # Qwen2.5. Eval must read the SAME token ids, even though the natural continuation
    # of "Answer:" is " Yes" (7414). Previous versions of this script read 7414, which
    # silently invalidated all R@1 numbers because the LoRA had never been trained to
    # raise that logit.
    yes_id = must_one("Yes", "yes")
    no_id  = must_one("No",  "no")
    # A/B tokens for pairwise are still read space-prefixed since pairwise training is
    # not relevant here (pairwise models in this repo use " A"/" B").
    a_id   = must_one(" A",   "a")
    b_id   = must_one(" B",   "b")
    print(f"  yes={yes_id}({tok.decode([yes_id])!r}) no={no_id}({tok.decode([no_id])!r}) "
          f"a={a_id}({tok.decode([a_id])!r}) b={b_id}({tok.decode([b_id])!r})", flush=True)
    return m, tok, yes_id, no_id, a_id, b_id


def score_one(m, tok, prompt, max_len, device, t1, t2, do_truncate=True):
    safe = truncate_safe(prompt, tok, max_len) if do_truncate else prompt
    inp = tok(safe, return_tensors="pt", truncation=False, add_special_tokens=False).to(device)
    with torch.no_grad():
        logits = m(**inp).logits[0, -1, :]
    return (logits[t1] - logits[t2]).item()


def rank_pointwise(m, tok, issue, repo, repo_dir, cands, code_mode, code_lines,
                   max_len, device, yes_id, no_id, hash_paths=False):
    scores = []
    for c in cands:
        c_disp = hash_path(c) if hash_paths else c
        if code_mode:
            code = read_head(repo_dir, repo, c, code_lines)
            prompt = POINT_CODE_PROMPT.format(issue_text=issue, fa=c_disp, ca=code)
        else:
            prompt = POINT_PROMPT.format(issue_text=issue, fa=c_disp)
        scores.append(score_one(m, tok, prompt, max_len, device, yes_id, no_id))
    # Tie-break by score margin (already the case)
    ranked = [c for c, _ in sorted(zip(cands, scores), key=lambda x: -x[1])]
    return ranked


def rank_pairwise(m, tok, issue, repo, repo_dir, cands, code_mode, code_lines,
                  max_len, device, a_id, b_id, n_opponents, ex_seed,
                  hash_paths=False):
    rng = random.Random(ex_seed)
    K = len(cands)
    n_opp = min(n_opponents, K - 1)
    code_cache = {}
    if code_mode:
        for c in cands:
            code_cache[c] = read_head(repo_dir, repo, c, code_lines)
    wins = {c: 0 for c in cands}
    margin = {c: 0.0 for c in cands}  # tiebreak by total signed margin
    for c in cands:
        opps = rng.sample([o for o in cands if o != c], n_opp)
        for opp in opps:
            # Stable per-pair order randomization to remove A/B position bias
            order_seed = stable_seed("order", c, opp)
            if order_seed % 2 == 0:
                fa, fb, gt_is_a = c, opp, True
            else:
                fa, fb, gt_is_a = opp, c, False
            fa_disp = hash_path(fa) if hash_paths else fa
            fb_disp = hash_path(fb) if hash_paths else fb
            if code_mode:
                ca = code_cache[fa]; cb = code_cache[fb]
                prompt = truncate_per_side_pair_code(
                    tok, issue, fa_disp, fb_disp, ca, cb, max_len,
                    PAIR_CODE_PROMPT)
                s = score_one(m, tok, prompt, max_len, device, a_id, b_id, do_truncate=False)
            else:
                prompt = PAIR_PROMPT.format(issue_text=issue, fa=fa_disp, fb=fb_disp)
                s = score_one(m, tok, prompt, max_len, device, a_id, b_id)
            # s>0 means A wins. Map back to {c, opp}.
            c_wins = (s > 0) == gt_is_a
            if c_wins:
                wins[c] += 1; margin[c] += abs(s); margin[opp] -= abs(s)
            else:
                wins[opp] += 1; margin[opp] += abs(s); margin[c] -= abs(s)
    # Sort by wins, tiebreak by margin
    ranked = sorted(cands, key=lambda c: (-wins[c], -margin[c]))
    return ranked


def metrics(ranked, gt_set):
    r1 = 1.0 if ranked and ranked[0] in gt_set else 0.0
    r5 = 1.0 if any(c in gt_set for c in ranked[:5]) else 0.0
    r10 = 1.0 if any(c in gt_set for c in ranked[:10]) else 0.0
    mrr = 0.0
    for i, c in enumerate(ranked, 1):
        if c in gt_set:
            mrr = 1.0 / i; break
    return r1, r5, r10, mrr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/data/shuyang/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--lora_path", default="/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/best")
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--bm25_candidates", required=True)
    ap.add_argument("--repo_dir", required=True)
    ap.add_argument("--objective", required=True, choices=["pointwise", "pairwise"])
    ap.add_argument("--code_mode", action="store_true")
    ap.add_argument("--hash_paths", action="store_true")
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--n_opponents", type=int, default=5)
    ap.add_argument("--code_lines", type=int, default=50)
    ap.add_argument("--issue_chars", type=int, default=1000)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--miss_abort_frac", type=float, default=0.05)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"
    data = [json.loads(l) for l in open(args.test_data)]
    if args.max_examples: data = data[:args.max_examples]
    bm25 = {}
    with open(args.bm25_candidates) as f:
        for l in f:
            r = json.loads(l)
            bm25[(r["repo"], str(r["issue_id"]))] = r
    print(f"Loaded {len(data)} test, {len(bm25)} bm25, obj={args.objective}, "
          f"code={args.code_mode}, hash={args.hash_paths}, K={args.top_k}, "
          f"opp={args.n_opponents}", flush=True)

    m, tok, yes_id, no_id, a_id, b_id = load_model(args.model_path, args.lora_path, device)
    R1=[]; R5=[]; R10=[]; MRR=[]; ORACLE=[]; start=time.time()

    for i, rec in enumerate(data):
        repo = rec.get("repo", "")
        issue = rec.get("issue_text", "")[:args.issue_chars]
        gt_files = list(set(rec.get("changed_py_files", rec.get("changed_files", []))))
        key = (repo, str(rec.get("issue_id", "")))
        if key not in bm25:
            _SKIP["no_bm25"] += 1; continue
        # De-dupe candidates preserving order
        seen = set(); cands = []
        # Support both 'bm25_candidates' (SWE) and 'candidates' (GREPO) key names
        cand_list = bm25[key].get("bm25_candidates") or bm25[key].get("candidates", [])
        for c in cand_list:
            if c not in seen:
                seen.add(c); cands.append(c)
            if len(cands) >= args.top_k: break
        if not gt_files:
            gt_files = list(set(bm25[key].get("ground_truth", [])))
        if not gt_files:
            _SKIP["no_gt"] += 1; continue
        if len(cands) < 2:
            _SKIP["few_cands"] += 1; continue
        gt_set = set(gt_files)
        # Oracle: any GT in cands?
        ORACLE.append(1.0 if any(c in gt_set for c in cands) else 0.0)
        ex_seed = stable_seed("ex", repo, rec.get("issue_id", ""))

        if args.objective == "pointwise":
            ranked = rank_pointwise(m, tok, issue, repo, args.repo_dir, cands,
                                    args.code_mode, args.code_lines,
                                    args.max_len, device, yes_id, no_id,
                                    hash_paths=args.hash_paths)
        else:
            ranked = rank_pairwise(m, tok, issue, repo, args.repo_dir, cands,
                                   args.code_mode, args.code_lines,
                                   args.max_len, device, a_id, b_id,
                                   args.n_opponents, ex_seed,
                                   hash_paths=args.hash_paths)

        r1, r5, r10, mrr = metrics(ranked, gt_set)
        R1.append(r1); R5.append(r5); R10.append(r10); MRR.append(mrr)

        if (i+1) % 10 == 0:
            mf = (_MISS["miss"] / max(1, _MISS["total"])) if args.code_mode else 0.0
            tf = _TRUNC["truncated"] / max(1, _TRUNC["total"])
            print(f"  [{i+1}] R@1={np.mean(R1)*100:.2f} R@5={np.mean(R5)*100:.2f} "
                  f"R@10={np.mean(R10)*100:.2f} MRR={np.mean(MRR)*100:.2f} "
                  f"oracle@K={np.mean(ORACLE)*100:.1f}% "
                  f"miss={mf*100:.1f}% trunc={tf*100:.1f}% "
                  f"({time.time()-start:.0f}s)", flush=True)
            if args.code_mode and _MISS["total"] >= 100 and mf > args.miss_abort_frac:
                raise RuntimeError(
                    f"Missing-code rate {mf*100:.1f}% > {args.miss_abort_frac*100:.1f}% "
                    f"(repo={repo}, repo_dir={args.repo_dir})")

    n = len(R1)
    mf = (_MISS["miss"] / max(1, _MISS["total"])) if args.code_mode else 0.0
    tf = _TRUNC["truncated"] / max(1, _TRUNC["total"])
    summary = {
        "objective": args.objective, "code_mode": args.code_mode,
        "hash_paths": args.hash_paths, "top_k": args.top_k,
        "n_opponents": args.n_opponents, "code_lines": args.code_lines,
        "issue_chars": args.issue_chars, "max_len": args.max_len,
        "n": n,
        "R@1": float(np.mean(R1)*100) if n else 0.0,
        "R@5": float(np.mean(R5)*100) if n else 0.0,
        "R@10": float(np.mean(R10)*100) if n else 0.0,
        "MRR": float(np.mean(MRR)*100) if n else 0.0,
        "oracle_at_K": float(np.mean(ORACLE)*100) if ORACLE else 0.0,
        "code_reads_total": _MISS["total"],
        "code_reads_missing": _MISS["miss"],
        "code_missing_frac": float(mf),
        "trunc_total": _TRUNC["total"],
        "trunc_truncated": _TRUNC["truncated"],
        "trunc_frac": float(tf),
        "skipped": dict(_SKIP),
        "model": args.model_path, "lora": args.lora_path,
        "yes_id": int(yes_id), "no_id": int(no_id), "a_id": int(a_id), "b_id": int(b_id),
        "metric_semantics": "Hit@k (any GT in top-k); MRR uses first GT rank",
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Done n={n} R@1={summary['R@1']:.2f} R@5={summary['R@5']:.2f} "
          f"R@10={summary['R@10']:.2f} MRR={summary['MRR']:.2f} "
          f"oracle@K={summary['oracle_at_K']:.1f}% "
          f"miss={mf*100:.1f}% trunc={tf*100:.1f}% skip={dict(_SKIP)} ===", flush=True)


if __name__ == "__main__":
    main()
