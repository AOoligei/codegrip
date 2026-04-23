#!/usr/bin/env python3
"""SweRankLLM listwise eval — uses official model + chat template + their prompt.

Codex audit v2 fixes:
  - Truncation: 2-pass shrink (reduce per_item, rebuild) instead of tail-chop
    (avoids destroying the chat assistant generation prefix)
  - Parse-failure tripwire: return (order, ok); count parse_fail
  - Context sizing: effective_max_len = max_len - max_new_tokens - safety
"""
import argparse, json, os, random, time, hashlib, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42); np.random.seed(42); torch.manual_seed(42)

PREFIX_USER = ("I will provide you with {num} code functions, each indicated by an "
               "alphabetical identifier []. Rank the code functions based on their "
               "relevance to fixing the GitHub issue: {query}\n")
ITEM_TEMPLATE = "\n[{label}] File: {path}\nCode:\n{code}\n"
SUFFIX_USER = ("\nRank the {num} code functions above based on their relevance to "
               "fixing the GitHub issue. The output format should be [] > [] > ..., "
               "e.g., [2] > [1] > [3]. Only respond with the ranking results, do not "
               "say any word or explain.")

_MISS = {"miss": 0, "total": 0}
_TRUNC = {"truncated": 0, "total": 0}
_SKIP = {"no_bm25": 0, "no_gt": 0, "few_cands": 0}
_PARSE_FAIL = {"count": 0}


def stable_seed(*parts):
    return int(hashlib.sha256("\n".join(str(p) for p in parts).encode()).hexdigest()[:16], 16)


def read_head(repo_dir, repo, fpath, n=50):
    full = os.path.join(repo_dir, repo, fpath)
    _MISS["total"] += 1
    if not os.path.isfile(full):
        _MISS["miss"] += 1
        return "# (not available)"
    try:
        with open(full, "r", errors="replace") as fh:
            lines = fh.readlines()[:n]
        return "".join(lines)[:min(800, n*80)]
    except Exception:
        _MISS["miss"] += 1
        return "# (unreadable)"


def parse_ranking(text, num_items):
    """Returns (order, ok). Supports BOTH numeric [1][2] AND alphabet [A][B] labels.
    SweRankLLM-Small is trained with NUMERIC labels."""
    t = text.upper()
    valid = set(range(num_items))
    seen = set(); order = []
    # Numeric brackets first (SweRankLLM convention)
    for m in re.findall(r"\[\s*(\d+)\s*\]", t):
        idx = int(m) - 1
        if idx in valid and idx not in seen:
            order.append(idx); seen.add(idx)
    if len(seen) < 2:
        # Fallback: alphabet brackets
        amatches = re.findall(r"\[\s*([A-Z])\s*\]", t)
        if not amatches:
            amatches = re.findall(r"\b([A-Z])\b(?=\s*(?:>|,|$))", t)
        for m in amatches:
            idx = ord(m) - ord("A")
            if idx in valid and idx not in seen:
                order.append(idx); seen.add(idx)
    ok = len(seen) >= 2
    for i in range(num_items):
        if i not in seen:
            order.append(i)
    return order, ok


def _build_once(tok, prefix, items_with_codes, suf, add_gen_prompt=True):
    """Build user message once with given codes. Returns formatted prompt string."""
    body = "".join(ITEM_TEMPLATE.format(label=L, path=P, code=c) for L, P, c in items_with_codes)
    user_msg = prefix + body + suf
    return tok.apply_chat_template([{"role": "user", "content": user_msg}],
                                    tokenize=False, add_generation_prompt=add_gen_prompt)


def build_chat_prompt(tok, issue, items, max_len):
    """Build SweRank prompt; 2-pass shrink to keep prompt <= max_len without chopping suffix."""
    num = len(items)
    prefix = PREFIX_USER.format(num=num, query=issue)
    suf = SUFFIX_USER.format(num=num)
    # First pass: estimate frame size with empty codes
    empty_items = [(L, P, "") for L, P, _ in items]
    frame = _build_once(tok, prefix, empty_items, suf)
    frame_len = len(tok.encode(frame, add_special_tokens=False))
    available = max_len - frame_len - 8  # small safety
    _TRUNC["total"] += 1
    if available <= 0:
        # Even empty body too big; fallback to single-item prompt with smallest code
        out = _build_once(tok, prefix, [(items[0][0], items[0][1], "")] + [(L, P, "") for L, P, _ in items[1:]], suf)
        if len(tok.encode(out, add_special_tokens=False)) > max_len:
            _TRUNC["truncated"] += 1
        return out
    per_item = max(0, available // num)
    truncated_any = False
    code_budgets = []
    for L, P, code in items:
        code_ids = tok.encode(code, add_special_tokens=False)
        if len(code_ids) > per_item:
            truncated_any = True
        code_budgets.append((L, P, tok.decode(code_ids[:per_item], skip_special_tokens=False)))
    if truncated_any: _TRUNC["truncated"] += 1
    out = _build_once(tok, prefix, code_budgets, suf)
    out_len = len(tok.encode(out, add_special_tokens=False))
    # 2-pass shrink: if still over, halve per_item and rebuild
    shrink_attempts = 0
    while out_len > max_len and per_item > 50 and shrink_attempts < 5:
        per_item = per_item // 2
        new_codes = []
        for L, P, code in items:
            code_ids = tok.encode(code, add_special_tokens=False)
            new_codes.append((L, P, tok.decode(code_ids[:per_item], skip_special_tokens=False)))
        out = _build_once(tok, prefix, new_codes, suf)
        out_len = len(tok.encode(out, add_special_tokens=False))
        shrink_attempts += 1
        _TRUNC["truncated"] += 1
    if out_len > max_len:
        # Last resort: build with empty codes (preserves chat template integrity)
        out = _build_once(tok, prefix, [(L, P, "") for L, P, _ in items], suf)
    return out


def listwise_window_pass(m, tok, issue, items, max_len, max_new_tokens, device):
    prompt = build_chat_prompt(tok, issue, items, max_len)
    inp = tok(prompt, return_tensors="pt", truncation=False, add_special_tokens=False).to(device)
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                         num_beams=1, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=False)
    order, ok = parse_ranking(gen, len(items))
    if not ok:
        _PARSE_FAIL["count"] += 1
    return [items[i] for i in order]


def sliding_window_rerank(m, tok, issue, repo, repo_dir, cands, code_lines,
                          window, step, max_len, max_new_tokens, device):
    assert window >= 2 and 0 < step <= window
    items = [(None, c, read_head(repo_dir, repo, c, code_lines)) for c in cands]
    K = len(items)
    if K < 2: return [p for _, p, _ in items]
    window = min(window, K)
    start_idx = K - window
    while start_idx > -step:
        s = max(0, start_idx); e = min(K, s + window)
        win = items[s:e]
        # SweRankLLM was trained with NUMERIC labels [1] [2] ... not alphabet
        win_labeled = [(str(i + 1), p, c) for i, (_, p, c) in enumerate(win)]
        ranked = listwise_window_pass(m, tok, issue, win_labeled, max_len, max_new_tokens, device)
        items[s:e] = [(None, p, c) for _, p, c in ranked]
        start_idx -= step
    return [p for _, p, _ in items]


def load_model(model_path, device):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb,
                                              device_map={"": device}, trust_remote_code=True,
                                              torch_dtype=torch.bfloat16)
    m.eval()
    return m, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--bm25_candidates", required=True)
    ap.add_argument("--repo_dir", required=True)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--code_lines", type=int, default=50)
    ap.add_argument("--issue_chars", type=int, default=1000)
    ap.add_argument("--max_len", type=int, default=8192,
                    help="TOTAL context budget (prompt + max_new_tokens)")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--miss_abort_frac", type=float, default=0.05)
    ap.add_argument("--parse_fail_abort_frac", type=float, default=0.30)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

    # Reserve room for max_new_tokens + safety in prompt budget
    prompt_budget = args.max_len - args.max_new_tokens - 16
    assert prompt_budget > 1024, f"max_len {args.max_len} too small for max_new_tokens {args.max_new_tokens}"

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"
    data = [json.loads(l) for l in open(args.test_data)]
    if args.max_examples: data = data[:args.max_examples]
    bm25 = {}
    with open(args.bm25_candidates) as f:
        for l in f:
            r = json.loads(l); bm25[(r["repo"], str(r["issue_id"]))] = r
    print(f"Loaded {len(data)} test, {len(bm25)} bm25, K={args.top_k}, "
          f"win={args.window}/{args.step}, prompt_budget={prompt_budget}", flush=True)
    m, tok = load_model(args.model_path, device)
    print(f"  Chat template present: {tok.chat_template is not None}", flush=True)

    R1=[]; R5=[]; R10=[]; MRR=[]; ORACLE=[]; passes=0; start=time.time()
    for i, rec in enumerate(data):
        repo = rec.get("repo", "")
        issue = rec.get("issue_text", "")[:args.issue_chars]
        gt_files = list(set(rec.get("changed_py_files", rec.get("changed_files", []))))
        key = (repo, str(rec.get("issue_id", "")))
        if key not in bm25: _SKIP["no_bm25"] += 1; continue
        seen=set(); cands=[]
        cand_list = bm25[key].get("bm25_candidates") or bm25[key].get("candidates", [])
        for c in cand_list:
            if c not in seen: seen.add(c); cands.append(c)
            if len(cands) >= args.top_k: break
        if not gt_files:
            gt_files = list(set(bm25[key].get("ground_truth", [])))
        if not gt_files: _SKIP["no_gt"] += 1; continue
        if len(cands) < 2: _SKIP["few_cands"] += 1; continue
        gt_set = set(gt_files)
        ORACLE.append(1.0 if any(c in gt_set for c in cands) else 0.0)

        ranked = sliding_window_rerank(m, tok, issue, repo, args.repo_dir, cands,
                                        args.code_lines, args.window, args.step,
                                        prompt_budget, args.max_new_tokens, device)

        r1 = 1.0 if ranked[0] in gt_set else 0.0
        r5 = 1.0 if any(c in gt_set for c in ranked[:5]) else 0.0
        r10 = 1.0 if any(c in gt_set for c in ranked[:10]) else 0.0
        mrr = 0.0
        for j, c in enumerate(ranked, 1):
            if c in gt_set: mrr = 1.0/j; break
        R1.append(r1); R5.append(r5); R10.append(r10); MRR.append(mrr)
        passes += 1

        if (i+1) % 5 == 0:
            mf = _MISS["miss"]/max(1,_MISS["total"]); tf = _TRUNC["truncated"]/max(1,_TRUNC["total"])
            pf_total = max(1, passes * (args.top_k // args.step + 1))  # approx total window passes
            pf = _PARSE_FAIL["count"] / pf_total
            print(f"  [{i+1}] R@1={np.mean(R1)*100:.2f} R@5={np.mean(R5)*100:.2f} "
                  f"MRR={np.mean(MRR)*100:.2f} ora@K={np.mean(ORACLE)*100:.1f}% "
                  f"miss={mf*100:.1f}% trunc={tf*100:.1f}% parse_fail={_PARSE_FAIL['count']} "
                  f"({time.time()-start:.0f}s)", flush=True)
            if _MISS["total"] >= 100 and mf > args.miss_abort_frac:
                raise RuntimeError(f"miss {mf*100:.1f}% > {args.miss_abort_frac*100:.1f}%")
            if pf > args.parse_fail_abort_frac:
                raise RuntimeError(f"parse_fail {pf*100:.1f}% > {args.parse_fail_abort_frac*100:.1f}% - check generation output format")

    n = len(R1)
    summary = {"protocol": "swerank-listwise-sliding-officialmodel",
               "model": args.model_path, "top_k": args.top_k,
               "window": args.window, "step": args.step, "code_lines": args.code_lines,
               "n": n,
               "R@1": float(np.mean(R1)*100) if n else 0,
               "R@5": float(np.mean(R5)*100) if n else 0,
               "R@10": float(np.mean(R10)*100) if n else 0,
               "MRR": float(np.mean(MRR)*100) if n else 0,
               "oracle_at_K": float(np.mean(ORACLE)*100) if ORACLE else 0,
               "miss_frac": float(_MISS["miss"]/max(1,_MISS["total"])),
               "trunc_frac": float(_TRUNC["truncated"]/max(1,_TRUNC["total"])),
               "parse_fail_count": _PARSE_FAIL["count"],
               "skipped": dict(_SKIP)}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Done n={n} R@1={summary['R@1']:.2f} R@5={summary['R@5']:.2f} "
          f"MRR={summary['MRR']:.2f} parse_fail={_PARSE_FAIL['count']} ===", flush=True)


if __name__ == "__main__":
    main()
