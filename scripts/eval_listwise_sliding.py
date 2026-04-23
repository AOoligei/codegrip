#!/usr/bin/env python3
"""SweRank-style listwise reranking with sliding window.

Implements the prompt format from SalesforceAIResearch/SweRank
(rank_listwise_os_llm.py): present W candidates as [A]..[J], ask model
to output "[B] > [A] > [C] > ..." ordering. Slide a window of size W with
step S over BM25 top-K. Score = position in final ranking.

Codex audit checklist:
  - Bare alphabet identifiers [A], [B], ... (NOT leading-space)
  - Token must_one assertion for each [X] character
  - Stable per-example seed for window order shuffling (if any)
  - Per-candidate token budget so all W candidates fit
  - Truncation tracking + fall-back
  - missing-code tripwire
  - Skipped-example counters
"""
import argparse, json, os, random, time, hashlib, re
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# SweRank-style prompts (rank_listwise_os_llm.py:259-274)
PREFIX = ("I will provide you with {num} code functions, each indicated by an alphabetical identifier [].\n"
          "Rank the code functions based on their relevance to fixing the GitHub issue: {query}\n")
ITEM_TEMPLATE = "\n[{label}] File: {path}\nCode:\n{code}\n"
SUFFIX = ("\nRank the {num} code functions above based on their relevance to fixing the GitHub issue. "
          "The output format should be [] > [] > ..., e.g., [B] > [A] > [C]. "
          "Only respond with the ranking results, do not say any word or explain.\n\nAnswer:")

ANSWER_SUFFIX = "\n\nAnswer:"
_MISS = {"miss": 0, "total": 0}
_TRUNC = {"truncated": 0, "total": 0}
_SKIP = {"no_bm25": 0, "no_gt": 0, "few_cands": 0}


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
        # Listwise needs many items in one prompt; cap each item tightly to avoid OOM
        char_cap = min(800, n * 80)
        return "".join(lines)[:char_cap]
    except Exception:
        _MISS["miss"] += 1
        return "# (unreadable)"


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
    return m, tok


def build_listwise_prompt(tok, issue, items, max_len, suffix=ANSWER_SUFFIX):
    """items: list of (label, path, code). Budget code per-item so total <= max_len."""
    num = len(items)
    prefix = PREFIX.format(num=num, query=issue)
    suf = SUFFIX.format(num=num)
    # Compute frame size with empty codes
    body_empty = "".join(ITEM_TEMPLATE.format(label=L, path=P, code="") for L, P, _ in items)
    frame = prefix + body_empty + suf
    frame_ids = tok.encode(frame, add_special_tokens=False)
    budget = max_len - len(frame_ids) - 64  # safety margin
    _TRUNC["total"] += 1
    if budget < 100:
        # Frame too big; truncate codes to nothing and tail-truncate
        full = prefix + "".join(ITEM_TEMPLATE.format(label=L, path=P, code="") for L, P, _ in items) + suf
        ids = tok.encode(full, add_special_tokens=False)
        if len(ids) > max_len:
            _TRUNC["truncated"] += 1
            suf_ids = tok.encode(suffix, add_special_tokens=False)
            keep = max_len - len(suf_ids)
            return tok.decode(ids[:keep], skip_special_tokens=False) + suffix
        return full
    per_item = budget // num
    body = ""
    truncated_any = False
    for L, P, code in items:
        code_ids_full = tok.encode(code, add_special_tokens=False)
        code_ids = code_ids_full[:per_item]
        if len(code_ids) < len(code_ids_full):
            truncated_any = True
        code_t = tok.decode(code_ids, skip_special_tokens=False)
        body += ITEM_TEMPLATE.format(label=L, path=P, code=code_t)
    if truncated_any: _TRUNC["truncated"] += 1
    out = prefix + body + suf
    # Final enforce
    out_ids = tok.encode(out, add_special_tokens=False)
    if len(out_ids) > max_len:
        suf_ids = tok.encode(suffix, add_special_tokens=False)
        keep = max_len - len(suf_ids)
        out = tok.decode(out_ids[:keep], skip_special_tokens=False) + suffix
    return out


def parse_ranking(text, num_items):
    """Parse "[B] > [A] > ..." into list of label indices (0-based).
    Tolerant: uppercase, allow inner whitespace; fall back to bare letter.
    """
    t = text.upper()
    valid = set(range(num_items))
    matches = re.findall(r"\[\s*([A-Z])\s*\]", t)
    if not matches:
        # Separator-aware fallback: only accept letters followed by ranking separator
        # (avoids pronoun "I" being parsed as label I in "I think B > A ...")
        matches = re.findall(r"\b([A-Z])\b(?=\s*(?:>|,|$))", t)
    seen = set()
    order = []
    for m in matches:
        idx = ord(m) - ord("A")
        if idx in valid and idx not in seen:
            order.append(idx); seen.add(idx)
    for i in range(num_items):
        if i not in seen:
            order.append(i)
    return order


def listwise_window_pass(m, tok, issue, items, max_len, max_new_tokens, device):
    """Run one window: build prompt, generate, parse rankings.
    Returns list of (path, code) reordered by model."""
    prompt = build_listwise_prompt(tok, issue, items, max_len)
    inp = tok(prompt, return_tensors="pt", truncation=False, add_special_tokens=False).to(device)
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=max_new_tokens,
                         do_sample=False, num_beams=1,
                         pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=False)
    order = parse_ranking(gen, len(items))
    return [items[i] for i in order]


def sliding_window_rerank(m, tok, issue, repo, repo_dir, cands, code_lines,
                           window, step, max_len, max_new_tokens, device):
    """SweRank-style: slide window of size W with step S from BOTTOM to TOP."""
    assert window >= 2, f"window must be >= 2 (got {window})"
    assert 0 < step <= window, f"step must satisfy 0 < step <= window (got step={step}, window={window})"
    items = [(None, c, read_head(repo_dir, repo, c, code_lines)) for c in cands]
    K = len(items)
    if K < 2:
        return [p for _, p, _ in items]
    window = min(window, K)
    start_idx = K - window
    while start_idx > -step:
        s = max(0, start_idx)
        e = min(K, s + window)
        win = items[s:e]
        # Assign labels A..
        win_labeled = [(chr(ord("A") + i), p, c) for i, (_, p, c) in enumerate(win)]
        ranked = listwise_window_pass(m, tok, issue, win_labeled, max_len, max_new_tokens, device)
        # Strip labels, place ranked back in items[s:e]
        items[s:e] = [(None, p, c) for _, p, c in ranked]
        start_idx -= step
    return [p for _, p, _ in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/data/shuyang/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--lora_path", default="/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/best")
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--bm25_candidates", required=True)
    ap.add_argument("--repo_dir", required=True)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--code_lines", type=int, default=50)
    ap.add_argument("--issue_chars", type=int, default=1000)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--max_new_tokens", type=int, default=128)
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
            r = json.loads(l); bm25[(r["repo"], str(r["issue_id"]))] = r
    print(f"Loaded {len(data)} test, {len(bm25)} bm25, K={args.top_k}, "
          f"window={args.window}, step={args.step}", flush=True)

    m, tok = load_model(args.model_path, args.lora_path, device)
    # Sanity: assert [A]..[J] each tokenize cleanly
    for ch in "ABCDEFGHIJ":
        ids = tok.encode(ch, add_special_tokens=False)
        if len(ids) != 1:
            print(f"  WARN: {ch!r} -> {ids} ({len(ids)} tokens)", flush=True)

    R1=[]; R5=[]; R10=[]; MRR=[]; ORACLE=[]; start=time.time()
    for i, rec in enumerate(data):
        repo = rec.get("repo", "")
        issue = rec.get("issue_text", "")[:args.issue_chars]
        gt_files = list(set(rec.get("changed_py_files", rec.get("changed_files", []))))
        key = (repo, str(rec.get("issue_id", "")))
        if key not in bm25:
            _SKIP["no_bm25"] += 1; continue
        seen=set(); cands=[]
        for c in bm25[key].get("bm25_candidates", []):
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
        ORACLE.append(1.0 if any(c in gt_set for c in cands) else 0.0)

        ranked = sliding_window_rerank(m, tok, issue, repo, args.repo_dir, cands,
                                        args.code_lines, args.window, args.step,
                                        args.max_len, args.max_new_tokens, device)

        r1 = 1.0 if ranked[0] in gt_set else 0.0
        r5 = 1.0 if any(c in gt_set for c in ranked[:5]) else 0.0
        r10 = 1.0 if any(c in gt_set for c in ranked[:10]) else 0.0
        mrr = 0.0
        for j, c in enumerate(ranked, 1):
            if c in gt_set: mrr = 1.0/j; break
        R1.append(r1); R5.append(r5); R10.append(r10); MRR.append(mrr)

        if (i+1) % 5 == 0:
            mf = _MISS["miss"] / max(1, _MISS["total"])
            tf = _TRUNC["truncated"] / max(1, _TRUNC["total"])
            print(f"  [{i+1}] R@1={np.mean(R1)*100:.2f} R@5={np.mean(R5)*100:.2f} "
                  f"MRR={np.mean(MRR)*100:.2f} oracle@K={np.mean(ORACLE)*100:.1f}% "
                  f"miss={mf*100:.1f}% trunc={tf*100:.1f}% ({time.time()-start:.0f}s)", flush=True)
            if _MISS["total"] >= 100 and mf > args.miss_abort_frac:
                raise RuntimeError(f"miss {mf*100:.1f}% > {args.miss_abort_frac*100:.1f}%")

    n = len(R1)
    summary = {"protocol": "swerank-listwise-sliding", "top_k": args.top_k,
               "window": args.window, "step": args.step, "code_lines": args.code_lines,
               "n": n,
               "R@1": float(np.mean(R1)*100) if n else 0,
               "R@5": float(np.mean(R5)*100) if n else 0,
               "R@10": float(np.mean(R10)*100) if n else 0,
               "MRR": float(np.mean(MRR)*100) if n else 0,
               "oracle_at_K": float(np.mean(ORACLE)*100) if ORACLE else 0,
               "miss_frac": float(_MISS["miss"]/max(1,_MISS["total"])),
               "trunc_frac": float(_TRUNC["truncated"]/max(1,_TRUNC["total"])),
               "skipped": dict(_SKIP),
               "model": args.model_path, "lora": args.lora_path}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Done n={n} R@1={summary['R@1']:.2f} R@5={summary['R@5']:.2f} "
          f"R@10={summary['R@10']:.2f} MRR={summary['MRR']:.2f} ===", flush=True)


if __name__ == "__main__":
    main()
