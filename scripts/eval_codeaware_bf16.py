#!/usr/bin/env python3
"""Code-aware bf16 eval (sanity check) for codeaware LoRA on SWE-bench / GREPO.
Identical to eval_codeaware_4bit.py except base model is loaded in pure bf16
(no 4-bit quantization), matching the bf16 training precision in train_rankft.py.
Use this to sanity-check whether 4-bit-inference R@1 numbers track bf16-inference R@1.
"""
import os, sys, json, argparse, time
from collections import defaultdict
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

torch.manual_seed(42); np.random.seed(42)

PROMPT_PATH = ("Given the bug report, is this file likely to need modification?\n\n"
               "Bug Report: {issue}\n\nFile: {path}\n\nAnswer:")
PROMPT_CODE = ("Given the bug report, is this file likely to need modification?\n\n"
               "Bug Report: {issue}\n\nFile: {path}\n\nCode:\n{code}\n\nAnswer:")


_ALIAS_MAP = None  # repo -> {hashed_path -> orig_path} (reverse of generator output)


def _load_reverse_alias(path):
    if not path or not os.path.isfile(path): return {}
    raw = json.load(open(path))
    return {repo: {h: o for o, h in m.items()} for repo, m in raw.items()}


def read_code(repo_dir, repo, fpath, n=50, cap=1500):
    if _ALIAS_MAP and repo in _ALIAS_MAP:
        fpath = _ALIAS_MAP[repo].get(fpath, fpath)
    full = os.path.join(repo_dir, repo, fpath)
    if not os.path.isfile(full): return "# (not available)"
    try:
        with open(full, errors="ignore") as f:
            return "".join(f.readlines()[:n])[:cap]
    except: return "# (unreadable)"


def partial_recall_at_k(ranked, gt_set, k):
    hits = [1 for c in ranked[:k] if c in gt_set]
    return sum(hits) / max(1, len(gt_set))


def build_ids_preserve_suffix(prompt, tok, max_len, suffix="\n\nAnswer:"):
    """Token-id level truncation: returns list of ids with suffix guaranteed at tail, len <= max_len."""
    suf_ids = tok.encode(suffix, add_special_tokens=False)
    if prompt.endswith(suffix):
        body_ids = tok.encode(prompt[:-len(suffix)], add_special_tokens=False)
    else:
        body_ids = tok.encode(prompt, add_special_tokens=False)
    keep = max_len - len(suf_ids)
    if keep <= 0:
        return suf_ids[-max_len:]  # suffix-only, trimmed from front if needed
    return body_ids[:keep] + suf_ids


@torch.no_grad()
def score_batch(model, tok, prompts, yes_id, no_id, max_len, dev, bs=8):
    """Build input_ids directly (no downstream tok() re-encode). Left-pad for rightmost-logit scoring."""
    batch_ids = [build_ids_preserve_suffix(p, tok, max_len) for p in prompts]
    pad_id = tok.pad_token_id
    scores = []
    for i in range(0, len(batch_ids), bs):
        chunk = batch_ids[i:i+bs]
        max_in = max(len(x) for x in chunk)
        padded = [[pad_id] * (max_in - len(x)) + x for x in chunk]
        attn = [[0] * (max_in - len(x)) + [1] * len(x) for x in chunk]
        input_ids = torch.tensor(padded, device=dev)
        attention_mask = torch.tensor(attn, device=dev)
        out = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        s = (out[:, yes_id].float() - out[:, no_id].float()).cpu().numpy()
        scores.extend(s.tolist())
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lora_path", default=None)
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--bm25_candidates", required=True)
    ap.add_argument("--repo_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--include_code", action="store_true")
    ap.add_argument("--code_max_lines", type=int, default=50)
    ap.add_argument("--alias_map", default=None,
                    help="Optional JSON: {repo: {orig_path: hashed_path}} for reading code under PathSwap eval")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--max_seq_length", type=int, default=768)
    ap.add_argument("--score_batch_size", type=int, default=4)
    args = ap.parse_args()

    global _ALIAS_MAP
    _ALIAS_MAP = _load_reverse_alias(args.alias_map)
    if _ALIAS_MAP:
        print(f"Loaded alias map for {len(_ALIAS_MAP)} repos (reverse hashed→orig).", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    dev = f"cuda:{args.gpu_id}"

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # bf16-sanity variant: NO 4-bit quantization. Loads base model in pure bf16.
    # Match training precision exactly (train_rankft.py also uses torch_dtype=bfloat16).
    m = AutoModelForCausalLM.from_pretrained(args.model_path,
                                              device_map={"": dev}, trust_remote_code=True,
                                              torch_dtype=torch.bfloat16)
    if args.lora_path: m = PeftModel.from_pretrained(m, args.lora_path)
    m.eval()
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    print(f"Loaded. yes_id={yes_id}, no_id={no_id}, include_code={args.include_code}", flush=True)

    bm25 = {}
    for line in open(args.bm25_candidates):
        r = json.loads(line)
        bm25[(r["repo"], str(r["issue_id"]))] = r.get("bm25_candidates") or r.get("candidates") or []
    print(f"BM25 keys: {len(bm25)}", flush=True)

    recalls = {1: [], 3: [], 5: [], 10: [], 20: []}
    n_eval = 0; start = time.time()
    for line in open(args.test_data):
        rec = json.loads(line)
        repo = rec.get("repo", "")
        iid = str(rec.get("issue_id", ""))
        gt = set(rec.get("changed_py_files") or rec.get("changed_files") or [])
        if not gt: continue
        cands = bm25.get((repo, iid), [])[:args.top_k]
        if not cands: continue
        issue = rec.get("issue_text", "")[:1500]
        if args.include_code:
            prompts = [PROMPT_CODE.format(issue=issue, path=c,
                                           code=read_code(args.repo_dir, repo, c, args.code_max_lines))
                       for c in cands]
        else:
            prompts = [PROMPT_PATH.format(issue=issue, path=c) for c in cands]
        scores = score_batch(m, tok, prompts, yes_id, no_id, args.max_seq_length, dev, args.score_batch_size)
        ranked = [c for c, _ in sorted(zip(cands, scores), key=lambda x: -x[1])]
        for k in recalls:
            recalls[k].append(partial_recall_at_k(ranked, gt, k))
        n_eval += 1
        if n_eval % 20 == 0:
            r1 = np.mean(recalls[1]) * 100
            print(f"  [{n_eval}] R@1={r1:.2f}% ({time.time()-start:.0f}s)", flush=True)

    summary = {
        "overall": {f"recall@{k}": float(np.mean(v) * 100) for k, v in recalls.items()},
        "n_eval": n_eval,
        "config": vars(args),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Final (n={n_eval}) ===", flush=True)
    for k, v in summary["overall"].items():
        print(f"  {k}: {v:.2f}%")


if __name__ == "__main__":
    main()
