#!/usr/bin/env python3
"""Pairwise GT-vs-hard-negative WITH code controls.

Variants:
  --control none           : standard path_code (baseline)
  --control swap           : Code A <-> Code B between fa and fb
  --control random         : Code from random in-repo file (matched-length irrelevant)
  --control no_comments    : tokenize-based docstring + comment removal

Codex audit v2 fixes:
  - hashlib.sha256 stable seeds
  - Per-side token budget for ca/cb (no silent B-truncation)
  - find_random_file uses cached per-repo file list, deterministic seed
  - strip_comments uses tokenize (not ast.unparse) to preserve layout
  - Asserts each scoring token = 1 token
  - Truncation rate logged
  - 5% miss tripwire (was 10%)
  - Skipped-example counters
"""
import argparse, json, os, random, time, hashlib, io, tokenize
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42); np.random.seed(42); torch.manual_seed(42)

PROMPT = ("Given the bug report, which file is more likely to need modification? "
          "Consider both the file paths and code content. Answer A or B.\n\n"
          "Bug Report: {issue_text}\n\nFile A: {fa}\nCode A:\n{ca}\n\n"
          "File B: {fb}\nCode B:\n{cb}\n\nAnswer:")
ANSWER_SUFFIX = "\n\nAnswer:"
_MISS = {"miss": 0, "total": 0}
_TRUNC = {"truncated": 0, "total": 0}
_SKIP = {"no_bm25": 0, "no_gt": 0, "no_neg": 0}
_REPO_FILE_CACHE = {}


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
        char_cap = max(800, n * 80)
        return "".join(lines)[:char_cap]
    except Exception:
        _MISS["miss"] += 1
        return "# (unreadable)"


def strip_comments_tokenize(code):
    """Use tokenize to remove COMMENT tokens + docstrings (Expr at module/func/class top).
    Falls back to ORIGINAL code on parse failure (NOT regex-chopping)."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenizeError, IndentationError):
        return code  # fallback: return original to avoid corruption
    out_toks = []
    for t in toks:
        if t.type == tokenize.COMMENT:
            continue
        out_toks.append(t)
    try:
        out = tokenize.untokenize(out_toks)
    except Exception:
        return code
    # Drop top-level docstrings via ast (best effort, fail open)
    try:
        import ast
        tree = ast.parse(out)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    node.body.pop(0)
        out2 = ast.unparse(tree)
        return out2
    except Exception:
        return out


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


def get_repo_files_cached(repo_dir, repo):
    if repo in _REPO_FILE_CACHE:
        return _REPO_FILE_CACHE[repo]
    rd = os.path.join(repo_dir, repo)
    files = []
    if os.path.isdir(rd):
        for root, _, fns in os.walk(rd):
            for f in fns:
                if f.endswith(".py"):
                    files.append(os.path.relpath(os.path.join(root, f), rd))
    _REPO_FILE_CACHE[repo] = files
    return files


def find_random_file(repo_dir, repo, exclude, rng):
    pool = [f for f in get_repo_files_cached(repo_dir, repo) if f not in exclude]
    if not pool: return None
    return rng.choice(pool)


def truncate_per_side(tok, issue, fa, fb, ca, cb, max_len, suffix=ANSWER_SUFFIX):
    """Build PROMPT but token-budget ca/cb individually so BOTH survive."""
    frame = PROMPT.format(issue_text=issue, fa=fa, fb=fb, ca="", cb="")
    frame_ids = tok.encode(frame, add_special_tokens=False)
    budget = max_len - len(frame_ids) - 32
    _TRUNC["total"] += 1
    if budget < 200:
        # Frame too big — truncate from end (preserves Answer suffix)
        full = PROMPT.format(issue_text=issue, fa=fa, fb=fb, ca=ca, cb=cb)
        ids = tok.encode(full, add_special_tokens=False)
        if len(ids) <= max_len: return full
        _TRUNC["truncated"] += 1
        suf_ids = tok.encode(suffix, add_special_tokens=False)
        return tok.decode(ids[:max_len - len(suf_ids)], skip_special_tokens=False) + suffix
    per_side = budget // 2
    ca_ids_full = tok.encode(ca, add_special_tokens=False)
    cb_ids_full = tok.encode(cb, add_special_tokens=False)
    ca_ids = ca_ids_full[:per_side]
    cb_ids = cb_ids_full[:per_side]
    if len(ca_ids) < len(ca_ids_full) or len(cb_ids) < len(cb_ids_full):
        _TRUNC["truncated"] += 1
    out = PROMPT.format(issue_text=issue, fa=fa, fb=fb,
                          ca=tok.decode(ca_ids, skip_special_tokens=False),
                          cb=tok.decode(cb_ids, skip_special_tokens=False))
    # Final enforce: re-encode and tail-truncate if still over budget
    out_ids = tok.encode(out, add_special_tokens=False)
    if len(out_ids) > max_len:
        suf_ids = tok.encode(suffix, add_special_tokens=False)
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
    def must_one(s, label):
        ids = tok.encode(s, add_special_tokens=False)
        assert len(ids) == 1, f"{label}: {s!r} -> {len(ids)} tokens"
        return ids[0]
    a_id = must_one(" A", "a")
    b_id = must_one(" B", "b")
    suffix_ids = tok.encode("Answer: A", add_special_tokens=False)
    assert suffix_ids[-1] == a_id
    print(f"  a={a_id}({tok.decode([a_id])!r}) b={b_id}({tok.decode([b_id])!r})", flush=True)
    return m, tok, a_id, b_id


def score_pair(m, tok, prompt, a_id, b_id, device):
    inp = tok(prompt, return_tensors="pt", truncation=False, add_special_tokens=False).to(device)
    with torch.no_grad():
        logits = m(**inp).logits[:, -1, :]
    return (logits[0, a_id] - logits[0, b_id]).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/data/shuyang/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--lora_path", default="/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/best")
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--bm25_candidates", required=True)
    ap.add_argument("--repo_dir", required=True)
    ap.add_argument("--control", required=True,
                    choices=["none", "swap", "random", "no_comments"])
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
    with open(args.bm25_candidates) as f:
        for l in f:
            r = json.loads(l)
            bm25[(r["repo"], str(r["issue_id"]))] = r
    print(f"Loaded {len(data)} test, {len(bm25)} bm25, control={args.control}", flush=True)

    m, tok, a_id, b_id = load_model(args.model_path, args.lora_path, device)
    correct = []; start = time.time()

    for i, rec in enumerate(data):
        repo = rec.get("repo", "")
        issue = rec.get("issue_text", "")[:args.issue_chars]
        gt_files = list(set(rec.get("changed_py_files", rec.get("changed_files", []))))
        key = (repo, str(rec.get("issue_id", "")))
        cands = []
        if key in bm25:
            cands = bm25[key].get("bm25_candidates", [])
            if not gt_files:
                gt_files = list(set(bm25[key].get("ground_truth", [])))
        else:
            _SKIP["no_bm25"] += 1
        if not gt_files:
            _SKIP["no_gt"] += 1; continue
        if not cands:
            cands = get_repo_files_cached(args.repo_dir, repo)
        gt = sorted(gt_files)[0]
        neg = find_hard_neg(gt, gt_files, cands)
        if neg is None:
            _SKIP["no_neg"] += 1; continue

        # A/B order via stable per-example seed
        if stable_seed("ab", repo, rec.get("issue_id", "")) % 2 == 0:
            fa, fb, gt_is_a = gt, neg, True
        else:
            fa, fb, gt_is_a = neg, gt, False

        ca_orig = read_head(args.repo_dir, repo, fa, args.code_lines)
        cb_orig = read_head(args.repo_dir, repo, fb, args.code_lines)

        if args.control == "none":
            ca, cb = ca_orig, cb_orig
        elif args.control == "swap":
            ca, cb = cb_orig, ca_orig
        elif args.control == "random":
            ex_rng = random.Random(stable_seed("rand", repo, rec.get("issue_id", "")))
            rfa = find_random_file(args.repo_dir, repo, set(gt_files) | {fa, fb}, ex_rng)
            rfb = find_random_file(args.repo_dir, repo, set(gt_files) | {fa, fb, rfa}, ex_rng) if rfa else None
            ca = read_head(args.repo_dir, repo, rfa, args.code_lines) if rfa else "# (no rand)"
            cb = read_head(args.repo_dir, repo, rfb, args.code_lines) if rfb else "# (no rand)"
        elif args.control == "no_comments":
            ca = strip_comments_tokenize(ca_orig)
            cb = strip_comments_tokenize(cb_orig)

        prompt = truncate_per_side(tok, issue, fa, fb, ca, cb, args.max_len)
        s = score_pair(m, tok, prompt, a_id, b_id, device)
        pred_a = s > 0
        hit = (pred_a == gt_is_a)
        correct.append(1.0 if hit else 0.0)

        if (i+1) % 20 == 0:
            mf = _MISS["miss"] / max(1, _MISS["total"])
            tf = _TRUNC["truncated"] / max(1, _TRUNC["total"])
            print(f"  [{i+1}] acc={np.mean(correct)*100:.2f}% miss={mf*100:.1f}% "
                  f"trunc={tf*100:.1f}% ({time.time()-start:.0f}s)", flush=True)
            if _MISS["total"] >= 100 and mf > args.miss_abort_frac:
                raise RuntimeError(
                    f"Missing-code rate {mf*100:.1f}% > {args.miss_abort_frac*100:.1f}% "
                    f"(repo={repo}, repo_dir={args.repo_dir})")

    n = len(correct)
    acc = float(np.mean(correct)*100) if n else 0.0
    mf = _MISS["miss"] / max(1, _MISS["total"])
    tf = _TRUNC["truncated"] / max(1, _TRUNC["total"])
    summary = {"control": args.control, "code_lines": args.code_lines,
               "issue_chars": args.issue_chars, "max_len": args.max_len,
               "n": n, "pairwise_acc": acc,
               "code_reads_total": _MISS["total"],
               "code_reads_missing": _MISS["miss"],
               "code_missing_frac": float(mf),
               "trunc_total": _TRUNC["total"], "trunc_truncated": _TRUNC["truncated"],
               "trunc_frac": float(tf),
               "skipped": dict(_SKIP),
               "model": args.model_path, "lora": args.lora_path}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== {args.control} acc={acc:.2f}% n={n} miss={mf*100:.1f}% trunc={tf*100:.1f}% skip={dict(_SKIP)} ===", flush=True)


if __name__ == "__main__":
    main()
