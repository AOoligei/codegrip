#!/usr/bin/env python3
"""End-to-end RRF: our codeaware LoRA ranks + SweRankEmbed ranks, fused with RRF.

Codex-audited (2026-04-20) fixes:
  - Yes/No token ID uses last token of "Answer: Yes" / "Answer: No" (space prefix)
  - Truncation preserves "Answer:" suffix
  - Missing-code tripwire
  - Coverage tracking
  - Candidate dedup
"""
import argparse, json, os, hashlib, time, torch
import numpy as np
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

PROMPT = ("Given the bug report, is this file likely to need modification?\n\n"
          "Bug Report: {issue_text}\n\nFile: {candidate_path}\nCode:\n{code}\n\nAnswer:")
SUFFIX = "\n\nAnswer:"

_MISS = {"miss": 0, "total": 0}


def read_head(repo_dir, repo, fpath, n=50):
    full = os.path.join(repo_dir, repo, fpath)
    _MISS["total"] += 1
    if not os.path.isfile(full):
        _MISS["miss"] += 1
        return "# (not available)"
    try:
        return "".join(open(full, "r", errors="replace").readlines()[:n])[:800]
    except Exception:
        _MISS["miss"] += 1
        return "# (unreadable)"


def truncate_preserve_suffix(prompt, tok, max_len):
    ids = tok.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_len:
        return prompt
    suf_ids = tok.encode(SUFFIX, add_special_tokens=False)
    keep = max_len - len(suf_ids)
    if keep <= 0:
        return SUFFIX
    return tok.decode(ids[:keep], skip_special_tokens=True) + SUFFIX


def load_path_model(mp, lp, device):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        mp, quantization_config=bnb, device_map={"": device},
        trust_remote_code=True, torch_dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, lp); m.eval()
    yes_ids = tok.encode("Answer: Yes", add_special_tokens=False)
    no_ids = tok.encode("Answer: No", add_special_tokens=False)
    yes_id, no_id = yes_ids[-1], no_ids[-1]
    assert yes_id != no_id
    print(f"  yes_id={yes_id} ({tok.decode([yes_id])!r}) "
          f"no_id={no_id} ({tok.decode([no_id])!r})", flush=True)
    return m, tok, yes_id, no_id


def score_path(m, tok, prompts, yes_id, no_id, device, max_len=1536, bs=4):
    prompts = [truncate_preserve_suffix(p, tok, max_len) for p in prompts]
    out = []
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i+bs]
        inp = tok(batch, return_tensors="pt", padding=True, truncation=False,
                  padding_side="left").to(device)
        with torch.no_grad():
            logits = m(**inp).logits[:, -1, :]
        s = (logits[:, yes_id].float() - logits[:, no_id].float()).cpu().numpy()
        out.extend(s.tolist())
    return np.array(out)


def score_code(embed_model, query, docs, batch_size=8):
    qv = embed_model.encode([query], prompt_name="query", convert_to_tensor=True,
                             normalize_embeddings=True, batch_size=1,
                             show_progress_bar=False)
    dv = embed_model.encode(docs, convert_to_tensor=True, normalize_embeddings=True,
                             batch_size=batch_size, show_progress_bar=False)
    return (qv @ dv.T).squeeze(0).cpu().float().numpy()


def rrf(rank, k=60):
    return 1.0 / (k + rank)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_model", default="/data/shuyang/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--path_lora", default="/home/chenlibin/grepo_agent/experiments/rankft_codeaware_swetrain/best")
    ap.add_argument("--code_model", required=True)
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--bm25_candidates", required=True)
    ap.add_argument("--repo_dir", required=True)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--rrf_k", type=int, default=60)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--miss_abort_frac", type=float, default=0.10)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"
    print("Loading path LoRA...", flush=True)
    m_path, tok_path, yes_id, no_id = load_path_model(args.path_model, args.path_lora, device)
    print("Loading SweRankEmbed...", flush=True)
    m_code = SentenceTransformer(args.code_model, trust_remote_code=True,
                                  device=device,
                                  model_kwargs={"torch_dtype": torch.bfloat16})

    data = [json.loads(l) for l in open(args.test_data)]
    bm25 = {}
    for l in open(args.bm25_candidates):
        r = json.loads(l); bm25[(r["repo"], str(r["issue_id"]))] = r
    print(f"Loaded {len(data)} test, {len(bm25)} bm25", flush=True)

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    hits = {a: [] for a in alphas}
    hits_path = []; hits_code = []
    start = time.time()
    n_skipped = {"no_bm25": 0, "no_gt": 0, "few_cands": 0}
    n_processed = 0

    for i, rec in enumerate(data):
        repo = rec.get("repo", "")
        issue = rec.get("issue_text", "")[:1000]
        gt = set(rec.get("changed_py_files", rec.get("changed_files", [])))
        if not gt:
            n_skipped["no_gt"] += 1; continue
        key = (repo, str(rec.get("issue_id", "")))
        if key not in bm25:
            n_skipped["no_bm25"] += 1; continue
        raw_cands = bm25[key].get("bm25_candidates", bm25[key].get("candidates", []))[:args.top_k]
        cands = list(dict.fromkeys(raw_cands))
        for g in gt:
            if g not in cands: cands.append(g)
        if len(cands) < 2:
            n_skipped["few_cands"] += 1; continue

        codes = [read_head(args.repo_dir, repo, c) for c in cands]
        prompts = [PROMPT.format(issue_text=issue, candidate_path=c, code=code)
                   for c, code in zip(cands, codes)]
        s_path = score_path(m_path, tok_path, prompts, yes_id, no_id, device, args.max_len)
        docs = [f"File: {c}\nCode:\n{code}" for c, code in zip(cands, codes)]
        s_code = score_code(m_code, issue, docs)
        assert len(s_path) == len(s_code) == len(cands)

        ranked_path = [c for c, _ in sorted(zip(cands, s_path), key=lambda x: -x[1])]
        ranked_code = [c for c, _ in sorted(zip(cands, s_code), key=lambda x: -x[1])]
        p_rank = {c: r for r, c in enumerate(ranked_path, 1)}
        c_rank = {c: r for r, c in enumerate(ranked_code, 1)}
        hits_path.append(1.0 if ranked_path[0] in gt else 0.0)
        hits_code.append(1.0 if ranked_code[0] in gt else 0.0)
        for a in alphas:
            scores = {c: a * rrf(p_rank[c], args.rrf_k) + (1-a) * rrf(c_rank[c], args.rrf_k)
                      for c in cands}
            ranked = sorted(cands, key=lambda c: -scores[c])
            hits[a].append(1.0 if ranked[0] in gt else 0.0)
        n_processed += 1

        if (i+1) % 20 == 0:
            mf = _MISS["miss"] / max(1, _MISS["total"])
            print(f"  [{i+1}] n_ok={n_processed} "
                  f"path={np.mean(hits_path)*100:.1f} code={np.mean(hits_code)*100:.1f} "
                  f"rrf0.5={np.mean(hits[0.5])*100:.1f} miss={mf*100:.1f}% "
                  f"({time.time()-start:.0f}s)", flush=True)
            if _MISS["total"] >= 100 and mf > args.miss_abort_frac:
                raise RuntimeError(
                    f"Missing-code rate {mf*100:.1f}% exceeds {args.miss_abort_frac*100:.1f}% "
                    f"(likely repo-layout mismatch; repo_dir={args.repo_dir})")

    mf = _MISS["miss"] / max(1, _MISS["total"])
    summary = {
        "n": n_processed, "n_skipped": n_skipped,
        "code_miss_frac": float(mf),
        "path_only_r1": float(np.mean(hits_path)*100) if hits_path else 0.0,
        "code_only_r1": float(np.mean(hits_code)*100) if hits_code else 0.0,
    }
    for a in alphas:
        summary[f"rrf_alpha={a:.2f}"] = float(np.mean(hits[a])*100) if hits[a] else 0.0
    summary["best_rrf_alpha"] = max(alphas, key=lambda a: np.mean(hits[a]) if hits[a] else 0)
    summary["best_rrf_r1"] = max(summary[f"rrf_alpha={a:.2f}"] for a in alphas)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
