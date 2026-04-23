#!/usr/bin/env python3
"""SweRankEmbed-Large dense retrieval — generate top-K candidates per (repo, issue).

Codex audit fixes:
  - model.config.use_cache = False (avoid KV-cache blowup at 8k context)
  - Tokenizer pad_token = eos + padding_side="left" (Qwen embedding convention)
  - Loud warnings + skip counts when repo/files missing
  - read_doc streams first N lines (no readlines() of whole file)
  - Skip __pycache__, .venv, build, dist, etc. in os.walk
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

QUERY_TEMPLATE = ("Instruct: Given a github issue, identify the code that needs "
                  "to be changed to fix the issue.\nQuery: {query}")
DOC_MAX_LINES = 200
DOC_CHAR_CAP = 16000
SKIP_DIRS = {"__pycache__", ".venv", "venv", "build", "dist", ".git",
             "node_modules", ".tox", ".pytest_cache", ".mypy_cache", "htmlcov"}
_SKIP = {"no_repo": 0, "empty_files": 0, "no_gt": 0}


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device),
                               sequence_lengths]


def encode(model, tokenizer, texts, max_length, device, batch_size, dtype):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inp = tokenizer(batch, padding=True, truncation=True,
                        return_tensors="pt", max_length=max_length).to(device)
        with torch.no_grad():
            out = model(**inp)
        e = last_token_pool(out.last_hidden_state, inp["attention_mask"])
        e = F.normalize(e.to(dtype), p=2, dim=-1)
        embs.append(e.cpu())
    return torch.cat(embs, dim=0)


def list_repo_files(repo_dir, repo):
    rd = os.path.join(repo_dir, repo)
    if not os.path.isdir(rd): return []
    files = []
    for root, dirs, fns in os.walk(rd):
        # Prune skip dirs in-place
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in fns:
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(root, f), rd)
                files.append(rel)
    return files


def read_doc(repo_dir, repo, fpath):
    """Stream first N lines + char cap; never readlines() entire file."""
    full = os.path.join(repo_dir, repo, fpath)
    try:
        with open(full, "r", errors="replace") as fh:
            chunks = []; total_chars = 0
            for i, line in enumerate(fh):
                if i >= DOC_MAX_LINES: break
                chunks.append(line)
                total_chars += len(line)
                if total_chars >= DOC_CHAR_CAP: break
            return "".join(chunks)[:DOC_CHAR_CAP]
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--repo_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--query_max_length", type=int, default=2048,
                    help="Smaller max_length for queries (issues are short)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

    device = f"cuda:{args.gpu_id}"
    dtype = torch.float32

    print(f"Loading model from {args.model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True,
                                       torch_dtype=torch.bfloat16).to(device)
    model.eval()
    model.config.use_cache = False  # avoid KV-cache blowup
    print(f"  Padding: {tokenizer.padding_side}, max_len doc={args.max_length} "
          f"query={args.query_max_length}", flush=True)

    data = [json.loads(l) for l in open(args.test_data)]
    if args.max_examples: data = data[:args.max_examples]

    by_repo = {}
    for r in data:
        by_repo.setdefault(r["repo"], []).append(r)
    print(f"Loaded {len(data)} examples across {len(by_repo)} repos", flush=True)

    out_records = []
    t0 = time.time()
    for repo_idx, (repo, recs) in enumerate(by_repo.items()):
        print(f"\n[{repo_idx+1}/{len(by_repo)}] repo={repo} ({len(recs)} examples)", flush=True)
        files = list_repo_files(args.repo_dir, repo)
        if len(files) < 2:
            print(f"  [WARN] repo dir missing or has <2 .py files (got {len(files)}); "
                  f"skipping {len(recs)} records", flush=True)
            _SKIP["no_repo"] += len(recs)
            continue
        docs = [read_doc(args.repo_dir, repo, f) for f in files]
        keep = [(f, d) for f, d in zip(files, docs) if d.strip()]
        if not keep:
            print(f"  [WARN] all {len(files)} files empty/unreadable; skipping {len(recs)}", flush=True)
            _SKIP["empty_files"] += len(recs)
            continue
        files_kept = [f for f, _ in keep]
        docs_kept = [d for _, d in keep]
        t_enc = time.time()
        doc_embs = encode(model, tokenizer, docs_kept, args.max_length,
                          device, args.batch_size, dtype)
        print(f"  encoded {len(files_kept)} docs in {time.time()-t_enc:.1f}s "
              f"-> shape={tuple(doc_embs.shape)}", flush=True)

        for rec in recs:
            issue = rec.get("issue_text", "")[:5000]
            query = QUERY_TEMPLATE.format(query=issue)
            q_emb = encode(model, tokenizer, [query], args.query_max_length, device, 1, dtype)
            scores = (q_emb @ doc_embs.T).squeeze(0)
            topk_idx = torch.topk(scores, k=min(args.top_k, len(files_kept))).indices
            out_records.append({
                "repo": repo, "issue_id": rec.get("issue_id"),
                "bm25_candidates": [files_kept[i] for i in topk_idx.tolist()],
                "ground_truth": list(rec.get("changed_py_files", []))
            })
        torch.cuda.empty_cache()
        print(f"  cumulative: {len(out_records)} retrieved ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(out_records)}/{len(data)} records to {args.output}", flush=True)
    print(f"Skipped: {dict(_SKIP)}", flush=True)
    if out_records:
        recall = [1.0 if any(g in r["bm25_candidates"] for g in r["ground_truth"]) else 0.0
                  for r in out_records if r["ground_truth"]]
        print(f"Oracle Hit@{args.top_k}: {np.mean(recall)*100:.1f}% (n={len(recall)})", flush=True)


if __name__ == "__main__":
    main()
