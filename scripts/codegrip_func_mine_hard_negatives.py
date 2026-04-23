#!/usr/bin/env python3
"""
Step 3: Per-query hard-negative mining at function granularity.

For each (repo, base_commit, issue) in func_gt_{swe,grepo}.jsonl:
  1) Locate corpus shard /data/chenlibin/codegrip_func/corpus/<repo>__<commit[:12]>.jsonl
  2) Tokenize each function record (_id + text body) with SweRank-byte-matched tokenizer
  3) Build BM25Okapi index per (repo, commit) — shared across all queries hitting the shard
  4) Query with issue_text -> top-(K + |GT|) function _ids
  5) pos_ids = intersection(GT, shard._ids); neg_ids = top-K excluding pos_ids
  6) Emit {query_id, query_text, repo, base_commit, pos_ids, neg_ids, gt_missing[]}

Missing GT policy:
  - Partial GT survives -> record gt_missing (and include instance with surviving pos)
  - All GT missing -> log to --drop_log, skip from main output

Deterministic: seed=42, sorted shard order, stable tie-break on BM25 ties.
Writes only under /data/chenlibin/codegrip_func/.
"""
import os, re, sys, json, time, argparse, random
from collections import defaultdict
from typing import Dict, List, Tuple
from multiprocessing import Pool

import numpy as np
from rank_bm25 import BM25Okapi

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def shard_path(corpus_dir: str, repo: str, commit: str) -> str:
    return os.path.join(corpus_dir, f"{repo}__{commit[:12]}.jsonl")


# Tokenizer verbatim from build_swebench_verified_bm25_strict.py (Codex-audited GO).
_STOP = {'the','and','for','not','but','are','was','has','had','can','may',
         'use','def','class','self','return','import','from','if','else','elif',
         'try','except','with','as','in','is','or','none','true','false','pass',
         'raise','this','that','will','would','should','could'}

def tok_code(t):
    t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
    t = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', t)
    t = re.sub(r'[_/\-.]', ' ', t)
    return [x for x in re.findall(r'[a-zA-Z][a-zA-Z0-9]*', t.lower())
            if len(x) > 1 and x not in _STOP]

def tok_path(p):
    p = re.sub(r'\.py$', '', p)
    out = []
    for part in re.split(r'[/_\-.]', p):
        sub = re.sub(r'([a-z])([A-Z])', r'\1 \2', part)
        out.extend(sub.lower().split())
    return [x for x in out if len(x) > 1]

def tok_query(t):
    toks = tok_code(t)
    for ref in re.findall(r'[\w/]+\.py\b', t):
        toks.extend(tok_path(ref) * 3)
    for q in re.findall(r'[`\'"](\w+)[`\'"]', t):
        if len(q) > 1: toks.append(q.lower())
    for d in re.findall(r'\b\w+(?:\.\w+){2,}\b', t):
        toks.extend(d.lower().split('.'))
    return toks


def tokenize_function_record(_id: str, text: str) -> List[str]:
    """Function-level adaptation of strict BM25's tok_doc:
    weights _id (path/class/method) tokens 3x and concatenates code body tokens.
    Strict BM25 uses tok_doc(path, content) = tok_path(path)*3 + tok_code(content);
    here we pass _id (which IS the path/class/method composite) as the path."""
    id_tokens = tok_path(_id) * 3
    if text.startswith(_id + "\n"):
        body = text[len(_id) + 1:]
    else:
        body = text
    body_tokens = tok_code(body)
    return id_tokens + body_tokens


def load_shard_tokens(shard_file: str) -> Tuple[List[str], List[List[str]]]:
    ids: List[str] = []
    docs: List[List[str]] = []
    seen = set()
    with open(shard_file) as f:
        for line in f:
            rec = json.loads(line)
            fid = rec["_id"]
            if fid in seen:
                continue
            seen.add(fid)
            ids.append(fid)
            docs.append(tokenize_function_record(fid, rec.get("text", "")))
    return ids, docs


def process_shard(task: Dict) -> List[Dict]:
    shard_file = task["shard_file"]
    queries = task["queries"]
    top_k = task["top_k"]

    if not os.path.isfile(shard_file):
        return [{"query_id": q["query_id"], "error": "shard_missing", "shard": shard_file} for q in queries]

    ids, docs = load_shard_tokens(shard_file)
    if not docs:
        return [{"query_id": q["query_id"], "error": "empty_shard", "shard": shard_file} for q in queries]
    bm25 = BM25Okapi(docs)
    id_to_idx = {fid: i for i, fid in enumerate(ids)}
    id_set = set(ids)

    out: List[Dict] = []
    for q in queries:
        gt = list(q["gt_ids"])
        gt_present = [g for g in gt if g in id_set]
        gt_missing = [g for g in gt if g not in id_set]

        if not gt_present:
            out.append({
                "query_id": q["query_id"], "repo": q["repo"], "base_commit": q["commit"],
                "error": "all_gt_missing", "gt_missing": gt_missing,
            })
            continue

        query_tokens = tok_query(q["issue_text"])
        scores = bm25.get_scores(query_tokens)
        order = np.lexsort((np.arange(len(scores)), -scores))
        pos_idx_set = {id_to_idx[g] for g in gt_present}

        neg_ids: List[str] = []
        for idx in order:
            if idx in pos_idx_set:
                continue
            neg_ids.append(ids[idx])
            if len(neg_ids) >= top_k:
                break

        out.append({
            "query_id": q["query_id"], "query_text": q["issue_text"],
            "repo": q["repo"], "base_commit": q["commit"],
            "pos_ids": gt_present, "neg_ids": neg_ids,
            "gt_missing": gt_missing, "shard_n_funcs": len(ids),
        })
    return out


def load_gt(paths: List[str]) -> List[Dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                commit = r.get("base_commit") or r.get("head_commit") or ""
                rows.append({
                    "query_id": r["issue_id"], "repo": r["repo"],
                    "commit": commit, "issue_text": r["issue_text"],
                    "gt_ids": r["gt_func_ids"],
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_files", nargs="+", required=True)
    ap.add_argument("--corpus_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--drop_log", required=True)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    if not args.output.startswith("/data/"):
        sys.exit(f"ERROR: --output must be under /data/ (got: {args.output})")
    if not args.drop_log.startswith("/data/"):
        sys.exit(f"ERROR: --drop_log must be under /data/ (got: {args.drop_log})")

    print(f"[seed={SEED}] loading GT from {args.gt_files}")
    rows = load_gt(args.gt_files)
    print(f"  {len(rows)} queries")

    shard_queries: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        shard_queries[shard_path(args.corpus_dir, r["repo"], r["commit"])].append(r)
    shard_list = sorted(shard_queries.keys())
    print(f"  {len(shard_list)} unique shards")

    tasks = [{"shard_file": s, "queries": shard_queries[s], "top_k": args.top_k} for s in shard_list]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.drop_log), exist_ok=True)
    t0 = time.time()
    n_ok = n_drop = n_partial = 0

    with open(args.output, "w") as fout, open(args.drop_log, "w") as fdrop:
        if args.num_workers > 1:
            with Pool(args.num_workers) as pool:
                # imap (ordered) — preserves shard_list order for deterministic output
                for i, results in enumerate(pool.imap(process_shard, tasks)):
                    for rec in results:
                        if rec.get("error"):
                            fdrop.write(json.dumps(rec) + "\n"); n_drop += 1
                        else:
                            if rec.get("gt_missing"): n_partial += 1
                            fout.write(json.dumps(rec) + "\n"); n_ok += 1
                    fout.flush(); fdrop.flush()
                    if (i + 1) % 50 == 0:
                        el = time.time() - t0
                        print(f"  shard {i+1}/{len(tasks)}  ok={n_ok} drop={n_drop} partial={n_partial}  elapsed={el:.0f}s", flush=True)
        else:
            for task in tasks:
                for rec in process_shard(task):
                    if rec.get("error"):
                        fdrop.write(json.dumps(rec) + "\n"); n_drop += 1
                    else:
                        if rec.get("gt_missing"): n_partial += 1
                        fout.write(json.dumps(rec) + "\n"); n_ok += 1

    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"  ok={n_ok}  dropped={n_drop}  partial_gt={n_partial}")
    print(f"  output: {args.output}")
    print(f"  drop log: {args.drop_log}")


if __name__ == "__main__":
    main()
