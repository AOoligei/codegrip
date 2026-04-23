#!/usr/bin/env python3
"""SPHT v2: better data design.
- Hard examples: PARTIAL hash (only 1-2 path components, not all)
- Easy examples: 10% augmented with random shuffles for regularization
- Output ready for train_rankft_code_residual.py (with --code in input)"""
import argparse, hashlib, json, os, random, re

random.seed(42)

def tokenize(s):
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", (s or "").lower()))

def hash_part(x):
    h = hashlib.sha256(x.encode()).hexdigest()[:8]
    if x.endswith(".py"):
        return f"m_{h}.py"
    return f"d_{h}"

def partial_hash_path(p, frac=0.5):
    """Hash a random subset of path components (frac of them)."""
    parts = p.split("/")
    n = len(parts)
    n_hash = max(1, int(n * frac))
    indices = sorted(random.sample(range(n), n_hash))
    return "/".join(hash_part(part) if i in indices else part for i, part in enumerate(parts))

def jaccard(a, b):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data", default="/home/chenlibin/grepo_agent/data/grepo_text/grepo_train.jsonl")
    ap.add_argument("--bm25_cands", default="/home/chenlibin/grepo_agent/data/rankft/grepo_train_bm25_top500.jsonl")
    ap.add_argument("--output_data", required=True)
    ap.add_argument("--output_bm25", required=True)
    ap.add_argument("--hard_frac", type=float, default=0.30)
    ap.add_argument("--partial_hash_frac", type=float, default=0.5,
                    help="For hard examples, hash this fraction of path components")
    ap.add_argument("--easy_aug_frac", type=float, default=0.10,
                    help="For easy examples, randomly hash 1-2 components in this fraction (regularization)")
    args = ap.parse_args()

    bm25 = {}
    with open(args.bm25_cands) as f:
        for l in f:
            r = json.loads(l)
            bm25[(r["repo"], r["issue_id"])] = r.get("bm25_candidates", r.get("candidates", []))

    data = [json.loads(l) for l in open(args.train_data)]
    records = []
    for rec in data:
        gt = rec.get("changed_py_files", rec.get("changed_files", []))
        if not gt: continue
        issue_tok = tokenize(rec["issue_text"])
        max_jac = max((jaccard(issue_tok, tokenize(g)) for g in gt), default=0)
        cands = bm25.get((rec["repo"], rec["issue_id"]), [])
        gt_rank = next((i for i, c in enumerate(cands[:200]) if c in set(gt)), 999)
        records.append({**rec, "_jac": max_jac, "_rank": gt_rank, "_cands": cands})

    records.sort(key=lambda r: (r["_jac"], -r["_rank"]))  # hardest first
    n_hard = int(len(records) * args.hard_frac)

    out_data = open(args.output_data, "w")
    out_bm25 = open(args.output_bm25, "w")

    n_partial = n_full_keep = n_easy_aug = 0
    for i, r in enumerate(records):
        is_hard = i < n_hard
        cands = r["_cands"]
        if is_hard:
            # PARTIAL hash: keep some path tokens, hash others
            mp = {p: partial_hash_path(p, args.partial_hash_frac)
                  for p in set(r["changed_py_files"]) | set(cands)}
            new_gt = [mp[p] for p in r["changed_py_files"]]
            new_cands = [mp.get(c, c) for c in cands]
            n_partial += 1
        elif random.random() < args.easy_aug_frac:
            # easy aug: hash 1 random component (regularization)
            mp = {p: partial_hash_path(p, 1.0/max(1,len(p.split("/"))))
                  for p in set(r["changed_py_files"]) | set(cands)}
            new_gt = [mp[p] for p in r["changed_py_files"]]
            new_cands = [mp.get(c, c) for c in cands]
            n_easy_aug += 1
        else:
            new_gt = r["changed_py_files"]
            new_cands = cands
            n_full_keep += 1

        out_data.write(json.dumps({
            "repo": r["repo"], "issue_id": r["issue_id"],
            "issue_text": r["issue_text"],
            "changed_py_files": new_gt, "changed_files": new_gt,
            "is_hard_spht": is_hard,
        }) + "\n")
        out_bm25.write(json.dumps({
            "repo": r["repo"], "issue_id": r["issue_id"],
            "candidates": new_cands, "bm25_candidates": new_cands,
        }) + "\n")

    out_data.close(); out_bm25.close()
    print(f"Wrote {len(records)} records.")
    print(f"  Hard (partial hash {args.partial_hash_frac:.0%}): {n_partial}")
    print(f"  Easy aug (single-component hash): {n_easy_aug}")
    print(f"  Easy untouched: {n_full_keep}")

if __name__ == "__main__":
    main()
