#!/usr/bin/env python3
"""Build SPHT training data: hash paths for HARD examples, leave EASY unchanged.

Hard = low issue-path Jaccard overlap OR low BM25 rank.
Easy = high overlap / BM25 top-1 is GT.
"""
import argparse, hashlib, json, os, re, sys

def tokenize(s):
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", (s or "").lower()))

def hash_path(p):
    parts = []
    for x in p.split("/"):
        if not x: continue
        h = hashlib.sha256(x.encode()).hexdigest()[:8]
        parts.append(f"m_{h}.py" if x.endswith(".py") else f"d_{h}")
    return "/".join(parts)

def jaccard(a, b):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data", default="/home/chenlibin/grepo_agent/data/grepo_text/grepo_train.jsonl")
    ap.add_argument("--bm25_cands", default="/home/chenlibin/grepo_agent/data/rankft/grepo_train_bm25_top500.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--jaccard_thresh", type=float, default=0.15)
    ap.add_argument("--hard_frac_target", type=float, default=0.30, help="target fraction of hard examples")
    args = ap.parse_args()

    # Load BM25 candidates to check rank of GT
    bm25 = {}
    with open(args.bm25_cands) as f:
        for line in f:
            r = json.loads(line)
            cands = r.get("candidates", r.get("bm25_candidates", []))
            bm25[(r["repo"], r["issue_id"])] = cands

    data = [json.loads(l) for l in open(args.train_data)]
    print(f"Loaded {len(data)} train examples")

    # Compute difficulty signals
    records = []
    for rec in data:
        repo = rec["repo"]; iid = rec["issue_id"]
        issue_tok = tokenize(rec["issue_text"])
        gt_files = rec.get("changed_py_files", rec.get("changed_files", []))
        if not gt_files: continue

        # issue-path jaccard (max over GT files)
        max_jac = 0.0
        for gt in gt_files:
            path_tok = tokenize(gt)
            max_jac = max(max_jac, jaccard(issue_tok, path_tok))

        # BM25 rank of GT (min rank over all GT)
        cands = bm25.get((repo, iid), [])
        gt_rank = 999
        for r_i, c in enumerate(cands[:200]):
            if c in set(gt_files):
                gt_rank = min(gt_rank, r_i); break

        records.append({
            "repo": repo, "issue_id": iid,
            "issue_text": rec["issue_text"],
            "changed_py_files": gt_files,
            "jaccard": max_jac,
            "bm25_rank": gt_rank,
        })

    # Sort by difficulty: low jaccard AND high BM25 rank = hardest
    # Simpler: rank by jaccard, mark bottom frac as hard
    records.sort(key=lambda r: (r["jaccard"], -r["bm25_rank"]))  # hardest first
    n_hard = int(len(records) * args.hard_frac_target)

    hard_ids = set()
    for i, r in enumerate(records):
        if i < n_hard:
            r["is_hard"] = True; hard_ids.add((r["repo"], r["issue_id"]))
        else:
            r["is_hard"] = False

    # Stats
    hard_recs = [r for r in records if r["is_hard"]]
    easy_recs = [r for r in records if not r["is_hard"]]
    print(f"Hard: {len(hard_recs)} examples")
    print(f"  avg jaccard: {sum(r['jaccard'] for r in hard_recs)/len(hard_recs):.3f}")
    print(f"  avg bm25_rank: {sum(r['bm25_rank'] for r in hard_recs)/len(hard_recs):.1f}")
    print(f"Easy: {len(easy_recs)} examples")
    print(f"  avg jaccard: {sum(r['jaccard'] for r in easy_recs)/len(easy_recs):.3f}")
    print(f"  avg bm25_rank: {sum(r['bm25_rank'] for r in easy_recs)/len(easy_recs):.1f}")

    # Write mixed training file
    # For hard examples: hash the paths of GT and candidates
    # For easy: keep as-is
    # Also need to emit the BM25 candidates (for the training pipeline to sample negatives from)
    out_bm25_cands = {}
    with open(args.output, "w") as fout:
        for r in records:
            rec_out = {
                "repo": r["repo"],
                "issue_id": r["issue_id"],
                "issue_text": r["issue_text"],
                "is_hard_spht": r["is_hard"],
                "jaccard": r["jaccard"],
                "bm25_rank": r["bm25_rank"],
            }
            cands = bm25.get((r["repo"], r["issue_id"]), [])
            if r["is_hard"]:
                # Hash all paths: GT + candidates
                rename_map = {}
                all_paths = set(r["changed_py_files"]) | set(cands)
                for p in all_paths:
                    rename_map[p] = hash_path(p)
                rec_out["changed_py_files"] = [rename_map[p] for p in r["changed_py_files"]]
                rec_out["bm25_candidates"] = [rename_map.get(p, hash_path(p)) for p in cands]
                rec_out["rename_map"] = rename_map
            else:
                rec_out["changed_py_files"] = r["changed_py_files"]
                rec_out["bm25_candidates"] = cands
            fout.write(json.dumps(rec_out) + "\n")

    print(f"\nWrote {len(records)} records to {args.output}")

if __name__ == "__main__":
    main()
