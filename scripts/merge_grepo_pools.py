#!/usr/bin/env python3
"""Merge A8E (git historical) + A41 (HEAD-BM25) pools into a HEAD-training pool.

Strategy (full_fallback per Codex R1 audit):
  - GT = union(A8E GT, A41 GT), filtered to HEAD disk
  - Candidates = A8E (filtered to HEAD) if valid (>=5 + GT-in); else A41 (same check); else drop
  - Output: {repo, issue_id, issue_text, changed_py_files, changed_files, candidates, source}
"""
import argparse, json, os, collections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a8e", default="/home/chenlibin/grepo_agent/data/rankft/grepo_train_git_historical_candidates.jsonl")
    ap.add_argument("--a41", default="/home/chenlibin/grepo_agent/data/rankft/grepo_train_head_candidates.jsonl")
    ap.add_argument("--repos", default="/home/chenlibin/grepo_agent/data/repos")
    ap.add_argument("--output", default="/home/chenlibin/grepo_agent/data/rankft/grepo_train_hybrid_candidates.jsonl")
    args = ap.parse_args()

    a8e_idx = {}
    with open(args.a8e, encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            a8e_idx[(r["repo"], str(r["issue_id"]))] = r
    a41_idx = {}
    with open(args.a41, encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            a41_idx[(r["repo"], str(r["issue_id"]))] = r
    print(f"A8E: {len(a8e_idx)} rows; A41: {len(a41_idx)} rows")

    def uniq(seq):
        out, seen = [], set()
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    def on_disk(repo, paths):
        rd = os.path.join(args.repos, repo)
        return [p for p in paths if os.path.isfile(os.path.join(rd, p))]

    stats = collections.defaultdict(int)
    merged = []
    keys = set(a8e_idx) | set(a41_idx)
    for k in sorted(keys):
        r8 = a8e_idx.get(k)
        r41 = a41_idx.get(k)
        if not r8 and not r41:
            stats["no_source"] += 1; continue
        repo = (r8 or r41)["repo"]

        gt_union = uniq((r8 or {}).get("ground_truth", []) +
                        (r8 or {}).get("changed_py_files", []) +
                        (r41 or {}).get("ground_truth", []) +
                        (r41 or {}).get("changed_py_files", []))
        gt_on_disk = on_disk(repo, gt_union)
        if not gt_on_disk:
            stats["no_gt_on_disk"] += 1; continue

        c8_raw = (r8 or {}).get("candidates", []) if (r8 and r8.get("skipped") != "shallow_clone") else []
        c8 = on_disk(repo, c8_raw)
        c41 = on_disk(repo, (r41 or {}).get("candidates", []))

        sgt = set(gt_on_disk)
        valid8 = len(c8) >= 5 and bool(sgt & set(c8))
        valid41 = len(c41) >= 5 and bool(sgt & set(c41))

        if valid8:
            chosen_cands, src = c8, "a8e_git"
        elif valid41:
            chosen_cands, src = c41, "a41_head"
        else:
            stats["neither_valid"] += 1; continue

        stats[f"ok_{src}"] += 1
        # A8E rows do not consistently carry issue_text; prefer the HEAD row when present.
        issue_text = ((r41 or {}).get("issue_text") or (r8 or {}).get("issue_text") or "")
        merged.append({
            "repo": repo,
            "issue_id": (r8 or r41)["issue_id"],
            "issue_text": issue_text,
            "changed_py_files": gt_on_disk,
            "changed_files": gt_on_disk,
            "candidates": chosen_cands,
            "source": src,
        })

    with open(args.output, "w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")
    print(f"\n=== Merge stats ===")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"\nFinal merged: {len(merged)} rows → {args.output}")


if __name__ == "__main__":
    main()
