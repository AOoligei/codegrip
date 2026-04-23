#!/usr/bin/env python3
"""Clean training pools: drop poisoned rows before codeaware training.

Inputs:
  --grepo_candidates: A41 output (grepo_train_head_candidates.jsonl)
  --swe_train: SWE-bench train data
  --output: clean combined training jsonl

Cleaning rules:
  1. ground_truth non-empty
  2. gt_in_candidates == True
  3. at least 5 candidates
  4. all candidates readable on disk
  5. no duplicate (repo, issue_id) across sources (SWE-bench wins on conflict)
  6. seed 42, stable sort
"""
import argparse, json, os, random, hashlib, collections

random.seed(42)

def check_row(rec, repo_root):
    """Return (clean, reason, updated_rec).
    Filters GT to HEAD-existing (fixes bughunter bug #1).
    """
    gt_raw = rec.get("ground_truth") or rec.get("changed_py_files") or rec.get("changed_files") or []
    if not gt_raw: return False, "empty_gt_raw", rec
    repo = rec.get("repo", "")
    rd = os.path.join(repo_root, repo)
    gt_on_disk = [g for g in gt_raw if os.path.isfile(os.path.join(rd, g))]
    if not gt_on_disk: return False, "no_gt_on_disk", rec
    cands = rec.get("candidates") or rec.get("bm25_candidates") or []
    if len(cands) < 5: return False, "too_few_cands", rec
    gt_in = any(g in set(cands) for g in gt_on_disk)
    if not gt_in: return False, "gt_not_in_cands", rec
    miss = [c for c in cands if not os.path.isfile(os.path.join(rd, c))]
    if len(miss) > 5: return False, f"{len(miss)}_cands_missing", rec
    new_rec = dict(rec)
    new_rec["changed_py_files"] = gt_on_disk
    new_rec["changed_files"] = gt_on_disk
    new_rec["ground_truth"] = gt_on_disk
    new_rec["_n_gt_filtered"] = len(gt_raw) - len(gt_on_disk)
    return True, "ok", new_rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grepo_candidates", default="/home/chenlibin/grepo_agent/data/rankft/grepo_train_head_candidates.jsonl")
    ap.add_argument("--grepo_repos", default="/home/chenlibin/grepo_agent/data/repos")
    ap.add_argument("--swe_train", default="/home/chenlibin/grepo_agent/data/swebench_train/swebench_train.jsonl")
    ap.add_argument("--swe_bm25", default="/home/chenlibin/grepo_agent/data/rankft/swebench_bm25_final_top500.jsonl",
                    help="SWE-bench train candidate pool (if exists); if not found, use training data's bm25 if embedded")
    ap.add_argument("--swe_repos", default="/home/chenlibin/grepo_agent/data/swebench_lite/repos")
    ap.add_argument("--output", default="/home/chenlibin/grepo_agent/data/rankft/clean_train_combined.jsonl")
    ap.add_argument("--grepo_only_output", default="/home/chenlibin/grepo_agent/data/rankft/clean_grepo_train.jsonl")
    ap.add_argument("--swe_only_output", default="/home/chenlibin/grepo_agent/data/rankft/clean_swe_train.jsonl")
    args = ap.parse_args()

    stats = collections.defaultdict(int)

    # --- GREPO pool ---
    print("[grepo] scanning...", flush=True)
    grepo_clean = []
    with open(args.grepo_candidates) as f:
        for line in f:
            rec = json.loads(line)
            ok, reason, new_rec = check_row(rec, args.grepo_repos)
            stats[f"grepo_{reason}"] += 1
            if ok:
                grepo_clean.append({
                    "repo": new_rec["repo"],
                    "issue_id": new_rec["issue_id"],
                    "issue_text": new_rec.get("issue_text", ""),
                    "changed_py_files": new_rec["changed_py_files"],
                    "changed_files": new_rec["changed_files"],
                    "candidates": new_rec["candidates"],
                    "source": "grepo",
                })
    print(f"[grepo] input: {stats['grepo_ok']+sum(v for k,v in stats.items() if k.startswith('grepo_') and k!='grepo_ok')}, clean: {len(grepo_clean)}", flush=True)

    # --- SWE-bench train pool (already known clean from memory, 2.8% miss) ---
    print("[swe] scanning...", flush=True)
    swe_clean = []
    swe_bm25_idx = {}
    if os.path.isfile(args.swe_bm25):
        for line in open(args.swe_bm25):
            r = json.loads(line)
            swe_bm25_idx[(r["repo"], str(r["issue_id"]))] = r.get("bm25_candidates", r.get("candidates", []))
    if os.path.isfile(args.swe_train):
        for line in open(args.swe_train):
            rec = json.loads(line)
            key = (rec.get("repo", ""), str(rec.get("issue_id", "")))
            cands = swe_bm25_idx.get(key) or rec.get("bm25_candidates", rec.get("candidates", []))
            merged = dict(rec); merged["candidates"] = cands
            ok, reason, new_rec = check_row(merged, args.swe_repos)
            stats[f"swe_{reason}"] += 1
            if ok:
                swe_clean.append({
                    "repo": new_rec["repo"],
                    "issue_id": new_rec["issue_id"],
                    "issue_text": new_rec.get("issue_text", ""),
                    "changed_py_files": new_rec["changed_py_files"],
                    "changed_files": new_rec["changed_files"],
                    "candidates": cands,
                    "source": "swe",
                })
    print(f"[swe] input: {stats['swe_ok']+sum(v for k,v in stats.items() if k.startswith('swe_') and k!='swe_ok')}, clean: {len(swe_clean)}", flush=True)

    # --- Dedupe & combine ---
    seen = set()
    combined = []
    # SWE first (priority), then GREPO
    for rec in swe_clean + grepo_clean:
        key = (rec["repo"], str(rec["issue_id"]))
        if key in seen: continue
        seen.add(key); combined.append(rec)
    combined.sort(key=lambda r: (r["repo"], str(r["issue_id"])))

    # Write outputs
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    for path, rows in [(args.grepo_only_output, grepo_clean), (args.swe_only_output, swe_clean), (args.output, combined)]:
        rows_sorted = sorted(rows, key=lambda r: (r["repo"], str(r["issue_id"])))
        with open(path, "w") as f:
            for r in rows_sorted: f.write(json.dumps(r) + "\n")
        print(f"[write] {path}: {len(rows_sorted)} rows", flush=True)

    # Stats
    print("\n=== Cleaning stats ===")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"\nFinal: GREPO-only={len(grepo_clean)}, SWE-only={len(swe_clean)}, combined-dedup={len(combined)}")


if __name__ == "__main__":
    main()
