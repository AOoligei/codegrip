#!/usr/bin/env python3
"""Per-base-commit BM25 candidates for SWE-bench Verified (strict).

For each unique (repo, base_commit), git-worktree the commit, BM25-index
its .py files (path*3 + first 200 content lines), query with issue_text,
emit top-500 paths. Tokenizer matches scripts/swebench_bm25_content.py.

Reproducibility: random.seed(42), np.random.seed(42), deterministic
sorting, deterministic tokenizer ordering. No GPU.
"""
import argparse, json, os, random, re, shutil, subprocess, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from rank_bm25 import BM25Okapi

random.seed(42); np.random.seed(42)

REPOS_DIR = "/home/chenlibin/grepo_agent/data/swebench_lite/repos"
WORKTREE_ROOT = "/data/chenlibin/tmp/verified_worktrees"
PREPARED = "/data/chenlibin/grepo_agent_experiments/swebench_verified/swebench_verified_prepared.jsonl"
OUT_DIR = "/data/chenlibin/grepo_agent_experiments/swebench_verified_strict"
OUT_JSONL = f"{OUT_DIR}/swebench_verified_bm25_strict.jsonl"
CKPT_JSONL = f"{OUT_DIR}/.checkpoint.jsonl"
TOP_K = 500
MAX_LINES = 200

# --- TOKENIZER (verbatim from swebench_bm25_content.py) ---
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

def tok_doc(path, content):
    return tok_path(path) * 3 + tok_code(content)

def tok_query(t):
    toks = tok_code(t)
    for ref in re.findall(r'[\w/]+\.py\b', t):
        toks.extend(tok_path(ref) * 3)
    for q in re.findall(r'[`\'"](\w+)[`\'"]', t):
        if len(q) > 1: toks.append(q.lower())
    for d in re.findall(r'\b\w+(?:\.\w+){2,}\b', t):
        toks.extend(d.lower().split('.'))
    return toks

def read_head(fp):
    try:
        with open(fp, 'r', errors='replace') as f:
            return '\n'.join(line.rstrip() for i, line in enumerate(f) if i < MAX_LINES)
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return ''

# --- WORKTREE LIFECYCLE ---
def make_worktree(repo, sha):
    src = os.path.join(REPOS_DIR, repo)
    wt  = os.path.join(WORKTREE_ROOT, f"{repo}_{sha[:12]}")
    if os.path.exists(wt):
        subprocess.run(["git", "-C", src, "worktree", "remove", "--force", wt],
                       check=False, capture_output=True)
        if os.path.exists(wt): shutil.rmtree(wt, ignore_errors=True)
    r = subprocess.run(["git", "-C", src, "worktree", "add", "--detach", wt, sha],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"worktree add failed for {repo}@{sha}: {r.stderr}")
    return wt, src

def drop_worktree(src, wt):
    subprocess.run(["git", "-C", src, "worktree", "remove", "--force", wt],
                   check=False, capture_output=True)
    if os.path.exists(wt): shutil.rmtree(wt, ignore_errors=True)
    subprocess.run(["git", "-C", src, "worktree", "prune"], check=False, capture_output=True)

# --- PER-COMMIT JOB ---
def process_commit(repo, sha, instances):
    """instances: list of dicts {issue_id, issue_text, ground_truth}"""
    wt, src = make_worktree(repo, sha)
    try:
        files = []
        for root, dirs, fs in os.walk(wt):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d != '__pycache__')
            for f in sorted(fs):
                if f.endswith('.py'):
                    files.append(os.path.relpath(os.path.join(root, f), wt))
        files.sort()  # deterministic doc order
        docs, valid = [], []
        for fp in files:
            toks = tok_doc(fp, read_head(os.path.join(wt, fp)))
            if toks:
                docs.append(toks); valid.append(fp)
        if not docs:
            return [{"repo": repo, "issue_id": ix["issue_id"], "issue_text": ix["issue_text"],
                     "ground_truth": ix["ground_truth"], "bm25_candidates": [],
                     "gt_in_candidates": False, "n_indexed_files": 0,
                     "base_commit": sha} for ix in instances]
        bm25 = BM25Okapi(docs)
        out = []
        for ix in instances:
            scores = bm25.get_scores(tok_query(ix["issue_text"]))
            top = np.argsort(-scores, kind='stable')[:TOP_K]
            cands = [valid[i] for i in top]
            gt = set(ix["ground_truth"])
            out.append({"repo": repo, "issue_id": ix["issue_id"],
                        "issue_text": ix["issue_text"],
                        "ground_truth": list(gt),
                        "bm25_candidates": cands,
                        "gt_in_candidates": bool(gt & set(cands)),
                        "n_indexed_files": len(valid),
                        "base_commit": sha})
        return out
    finally:
        drop_worktree(src, wt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(WORKTREE_ROOT, exist_ok=True)

    recs = [json.loads(l) for l in open(PREPARED)]
    groups = defaultdict(list)
    for r in recs:
        groups[(r["repo"], r["base_commit"])].append({
            "issue_id": r.get("issue_id") or r.get("instance_id"),
            "issue_text": r["issue_text"],
            "ground_truth": list(r.get("changed_py_files", [])),
        })
    keys = sorted(groups.keys())
    print(f"unique (repo, sha): {len(keys)}; instances: {len(recs)}", flush=True)

    done = set()
    if args.resume and os.path.exists(CKPT_JSONL):
        # tolerate malformed tail line from a prior crash
        for line in open(CKPT_JSONL):
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARN: dropped malformed checkpoint line (likely truncated tail)", flush=True)
                continue
            done.add((d["repo"], d["base_commit"], str(d["issue_id"])))
        print(f"resume: {len(done)} instances already done", flush=True)
    elif os.path.exists(CKPT_JSONL):
        # fresh run: clear stale checkpoint to avoid contamination
        os.remove(CKPT_JSONL)
        print("Fresh run: cleared stale checkpoint", flush=True)

    pending = [(r, s) for (r, s) in keys
               if not all((r, s, str(ix["issue_id"])) in done for ix in groups[(r, s)])]
    print(f"pending commits: {len(pending)}", flush=True)

    # Pre-flight: every pending sha must be reachable in its source repo
    print("Pre-flight: checking commits exist locally...", flush=True)
    missing_shas = []
    for (repo, sha) in pending:
        src = os.path.join(REPOS_DIR, repo)
        r = subprocess.run(["git", "-C", src, "cat-file", "-e", f"{sha}^{{commit}}"],
                           capture_output=True)
        if r.returncode != 0:
            missing_shas.append((repo, sha))
    if missing_shas:
        print(f"WARN: {len(missing_shas)} commits not found locally; attempting fetch...", flush=True)
        for repo, sha in missing_shas[:5]:
            print(f"  missing: {repo}@{sha}", flush=True)
        # Try a single fetch per affected repo
        affected_repos = sorted({r for r, _ in missing_shas})
        for repo in affected_repos:
            src = os.path.join(REPOS_DIR, repo)
            print(f"  git fetch --all in {repo}...", flush=True)
            subprocess.run(["git", "-C", src, "fetch", "--all", "--tags"],
                           check=False, capture_output=True)
        # Recheck
        still_missing = []
        for repo, sha in missing_shas:
            src = os.path.join(REPOS_DIR, repo)
            r = subprocess.run(["git", "-C", src, "cat-file", "-e", f"{sha}^{{commit}}"],
                               capture_output=True)
            if r.returncode != 0:
                still_missing.append((repo, sha))
        if still_missing:
            print(f"ERROR: {len(still_missing)} commits unreachable even after fetch.", flush=True)
            for repo, sha in still_missing[:10]:
                print(f"  unreachable: {repo}@{sha}", flush=True)
            raise SystemExit("Aborting: cannot proceed with missing commits (would silently lose recall data)")

    t0 = time.time()
    ckpt = open(CKPT_JSONL, "a")
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_commit, r, s, groups[(r, s)]): (r, s) for (r, s) in pending}
        for f in as_completed(futs):
            r, s = futs[f]
            try:
                rows = f.result()
            except Exception as e:
                print(f"FAIL {r}@{s[:12]}: {e}", flush=True)
                ckpt.close()
                raise SystemExit(f"Aborting on commit failure to prevent silent recall loss")
            for row in rows:
                ckpt.write(json.dumps(row) + "\n")
            ckpt.flush()
            n_done += 1
            if n_done % 25 == 0:
                print(f"  {n_done}/{len(pending)} commits, {time.time()-t0:.0f}s elapsed", flush=True)
    ckpt.close()

    # Reorder checkpoint -> final jsonl in prepared.jsonl order
    # Tolerate malformed tail lines (e.g. from prior crash). Write via temp+rename for atomicity.
    rows = {}
    for line in open(CKPT_JSONL):
        line = line.strip()
        if not line: continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            print(f"WARN: dropped malformed checkpoint line during final assembly", flush=True)
            continue
        rows[(d["repo"], str(d["issue_id"]))] = d
    n_missing = 0
    OUT_TMP = OUT_JSONL + ".tmp"
    with open(OUT_TMP, "w") as f:
        for r in recs:
            iid = str(r.get("issue_id") or r.get("instance_id"))
            row = rows.get((r["repo"], iid))
            if row is None:
                n_missing += 1; continue
            f.write(json.dumps({k: row[k] for k in
                ("repo","issue_id","issue_text","ground_truth","bm25_candidates","gt_in_candidates")}) + "\n")
    if n_missing > 0:
        print(f"ERROR: {n_missing} instances missing from final output. Refusing to commit. Inspect checkpoint.", flush=True)
        os.remove(OUT_TMP)
        raise SystemExit(2)
    os.replace(OUT_TMP, OUT_JSONL)
    print(f"wrote {OUT_JSONL}; all {len(recs)} instances present", flush=True)

    # Metrics
    rs = [json.loads(l) for l in open(OUT_JSONL)]
    for k in (1,5,10,20,50,100,200,500):
        acc = np.mean([1.0 if set(x["ground_truth"]).issubset(set(x["bm25_candidates"][:k])) else 0.0 for x in rs]) * 100
        print(f"  Acc@{k}: {acc:.2f}%")
    print(f"  gt_in_top500: {np.mean([x['gt_in_candidates'] for x in rs])*100:.2f}%")

if __name__ == "__main__":
    main()
