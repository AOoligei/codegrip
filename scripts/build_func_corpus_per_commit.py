#!/usr/bin/env python3
"""Step 2: per-commit function-level corpus shards, byte-compatible with SweRank.

For each unique (repo, commit) across func_gt_swe.jsonl + func_gt_grepo.jsonl:
  1. git-worktree the commit (race-safe via fcntl lock per source repo),
  2. AST-walk every .py file; emit FunctionDef records (depth 1) and ClassDef.body
     FunctionDef records (depth 2). AsyncFunctionDef EXCLUDED to match SweRank.
  3. Render text exactly as SweRank: `_id\\n<body>` where body for methods is
     `class X:` literally concatenated (no newline) with method source lines.
  4. Atomic-rename shard to /data/chenlibin/codegrip_func/corpus/<repo>__<sha[:12]>.jsonl.

Reproducibility: seed 42, sorted dir/file traversal, deterministic record order
(class_body order within class; top-level AST order within module). No GPU.
Reuses worktree lifecycle patterns from build_swebench_verified_bm25_strict.py.

Writes only under /data/chenlibin/codegrip_func/corpus/; aborts on any per-commit
failure to prevent silent GT loss.
"""
import argparse, ast, contextlib, fcntl, json, os, random, shutil, subprocess, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

random.seed(42)

REPOS_DIR_SWE   = "/home/chenlibin/grepo_agent/data/swebench_lite/repos"
REPOS_DIR_GREPO = "/home/chenlibin/grepo_agent/data/repos"
WORKTREE_ROOT   = "/data/chenlibin/tmp/func_corpus_worktrees"
GT_SWE          = "/data/chenlibin/codegrip_func/func_gt_swe.jsonl"
GT_GREPO        = "/data/chenlibin/codegrip_func/func_gt_grepo.jsonl"
OUT_DIR         = Path("/data/chenlibin/codegrip_func/corpus")
STATS_DIR       = Path("/data/chenlibin/codegrip_func/corpus_stats")
CKPT_PATH       = Path("/data/chenlibin/codegrip_func/corpus/.checkpoint.jsonl")
MIN_FREE_GB     = 60.0  # refuse to start below this


def repo_dir_for(repo: str) -> str:
    p1 = os.path.join(REPOS_DIR_SWE, repo)
    if os.path.isdir(p1): return p1
    p2 = os.path.join(REPOS_DIR_GREPO, repo)
    if os.path.isdir(p2): return p2
    raise RuntimeError(f"repo not found under SWE or GREPO roots: {repo}")


@contextlib.contextmanager
def source_repo_lock(src_dir: str):
    lock_fp = os.path.join(src_dir, ".git", ".codegrip_worktree.lock")
    fd = os.open(lock_fp, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

def make_worktree(repo: str, sha: str):
    src = repo_dir_for(repo)
    wt  = os.path.join(WORKTREE_ROOT, f"{repo}_{sha[:12]}")
    with source_repo_lock(src):
        if os.path.exists(wt):
            subprocess.run(["git", "-C", src, "worktree", "remove", "--force", wt],
                           check=False, capture_output=True)
            if os.path.exists(wt): shutil.rmtree(wt, ignore_errors=True)
        r = subprocess.run(["git", "-C", src, "worktree", "add", "--detach", wt, sha],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"worktree add failed for {repo}@{sha}: {r.stderr}")
    return wt, src

def drop_worktree(src: str, wt: str):
    with source_repo_lock(src):
        subprocess.run(["git", "-C", src, "worktree", "remove", "--force", wt],
                       check=False, capture_output=True)
        if os.path.exists(wt): shutil.rmtree(wt, ignore_errors=True)
        subprocess.run(["git", "-C", src, "worktree", "prune"], check=False, capture_output=True)


def extract_records(file_path_rel: str, source: str, mode: str):
    """Return (list of dicts, n_parse_fail, n_funcs, n_methods).
    SweRank-mode: depth<=2, no AsyncFunctionDef, method text = `class X:` + body lines (no newline between).
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, MemoryError):
        return [], 1, 0, 0
    lines = source.splitlines()

    def body_text(node):
        return "\n".join(lines[node.lineno - 1 : node.end_lineno])

    records, n_func, n_method = [], 0, 0

    if mode == "swerank":
        # Byte-compatible with SweRank's parse_python_file() in get_repo_structure.py:
        #   - ast.walk() visits ALL nodes recursively
        #   - ClassDef: collect direct-child FunctionDef as methods (method names -> class_methods set)
        #   - Any FunctionDef (NOT AsyncFunctionDef) whose name is NOT in class_methods -> top-level func
        # Note: this deduplicates by FUNCTION NAME, not path. Collisions matter.
        class_methods_set = set()
        # First pass: gather all class methods (direct children only)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for n in node.body:
                    if isinstance(n, ast.FunctionDef):
                        class_methods_set.add(n.name)
        # Second pass: emit records. ClassDef methods first (discovered order), then free funcs.
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for m in node.body:
                    if isinstance(m, ast.FunctionDef):
                        _id = f"{file_path_rel}/{node.name}/{m.name}"
                        mtxt = body_text(m)
                        text = f"{_id}\nclass {node.name}:{mtxt}"
                        records.append({"_id": _id, "title": "", "text": text,
                                        "metadata": {}})
                        n_method += 1
            elif isinstance(node, ast.FunctionDef) and not isinstance(node, ast.AsyncFunctionDef):
                if node.name not in class_methods_set:
                    _id = f"{file_path_rel}/{node.name}"
                    records.append({"_id": _id, "title": "",
                                    "text": f"{_id}\n{body_text(node)}",
                                    "metadata": {}})
                    n_func += 1
        return records, 0, n_func, n_method

    # extended mode: include async + nested
    nonlocal_n_func, nonlocal_n_method = [0], [0]
    def walk(node, chain, cls_chain):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                new_chain = chain + [child.name]
                _id = f"{file_path_rel}/" + "/".join(new_chain)
                body = body_text(child)
                if cls_chain:
                    prefix = "\n".join(f"class {c}:" for c in cls_chain)
                    text = f"{_id}\n{prefix}{body}"
                    nonlocal_n_method[0] += 1
                else:
                    text = f"{_id}\n{body}"
                    nonlocal_n_func[0] += 1
                records.append({"_id": _id, "title": "", "text": text, "metadata": {}})
                walk(child, new_chain, cls_chain)
            elif isinstance(child, ast.ClassDef):
                walk(child, chain + [child.name], cls_chain + [child.name])
            else:
                walk(child, chain, cls_chain)
    walk(tree, [], [])
    return records, 0, nonlocal_n_func[0], nonlocal_n_method[0]


def process_commit(repo: str, sha: str, mode: str):
    shard = OUT_DIR / f"{repo}__{sha[:12]}.jsonl"
    stats_fp = STATS_DIR / f"{repo}__{sha[:12]}.stats.json"
    tmp = shard.with_suffix(".jsonl.tmp")
    wt, src = make_worktree(repo, sha)
    try:
        py_files = []
        for root, dirs, fs in os.walk(wt):
            # SweRank-style: skip test and hidden dirs
            dirs[:] = sorted(d for d in dirs
                              if not d.startswith('.')
                              and d != '__pycache__'
                              and d != 'tests'
                              and d != 'test')
            for f in sorted(fs):
                if f.endswith('.py'):
                    rel = os.path.relpath(os.path.join(root, f), wt)
                    # Also skip files whose path contains /test/ or /tests/
                    parts = rel.split('/')
                    if any(p in ('test', 'tests') for p in parts):
                        continue
                    py_files.append(rel)
        py_files.sort()

        n_files = len(py_files); n_parse_fail = 0
        n_funcs_total = n_methods_total = n_records = 0
        seen_ids = set(); n_dup_ids = 0

        with open(tmp, "w") as out:
            for rel in py_files:
                fp = os.path.join(wt, rel)
                try:
                    with open(fp, "r", errors="replace") as g:
                        src_text = g.read()
                except (PermissionError, IsADirectoryError, OSError):
                    continue
                recs, pf, nf, nm = extract_records(rel, src_text, mode)
                n_parse_fail += pf
                n_funcs_total += nf
                n_methods_total += nm
                for r in recs:
                    if r["_id"] in seen_ids:
                        n_dup_ids += 1
                    seen_ids.add(r["_id"])
                    out.write(json.dumps(r, ensure_ascii=True) + "\n")
                    n_records += 1
        os.replace(tmp, shard)

        stats = {"repo": repo, "commit": sha, "shard": str(shard),
                 "n_py_files": n_files, "n_parse_fail": n_parse_fail,
                 "n_funcs": n_funcs_total, "n_methods": n_methods_total,
                 "n_records": n_records, "n_dup_ids": n_dup_ids,
                 "mode": mode}
        with open(stats_fp, "w") as s:
            s.write(json.dumps(stats))
        return stats
    finally:
        drop_worktree(src, wt)


def free_gb(path: str) -> float:
    st = os.statvfs(path)
    return (st.f_bavail * st.f_frsize) / (1024**3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mode", choices=["swerank", "extended"], default="swerank")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gt-swe", default=GT_SWE)
    ap.add_argument("--gt-grepo", default=GT_GREPO)
    ap.add_argument("--limit-pairs", type=int, default=0,
                    help="for smoke test: cap pairs to this many")
    ap.add_argument("--filter-repo-commit", default=None,
                    help="for byte-match: only process this single 'repo,sha' pair")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    os.makedirs(WORKTREE_ROOT, exist_ok=True)

    fg = free_gb("/data/chenlibin")
    if fg < MIN_FREE_GB:
        raise SystemExit(f"ERROR: only {fg:.1f} GB free under /data; need >= {MIN_FREE_GB} GB")
    print(f"free space: {fg:.1f} GB", flush=True)

    pairs = set()
    for p in (args.gt_swe, args.gt_grepo):
        if not os.path.isfile(p):
            print(f"WARN: missing {p}", flush=True); continue
        for line in open(p):
            d = json.loads(line)
            sha = d.get("base_commit") or d.get("head_commit")
            if not sha: continue
            pairs.add((d["repo"], sha))
    pairs = sorted(pairs)
    print(f"unique (repo, commit) pairs: {len(pairs)}", flush=True)

    if args.filter_repo_commit:
        repo_f, sha_f = args.filter_repo_commit.split(",", 1)
        pairs = [(r, s) for (r, s) in pairs if r == repo_f and s.startswith(sha_f)]
        print(f"filter: {len(pairs)} pairs match", flush=True)
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]
        print(f"limit: {len(pairs)} pairs", flush=True)

    done = set()
    if args.resume and CKPT_PATH.exists():
        for line in open(CKPT_PATH):
            try:
                d = json.loads(line)
                done.add((d["repo"], d["commit"]))
            except Exception:
                continue
        print(f"resume: {len(done)} commits already done", flush=True)
    elif CKPT_PATH.exists():
        CKPT_PATH.unlink()
        print("fresh run: cleared stale checkpoint", flush=True)

    pending = [(r, s) for (r, s) in pairs if (r, s) not in done]
    print(f"pending: {len(pending)} commits", flush=True)

    print("pre-flight: verifying commits reachable locally...", flush=True)
    missing = []
    for repo, sha in pending:
        src = repo_dir_for(repo)
        r = subprocess.run(["git", "-C", src, "cat-file", "-e", f"{sha}^{{commit}}"],
                           capture_output=True)
        if r.returncode != 0:
            missing.append((repo, sha))
    if missing:
        print(f"WARN: {len(missing)} commits missing; attempting fetch per repo...", flush=True)
        for repo in sorted({r for r, _ in missing}):
            src = repo_dir_for(repo)
            subprocess.run(["git", "-C", src, "fetch", "--all", "--tags"],
                           check=False, capture_output=True)
        still = [(r, s) for (r, s) in missing
                 if subprocess.run(["git", "-C", repo_dir_for(r), "cat-file", "-e", f"{s}^{{commit}}"],
                                   capture_output=True).returncode != 0]
        if still:
            for r, s in still[:10]:
                print(f"  unreachable: {r}@{s}", flush=True)
            raise SystemExit(f"aborting: {len(still)} commits unreachable")

    t0 = time.time(); n_done = 0
    ckpt = open(CKPT_PATH, "a")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_commit, r, s, args.mode): (r, s) for (r, s) in pending}
        for f in as_completed(futs):
            r, s = futs[f]
            try:
                stats = f.result()
            except Exception as e:
                print(f"FAIL {r}@{s[:12]}: {e}", flush=True)
                ckpt.close()
                raise SystemExit(f"aborting on commit failure to prevent silent corpus loss")
            ckpt.write(json.dumps(stats) + "\n"); ckpt.flush()
            n_done += 1
            if n_done % 25 == 0:
                dt = time.time() - t0
                rate = n_done / max(1, dt)
                eta = (len(pending) - n_done) / max(1e-6, rate)
                print(f"  {n_done}/{len(pending)} commits | {dt:.0f}s elapsed | ETA {eta:.0f}s", flush=True)
    ckpt.close()

    total_records = 0
    for sf in STATS_DIR.glob("*.stats.json"):
        total_records += json.loads(sf.read_text())["n_records"]
    print(f"DONE: {len(pending)} shards written, {total_records} total records, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
