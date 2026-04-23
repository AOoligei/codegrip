#!/usr/bin/env python3
"""
Step 1a: extract function-level ground truth for the 1870 SWE-bench-derived
training instances (clean_swe_train.jsonl).

For each instance:
  1. Look up base_commit + patch from princeton-nlp/SWE-bench TEST parquet
     (joined by instance_id; verified 1870/1870 hit).
  2. Parse the unified diff to get (path, hunk_line_numbers) on the BASE side.
  3. `git -C <repo> show <base_commit>:<path>` to read pre-patch source.
  4. AST-walk to find the deepest enclosing FunctionDef/AsyncFunctionDef chain
     for each touched line; emit SweRank-style id "path/file.py/[Cls/]func".
     If no enclosing function exists, emit "path/file.py/<MODULE>" with
     gt_kind="module" (filtered by trainer).
Output: /data/chenlibin/codegrip_func/func_gt_swe.jsonl
        {repo, issue_id, base_commit, gt_func_ids: [...], gt_kind: {...},
         coverage: {n_changed_lines, n_resolved}}

Read-only on /home; writes only under /data/chenlibin/codegrip_func/.
Deterministic (no parallelism in v1; ~1870 instances, ~10 min wall).
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

PARQUET = ("/data/chenlibin/hf_cache/hub/hub/datasets--princeton-nlp--SWE-bench/"
           "snapshots/e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6/data/"
           "test-00000-of-00001.parquet")
REPOS_DIR = "/home/chenlibin/grepo_agent/data/swebench_lite/repos"
TRAIN_JSONL = "/home/chenlibin/grepo_agent/data/rankft/clean_swe_train.jsonl"
OUT_DIR = Path("/data/chenlibin/codegrip_func")
OUT_PATH = OUT_DIR / "func_gt_swe.jsonl"

DIFF_FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)")
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")


def load_swebench_index():
    t = pq.read_table(PARQUET, columns=["instance_id", "base_commit", "patch"])
    return {iid: (bc, pt) for iid, bc, pt in zip(
        t["instance_id"].to_pylist(),
        t["base_commit"].to_pylist(),
        t["patch"].to_pylist(),
    )}


def parse_patch_basefiles(patch: str):
    """Yield (path, [line_numbers_on_base_side])."""
    cur_path, cur_base, lines = None, 0, []
    out = {}
    for line in patch.splitlines():
        m = DIFF_FILE_RE.match(line)
        if m:
            if cur_path and lines:
                out.setdefault(cur_path, set()).update(lines)
            cur_path = m.group(1)
            lines = []
            continue
        if cur_path and not cur_path.endswith(".py"):
            continue
        m = HUNK_RE.match(line)
        if m:
            cur_base = int(m.group(1))
            continue
        if line.startswith("-") and not line.startswith("---"):
            lines.append(cur_base); cur_base += 1
        elif line.startswith(" "):
            cur_base += 1
        # "+" lines do not advance base counter
    if cur_path and lines:
        out.setdefault(cur_path, set()).update(lines)
    return {p: sorted(s) for p, s in out.items() if p.endswith(".py")}


def git_show(repo_dir, commit, path):
    try:
        return subprocess.check_output(
            ["git", "-C", repo_dir, "show", f"{commit}:{path}"],
            stderr=subprocess.DEVNULL, text=True, errors="replace",
        )
    except subprocess.CalledProcessError:
        return None


def resolve_funcs(source: str, lines, path: str):
    """Return (ids, kinds, n_func_lines).
    Only returns function-level IDs (deepest=func). Other lines (class-body /
    module-level) fall through to <MODULE>. n_func_lines is the count of input
    lines that mapped to a function (precise per-line coverage).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], "parse_fail", 0
    spans = []  # (start, end, chain_list, kind)

    def walk(node, chain):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = child.lineno
                end = getattr(child, "end_lineno", start)
                new_chain = chain + [child.name]
                kind = "class" if isinstance(child, ast.ClassDef) else "func"
                spans.append((start, end, new_chain, kind))
                walk(child, new_chain)

    walk(tree, [])
    ids = set()
    kinds = set()
    n_func_lines = 0
    for ln in lines:
        best = None
        for start, end, chain, kind in spans:
            if start <= ln <= end:
                if best is None or (end - start) < (best[1] - best[0]):
                    best = (start, end, chain, kind)
        if best is not None and best[3] == "func":
            ids.add(f"{path}/" + "/".join(best[2])); kinds.add("func")
            n_func_lines += 1
        else:
            ids.add(f"{path}/<MODULE>"); kinds.add("module")
    return sorted(ids), "|".join(sorted(kinds)) or "none", n_func_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="dry-run cap")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    # Restrict --out to /data prefix to honor disk policy
    if not str(args.out).startswith("/data/"):
        sys.exit(f"ERROR: --out must be under /data/ (got: {args.out})")

    # mkdir parent of args.out (handles arbitrary --out paths)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    swe_idx = load_swebench_index()
    print(f"Loaded {len(swe_idx)} SWE-bench test instances", file=sys.stderr)
    n_ok = n_skip_no_id = n_skip_no_repo = n_skip_no_funcs = 0
    n_git_show_fail = n_git_show_ok = 0
    n_resolved_lines_total = n_changed_lines_total = 0
    with open(TRAIN_JSONL) as f, open(args.out, "w") as g:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            d = json.loads(line)
            iid, repo = d["issue_id"], d["repo"]
            if iid not in swe_idx:
                n_skip_no_id += 1; continue
            base_commit, patch = swe_idx[iid]
            repo_dir = os.path.join(REPOS_DIR, repo)
            if not os.path.isdir(repo_dir):
                n_skip_no_repo += 1; continue
            file_lines = parse_patch_basefiles(patch or "")
            gt_ids, kinds = [], []
            n_changed = sum(len(v) for v in file_lines.values())
            n_resolved_lines = 0
            for path, lns in file_lines.items():
                src = git_show(repo_dir, base_commit, path)
                if src is None:
                    n_git_show_fail += 1
                    continue
                n_git_show_ok += 1
                ids, kind, n_func_lines = resolve_funcs(src, lns, path)
                gt_ids.extend(ids); kinds.append(kind)
                n_resolved_lines += n_func_lines  # per-line precise count
            gt_ids = sorted(set(gt_ids))
            n_changed_lines_total += n_changed
            n_resolved_lines_total += n_resolved_lines
            if not gt_ids:
                n_skip_no_funcs += 1; continue
            g.write(json.dumps({
                "repo": repo, "issue_id": iid, "base_commit": base_commit,
                "gt_func_ids": gt_ids, "gt_kinds": kinds,
                "coverage": {"n_changed_lines": n_changed,
                              "n_resolved_lines": n_resolved_lines,
                              "n_unique_func_ids": len([x for x in gt_ids if not x.endswith("<MODULE>")])},
                "issue_text": d["issue_text"],
            }) + "\n")
            n_ok += 1
            if (i + 1) % 200 == 0:
                print(f"[{i+1}] ok={n_ok} skip(no_id={n_skip_no_id},no_repo={n_skip_no_repo},no_funcs={n_skip_no_funcs}) "
                      f"git_show_fail={n_git_show_fail}/{n_git_show_fail+n_git_show_ok}", file=sys.stderr, flush=True)
    line_resolution_pct = 100.0 * n_resolved_lines_total / max(1, n_changed_lines_total)
    print(f"DONE ok={n_ok} skip(no_id={n_skip_no_id}, no_repo={n_skip_no_repo}, no_funcs={n_skip_no_funcs})", file=sys.stderr)
    print(f"  git_show: {n_git_show_ok} ok / {n_git_show_fail} fail ({100*n_git_show_fail/max(1,n_git_show_fail+n_git_show_ok):.1f}%)", file=sys.stderr)
    print(f"  changed lines: {n_changed_lines_total}, mapped to func: {n_resolved_lines_total} ({line_resolution_pct:.1f}%)", file=sys.stderr)
    print(f"  Output: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
