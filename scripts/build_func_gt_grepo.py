#!/usr/bin/env python3
"""
Step 1b: extract function-level ground truth for GREPO training instances.

GREPO data already has `changed_functions` (function names, e.g. ['_collapse_state'])
plus `changed_py_files`. We AST-parse each changed file at HEAD (or closest commit)
and find every FunctionDef whose terminal name matches any of `changed_functions`,
emitting SweRank-style id 'path/file.py/[Cls/]func_name'.

If a name matches multiple defs in the file (e.g. overloaded init in different
classes), all matches are emitted; the trainer can dedupe.

Output: /data/chenlibin/codegrip_func/func_gt_grepo.jsonl
        {repo, issue_id, head_commit, gt_func_ids, coverage}

Read-only on /home; writes only under /data/chenlibin/codegrip_func/.
Deterministic. No GPU.
"""
import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

REPOS_DIR = "/home/chenlibin/grepo_agent/data/repos"
TRAIN_JSONL = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_train.jsonl"
OUT_DIR = Path("/data/chenlibin/codegrip_func")
OUT_PATH = OUT_DIR / "func_gt_grepo.jsonl"


def read_file_at_head(repo_dir, path):
    full = os.path.join(repo_dir, path)
    if not os.path.isfile(full):
        return None
    try:
        with open(full, errors="replace") as f:
            return f.read()
    except (PermissionError, IsADirectoryError):
        return None


def get_head_commit(repo_dir):
    try:
        return subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def find_matching_funcs(source: str, target_names: set, path: str):
    """AST-walk and return list of SweRank-style ids matching target_names."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    matches = []

    def walk(node, chain):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name in target_names:
                    matches.append(f"{path}/" + "/".join(chain + [child.name]))
                walk(child, chain + [child.name])
            elif isinstance(child, ast.ClassDef):
                walk(child, chain + [child.name])
            else:
                walk(child, chain)

    walk(tree, [])
    return matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    if not str(args.out).startswith("/data/"):
        sys.exit(f"ERROR: --out must be under /data/ (got: {args.out})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip_no_repo = n_skip_no_funcs = 0
    n_file_read_fail = 0
    n_total_targets = n_resolved_targets = 0

    with open(TRAIN_JSONL) as f, open(args.out, "w") as g:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            d = json.loads(line)
            repo = d["repo"]
            iid = d.get("issue_id")
            issue_text = d.get("issue_text", "")
            target_names = set(d.get("changed_functions") or [])
            files = [p for p in (d.get("changed_py_files") or []) if p.endswith(".py")]

            if not target_names or not files:
                n_skip_no_funcs += 1; continue

            repo_dir = os.path.join(REPOS_DIR, repo)
            if not os.path.isdir(repo_dir):
                n_skip_no_repo += 1; continue
            head = get_head_commit(repo_dir)

            gt_ids = []
            n_total_targets += len(target_names)
            for path in files:
                src = read_file_at_head(repo_dir, path)
                if src is None:
                    n_file_read_fail += 1; continue
                ids = find_matching_funcs(src, target_names, path)
                gt_ids.extend(ids)
            gt_ids = sorted(set(gt_ids))
            n_resolved_targets += len({i.rsplit("/", 1)[-1] for i in gt_ids} & target_names)

            if not gt_ids:
                n_skip_no_funcs += 1; continue
            g.write(json.dumps({
                "repo": repo, "issue_id": iid, "head_commit": head,
                "gt_func_ids": gt_ids,
                "coverage": {"n_target_names": len(target_names),
                              "n_resolved_unique": len(gt_ids),
                              "n_files": len(files)},
                "issue_text": issue_text,
            }) + "\n")
            n_ok += 1
            if (i + 1) % 500 == 0:
                print(f"[{i+1}] ok={n_ok} skip(no_repo={n_skip_no_repo}, no_funcs={n_skip_no_funcs}) file_read_fail={n_file_read_fail}",
                      file=sys.stderr, flush=True)

    name_resolution_pct = 100.0 * n_resolved_targets / max(1, n_total_targets)
    print(f"DONE ok={n_ok} skip(no_repo={n_skip_no_repo}, no_funcs={n_skip_no_funcs})", file=sys.stderr)
    print(f"  file_read_fail: {n_file_read_fail}", file=sys.stderr)
    print(f"  function-name resolution: {n_resolved_targets}/{n_total_targets} ({name_resolution_pct:.1f}%)", file=sys.stderr)
    print(f"  Output: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
