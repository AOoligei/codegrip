#!/usr/bin/env python3
"""Generate SWE-bench Lite SHA-256 PathSwap variants.

For each repo, build a deterministic per-component SHA-256 alias map.
Apply to test data + BM25 candidates. Output keeps the SAME GT-content
correspondence (only paths renamed; files on disk also need to be
accessible via the new names — but eval scripts typically resolve
through the alias map at code-read time).

Output:
  data/swebench_lite/swebench_lite_test_pathswap.jsonl
  data/swebench_lite/swebench_bm25_pathswap.jsonl
  data/swebench_lite/pathswap_alias_map.json
"""
import json, os, hashlib

BASE = "/home/chenlibin/grepo_agent"
TEST = f"{BASE}/data/swebench_lite/swebench_lite_test.jsonl"
BM25 = f"{BASE}/data/rankft/swebench_bm25_final_top500.jsonl"
REPOS = f"{BASE}/data/swebench_lite/repos"
OUT_TEST = f"{BASE}/data/swebench_lite/swebench_lite_test_pathswap.jsonl"
OUT_BM25 = f"{BASE}/data/swebench_lite/swebench_bm25_pathswap.jsonl"
OUT_MAP = f"{BASE}/data/swebench_lite/pathswap_alias_map.json"


def hash_component(comp):
    if not comp: return comp
    h = hashlib.sha256(comp.encode()).hexdigest()[:16]
    if comp.endswith(".py"): return f"m_{h}.py"
    return f"d_{h}"


def hash_path(p):
    return "/".join(hash_component(c) for c in p.split("/") if c)


def main():
    test_records = [json.loads(l) for l in open(TEST)]
    bm25_records = [json.loads(l) for l in open(BM25)]

    # Build per-repo alias map (orig path → hashed path)
    repo_to_aliases = {}
    repos = set(r.get("repo", "") for r in test_records)
    for repo in repos:
        rd = os.path.join(REPOS, repo)
        if not os.path.isdir(rd): continue
        aliases = {}
        for root, _, files in os.walk(rd):
            for f in files:
                if not f.endswith(".py"): continue
                rel = os.path.relpath(os.path.join(root, f), rd)
                aliases[rel] = hash_path(rel)
        repo_to_aliases[repo] = aliases
        print(f"  {repo}: {len(aliases)} files aliased")

    json.dump(repo_to_aliases, open(OUT_MAP, "w"))
    print(f"Saved alias map: {OUT_MAP}")

    # Apply to test data
    n_test = 0
    with open(OUT_TEST, "w") as f:
        for rec in test_records:
            repo = rec.get("repo", "")
            am = repo_to_aliases.get(repo, {})
            new = dict(rec)
            for k in ("changed_py_files", "changed_files"):
                if k in rec:
                    new[k] = [am.get(p, hash_path(p)) for p in rec[k]]
            f.write(json.dumps(new) + "\n")
            n_test += 1
    print(f"Wrote {OUT_TEST} ({n_test} records)")

    # Apply to BM25 candidates
    n_bm25 = 0
    with open(OUT_BM25, "w") as f:
        for rec in bm25_records:
            repo = rec.get("repo", "")
            am = repo_to_aliases.get(repo, {})
            new = dict(rec)
            for k in ("bm25_candidates", "candidates", "ground_truth"):
                if k in rec:
                    new[k] = [am.get(p, hash_path(p)) for p in rec[k]]
            f.write(json.dumps(new) + "\n")
            n_bm25 += 1
    print(f"Wrote {OUT_BM25} ({n_bm25} records)")


if __name__ == "__main__":
    main()
