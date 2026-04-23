#!/usr/bin/env python3
"""Generate SWE-bench Verified SHA-256 PathSwap variants.

Mirror of build_swebench_pathswap.py with Verified I/O paths.
Repo set is identical to Lite (12 repos), so we reuse data/swebench_lite/repos
to walk the file tree. Outputs go to /data (root disk is full)."""
import json, os, hashlib

LITE_REPOS = "/home/chenlibin/grepo_agent/data/swebench_lite/repos"
VERIFIED_DIR = "/data/chenlibin/grepo_agent_experiments/swebench_verified"
TEST = f"{VERIFIED_DIR}/swebench_verified_prepared.jsonl"
BM25 = f"{VERIFIED_DIR}/swebench_verified_bm25_top500.jsonl"
OUT_TEST = f"{VERIFIED_DIR}/swebench_verified_test_pathswap.jsonl"
OUT_BM25 = f"{VERIFIED_DIR}/swebench_verified_bm25_pathswap.jsonl"
OUT_MAP  = f"{VERIFIED_DIR}/pathswap_alias_map_verified.json"

def hash_component(comp):
    if not comp: return comp
    h = hashlib.sha256(comp.encode()).hexdigest()[:16]
    return f"m_{h}.py" if comp.endswith(".py") else f"d_{h}"

def hash_path(p):
    return "/".join(hash_component(c) for c in p.split("/") if c)

def main():
    test_records = [json.loads(l) for l in open(TEST)]
    bm25_records = [json.loads(l) for l in open(BM25)]

    repos = sorted({r["repo"] for r in test_records})
    repo_to_aliases = {}
    for repo in repos:
        rd = os.path.join(LITE_REPOS, repo)
        assert os.path.isdir(rd), f"missing repo dir: {rd}"
        aliases = {}
        for root, _, files in os.walk(rd):
            for f in files:
                if not f.endswith(".py"): continue
                rel = os.path.relpath(os.path.join(root, f), rd)
                aliases[rel] = hash_path(rel)
        repo_to_aliases[repo] = aliases
        print(f"  {repo}: {len(aliases)} files aliased")

    # GT-coverage check: any GT path missing from repo scan gets a deterministic
    # fallback alias INSERTED into the alias map so downstream reverse lookups
    # (e.g. hashed→orig in eval_codeaware_4bit.py) still resolve.
    inserted_fallback = 0
    for r in test_records:
        am = repo_to_aliases.setdefault(r["repo"], {})
        for p in r.get("changed_py_files", []):
            if p not in am:
                am[p] = hash_path(p)
                inserted_fallback += 1
    if inserted_fallback:
        print(f"INFO: inserted {inserted_fallback} fallback aliases (GT paths absent from on-disk repo snapshot)")

    os.makedirs(VERIFIED_DIR, exist_ok=True)
    json.dump(repo_to_aliases, open(OUT_MAP, "w"))
    print(f"Saved alias map: {OUT_MAP}")

    with open(OUT_TEST, "w") as f:
        for rec in test_records:
            am = repo_to_aliases.get(rec["repo"], {})
            new = dict(rec)
            for k in ("changed_py_files", "changed_files"):
                if k in rec:
                    new[k] = [am.get(p, hash_path(p)) for p in rec[k]]
            f.write(json.dumps(new) + "\n")
    print(f"Wrote {OUT_TEST}")

    with open(OUT_BM25, "w") as f:
        for rec in bm25_records:
            am = repo_to_aliases.get(rec["repo"], {})
            new = dict(rec)
            for k in ("bm25_candidates", "candidates", "ground_truth"):
                if k in rec:
                    new[k] = [am.get(p, hash_path(p)) for p in rec[k]]
            f.write(json.dumps(new) + "\n")
    print(f"Wrote {OUT_BM25}")

if __name__ == "__main__":
    main()
