"""Build the PathSwap-GREPO benchmark package.

Merges original test data with PathSwap-transformed data into a single
self-contained benchmark file. Each example contains both original and
shuffled paths, plus the candidate pool with both path versions.

Usage:
    python scripts/build_pathswap_benchmark.py
"""

import json
import os

BASE = "/home/chenlibin/grepo_agent"
OUT_DIR = os.path.join(BASE, "data/pathswap_benchmark")
os.makedirs(OUT_DIR, exist_ok=True)

# Load original test data (keyed by repo+issue_id)
orig_test = {}
with open(os.path.join(BASE, "data/grepo_text/grepo_test.jsonl")) as f:
    for line in f:
        d = json.loads(line)
        key = (d["repo"], d["issue_id"])
        orig_test[key] = d

# Load pathswap test data
ps_test = {}
with open(os.path.join(BASE, "data/pathswap/grepo_test_pathswap.jsonl")) as f:
    for line in f:
        d = json.loads(line)
        key = (d["repo"], d["issue_id"])
        ps_test[key] = d

# Load original BM25 candidates
orig_cands = {}
with open(os.path.join(BASE, "data/rankft/grepo_test_bm25_top500.jsonl")) as f:
    for line in f:
        d = json.loads(line)
        key = (d["repo"], d["issue_id"])
        cand_key = "candidates" if "candidates" in d else "bm25_candidates"
        orig_cands[key] = d.get(cand_key, [])

# Load pathswap BM25 candidates
ps_cands = {}
with open(os.path.join(BASE, "data/pathswap/grepo_test_bm25_top500_pathswap.jsonl")) as f:
    for line in f:
        d = json.loads(line)
        key = (d["repo"], d["issue_id"])
        cand_key = "candidates" if "candidates" in d else "bm25_candidates"
        ps_cands[key] = d.get(cand_key, [])

# Build merged test file: only include examples that have candidates
test_keys = sorted(orig_cands.keys(), key=lambda k: (k[0], k[1]))
print(f"Building benchmark with {len(test_keys)} examples")

test_out = []
for key in test_keys:
    orig = orig_test[key]
    ps = ps_test[key]
    entry = {
        "repo": orig["repo"],
        "issue_id": orig["issue_id"],
        "issue_text": orig["issue_text"],
        "ground_truth_files": orig["changed_files"],
        "ground_truth_files_pathswap": ps["changed_files"],
    }
    test_out.append(entry)

with open(os.path.join(OUT_DIR, "pathswap_test.jsonl"), "w") as f:
    for entry in test_out:
        f.write(json.dumps(entry) + "\n")
print(f"Wrote {len(test_out)} test examples to pathswap_test.jsonl")

# Build merged candidate file
cands_out = []
for key in test_keys:
    entry = {
        "repo": key[0],
        "issue_id": key[1],
        "candidates": orig_cands[key],
        "candidates_pathswap": ps_cands[key],
    }
    cands_out.append(entry)

with open(os.path.join(OUT_DIR, "pathswap_candidates.jsonl"), "w") as f:
    for entry in cands_out:
        f.write(json.dumps(entry) + "\n")
print(f"Wrote {len(cands_out)} candidate entries to pathswap_candidates.jsonl")

# Stats
n_cands = [len(e["candidates"]) for e in cands_out]
print(f"\nCandidate pool stats: mean={sum(n_cands)/len(n_cands):.1f}, "
      f"min={min(n_cands)}, max={max(n_cands)}")
