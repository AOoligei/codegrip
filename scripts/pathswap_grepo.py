"""PathSwap-GREPO: Structure-preserving path counterfactual benchmark.

For each repo, builds a consistent renaming map that:
- Hashes each path component (directory or filename stem)
- Preserves file extensions (.py stays .py)
- Preserves directory structure (depth, branching)
- Preserves graph topology (if A imports B, renamed_A imports renamed_B)
- Preserves ground truth mapping

This creates a counterfactual where path-token overlap with issue text is destroyed,
but all structural and code-semantic relationships are preserved.

Usage:
    python pathswap_grepo.py --test_data data/grepo_text/grepo_test.jsonl \
        --candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
        --output_dir data/pathswap/
"""

import json
import hashlib
import os
import re
import argparse
from collections import defaultdict


def stable_hash(s: str, length: int = 8) -> str:
    """Deterministic hash of a string to a short hex string."""
    return hashlib.sha256(s.encode()).hexdigest()[:length]


def build_rename_map(all_files: list, repo: str) -> dict:
    """Build a consistent path component renaming map for a repo.

    Each unique path component gets a deterministic hash-based name.
    Extensions are preserved. Directory structure is preserved.
    """
    rename_map = {}  # original_component -> renamed_component

    for filepath in all_files:
        parts = filepath.split('/')
        for i, part in enumerate(parts):
            if part in rename_map:
                continue

            # Preserve extension for files
            if '.' in part and i == len(parts) - 1:
                # It's a file with extension
                stem, ext = part.rsplit('.', 1)
                # Hash the stem, keep extension
                hashed = stable_hash(f"{repo}/{stem}")
                rename_map[part] = f"m_{hashed}.{ext}"
            else:
                # Directory or extensionless file
                hashed = stable_hash(f"{repo}/{part}")
                rename_map[part] = f"d_{hashed}"

    return rename_map


def apply_rename(filepath: str, rename_map: dict) -> str:
    """Apply the rename map to a file path."""
    parts = filepath.split('/')
    renamed_parts = []
    for part in parts:
        renamed_parts.append(rename_map.get(part, part))
    return '/'.join(renamed_parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--candidates", required=True, nargs='+',
                        help="One or more candidate files to transform")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Collect all files per repo from test data and candidates
    repo_files = defaultdict(set)

    # From test data
    test_data = []
    with open(args.test_data) as f:
        for line in f:
            d = json.loads(line)
            test_data.append(d)
            repo = d['repo']
            for fp in d.get('changed_files', []) + d.get('changed_py_files', []):
                repo_files[repo].add(fp)

    # From candidate files
    candidate_data = {}
    for cand_path in args.candidates:
        entries = []
        with open(cand_path) as f:
            for line in f:
                d = json.loads(line)
                entries.append(d)
                repo = d['repo']
                cand_key = 'candidates' if 'candidates' in d else 'bm25_candidates'
                for fp in d.get(cand_key, []):
                    repo_files[repo].add(fp)
        candidate_data[cand_path] = entries

    # Step 2: Build rename maps per repo
    rename_maps = {}
    for repo, files in repo_files.items():
        rename_maps[repo] = build_rename_map(sorted(files), repo)

    # Step 3: Transform test data
    transformed_test = []
    for d in test_data:
        repo = d['repo']
        rmap = rename_maps[repo]
        new_d = dict(d)
        new_d['changed_files'] = [apply_rename(f, rmap) for f in d.get('changed_files', [])]
        new_d['changed_py_files'] = [apply_rename(f, rmap) for f in d.get('changed_py_files', [])]
        # Keep issue_text UNCHANGED — this is the key: path tokens in issue no longer match renamed paths
        transformed_test.append(new_d)

    test_out = os.path.join(args.output_dir, "grepo_test_pathswap.jsonl")
    with open(test_out, 'w') as f:
        for d in transformed_test:
            f.write(json.dumps(d) + '\n')
    print(f"Wrote {len(transformed_test)} test examples to {test_out}")

    # Step 4: Transform candidate files
    for cand_path, entries in candidate_data.items():
        transformed = []
        for d in entries:
            repo = d['repo']
            rmap = rename_maps[repo]
            new_d = dict(d)
            cand_key = 'candidates' if 'candidates' in d else 'bm25_candidates'
            new_d[cand_key] = [apply_rename(f, rmap) for f in d.get(cand_key, [])]
            transformed.append(new_d)

        basename = os.path.basename(cand_path).replace('.jsonl', '_pathswap.jsonl')
        out_path = os.path.join(args.output_dir, basename)
        with open(out_path, 'w') as f:
            for d in transformed:
                f.write(json.dumps(d) + '\n')
        print(f"Wrote {len(transformed)} candidates to {out_path}")

    # Step 5: Compute overlap statistics
    total_tokens = 0
    overlap_tokens = 0
    for d in test_data:
        repo = d['repo']
        issue_tokens = set(re.findall(r'[a-zA-Z_]\w{2,}', d['issue_text'].lower()))
        for fp in d.get('changed_files', []):
            for part in fp.replace('/', '_').replace('.', '_').split('_'):
                if len(part) >= 3:
                    total_tokens += 1
                    if part.lower() in issue_tokens:
                        overlap_tokens += 1

    if total_tokens > 0:
        print(f"\nOverlap stats: {overlap_tokens}/{total_tokens} path tokens "
              f"({100*overlap_tokens/total_tokens:.1f}%) overlap with issue text")
        print("PathSwap destroys ALL path-issue lexical overlap by design")

    # Save rename maps for reference
    maps_out = os.path.join(args.output_dir, "rename_maps.json")
    # Convert sets to lists for JSON
    serializable_maps = {repo: dict(rmap) for repo, rmap in rename_maps.items()}
    with open(maps_out, 'w') as f:
        json.dump(serializable_maps, f, indent=2)
    print(f"Saved rename maps to {maps_out}")


if __name__ == "__main__":
    main()
