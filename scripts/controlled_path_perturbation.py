#!/usr/bin/env python3
"""
Controlled path perturbation experiment for CodeGRIP.

Disentangles what path components the reranker relies on:
  1. shuffle_dirs      - keep filenames, randomly reassign directory paths
  2. shuffle_filenames - keep dirs, randomly shuffle filenames within each dir
  3. remove_module_names - hash module-semantic filename parts, keep structure
  4. flatten_dirs      - remove all dirs, keep original filenames in root
  5. swap_leaf_dirs    - shuffle only the last directory component

Each condition creates perturbed test.jsonl + bm25_candidates.jsonl under
experiments/path_perturb_{condition}/.

Usage:
    python scripts/controlled_path_perturbation.py [--conditions ...]
    python scripts/controlled_path_perturbation.py --conditions shuffle_dirs flatten_dirs
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_PATH = BASE_DIR / "data" / "grepo_text" / "grepo_test.jsonl"
BM25_PATH = BASE_DIR / "data" / "rankft" / "merged_bm25_exp6_candidates.jsonl"

ALL_CONDITIONS = [
    "shuffle_dirs",
    "shuffle_filenames",
    "remove_module_names",
    "flatten_dirs",
    "swap_leaf_dirs",
]

# Module-semantic keywords commonly found in Python filenames.
# We hash these while preserving prefixes like test_ and suffixes like .py.
MODULE_SEMANTIC_WORDS = {
    "models", "model", "views", "view", "forms", "form",
    "serializers", "serializer", "admin", "urls", "managers", "manager",
    "signals", "signal", "middleware", "middlewares",
    "utils", "util", "helpers", "helper", "mixins", "mixin",
    "decorators", "decorator", "validators", "validator",
    "exceptions", "exception", "constants", "constant",
    "handlers", "handler", "services", "service",
    "tasks", "task", "commands", "command",
    "factories", "factory", "fixtures", "fixture",
    "backends", "backend", "adapters", "adapter",
    "parsers", "parser", "renderers", "renderer",
    "filters", "filter", "permissions", "permission",
    "routers", "router", "schemas", "schema",
    "config", "settings", "conf",
    "api", "core", "base", "common", "main", "app",
    "cli", "server", "client", "protocol",
    "types", "interfaces", "abstract",
    "ops", "gates", "circuits", "protocols",
}


def _stable_hash(s: str, length: int = 8) -> str:
    """Deterministic short hash of a string."""
    return hashlib.sha256(s.encode()).hexdigest()[:length]


# ============================================================
# Perturbation functions
# ============================================================

def perturb_shuffle_dirs(paths: List[str]) -> Dict[str, str]:
    """Keep filenames, randomly shuffle directory assignments.

    Collect all unique directory prefixes and all unique filenames,
    then randomly reassign each file to a directory (1-to-1 mapping of
    original full paths to new full paths, preserving filename).
    """
    mapping = {}
    if not paths:
        return mapping

    # Separate into (dir_prefix, filename) pairs
    dir_file_pairs = []
    for p in paths:
        parts = p.rsplit("/", 1)
        if len(parts) == 2:
            dir_file_pairs.append((parts[0], parts[1]))
        else:
            dir_file_pairs.append(("", parts[0]))

    # Collect unique directory prefixes
    unique_dirs = list(set(d for d, _ in dir_file_pairs if d))
    if not unique_dirs:
        # No directories to shuffle -- identity mapping
        return {p: p for p in paths}

    # For each path, assign a random directory from the pool
    shuffled_dirs = list(unique_dirs)
    rng = random.Random(42)
    # We want a permutation that's applied to directory assignments
    # Build a mapping from old_dir -> new_dir
    dir_perm = list(unique_dirs)
    rng.shuffle(dir_perm)
    dir_map = dict(zip(unique_dirs, dir_perm))

    for p in paths:
        parts = p.rsplit("/", 1)
        if len(parts) == 2:
            old_dir, fname = parts
            new_dir = dir_map.get(old_dir, old_dir)
            mapping[p] = f"{new_dir}/{fname}"
        else:
            mapping[p] = p

    return mapping


def perturb_shuffle_filenames(paths: List[str]) -> Dict[str, str]:
    """Keep directory structure, randomly shuffle filenames within each directory."""
    mapping = {}
    if not paths:
        return mapping

    # Group files by directory
    dir_to_files = defaultdict(list)
    for p in paths:
        parts = p.rsplit("/", 1)
        if len(parts) == 2:
            dir_to_files[parts[0]].append(parts[1])
        else:
            dir_to_files[""].append(parts[0])

    rng = random.Random(42)
    dir_to_shuffled = {}
    for d, fnames in dir_to_files.items():
        shuffled = list(fnames)
        rng.shuffle(shuffled)
        dir_to_shuffled[d] = dict(zip(fnames, shuffled))

    for p in paths:
        parts = p.rsplit("/", 1)
        if len(parts) == 2:
            d, f = parts
            new_f = dir_to_shuffled[d][f]
            mapping[p] = f"{d}/{new_f}"
        else:
            new_f = dir_to_shuffled[""][parts[0]]
            mapping[p] = new_f

    return mapping


def perturb_remove_module_names(paths: List[str]) -> Dict[str, str]:
    """Replace module-semantic parts of filenames with hashes.

    Preserves: test_ prefix, _test suffix, .py extension, directory structure.
    Hashes: the 'semantic core' of the filename if it matches MODULE_SEMANTIC_WORDS.
    """
    mapping = {}
    fname_cache = {}  # cache per unique filename to ensure consistency

    for p in paths:
        parts = p.rsplit("/", 1)
        if len(parts) == 2:
            d, f = parts
        else:
            d, f = "", parts[0]

        if f in fname_cache:
            new_f = fname_cache[f]
        else:
            new_f = _anonymize_module_filename(f)
            fname_cache[f] = new_f

        if d:
            mapping[p] = f"{d}/{new_f}"
        else:
            mapping[p] = new_f

    return mapping


def _anonymize_module_filename(fname: str) -> str:
    """Anonymize module-semantic parts of a single filename.

    Examples:
        models.py       -> m_a1b2c3d4.py
        test_models.py  -> test_a1b2c3d4.py
        models_test.py  -> a1b2c3d4_test.py
        my_custom.py    -> my_custom.py  (no module keyword, unchanged)
    """
    stem = Path(fname).stem
    ext = Path(fname).suffix or ".py"

    # Detect test prefix/suffix
    prefix = ""
    suffix = ""
    core = stem

    if core.startswith("test_"):
        prefix = "test_"
        core = core[5:]
    elif core.startswith("tests_"):
        prefix = "tests_"
        core = core[6:]

    if core.endswith("_test"):
        suffix = "_test"
        core = core[:-5]
    elif core.endswith("_tests"):
        suffix = "_tests"
        core = core[:-6]

    # Check if the core (or any part split by _) contains module-semantic words
    core_parts = core.split("_")
    has_semantic = any(part.lower() in MODULE_SEMANTIC_WORDS for part in core_parts)

    if has_semantic:
        # Hash the semantic parts, keep non-semantic parts
        new_parts = []
        for part in core_parts:
            if part.lower() in MODULE_SEMANTIC_WORDS:
                new_parts.append(_stable_hash(part, 6))
            else:
                new_parts.append(part)
        new_core = "_".join(new_parts)
        return f"{prefix}{new_core}{suffix}{ext}"
    else:
        # No module keyword found, return unchanged
        return fname


def perturb_flatten_dirs(paths: List[str]) -> Dict[str, str]:
    """Remove all directory structure, keep original filenames in root.

    If filenames collide (same filename in different dirs), disambiguate
    with a numeric suffix.
    """
    mapping = {}
    fname_count = defaultdict(int)

    # First pass: count filename occurrences
    fnames_all = []
    for p in paths:
        fname = p.rsplit("/", 1)[-1]
        fnames_all.append(fname)
        fname_count[fname] += 1

    # Second pass: assign unique names
    fname_seen = defaultdict(int)
    for p in paths:
        fname = p.rsplit("/", 1)[-1]
        if fname_count[fname] > 1:
            idx = fname_seen[fname]
            fname_seen[fname] += 1
            stem = Path(fname).stem
            ext = Path(fname).suffix or ".py"
            mapping[p] = f"{stem}_{idx}{ext}"
        else:
            mapping[p] = fname

    return mapping


def perturb_swap_leaf_dirs(paths: List[str]) -> Dict[str, str]:
    """Shuffle only the last directory component (immediate parent dir).

    Keep deeper directory structure and filenames intact.
    E.g., a/b/c/file.py -> parent is 'c', deeper is 'a/b'.
    We shuffle the leaf dirs among paths that share the same depth.
    """
    mapping = {}
    if not paths:
        return mapping

    # Parse each path into (deeper_prefix, leaf_dir, filename)
    parsed = []
    for p in paths:
        parts = p.split("/")
        if len(parts) >= 3:
            deeper = "/".join(parts[:-2])
            leaf = parts[-2]
            fname = parts[-1]
            parsed.append((deeper, leaf, fname, p))
        elif len(parts) == 2:
            parsed.append(("", parts[0], parts[1], p))
        else:
            parsed.append(("", "", parts[0], p))

    # Collect unique leaf dirs (only from paths that have a leaf dir)
    leaf_dirs = list(set(leaf for _, leaf, _, _ in parsed if leaf))
    if not leaf_dirs:
        return {p: p for p in paths}

    rng = random.Random(42)
    shuffled_leafs = list(leaf_dirs)
    rng.shuffle(shuffled_leafs)
    leaf_map = dict(zip(leaf_dirs, shuffled_leafs))

    for deeper, leaf, fname, orig in parsed:
        if leaf:
            new_leaf = leaf_map.get(leaf, leaf)
            if deeper:
                mapping[orig] = f"{deeper}/{new_leaf}/{fname}"
            else:
                mapping[orig] = f"{new_leaf}/{fname}"
        else:
            mapping[orig] = orig

    return mapping


# ============================================================
# Perturbation dispatcher
# ============================================================

PERTURB_FN = {
    "shuffle_dirs": perturb_shuffle_dirs,
    "shuffle_filenames": perturb_shuffle_filenames,
    "remove_module_names": perturb_remove_module_names,
    "flatten_dirs": perturb_flatten_dirs,
    "swap_leaf_dirs": perturb_swap_leaf_dirs,
}


def apply_perturbation(condition: str, paths: List[str]) -> Dict[str, str]:
    """Apply a perturbation condition and return old->new path mapping."""
    fn = PERTURB_FN[condition]
    return fn(paths)


# ============================================================
# Data processing
# ============================================================

def load_data() -> Tuple[List[dict], Dict[Tuple[str, int], dict]]:
    """Load test data and BM25 candidates."""
    test_data = []
    with open(TEST_PATH) as f:
        for line in f:
            test_data.append(json.loads(line))

    bm25_data = {}
    with open(BM25_PATH) as f:
        for line in f:
            item = json.loads(line)
            key = (item["repo"], item["issue_id"])
            bm25_data[key] = item

    return test_data, bm25_data


def create_perturbed_data(condition: str, test_data: List[dict],
                          bm25_data: dict, out_dir: str):
    """Create perturbed test and BM25 candidate files for one condition."""
    os.makedirs(out_dir, exist_ok=True)
    out_test = os.path.join(out_dir, "test.jsonl")
    out_bm25 = os.path.join(out_dir, "bm25_candidates.jsonl")

    perturbed_test = []
    perturbed_bm25 = []
    n_skipped = 0
    n_changed_paths = 0
    n_total_paths = 0

    for item in test_data:
        repo = item["repo"]
        issue_id = item["issue_id"]
        key = (repo, issue_id)

        if key not in bm25_data:
            n_skipped += 1
            continue

        bm25_item = bm25_data[key]
        candidates = bm25_item.get("candidates", [])
        gt_files = item.get("changed_py_files", [])

        # Build per-example path mapping (all unique paths in this example)
        all_paths = list(set(candidates + gt_files))
        path_map = apply_perturbation(condition, all_paths)

        # Count changes
        for p in all_paths:
            n_total_paths += 1
            if path_map.get(p, p) != p:
                n_changed_paths += 1

        # Create perturbed test item
        p_item = dict(item)
        p_item["changed_py_files"] = [path_map.get(f, f) for f in gt_files]
        if "changed_files" in p_item:
            p_item["changed_files"] = [path_map.get(f, f) for f in item["changed_files"]]
        perturbed_test.append(p_item)

        # Create perturbed BM25 item
        p_bm25 = dict(bm25_item)
        p_bm25["candidates"] = [path_map.get(c, c) for c in candidates]
        perturbed_bm25.append(p_bm25)

    with open(out_test, "w") as f:
        for item in perturbed_test:
            f.write(json.dumps(item) + "\n")

    with open(out_bm25, "w") as f:
        for item in perturbed_bm25:
            f.write(json.dumps(item) + "\n")

    pct = (n_changed_paths / n_total_paths * 100) if n_total_paths else 0
    print(f"  [{condition}] {len(perturbed_test)} examples, "
          f"{n_skipped} skipped, "
          f"{n_changed_paths}/{n_total_paths} paths changed ({pct:.1f}%)")
    print(f"  -> {out_test}")
    print(f"  -> {out_bm25}")

    return len(perturbed_test)


def show_examples(condition: str, test_data: List[dict],
                  bm25_data: dict, n: int = 3):
    """Show a few example perturbations for sanity checking."""
    print(f"\n  Sample perturbations for '{condition}':")
    count = 0
    for item in test_data:
        key = (item["repo"], item["issue_id"])
        if key not in bm25_data:
            continue
        bm25_item = bm25_data[key]
        candidates = bm25_item.get("candidates", [])
        gt_files = item.get("changed_py_files", [])
        all_paths = list(set(candidates + gt_files))
        path_map = apply_perturbation(condition, all_paths)

        # Show first few changed paths
        changed = [(old, new) for old, new in path_map.items() if old != new]
        if changed:
            print(f"    repo={item['repo']} issue={item['issue_id']}:")
            for old, new in changed[:5]:
                print(f"      {old}")
                print(f"      -> {new}")
            count += 1
            if count >= n:
                break
    if count == 0:
        print("    (no paths changed in first examples)")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Controlled path perturbation for CodeGRIP ablation."
    )
    parser.add_argument(
        "--conditions", nargs="+", default=ALL_CONDITIONS,
        choices=ALL_CONDITIONS,
        help="Which perturbation conditions to create."
    )
    parser.add_argument(
        "--show_examples", action="store_true", default=True,
        help="Show sample perturbations for sanity check."
    )
    parser.add_argument(
        "--no_examples", action="store_true",
        help="Suppress sample output."
    )
    args = parser.parse_args()

    print(f"Loading data from:")
    print(f"  test:  {TEST_PATH}")
    print(f"  bm25:  {BM25_PATH}")
    test_data, bm25_data = load_data()
    print(f"  Loaded {len(test_data)} test examples, {len(bm25_data)} BM25 entries")

    for cond in args.conditions:
        out_dir = str(BASE_DIR / "experiments" / f"path_perturb_{cond}")
        print(f"\n=== Condition: {cond} ===")
        create_perturbed_data(cond, test_data, bm25_data, out_dir)
        if args.show_examples and not args.no_examples:
            show_examples(cond, test_data, bm25_data)

    print("\n=== All conditions created ===")
    print("Output directories:")
    for cond in args.conditions:
        out_dir = BASE_DIR / "experiments" / f"path_perturb_{cond}"
        print(f"  {out_dir}")


if __name__ == "__main__":
    main()


def perturb_delexicalize(paths: List[str], issue_text: str = "") -> Dict[str, str]:
    """Remove path tokens that literally appear in the issue text.
    
    This isolates lexical shortcuts: if the model uses literal token overlap
    between issue text and file paths, this perturbation will hurt.
    If the model uses structural co-location patterns, this should survive.
    """
    # Extract tokens from issue text (lowercased, alphanumeric, len >= 3)
    issue_tokens = set()
    for word in re.findall(r'[a-zA-Z_]\w{2,}', issue_text.lower()):
        issue_tokens.add(word)
    # Also extract from path-like mentions
    for p in re.findall(r'[\w./]+\.py', issue_text):
        for part in p.replace('.py', '').split('/'):
            for sub in part.split('_'):
                if len(sub) >= 3:
                    issue_tokens.add(sub.lower())
    
    mapping = {}
    for path in paths:
        parts = path.split('/')
        new_parts = []
        for part in parts:
            if part.endswith('.py'):
                base = part[:-3]
                # Split on underscore, hash any token that appears in issue
                sub_parts = base.split('_')
                new_sub = []
                for s in sub_parts:
                    if s.lower() in issue_tokens and len(s) >= 3:
                        new_sub.append(_stable_hash(s, 6))
                    else:
                        new_sub.append(s)
                new_parts.append('_'.join(new_sub) + '.py')
            else:
                # For directory components
                sub_parts = part.split('_')
                new_sub = []
                for s in sub_parts:
                    if s.lower() in issue_tokens and len(s) >= 3:
                        new_sub.append(_stable_hash(s, 6))
                    else:
                        new_sub.append(s)
                new_parts.append('_'.join(new_sub))
        mapping[path] = '/'.join(new_parts)
    return mapping
