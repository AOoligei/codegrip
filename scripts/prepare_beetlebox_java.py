"""
Prepare BeetleBox Java test data for CodeGRIP reranker evaluation.

This script:
1. Loads BeetleBox test.parquet and filters to Java examples
2. Clones each repo at before_fix_sha (cached)
3. Lists all .java files in each repo snapshot
4. Runs BM25 (rank_bm25) over file paths using issue title+body as query
5. Outputs two JSONL files compatible with eval_rankft.py:
   - java_test.jsonl  (test data, same schema as grepo_test.jsonl)
   - java_bm25_top500.jsonl  (BM25 candidates, same schema as grepo_test_bm25_top500.jsonl)

Usage:
    python scripts/prepare_beetlebox_java.py

Cross-language generalization test: model trained on Python, evaluated on Java.
"""

import json
import os
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

# Deterministic
np.random.seed(42)

# ============================================================
# Config
# ============================================================

BEETLEBOX_PARQUET = "/data/chenlibin/beetlebox/test.parquet"
CLONE_CACHE_DIR = "/data/chenlibin/beetlebox/repo_cache"
OUTPUT_DIR = "/data/chenlibin/beetlebox"
TOP_K = 500

OUTPUT_TEST = os.path.join(OUTPUT_DIR, "java_test.jsonl")
OUTPUT_BM25 = os.path.join(OUTPUT_DIR, "java_bm25_top500.jsonl")


# ============================================================
# Repo cloning & file listing
# ============================================================

def clone_repo(repo_name: str, repo_url: str) -> str:
    """Clone repo to cache dir (if not already cached). Returns local path."""
    # repo_name like "apache/dubbo" -> use as directory
    local_path = os.path.join(CLONE_CACHE_DIR, repo_name.replace("/", "__"))
    if os.path.exists(local_path):
        print(f"  [cache hit] {repo_name} -> {local_path}")
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"  [cloning] {repo_url} -> {local_path}")
    try:
        subprocess.run(
            ["git", "clone", "--bare", "--filter=tree:0", repo_url + ".git", local_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        print(f"  [WARN] Clone failed for {repo_name}: {e}")
        # Try shallow clone as fallback
        try:
            subprocess.run(
                ["git", "clone", "--bare", "--depth=1", repo_url + ".git", local_path + "_shallow"],
                check=True, capture_output=True, text=True, timeout=600,
            )
            os.rename(local_path + "_shallow", local_path)
            print(f"  [fallback] Shallow clone succeeded for {repo_name}")
        except Exception as e2:
            print(f"  [ERROR] All clone attempts failed for {repo_name}: {e2}")
            return None
    return local_path


def list_java_files_at_sha(bare_repo_path: str, sha: str) -> list[str]:
    """List all .java files in a bare repo at a given commit SHA.

    Uses `git ls-tree` which is fast and doesn't require checkout.
    Returns relative paths like 'src/main/java/com/example/Foo.java'.
    """
    result = subprocess.run(
        ["git", "--git-dir", bare_repo_path, "ls-tree", "-r", "--name-only", sha],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print(f"    WARNING: git ls-tree failed for {sha}: {result.stderr.strip()}")
        return []

    all_files = result.stdout.strip().split("\n")
    java_files = [f for f in all_files if f.endswith(".java")]
    return java_files


# ============================================================
# BM25 retrieval over file paths
# ============================================================

def tokenize_path(path: str) -> list[str]:
    """Tokenize a file path into searchable tokens.

    'src/main/java/org/apache/dubbo/rpc/TriRpcStatus.java'
    -> ['src', 'main', 'java', 'org', 'apache', 'dubbo', 'rpc', 'tri', 'rpc', 'status']
    """
    # Split on / and .
    parts = path.replace("/", " ").replace(".", " ").replace("_", " ").replace("-", " ")
    # CamelCase split
    tokens = []
    for part in parts.split():
        # Split CamelCase: 'TriRpcStatus' -> ['Tri', 'Rpc', 'Status']
        camel_tokens = []
        current = ""
        for c in part:
            if c.isupper() and current:
                camel_tokens.append(current)
                current = c
            else:
                current += c
        if current:
            camel_tokens.append(current)
        tokens.extend(camel_tokens)

    return [t.lower() for t in tokens if len(t) > 1]


def tokenize_query(text: str) -> list[str]:
    """Tokenize issue text for BM25 query."""
    # Simple tokenization: lowercase, split on non-alphanumeric
    import re
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1]


def bm25_rank_files(
    query_text: str,
    file_paths: list[str],
    top_k: int = TOP_K,
) -> list[str]:
    """Rank file paths by BM25 relevance to query text.

    Each file path is treated as a "document" with tokens from the path.
    """
    if not file_paths:
        return []

    # Tokenize all file paths
    corpus = [tokenize_path(p) for p in file_paths]
    bm25 = BM25Okapi(corpus)

    # Tokenize query
    query_tokens = tokenize_query(query_text)
    if not query_tokens:
        return file_paths[:top_k]

    scores = bm25.get_scores(query_tokens)
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [file_paths[i] for i in ranked_indices]


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Preparing BeetleBox Java data for CodeGRIP evaluation")
    print("=" * 70)

    # Load data
    print(f"\nLoading {BEETLEBOX_PARQUET}...")
    df = pd.read_parquet(BEETLEBOX_PARQUET)
    java_df = df[df["language"] == "java"].copy()
    print(f"  Total test examples: {len(df)}")
    print(f"  Java test examples: {len(java_df)}")

    # Get unique repos
    repos = java_df[["repo_name", "repo_url"]].drop_duplicates()
    print(f"\n  Unique Java repos: {len(repos)}")
    for _, row in repos.iterrows():
        count = (java_df["repo_name"] == row["repo_name"]).sum()
        print(f"    {row['repo_name']}: {count} examples")

    # Clone all repos (bare clones for efficiency)
    print(f"\nCloning repos to {CLONE_CACHE_DIR}...")
    os.makedirs(CLONE_CACHE_DIR, exist_ok=True)
    repo_paths = {}
    for _, row in repos.iterrows():
        repo_name = row["repo_name"]
        repo_url = row["repo_url"]
        path = clone_repo(repo_name, repo_url)
        if path is not None:
            repo_paths[repo_name] = path
        else:
            print(f"  SKIPPING {repo_name} (clone failed)")

    # Process each example
    print(f"\nProcessing {len(java_df)} Java examples...")
    test_records = []
    bm25_records = []
    stats = defaultdict(int)

    # Cache: (repo_name, sha) -> java_files
    file_cache: dict[tuple[str, str], list[str]] = {}

    for idx, (_, row) in enumerate(java_df.iterrows()):
        repo_name = row["repo_name"]
        issue_id = int(row["issue_id"])
        sha = row["before_fix_sha"]
        title = str(row["title"] or "")
        body = str(row["body"] or "")
        updated_files_raw = row["updated_files"]

        # Parse updated_files (stored as JSON string)
        if isinstance(updated_files_raw, str):
            updated_files = json.loads(updated_files_raw)
        elif isinstance(updated_files_raw, list):
            updated_files = updated_files_raw
        else:
            updated_files = []

        # Filter to .java files only
        changed_java_files = [f for f in updated_files if f.endswith(".java")]
        if not changed_java_files:
            stats["skipped_no_java_gt"] += 1
            continue

        # Build issue text (matching GREPO format)
        issue_text = f"Title: {title}"
        if body:
            issue_text += f"\n\nDescription: {body}"

        # Short repo name (without org prefix) for consistency with GREPO format
        short_repo = repo_name.split("/")[-1]

        # Skip if repo wasn't cloned
        if repo_name not in repo_paths:
            stats["skipped_no_repo"] += 1
            continue

        # List all Java files at before_fix_sha
        cache_key = (repo_name, sha)
        if cache_key not in file_cache:
            bare_path = repo_paths[repo_name]
            file_cache[cache_key] = list_java_files_at_sha(bare_path, sha)
        java_files = file_cache[cache_key]

        if not java_files:
            stats["skipped_no_files"] += 1
            continue

        # BM25 ranking
        bm25_ranked = bm25_rank_files(issue_text, java_files, top_k=TOP_K)

        # Check if GT files are in candidates
        gt_set = set(changed_java_files)
        gt_in_candidates = bool(gt_set & set(bm25_ranked))

        # Test data record (matching grepo_test.jsonl schema)
        # NOTE: We use "changed_py_files" key to match eval_rankft.py's filter
        # (line 181: `if item.get("changed_py_files")`), even though these are Java files.
        test_record = {
            "repo": short_repo,
            "issue_id": issue_id,
            "issue_text": issue_text,
            "changed_files": changed_java_files,
            "changed_py_files": changed_java_files,  # Compatibility with eval_rankft.py
            "changed_functions": [],
            "split": "test",
            "timestamp": str(row.get("report_datetime", "")),
            # Extra metadata for cross-language analysis
            "language": "java",
            "full_repo_name": repo_name,
            "before_fix_sha": sha,
        }
        test_records.append(test_record)

        # BM25 candidates record
        bm25_record = {
            "repo": short_repo,
            "issue_id": issue_id,
            "issue_text": issue_text,
            "ground_truth": changed_java_files,
            "bm25_candidates": bm25_ranked,
            "gt_in_candidates": gt_in_candidates,
            "num_java_files_in_repo": len(java_files),
        }
        bm25_records.append(bm25_record)

        stats["processed"] += 1
        if gt_in_candidates:
            stats["gt_in_bm25"] += 1

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(java_df)}] processed={stats['processed']} "
                  f"gt_in_bm25={stats['gt_in_bm25']}")

    # Save outputs
    print(f"\nSaving outputs...")

    with open(OUTPUT_TEST, "w") as f:
        for r in test_records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"  Test data: {OUTPUT_TEST} ({len(test_records)} examples)")

    with open(OUTPUT_BM25, "w") as f:
        for r in bm25_records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"  BM25 candidates: {OUTPUT_BM25} ({len(bm25_records)} examples)")

    # Summary stats
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Total Java examples: {len(java_df)}")
    print(f"  Processed (have Java GT files): {stats['processed']}")
    print(f"  Skipped (no Java GT files): {stats.get('skipped_no_java_gt', 0)}")
    print(f"  Skipped (no files at SHA): {stats.get('skipped_no_files', 0)}")
    print(f"  GT in BM25 top-{TOP_K}: {stats['gt_in_bm25']}/{stats['processed']} "
          f"({100*stats['gt_in_bm25']/max(stats['processed'],1):.1f}%)")

    # Print eval command
    print(f"\n{'='*70}")
    print(f"EVAL COMMAND (run when GPUs are free):")
    print(f"{'='*70}")
    print(f"""
/home/chenlibin/miniconda3/envs/tgn/bin/python src/eval/eval_rankft.py \\
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \\
    --lora_path experiments/rankft_runB_graph/best \\
    --test_data {OUTPUT_TEST} \\
    --bm25_candidates {OUTPUT_BM25} \\
    --output_dir experiments/beetlebox_java_eval \\
    --gpu_id 0 \\
    --top_k 200 \\
    --max_seq_length 512
""")


if __name__ == "__main__":
    main()
