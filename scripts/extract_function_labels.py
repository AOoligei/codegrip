#!/usr/bin/env python3
"""
Extract function-level labels from git diffs (or AST matching when diffs unavailable).

For each training example, determines which functions were modified in the fix.

Two modes of operation:
  1. DIFF MODE (base_commit present): Parse pre-fix file via `git show base_commit:file`,
     extract function spans with AST, get diff line numbers, map to enclosing functions.
  2. MATCH MODE (no base_commit / shallow clone fallback): Parse HEAD file with AST,
     match the existing `changed_functions` bare names to qualified names per file.

Output: augmented JSONL with `changed_functions_detailed` per file, containing
positive_functions (qualified), all_functions, and module_level_edit flag.

Usage:
    python scripts/extract_function_labels.py \
        --repos_dir data/repos \
        --train_data data/grepo_text/grepo_train.jsonl \
        --output data/grepo_text/grepo_train_function_labels.jsonl
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

class FunctionSpan:
    """A function/method definition with its line range and qualified name."""
    __slots__ = ("name", "qualified_name", "start", "end")

    def __init__(self, name: str, qualified_name: str, start: int, end: int):
        self.name = name
        self.qualified_name = qualified_name
        self.start = start
        self.end = end

    def __repr__(self):
        return f"FunctionSpan({self.qualified_name}, {self.start}-{self.end})"


def extract_function_spans(source: str) -> Tuple[List[FunctionSpan], bool]:
    """Parse Python source and extract all function/method spans.

    Returns:
        (list of FunctionSpan, parse_ok)
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], False

    spans = []
    _walk_node(tree, prefix="", spans=spans)
    return spans, True


def _walk_node(node: ast.AST, prefix: str, spans: List[FunctionSpan]):
    """Recursively walk AST collecting function/method spans."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            class_prefix = f"{prefix}{child.name}." if prefix else f"{child.name}."
            # Also record methods inside the class
            _walk_node(child, class_prefix, spans)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qname = f"{prefix}{child.name}"
            start = child.lineno
            end = child.end_lineno if hasattr(child, "end_lineno") and child.end_lineno else start
            spans.append(FunctionSpan(child.name, qname, start, end))
            # Handle nested functions/classes inside the function
            nested_prefix = f"{qname}."
            _walk_node(child, nested_prefix, spans)


# ---------------------------------------------------------------------------
# Diff parsing
# ---------------------------------------------------------------------------

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_diff_old_lines(diff_text: str) -> Set[int]:
    """Extract old-side (pre-fix) modified line numbers from a unified diff.

    We collect lines that were removed or changed (lines starting with '-'
    that are not diff headers).  These represent lines in the old file that
    were touched by the fix.
    """
    modified_lines: Set[int] = set()
    old_lineno = 0

    for line in diff_text.splitlines():
        hunk_match = _HUNK_RE.match(line)
        if hunk_match:
            old_lineno = int(hunk_match.group(1))
            continue

        if old_lineno == 0:
            # Haven't hit a hunk yet (file header lines)
            continue

        if line.startswith("-") and not line.startswith("---"):
            modified_lines.add(old_lineno)
            old_lineno += 1
        elif line.startswith("+") and not line.startswith("+++"):
            # Added line — does not consume old line numbers
            pass
        else:
            # Context line
            old_lineno += 1

    return modified_lines


def parse_diff_new_lines(diff_text: str) -> Set[int]:
    """Extract new-side (post-fix) modified line numbers from a unified diff.

    Collects lines that were added (lines starting with '+').
    """
    modified_lines: Set[int] = set()
    new_lineno = 0

    for line in diff_text.splitlines():
        hunk_match = _HUNK_RE.match(line)
        if hunk_match:
            new_lineno = int(hunk_match.group(1))
            continue

        if new_lineno == 0:
            continue

        if line.startswith("+") and not line.startswith("+++"):
            modified_lines.add(new_lineno)
            new_lineno += 1
        elif line.startswith("-") and not line.startswith("---"):
            # Removed line — does not consume new line numbers
            pass
        else:
            new_lineno += 1

    return modified_lines


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_show_file(repo_dir: str, commit: str, filepath: str) -> Optional[str]:
    """Get file content at a specific commit via `git show`."""
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{filepath}"],
            cwd=repo_dir, capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, UnicodeDecodeError):
        pass
    return None


def git_diff(repo_dir: str, base_commit: str, filepath: str,
             target: str = "HEAD") -> Optional[str]:
    """Get unified diff between base_commit and target for a file."""
    try:
        result = subprocess.run(
            ["git", "diff", base_commit, target, "--", filepath],
            cwd=repo_dir, capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, UnicodeDecodeError):
        pass
    return None


def git_has_commit(repo_dir: str, commit: str) -> bool:
    """Check if a commit exists in the repository."""
    try:
        result = subprocess.run(
            ["git", "cat-file", "-t", commit],
            cwd=repo_dir, capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and result.stdout.strip() == "commit"
    except subprocess.TimeoutExpired:
        return False


def read_file_at_head(repo_dir: str, filepath: str) -> Optional[str]:
    """Read a file from the working tree (HEAD checkout)."""
    full_path = os.path.join(repo_dir, filepath)
    if not os.path.isfile(full_path):
        return None
    try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except (OSError, IOError):
        return None


# ---------------------------------------------------------------------------
# Core label extraction
# ---------------------------------------------------------------------------

def map_lines_to_functions(
    modified_lines: Set[int], spans: List[FunctionSpan]
) -> Tuple[List[str], bool]:
    """Map a set of modified line numbers to enclosing function spans.

    Returns:
        (list of qualified function names that overlap, module_level_edit)
    """
    if not modified_lines:
        return [], False

    hit_functions: List[str] = []
    covered_lines: Set[int] = set()

    for span in spans:
        span_range = set(range(span.start, span.end + 1))
        if modified_lines & span_range:
            hit_functions.append(span.qualified_name)
            covered_lines |= span_range

    module_level = bool(modified_lines - covered_lines)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for fn in hit_functions:
        if fn not in seen:
            seen.add(fn)
            unique.append(fn)

    return unique, module_level


def extract_labels_diff_mode(
    repo_dir: str, base_commit: str, filepath: str
) -> Optional[Dict]:
    """Extract function labels using git diff (base_commit available).

    Strategy:
      1. git show base_commit:file -> parse AST for old-side function spans
      2. git diff base_commit HEAD -- file -> extract old-side modified lines
      3. Map modified lines -> enclosing functions (positive labels)
      4. Also parse HEAD file for new-side additions (new functions, additions)
    """
    # --- Old side ---
    old_source = git_show_file(repo_dir, base_commit, filepath)
    diff_text = git_diff(repo_dir, base_commit, filepath)

    if diff_text is None or diff_text.strip() == "":
        # No diff means no change (or error)
        return None

    # Parse old-side AST for function spans
    old_spans: List[FunctionSpan] = []
    if old_source is not None:
        old_spans, _ = extract_function_spans(old_source)

    # Parse diff for old-side modified lines
    old_modified = parse_diff_old_lines(diff_text)

    # Map to functions
    old_hits, old_module_edit = map_lines_to_functions(old_modified, old_spans)

    # --- New side (to catch newly added functions) ---
    new_source = read_file_at_head(repo_dir, filepath)
    new_spans: List[FunctionSpan] = []
    if new_source is not None:
        new_spans, _ = extract_function_spans(new_source)

    new_modified = parse_diff_new_lines(diff_text)
    new_hits, new_module_edit = map_lines_to_functions(new_modified, new_spans)

    # Merge positive functions from both sides
    seen = set()
    positive = []
    for fn in old_hits + new_hits:
        if fn not in seen:
            seen.add(fn)
            positive.append(fn)

    # All functions = union of old and new spans
    all_funcs_seen = set()
    all_funcs = []
    for span in old_spans + new_spans:
        if span.qualified_name not in all_funcs_seen:
            all_funcs_seen.add(span.qualified_name)
            all_funcs.append(span.qualified_name)

    module_level = old_module_edit or new_module_edit

    return {
        "positive_functions": positive,
        "all_functions": sorted(all_funcs),
        "module_level_edit": module_level,
    }


def extract_labels_match_mode(
    repo_dir: str, filepath: str, changed_functions_bare: List[str]
) -> Optional[Dict]:
    """Extract function labels by matching bare names to AST (no base_commit).

    Fallback when we cannot do a git diff. Parses the HEAD file to get all
    function spans with qualified names, then matches the bare `changed_functions`
    names from the training data to qualified names found in this file.
    """
    source = read_file_at_head(repo_dir, filepath)
    if source is None:
        return None

    spans, ok = extract_function_spans(source)
    if not ok:
        return None

    all_funcs = sorted(set(s.qualified_name for s in spans))

    # Build bare-name -> list of qualified names
    bare_to_qualified: Dict[str, List[str]] = defaultdict(list)
    for s in spans:
        bare_to_qualified[s.name].append(s.qualified_name)

    # Match
    changed_set = set(changed_functions_bare)
    positive = []
    positive_set = set()
    for bare_name in sorted(changed_set):
        for qname in bare_to_qualified.get(bare_name, []):
            if qname not in positive_set:
                positive_set.add(qname)
                positive.append(qname)

    return {
        "positive_functions": positive,
        "all_functions": all_funcs,
        "module_level_edit": False,  # Cannot determine without diff
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_example(
    item: Dict, repos_dir: str
) -> Optional[Dict]:
    """Process a single training example and extract function-level labels.

    Returns the changed_functions_detailed dict, or None on failure.
    """
    repo = item["repo"]
    repo_dir = os.path.join(repos_dir, repo)
    if not os.path.isdir(repo_dir):
        return None

    base_commit = item.get("base_commit")
    changed_py_files = item.get("changed_py_files", [])
    changed_functions_bare = item.get("changed_functions", [])

    if not changed_py_files:
        return {}

    # Determine mode: diff vs match
    use_diff = (
        base_commit is not None
        and base_commit.strip() != ""
        and git_has_commit(repo_dir, base_commit)
    )

    result = {}

    for filepath in sorted(changed_py_files):
        if use_diff:
            file_labels = extract_labels_diff_mode(repo_dir, base_commit, filepath)
        else:
            file_labels = extract_labels_match_mode(
                repo_dir, filepath, changed_functions_bare
            )

        if file_labels is not None:
            result[filepath] = file_labels

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract function-level labels from git diffs or AST matching."
    )
    parser.add_argument(
        "--repos_dir", default="data/repos",
        help="Directory containing cloned repos (default: data/repos)"
    )
    parser.add_argument(
        "--train_data", default="data/grepo_text/grepo_train.jsonl",
        help="Input JSONL training data (default: data/grepo_text/grepo_train.jsonl)"
    )
    parser.add_argument(
        "--output", default="data/grepo_text/grepo_train_function_labels.jsonl",
        help="Output JSONL path (default: data/grepo_text/grepo_train_function_labels.jsonl)"
    )
    parser.add_argument(
        "--max_examples", type=int, default=0,
        help="Process at most N examples (0 = all, for debugging)"
    )
    args = parser.parse_args()

    # Load training data
    print(f"Loading training data from {args.train_data} ...")
    examples = []
    with open(args.train_data, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"  Loaded {len(examples)} examples")

    if args.max_examples > 0:
        examples = examples[: args.max_examples]
        print(f"  Truncated to {len(examples)} examples (--max_examples)")

    # Sort deterministically
    examples.sort(key=lambda x: (x["repo"], x.get("issue_id", 0)))

    # Process
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    stats = {
        "total": len(examples),
        "processed": 0,
        "diff_mode": 0,
        "match_mode": 0,
        "skipped_no_repo": 0,
        "skipped_no_py_files": 0,
        "files_processed": 0,
        "files_with_positives": 0,
        "total_positive_functions": 0,
        "total_all_functions": 0,
        "module_level_edits": 0,
    }

    results = []
    prev_repo = None

    for i, item in enumerate(examples):
        repo = item["repo"]
        if repo != prev_repo:
            print(f"  [{i+1}/{len(examples)}] Processing repo: {repo}")
            prev_repo = repo

        detailed = process_example(item, args.repos_dir)

        if detailed is None:
            stats["skipped_no_repo"] += 1
            out_item = dict(item)
            out_item["changed_functions_detailed"] = {}
            results.append(out_item)
            continue

        if not item.get("changed_py_files"):
            stats["skipped_no_py_files"] += 1

        # Determine which mode was used
        base_commit = item.get("base_commit")
        repo_dir = os.path.join(args.repos_dir, repo)
        if (
            base_commit
            and base_commit.strip()
            and os.path.isdir(repo_dir)
            and git_has_commit(repo_dir, base_commit)
        ):
            stats["diff_mode"] += 1
        else:
            stats["match_mode"] += 1

        # Accumulate stats
        stats["processed"] += 1
        for fpath, flabels in detailed.items():
            stats["files_processed"] += 1
            if flabels["positive_functions"]:
                stats["files_with_positives"] += 1
            stats["total_positive_functions"] += len(flabels["positive_functions"])
            stats["total_all_functions"] += len(flabels["all_functions"])
            if flabels["module_level_edit"]:
                stats["module_level_edits"] += 1

        out_item = dict(item)
        out_item["changed_functions_detailed"] = detailed
        results.append(out_item)

        if (i + 1) % 500 == 0:
            print(f"    Progress: {i+1}/{len(examples)} "
                  f"({stats['processed']} processed, "
                  f"{stats['files_with_positives']} files with positives)")

    # Write output
    with open(args.output, "w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nOutput written to {args.output}")
    print(f"\n{'='*60}")
    print(f"Statistics:")
    print(f"{'='*60}")
    print(f"  Total examples:           {stats['total']}")
    print(f"  Processed:                {stats['processed']}")
    print(f"    Diff mode:              {stats['diff_mode']}")
    print(f"    Match mode:             {stats['match_mode']}")
    print(f"  Skipped (no repo):        {stats['skipped_no_repo']}")
    print(f"  Skipped (no py files):    {stats['skipped_no_py_files']}")
    print(f"  Files processed:          {stats['files_processed']}")
    print(f"  Files with positives:     {stats['files_with_positives']}")
    print(f"  Total positive functions: {stats['total_positive_functions']}")
    print(f"  Total all functions:      {stats['total_all_functions']}")
    print(f"  Module-level edits:       {stats['module_level_edits']}")

    if stats["total_all_functions"] > 0:
        ratio = stats["total_positive_functions"] / stats["total_all_functions"]
        print(f"  Positive ratio:           {ratio:.4f} "
              f"({stats['total_positive_functions']}/{stats['total_all_functions']})")


if __name__ == "__main__":
    main()
