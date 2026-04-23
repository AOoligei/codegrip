#!/usr/bin/env python3
"""Build a clean-lite corpus on top of funcswap_v1.

Clean-lite preserves the existing funcswap `_id` space exactly and only edits
the scorer-visible function body. It sanitizes repo-local module/path mentions
in three places:
  1. absolute import statements
  2. comments and docstrings
  3. string literals that look like module/path references

The goal is to remove residual lexical leakage without over-sanitizing normal
code identifiers or natural language.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tokenize
from collections import Counter
from pathlib import Path


GENERIC_WRAPPERS = {"src", "lib", "python", "py", "package", "packages"}
STDLIB_ROOTS = set(getattr(sys, "stdlib_module_names", set())) | {
    "abc",
    "argparse",
    "asyncio",
    "collections",
    "contextlib",
    "dataclasses",
    "functools",
    "io",
    "itertools",
    "json",
    "math",
    "os",
    "pathlib",
    "re",
    "sys",
    "tempfile",
    "threading",
    "time",
    "typing",
    "unittest",
    "urllib",
}

FROM_IMPORT_RE = re.compile(r"^(\s*from\s+)([A-Za-z_][A-Za-z0-9_\.]*)(\s+import\b.*)$")
IMPORT_STMT_RE = re.compile(r"^(\s*import\s+)(.+)$")
IMPORT_PART_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_\.]*)(\s+as\s+[A-Za-z_][A-Za-z0-9_]*)?$")
DOTTED_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)(?![A-Za-z0-9_./-])"
)
SLASH_FILE_RE = re.compile(
    r"(?<!://)(?<![A-Za-z0-9_./-])([A-Za-z_][A-Za-z0-9_-]*(?:/[A-Za-z0-9_][A-Za-z0-9_.-]*)+\.py)(?![A-Za-z0-9_./-])"
)
SLASH_DIR_RE = re.compile(
    r"(?<!://)(?<![A-Za-z0-9_./-])([A-Za-z_][A-Za-z0-9_-]*(?:/[A-Za-z0-9_][A-Za-z0-9_.-]*)+/)(?![A-Za-z0-9_./-])"
)


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class AliasMapper:
    """Deterministic componentwise aliases aligned with pathswap hashing."""

    def __init__(self):
        self.dir_cache = {}
        self.file_cache = {}
        self.mod_cache = {}

    def dir_alias(self, part: str) -> str:
        part = part.lower()
        if part not in self.dir_cache:
            self.dir_cache[part] = f"d_{sha16(part)}"
        return self.dir_cache[part]

    def file_alias(self, filename: str) -> str:
        filename = filename.lower()
        if filename not in self.file_cache:
            self.file_cache[filename] = f"m_{sha16(filename)}.py"
        return self.file_cache[filename]

    def module_alias(self, part: str) -> str:
        part = part.lower()
        if part not in self.mod_cache:
            self.mod_cache[part] = f"m_{sha16(part + '.py')}"
        return self.mod_cache[part]

    def dotted_alias(self, dotted: str) -> str:
        parts = [p.lower() for p in dotted.split(".") if p]
        if not parts:
            return dotted
        if len(parts) == 1:
            return self.dir_alias(parts[0])
        out = [self.dir_alias(p) for p in parts[:-1]]
        out.append(self.module_alias(parts[-1]))
        return ".".join(out)

    def slash_alias(self, path: str) -> str:
        trailing = path.endswith("/")
        parts = [p.lower() for p in path.strip("/").split("/") if p]
        if not parts:
            return path
        out = []
        for i, part in enumerate(parts):
            is_last = i == len(parts) - 1
            if is_last and part.endswith(".py"):
                out.append(self.file_alias(part))
            else:
                out.append(self.dir_alias(part))
        text = "/".join(out)
        if trailing:
            text += "/"
        return text


def split_item_id(item_id: str) -> tuple[str, str]:
    m = re.search(r"\.py/", item_id)
    if not m:
        return item_id, ""
    end = m.end()
    return item_id[: end - 1], item_id[end:]


def infer_repo_roots(instance_id: str, normal_inst_dir: str) -> set[str]:
    roots = set()
    if os.path.isfile(os.path.join(normal_inst_dir, "corpus.jsonl")):
        with open(os.path.join(normal_inst_dir, "corpus.jsonl"), encoding="utf-8") as handle:
            for line in handle:
                obj = json.loads(line)
                filepath, _ = split_item_id(obj["_id"])
                parts = [p for p in filepath.split("/") if p]
                if not parts:
                    continue
                roots.add(parts[0].lower())
                if len(parts) >= 2 and parts[0].lower() in GENERIC_WRAPPERS:
                    roots.add(parts[1].lower())
    repo_part = instance_id.split("__", 1)[-1].rsplit("-", 1)[0].lower()
    for cand in {repo_part, repo_part.replace("-", "_"), repo_part.replace("-", "")}:
        if cand and cand not in STDLIB_ROOTS:
            roots.add(cand)
    return roots


def is_repo_local_root(root: str, repo_roots: set[str]) -> bool:
    root = root.lower()
    return root in repo_roots and root not in STDLIB_ROOTS


def clean_import_line(line: str, aliases: AliasMapper, repo_roots: set[str], stats: Counter | None = None) -> str:
    match = FROM_IMPORT_RE.match(line)
    if match:
        module = match.group(2)
        root = module.split(".", 1)[0].lower()
        if is_repo_local_root(root, repo_roots):
            repl = aliases.dotted_alias(module)
            if stats is not None and repl != module:
                stats["import_from_rewrites"] += 1
            return match.group(1) + repl + match.group(3)
        return line

    match = IMPORT_STMT_RE.match(line)
    if not match:
        return line

    changed = False
    parts = []
    for piece in match.group(2).split(","):
        stripped = piece.strip()
        alias_match = IMPORT_PART_RE.match(stripped)
        if not alias_match:
            parts.append(piece)
            continue
        module = alias_match.group(1)
        root = module.split(".", 1)[0].lower()
        if is_repo_local_root(root, repo_roots):
            repl = aliases.dotted_alias(module) + (alias_match.group(2) or "")
            changed = True
            parts.append((" " if piece.startswith(" ") else "") + repl)
        else:
            parts.append(piece)
    if changed and stats is not None:
        stats["import_rewrites"] += 1
    if not changed:
        return line
    return match.group(1) + ",".join(parts)


def _looks_like_url_context(text: str, start: int) -> bool:
    return text[max(0, start - 3):start] == "://"


def replace_dotted_paths(text: str, aliases: AliasMapper, repo_roots: set[str], stats: Counter | None = None) -> str:
    def repl(match: re.Match[str]) -> str:
        value = match.group(1)
        if _looks_like_url_context(text, match.start(1)):
            return value
        root = value.split(".", 1)[0].lower()
        if not is_repo_local_root(root, repo_roots):
            return value
        replaced = aliases.dotted_alias(value)
        if stats is not None and replaced != value:
            stats["dotted_rewrites"] += 1
        return replaced

    return DOTTED_RE.sub(repl, text)


def replace_slash_paths(text: str, aliases: AliasMapper, repo_roots: set[str], stats: Counter | None = None) -> str:
    def slash_repl(match: re.Match[str]) -> str:
        value = match.group(1)
        root = value.split("/", 1)[0].lower()
        if not is_repo_local_root(root, repo_roots):
            return value
        replaced = aliases.slash_alias(value)
        if stats is not None and replaced != value:
            stats["slash_rewrites"] += 1
        return replaced

    text = SLASH_FILE_RE.sub(slash_repl, text)
    text = SLASH_DIR_RE.sub(slash_repl, text)
    return text


def clean_free_text(text: str, aliases: AliasMapper, repo_roots: set[str], stats: Counter | None = None) -> str:
    text = replace_slash_paths(text, aliases, repo_roots, stats)
    text = replace_dotted_paths(text, aliases, repo_roots, stats)
    return text


def clean_body(body: str, aliases: AliasMapper, repo_roots: set[str], stats: Counter | None = None) -> str:
    line_cleaned = "".join(clean_import_line(line, aliases, repo_roots, stats) for line in body.splitlines(keepends=True))
    string_token_types = {tokenize.STRING}
    for name in ("FSTRING_MIDDLE",):
        if hasattr(tokenize, name):
            string_token_types.add(getattr(tokenize, name))
    try:
        tokens = []
        for token in tokenize.generate_tokens(io.StringIO(line_cleaned).readline):
            tok_type = token.type
            tok_str = token.string
            if tok_type == tokenize.COMMENT:
                tok_str = clean_free_text(tok_str, aliases, repo_roots, stats)
            elif tok_type in string_token_types:
                tok_str = clean_free_text(tok_str, aliases, repo_roots, stats)
            if tok_str != token.string:
                token = token._replace(string=tok_str)
            tokens.append(token)
        return tokenize.untokenize(tokens)
    except tokenize.TokenError:
        if stats is not None:
            stats["tokenize_fallbacks"] += 1
        return line_cleaned


def rewrite_corpus_row(obj: dict, aliases: AliasMapper, repo_roots: set[str], stats: Counter) -> dict:
    old_id = obj["_id"]
    text = obj.get("text", "")
    if text.startswith(old_id + "\n"):
        body = text[len(old_id) + 1:]
        cleaned = clean_body(body, aliases, repo_roots, stats)
        obj["text"] = old_id + "\n" + cleaned
    else:
        obj["text"] = clean_body(text, aliases, repo_roots, stats)
    return obj


def build_cleanlite(
    src_dir: str,
    normal_dir: str,
    retriever_jsonl: str,
    out_corpus_root: str,
    out_retriever_jsonl: str,
    prefix: str,
    limit: int | None = None,
) -> Counter:
    if os.path.exists(out_corpus_root) and os.listdir(out_corpus_root):
        raise RuntimeError(f"REFUSE: out_corpus_root {out_corpus_root} is non-empty")
    if os.path.exists(out_retriever_jsonl):
        raise RuntimeError(f"REFUSE: out_retriever_jsonl {out_retriever_jsonl} exists")
    os.makedirs(out_corpus_root, exist_ok=True)

    rows = [json.loads(line) for line in open(retriever_jsonl, encoding="utf-8")]
    if limit is not None:
        rows = rows[:limit]

    stats = Counter()
    out_rows = []

    for row in rows:
        iid = row.get("instance_id")
        if not iid:
            stats["skip_missing_iid"] += 1
            continue
        src_inst = os.path.join(src_dir, f"{prefix}{iid}")
        if not os.path.isdir(src_inst):
            stats["skip_missing_instance_dir"] += 1
            continue

        dst_inst = os.path.join(out_corpus_root, f"{prefix}{iid}")
        os.makedirs(os.path.join(dst_inst, "qrels"), exist_ok=True)
        shutil.copy2(os.path.join(src_inst, "queries.jsonl"), os.path.join(dst_inst, "queries.jsonl"))
        shutil.copy2(os.path.join(src_inst, "qrels", "test.tsv"), os.path.join(dst_inst, "qrels", "test.tsv"))

        normal_inst = os.path.join(normal_dir, f"{prefix}{iid}")
        repo_roots = infer_repo_roots(iid, normal_inst)
        aliases = AliasMapper()

        emitted_ids = set()
        with open(os.path.join(src_inst, "corpus.jsonl"), encoding="utf-8") as handle, \
             open(os.path.join(dst_inst, "corpus.jsonl"), "w", encoding="utf-8") as out_handle:
            for line in handle:
                obj = json.loads(line)
                stats["corpus_rows"] += 1
                obj = rewrite_corpus_row(obj, aliases, repo_roots, stats)
                emitted_ids.add(obj["_id"])
                out_handle.write(json.dumps(obj) + "\n")

        with open(os.path.join(dst_inst, "qrels", "test.tsv"), encoding="utf-8") as handle:
            next(handle)
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3 and parts[1] not in emitted_ids:
                    raise RuntimeError(f"qrels id missing from clean-lite corpus for {iid}: {parts[1]}")

        out_rows.append(dict(row))
        stats["instances_done"] += 1

    with open(out_retriever_jsonl, "w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row) + "\n")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True, help="funcswap_v1 corpus root")
    parser.add_argument("--normal_dir", required=True, help="original SweRank corpus root")
    parser.add_argument("--retriever_jsonl", required=True, help="funcswap_v1 retriever JSONL")
    parser.add_argument("--out_corpus_root", required=True)
    parser.add_argument("--out_retriever_jsonl", required=True)
    parser.add_argument("--prefix", default="swe-bench-lite-function_")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    stats = build_cleanlite(
        src_dir=args.src_dir,
        normal_dir=args.normal_dir,
        retriever_jsonl=args.retriever_jsonl,
        out_corpus_root=args.out_corpus_root,
        out_retriever_jsonl=args.out_retriever_jsonl,
        prefix=args.prefix,
        limit=args.limit,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"Out corpus: {args.out_corpus_root}")
    print(f"Out retriever JSONL: {args.out_retriever_jsonl}")


if __name__ == "__main__":
    main()
