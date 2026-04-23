#!/usr/bin/env python3
"""
PathSwap v2: rewrite imports in code as well.

For each GREPO/SWE-bench repo:
  1. Build a global rename_map: every .py file's import module path -> hashed
  2. Use `ast` to parse each .py file and rewrite Import/ImportFrom nodes
  3. Write rewritten files to a new location

Produces path-hashed code where both file paths AND import statements
use hashed identifiers, closing the lexical-leak loophole.
"""
import argparse, ast, hashlib, json, os, shutil, sys

def hash_component(x):
    if not x: return x
    h = hashlib.sha256(x.encode()).hexdigest()[:8]
    return f"m_{h}" if not x.endswith(".py") else f"m_{h}.py"

def hash_module(dotted):
    """Hash a dotted module path like cirq.value.linear_dict"""
    parts = dotted.split(".")
    return ".".join(hash_component(p) for p in parts)

def rewrite_imports_ast(src, hash_map=None):
    """Rewrite all Import/ImportFrom module names. hash_map=None means default sha256."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, False
    class Rewriter(ast.NodeTransformer):
        def visit_Import(self, node):
            for alias in node.names:
                if alias.name:
                    alias.name = hash_module(alias.name)
            return node
        def visit_ImportFrom(self, node):
            if node.module:
                node.module = hash_module(node.module)
            return node
    new_tree = Rewriter().visit(tree)
    try:
        return ast.unparse(new_tree), True
    except Exception:
        return src, False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_repo", required=True)
    ap.add_argument("--dst_repo", required=True)
    args = ap.parse_args()

    n_rewrite = n_skip = 0
    for root, dirs, files in os.walk(args.src_repo):
        rel_root = os.path.relpath(root, args.src_repo)
        for f in files:
            if not f.endswith(".py"):
                continue
            src_path = os.path.join(root, f)
            dst_dir = os.path.join(args.dst_repo, rel_root)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, f)
            try:
                with open(src_path, "r", errors="replace") as fh:
                    src = fh.read()
                new_src, ok = rewrite_imports_ast(src)
                if ok:
                    with open(dst_path, "w") as fh:
                        fh.write(new_src)
                    n_rewrite += 1
                else:
                    shutil.copy(src_path, dst_path)
                    n_skip += 1
            except Exception:
                n_skip += 1
    print(f"Rewrote {n_rewrite}, skipped {n_skip}")

if __name__ == "__main__":
    main()
