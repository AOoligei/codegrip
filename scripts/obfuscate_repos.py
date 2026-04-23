#!/usr/bin/env python3
"""
Obfuscate all Python files in data/repos/ by replacing user-defined identifiers
with random tokens (v_001, f_002, c_003, etc.), preserving syntax.

Goal: create data/repos_obfuscated/ for the identifier obfuscation experiment.
If the code-centric scorer performs similarly on obfuscated code, then the
"code signal" is structural/syntactic, not from meaningful identifier names.

Strategy:
  1. Parse each .py file with AST
  2. Collect all user-defined identifiers (variables, functions, classes, args)
  3. Build a deterministic mapping: original_name -> obfuscated_name
  4. Apply replacement via AST (using ast.unparse on Python 3.9+)
  5. If AST fails, fall back to regex-based token replacement

Preserves:
  - Python keywords and builtins
  - Import module names (structural, not semantic)
  - String literals
  - Numeric literals
  - Comments (best-effort)
"""

import ast
import builtins
import hashlib
import keyword
import os
import re
import sys
import time
import random
from collections import defaultdict
from typing import Dict, Set, Optional

random.seed(42)

# Python builtins and keywords to never replace
PYTHON_BUILTINS = set(dir(builtins))
PYTHON_KEYWORDS = set(keyword.kwlist)
# Also protect common dunder names and test framework names
PROTECTED = PYTHON_BUILTINS | PYTHON_KEYWORDS | {
    'self', 'cls', 'super', 'None', 'True', 'False',
    '__init__', '__new__', '__del__', '__repr__', '__str__',
    '__len__', '__getitem__', '__setitem__', '__delitem__',
    '__iter__', '__next__', '__contains__', '__call__',
    '__enter__', '__exit__', '__get__', '__set__', '__delete__',
    '__eq__', '__ne__', '__lt__', '__gt__', '__le__', '__ge__',
    '__hash__', '__bool__', '__add__', '__sub__', '__mul__',
    '__truediv__', '__floordiv__', '__mod__', '__pow__',
    '__and__', '__or__', '__xor__', '__invert__', '__neg__',
    '__pos__', '__abs__', '__radd__', '__rsub__', '__rmul__',
    '__iadd__', '__isub__', '__imul__', '__itruediv__',
    '__getattr__', '__setattr__', '__delattr__', '__getattribute__',
    '__class__', '__dict__', '__doc__', '__module__', '__name__',
    '__qualname__', '__slots__', '__weakref__', '__all__',
    '__file__', '__path__', '__package__', '__spec__',
    '__loader__', '__builtins__', '__cached__', '__import__',
    '__annotations__', '__bases__', '__mro__', '__subclasses__',
    '__init_subclass__', '__class_getitem__', '__set_name__',
    '__prepare__', '__instancecheck__', '__subclasscheck__',
    '__abstractmethods__', '__match_args__',
    # Common pytest/unittest
    'setUp', 'tearDown', 'setUpClass', 'tearDownClass',
    'test', 'main',
}


def collect_import_names(tree: ast.AST) -> Set[str]:
    """Collect all names that come from import statements — don't obfuscate these."""
    import_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                import_names.add(name.split('.')[0])
                # Also add the full dotted name parts
                for part in alias.name.split('.'):
                    import_names.add(part)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for part in node.module.split('.'):
                    import_names.add(part)
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                import_names.add(name)
    return import_names


def collect_attribute_names(tree: ast.AST) -> Set[str]:
    """Collect attribute names used in obj.attr — these often reference external APIs."""
    attrs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attrs.add(node.attr)
    return attrs


def collect_user_defined_names(tree: ast.AST) -> Set[str]:
    """Collect names that are definitely user-defined (function defs, class defs,
    variable assignments, argument names)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
            # Arguments
            for arg in node.args.args:
                names.add(arg.arg)
            for arg in node.args.posonlyargs:
                names.add(arg.arg)
            for arg in node.args.kwonlyargs:
                names.add(arg.arg)
            if node.args.vararg:
                names.add(node.args.vararg.arg)
            if node.args.kwarg:
                names.add(node.args.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.Global):
            names.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            names.update(node.names)
    return names


def build_obfuscation_map(user_names: Set[str], import_names: Set[str]) -> Dict[str, str]:
    """Build deterministic mapping from original names to obfuscated tokens."""
    # Filter out protected names and import names
    to_obfuscate = set()
    for name in user_names:
        if name in PROTECTED or name in import_names:
            continue
        if name.startswith('__') and name.endswith('__'):
            continue  # dunder methods
        if name.startswith('_') and len(name) > 1 and name[1] == '_':
            continue  # name-mangled
        to_obfuscate.add(name)

    # Sort for determinism
    sorted_names = sorted(to_obfuscate)
    mapping = {}
    counters = {'v': 0, 'f': 0, 'c': 0, 'a': 0}

    return sorted_names, mapping


class NameObfuscator(ast.NodeTransformer):
    """AST transformer that replaces user-defined names with obfuscated tokens."""

    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping

    def visit_Name(self, node):
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node

    def visit_FunctionDef(self, node):
        if node.name in self.mapping:
            node.name = self.mapping[node.name]
        self._obfuscate_args(node.args)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node):
        if node.name in self.mapping:
            node.name = self.mapping[node.name]
        self._obfuscate_args(node.args)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        if node.name in self.mapping:
            node.name = self.mapping[node.name]
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node):
        # Only obfuscate attribute names if they're in our mapping
        if node.attr in self.mapping:
            node.attr = self.mapping[node.attr]
        self.generic_visit(node)
        return node

    def visit_Global(self, node):
        node.names = [self.mapping.get(n, n) for n in node.names]
        return node

    def visit_Nonlocal(self, node):
        node.names = [self.mapping.get(n, n) for n in node.names]
        return node

    def visit_arg(self, node):
        if node.arg in self.mapping:
            node.arg = self.mapping[node.arg]
        return node

    def _obfuscate_args(self, args):
        for arg in args.args:
            if arg.arg in self.mapping:
                arg.arg = self.mapping[arg.arg]
        for arg in args.posonlyargs:
            if arg.arg in self.mapping:
                arg.arg = self.mapping[arg.arg]
        for arg in args.kwonlyargs:
            if arg.arg in self.mapping:
                arg.arg = self.mapping[arg.arg]
        if args.vararg and args.vararg.arg in self.mapping:
            args.vararg.arg = self.mapping[args.vararg.arg]
        if args.kwarg and args.kwarg.arg in self.mapping:
            args.kwarg.arg = self.mapping[args.kwarg.arg]


def obfuscate_with_ast(source: str) -> Optional[str]:
    """Try AST-based obfuscation. Returns None if it fails."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    import_names = collect_import_names(tree)
    user_names = collect_user_defined_names(tree)

    # Build mapping
    to_obfuscate = set()
    for name in user_names:
        if name in PROTECTED or name in import_names:
            continue
        if name.startswith('__') and name.endswith('__'):
            continue
        to_obfuscate.add(name)

    if not to_obfuscate:
        return source  # Nothing to obfuscate

    # Sort for determinism, build mapping with prefixes based on definition type
    func_names = set()
    class_names = set()
    arg_names = set()
    var_names = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in to_obfuscate:
                func_names.add(node.name)
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                if arg.arg in to_obfuscate:
                    arg_names.add(arg.arg)
            if node.args.vararg and node.args.vararg.arg in to_obfuscate:
                arg_names.add(node.args.vararg.arg)
            if node.args.kwarg and node.args.kwarg.arg in to_obfuscate:
                arg_names.add(node.args.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            if node.name in to_obfuscate:
                class_names.add(node.name)

    # Everything else that's in to_obfuscate but not func/class/arg
    var_names = to_obfuscate - func_names - class_names - arg_names

    mapping = {}
    for i, name in enumerate(sorted(func_names)):
        mapping[name] = f"f_{i:03d}"
    for i, name in enumerate(sorted(class_names)):
        mapping[name] = f"c_{i:03d}"
    for i, name in enumerate(sorted(arg_names)):
        mapping[name] = f"a_{i:03d}"
    for i, name in enumerate(sorted(var_names)):
        mapping[name] = f"v_{i:03d}"

    # Apply transformation
    transformer = NameObfuscator(mapping)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        return ast.unparse(new_tree)
    except Exception:
        return None


def obfuscate_with_regex(source: str) -> str:
    """Fallback: regex-based obfuscation for files that fail AST parsing."""
    # Tokenize with a simple regex, replace identifiers
    # This is less precise but handles syntax errors

    # Collect candidate identifiers (words that look like Python names)
    token_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')

    # Find all unique identifiers
    all_ids = set(token_pattern.findall(source))

    # Filter out protected names
    to_replace = set()
    for name in all_ids:
        if name in PROTECTED:
            continue
        if name.startswith('__') and name.endswith('__'):
            continue
        to_replace.add(name)

    if not to_replace:
        return source

    # Build mapping
    mapping = {}
    for i, name in enumerate(sorted(to_replace)):
        mapping[name] = f"x_{i:03d}"

    # Replace using word boundaries
    def replacer(match):
        word = match.group(0)
        return mapping.get(word, word)

    # Process line by line, skip string literals (best-effort)
    lines = source.split('\n')
    result = []
    in_multiline_string = False

    for line in lines:
        stripped = line.lstrip()
        # Very rough heuristic: skip lines that are in docstrings
        if '"""' in line or "'''" in line:
            triple_count = line.count('"""') + line.count("'''")
            if triple_count % 2 == 1:
                in_multiline_string = not in_multiline_string
            result.append(line)
            continue
        if in_multiline_string:
            result.append(line)
            continue

        # For comment lines, don't replace
        if stripped.startswith('#'):
            result.append(line)
            continue

        # Split line into code and comment parts
        # (naive: doesn't handle # inside strings)
        comment_idx = line.find('#')
        if comment_idx >= 0:
            code_part = line[:comment_idx]
            comment_part = line[comment_idx:]
            code_part = token_pattern.sub(replacer, code_part)
            result.append(code_part + comment_part)
        else:
            result.append(token_pattern.sub(replacer, line))

    return '\n'.join(result)


def obfuscate_file(source: str) -> str:
    """Obfuscate a Python file, trying AST first, regex fallback."""
    result = obfuscate_with_ast(source)
    if result is not None:
        return result
    return obfuscate_with_regex(source)


def process_repos(src_dir: str, dst_dir: str, sample_n: int = 0):
    """Process all Python files in all repos."""
    repos = sorted([d for d in os.listdir(src_dir)
                    if os.path.isdir(os.path.join(src_dir, d))])

    stats = {
        'total_files': 0,
        'ast_success': 0,
        'regex_fallback': 0,
        'copy_nonpy': 0,
        'errors': 0,
    }

    all_py_files = []
    for repo in repos:
        repo_src = os.path.join(src_dir, repo)
        for root, dirs, files in os.walk(repo_src):
            for f in files:
                if f.endswith('.py'):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, src_dir)
                    all_py_files.append((full_path, rel_path))

    print(f"Found {len(all_py_files)} Python files across {len(repos)} repos")

    if sample_n > 0:
        random.seed(42)
        sample = random.sample(all_py_files, min(sample_n, len(all_py_files)))
        print(f"Sampling {len(sample)} files for verification")
        all_py_files = sample

    start = time.time()
    for i, (full_path, rel_path) in enumerate(all_py_files):
        dst_path = os.path.join(dst_dir, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        try:
            with open(full_path, 'r', errors='replace') as f:
                source = f.read()
        except Exception as e:
            stats['errors'] += 1
            continue

        stats['total_files'] += 1

        # Try AST first
        ast_result = obfuscate_with_ast(source)
        if ast_result is not None:
            stats['ast_success'] += 1
            with open(dst_path, 'w') as f:
                f.write(ast_result)
        else:
            stats['regex_fallback'] += 1
            regex_result = obfuscate_with_regex(source)
            with open(dst_path, 'w') as f:
                f.write(regex_result)

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(all_py_files)}] {elapsed:.0f}s "
                  f"(AST: {stats['ast_success']}, regex: {stats['regex_fallback']})")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Total Python files: {stats['total_files']}")
    print(f"  AST success: {stats['ast_success']}")
    print(f"  Regex fallback: {stats['regex_fallback']}")
    print(f"  Errors: {stats['errors']}")

    return stats


def verify_sample(dst_dir: str, n: int = 10):
    """Verify that obfuscated files are valid Python."""
    py_files = []
    for root, dirs, files in os.walk(dst_dir):
        for f in files:
            if f.endswith('.py'):
                py_files.append(os.path.join(root, f))

    if not py_files:
        print("No Python files found in output!")
        return

    random.seed(42)
    sample = random.sample(py_files, min(n, len(py_files)))

    parse_ok = 0
    for path in sample:
        with open(path, 'r') as f:
            source = f.read()
        try:
            ast.parse(source)
            parse_ok += 1
        except SyntaxError:
            rel = os.path.relpath(path, dst_dir)
            print(f"  WARN: {rel} doesn't parse (may be regex-fallback)")

    print(f"\nVerification: {parse_ok}/{len(sample)} sample files parse successfully")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', default='data/repos',
                        help='Source repos directory')
    parser.add_argument('--dst_dir', default='/data/chenlibin/grepo_agent_experiments/repos_obfuscated',
                        help='Output directory for obfuscated repos')
    parser.add_argument('--sample', type=int, default=0,
                        help='If >0, only process this many files (for testing)')
    parser.add_argument('--verify', action='store_true',
                        help='Run verification on output')
    args = parser.parse_args()

    print(f"Obfuscating repos: {args.src_dir} -> {args.dst_dir}")
    stats = process_repos(args.src_dir, args.dst_dir, sample_n=args.sample)

    if args.verify or args.sample > 0:
        verify_sample(args.dst_dir, n=50)
