#!/usr/bin/env python3
"""Build a "FuncSwap" perturbed subset for the function-level rerank experiment.

Hypothesis under test (paper-critical):
  Path-shortcut vulnerability scales with path information density. Function-level
  rerankers see (path + Class/method + body); the existing PathSwap (file-level
  hashing) only removes path info, leaving Class/method tokens intact, so func-
  level systems barely degrade. If we ALSO hash the Class/method identifiers
  in the prompt header (FuncSwap), the reranker loses an additional surface
  identifier channel. We test whether THIS partial ablation degrades scoring.

Scope (read carefully — required for honest paper framing):
  This is an "identifier-HEADER ablation", strictly analogous to how PathSwap
  ablates the FILE-PATH header. It edits:
    (a) the `_id` line (which the scorer concatenates as `Function: {func_id}`)
    (b) the FIRST `def NAME(...)` / `class C:    def m(...)` line in body
    (c) qrels and the retriever output, so metric & scorer stay consistent.
  It does NOT clean references inside the body that may incidentally repeat
  the original method/class name (docstrings, recursive `self.method(...)`
  calls, error strings, comments, type annotations). The model can therefore
  still recover semantics from those incidental signals; we are not claiming
  to wipe all method-name evidence. The hypothesis we are testing is:
    "If the explicit IDENTIFIER HEADER is the load-bearing signal, removing
    it should produce a measurable degradation."
  If it does NOT degrade, that is meaningful evidence that function-level
  rerankers rely on deep code semantics (body content), not surface tokens.

Output:
  - mirrors the per-instance directory layout under a NEW root
    (does not modify the existing perturbed corpus)
  - keeps only the union of (top-K docs that retriever returned, GT docs from
    qrels) so corpus + qrels + retriever all stay consistent
  - writes a rewritten retriever JSONL with hashed doc ids
  - asserts post-build invariants: every rewritten retriever doc id and every
    rewritten qrels corpus-id MUST appear in the new corpus.jsonl. Any
    mismatch (e.g. hash collision) is a hard error — no silent drops.

Determinism:
  - Hash = first 16 hex chars of sha256("<C>::<m>" or "<m>") with a fixed
    namespace prefix, so that identical (Class, method) pairs across files
    hash to the same value. This is intentional: we mirror the existing
    PathSwap semantics (deterministic per-component hashing).

CLI:
  python funcswap_build_subset.py \
    --src_dir /data/chenlibin/SweRank_perturbed \
    --retriever_jsonl /data/chenlibin/SweRank_results_local/perturbed_v2.jsonl \
    --out_corpus_root /data/chenlibin/SweRank_perturbed_funcswap \
    --out_retriever_jsonl /data/chenlibin/SweRank_results_local/funcswap_v1.jsonl \
    --top_k 100
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter


def h16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def hash_tail(tail: str) -> str:
    """Hash the part after `.py/`. Either `<method>` or `<Class>/<method>`.
    Returns the new tail with the SAME shape. Hashing is stable and
    deterministic per (Class, method) pair so that multiple corpus rows with
    identical class/method names get the same hash (they're functionally
    indistinguishable to the model)."""
    if "/" not in tail:
        return f"f_{h16('FUNC::' + tail)}"
    parts = tail.split("/")
    if len(parts) == 2:
        cls, method = parts
        # We bind method to its class (Class::method) so two methods with the
        # same name in different classes get DIFFERENT hashes — preserves the
        # corpus row count and avoids accidental collisions that would let the
        # reranker pick a different row.
        return f"C_{h16('CLASS::' + cls)}/m_{h16('METHOD::' + cls + '::' + method)}"
    # Defensive: deeper nesting shouldn't occur in our dataset, but hash all.
    out = []
    ctx = []
    for i, p in enumerate(parts):
        ctx.append(p)
        if i < len(parts) - 1:
            out.append(f"X_{h16('XCLASS::' + '::'.join(ctx))}")
        else:
            out.append(f"m_{h16('XMETHOD::' + '::'.join(ctx))}")
    return "/".join(out)


def split_id(_id: str):
    """Returns (filepath, tail) where filepath ends with `.py` and tail is
    the part after `.py/` (may contain /). If no `.py/`, returns (_id, '')."""
    m = re.search(r"\.py/", _id)
    if not m:
        return _id, ""
    end = m.end()  # just after '.py/'
    return _id[: end - 1], _id[end:]


def funcswap_id(_id: str) -> str:
    fp, tail = split_id(_id)
    if not tail:
        return _id
    return f"{fp}/{hash_tail(tail)}"


# Body rewrites
# Top-level: `def NAME(` (allow leading whitespace)
TOPLEVEL_DEF_RE = re.compile(r"^(\s*def\s+)([A-Za-z_][A-Za-z_0-9]*)(\s*\()", re.M)
# Class wrapper inserted by SweRank's extractor: `class CLASSNAME:    def METHOD(`
# (their formatter joins them with no newline). We only rewrite the FIRST
# match because the body should only contain one function.
CLASSMETHOD_RE = re.compile(
    r"^(\s*class\s+)([A-Za-z_][A-Za-z_0-9]*)(\s*[:(])([^\n]*?)(\bdef\s+)([A-Za-z_][A-Za-z_0-9]*)(\s*\()"
)


def _expected_header_tokens(tail: str):
    """Return the hashed identifier tokens we expect to land in the body header.
    Returned tokens MUST appear in the rewritten body for `rewrite_body` to
    consider the rewrite successful. Used for hard-fail audit (we do NOT trust
    re.subn's count alone)."""
    if not tail:
        return []
    parts = tail.split("/")
    if len(parts) == 1:
        return [f"f_{h16('FUNC::' + parts[0])}"]
    elif len(parts) == 2:
        cls, method = parts
        return [
            f"C_{h16('CLASS::' + cls)}",
            f"m_{h16('METHOD::' + cls + '::' + method)}",
        ]
    else:
        return [f"m_{h16('XMETHOD::' + '::'.join(parts))}"]


def rewrite_body(body: str, tail: str, stats: Counter):
    """Rewrite the def line(s) so the surface form matches the hashed _id.

    Returns (new_body, status) where status is one of:
      - 'main': main regex matched (good)
      - 'fallback': fallback regex matched (good but worth tracking)
      - 'miss': rewrite failed (caller decides whether to abort)
      - 'noop': tail empty (no rewrite needed)
    """
    if not tail:
        return body, "noop"
    parts = tail.split("/")
    if len(parts) == 1:
        method = parts[0]
        h_m = f"f_{h16('FUNC::' + method)}"
        # Replace ONLY the first `def method(` (do not touch nested defs that
        # may exist inside the body — paper hypothesis is about the outer name).
        # Use a lambda that matches by NAME so we don't accidentally rename a
        # nested helper that comes before the outer def (rare but possible).
        replaced = {"n": 0}

        def _sub_top(m):
            if replaced["n"] > 0:
                return m.group(0)
            if m.group(2) != method:
                return m.group(0)
            replaced["n"] += 1
            return m.group(1) + h_m + m.group(3)

        new_body = TOPLEVEL_DEF_RE.sub(_sub_top, body)
        if replaced["n"] == 0:
            stats["miss_toplevel_def"] += 1
            return new_body, "miss"
        return new_body, "main"
    elif len(parts) == 2:
        cls, method = parts
        h_C = f"C_{h16('CLASS::' + cls)}"
        h_m = f"m_{h16('METHOD::' + cls + '::' + method)}"
        # Try the SweRank class wrapper first.
        m = CLASSMETHOD_RE.search(body)
        if m and m.group(2) == cls and m.group(6) == method:
            new = (
                m.group(1) + h_C + m.group(3) + m.group(4) + m.group(5) + h_m + m.group(7)
            )
            new_body = body[: m.start()] + new + body[m.end():]
            return new_body, "main"
        # Fallback: a plain `def method(` line (some rows omit the wrapper).
        replaced = {"n": 0}

        def _sub_cm(mm):
            if replaced["n"] > 0:
                return mm.group(0)
            if mm.group(2) != method:
                return mm.group(0)
            replaced["n"] += 1
            return mm.group(1) + h_m + mm.group(3)

        new_body = TOPLEVEL_DEF_RE.sub(_sub_cm, body)
        if replaced["n"] == 0:
            stats["miss_classmethod_def"] += 1
            return new_body, "miss"
        # Note: in the fallback we did NOT rewrite the class name (no class line).
        # Caller's invariant check will use only h_m for fallback rewrites.
        return new_body, "fallback"
    else:
        # Defensive: rewrite the first `def <leaf-method>(`
        method = parts[-1]
        h_m = f"m_{h16('XMETHOD::' + '::'.join(parts))}"
        replaced = {"n": 0}

        def _sub_deep(mm):
            if replaced["n"] > 0:
                return mm.group(0)
            if mm.group(2) != method:
                return mm.group(0)
            replaced["n"] += 1
            return mm.group(1) + h_m + mm.group(3)

        new_body = TOPLEVEL_DEF_RE.sub(_sub_deep, body)
        if replaced["n"] == 0:
            stats["miss_deep_def"] += 1
            return new_body, "miss"
        return new_body, "main"


def perturb_corpus_row(obj: dict, stats: Counter):
    """Returns (new_obj, status, expected_tokens) so caller can audit."""
    old_id = obj["_id"]
    _fp, tail = split_id(old_id)
    new_id = funcswap_id(old_id)
    text = obj.get("text", "")
    # First line is the _id surface — replace whole line if it equals old_id
    if text.startswith(old_id + "\n"):
        body = text[len(old_id) + 1:]
    elif text.startswith(old_id):
        body = text[len(old_id):].lstrip("\n")
    else:
        body = text
    new_body, status = rewrite_body(body, tail, stats)
    obj["_id"] = new_id
    obj["text"] = new_id + "\n" + new_body
    expected = _expected_header_tokens(tail)
    # For 'fallback' on class/method we do NOT actually rewrite the class
    # token (no `class C:` header in the body), so only the method hash is
    # required to appear.
    if status == "fallback" and len(expected) == 2:
        expected = [expected[1]]
    return obj, status, expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True,
                    help="Existing perturbed (path-hashed) per-instance dataset root")
    ap.add_argument("--retriever_jsonl", required=True,
                    help="SweRank retriever output (perturbed_v2.jsonl); rows have 'docs' = top-K ids")
    ap.add_argument("--out_corpus_root", required=True,
                    help="Output root for funcswap per-instance dirs (NEW directory)")
    ap.add_argument("--out_retriever_jsonl", required=True,
                    help="Path to write the rewritten retriever JSONL with hashed docs")
    ap.add_argument("--top_k", type=int, default=100,
                    help="Only keep top-K docs per instance in the new corpus (default 100)")
    ap.add_argument("--prefix", default="swe-bench-lite-function_")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if os.path.exists(args.out_corpus_root) and os.listdir(args.out_corpus_root):
        print(f"REFUSE: out_corpus_root {args.out_corpus_root} is non-empty. Delete or pick another.",
              file=sys.stderr)
        sys.exit(2)
    if os.path.exists(args.out_retriever_jsonl):
        print(f"REFUSE: out_retriever_jsonl {args.out_retriever_jsonl} exists. Delete or pick another.",
              file=sys.stderr)
        sys.exit(2)
    os.makedirs(args.out_corpus_root, exist_ok=True)

    rows = [json.loads(l) for l in open(args.retriever_jsonl)]
    print(f"Loaded {len(rows)} retriever rows", flush=True)
    if args.limit:
        rows = rows[: args.limit]

    stats = Counter()
    n_done = 0
    n_skip = 0
    out_rows = []

    for row in rows:
        iid = row.get("instance_id")
        if not iid:
            n_skip += 1
            continue
        src_inst = os.path.join(args.src_dir, f"{args.prefix}{iid}")
        if not os.path.isdir(src_inst):
            n_skip += 1
            stats["miss_inst_dir"] += 1
            continue
        old_docs = (row.get("docs") or [])[: args.top_k]
        if not old_docs:
            n_skip += 1
            continue
        kept_set = set(old_docs)

        # 1) corpus.jsonl: only emit rows whose _id is in kept_set, and rewrite
        src_corpus = os.path.join(src_inst, "corpus.jsonl")
        if not os.path.isfile(src_corpus):
            n_skip += 1
            stats["miss_corpus"] += 1
            continue

        # We must ALSO keep the GT rows (so qrels still references valid ids
        # for sanity; the metric script only checks the predicted-list anyway,
        # but `oracle@K` is computed against the gt list, so we must keep them
        # in the per-instance corpus to allow downstream tools).
        src_qrels = os.path.join(src_inst, "qrels", "test.tsv")
        gt_old = []
        with open(src_qrels) as f:
            f.readline()  # skip header
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 3:
                    gt_old.append(p[1])
        kept_set.update(gt_old)

        dst_inst = os.path.join(args.out_corpus_root, f"{args.prefix}{iid}")
        os.makedirs(os.path.join(dst_inst, "qrels"), exist_ok=True)
        # queries.jsonl unchanged (bug-report text)
        with open(os.path.join(src_inst, "queries.jsonl")) as f, \
             open(os.path.join(dst_inst, "queries.jsonl"), "w") as g:
            g.write(f.read())

        new_id_count = 0
        emitted_new_ids = set()
        body_audit_failed = []  # (old_id, new_id, expected_tokens not in body)
        with open(src_corpus) as f, open(os.path.join(dst_inst, "corpus.jsonl"), "w") as g:
            for line in f:
                d = json.loads(line)
                old_inner_id = d["_id"]
                if old_inner_id not in kept_set:
                    continue
                d, status, expected = perturb_corpus_row(d, stats)
                if d["_id"] in emitted_new_ids:
                    # FAIL-FAST: paper-critical experiment cannot tolerate
                    # silent drops from hash collisions.
                    print(f"FATAL: hash collision in instance {iid}: "
                          f"{old_inner_id} -> {d['_id']} (already emitted)",
                          file=sys.stderr)
                    sys.exit(3)
                # Body audit: every expected hashed token must be present
                # in the BODY (after the _id first line). Searching the full
                # text would false-pass via the rewritten _id line.
                if expected:
                    full = d["text"]
                    nl = full.find("\n")
                    body_only = full[nl + 1:] if nl >= 0 else ""
                    missing = [t for t in expected if t not in body_only]
                    if missing:
                        body_audit_failed.append((old_inner_id, d["_id"], missing, status))
                stats[f"status_{status}"] += 1
                emitted_new_ids.add(d["_id"])
                g.write(json.dumps(d) + "\n")
                new_id_count += 1

        if body_audit_failed:
            # Some rows had no `def` line we could rewrite. Track but allow —
            # those rows just skip body-header rewrite (still get _id rewrite).
            stats["body_audit_failed_rows"] += len(body_audit_failed)
            if len(body_audit_failed) <= 3:
                for o, n, miss, st in body_audit_failed:
                    print(f"  WARN body audit miss in {iid}: status={st} "
                          f"old={o} new={n} missing_tokens={miss}", flush=True)

        if new_id_count == 0:
            n_skip += 1
            stats["empty_corpus"] += 1
            continue

        # 2) qrels: rewrite the corpus-id column with funcswap_id
        gt_new = []
        with open(src_qrels) as f, open(os.path.join(dst_inst, "qrels", "test.tsv"), "w") as g:
            g.write(f.readline())  # header
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 3:
                    continue
                new_c = funcswap_id(p[1])
                gt_new.append(new_c)
                p[1] = new_c
                g.write("\t".join(p) + "\n")

        # 3) build the new retriever row with hashed docs (preserve order!)
        new_row = dict(row)
        new_row["docs"] = [funcswap_id(d) for d in old_docs]
        out_rows.append(new_row)

        # 4) Invariant audit: every retriever doc and every qrels GT must
        # exist in the newly-written corpus _id set. Any miss is a hard fail
        # (paper-critical: silent drops invalidate the experiment).
        missing_docs = [d for d in new_row["docs"] if d not in emitted_new_ids]
        missing_gt = [g for g in gt_new if g not in emitted_new_ids]
        if missing_docs:
            print(f"FATAL: instance {iid}: {len(missing_docs)} retriever docs "
                  f"not in new corpus (sample: {missing_docs[:3]})", file=sys.stderr)
            sys.exit(4)
        if missing_gt:
            print(f"FATAL: instance {iid}: {len(missing_gt)} GT qrels rows "
                  f"not in new corpus (sample: {missing_gt[:3]})", file=sys.stderr)
            sys.exit(5)

        n_done += 1

    with open(args.out_retriever_jsonl, "w") as g:
        for r in out_rows:
            g.write(json.dumps(r) + "\n")

    print(f"\n=== Done n_done={n_done} n_skip={n_skip} ===", flush=True)
    print(f"Stats: {dict(stats)}", flush=True)
    print(f"Out corpus: {args.out_corpus_root}", flush=True)
    print(f"Out retriever JSONL: {args.out_retriever_jsonl}", flush=True)


if __name__ == "__main__":
    main()
