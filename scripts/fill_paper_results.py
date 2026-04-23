#!/usr/bin/env python3
"""
Fill TBD values in paper/latex/main.tex with actual experimental results.
Run this after experiments complete to auto-update tables.

Usage: python scripts/fill_paper_results.py [--dry-run]
"""
import json
import sys
from pathlib import Path

BASE = Path("experiments")
PAPER = Path("paper/latex/main.tex")
DRY_RUN = "--dry-run" in sys.argv


def load_summary(exp_name):
    p = BASE / exp_name / "eval_merged_rerank" / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["overall"]


def fill_samedir():
    """Fill same-directory row in tab:structural."""
    s = load_summary("rankft_ablation_samedir")
    if s is None:
        print("[SKIP] samedir: no results yet")
        return None
    r1, r5, r10 = s["recall@1"], s["recall@5"], s["recall@10"]
    old = "Same-directory & TBD & TBD & TBD & --- \\\\"
    new = f"Same-directory & {r1:.2f} & {r5:.2f} & {r10:.2f} & --- \\\\"
    print(f"[FILL] samedir: R@1={r1:.2f}, R@5={r5:.2f}, R@10={r10:.2f}")
    return (old, new)


def fill_scale():
    """Fill scale ablation table."""
    # Exact spacing from paper: {size}{padding} & ...
    size_patterns = {
        "0.5B": "0.5B  &",
        "1.5B": "1.5B  &",
        "3B":   "3B    &",
        "7B":   "7B    &",
        "14B":  "14B   &",
    }
    replacements = []
    for size, prefix in size_patterns.items():
        g = load_summary(f"scale_{size}_graph")
        b = load_summary(f"scale_{size}_bm25only")
        if g is None or b is None:
            print(f"[SKIP] scale {size}: graph={'OK' if g else 'MISSING'}, bm25={'OK' if b else 'MISSING'}")
            continue
        gr1, br1 = g["recall@1"], b["recall@1"]
        delta = gr1 - br1
        old = f"{prefix} \\texttt{{TBD}} & \\texttt{{TBD}} & \\texttt{{TBD}} \\\\"
        new = f"{prefix} {gr1:.2f} & {br1:.2f} & {delta:+.2f} \\\\"
        replacements.append((old, new))
        print(f"[FILL] scale {size}: graph={gr1:.2f}, bm25={br1:.2f}, delta={delta:+.2f}")
    return replacements


def fill_seeds():
    """Fill seed robustness table. Handles both TBD and already-filled values."""
    import re
    graph_r1s, graph_r5s = [], []
    bm25_r1s, bm25_r5s = [], []

    # Seed 42 = original runs
    g42 = load_summary("rankft_runB_graph")
    b42 = load_summary("rankft_runA_bm25only")
    if g42:
        graph_r1s.append(g42["recall@1"])
        graph_r5s.append(g42["recall@5"])
    if b42:
        bm25_r1s.append(b42["recall@1"])
        bm25_r5s.append(b42["recall@5"])

    for seed in [1, 2, 3, 4]:
        g = load_summary(f"rankft_runB_graph_seed{seed}")
        b = load_summary(f"rankft_runA_bm25only_seed{seed}")
        if g:
            graph_r1s.append(g["recall@1"])
            graph_r5s.append(g["recall@5"])
        if b:
            bm25_r1s.append(b["recall@1"])
            bm25_r5s.append(b["recall@5"])

    replacements = []
    if len(graph_r1s) >= 2:
        import numpy as np
        gr1_mean, gr1_std = np.mean(graph_r1s), np.std(graph_r1s)
        gr5_mean, gr5_std = np.mean(graph_r5s), np.std(graph_r5s)
        # Match both TBD and already-filled patterns
        old_g = re.compile(r"Graph-hard\s+& .+? \\\\")
        new_g = f"Graph-hard   & ${gr1_mean:.2f} \\pm {gr1_std:.2f}$ & ${gr5_mean:.2f} \\pm {gr5_std:.2f}$ \\\\"
        replacements.append(("regex", old_g, new_g))
        print(f"[FILL] seeds graph: R@1={gr1_mean:.2f}+/-{gr1_std:.2f} (n={len(graph_r1s)})")
    else:
        print(f"[SKIP] seeds graph: only {len(graph_r1s)} seed(s) done")

    if len(bm25_r1s) >= 2:
        import numpy as np
        br1_mean, br1_std = np.mean(bm25_r1s), np.std(bm25_r1s)
        br5_mean, br5_std = np.mean(bm25_r5s), np.std(bm25_r5s)
        old_b = re.compile(r"BM25-only\s+& .+? \\\\")
        new_b = f"BM25-only    & ${br1_mean:.2f} \\pm {br1_std:.2f}$ & ${br5_mean:.2f} \\pm {br5_std:.2f}$ \\\\"
        replacements.append(("regex", old_b, new_b))
        print(f"[FILL] seeds bm25: R@1={br1_mean:.2f}+/-{br1_std:.2f} (n={len(bm25_r1s)})")
    else:
        print(f"[SKIP] seeds bm25: only {len(bm25_r1s)} seed(s) done")

    return replacements


def fill_beetlebox():
    """Fill BeetleBox cross-language table."""
    # Graph reranker on BeetleBox — try multiple possible paths
    reranker = None
    for name in ["beetlebox_java_eval", "beetlebox_java_rerank",
                  "rankft_runB_graph/eval_beetlebox_java"]:
        reranker = load_summary(name)
        if reranker is None:
            # Also try direct summary.json (not under eval_merged_rerank/)
            p = BASE / name / "summary.json"
            if p.exists():
                reranker = json.loads(p.read_text())["overall"]
        if reranker is not None:
            break
    if reranker is None:
        print("[SKIP] beetlebox: no reranker results yet")
        return []

    replacements = []
    r1 = reranker["recall@1"]
    r5 = reranker["recall@5"]
    r10 = reranker["recall@10"]
    old = "CodeGRIP reranker & \\texttt{TBD} & \\texttt{TBD} & \\texttt{TBD} \\\\"
    new = f"CodeGRIP reranker & {r1:.2f} & {r5:.2f} & {r10:.2f} \\\\"
    replacements.append((old, new))
    print(f"[FILL] beetlebox reranker: R@1={r1:.2f}, R@5={r5:.2f}, R@10={r10:.2f}")

    # BM25 baseline on BeetleBox (if available)
    # This would need a separate BM25-only eval
    return replacements


def main():
    text = PAPER.read_text()
    changes = []

    # Collect all replacements
    r = fill_samedir()
    if r:
        changes.append(r)

    for r in fill_scale():
        changes.append(r)

    for r in fill_seeds():
        changes.append(r)

    for r in fill_beetlebox():
        changes.append(r)

    if not changes:
        print("\nNo results to fill yet.")
        return

    # Apply replacements
    import re
    for change in changes:
        if len(change) == 3 and change[0] == "regex":
            _, pattern, new = change
            if pattern.search(text):
                text = pattern.sub(lambda m: new, text, count=1)
                print(f"  [OK] replaced (regex)")
            else:
                print(f"  [WARN] regex pattern not found: {pattern.pattern[:60]}...")
        else:
            old, new = change
            if old in text:
                text = text.replace(old, new)
                print(f"  [OK] replaced")
            else:
                print(f"  [WARN] pattern not found in paper: {old[:60]}...")

    # Also remove "(Same-directory results pending.)" if samedir was filled
    pending_note = "(Same-directory results pending.)"
    if pending_note in text and any("samedir" in str(c) for c in changes):
        text = text.replace(pending_note, "")
        print("  [OK] removed samedir pending note")

    if DRY_RUN:
        print(f"\n[DRY RUN] Would write {len(changes)} changes to {PAPER}")
    else:
        PAPER.write_text(text)
        print(f"\n[DONE] Wrote {len(changes)} changes to {PAPER}")


if __name__ == "__main__":
    main()
