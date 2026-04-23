#!/usr/bin/env python3
"""
After structural ablation evals complete, read results and print
the filled table rows for paper/latex/main.tex Table tab:structural.
Usage: python scripts/fill_structural_table.py
"""
import json
from pathlib import Path

BASE = Path("experiments")

configs = [
    ("Graph-hard (co-change + import)", "rankft_runB_graph"),
    ("Same-directory", "rankft_ablation_samedir"),
    ("Path-edit-distance ($\\leq 2$ components)", "rankft_ablation_pathdist"),
    ("Tree-neighbor (sibling dirs)", "rankft_ablation_treeneighbor"),
    ("Combined struct.\\ (10\\%+10\\%+5\\%)", "rankft_ablation_allstruct"),
]

rows = []
for label, exp in configs:
    summary_path = BASE / exp / "eval_merged_rerank" / "summary.json"
    if not summary_path.exists():
        print(f"MISSING: {summary_path}")
        rows.append((label, None))
        continue
    d = json.loads(summary_path.read_text())["overall"]
    r1 = d["recall@1"]
    r5 = d["recall@5"]
    r10 = d["recall@10"]
    # conditional acc@1: fraction of correct top-1 among examples where GT is in pool
    cond = d.get("cond_acc@1", d.get("conditional_recall@1", None))
    rows.append((label, r1, r5, r10, cond))

print("\n=== LaTeX rows for Table tab:structural ===\n")
best_r1 = max((r[1] for r in rows if r[1] is not None), default=0)
for row in rows:
    if row[1] is None:
        print(f"{row[0]} & TBD & TBD & TBD & TBD \\\\")
        continue
    label, r1, r5, r10, cond = row
    bold = r1 >= best_r1 - 0.01
    r1_s = f"\\textbf{{{r1:.2f}}}" if bold else f"{r1:.2f}"
    r5_s = f"\\textbf{{{r5:.2f}}}" if bold else f"{r5:.2f}"
    r10_s = f"\\textbf{{{r10:.2f}}}" if bold else f"{r10:.2f}"
    cond_s = f"{cond*100:.1f}\\%" if cond is not None else "---"
    print(f"{label} & {r1_s} & {r5_s} & {r10_s} & {cond_s} \\\\")

print("\n=== Summary ===")
for row in rows:
    if row[1] is not None:
        print(f"  {row[0]}: R@1={row[1]:.2f}")
    else:
        print(f"  {row[0]}: MISSING")

# ---- Analysis paragraph suggestion ----
graph_r1 = rows[0][1] if rows[0][1] is not None else None
heuristic_rows = [r for r in rows[1:] if r[1] is not None]
if graph_r1 is not None and heuristic_rows:
    best_heuristic = max(heuristic_rows, key=lambda r: r[1])
    gap = graph_r1 - best_heuristic[1]
    print(f"\n=== Suggested analysis paragraph (gap={gap:.2f}) ===\n")
    if gap >= 0.5:
        # Scenario A: graph clearly wins
        heuristic_summary = "; ".join(
            f"{r[0].split('(')[0].strip()} ({r[1]:.2f}\\%)" for r in heuristic_rows
        )
        print(
            f"Table~\\ref{{tab:structural}} shows that graph-hard negatives ({graph_r1:.2f}\\% R@1) "
            f"outperform all evaluated simpler structural heuristics: {heuristic_summary}. "
            f"This demonstrates that the benefit is not attributable solely to structural proximity but to "
            f"the \\emph{{specific co-change and import relationships}} encoded in the graph, "
            f"which identify semantically coupled files in ways that directory structure alone cannot capture."
        )
        print("\n[FRAMING: Keep current thesis. Graph > structural proximity.]\n")
    else:
        # Scenario B: simpler heuristics match
        vals = ", ".join(
            f"{r[0].split('(')[0].strip()} ({r[1]:.2f}\\%)" for r in heuristic_rows
        )
        print(
            f"Table~\\ref{{tab:structural}} reveals that simpler structural heuristics perform comparably "
            f"to graph-hard negatives ({graph_r1:.2f}\\% R@1): {vals}. "
            f"This result clarifies the \\emph{{mechanism}}: the primary driver of improvement is "
            f"\\emph{{structural proximity}}---selecting negatives whose paths are confusable with ground "
            f"truth---rather than the specific co-change or import topology of the dependency graph. "
            f"Graph-hard negatives achieve this naturally (29.7\\% parent-dir overlap vs.\\ 21.4\\% for "
            f"BM25-hard), but any structurally local sampling strategy provides similar benefit. "
            f"The key failure mode of BM25-hard negatives is not keyword hardness but path distinguishability."
        )
        print("\n[FRAMING: Pivot to 'structural proximity > BM25-hard'. Graph is one instance of this.]\n")
else:
    print("\n=== Analysis paragraph: waiting for all results ===\n")
