#!/usr/bin/env python3
"""
Collect model scale ablation results.
Usage: python scripts/collect_scale_results.py
"""
import json
from pathlib import Path

BASE = Path("experiments")
SIZES = ["0.5B", "1.5B", "3B", "7B", "14B"]

# 7B uses original experiment names (with SFT), scale ablation uses scale_* (no SFT)
GRAPH_MAP = {
    "0.5B": "scale_0.5B_graph",
    "1.5B": "scale_1.5B_graph",
    "3B":   "scale_3B_graph",
    "7B":   "rankft_runB_graph",       # original 7B with SFT
    "14B":  "scale_14B_graph",
}
BM25_MAP = {
    "0.5B": "scale_0.5B_bm25only",
    "1.5B": "scale_1.5B_bm25only",
    "3B":   "scale_3B_bm25only",
    "7B":   "rankft_runA_bm25only",    # original 7B with SFT
    "14B":  "scale_14B_bm25only",
}

# Also check scale_7B_* (without SFT, fair comparison)
GRAPH_MAP_NOSFT = {s: f"scale_{s}_graph" for s in SIZES}
BM25_MAP_NOSFT = {s: f"scale_{s}_bm25only" for s in SIZES}


def load_r1(exp_name):
    p = BASE / exp_name / "eval_merged_rerank" / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())["overall"]
    return d["recall@1"]


print("=" * 60)
print("MODEL SCALE ABLATION: Graph-hard vs BM25-only")
print("=" * 60)

# Try no-SFT versions first (fair comparison), fallback to original for 7B
print(f"\n{'Size':<8} {'Graph R@1':>10} {'BM25 R@1':>10} {'Delta':>8} {'Note'}")
print("-" * 50)
for size in SIZES:
    # Try no-SFT first
    g = load_r1(GRAPH_MAP_NOSFT[size])
    b = load_r1(BM25_MAP_NOSFT[size])
    note = ""

    # Fallback to with-SFT for 7B
    if g is None and size == "7B":
        g = load_r1(GRAPH_MAP["7B"])
        note = "(+SFT)"
    if b is None and size == "7B":
        b = load_r1(BM25_MAP["7B"])
        if not note:
            note = "(+SFT)"

    g_s = f"{g:.2f}" if g is not None else "PENDING"
    b_s = f"{b:.2f}" if b is not None else "PENDING"
    if g is not None and b is not None:
        d_s = f"{g-b:+.2f}"
    else:
        d_s = "N/A"
    print(f"{size:<8} {g_s:>10} {b_s:>10} {d_s:>8} {note}")

# LaTeX output
print("\n=== LaTeX table rows ===")
for size in SIZES:
    g = load_r1(GRAPH_MAP_NOSFT[size])
    b = load_r1(BM25_MAP_NOSFT[size])
    if g is None and size == "7B":
        g = load_r1(GRAPH_MAP["7B"])
    if b is None and size == "7B":
        b = load_r1(BM25_MAP["7B"])
    if g is not None and b is not None:
        delta = g - b
        print(f"Qwen2.5-{size} & {b:.2f} & {g:.2f} & {delta:+.2f} \\\\")
    else:
        print(f"Qwen2.5-{size} & --- & --- & --- \\\\")
