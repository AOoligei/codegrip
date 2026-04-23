"""Plot budget curves for CodeGRIP paper.

Generates publication-quality figures showing oracle recall and R@1
as a function of candidate pool size.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# BM25 oracle recall at different pool sizes
bm25_sizes = np.array([50, 100, 150, 200, 250, 300, 400, 500])
bm25_recall = np.array([0.6714, 0.7447, 0.7876, 0.8363, 0.8527, 0.8838, 0.9090, 0.9261])

# Fixed-size expansion pools (avg size ~208)
graph_size, graph_recall = 208, 0.9073
random_size, random_recall = 208, 0.8797

# R@1 data points
r1_data = {
    "BM25 top-200": (200, 19.00),
    "Random expansion": (208, 24.69),
    "Graph expansion": (208, 27.01),
    "BM25 top-500": (500, 27.01),
}

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

OUTDIR = Path(__file__).resolve().parent.parent / "paper" / "latex" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# NeurIPS single-column width
SINGLE_COL = 3.25  # inches
DOUBLE_COL = 6.75

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Colors – colorblind-friendly
C_BM25 = "#4472C4"      # blue
C_GRAPH = "#C0392B"      # red
C_RANDOM = "#7F8C8D"     # gray


# ---------------------------------------------------------------------------
# Figure 1: Oracle Recall vs Pool Size
# ---------------------------------------------------------------------------

def plot_oracle_recall():
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))

    # BM25 line
    ax.plot(
        bm25_sizes, bm25_recall,
        marker="o", color=C_BM25, zorder=3,
        label="BM25 retrieval", markersize=4,
    )

    # Graph expansion marker
    ax.plot(
        graph_size, graph_recall,
        marker="*", color=C_GRAPH, markersize=10, zorder=5,
        markeredgewidth=0.4, markeredgecolor="k",
        label="Graph expansion",
    )

    # Random expansion marker
    ax.plot(
        random_size, random_recall,
        marker="^", color=C_RANDOM, markersize=6, zorder=4,
        markeredgewidth=0.4, markeredgecolor="k",
        label="Random expansion",
    )

    # Horizontal dashed line from graph expansion to BM25 curve intersection
    # Graph expansion recall 0.9073 intersects BM25 curve between 400 (0.9090)
    # and 300 (0.8838).  Interpolate:
    #   0.9073 = 0.8838 + (0.9090 - 0.8838) * (x - 300) / (400 - 300)
    #   x = 300 + (0.9073 - 0.8838) / (0.9090 - 0.8838) * 100
    x_intersect = 300 + (graph_recall - 0.8838) / (0.9090 - 0.8838) * 100
    ax.hlines(
        graph_recall, graph_size, x_intersect,
        linestyles="dashed", colors=C_GRAPH, linewidth=0.8, zorder=2,
    )
    # Small vertical tick at intersection
    ax.plot(
        x_intersect, graph_recall,
        marker="|", color=C_GRAPH, markersize=6, zorder=2,
    )

    # Annotation: savings arrow
    ax.annotate(
        f"~{int(round(x_intersect)) - graph_size} fewer\ncandidates",
        xy=((graph_size + x_intersect) / 2, graph_recall),
        xytext=((graph_size + x_intersect) / 2, graph_recall + 0.04),
        fontsize=6.5, color=C_GRAPH, ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-", color=C_GRAPH, lw=0.5),
    )

    ax.set_xlabel("Candidate pool size")
    ax.set_ylabel("Oracle recall")
    ax.set_xlim(0, 530)
    ax.set_ylim(0.63, 0.95)
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(OUTDIR / "budget_curve.pdf")
    fig.savefig(OUTDIR / "budget_curve.png")
    print(f"Saved oracle recall figure to {OUTDIR / 'budget_curve.pdf'}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: R@1 vs Pool Size
# ---------------------------------------------------------------------------

def plot_r1():
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))

    # BM25 points (200, 500)
    bm25_r1_sizes = [200, 500]
    bm25_r1_vals = [19.00, 27.01]
    ax.plot(
        bm25_r1_sizes, bm25_r1_vals,
        marker="o", color=C_BM25, markersize=4, zorder=3,
        label="BM25 retrieval", linestyle="--", linewidth=0.8,
    )

    # Graph expansion
    ax.plot(
        r1_data["Graph expansion"][0], r1_data["Graph expansion"][1],
        marker="*", color=C_GRAPH, markersize=10, zorder=5,
        markeredgewidth=0.4, markeredgecolor="k",
        label="Graph expansion",
    )

    # Random expansion
    ax.plot(
        r1_data["Random expansion"][0], r1_data["Random expansion"][1],
        marker="^", color=C_RANDOM, markersize=6, zorder=4,
        markeredgewidth=0.4, markeredgecolor="k",
        label="Random expansion",
    )

    # Horizontal dashed line: graph expansion R@1 = 27.01 matches BM25 top-500
    ax.hlines(
        27.01, 208, 500,
        linestyles="dashed", colors=C_GRAPH, linewidth=0.8, zorder=2,
    )
    ax.annotate(
        "Same R@1,\n~2.4x fewer candidates",
        xy=(354, 27.01),
        xytext=(354, 22.0),
        fontsize=6.5, color=C_GRAPH, ha="center", va="top",
        arrowprops=dict(arrowstyle="-", color=C_GRAPH, lw=0.5),
    )

    ax.set_xlabel("Candidate pool size")
    ax.set_ylabel("Recall@1 (%)")
    ax.set_xlim(150, 550)
    ax.set_ylim(15, 31)
    ax.set_xticks([200, 300, 400, 500])
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(OUTDIR / "budget_curve_r1.pdf")
    fig.savefig(OUTDIR / "budget_curve_r1.png")
    print(f"Saved R@1 figure to {OUTDIR / 'budget_curve_r1.pdf'}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_oracle_recall()
    plot_r1()
    print("Done.")
