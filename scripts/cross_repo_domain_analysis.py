#!/usr/bin/env python3
"""Cross-repo domain analysis: compare graph-hard vs bm25-only R@1 by domain."""

import json
from collections import defaultdict

# ---------- Paths ----------
GRAPH_PATH = "experiments/rankft_runB_graph/eval_merged_rerank/summary.json"
BM25_PATH = "experiments/rankft_runA_bm25only/eval_exp6_rerank/summary.json"

# ---------- Domain mapping ----------
DOMAIN_MAP = {
    "Scientific Computing": [
        "scipy", "jax", "xarray", "astropy", "networkx", "arviz",
        "PyBaMM", "pyvista", "pyomo", "PyPSA", "shapely",
        "feature_engine", "python-control",
    ],
    "Data/ML Infrastructure": [
        "datasets", "litellm", "haystack", "llama-stack", "llama_deploy",
        "dspy", "torchtune", "instructlab", "smolagents",
    ],
    "Developer Tools": [
        "pylint", "astroid", "wemake-python-styleguide", "ipython",
        "jupyter-ai", "twine", "briefcase", "kedro", "dvc", "attrs",
        "poetry", "pipenv",
    ],
    "Web/API": [
        "falcon", "flask", "faststream", "urllib3",
        "python-telegram-bot", "openai-agents-python", "aiogram", "Radicale",
    ],
    "Data Processing": [
        "sqlfluff", "fonttools", "geopandas", "pydicom",
        "pvlib-python", "tablib", "csvkit", "marshmallow",
    ],
    "Infrastructure/DevOps": [
        "cfn-lint", "patroni", "crawlee-python", "dynaconf", "transitions",
    ],
    "Desktop/Media": [
        "Solaar", "qtile", "Cirq", "streamlink", "beets",
        "privacyidea", "scrapy-splash",
    ],
    "Documentation": [
        "sphinx", "WeasyPrint", "babel", "filesystem_spec",
    ],
    "Other": [
        "mesa", "segmentation_models.pytorch", "Flexget",
        "icloud_photos_downloader",
    ],
}

# Invert for lookup
REPO_TO_DOMAIN = {}
for domain, repos in DOMAIN_MAP.items():
    for repo in repos:
        REPO_TO_DOMAIN[repo] = domain


def load_per_repo(path):
    with open(path) as f:
        data = json.load(f)
    return data["per_repo"]


def main():
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    graph_repos = load_per_repo(GRAPH_PATH)
    bm25_repos = load_per_repo(BM25_PATH)

    # Collect all repo names
    all_repos = sorted(set(list(graph_repos.keys()) + list(bm25_repos.keys())))

    # Build per-repo comparison list
    per_repo_rows = []
    for repo in all_repos:
        g = graph_repos.get(repo)
        b = bm25_repos.get(repo)
        g_r1 = g["hit@1"] if g else None
        b_r1 = b["hit@1"] if b else None
        g_count = g["count"] if g else 0
        b_count = b["count"] if b else 0
        count = max(g_count, b_count)
        domain = REPO_TO_DOMAIN.get(repo, "Other")

        delta = None
        if g_r1 is not None and b_r1 is not None:
            delta = g_r1 - b_r1

        per_repo_rows.append({
            "repo": repo,
            "domain": domain,
            "graph_r1": g_r1,
            "bm25_r1": b_r1,
            "delta": delta,
            "count": count,
        })

    # ---------- Domain aggregation ----------
    domain_stats = defaultdict(lambda: {
        "repos": [],
        "total_examples": 0,
        "graph_r1_vals": [],
        "bm25_r1_vals": [],
        "wins": 0,
        "losses": 0,
        "ties": 0,
    })

    for row in per_repo_rows:
        d = domain_stats[row["domain"]]
        d["repos"].append(row["repo"])
        d["total_examples"] += row["count"]
        if row["graph_r1"] is not None:
            d["graph_r1_vals"].append(row["graph_r1"])
        if row["bm25_r1"] is not None:
            d["bm25_r1_vals"].append(row["bm25_r1"])
        if row["delta"] is not None:
            if row["delta"] > 0.5:
                d["wins"] += 1
            elif row["delta"] < -0.5:
                d["losses"] += 1
            else:
                d["ties"] += 1

    # ---------- Print text table: domain summary ----------
    print("=" * 110)
    print("DOMAIN ANALYSIS: Graph-Hard (runB) vs BM25-Only (runA) -- R@1 (hit@1)")
    print("=" * 110)

    header = f"{'Domain':<25} {'#Repo':>5} {'#Ex':>5} {'Graph R@1':>10} {'BM25 R@1':>10} {'Delta':>8} {'Win':>4} {'Lose':>4} {'Tie':>4}"
    print(header)
    print("-" * 110)

    domain_order = [
        "Scientific Computing", "Data/ML Infrastructure", "Developer Tools",
        "Web/API", "Data Processing", "Infrastructure/DevOps",
        "Desktop/Media", "Documentation", "Other",
    ]

    domain_summary_rows = []
    for domain in domain_order:
        d = domain_stats[domain]
        n_repos = len(d["repos"])
        n_ex = d["total_examples"]
        mean_g = sum(d["graph_r1_vals"]) / len(d["graph_r1_vals"]) if d["graph_r1_vals"] else float("nan")
        mean_b = sum(d["bm25_r1_vals"]) / len(d["bm25_r1_vals"]) if d["bm25_r1_vals"] else float("nan")
        delta = mean_g - mean_b
        print(f"{domain:<25} {n_repos:>5} {n_ex:>5} {mean_g:>10.2f} {mean_b:>10.2f} {delta:>+8.2f} {d['wins']:>4} {d['losses']:>4} {d['ties']:>4}")
        domain_summary_rows.append((domain, n_repos, n_ex, mean_g, mean_b, delta, d["wins"], d["losses"], d["ties"]))

    # Overall
    all_g = [r["graph_r1"] for r in per_repo_rows if r["graph_r1"] is not None]
    all_b = [r["bm25_r1"] for r in per_repo_rows if r["bm25_r1"] is not None]
    total_ex = sum(r["count"] for r in per_repo_rows)
    total_wins = sum(d["wins"] for d in domain_stats.values())
    total_losses = sum(d["losses"] for d in domain_stats.values())
    total_ties = sum(d["ties"] for d in domain_stats.values())
    print("-" * 110)
    mean_g_all = sum(all_g) / len(all_g)
    mean_b_all = sum(all_b) / len(all_b)
    print(f"{'OVERALL':<25} {len(all_repos):>5} {total_ex:>5} {mean_g_all:>10.2f} {mean_b_all:>10.2f} {mean_g_all - mean_b_all:>+8.2f} {total_wins:>4} {total_losses:>4} {total_ties:>4}")
    print()

    # ---------- Print text table: per-repo sorted by delta ----------
    print("=" * 100)
    print("PER-REPO COMPARISON (sorted by delta, descending)")
    print("=" * 100)
    header2 = f"{'Repo':<35} {'Domain':<22} {'#Ex':>4} {'Graph':>7} {'BM25':>7} {'Delta':>8}"
    print(header2)
    print("-" * 100)

    sorted_rows = sorted(per_repo_rows, key=lambda r: r["delta"] if r["delta"] is not None else -999, reverse=True)
    for row in sorted_rows:
        g_str = f"{row['graph_r1']:.1f}" if row["graph_r1"] is not None else "N/A"
        b_str = f"{row['bm25_r1']:.1f}" if row["bm25_r1"] is not None else "N/A"
        d_str = f"{row['delta']:+.1f}" if row["delta"] is not None else "N/A"
        print(f"{row['repo']:<35} {row['domain']:<22} {row['count']:>4} {g_str:>7} {b_str:>7} {d_str:>8}")

    print()

    # ---------- LaTeX table: domain summary ----------
    print("=" * 110)
    print("LATEX TABLE: Domain Summary")
    print("=" * 110)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Domain-level R@1 comparison: Graph-Hard vs BM25-Only.}")
    print(r"\label{tab:domain_analysis}")
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begin{tabular}{l r r r r r r r r}")
    print(r"\toprule")
    print(r"Domain & \#Repos & \#Examples & Graph R@1 & BM25 R@1 & $\Delta$ & Win & Lose & Tie \\")
    print(r"\midrule")
    for (domain, n_repos, n_ex, mean_g, mean_b, delta, wins, losses, ties) in domain_summary_rows:
        delta_str = f"{delta:+.2f}"
        print(f"{domain} & {n_repos} & {n_ex} & {mean_g:.2f} & {mean_b:.2f} & {delta_str} & {wins} & {losses} & {ties} \\\\")
    print(r"\midrule")
    delta_all = mean_g_all - mean_b_all
    print(f"Overall & {len(all_repos)} & {total_ex} & {mean_g_all:.2f} & {mean_b_all:.2f} & {delta_all:+.2f} & {total_wins} & {total_losses} & {total_ties} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")
    print()

    # ---------- LaTeX table: per-repo ----------
    print("=" * 110)
    print("LATEX TABLE: Per-Repo Comparison (sorted by delta)")
    print("=" * 110)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Per-repo R@1 comparison: Graph-Hard vs BM25-Only (sorted by $\Delta$).}")
    print(r"\label{tab:per_repo_analysis}")
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begin{tabular}{l l r r r r}")
    print(r"\toprule")
    print(r"Repository & Domain & \#Ex & Graph R@1 & BM25 R@1 & $\Delta$ \\")
    print(r"\midrule")
    for row in sorted_rows:
        g_str = f"{row['graph_r1']:.1f}" if row["graph_r1"] is not None else "---"
        b_str = f"{row['bm25_r1']:.1f}" if row["bm25_r1"] is not None else "---"
        d_str = f"{row['delta']:+.1f}" if row["delta"] is not None else "---"
        repo_escaped = row["repo"].replace("_", r"\_")
        domain_short = row["domain"]
        print(f"{repo_escaped} & {domain_short} & {row['count']} & {g_str} & {b_str} & {d_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
