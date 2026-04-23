"""Collect all main results into paper-ready tables with CI and 9-repo slice."""
import json, os
import numpy as np

np.random.seed(42)

GREPO_9 = {'astropy', 'dvc', 'ipython', 'pylint', 'scipy', 'sphinx', 'streamlink', 'xarray', 'geopandas'}
BASE = "/data/chenlibin/grepo_agent_experiments/v2_with_ci"

def load_summary(path):
    sf = os.path.join(path, "summary.json")
    if os.path.exists(sf):
        return json.load(open(sf))
    return None

def fmt_ci(d, metric="recall@1"):
    """Format metric with bootstrap CI."""
    ci = d.get("bootstrap_ci", {}).get(metric, {})
    val = d["overall"].get(metric.replace("recall", "hit"), d["overall"].get(metric, 0))
    if ci:
        return f"{val:.2f} [{ci['ci_lo']:.2f}, {ci['ci_hi']:.2f}]"
    return f"{val:.2f}"

def nine_repo_slice(d):
    """Compute 9-repo weighted average (weighted by n_examples, matching GREPO paper)."""
    pr = d.get("per_repo", {})
    matched = {r: v for r, v in pr.items() if r.lower() in GREPO_9}
    if not matched:
        return None
    total_n = sum(v.get('n_examples', 1) for v in matched.values())
    return sum(v['hit@1'] * v.get('n_examples', 1) for v in matched.values()) / total_n


# ============================================================
# Table 1: Main Baselines
# ============================================================
print("=" * 80)
print("TABLE 1: Main Baselines (graph-expanded pool, top-200)")
print("=" * 80)
print(f"{'Model':<25} {'Recall@1 [95% CI]':>25} {'Acc@1':>8} {'9-repo':>8}")
print("-" * 70)

for name, path in [
    ("Qwen2.5-7B (best)", f"{BASE}/qwen25_7b/eval_graph"),
    ("Qwen2.5-7B (final)", f"{BASE}/qwen25_7b/eval_graph_final"),
    ("Llama-3.1-8B", f"{BASE}/llama31_8b/eval_graph"),
    ("Qwen3-8B", f"{BASE}/qwen3_8b/eval_graph"),
    ("Code-Residual-7B", f"{BASE}/code_residual_7b/eval_graph"),
]:
    d = load_summary(path)
    if d:
        r1_ci = fmt_ci(d)
        acc1 = d["overall"].get("acc@1", "?")
        nine = nine_repo_slice(d)
        nine_s = f"{nine:.2f}" if nine else "?"
        print(f"{name:<25} {r1_ci:>25} {acc1:>7.2f}% {nine_s:>7}%")
    else:
        print(f"{name:<25} {'(not yet available)':>25}")

# ============================================================
# Table 2: Cross-LLM Perturbation
# ============================================================
print("\n" + "=" * 80)
print("TABLE 2: Cross-LLM Perturbation Results (Recall@1)")
print("=" * 80)

baselines = {}
perturbations = ['shuffle_filenames', 'shuffle_dirs', 'flatten_dirs',
                 'swap_leaf_dirs', 'remove_module_names', 'delexicalize']
models = {
    'Qwen2.5-7B': 'qwen25_7b',
    'Llama-3.1-8B': 'llama31_8b',
    'Qwen3-8B': 'qwen3_8b',
}

# Get baselines
for m_name, m_dir in models.items():
    d = load_summary(f"{BASE}/{m_dir}/eval_graph")
    if d:
        baselines[m_name] = d["overall"]["hit@1"]

print(f"{'Perturbation':<22}", end='')
for m in models:
    print(f" {m:>16}", end='')
print()
print("-" * 72)

print(f"{'baseline':<22}", end='')
for m in models:
    b = baselines.get(m)
    print(f" {b:>15.2f}%" if b else f" {'?':>16}", end='')
print()

for cond in perturbations:
    print(f"{cond:<22}", end='')
    for m_name, m_dir in models.items():
        d = load_summary(f"{BASE}/{m_dir}/eval_perturb_{cond}")
        if d:
            h1 = d["overall"]["hit@1"]
            b = baselines.get(m_name, h1)
            drop = (h1 / b - 1) * 100 if b else 0
            print(f" {h1:>6.2f} ({drop:>+4.0f}%)", end='')
        else:
            print(f" {'?':>16}", end='')
    print()

# ============================================================
# Table 3: Code-Residual Robustness
# ============================================================
print("\n" + "=" * 80)
print("TABLE 3: Code-Residual Model Robustness Under Path Perturbation")
print("=" * 80)

cr_baseline = None
d = load_summary(f"{BASE}/code_residual_7b/eval_graph")
if d:
    cr_baseline = d["overall"]["hit@1"]

print(f"{'Perturbation':<22} {'Path-only R@1':>14} {'Code-Residual R@1':>18}")
print("-" * 56)
print(f"{'baseline':<22} {baselines.get('Qwen2.5-7B', '?'):>13.2f}% {cr_baseline:>17.2f}%" if cr_baseline else "")

for cond in perturbations:
    po = load_summary(f"{BASE}/qwen25_7b/eval_perturb_{cond}")
    cr = load_summary(f"{BASE}/code_residual_7b/eval_perturb_{cond}")
    po_val = f"{po['overall']['hit@1']:>13.2f}%" if po else f"{'?':>14}"
    cr_val = f"{cr['overall']['hit@1']:>17.2f}%" if cr else f"{'?':>18}"
    print(f"{cond:<22} {po_val} {cr_val}")

print("\n(Run scripts/rerun_main_results.sh first if tables show '?')")
