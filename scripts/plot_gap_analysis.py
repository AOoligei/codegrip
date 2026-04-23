#!/usr/bin/env python3
"""Plot oracle fallacy gap across training checkpoints."""
import json
import glob
import os
import sys

def collect_results(exp_dir):
    graph_r = {}
    hybrid_r = {}
    bm25_r = {}
    
    for pool, store in [('graph', graph_r), ('hybrid', hybrid_r), ('bm25', bm25_r)]:
        for d in glob.glob(f'{exp_dir}/eval_*_{pool}'):
            s = os.path.join(d, 'summary.json')
            if not os.path.exists(s):
                continue
            h1 = json.load(open(s))['overall']['hit@1']
            ckpt = os.path.basename(d).replace('eval_', '').replace(f'_{pool}', '')
            if 'checkpoint' in ckpt:
                step = int(ckpt.split('-')[1])
            elif ckpt == 'best':
                step = -1
            elif ckpt == 'final':
                step = 99999
            else:
                continue
            store[step] = h1
    
    return graph_r, hybrid_r, bm25_r

def print_table(graph_r, hybrid_r, bm25_r):
    all_steps = sorted(set(list(graph_r.keys()) + list(hybrid_r.keys()) + list(bm25_r.keys())))
    
    print(f"{'step':<10s} {'graph':>8s} {'hybrid':>8s} {'bm25':>8s} {'gap(g-h)':>10s}")
    print("=" * 48)
    
    gaps = []
    for step in all_steps:
        label = f"best" if step == -1 else (f"final" if step == 99999 else f"step-{step}")
        g = graph_r.get(step)
        h = hybrid_r.get(step)
        b = bm25_r.get(step)
        
        gs = f"{g:.2f}%" if g else "-"
        hs = f"{h:.2f}%" if h else "-"
        bs = f"{b:.2f}%" if b else "-"
        
        if g and h:
            gap = g - h
            gaps.append((step, gap))
            marker = " <<< FALLACY" if gap > 3 else (" < mild" if gap > 1 else "")
            print(f"{label:<10s} {gs:>8s} {hs:>8s} {bs:>8s} {gap:+7.2f}pp{marker}")
        else:
            print(f"{label:<10s} {gs:>8s} {hs:>8s} {bs:>8s}")
    
    if gaps:
        import numpy as np
        gap_vals = [g for _, g in gaps]
        print(f"\n--- Summary ({len(gaps)} paired checkpoints) ---")
        print(f"Mean gap: {np.mean(gap_vals):+.2f}pp")
        print(f"Max gap:  {max(gap_vals):+.2f}pp")
        print(f"Min gap:  {min(gap_vals):+.2f}pp")
        print(f"Gap > 3pp: {sum(1 for g in gap_vals if g > 3)}/{len(gaps)}")
        print(f"Gap > 1pp: {sum(1 for g in gap_vals if g > 1)}/{len(gaps)}")
    
    return gaps

def plot_gap(gaps, output_path="paper/figures/gap_vs_step.pdf"):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plot")
        return
    
    steps = [s for s, _ in gaps if s >= 0 and s < 99999]
    gap_vals = [g for s, g in gaps if s >= 0 and s < 99999]
    
    if not steps:
        print("No checkpoint data to plot")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(steps, gap_vals, 'b-o', markersize=3, linewidth=1)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=7.01, color='red', linestyle=':', alpha=0.5, label='Old gap (7.01pp)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gap (graph - hybrid, pp)')
    ax.set_title('Oracle Fallacy Gap Across Training Checkpoints')
    ax.legend()
    ax.grid(alpha=0.3)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments/rankft_runB_graph_v2"
    graph_r, hybrid_r, bm25_r = collect_results(exp_dir)
    gaps = print_table(graph_r, hybrid_r, bm25_r)
    if gaps:
        plot_gap(gaps)
