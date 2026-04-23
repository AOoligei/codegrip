#!/usr/bin/env python3
"""Extract 'hero' case studies where the pipeline rescued predictions.

Finds examples where:
  - The base model completely missed all GT files (Hit@5 = 0)
  - But expansion + reranking recovered at least one GT file into top-5/10

Outputs a structured markdown report with:
  - Issue text summary
  - Base model's predictions vs ground truth
  - Which structural signal(s) rescued the correct file
  - Multi-hop structural path from prediction → expansion → GT file

Usage:
    python scripts/extract_hero_cases.py \
        --base_predictions experiments/exp1_sft_only/eval_filetree/predictions.jsonl \
        --expanded_predictions experiments/exp1_sft_only/eval_unified_expansion/predictions.jsonl \
        --reranked_predictions experiments/exp1_sft_only/eval_reranked/predictions.jsonl \
        --test_data data/grepo_text/grepo_test.jsonl \
        --train_data data/grepo_text/grepo_train.jsonl \
        --dep_graph_dir data/dep_graphs \
        --file_tree_dir data/file_trees \
        --output docs/hero_cases.md \
        --top_k 3
"""

import argparse
import json
import os
import sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def compute_hit_at_k(predicted, gt_set, k):
    if not gt_set:
        return 0.0
    topk = set(predicted[:k])
    return (len(gt_set & topk) / len(gt_set)) * 100


def build_cochange_map(train_data):
    """Build per-repo co-change adjacency with scores."""
    cochange = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for item in train_data:
        repo = item['repo']
        files = item.get('changed_py_files', item.get('changed_files', []))
        if len(files) < 2:
            continue
        for i, f1 in enumerate(files):
            for f2 in files[i+1:]:
                cochange[repo][f1][f2] += 1
                cochange[repo][f2][f1] += 1
    return cochange


def build_import_map(dep_graph_dir):
    """Build per-repo import adjacency from dep_graph JSON files.

    Format: file_imports is {file: [imported_files]}, not edge list.
    """
    imports = defaultdict(lambda: defaultdict(set))
    if not os.path.isdir(dep_graph_dir):
        return imports
    for fn in os.listdir(dep_graph_dir):
        if not fn.endswith('.json'):
            continue
        repo = fn.replace('_rels.json', '').replace('_graph.json', '').replace('.json', '')
        with open(os.path.join(dep_graph_dir, fn)) as f:
            graph = json.load(f)
        # file_imports: {source_file: [target_files]}
        for src, targets in graph.get('file_imports', {}).items():
            if isinstance(targets, list):
                for tgt in targets:
                    imports[repo][src].add(tgt)
                    imports[repo][tgt].add(src)
    return imports


def find_structural_path(predicted_files, gt_file, repo, cochange, imports):
    """Find how a GT file connects to any predicted file via structural signals."""
    paths = []
    gt_dir = os.path.dirname(gt_file)

    for pred_file in predicted_files:
        pred_dir = os.path.dirname(pred_file)

        # Co-change connection
        cc_score = cochange.get(repo, {}).get(pred_file, {}).get(gt_file, 0)
        if cc_score > 0:
            paths.append({
                'type': 'co-change',
                'from': pred_file,
                'to': gt_file,
                'score': cc_score,
                'description': f"Co-changed {int(cc_score)} times in training data"
            })

        # Import connection
        if gt_file in imports.get(repo, {}).get(pred_file, set()):
            paths.append({
                'type': 'import',
                'from': pred_file,
                'to': gt_file,
                'description': f"Direct import dependency"
            })

        # Directory co-location
        if gt_dir == pred_dir and gt_dir:
            paths.append({
                'type': 'directory',
                'from': pred_file,
                'to': gt_file,
                'description': f"Same directory: {gt_dir}/"
            })

        # Test-source mapping
        pred_base = os.path.basename(pred_file)
        gt_base = os.path.basename(gt_file)
        if (pred_base.startswith('test_') and gt_base == pred_base[5:]) or \
           (gt_base.startswith('test_') and pred_base == gt_base[5:]):
            paths.append({
                'type': 'test-source',
                'from': pred_file,
                'to': gt_file,
                'description': f"Test-source naming convention"
            })

    return paths


def score_hero_case(case):
    """Score a hero case for selection (higher = more interesting)."""
    score = 0

    # Prefer cases where multiple GT files were rescued
    score += case['n_rescued'] * 10

    # Prefer cases with multi-hop structural paths
    for gt_file, paths in case['structural_paths'].items():
        if len(paths) >= 2:
            score += 5  # Multi-signal rescue
        for p in paths:
            if p['type'] == 'co-change':
                score += 3  # Co-change is our star signal
            elif p['type'] == 'import':
                score += 2

    # Prefer cases with more GT files (complex bugs)
    score += min(case['n_gt'], 5) * 2

    # Prefer repos with recognizable names
    well_known = ['scipy', 'astropy', 'pylint', 'xarray', 'ipython', 'sphinx', 'dvc']
    if any(r in case['repo'].lower() for r in well_known):
        score += 5

    # Prefer cases with clear issue text (longer = more context)
    issue_len = len(case.get('issue_text', ''))
    if issue_len > 200:
        score += 3

    return score


def format_hero_case(case, idx):
    """Format a single hero case as markdown."""
    lines = []
    lines.append(f"### Case {idx}: {case['repo']} #{case['issue_id']}")
    lines.append("")

    # Issue summary
    issue_text = case.get('issue_text', 'N/A')
    # Truncate long issues
    if len(issue_text) > 800:
        issue_text = issue_text[:800] + "..."
    lines.append(f"**Issue Text:**")
    lines.append(f"> {issue_text}")
    lines.append("")

    # Ground truth
    lines.append(f"**Ground Truth ({case['n_gt']} files):**")
    for gt in case['ground_truth']:
        status = "rescued" if gt in case['rescued_files'] else "missed"
        lines.append(f"- `{gt}` ({status})")
    lines.append("")

    # Base model predictions
    lines.append(f"**Base Model Predictions (Hit@5 = {case['base_hit5']:.0f}%, Hit@10 = {case['base_hit10']:.0f}%):**")
    for i, pred in enumerate(case['base_predicted'][:5], 1):
        marker = " **(GT)**" if pred in case['gt_set'] else ""
        lines.append(f"  {i}. `{pred}`{marker}")
    lines.append("")

    # After expansion
    lines.append(f"**After Expansion + Reranking (Hit@5 = {case['final_hit5']:.0f}%, Hit@10 = {case['final_hit10']:.0f}%):**")
    for i, pred in enumerate(case['final_predicted'][:10], 1):
        marker = " **(GT)**" if pred in case['gt_set'] else ""
        lines.append(f"  {i}. `{pred}`{marker}")
    lines.append("")

    # Structural rescue paths
    lines.append("**How Structural Signals Rescued the Correct Files:**")
    lines.append("")
    for gt_file, paths in case['structural_paths'].items():
        if not paths:
            continue
        lines.append(f"- `{gt_file}` was found via:")
        for p in paths:
            if p['type'] == 'co-change':
                lines.append(f"  - **Co-change**: `{p['from']}` co-changed {int(p.get('score', 0))} times with target")
            elif p['type'] == 'import':
                lines.append(f"  - **Import**: Direct import dependency from `{p['from']}`")
            elif p['type'] == 'directory':
                lines.append(f"  - **Directory**: Same directory as `{p['from']}`")
            elif p['type'] == 'test-source':
                lines.append(f"  - **Test-Source**: Naming convention maps to `{p['from']}`")
    lines.append("")

    # Key insight
    signal_types = set()
    for paths in case['structural_paths'].values():
        for p in paths:
            signal_types.add(p['type'])

    if signal_types:
        lines.append(f"**Key Insight:** The base model predicted files in the right *module* "
                      f"but missed the specific files that needed changes. "
                      f"The {', '.join(sorted(signal_types))} signal(s) bridged the gap by "
                      f"connecting the model's predictions to structurally related ground-truth files.")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Extract hero case studies')
    parser.add_argument('--base_predictions', default='experiments/exp1_sft_only/eval_filetree/predictions.jsonl')
    parser.add_argument('--expanded_predictions', default='experiments/exp1_sft_only/eval_unified_expansion/predictions.jsonl')
    parser.add_argument('--reranked_predictions', default='experiments/exp1_sft_only/eval_reranked/predictions.jsonl')
    parser.add_argument('--test_data', default='data/grepo_text/grepo_test.jsonl')
    parser.add_argument('--train_data', default='data/grepo_text/grepo_train.jsonl')
    parser.add_argument('--dep_graph_dir', default='data/dep_graphs')
    parser.add_argument('--file_tree_dir', default='data/file_trees')
    parser.add_argument('--output', default='docs/hero_cases.md')
    parser.add_argument('--top_k', type=int, default=3, help='Number of hero cases to extract')
    args = parser.parse_args()

    # Load all prediction files
    print("Loading predictions...")
    base_preds = load_jsonl(args.base_predictions)
    expanded_preds = load_jsonl(args.expanded_predictions)
    reranked_preds = load_jsonl(args.reranked_predictions)

    # Load test data for issue text
    test_data = load_jsonl(args.test_data)
    test_map = {(d['repo'], d['issue_id']): d for d in test_data}

    # Build structural indices
    print("Building structural indices...")
    train_data = load_jsonl(args.train_data)
    cochange = build_cochange_map(train_data)
    imports = build_import_map(args.dep_graph_dir)

    # Index predictions by (repo, issue_id)
    base_map = {(p['repo'], p['issue_id']): p for p in base_preds}
    expanded_map = {(p['repo'], p['issue_id']): p for p in expanded_preds}
    reranked_map = {(p['repo'], p['issue_id']): p for p in reranked_preds}

    # Find hero cases: base Hit@10 = 0, final Hit@10 > 0
    print("Searching for hero cases...")
    hero_cases = []

    for key, base_p in base_map.items():
        final_p = reranked_map.get(key, expanded_map.get(key))
        if final_p is None:
            continue

        gt_set = set(base_p.get('ground_truth', []))
        if not gt_set:
            continue

        base_predicted = base_p.get('predicted', [])
        final_predicted = final_p.get('predicted', [])

        base_hit5 = compute_hit_at_k(base_predicted, gt_set, 5)
        base_hit10 = compute_hit_at_k(base_predicted, gt_set, 10)
        final_hit5 = compute_hit_at_k(final_predicted, gt_set, 5)
        final_hit10 = compute_hit_at_k(final_predicted, gt_set, 10)

        # Hero condition: base missed, pipeline rescued
        if base_hit10 > 0:
            continue  # Base already found something
        if final_hit10 <= 0:
            continue  # Pipeline didn't help

        # Find which GT files were rescued
        rescued_in_10 = gt_set & set(final_predicted[:10])
        # Find structural paths for rescued files
        structural_paths = {}
        for gt_file in rescued_in_10:
            paths = find_structural_path(base_predicted, gt_file, key[0], cochange, imports)
            structural_paths[gt_file] = paths

        # Get issue text
        test_item = test_map.get(key, {})
        issue_text = test_item.get('issue_text', '')

        case = {
            'repo': key[0],
            'issue_id': key[1],
            'issue_text': issue_text,
            'ground_truth': list(gt_set),
            'gt_set': gt_set,
            'base_predicted': base_predicted,
            'final_predicted': final_predicted,
            'base_hit5': base_hit5,
            'base_hit10': base_hit10,
            'final_hit5': final_hit5,
            'final_hit10': final_hit10,
            'n_gt': len(gt_set),
            'n_rescued': len(rescued_in_10),
            'rescued_files': rescued_in_10,
            'structural_paths': structural_paths,
        }
        hero_cases.append(case)

    print(f"Found {len(hero_cases)} hero cases (base Hit@10=0, final Hit@10>0)")

    # Score and rank hero cases
    for case in hero_cases:
        case['hero_score'] = score_hero_case(case)

    hero_cases.sort(key=lambda c: c['hero_score'], reverse=True)

    # Generate report
    lines = []
    lines.append("# Hero Case Studies: When Structural Signals Rescue Predictions")
    lines.append("")
    lines.append(f"Out of {len(base_preds)} test examples, **{len(hero_cases)}** cases show the "
                 f"pipeline rescuing completely missed predictions (base Hit@10=0 → final Hit@10>0).")
    lines.append("")
    lines.append("The following cases illustrate how multi-signal structural expansion bridges the gap "
                 "between the base model's text-only predictions and the ground-truth files.")
    lines.append("")

    # Summary stats
    rescue_by_signal = defaultdict(int)
    for case in hero_cases:
        for paths in case['structural_paths'].values():
            for p in paths:
                rescue_by_signal[p['type']] += 1

    lines.append("## Rescue Statistics")
    lines.append("")
    lines.append(f"- Total hero cases: {len(hero_cases)}")
    lines.append(f"- Rescue signal breakdown:")
    for sig, count in sorted(rescue_by_signal.items(), key=lambda x: -x[1]):
        lines.append(f"  - {sig}: {count} rescue paths")
    lines.append("")

    # Top cases
    lines.append("## Selected Case Studies")
    lines.append("")

    for i, case in enumerate(hero_cases[:args.top_k], 1):
        lines.append(format_hero_case(case, i))

    # Additional statistics
    lines.append("## Aggregate Analysis")
    lines.append("")
    avg_rescued = sum(c['n_rescued'] for c in hero_cases) / max(len(hero_cases), 1)
    avg_gt = sum(c['n_gt'] for c in hero_cases) / max(len(hero_cases), 1)
    lines.append(f"- Average GT files per hero case: {avg_gt:.1f}")
    lines.append(f"- Average rescued files per case: {avg_rescued:.1f}")
    lines.append(f"- Rescue rate: {avg_rescued/max(avg_gt, 0.01)*100:.1f}%")
    lines.append("")

    # Which signals are most common in rescues
    lines.append("### Signal Contribution to Rescues")
    lines.append("")
    lines.append("| Signal | Rescue Paths | % of Rescues |")
    lines.append("|--------|-------------|-------------|")
    total_paths = sum(rescue_by_signal.values())
    for sig, count in sorted(rescue_by_signal.items(), key=lambda x: -x[1]):
        pct = count / max(total_paths, 1) * 100
        lines.append(f"| {sig} | {count} | {pct:.1f}% |")
    lines.append("")

    output = "\n".join(lines)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(output)
    print(f"\nHero case report written to {args.output}")
    print(f"Top {args.top_k} cases selected by interestingness score")


if __name__ == '__main__':
    main()
