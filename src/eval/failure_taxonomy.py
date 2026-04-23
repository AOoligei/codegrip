"""
Failure Taxonomy: Categorize the 798 complete misses (H@10=0) into
distinct, publishable failure modes.

Categories derived from structural, textual, and data-level analysis:
1. Isolated Files: GT files with zero structural connections in the repo graph
2. Cross-Module Bugs: GT files in different directories from all predictions
3. Test-Only Misses: GT is exclusively test files that model never predicts
4. Long-Tail Repos: Repos with <5 training examples (model never learned patterns)
5. Multi-File Scatter: Issues requiring changes across 5+ files (model can't cover all)

Usage:
    python src/eval/failure_taxonomy.py \
        --predictions experiments/exp1_sft_only/eval_reranked/predictions.jsonl \
        --test_data data/grepo_text/grepo_test.jsonl \
        --train_data data/grepo_text/grepo_train.jsonl \
        --dep_graph_dir data/dep_graphs \
        --output docs/failure_taxonomy.md
"""

import json
import os
import re
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple


def load_predictions(path: str) -> Dict[Tuple[str, int], dict]:
    preds = {}
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            key = (p['repo'], p['issue_id'])
            preds[key] = p
    return preds


def build_repo_graph(train_data: List[dict], dep_graph_dir: str):
    """Build per-repo structural adjacency (co-change + import)."""
    # Co-change adjacency
    cochange_adj = defaultdict(lambda: defaultdict(set))
    for item in train_data:
        repo = item['repo']
        files = item.get('changed_py_files', [])
        for f in files:
            for other in files:
                if other != f:
                    cochange_adj[repo][f].add(other)

    # Import adjacency
    import_adj = defaultdict(lambda: defaultdict(set))
    if os.path.isdir(dep_graph_dir):
        for fname in os.listdir(dep_graph_dir):
            if not fname.endswith("_rels.json"):
                continue
            repo = fname.replace("_rels.json", "")
            with open(os.path.join(dep_graph_dir, fname)) as f:
                rels = json.load(f)
            for importer, imported_list in rels.get('file_imports', {}).items():
                for imported in imported_list:
                    if importer.endswith('.py') and imported.endswith('.py'):
                        import_adj[repo][importer].add(imported)
                        import_adj[repo][imported].add(importer)

    return cochange_adj, import_adj


def classify_issue(
    repo: str,
    gt_files: Set[str],
    predicted: List[str],
    cochange_adj: dict,
    import_adj: dict,
    repo_train_count: int,
    issue_text: str,
):
    """Classify a complete miss into failure mode categories.
    Returns list of applicable categories.
    """
    categories = []
    pred_set = set(predicted)
    pred_dirs = set(os.path.dirname(f) for f in predicted if f)

    # Category 1: Isolated Files — GT files have zero structural connections
    gt_total_connections = 0
    gt_isolated_count = 0
    for f in gt_files:
        cc = cochange_adj.get(repo, {}).get(f, set())
        imp = import_adj.get(repo, {}).get(f, set())
        total_conn = len(cc | imp)
        gt_total_connections += total_conn
        if total_conn == 0:
            gt_isolated_count += 1

    if gt_isolated_count == len(gt_files):
        categories.append('isolated_files')
    elif gt_isolated_count > 0:
        categories.append('partially_isolated')

    # Category 2: Cross-Module — GT files are in different directories from predictions
    gt_dirs = set(os.path.dirname(f) for f in gt_files)
    dir_overlap = gt_dirs & pred_dirs
    if len(dir_overlap) == 0 and len(predicted) > 0:
        categories.append('cross_module')

    # Category 3: Test-Only — all GT files are test files
    gt_test_count = sum(1 for f in gt_files
                       if 'test' in os.path.basename(f).lower()
                       or '/test/' in f or '/tests/' in f)
    if gt_test_count == len(gt_files):
        categories.append('test_only')
    elif gt_test_count > 0:
        categories.append('includes_tests')

    # Category 4: Long-Tail Repo — very few training examples
    if repo_train_count <= 5:
        categories.append('long_tail_repo')

    # Category 5: Multi-File Scatter — many GT files spread across repo
    if len(gt_files) >= 5:
        categories.append('multi_file_scatter')
    elif len(gt_files) >= 3:
        categories.append('moderate_scatter')

    # Category 6: Deep files (4+ directory levels)
    deep_count = sum(1 for f in gt_files if f.count('/') >= 4)
    if deep_count == len(gt_files):
        categories.append('deep_files_only')
    elif deep_count > 0:
        categories.append('includes_deep_files')

    # Category 7: Vague issue text (very short, no code references)
    code_refs = len(re.findall(r'`[^`]+`', issue_text))
    path_refs = len(re.findall(r'[\w./]+\.py', issue_text))
    words = len(issue_text.split())
    if words < 30 and code_refs == 0 and path_refs == 0:
        categories.append('vague_issue')

    if not categories:
        categories.append('uncategorized')

    return categories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--test_data', default='data/grepo_text/grepo_test.jsonl')
    parser.add_argument('--train_data', default='data/grepo_text/grepo_train.jsonl')
    parser.add_argument('--dep_graph_dir', default='data/dep_graphs')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    # Load data
    preds = load_predictions(args.predictions)
    test_items = {}
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            key = (item['repo'], item['issue_id'])
            test_items[key] = item

    with open(args.train_data) as f:
        train_data = [json.loads(l) for l in f]

    # Count training examples per repo
    repo_train_count = Counter(item['repo'] for item in train_data)

    # Build structural graph
    cochange_adj, import_adj = build_repo_graph(train_data, args.dep_graph_dir)

    # Find complete misses (H@10 = 0)
    complete_misses = []
    for key, test_item in test_items.items():
        gt = set(test_item.get('changed_py_files', []))
        if not gt:
            continue
        pred = preds.get(key, {}).get('predicted', [])
        h10 = len(gt & set(pred[:10])) / len(gt) * 100 if gt else 0
        if h10 == 0:
            complete_misses.append({
                'key': key,
                'repo': key[0],
                'issue_id': key[1],
                'gt_files': gt,
                'predicted': pred,
                'issue_text': test_item.get('issue_text', ''),
                'n_gt': len(gt),
            })

    print(f"Complete misses (H@10=0): {len(complete_misses)}")

    # Classify each miss
    category_counts = Counter()
    category_examples = defaultdict(list)
    per_issue_cats = []

    for miss in complete_misses:
        cats = classify_issue(
            repo=miss['repo'],
            gt_files=miss['gt_files'],
            predicted=miss['predicted'],
            cochange_adj=cochange_adj,
            import_adj=import_adj,
            repo_train_count=repo_train_count[miss['repo']],
            issue_text=miss['issue_text'],
        )
        per_issue_cats.append(cats)
        for cat in cats:
            category_counts[cat] += 1
            if len(category_examples[cat]) < 5:
                category_examples[cat].append(miss)

    # Assign PRIMARY category (most specific applicable)
    priority_order = [
        'isolated_files', 'long_tail_repo', 'cross_module', 'test_only',
        'multi_file_scatter', 'deep_files_only', 'vague_issue',
        'partially_isolated', 'includes_tests', 'moderate_scatter',
        'includes_deep_files', 'uncategorized',
    ]

    primary_counts = Counter()
    for cats in per_issue_cats:
        for cat in priority_order:
            if cat in cats:
                primary_counts[cat] += 1
                break

    # Generate report
    lines = []
    lines.append("# Failure Taxonomy: Complete Misses (H@10 = 0)\n")
    lines.append(f"**Total complete misses**: {len(complete_misses)} out of "
                f"{len(test_items)} test examples "
                f"({100 * len(complete_misses) / len(test_items):.1f}%)\n")

    # Primary category distribution
    lines.append("## Primary Failure Mode Distribution\n")
    lines.append("Each issue assigned to its most specific applicable category.\n")
    lines.append("| Failure Mode | Count | % | Description |")
    lines.append("|-------------|------:|---:|-------------|")

    descriptions = {
        'isolated_files': 'ALL GT files have zero co-change and import connections',
        'long_tail_repo': 'Repo has <= 5 training examples (insufficient learning)',
        'cross_module': 'GT files in different directories from all predictions',
        'test_only': 'All GT files are test files (model systematically misses tests)',
        'multi_file_scatter': 'Issue requires changes across 5+ files',
        'deep_files_only': 'All GT files are 4+ directories deep',
        'vague_issue': 'Issue text <30 words, no code/path references',
        'partially_isolated': 'Some GT files have zero structural connections',
        'includes_tests': 'Some GT files are test files',
        'moderate_scatter': 'Issue requires changes across 3-4 files',
        'includes_deep_files': 'Some GT files are 4+ directories deep',
        'uncategorized': 'No clear pattern identified',
    }

    for cat in priority_order:
        if primary_counts[cat] > 0:
            pct = 100 * primary_counts[cat] / len(complete_misses)
            desc = descriptions.get(cat, '')
            lines.append(f"| **{cat}** | {primary_counts[cat]} | {pct:.1f}% | {desc} |")

    # Multi-label distribution (issues can have multiple categories)
    lines.append("\n## Multi-Label Category Frequency\n")
    lines.append("Issues can belong to multiple categories simultaneously.\n")
    lines.append("| Category | Count | % |")
    lines.append("|----------|------:|---:|")
    for cat, count in category_counts.most_common():
        pct = 100 * count / len(complete_misses)
        lines.append(f"| {cat} | {count} | {pct:.1f}% |")

    # Category co-occurrence
    lines.append("\n## Category Co-Occurrence\n")
    co_occur = Counter()
    for cats in per_issue_cats:
        for i, a in enumerate(sorted(cats)):
            for b in sorted(cats)[i+1:]:
                co_occur[(a, b)] += 1

    lines.append("| Category A | Category B | Count |")
    lines.append("|-----------|-----------|------:|")
    for (a, b), count in co_occur.most_common(10):
        lines.append(f"| {a} | {b} | {count} |")

    # Detailed examples per major category
    lines.append("\n## Representative Examples\n")

    major_cats = ['isolated_files', 'cross_module', 'test_only', 'long_tail_repo',
                  'multi_file_scatter', 'deep_files_only']
    for cat in major_cats:
        if cat not in category_examples:
            continue
        lines.append(f"\n### {cat.replace('_', ' ').title()}\n")
        lines.append(f"*{descriptions.get(cat, '')}*\n")
        for ex in category_examples[cat][:3]:
            lines.append(f"**{ex['repo']} / #{ex['issue_id']}** ({ex['n_gt']} GT files)")
            for f in sorted(ex['gt_files'])[:5]:
                lines.append(f"  - `{f}`")
            if ex['n_gt'] > 5:
                lines.append(f"  - ... and {ex['n_gt'] - 5} more")
            lines.append(f"  - Issue: {ex['issue_text'][:150]}...")
            lines.append("")

    # Summary for paper
    lines.append("\n---\n")
    lines.append("## Summary for Paper Error Analysis\n")
    lines.append("The 798 complete misses decompose into four primary failure modes:\n")

    # Consolidate into 4 paper-ready categories
    paper_cats = {
        'Structurally Isolated': primary_counts.get('isolated_files', 0) + primary_counts.get('partially_isolated', 0),
        'Cross-Module Dependencies': primary_counts.get('cross_module', 0),
        'Test File Blindness': primary_counts.get('test_only', 0) + primary_counts.get('includes_tests', 0),
        'Data Scarcity / Multi-File': (primary_counts.get('long_tail_repo', 0) +
                                        primary_counts.get('multi_file_scatter', 0) +
                                        primary_counts.get('moderate_scatter', 0)),
    }
    remaining = len(complete_misses) - sum(paper_cats.values())
    paper_cats['Other'] = remaining

    for cat_name, count in paper_cats.items():
        pct = 100 * count / len(complete_misses)
        lines.append(f"- **{cat_name}**: {count} ({pct:.1f}%)")

    lines.append(f"\nThese categories suggest that improved structural coverage "
                f"(addressing isolated files and cross-module dependencies) and "
                f"explicit test-file prediction are the most promising directions "
                f"for reducing complete misses.")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(report)

    print(f"\nTaxonomy report saved to {args.output}")
    print(f"\nPaper-ready summary:")
    for cat_name, count in paper_cats.items():
        pct = 100 * count / len(complete_misses)
        print(f"  {cat_name:30s}: {count:4d} ({pct:5.1f}%)")


if __name__ == '__main__':
    main()
