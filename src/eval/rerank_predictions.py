"""
Issue-text-aware reranking for expanded predictions.

Extracts file path mentions and keywords from bug report text,
then promotes expansion candidates with text relevance into
higher positions (top-5, top-10).

This is a post-processing step applied AFTER multi_signal_expansion.

Usage:
    python src/eval/rerank_predictions.py \
        --predictions experiments/exp1_sft_only/eval_unified_expansion/predictions.jsonl \
        --test_data data/grepo_text/grepo_test.jsonl \
        --output experiments/exp1_sft_only/eval_reranked/predictions.jsonl \
        --promote_top5 1 --promote_top10 2 --threshold 0.3
"""

import json
import argparse
import os
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def extract_file_mentions(issue_text: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Extract file path mentions, partial names, and keywords from issue text.

    Returns:
        - full_paths: complete file paths (e.g., "src/foo/bar.py")
        - partial_names: filename without directory (e.g., "bar.py")
        - keywords: module/class/function names mentioned in text
    """
    full_paths = set()
    partial_names = set()
    keywords = set()

    # Split into title and body for weighted extraction
    lines = issue_text.split('\n')
    title = lines[0] if lines else ''
    body = '\n'.join(lines[1:]) if len(lines) > 1 else ''

    # 1. Full file paths (containing / and ending in .py)
    for m in re.finditer(r'[\w./\\-]+\.py\b', issue_text):
        path = m.group(0).lstrip('./')
        if '/' in path:
            full_paths.add(path)
        partial_names.add(os.path.basename(path))

    # 2. Module-style references (foo.bar.baz -> foo/bar/baz.py or foo/bar.py)
    for m in re.finditer(r'`([\w.]+)`', issue_text):
        name = m.group(1)
        parts = name.split('.')
        if len(parts) >= 2:
            path_py = '/'.join(parts) + '.py'
            full_paths.add(path_py)
            if len(parts) >= 3:
                path_py2 = '/'.join(parts[:-1]) + '.py'
                full_paths.add(path_py2)
            for part in parts:
                if len(part) > 2:
                    keywords.add(part.lower())
        else:
            if len(name) > 2:
                keywords.add(name.lower())

    # 3. Class/function names from backtick code (including compound expressions)
    for m in re.finditer(r'`([^`]+)`', issue_text):
        code = m.group(1).strip()
        # Extract identifiers from code snippets
        for ident in re.findall(r'\b([a-zA-Z_]\w+)\b', code):
            if len(ident) > 2 and not ident.isupper() and ident not in ('the', 'and', 'for', 'not', 'with'):
                keywords.add(ident.lower())

    # 4. Title keywords (strongest signal - titles are very informative)
    # CamelCase identifiers
    for m in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z]*)+)\b', title):
        keywords.add(m.group(1).lower())
    # snake_case identifiers
    for m in re.finditer(r'\b([a-z]\w*_\w+)\b', title):
        keywords.add(m.group(1).lower())
    # Single meaningful words from title (nouns that could be module/file names)
    for m in re.finditer(r'\b([a-z]{3,})\b', title.lower()):
        word = m.group(1)
        # Skip common English words that aren't likely file/module names
        stopwords = {
            'the', 'and', 'for', 'not', 'with', 'this', 'that', 'from',
            'have', 'has', 'had', 'are', 'was', 'were', 'been', 'being',
            'can', 'could', 'should', 'would', 'may', 'might', 'will',
            'fix', 'bug', 'add', 'new', 'use', 'when', 'after', 'before',
            'into', 'about', 'than', 'then', 'also', 'just', 'only',
            'some', 'all', 'any', 'each', 'every', 'both', 'few',
            'more', 'most', 'other', 'such', 'too', 'very', 'how',
            'what', 'which', 'who', 'whom', 'why', 'where', 'while',
            'does', 'don', 'did', 'doing', 'make', 'made', 'get',
            'got', 'set', 'let', 'put', 'take', 'give', 'keep',
            'still', 'already', 'yet', 'now', 'here', 'there',
            'update', 'change', 'remove', 'delete', 'create', 'move',
            'allow', 'ensure', 'handle', 'support', 'improve',
            'instead', 'error', 'issue', 'problem', 'file', 'files',
            'method', 'function', 'class', 'module', 'package',
            'return', 'value', 'type', 'name', 'path', 'test',
            'using', 'calls', 'case', 'properly', 'correctly',
            'behavior', 'behaviour', 'output', 'input', 'default',
            'prepare', 'release', 'version', 'minor', 'major',
        }
        if word not in stopwords:
            keywords.add(word)

    # 5. Title: directory/component mentions (e.g., "cli:", "io:", "GUI")
    title_prefix = re.match(r'^(\w+):', title)
    if title_prefix:
        prefix = title_prefix.group(1).lower()
        if len(prefix) > 1:
            keywords.add(prefix)

    return full_paths, partial_names, keywords


def score_candidate(
    filepath: str,
    full_paths: Set[str],
    partial_names: Set[str],
    keywords: Set[str],
) -> float:
    """Score a candidate file based on its relevance to issue text mentions."""
    score = 0.0
    basename = os.path.basename(filepath)
    basename_lower = basename.lower()
    filepath_lower = filepath.lower()
    # Decompose path into meaningful tokens
    stem = basename_lower.replace('.py', '')
    path_components = filepath_lower.replace('.py', '').split('/')
    # All tokens from path (split on / and _)
    path_tokens = set()
    for comp in path_components:
        path_tokens.add(comp)
        for part in comp.split('_'):
            if part:
                path_tokens.add(part)

    # Direct full path match (strongest signal)
    for fp in full_paths:
        if filepath.endswith(fp) or fp.endswith(filepath):
            score += 2.0
            break
        if fp in filepath or filepath in fp:
            score += 1.0

    # Basename match
    for pn in partial_names:
        if basename == pn:
            score += 1.5
            break
        if pn in basename:
            score += 0.5

    # Keyword match against path components
    matched_keywords = 0
    for kw in keywords:
        if kw == stem:
            # Exact stem match (e.g., keyword "probe" matches "probe.py")
            score += 0.5
            matched_keywords += 1
        elif kw in path_tokens:
            # Token match (e.g., keyword "cli" matches "cli/probe.py")
            score += 0.3
            matched_keywords += 1
        elif kw in filepath_lower:
            # Substring match
            score += 0.15
            matched_keywords += 1

    # Bonus for multiple keyword matches (convergent evidence)
    if matched_keywords >= 2:
        score *= 1.3

    return score


def rerank_predictions(
    predictions_path: str,
    test_data_path: str,
    output_path: str,
    promote_top5: int = 1,
    promote_top10: int = 2,
    threshold: float = 0.3,
) -> dict:
    """
    Rerank expanded predictions using issue text relevance.

    For each prediction:
    1. Extract file mentions from the original issue text
    2. Score expansion candidates by text relevance
    3. Promote high-scoring candidates into top-5 and top-10 positions
    4. Keep position-1 (original top prediction) fixed

    Args:
        predictions_path: Path to expanded predictions JSONL
        test_data_path: Path to test data JSONL (for issue_text)
        output_path: Path to write reranked predictions JSONL
        promote_top5: Max candidates to promote into top-5
        promote_top10: Max additional candidates to promote into top-10
        threshold: Minimum text relevance score to promote
    """
    # Load issue texts
    issue_texts = {}
    with open(test_data_path) as f:
        for line in f:
            item = json.loads(line)
            key = (item['repo'], str(item['issue_id']))
            issue_texts[key] = item.get('issue_text', '')

    # Load predictions
    preds = []
    with open(predictions_path) as f:
        for line in f:
            preds.append(json.loads(line))

    reranked_preds = []
    rerank_stats = {'promoted_5': 0, 'promoted_10': 0, 'total': 0, 'with_mentions': 0}

    for p in preds:
        repo = p['repo']
        issue_id = str(p.get('issue_id', ''))
        predicted = list(p['predicted'])
        original = list(p.get('predicted_original', predicted[:1]))
        original_count = len(original)

        rerank_stats['total'] += 1

        # Get issue text
        issue_text = issue_texts.get((repo, issue_id), '')
        if not issue_text:
            # No issue text available, keep original order
            reranked_preds.append(p)
            continue

        # Extract mentions
        full_paths, partial_names, keywords = extract_file_mentions(issue_text)

        if not full_paths and not partial_names and not keywords:
            reranked_preds.append(p)
            continue

        rerank_stats['with_mentions'] += 1

        # Score expansion candidates (files beyond original predictions)
        expansion_files = predicted[original_count:]
        if not expansion_files:
            reranked_preds.append(p)
            continue

        scored = []
        for f in expansion_files:
            s = score_candidate(f, full_paths, partial_names, keywords)
            scored.append((f, s))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Promote high-scoring candidates
        new_predicted = list(predicted)  # Start with original order
        promoted_files = set()

        # Promote into top-5 (positions 2-5, keep position 1 fixed)
        promoted_5 = 0
        for fname, s in scored:
            if promoted_5 >= promote_top5:
                break
            if s >= threshold and fname in new_predicted[original_count:]:
                # Remove from current position and insert after original preds
                new_predicted.remove(fname)
                insert_pos = min(original_count + promoted_5, 4)  # positions 2-5
                new_predicted.insert(insert_pos, fname)
                promoted_files.add(fname)
                promoted_5 += 1

        # Promote into top-10 (positions 6-10)
        promoted_10 = 0
        threshold_10 = threshold * 0.7  # Lower threshold for top-10
        for fname, s in scored:
            if promoted_10 >= promote_top10:
                break
            if fname in promoted_files:
                continue
            if s >= threshold_10 and fname in new_predicted[5:]:
                new_predicted.remove(fname)
                insert_pos = min(5 + promoted_10, 9)
                new_predicted.insert(insert_pos, fname)
                promoted_files.add(fname)
                promoted_10 += 1

        rerank_stats['promoted_5'] += promoted_5
        rerank_stats['promoted_10'] += promoted_10

        # Recompute metrics
        gt = set(p.get('ground_truth', p.get('changed_py_files', [])))
        metrics = {}
        for k in [1, 3, 5, 10, 20]:
            topk = set(new_predicted[:k])
            hits = len(gt & topk)
            metrics[f'hit@{k}'] = (hits / len(gt)) * 100 if gt else 0.0

        new_p = dict(p)
        new_p['predicted'] = new_predicted
        new_p['metrics'] = metrics
        reranked_preds.append(new_p)

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for rp in reranked_preds:
            f.write(json.dumps(rp) + '\n')

    # Compute summary
    summary = compute_summary(reranked_preds)
    summary_path = output_path.replace('predictions.jsonl', 'summary.json')
    summary['rerank_stats'] = rerank_stats
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    o = summary['overall']
    print(f"Reranking stats: {rerank_stats['promoted_5']} promoted to top-5, "
          f"{rerank_stats['promoted_10']} to top-10, "
          f"{rerank_stats['with_mentions']}/{rerank_stats['total']} had text mentions")
    print(f"Hit@1={o['hit@1']:.2f}% Hit@3={o['hit@3']:.2f}% Hit@5={o['hit@5']:.2f}% "
          f"Hit@10={o['hit@10']:.2f}% Hit@20={o['hit@20']:.2f}%")

    return summary


def compute_summary(preds: List[dict]) -> dict:
    repo_metrics = defaultdict(lambda: defaultdict(list))
    overall = defaultdict(list)
    for p in preds:
        repo = p['repo']
        for k, v in p['metrics'].items():
            repo_metrics[repo][k].append(v)
            overall[k].append(v)

    summary = {
        'overall': {k: sum(v) / len(v) for k, v in overall.items()},
        'per_repo': {}
    }
    for repo, metrics in repo_metrics.items():
        summary['per_repo'][repo] = {}
        for k, v in metrics.items():
            summary['per_repo'][repo][k] = sum(v) / len(v)
        summary['per_repo'][repo]['count'] = len(repo_metrics[repo].get('hit@1', []))

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True,
                        help='Path to expanded predictions JSONL')
    parser.add_argument('--test_data', required=True,
                        help='Path to test data JSONL (for issue text)')
    parser.add_argument('--output', required=True,
                        help='Output path for reranked predictions')
    parser.add_argument('--promote_top5', type=int, default=1,
                        help='Max candidates to promote into top-5')
    parser.add_argument('--promote_top10', type=int, default=2,
                        help='Max additional candidates to promote into top-10')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Min text relevance score for promotion')
    args = parser.parse_args()

    rerank_predictions(
        args.predictions, args.test_data, args.output,
        promote_top5=args.promote_top5,
        promote_top10=args.promote_top10,
        threshold=args.threshold,
    )


if __name__ == '__main__':
    main()
