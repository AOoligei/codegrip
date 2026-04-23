"""
Iterative Expansion-Reranking: multi-round expand-rerank loop.

Instead of a single expand → rerank, do:
  Round 1: SFT predictions → graph expand → BM25 merge → rerank → top-K
  Round 2: take top-K → graph expand from top-K → merge with round-1 pool → rerank again

This finds 2-hop graph neighbors that aren't reachable in a single expansion round.

This script simulates round 2 using the existing predictions (round 1 output).
For each example:
1. Take top-K reranked files (from round 1)
2. Expand these through the graph to find new candidates
3. The new candidates don't have reranker scores, so we use graph-based heuristic scores
4. Measure how often 2-hop neighbors contain GT files not in the original pool
"""

import json
import os
from collections import defaultdict
from typing import Dict, Set


def build_cochange_index(train_data_path: str, min_cochange: int = 1) -> Dict:
    repo_cochanges = defaultdict(lambda: defaultdict(int))
    repo_file_count = defaultdict(lambda: defaultdict(int))

    with open(train_data_path) as f:
        for line in f:
            item = json.loads(line)
            if item.get('split') != 'train':
                continue
            repo = item['repo']
            files = item.get('changed_py_files', [])
            if not files:
                files = [ff for ff in item.get('changed_files', []) if ff.endswith('.py')]
            for f_item in files:
                repo_file_count[repo][f_item] += 1
            for i, fa in enumerate(files):
                for j, fb in enumerate(files):
                    if i != j:
                        repo_cochanges[repo][(fa, fb)] += 1

    index = {}
    for repo in repo_cochanges:
        index[repo] = defaultdict(dict)
        for (fa, fb), count in repo_cochanges[repo].items():
            if count >= min_cochange:
                score = count / max(repo_file_count[repo][fa], 1)
                index[repo][fa][fb] = score
    return index


def build_import_index(dep_graph_dir: str) -> Dict[str, Dict[str, Set[str]]]:
    index = {}
    if not os.path.isdir(dep_graph_dir):
        return index
    for fname in os.listdir(dep_graph_dir):
        if not fname.endswith('_rels.json'):
            continue
        repo = fname.replace('_rels.json', '')
        with open(os.path.join(dep_graph_dir, fname)) as f:
            rels = json.load(f)
        neighbors = defaultdict(set)
        for src, targets in rels.get('file_imports', {}).items():
            for tgt in targets:
                neighbors[src].add(tgt)
                neighbors[tgt].add(src)
        for src_func, callees in rels.get('call_graph', {}).items():
            src_file = src_func.split(':')[0] if ':' in src_func else src_func
            for callee in callees:
                tgt_file = callee.split(':')[0] if ':' in callee else callee
                if src_file != tgt_file:
                    neighbors[src_file].add(tgt_file)
                    neighbors[tgt_file].add(src_file)
        index[repo] = dict(neighbors)
    return index


def expand_from_seeds(seeds, repo, cc_index, imp_index, existing_pool,
                       max_cc=10, max_imp=10, min_cc_score=0.05):
    """Expand from seed files, returning new candidates not in existing_pool."""
    new_candidates = {}
    repo_cc = cc_index.get(repo, {})
    repo_imp = imp_index.get(repo, {})

    for seed in seeds:
        # Co-change neighbors
        cc_neighbors = sorted(repo_cc.get(seed, {}).items(), key=lambda x: -x[1])
        for neighbor, score in cc_neighbors[:max_cc]:
            if neighbor not in existing_pool and score >= min_cc_score:
                if neighbor not in new_candidates or new_candidates[neighbor] < score:
                    new_candidates[neighbor] = score

        # Import neighbors
        for neighbor in repo_imp.get(seed, set()):
            if neighbor not in existing_pool:
                if neighbor not in new_candidates:
                    new_candidates[neighbor] = 0.3  # import weight

    return new_candidates


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--train_data', default='data/grepo_text/grepo_train.jsonl')
    parser.add_argument('--dep_graph_dir', default='data/dep_graphs')
    parser.add_argument('--test_data', default='data/grepo_text/grepo_test.jsonl',
                        help='Test data to get full GT file lists')
    args = parser.parse_args()

    # Load predictions
    print("Loading predictions...")
    predictions = []
    with open(args.predictions) as f:
        for line in f:
            predictions.append(json.loads(line))
    print(f"  {len(predictions)} examples")

    # Load full test data for GT
    test_gt = {}
    if os.path.exists(args.test_data):
        with open(args.test_data) as f:
            for line in f:
                item = json.loads(line)
                key = (item['repo'], item['issue_id'])
                files = item.get('changed_py_files', [])
                if not files:
                    files = [ff for ff in item.get('changed_files', []) if ff.endswith('.py')]
                test_gt[key] = set(files)

    # Build graphs
    print("Building co-change index...")
    cc_index = build_cochange_index(args.train_data)
    print(f"  {len(cc_index)} repos")

    print("Building import index...")
    imp_index = build_import_index(args.dep_graph_dir)
    print(f"  {len(imp_index)} repos")

    # Analyze iterative expansion potential
    print("\n=== Iterative Expansion Analysis ===\n")

    for top_k in [5, 10, 20]:
        total = 0
        gt_only_in_2hop = 0
        gt_in_2hop = 0
        gt_in_1hop = 0
        new_candidates_added = 0
        gt_not_in_pool = 0

        for pred in predictions:
            repo = pred['repo']
            predicted = pred['predicted']
            gt = set(pred['ground_truth'])
            pool = set(predicted)

            # Round 1 result
            gt_in_pool = gt & pool
            gt_missing = gt - pool

            # Round 2: expand from top-K
            seeds = predicted[:top_k]
            new_cands = expand_from_seeds(seeds, repo, cc_index, imp_index, pool)

            new_cands_set = set(new_cands.keys())
            gt_found_in_2hop = gt_missing & new_cands_set
            gt_found_in_1or2 = gt_in_pool | gt_found_in_2hop

            total += len(gt)
            gt_in_1hop += len(gt_in_pool)
            gt_in_2hop += len(gt_found_in_1or2)
            gt_only_in_2hop += len(gt_found_in_2hop)
            gt_not_in_pool += len(gt_missing)
            new_candidates_added += len(new_cands)

        n = len(predictions)
        print(f"Top-K={top_k} seeds:")
        print(f"  GT files: {total} total, {gt_in_1hop} in round-1 pool ({gt_in_1hop/total*100:.1f}%)")
        print(f"  GT missing from round-1: {gt_not_in_pool}")
        print(f"  GT found by 2-hop expansion: {gt_only_in_2hop} ({gt_only_in_2hop/max(gt_not_in_pool,1)*100:.1f}% of missing)")
        print(f"  Coverage: round-1={gt_in_1hop/total*100:.1f}%, round-1+2={gt_in_2hop/total*100:.1f}%")
        print(f"  Avg new candidates per example: {new_candidates_added/n:.1f}")
        print()

    # Detailed analysis: how many examples gain new GT files from 2-hop?
    print("=== Per-example analysis (top-K=10) ===\n")
    examples_gaining = 0
    examples_gaining_all_missing = 0
    rank_1_in_2hop = 0

    for pred in predictions:
        repo = pred['repo']
        predicted = pred['predicted']
        gt = set(pred['ground_truth'])
        pool = set(predicted)
        gt_missing = gt - pool

        if not gt_missing:
            continue

        seeds = predicted[:10]
        new_cands = expand_from_seeds(seeds, repo, cc_index, imp_index, pool)
        new_cands_set = set(new_cands.keys())
        gt_found = gt_missing & new_cands_set

        if gt_found:
            examples_gaining += 1
            if gt_found == gt_missing:
                examples_gaining_all_missing += 1

        # Check if the GT file that's rank-1 in original is in 2-hop
        if predicted and predicted[0] not in gt:
            # Current rank-1 is wrong; check if any GT is in 2-hop
            if gt_missing & new_cands_set:
                rank_1_in_2hop += 1

    n_with_missing = sum(1 for p in predictions if set(p['ground_truth']) - set(p['predicted']))
    print(f"Examples with GT missing from round-1 pool: {n_with_missing}")
    print(f"Examples gaining GT from 2-hop: {examples_gaining} ({examples_gaining/max(n_with_missing,1)*100:.1f}%)")
    print(f"Examples recovering ALL missing GT: {examples_gaining_all_missing}")
    print(f"Examples where rank-1 wrong AND GT in 2-hop: {rank_1_in_2hop}")


if __name__ == '__main__':
    main()
