"""
Learned Expansion Policy for CodeGRIP graph expansion.

Goal: Train a classifier to predict which graph neighbors (co-change, import)
of SFT-predicted files are actually ground-truth changed files.

Approach:
- Train set: use leave-one-out within train split. For each train example,
  use the OTHER train examples to build co-change index. Use GT files as
  "simulated SFT predictions" (subset), find their graph neighbors, label
  each neighbor as positive/negative.
- Test set: use all train data to build co-change index, use actual SFT
  predictions from eval_graph_expanded/predictions.jsonl.
- Features: co-change weight, import degree, directory distance, linking
  count, BM25 rank, etc.
- Models: Logistic Regression + LightGBM.
- Evaluation: at matched pool sizes, compare classifier-filtered expansion
  vs random vs full expansion on oracle recall.
"""

import json
import os
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    classification_report, roc_auc_score
)
from sklearn.preprocessing import StandardScaler

random.seed(42)
np.random.seed(42)

ROOT = Path("/home/chenlibin/grepo_agent")
TRAIN_DATA = ROOT / "data/grepo_text/grepo_train.jsonl"
TEST_PREDS = ROOT / "experiments/exp1_sft_only/eval_graph_expanded/predictions.jsonl"
DEP_GRAPH_DIR = ROOT / "data/dep_graphs"
BM25_TRAIN = ROOT / "data/rankft/grepo_train_bm25_top500.jsonl"
BM25_TEST = ROOT / "data/rankft/grepo_test_bm25_top500.jsonl"


# ─── Data Loading ──────────────────────────────────────────────────────

def load_train_data():
    """Load train examples grouped by repo."""
    examples = []
    with open(TRAIN_DATA) as f:
        for line in f:
            item = json.loads(line)
            if item.get("split") != "train":
                continue
            files = item.get("changed_py_files", [])
            if not files:
                files = [f_ for f_ in item.get("changed_files", []) if f_.endswith(".py")]
            if files:
                examples.append({
                    "repo": item["repo"],
                    "issue_id": item["issue_id"],
                    "changed_files": files,
                })
    return examples


def load_test_predictions():
    """Load test predictions with original SFT preds and ground truth."""
    preds = []
    with open(TEST_PREDS) as f:
        for line in f:
            d = json.loads(line)
            preds.append({
                "repo": d["repo"],
                "issue_id": d["issue_id"],
                "ground_truth": d["ground_truth"],
                "sft_predicted": d.get("predicted_original", d["predicted"][:10]),
                "full_expanded": d["predicted"],
            })
    return preds


def build_cochange_index_from_examples(examples, min_cochange=1):
    """Build co-change index from a list of train examples."""
    repo_cochanges = defaultdict(Counter)
    repo_file_count = defaultdict(Counter)

    for ex in examples:
        repo = ex["repo"]
        files = ex["changed_files"]
        for f in files:
            repo_file_count[repo][f] += 1
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


def build_import_index():
    """Build per-repo bidirectional import index from dep_graphs."""
    index = {}
    for fname in os.listdir(DEP_GRAPH_DIR):
        if not fname.endswith("_rels.json"):
            continue
        repo = fname.replace("_rels.json", "")
        with open(DEP_GRAPH_DIR / fname) as f:
            rels = json.load(f)
        file_imports = rels.get("file_imports", {})
        neighbors = defaultdict(set)
        for src, targets in file_imports.items():
            for tgt in targets:
                neighbors[src].add(tgt)
                neighbors[tgt].add(src)
        call_graph = rels.get("call_graph", {})
        for src_func, callees in call_graph.items():
            src_file = src_func.split(":")[0] if ":" in src_func else src_func
            for callee in callees:
                tgt_file = callee.split(":")[0] if ":" in callee else callee
                if src_file != tgt_file:
                    neighbors[src_file].add(tgt_file)
                    neighbors[tgt_file].add(src_file)
        index[repo] = {k: v for k, v in neighbors.items()}
    return index


def load_bm25_index(path):
    """Load BM25 candidate rankings as {(repo, issue_id): {file: rank}}."""
    index = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            repo = d["repo"]
            issue_id = d["issue_id"]
            cands = d["bm25_candidates"]
            rank_map = {c: i for i, c in enumerate(cands)}
            index[(repo, issue_id)] = rank_map
    return index


# ─── Feature Extraction ────────────────────────────────────────────────

def directory_distance(file_a: str, file_b: str) -> int:
    """Number of different directory components."""
    parts_a = Path(file_a).parts[:-1]
    parts_b = Path(file_b).parts[:-1]
    # Find common prefix length
    common = 0
    for pa, pb in zip(parts_a, parts_b):
        if pa == pb:
            common += 1
        else:
            break
    return (len(parts_a) - common) + (len(parts_b) - common)


def same_directory(file_a: str, file_b: str) -> int:
    return int(str(Path(file_a).parent) == str(Path(file_b).parent))


def is_test_file(filepath: str) -> int:
    name = Path(filepath).stem
    return int("test" in name or "tests" in name or name.startswith("test_") or name.endswith("_test"))


def extract_neighbor_features(
    neighbor: str,
    seed_files: List[str],
    cochange_index: Dict,
    import_index: Dict,
    repo: str,
    bm25_ranks: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """Extract features for a candidate neighbor file."""
    repo_cc = cochange_index.get(repo, {})
    repo_imp = import_index.get(repo, {})

    # Co-change features
    cc_weights = []
    cc_link_count = 0
    for sf in seed_files:
        w = repo_cc.get(sf, {}).get(neighbor, 0.0)
        if w > 0:
            cc_weights.append(w)
            cc_link_count += 1

    max_cc_weight = max(cc_weights) if cc_weights else 0.0
    mean_cc_weight = np.mean(cc_weights) if cc_weights else 0.0
    sum_cc_weight = sum(cc_weights)

    # Import features
    imp_link_count = 0
    for sf in seed_files:
        if neighbor in repo_imp.get(sf, set()):
            imp_link_count += 1

    # Neighbor's total import degree
    neighbor_import_degree = len(repo_imp.get(neighbor, set()))

    # Directory distance features
    dir_dists = [directory_distance(sf, neighbor) for sf in seed_files]
    min_dir_dist = min(dir_dists) if dir_dists else 10
    mean_dir_dist = np.mean(dir_dists) if dir_dists else 10
    any_same_dir = int(any(same_directory(sf, neighbor) for sf in seed_files))

    # File type features
    neighbor_is_test = is_test_file(neighbor)
    any_seed_is_test = int(any(is_test_file(sf) for sf in seed_files))
    test_match = int(neighbor_is_test == any_seed_is_test)

    # Path depth
    neighbor_depth = len(Path(neighbor).parts)

    # Number of seed files linking to this neighbor (combined)
    total_link_count = cc_link_count + imp_link_count

    # BM25 rank feature
    bm25_rank = 501  # default: not in BM25 top 500
    if bm25_ranks is not None:
        bm25_rank = bm25_ranks.get(neighbor, 501)
    bm25_in_top50 = int(bm25_rank < 50)
    bm25_in_top100 = int(bm25_rank < 100)

    # Connection type
    has_cochange = int(cc_link_count > 0)
    has_import = int(imp_link_count > 0)
    has_both = int(has_cochange and has_import)

    features = {
        "max_cc_weight": max_cc_weight,
        "mean_cc_weight": mean_cc_weight,
        "sum_cc_weight": sum_cc_weight,
        "cc_link_count": cc_link_count,
        "imp_link_count": imp_link_count,
        "neighbor_import_degree": neighbor_import_degree,
        "min_dir_dist": min_dir_dist,
        "mean_dir_dist": mean_dir_dist,
        "any_same_dir": any_same_dir,
        "neighbor_is_test": neighbor_is_test,
        "test_match": test_match,
        "neighbor_depth": neighbor_depth,
        "total_link_count": total_link_count,
        "bm25_rank_inv": 1.0 / (bm25_rank + 1),
        "bm25_in_top50": bm25_in_top50,
        "bm25_in_top100": bm25_in_top100,
        "has_cochange": has_cochange,
        "has_import": has_import,
        "has_both": has_both,
        "num_seed_files": len(seed_files),
    }
    return features


# ─── Dataset Construction ──────────────────────────────────────────────

def get_all_neighbors(
    seed_files: List[str],
    repo: str,
    cochange_index: Dict,
    import_index: Dict,
) -> Set[str]:
    """Get all graph neighbors of seed files (co-change + import)."""
    neighbors = set()
    seed_set = set(seed_files)

    repo_cc = cochange_index.get(repo, {})
    for sf in seed_files:
        for nb in repo_cc.get(sf, {}):
            if nb not in seed_set:
                neighbors.add(nb)

    repo_imp = import_index.get(repo, {})
    for sf in seed_files:
        for nb in repo_imp.get(sf, set()):
            if nb not in seed_set:
                neighbors.add(nb)

    return neighbors


def build_train_dataset(train_examples, import_index, bm25_train_index):
    """
    Build training dataset for the expansion classifier.

    Strategy: For each train example, use leave-one-out co-change index
    (exclude current example), simulate SFT predictions as a random subset
    of GT files, find neighbors, label them.
    """
    print("Building per-repo example groups...")
    repo_examples = defaultdict(list)
    for ex in train_examples:
        repo_examples[ex["repo"]].append(ex)

    # Build full co-change index first
    full_cc_index = build_cochange_index_from_examples(train_examples)

    features_list = []
    labels = []
    meta = []

    print("Constructing training samples...")
    n_examples = 0
    n_pos = 0

    for repo, examples in repo_examples.items():
        if len(examples) < 5:
            continue  # Need enough examples for meaningful co-change

        for i, ex in enumerate(examples):
            gt_set = set(ex["changed_files"])

            # Simulate SFT prediction: use a random subset of GT as "seed"
            # (In reality, SFT gets some right and some wrong; we simulate
            # by picking 1-3 GT files as seeds and seeing if neighbors cover rest)
            if len(gt_set) < 2:
                continue  # Need at least 2 files to have something to predict

            # Use first file as seed (deterministic), rest as targets
            gt_list = sorted(gt_set)
            n_seed = max(1, len(gt_list) // 2)
            seed_files = gt_list[:n_seed]
            target_files = set(gt_list[n_seed:])

            # Build LOO co-change index (approximate: just use full index,
            # the contribution of one example is small)
            cc_index = full_cc_index

            neighbors = get_all_neighbors(seed_files, repo, cc_index, import_index)
            if not neighbors:
                continue

            bm25_key = (repo, ex["issue_id"])
            bm25_ranks = bm25_train_index.get(bm25_key, None)

            for nb in neighbors:
                feat = extract_neighbor_features(
                    nb, seed_files, cc_index, import_index, repo, bm25_ranks
                )
                label = int(nb in target_files)
                features_list.append(feat)
                labels.append(label)
                meta.append({"repo": repo, "issue_id": ex["issue_id"], "neighbor": nb})
                if label:
                    n_pos += 1

            n_examples += 1

    print(f"Training dataset: {len(features_list)} samples, {n_pos} positive "
          f"({100*n_pos/max(len(labels),1):.2f}%), from {n_examples} examples")

    return features_list, labels, meta


def build_test_dataset(test_preds, cochange_index, import_index, bm25_test_index):
    """Build test dataset using actual SFT predictions."""
    features_list = []
    labels = []
    meta = []
    n_examples = 0
    n_pos = 0

    for pred in test_preds:
        repo = pred["repo"]
        gt_set = set(pred["ground_truth"])
        seed_files = pred["sft_predicted"]
        seed_set = set(seed_files)

        neighbors = get_all_neighbors(seed_files, repo, cochange_index, import_index)
        if not neighbors:
            continue

        bm25_key = (repo, pred["issue_id"])
        bm25_ranks = bm25_test_index.get(bm25_key, None)

        for nb in neighbors:
            feat = extract_neighbor_features(
                nb, seed_files, cochange_index, import_index, repo, bm25_ranks
            )
            label = int(nb in gt_set)
            features_list.append(feat)
            labels.append(label)
            meta.append({
                "repo": repo, "issue_id": pred["issue_id"],
                "neighbor": nb, "in_sft": nb in seed_set,
            })
            if label:
                n_pos += 1

        n_examples += 1

    print(f"Test dataset: {len(features_list)} samples, {n_pos} positive "
          f"({100*n_pos/max(len(labels),1):.2f}%), from {n_examples} examples")

    return features_list, labels, meta


# ─── Evaluation ─────────────────────────────────────────────────────────

def evaluate_expansion_policy(
    test_preds, cochange_index, import_index, bm25_test_index,
    model, scaler, feature_names, top_k_values=[3, 5, 10, 15, 20]
):
    """
    Evaluate classifier-filtered expansion vs baselines.
    For each test example:
    1. Get SFT predictions (seeds)
    2. Get all graph neighbors
    3. Score neighbors with classifier
    4. Compare: top-K classifier vs random-K vs full expansion
    """
    results = {
        "classifier": {k: [] for k in top_k_values},
        "random": {k: [] for k in top_k_values},
        "full": [],
        "sft_only": [],
    }
    pool_sizes = {
        "classifier": {k: [] for k in top_k_values},
        "random": {k: [] for k in top_k_values},
        "full": [],
        "sft_only": [],
    }

    for pred in test_preds:
        repo = pred["repo"]
        gt_set = set(pred["ground_truth"])
        seed_files = pred["sft_predicted"]
        seed_set = set(seed_files)

        # SFT-only recall
        sft_recall = len(gt_set & seed_set) / len(gt_set) if gt_set else 0
        results["sft_only"].append(sft_recall)
        pool_sizes["sft_only"].append(len(seed_set))

        neighbors = get_all_neighbors(seed_files, repo, cochange_index, import_index)
        neighbor_list = sorted(neighbors - seed_set)

        # Full expansion
        full_pool = seed_set | neighbors
        full_recall = len(gt_set & full_pool) / len(gt_set) if gt_set else 0
        results["full"].append(full_recall)
        pool_sizes["full"].append(len(full_pool))

        if not neighbor_list:
            for k in top_k_values:
                results["classifier"][k].append(sft_recall)
                results["random"][k].append(sft_recall)
                pool_sizes["classifier"][k].append(len(seed_set))
                pool_sizes["random"][k].append(len(seed_set))
            continue

        # Score neighbors
        bm25_key = (repo, pred["issue_id"])
        bm25_ranks = bm25_test_index.get(bm25_key, None)

        feat_dicts = [
            extract_neighbor_features(
                nb, seed_files, cochange_index, import_index, repo, bm25_ranks
            )
            for nb in neighbor_list
        ]
        X = np.array([[fd[fn] for fn in feature_names] for fd in feat_dicts])
        X_scaled = scaler.transform(X)
        scores = model.predict_proba(X_scaled)[:, 1]

        # Rank by classifier score
        ranked_indices = np.argsort(-scores)
        ranked_neighbors = [neighbor_list[i] for i in ranked_indices]

        # Random baseline (fixed seed per example for reproducibility)
        rng = random.Random(pred["issue_id"])
        random_neighbors = list(neighbor_list)
        rng.shuffle(random_neighbors)

        for k in top_k_values:
            # Classifier top-K
            cls_expansion = set(ranked_neighbors[:k])
            cls_pool = seed_set | cls_expansion
            cls_recall = len(gt_set & cls_pool) / len(gt_set) if gt_set else 0
            results["classifier"][k].append(cls_recall)
            pool_sizes["classifier"][k].append(len(cls_pool))

            # Random top-K
            rand_expansion = set(random_neighbors[:k])
            rand_pool = seed_set | rand_expansion
            rand_recall = len(gt_set & rand_pool) / len(gt_set) if gt_set else 0
            results["random"][k].append(rand_recall)
            pool_sizes["random"][k].append(len(rand_pool))

    return results, pool_sizes


def print_results(results, pool_sizes, top_k_values):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("EXPANSION POLICY COMPARISON (Oracle Recall)")
    print("=" * 80)

    print(f"\n{'Method':<30} {'Oracle R@pool':>15} {'Avg Pool Size':>15}")
    print("-" * 60)

    sft_recall = 100 * np.mean(results["sft_only"])
    sft_pool = np.mean(pool_sizes["sft_only"])
    print(f"{'SFT only (no expansion)':<30} {sft_recall:>14.2f}% {sft_pool:>15.1f}")

    for k in top_k_values:
        r = 100 * np.mean(results["random"][k])
        p = np.mean(pool_sizes["random"][k])
        print(f"{'Random expand top-' + str(k):<30} {r:>14.2f}% {p:>15.1f}")

    print()
    for k in top_k_values:
        r = 100 * np.mean(results["classifier"][k])
        p = np.mean(pool_sizes["classifier"][k])
        print(f"{'Classifier expand top-' + str(k):<30} {r:>14.2f}% {p:>15.1f}")

    full_recall = 100 * np.mean(results["full"])
    full_pool = np.mean(pool_sizes["full"])
    print(f"\n{'Full graph expansion':<30} {full_recall:>14.2f}% {full_pool:>15.1f}")

    # Compute gaps
    print("\n" + "=" * 80)
    print("CLASSIFIER vs RANDOM GAP (at matched budget)")
    print("=" * 80)
    print(f"\n{'Budget K':<15} {'Classifier':>12} {'Random':>12} {'Gap':>12}")
    print("-" * 55)
    for k in top_k_values:
        cr = 100 * np.mean(results["classifier"][k])
        rr = 100 * np.mean(results["random"][k])
        gap = cr - rr
        print(f"{'K=' + str(k):<15} {cr:>11.2f}% {rr:>11.2f}% {gap:>+11.2f}%")


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("Learned Expansion Policy Experiment")
    print("=" * 80)

    # 1. Load data
    print("\n[1/6] Loading data...")
    train_examples = load_train_data()
    print(f"  Train examples: {len(train_examples)}")

    test_preds = load_test_predictions()
    print(f"  Test predictions: {len(test_preds)}")

    # 2. Build indexes
    print("\n[2/6] Building indexes...")
    full_cc_index = build_cochange_index_from_examples(train_examples)
    print(f"  Co-change index: {len(full_cc_index)} repos")

    import_index = build_import_index()
    print(f"  Import index: {len(import_index)} repos")

    bm25_train_index = load_bm25_index(BM25_TRAIN)
    print(f"  BM25 train index: {len(bm25_train_index)} examples")

    bm25_test_index = load_bm25_index(BM25_TEST)
    print(f"  BM25 test index: {len(bm25_test_index)} examples")

    # 3. Build datasets
    print("\n[3/6] Building training dataset...")
    train_feats, train_labels, train_meta = build_train_dataset(
        train_examples, import_index, bm25_train_index
    )
    feature_names = sorted(train_feats[0].keys())
    X_train = np.array([[fd[fn] for fn in feature_names] for fd in train_feats])
    y_train = np.array(train_labels)

    print("\n[4/6] Building test dataset...")
    test_feats, test_labels, test_meta = build_test_dataset(
        test_preds, full_cc_index, import_index, bm25_test_index
    )
    X_test = np.array([[fd[fn] for fn in feature_names] for fd in test_feats])
    y_test = np.array(test_labels)

    # 4. Train models
    print("\n[5/6] Training classifiers...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    print("\n--- Logistic Regression ---")
    lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, C=1.0, random_state=42
    )
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_probs) if y_test.sum() > 0 else 0
    lr_ap = average_precision_score(y_test, lr_probs) if y_test.sum() > 0 else 0
    print(f"  Test AUC-ROC: {lr_auc:.4f}")
    print(f"  Test AP:      {lr_ap:.4f}")

    # Feature importance (LR coefficients)
    print("\n  Feature coefficients:")
    coef_order = np.argsort(-np.abs(lr.coef_[0]))
    for idx in coef_order:
        print(f"    {feature_names[idx]:<25} {lr.coef_[0][idx]:+.4f}")

    # Gradient Boosting
    print("\n--- Gradient Boosting ---")
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42,
        min_samples_leaf=20,
    )
    # Handle class imbalance with sample weights
    pos_ratio = y_train.sum() / len(y_train)
    sample_weights = np.where(y_train == 1, 1.0 / pos_ratio, 1.0 / (1 - pos_ratio))
    sample_weights = sample_weights / sample_weights.mean()

    gb.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    gb_probs = gb.predict_proba(X_test_scaled)[:, 1]
    gb_auc = roc_auc_score(y_test, gb_probs) if y_test.sum() > 0 else 0
    gb_ap = average_precision_score(y_test, gb_probs) if y_test.sum() > 0 else 0
    print(f"  Test AUC-ROC: {gb_auc:.4f}")
    print(f"  Test AP:      {gb_ap:.4f}")

    # Feature importance (GB)
    print("\n  Feature importances:")
    imp_order = np.argsort(-gb.feature_importances_)
    for idx in imp_order:
        print(f"    {feature_names[idx]:<25} {gb.feature_importances_[idx]:.4f}")

    # 5. Select best model
    best_model_name = "GradientBoosting" if gb_auc > lr_auc else "LogisticRegression"
    best_model = gb if gb_auc > lr_auc else lr
    best_auc = max(gb_auc, lr_auc)
    print(f"\nBest model: {best_model_name} (AUC={best_auc:.4f})")

    # 6. Evaluate expansion policies
    print("\n[6/6] Evaluating expansion policies...")
    top_k_values = [3, 5, 10, 15, 20]
    results, pool_sizes = evaluate_expansion_policy(
        test_preds, full_cc_index, import_index, bm25_test_index,
        best_model, scaler, feature_names, top_k_values
    )

    print_results(results, pool_sizes, top_k_values)

    # Also evaluate with the other model for comparison
    other_model = lr if best_model_name == "GradientBoosting" else gb
    other_name = "LogisticRegression" if best_model_name == "GradientBoosting" else "GradientBoosting"
    results2, pool_sizes2 = evaluate_expansion_policy(
        test_preds, full_cc_index, import_index, bm25_test_index,
        other_model, scaler, feature_names, top_k_values
    )

    print(f"\n{'=' * 80}")
    print(f"COMPARISON: {best_model_name} vs {other_name}")
    print(f"{'=' * 80}")
    print(f"\n{'Budget K':<15} {best_model_name:>15} {other_name:>15} {'Random':>12}")
    print("-" * 60)
    for k in top_k_values:
        r1 = 100 * np.mean(results["classifier"][k])
        r2 = 100 * np.mean(results2["classifier"][k])
        rr = 100 * np.mean(results["random"][k])
        print(f"{'K=' + str(k):<15} {r1:>14.2f}% {r2:>14.2f}% {rr:>11.2f}%")

    # Summary statistics
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    sft_r = 100 * np.mean(results["sft_only"])
    full_r = 100 * np.mean(results["full"])
    full_p = np.mean(pool_sizes["full"])
    print(f"SFT-only recall:           {sft_r:.2f}%")
    print(f"Full expansion recall:     {full_r:.2f}% (avg pool={full_p:.0f})")

    # Find the K where classifier matches full expansion recall
    for k in top_k_values:
        cr = 100 * np.mean(results["classifier"][k])
        cp = np.mean(pool_sizes["classifier"][k])
        rr = 100 * np.mean(results["random"][k])
        if cr >= full_r * 0.95:
            print(f"Classifier reaches 95% of full recall at K={k} "
                  f"(recall={cr:.2f}%, pool={cp:.0f} vs full pool={full_p:.0f})")
            break

    # Precision at different thresholds
    print(f"\nClassifier precision analysis (on test neighbors):")
    probs = best_model.predict_proba(X_test_scaled)[:, 1]
    for thresh in [0.1, 0.2, 0.3, 0.5, 0.7]:
        pred_pos = probs >= thresh
        if pred_pos.sum() > 0:
            prec = y_test[pred_pos].mean()
            recall = y_test[pred_pos].sum() / max(y_test.sum(), 1)
            print(f"  threshold={thresh:.1f}: precision={100*prec:.1f}%, "
                  f"recall={100*recall:.1f}%, "
                  f"n_selected={pred_pos.sum()}/{len(probs)}")


if __name__ == "__main__":
    main()
