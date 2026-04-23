"""
Graph-RAG baseline evaluation for GREPO.

Compares "in-context" graph knowledge (Graph-RAG) against our
"in-parameter" approach (cross-encoder trained with graph-based hard negatives).

Method: For each (issue, candidate_file) pair, we enrich the prompt with
the file's graph neighbors (co-change, imports) so the base LLM can use
structural context when scoring Yes/No.  No LoRA is loaded -- this is the
pure base model with retrieval-augmented prompts.

Metrics: Hit@k, Acc@k, Recall@k, Conditional Acc@1|GT in candidates.

Usage:
    python scripts/eval_graph_rag_baseline.py \
        --gpu_id 0 \
        --output_dir experiments/graph_rag_baseline/eval \
        --top_k 200 \
        --max_seq_length 768
"""

import os
import json
import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Deterministic
torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# Paths (defaults)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
DEFAULT_TEST_DATA = str(PROJECT_ROOT / "data" / "grepo_text" / "grepo_test.jsonl")
DEFAULT_BM25_CANDIDATES = str(
    PROJECT_ROOT / "data" / "rankft" / "exp6_expanded_candidates.jsonl"
)
DEFAULT_DEP_GRAPH_DIR = str(PROJECT_ROOT / "data" / "dep_graphs")
DEFAULT_TRAIN_DATA = str(PROJECT_ROOT / "data" / "grepo_text" / "grepo_train.jsonl")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "experiments" / "graph_rag_baseline" / "eval")


# ============================================================
# Graph-RAG prompt template
# ============================================================

PROMPT_TEMPLATE = (
    "Given the bug report and file context, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n"
    "Related files (co-change): {cochange_neighbors}\n"
    "Related files (imports): {import_neighbors}\n\n"
    "Answer:"
)


def build_graph_prompt(
    issue_text: str,
    candidate_path: str,
    cochange_neighbors: List[str],
    import_neighbors: List[str],
    max_neighbors: int = 10,
) -> str:
    """Build the Graph-RAG scoring prompt for a single (issue, file) pair."""
    cc_str = ", ".join(cochange_neighbors[:max_neighbors]) if cochange_neighbors else "none"
    imp_str = ", ".join(import_neighbors[:max_neighbors]) if import_neighbors else "none"
    return PROMPT_TEMPLATE.format(
        issue_text=issue_text,
        candidate_path=candidate_path,
        cochange_neighbors=cc_str,
        import_neighbors=imp_str,
    )


# ============================================================
# Graph loading
# ============================================================

def load_dep_graphs(dep_graph_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """Load dependency (import) graphs from *_rels.json files.

    Returns:
        {repo: {file_path: [imported_file_1, ...]}}
    """
    import_graph: Dict[str, Dict[str, List[str]]] = {}

    rels_dir = Path(dep_graph_dir)
    for fpath in sorted(rels_dir.glob("*_rels.json")):
        repo_name = fpath.stem.replace("_rels", "")
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  WARNING: Failed to load {fpath}: {e}")
            continue

        file_imports = data.get("file_imports", {})
        if isinstance(file_imports, dict):
            import_graph[repo_name] = file_imports
        else:
            print(f"  WARNING: Unexpected file_imports type in {fpath}: {type(file_imports)}")

    print(f"  Loaded import graphs for {len(import_graph)} repos")
    return import_graph


def build_reverse_import_graph(
    import_graph: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Dict[str, List[str]]]:
    """Build reverse import graph: {repo: {file: [files that import it]}}."""
    reverse: Dict[str, Dict[str, List[str]]] = {}
    for repo, file_imports in import_graph.items():
        rev = defaultdict(list)
        for src_file, imported_files in file_imports.items():
            for tgt_file in imported_files:
                rev[tgt_file].append(src_file)
        reverse[repo] = dict(rev)
    return reverse


def build_cochange_graph(train_data_path: str) -> Dict[str, Dict[str, Set[str]]]:
    """Build co-change graph from training data.

    Two files co-change if they both appear in changed_py_files for the
    same issue in the training set.

    Returns:
        {repo: {file_path: set_of_cochange_files}}
    """
    cochange: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    with open(train_data_path) as f:
        for line in f:
            item = json.loads(line)
            repo = item["repo"]
            changed = item.get("changed_py_files", [])
            # Each pair of changed files in the same issue is a co-change edge
            for i, f1 in enumerate(changed):
                for f2 in changed[i + 1 :]:
                    cochange[repo][f1].add(f2)
                    cochange[repo][f2].add(f1)

    # Convert inner defaultdicts to regular dicts for cleaner access
    result = {}
    for repo, file_map in cochange.items():
        result[repo] = {fpath: neighbors for fpath, neighbors in file_map.items()}

    print(f"  Built co-change graph for {len(result)} repos")
    total_edges = sum(
        sum(len(v) for v in repo_map.values()) for repo_map in result.values()
    )
    print(f"  Total co-change edges (undirected, counted twice): {total_edges}")
    return result


def get_directory_neighbors(
    candidate: str, all_candidates: List[str], max_depth: int = 1
) -> List[str]:
    """Find files in the same directory (or parent directory) as the candidate.

    Only returns files that are also in the candidate list (for relevance).
    """
    parts = candidate.rsplit("/", 1)
    if len(parts) < 2:
        return []
    candidate_dir = parts[0]
    neighbors = []
    for other in all_candidates:
        if other == candidate:
            continue
        if other.startswith(candidate_dir + "/"):
            neighbors.append(other)
    return neighbors


# ============================================================
# Scoring
# ============================================================

def get_yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    """Get token IDs for 'Yes' and 'No'."""
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    return yes_ids[0], no_ids[0]


@torch.no_grad()
def score_candidates_batched(
    model,
    tokenizer,
    prompts: List[str],
    yes_id: int,
    no_id: int,
    max_seq_length: int,
    device: str,
    batch_size: int = 8,
) -> List[float]:
    """Score all candidates for a single issue, in batches.

    Returns list of scores (logit_yes - logit_no) for each prompt.
    """
    all_scores = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        encodings = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Fall back to one-at-a-time
                torch.cuda.empty_cache()
                for prompt in batch_prompts:
                    enc = tokenizer(
                        [prompt],
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_seq_length,
                    )
                    ids = enc["input_ids"].to(device)
                    mask = enc["attention_mask"].to(device)
                    out = model(input_ids=ids, attention_mask=mask)
                    logits = out.logits[0, -1]
                    score = (logits[yes_id] - logits[no_id]).item()
                    all_scores.append(score)
                continue
            raise

        logits = outputs.logits  # (batch, seq_len, vocab_size)

        # Get logits at last non-padding position for each sequence
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.size(0), device=device)
        last_logits = logits[batch_indices, seq_lengths]  # (batch, vocab_size)

        scores = (last_logits[:, yes_id] - last_logits[:, no_id]).cpu().tolist()
        all_scores.extend(scores)

    return all_scores


# ============================================================
# Metrics
# ============================================================

def compute_hit_at_k(predicted: List[str], gt: Set[str], k: int) -> float:
    """Hit@k: fraction of GT files found in top-k predictions."""
    if not gt:
        return 0.0
    top_k = set(predicted[:k])
    return len(top_k & gt) / len(gt)


def compute_acc_at_k(predicted: List[str], gt: Set[str], k: int) -> float:
    """Acc@k: 1.0 if ALL GT files are in top-k, else 0.0."""
    if not gt:
        return 0.0
    top_k = set(predicted[:k])
    return 1.0 if gt.issubset(top_k) else 0.0


def compute_recall_at_k(predicted: List[str], gt: Set[str], k: int) -> float:
    """Recall@k: same as hit@k for this task."""
    return compute_hit_at_k(predicted, gt, k)


# ============================================================
# Evaluation
# ============================================================

def evaluate(args):
    """Main evaluation routine."""
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    # ---- Load test data ----
    print(f"Loading test data from {args.test_data}...")
    test_data = []
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            if item.get("changed_py_files"):
                test_data.append(item)
    print(f"  {len(test_data)} test examples")

    # ---- Load BM25 candidates ----
    print(f"Loading BM25 candidates from {args.bm25_candidates}...")
    bm25_map: Dict[str, List[str]] = {}
    with open(args.bm25_candidates) as f:
        for line in f:
            item = json.loads(line)
            key = f"{item['repo']}_{item['issue_id']}"
            bm25_map[key] = item.get("candidates", item.get("bm25_candidates", []))
    print(f"  BM25 candidates for {len(bm25_map)} examples")

    # ---- Load dependency graphs ----
    print(f"Loading dependency graphs from {args.dep_graph_dir}...")
    import_graph = load_dep_graphs(args.dep_graph_dir)
    reverse_import_graph = build_reverse_import_graph(import_graph)

    # ---- Build co-change graph from training data ----
    print(f"Building co-change graph from {args.train_data}...")
    cochange_graph = build_cochange_graph(args.train_data)

    # ---- Load model (NO LoRA) ----
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    yes_id, no_id = get_yes_no_token_ids(tokenizer)
    print(f"  Yes ID: {yes_id}, No ID: {no_id}")

    print(f"Loading base model from {args.model_path} (NO LoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print("  Model loaded and set to eval mode.")

    # ---- Evaluate ----
    k_values = [1, 3, 5, 10, 20]
    results = []
    overall_metrics = {f"hit@{k}": [] for k in k_values}
    overall_metrics.update({f"acc@{k}": [] for k in k_values})
    overall_metrics.update({f"recall@{k}": [] for k in k_values})

    # Conditional Acc@1
    cond_acc1_correct = 0
    cond_acc1_total = 0

    per_repo_metrics = defaultdict(lambda: defaultdict(list))

    total = len(test_data)
    start_time = time.time()

    # Stats for graph context coverage
    graph_stats = {
        "has_cochange": 0,
        "has_import": 0,
        "has_any_graph": 0,
        "total_candidates_scored": 0,
    }

    for idx, example in enumerate(test_data):
        repo = example["repo"]
        issue_id = example["issue_id"]
        issue_text = example["issue_text"]
        gt_files = set(example["changed_py_files"])

        bm25_key = f"{repo}_{issue_id}"
        candidates = bm25_map.get(bm25_key, [])

        if not candidates:
            continue

        # Truncate to top_k
        candidates = candidates[: args.top_k]

        # Check if any GT file is in the candidate list
        gt_in_candidates = bool(gt_files & set(candidates))

        if idx % 20 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / max(elapsed, 1)
            eta = (total - idx - 1) / max(rate, 0.001)
            print(
                f"  [{idx+1}/{total}] {repo}#{issue_id} | "
                f"{len(candidates)} candidates | "
                f"GT in candidates: {gt_in_candidates} | "
                f"ETA: {eta:.0f}s"
            )

        # ---- Build Graph-RAG prompts ----
        repo_imports = import_graph.get(repo, {})
        repo_reverse_imports = reverse_import_graph.get(repo, {})
        repo_cochange = cochange_graph.get(repo, {})
        candidate_set = set(candidates)

        prompts = []
        per_candidate_graph_info = []

        for cand in candidates:
            # Co-change neighbors (from training data)
            cc_neighbors_raw = repo_cochange.get(cand, set())
            # Filter to candidates that are in the current candidate list for relevance
            cc_neighbors = sorted(cc_neighbors_raw & candidate_set - {cand})

            # Import neighbors (files this file imports + files that import this file)
            imports_out = repo_imports.get(cand, [])
            imports_in = repo_reverse_imports.get(cand, [])
            imp_neighbors_raw = set(imports_out) | set(imports_in)
            imp_neighbors = sorted(imp_neighbors_raw & candidate_set - {cand})

            # Track stats
            has_cc = len(cc_neighbors) > 0
            has_imp = len(imp_neighbors) > 0
            if has_cc:
                graph_stats["has_cochange"] += 1
            if has_imp:
                graph_stats["has_import"] += 1
            if has_cc or has_imp:
                graph_stats["has_any_graph"] += 1
            graph_stats["total_candidates_scored"] += 1

            prompt = build_graph_prompt(
                issue_text=issue_text,
                candidate_path=cand,
                cochange_neighbors=cc_neighbors,
                import_neighbors=imp_neighbors,
                max_neighbors=args.max_neighbors,
            )
            prompts.append(prompt)
            per_candidate_graph_info.append({
                "cochange": cc_neighbors[:5],
                "imports": imp_neighbors[:5],
            })

        # ---- Score all candidates ----
        t0 = time.time()
        scores = score_candidates_batched(
            model,
            tokenizer,
            prompts,
            yes_id,
            no_id,
            args.max_seq_length,
            device,
            batch_size=args.score_batch_size,
        )
        scoring_time = time.time() - t0

        # Rerank by score (descending)
        scored_candidates = sorted(
            zip(candidates, scores), key=lambda x: -x[1]
        )
        reranked = [c for c, _ in scored_candidates]

        # Compute metrics
        metrics = {}
        for k in k_values:
            hit = compute_hit_at_k(reranked, gt_files, k)
            acc = compute_acc_at_k(reranked, gt_files, k)
            recall = compute_recall_at_k(reranked, gt_files, k)
            metrics[f"hit@{k}"] = hit
            metrics[f"acc@{k}"] = acc
            metrics[f"recall@{k}"] = recall
            overall_metrics[f"hit@{k}"].append(hit)
            overall_metrics[f"acc@{k}"].append(acc)
            overall_metrics[f"recall@{k}"].append(recall)
            per_repo_metrics[repo][f"hit@{k}"].append(hit)
            per_repo_metrics[repo][f"acc@{k}"].append(acc)
            per_repo_metrics[repo][f"recall@{k}"].append(recall)

        # Conditional Acc@1
        if gt_in_candidates:
            cond_acc1_total += 1
            if reranked[0] in gt_files:
                cond_acc1_correct += 1

        # Store result
        result = {
            "repo": repo,
            "issue_id": issue_id,
            "ground_truth": list(gt_files),
            "predicted": reranked[:50],
            "bm25_original": candidates[:20],
            "scores": [s for _, s in scored_candidates[:50]],
            "metrics": metrics,
            "gt_in_candidates": gt_in_candidates,
            "num_candidates": len(candidates),
            "scoring_time": round(scoring_time, 3),
        }
        results.append(result)

    # ---- Aggregate metrics ----
    total_evaluated = len(results)
    elapsed_total = time.time() - start_time

    avg_overall = {}
    for metric_name, values in overall_metrics.items():
        if values:
            avg_overall[metric_name] = sum(values) / len(values) * 100
        else:
            avg_overall[metric_name] = 0.0

    cond_acc1 = (
        (cond_acc1_correct / cond_acc1_total * 100)
        if cond_acc1_total > 0
        else 0.0
    )
    avg_overall["cond_acc@1|gt_in_candidates"] = cond_acc1

    avg_per_repo = {}
    for repo, repo_m in per_repo_metrics.items():
        avg_per_repo[repo] = {}
        for metric_name, values in repo_m.items():
            avg_per_repo[repo][metric_name] = sum(values) / len(values) * 100
        avg_per_repo[repo]["count"] = len(repo_m.get("hit@1", []))

    # ---- Graph context coverage stats ----
    total_scored = max(graph_stats["total_candidates_scored"], 1)
    cc_pct = graph_stats["has_cochange"] / total_scored * 100
    imp_pct = graph_stats["has_import"] / total_scored * 100
    any_pct = graph_stats["has_any_graph"] / total_scored * 100

    # ---- Print results ----
    print(f"\n{'='*70}")
    print(f"GRAPH-RAG BASELINE EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  Model: {args.model_path} (base, NO LoRA)")
    print(f"  Examples evaluated: {total_evaluated}")
    print(f"  Total time: {elapsed_total:.0f}s ({elapsed_total/3600:.2f}h)")
    print(f"  Avg time/example: {elapsed_total/max(total_evaluated,1):.2f}s")

    print(f"\nGRAPH CONTEXT COVERAGE:")
    print(f"  Candidates with co-change neighbors: {graph_stats['has_cochange']}/{total_scored} ({cc_pct:.1f}%)")
    print(f"  Candidates with import neighbors:    {graph_stats['has_import']}/{total_scored} ({imp_pct:.1f}%)")
    print(f"  Candidates with any graph context:   {graph_stats['has_any_graph']}/{total_scored} ({any_pct:.1f}%)")

    print(f"\nOVERALL METRICS:")
    for k in k_values:
        print(
            f"  Hit@{k}: {avg_overall[f'hit@{k}']:.2f}%  |  "
            f"Acc@{k}: {avg_overall[f'acc@{k}']:.2f}%  |  "
            f"Recall@{k}: {avg_overall[f'recall@{k}']:.2f}%"
        )
    print(
        f"\n  Cond. Acc@1|GT in candidates: {cond_acc1:.2f}% "
        f"({cond_acc1_correct}/{cond_acc1_total})"
    )

    print(f"\nPER-REPO RESULTS:")
    header = (
        "Repo".ljust(18)
        + "".join(f"H@{k}".rjust(8) for k in k_values)
        + "  Count"
    )
    print(header)
    print("-" * len(header))
    for repo in sorted(avg_per_repo.keys()):
        m = avg_per_repo[repo]
        line = repo[:17].ljust(18)
        line += "".join(f"{m.get(f'hit@{k}', 0):7.2f}%" for k in k_values)
        line += f"  {m['count']:5d}"
        print(line)

    # ---- Comparison with BM25 baseline ----
    bm25_metrics = {f"hit@{k}": [] for k in k_values}
    for result in results:
        gt = set(result["ground_truth"])
        bm25_orig = result.get("bm25_original", result["predicted"])
        for k in k_values:
            bm25_metrics[f"hit@{k}"].append(compute_hit_at_k(bm25_orig, gt, k))

    print(f"\nCOMPARISON (BM25 -> Graph-RAG):")
    for k in k_values:
        bm25_val = (
            sum(bm25_metrics[f"hit@{k}"])
            / max(len(bm25_metrics[f"hit@{k}"]), 1)
            * 100
        )
        grag_val = avg_overall[f"hit@{k}"]
        delta = grag_val - bm25_val
        direction = "+" if delta >= 0 else ""
        print(
            f"  Hit@{k}: {bm25_val:.2f}% -> {grag_val:.2f}% ({direction}{delta:.2f}%)"
        )

    # ---- Save results ----
    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    with open(pred_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nPredictions saved to {pred_path}")

    summary = {
        "method": "Graph-RAG baseline (in-context graph knowledge, NO LoRA)",
        "overall": avg_overall,
        "per_repo": avg_per_repo,
        "graph_context_coverage": {
            "has_cochange": graph_stats["has_cochange"],
            "has_import": graph_stats["has_import"],
            "has_any_graph": graph_stats["has_any_graph"],
            "total_candidates_scored": graph_stats["total_candidates_scored"],
            "cochange_pct": round(cc_pct, 2),
            "import_pct": round(imp_pct, 2),
            "any_graph_pct": round(any_pct, 2),
        },
        "config": {
            "model_path": args.model_path,
            "lora": "NONE (base model only)",
            "top_k": args.top_k,
            "max_seq_length": args.max_seq_length,
            "score_batch_size": args.score_batch_size,
            "max_neighbors": args.max_neighbors,
            "total_examples": total_evaluated,
            "cond_acc1_total": cond_acc1_total,
            "cond_acc1_correct": cond_acc1_correct,
        },
        "wall_clock_seconds": round(elapsed_total, 2),
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Graph-RAG baseline (in-context graph knowledge) on GREPO test set"
    )

    # Data paths
    parser.add_argument(
        "--test_data",
        default=DEFAULT_TEST_DATA,
        help="Test JSONL (GREPO format)",
    )
    parser.add_argument(
        "--bm25_candidates",
        default=DEFAULT_BM25_CANDIDATES,
        help="Precomputed BM25 top-K JSONL",
    )
    parser.add_argument(
        "--dep_graph_dir",
        default=DEFAULT_DEP_GRAPH_DIR,
        help="Directory containing *_rels.json dependency graph files",
    )
    parser.add_argument(
        "--train_data",
        default=DEFAULT_TRAIN_DATA,
        help="Training JSONL for building co-change graph",
    )
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
        help="Path to base Qwen2.5-7B-Instruct (NO LoRA)",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Results directory",
    )

    # Hardware
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU to use")

    # Evaluation params
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="How many BM25 candidates to rerank",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=768,
        help="Max tokens per prompt (longer due to graph context)",
    )
    parser.add_argument(
        "--score_batch_size",
        type=int,
        default=8,
        help="Batch size for scoring candidates (8 for 24GB RTX 4090)",
    )
    parser.add_argument(
        "--max_neighbors",
        type=int,
        default=10,
        help="Max graph neighbors to include per edge type in the prompt",
    )

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
