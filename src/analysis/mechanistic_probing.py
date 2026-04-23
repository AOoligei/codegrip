"""
Mechanistic probing / attention analysis for GREPO fine-tuned models.

Analyzes whether a LoRA-finetuned Qwen2.5-7B model has internalized graph
structure by examining attention patterns. Core question: do attention heads
attend more strongly to structurally-related files (co-changed, import-linked)
after fine-tuning compared to the base model?

Methodology:
  1. Load base model and LoRA-finetuned model.
  2. Feed in file-tree prompts (from GREPO eval data).
  3. Extract attention patterns from all layers.
  4. For each sample, identify token positions corresponding to each file path
     in the prompt.
  5. Compare attention from the last token (prediction position) to:
     - Tokens of files that co-change with ground truth files
     - Tokens of files that are import-linked to ground truth files
     - Tokens of random unrelated files
  6. Compute statistics (mean attention difference, paired t-test / Wilcoxon).
  7. Save results as JSON.

Usage:
    python src/analysis/mechanistic_probing.py \
        --model_path /path/to/Qwen2.5-Coder-7B-Instruct \
        --lora_path experiments/lora_v3_graph/final \
        --test_data data/grepo_text/grepo_test.jsonl \
        --file_tree_dir data/file_trees \
        --dep_graph_dir data/dep_graphs \
        --train_data data/grepo_text/grepo_train.jsonl \
        --output_dir experiments/mechanistic_probing \
        --num_samples 50 --gpu_id 0
"""

import os
import json
import argparse
import random
import time
import logging
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from scipy import stats

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data(path: str) -> List[Dict]:
    """Load test JSONL. Each line: {repo, issue_id, issue_text, changed_py_files, ...}."""
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if item.get("changed_py_files"):
                data.append(item)
    return data


def load_file_trees(directory: str) -> Dict[str, Dict]:
    """Load per-repo file trees. Returns {repo: {py_files: [...], ...}}."""
    trees = {}
    for fname in os.listdir(directory):
        if fname.endswith(".json"):
            repo = fname.replace(".json", "")
            with open(os.path.join(directory, fname)) as f:
                trees[repo] = json.load(f)
    return trees


def build_cochange_index(train_data_path: str) -> Dict[str, Dict[str, Set[str]]]:
    """Build co-change index from training data.

    Returns {repo: {file_a: {file_b, file_c, ...}}} where files co-changed
    in at least one training PR.
    """
    repo_cochange = defaultdict(lambda: defaultdict(set))
    with open(train_data_path) as f:
        for line in f:
            item = json.loads(line)
            repo = item["repo"]
            files = item.get("changed_py_files", [])
            for i, fa in enumerate(files):
                for fb in files[i + 1:]:
                    repo_cochange[repo][fa].add(fb)
                    repo_cochange[repo][fb].add(fa)
    # Convert inner defaultdicts to regular dicts
    return {repo: dict(mapping) for repo, mapping in repo_cochange.items()}


def build_import_index(dep_graph_dir: str) -> Dict[str, Dict[str, Set[str]]]:
    """Build import-neighbor index from dependency graphs.

    Returns {repo: {file: {import_neighbor_1, ...}}} (bidirectional).
    """
    repo_imports = defaultdict(lambda: defaultdict(set))
    for fname in os.listdir(dep_graph_dir):
        if not fname.endswith("_rels.json"):
            continue
        repo = fname.replace("_rels.json", "")
        with open(os.path.join(dep_graph_dir, fname)) as f:
            rels = json.load(f)
        for importer, imported_list in rels.get("file_imports", {}).items():
            for imported in imported_list:
                if importer.endswith(".py") and imported.endswith(".py"):
                    repo_imports[repo][importer].add(imported)
                    repo_imports[repo][imported].add(importer)
    return {repo: dict(mapping) for repo, mapping in repo_imports.items()}


# ---------------------------------------------------------------------------
# Prompt construction (mirrors eval_grepo_file_level.build_filetree_prompt)
# ---------------------------------------------------------------------------

def build_filetree_prompt(issue_text: str, repo: str, py_files: List[str]) -> str:
    """Build file-tree prompt matching SFT v1_filetree format."""
    file_list = "\n".join(py_files[:300])
    return (
        f'You are analyzing a bug report for the Python repository "{repo}".\n\n'
        f"The repository contains the following Python files:\n{file_list}\n\n"
        f"Bug Report:\n{issue_text}\n\n"
        f"From the file list above, identify which Python files need to be modified "
        f"to fix this bug. Output file paths only, one per line, most relevant first."
    )


# ---------------------------------------------------------------------------
# Token-to-file mapping
# ---------------------------------------------------------------------------

def find_file_token_spans(
    token_ids: List[int],
    tokenizer,
    py_files: List[str],
) -> Dict[str, List[int]]:
    """Map each file path to its token positions within the input.

    Strategy:
      - For each file path, tokenize it independently to get its token ids.
      - Scan the full token sequence for exact sub-sequence matches.
      - Return {file_path: [token_idx_0, token_idx_1, ...]} for files found.

    We search for each file as a standalone substring. Because the file list
    is formatted as newline-separated paths, each path is typically preceded
    by a newline token.
    """
    file_to_positions: Dict[str, List[int]] = {}

    for fpath in py_files:
        # Tokenize the file path in isolation (no special tokens).
        # Prepend a newline because that is the delimiter in the prompt.
        fpath_tokens = tokenizer.encode("\n" + fpath, add_special_tokens=False)
        if not fpath_tokens:
            continue

        span_len = len(fpath_tokens)
        positions = []

        for i in range(len(token_ids) - span_len + 1):
            if token_ids[i : i + span_len] == fpath_tokens:
                # Record positions of the actual file tokens (skip leading \n token)
                # The leading \n token is fpath_tokens[0]; the file itself starts at [1:]
                # but we include all positions for more signal.
                positions.extend(range(i, i + span_len))

        if positions:
            file_to_positions[fpath] = positions

    return file_to_positions


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def extract_attention_to_files(
    model,
    tokenizer,
    prompt: str,
    py_files: List[str],
    max_length: int = 4096,
    device: str = "cuda:0",
) -> Optional[Dict]:
    """Run a forward pass and extract attention from the last token to file tokens.

    Returns:
        {
            "file_positions": {file_path: [positions]},
            "attention_per_layer": np.ndarray of shape (num_layers, num_heads, num_files)
                where each entry is the summed attention from the last token to all
                positions of that file, for each (layer, head).
            "file_order": [file_path_0, file_path_1, ...]  (matches axis-2 order)
        }
    Returns None if tokenization exceeds max_length or no files are found.
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    if seq_len >= max_length:
        logger.warning("Prompt truncated to %d tokens; results may be unreliable.", max_length)

    # Map file paths to token positions
    token_id_list = input_ids[0].tolist()
    file_positions = find_file_token_spans(token_id_list, tokenizer, py_files)

    if not file_positions:
        return None

    # Forward pass with attention
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            use_cache=False,
        )

    # outputs.attentions is a tuple of (num_layers,) tensors
    # each tensor shape: (batch=1, num_heads, seq_len, seq_len)
    attentions = outputs.attentions  # tuple of length num_layers

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    file_order = sorted(file_positions.keys())
    num_files = len(file_order)

    # Extract attention from last token to each file's tokens
    # Shape: (num_layers, num_heads, num_files)
    attn_to_files = np.zeros((num_layers, num_heads, num_files), dtype=np.float32)

    last_tok_idx = seq_len - 1

    for layer_idx in range(num_layers):
        # (1, num_heads, seq_len, seq_len) -> (num_heads, seq_len)
        # We want attn[head, last_tok, :] for each head
        layer_attn = attentions[layer_idx][0, :, last_tok_idx, :]  # (num_heads, seq_len)
        layer_attn_np = layer_attn.float().cpu().numpy()

        for file_idx, fpath in enumerate(file_order):
            positions = file_positions[fpath]
            # Sum attention over all token positions belonging to this file
            attn_to_files[layer_idx, :, file_idx] = layer_attn_np[:, positions].sum(axis=1)

    # Free GPU memory from attention tensors
    del outputs, attentions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "file_positions": file_positions,
        "attention_per_layer": attn_to_files,
        "file_order": file_order,
        "seq_len": seq_len,
    }


# ---------------------------------------------------------------------------
# Per-sample analysis: classify files into structural categories
# ---------------------------------------------------------------------------

def classify_files(
    file_order: List[str],
    gt_files: Set[str],
    cochange_neighbors: Dict[str, Set[str]],
    import_neighbors: Dict[str, Set[str]],
) -> Dict[str, List[int]]:
    """Classify each file in file_order into structural categories.

    Categories (mutually exclusive, assigned by priority):
      1. "ground_truth" - the file is one of the actually changed files
      2. "cochange"     - the file co-changed with a GT file in training data
      3. "import"       - the file is an import-neighbor of a GT file
      4. "random"       - none of the above

    Returns {category: [indices_into_file_order]}.
    """
    # Compute the union of co-change and import neighbors for all GT files
    cochange_set: Set[str] = set()
    import_set: Set[str] = set()
    for gt_f in gt_files:
        cochange_set |= cochange_neighbors.get(gt_f, set())
        import_set |= import_neighbors.get(gt_f, set())
    # Remove GT files themselves from neighbor sets
    cochange_set -= gt_files
    import_set -= gt_files

    categories: Dict[str, List[int]] = {
        "ground_truth": [],
        "cochange": [],
        "import": [],
        "random": [],
    }

    for idx, fpath in enumerate(file_order):
        if fpath in gt_files:
            categories["ground_truth"].append(idx)
        elif fpath in cochange_set:
            categories["cochange"].append(idx)
        elif fpath in import_set:
            categories["import"].append(idx)
        else:
            categories["random"].append(idx)

    return categories


def compute_category_attention(
    attn_to_files: np.ndarray,
    categories: Dict[str, List[int]],
) -> Dict[str, np.ndarray]:
    """Compute mean attention per category, averaged over heads and files.

    Args:
        attn_to_files: (num_layers, num_heads, num_files)
        categories: {category: [file_indices]}

    Returns:
        {category: np.ndarray of shape (num_layers,)} -- mean attention
        across all heads and all files in that category, per layer.
    """
    result = {}
    for cat, indices in categories.items():
        if not indices:
            result[cat] = None
            continue
        # (num_layers, num_heads, len(indices))
        subset = attn_to_files[:, :, indices]
        # Mean over heads and files -> (num_layers,)
        result[cat] = subset.mean(axis=(1, 2))
    return result


def compute_per_head_category_attention(
    attn_to_files: np.ndarray,
    categories: Dict[str, List[int]],
) -> Dict[str, np.ndarray]:
    """Compute mean attention per category per head.

    Returns {category: np.ndarray of shape (num_layers, num_heads)}.
    """
    result = {}
    for cat, indices in categories.items():
        if not indices:
            result[cat] = None
            continue
        subset = attn_to_files[:, :, indices]
        result[cat] = subset.mean(axis=2)  # (num_layers, num_heads)
    return result


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def paired_comparison(
    samples_a: List[float],
    samples_b: List[float],
    test_name: str = "wilcoxon",
) -> Dict:
    """Compare two paired sample distributions.

    Uses Wilcoxon signed-rank test (non-parametric) or paired t-test.
    """
    a = np.array(samples_a)
    b = np.array(samples_b)
    diff = a - b
    n_valid = np.sum(diff != 0)

    result = {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff)),
        "n_samples": len(a),
        "n_nonzero_diffs": int(n_valid),
    }

    if n_valid < 5:
        result["test"] = "insufficient_samples"
        result["p_value"] = None
        result["statistic"] = None
        return result

    if test_name == "wilcoxon":
        try:
            stat, p = stats.wilcoxon(a, b, alternative="greater")
            result["test"] = "wilcoxon_signed_rank"
        except ValueError:
            # Fallback if all differences are zero
            stat, p = float("nan"), 1.0
            result["test"] = "wilcoxon_failed"
    else:
        stat, p = stats.ttest_rel(a, b)
        # Convert to one-sided (a > b)
        p = p / 2 if stat > 0 else 1 - p / 2
        result["test"] = "paired_ttest"

    result["statistic"] = float(stat) if not np.isnan(stat) else None
    result["p_value"] = float(p) if not np.isnan(p) else None

    return result


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def run_analysis_for_model(
    model,
    tokenizer,
    samples: List[Dict],
    file_trees: Dict[str, Dict],
    cochange_index: Dict[str, Dict[str, Set[str]]],
    import_index: Dict[str, Dict[str, Set[str]]],
    device: str = "cuda:0",
    max_length: int = 4096,
) -> Dict:
    """Run attention analysis over all samples for a single model.

    Returns per-sample and aggregated results.
    """
    per_sample_results = []
    # For aggregate statistics: per-sample mean attention to each category
    # We aggregate the "average across all layers" value per sample.
    agg_by_category = defaultdict(list)  # {category: [per_sample_mean_attn]}

    for idx, sample in enumerate(samples):
        repo = sample["repo"]
        issue_id = sample["issue_id"]
        gt_files = set(sample["changed_py_files"])

        tree = file_trees.get(repo, {})
        py_files = tree.get("py_files", [])

        if not py_files:
            logger.warning(
                "[%d/%d] %s #%s: no file tree, skipping.",
                idx + 1, len(samples), repo, issue_id,
            )
            continue

        prompt = build_filetree_prompt(sample["issue_text"], repo, py_files)

        logger.info(
            "[%d/%d] %s #%s (%d py_files, %d gt_files)...",
            idx + 1, len(samples), repo, issue_id, len(py_files), len(gt_files),
        )

        t0 = time.time()
        result = extract_attention_to_files(
            model, tokenizer, prompt, py_files,
            max_length=max_length, device=device,
        )
        elapsed = time.time() - t0

        if result is None:
            logger.warning(
                "  -> No file tokens found in tokenized prompt, skipping."
            )
            continue

        file_order = result["file_order"]
        attn = result["attention_per_layer"]  # (L, H, F)

        # Classify files
        cochange_neighbors = cochange_index.get(repo, {})
        import_neighbors = import_index.get(repo, {})
        categories = classify_files(file_order, gt_files, cochange_neighbors, import_neighbors)

        cat_counts = {cat: len(idxs) for cat, idxs in categories.items()}
        logger.info(
            "  -> %d files mapped, categories: %s, %.1fs",
            len(file_order), cat_counts, elapsed,
        )

        # Skip if we lack enough files in comparison categories
        if not categories["random"]:
            logger.warning("  -> No random files for comparison, skipping.")
            continue

        # Compute per-layer mean attention by category
        cat_attn = compute_category_attention(attn, categories)
        per_head_attn = compute_per_head_category_attention(attn, categories)

        # Store per-sample summary: mean attention across ALL layers
        sample_result = {
            "repo": repo,
            "issue_id": issue_id,
            "num_gt_files": len(gt_files),
            "num_files_mapped": len(file_order),
            "category_counts": cat_counts,
            "seq_len": result["seq_len"],
            "inference_time_s": round(elapsed, 2),
            "per_layer_mean_attn": {},  # {cat: [per_layer_float]}
        }

        for cat in ["ground_truth", "cochange", "import", "random"]:
            if cat_attn[cat] is not None:
                layer_means = cat_attn[cat].tolist()
                sample_result["per_layer_mean_attn"][cat] = [
                    round(v, 8) for v in layer_means
                ]
                overall_mean = float(np.mean(cat_attn[cat]))
                agg_by_category[cat].append(overall_mean)
            else:
                sample_result["per_layer_mean_attn"][cat] = None

        # Identify top-attending heads for structural categories
        # (layer, head) pairs where cochange/import attention most exceeds random
        if per_head_attn["random"] is not None:
            random_attn = per_head_attn["random"]  # (L, H)
            for cat in ["cochange", "import", "ground_truth"]:
                if per_head_attn[cat] is not None:
                    diff = per_head_attn[cat] - random_attn  # (L, H)
                    top_k = 5
                    flat_indices = np.argsort(diff.ravel())[::-1][:top_k]
                    top_heads = []
                    for fi in flat_indices:
                        layer = int(fi // diff.shape[1])
                        head = int(fi % diff.shape[1])
                        top_heads.append({
                            "layer": layer,
                            "head": head,
                            "diff": round(float(diff[layer, head]), 8),
                            "cat_attn": round(float(per_head_attn[cat][layer, head]), 8),
                            "random_attn": round(float(random_attn[layer, head]), 8),
                        })
                    sample_result[f"top_heads_{cat}_vs_random"] = top_heads

        per_sample_results.append(sample_result)

    return {
        "per_sample": per_sample_results,
        "agg_by_category": {k: v for k, v in agg_by_category.items()},
    }


def aggregate_results(
    base_results: Dict,
    finetuned_results: Dict,
) -> Dict:
    """Aggregate and compare base vs. finetuned model results.

    Computes:
      - Per-category mean attention for each model
      - Base vs. finetuned differences
      - Statistical tests: does finetuning increase attention to structural neighbors?
    """
    summary = {
        "base_model": {},
        "finetuned_model": {},
        "comparisons": {},
    }

    # Per-model category means
    for model_key, results in [("base_model", base_results), ("finetuned_model", finetuned_results)]:
        agg = results["agg_by_category"]
        for cat, values in agg.items():
            summary[model_key][cat] = {
                "mean_attention": float(np.mean(values)) if values else None,
                "std_attention": float(np.std(values)) if values else None,
                "n_samples": len(values),
            }

    # Comparison 1: Within each model, structural vs random
    for model_key, results in [("base_model", base_results), ("finetuned_model", finetuned_results)]:
        agg = results["agg_by_category"]
        random_vals = agg.get("random", [])
        comparisons = {}
        for cat in ["ground_truth", "cochange", "import"]:
            cat_vals = agg.get(cat, [])
            if len(cat_vals) >= 5 and len(random_vals) >= 5:
                # Align by sample index (they correspond to the same samples)
                n = min(len(cat_vals), len(random_vals))
                comparisons[f"{cat}_vs_random"] = paired_comparison(
                    cat_vals[:n], random_vals[:n], test_name="wilcoxon"
                )
        summary[f"{model_key}_internal"] = comparisons

    # Comparison 2: Finetuned vs base for each category
    # This requires samples to be aligned (same samples in same order)
    base_agg = base_results["agg_by_category"]
    ft_agg = finetuned_results["agg_by_category"]

    for cat in ["ground_truth", "cochange", "import", "random"]:
        base_vals = base_agg.get(cat, [])
        ft_vals = ft_agg.get(cat, [])
        if len(base_vals) >= 5 and len(ft_vals) >= 5:
            n = min(len(base_vals), len(ft_vals))
            summary["comparisons"][f"ft_vs_base_{cat}"] = paired_comparison(
                ft_vals[:n], base_vals[:n], test_name="wilcoxon"
            )

    # Comparison 3: Does finetuning increase the *gap* between structural and random?
    for cat in ["cochange", "import"]:
        base_cat = base_agg.get(cat, [])
        base_rand = base_agg.get("random", [])
        ft_cat = ft_agg.get(cat, [])
        ft_rand = ft_agg.get("random", [])

        n = min(len(base_cat), len(base_rand), len(ft_cat), len(ft_rand))
        if n >= 5:
            base_gap = [base_cat[i] - base_rand[i] for i in range(n)]
            ft_gap = [ft_cat[i] - ft_rand[i] for i in range(n)]
            summary["comparisons"][f"gap_increase_{cat}"] = paired_comparison(
                ft_gap, base_gap, test_name="wilcoxon"
            )

    return summary


# ---------------------------------------------------------------------------
# Per-layer breakdown for visualization
# ---------------------------------------------------------------------------

def compute_per_layer_aggregate(results: Dict) -> Dict[str, List[Optional[float]]]:
    """Compute mean attention per layer per category, averaged across samples.

    Returns {category: [mean_at_layer_0, mean_at_layer_1, ...]}.
    """
    # Collect per-layer values across samples
    layer_values = defaultdict(lambda: defaultdict(list))  # {cat: {layer: [values]}}

    for sample in results["per_sample"]:
        for cat, layer_means in sample["per_layer_mean_attn"].items():
            if layer_means is None:
                continue
            for layer_idx, val in enumerate(layer_means):
                layer_values[cat][layer_idx].append(val)

    # Average
    output = {}
    for cat in ["ground_truth", "cochange", "import", "random"]:
        if cat not in layer_values:
            output[cat] = None
            continue
        max_layer = max(layer_values[cat].keys()) + 1
        means = []
        for l in range(max_layer):
            vals = layer_values[cat].get(l, [])
            means.append(round(float(np.mean(vals)), 8) if vals else None)
        output[cat] = means

    return output


# ---------------------------------------------------------------------------
# Top heads discovery
# ---------------------------------------------------------------------------

def find_top_structural_heads(results: Dict, top_k: int = 20) -> List[Dict]:
    """Find (layer, head) pairs that most consistently attend to structural neighbors.

    Aggregates per-sample top-head rankings and returns the most frequent ones.
    """
    head_scores = Counter()  # {(layer, head): cumulative_score}

    for cat in ["cochange", "import", "ground_truth"]:
        key = f"top_heads_{cat}_vs_random"
        for sample in results["per_sample"]:
            if key not in sample:
                continue
            for rank, entry in enumerate(sample[key]):
                # Weight by inverse rank (top-1 gets most credit)
                weight = 1.0 / (rank + 1)
                head_scores[(entry["layer"], entry["head"], cat)] += weight

    # Sort by score
    ranked = sorted(head_scores.items(), key=lambda x: -x[1])[:top_k]

    return [
        {
            "layer": layer,
            "head": head,
            "category": cat,
            "aggregate_score": round(score, 4),
        }
        for (layer, head, cat), score in ranked
    ]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_path: str,
    lora_path: Optional[str],
    device: str,
    dtype: torch.dtype = torch.bfloat16,
):
    """Load model (optionally with LoRA) and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer from %s...", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model from %s (dtype=%s)...", model_path, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    if lora_path:
        from peft import PeftModel
        logger.info("Loading LoRA adapter from %s...", lora_path)
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    logger.info("Model loaded on %s.", device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mechanistic probing: attention analysis for GREPO fine-tuned models."
    )
    parser.add_argument("--model_path", required=True, help="Path to base model (e.g. Qwen2.5-Coder-7B-Instruct)")
    parser.add_argument("--lora_path", default=None, help="Path to LoRA adapter for finetuned model")
    parser.add_argument("--test_data", required=True, help="Path to test JSONL (grepo_test.jsonl)")
    parser.add_argument("--file_tree_dir", default="data/file_trees", help="Directory with per-repo file tree JSONs")
    parser.add_argument("--dep_graph_dir", default="data/dep_graphs", help="Directory with per-repo dep graph JSONs")
    parser.add_argument("--train_data", default="data/grepo_text/grepo_train.jsonl",
                        help="Training data JSONL for co-change computation")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of test samples to analyze")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length for tokenization")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"],
                        help="Model dtype")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Config: %s", json.dumps(config, indent=2))

    device = f"cuda:{args.gpu_id}"
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    logger.info("Loading test data from %s...", args.test_data)
    test_data = load_test_data(args.test_data)
    logger.info("  Loaded %d examples with Python file changes.", len(test_data))

    logger.info("Loading file trees from %s...", args.file_tree_dir)
    file_trees = load_file_trees(args.file_tree_dir)
    logger.info("  Loaded trees for %d repos.", len(file_trees))

    # Filter test data to repos with file trees
    test_data = [d for d in test_data if d["repo"] in file_trees]
    logger.info("  After filtering (has file tree): %d examples.", len(test_data))

    logger.info("Building co-change index from %s...", args.train_data)
    cochange_index = build_cochange_index(args.train_data)
    logger.info("  Co-change data for %d repos.", len(cochange_index))

    logger.info("Building import index from %s...", args.dep_graph_dir)
    import_index = build_import_index(args.dep_graph_dir)
    logger.info("  Import data for %d repos.", len(import_index))

    # Sample
    if args.num_samples > 0 and args.num_samples < len(test_data):
        random.shuffle(test_data)
        test_data = test_data[:args.num_samples]
    logger.info("Analyzing %d samples.", len(test_data))

    # -----------------------------------------------------------------------
    # Run analysis: base model
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Phase 1: Base model analysis")
    logger.info("=" * 70)

    base_model, tokenizer = load_model_and_tokenizer(
        args.model_path, lora_path=None, device=device, dtype=dtype,
    )

    t0 = time.time()
    base_results = run_analysis_for_model(
        base_model, tokenizer, test_data, file_trees,
        cochange_index, import_index,
        device=device, max_length=args.max_length,
    )
    base_elapsed = time.time() - t0
    logger.info("Base model analysis done in %.1fs (%d samples).",
                base_elapsed, len(base_results["per_sample"]))

    # Free base model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Run analysis: finetuned model
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Phase 2: Finetuned model analysis")
    logger.info("=" * 70)

    if args.lora_path:
        ft_model, tokenizer = load_model_and_tokenizer(
            args.model_path, lora_path=args.lora_path, device=device, dtype=dtype,
        )
    else:
        logger.warning("No --lora_path provided; skipping finetuned model analysis.")
        ft_model = None

    if ft_model is not None:
        t0 = time.time()
        ft_results = run_analysis_for_model(
            ft_model, tokenizer, test_data, file_trees,
            cochange_index, import_index,
            device=device, max_length=args.max_length,
        )
        ft_elapsed = time.time() - t0
        logger.info("Finetuned model analysis done in %.1fs (%d samples).",
                     ft_elapsed, len(ft_results["per_sample"]))

        del ft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        ft_results = {"per_sample": [], "agg_by_category": {}}

    # -----------------------------------------------------------------------
    # Aggregate and compare
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Phase 3: Aggregation and statistical comparison")
    logger.info("=" * 70)

    # Per-layer breakdown
    base_per_layer = compute_per_layer_aggregate(base_results)
    ft_per_layer = compute_per_layer_aggregate(ft_results) if ft_model is not None or ft_results["per_sample"] else {}

    # Top structural heads
    base_top_heads = find_top_structural_heads(base_results)
    ft_top_heads = find_top_structural_heads(ft_results) if ft_results["per_sample"] else []

    # Statistical comparison
    if ft_results["per_sample"]:
        comparison_summary = aggregate_results(base_results, ft_results)
    else:
        # Base-only internal comparisons
        comparison_summary = {
            "base_model": {},
            "finetuned_model": {},
            "comparisons": {},
        }
        agg = base_results["agg_by_category"]
        for cat, values in agg.items():
            comparison_summary["base_model"][cat] = {
                "mean_attention": float(np.mean(values)) if values else None,
                "std_attention": float(np.std(values)) if values else None,
                "n_samples": len(values),
            }
        random_vals = agg.get("random", [])
        for cat in ["ground_truth", "cochange", "import"]:
            cat_vals = agg.get(cat, [])
            if len(cat_vals) >= 5 and len(random_vals) >= 5:
                n = min(len(cat_vals), len(random_vals))
                comparison_summary["comparisons"][f"base_{cat}_vs_random"] = paired_comparison(
                    cat_vals[:n], random_vals[:n], test_name="wilcoxon"
                )

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    for model_key in ["base_model", "finetuned_model"]:
        if model_key not in comparison_summary or not comparison_summary[model_key]:
            continue
        logger.info("\n%s:", model_key.upper().replace("_", " "))
        for cat, stats_dict in comparison_summary[model_key].items():
            if stats_dict["mean_attention"] is not None:
                logger.info(
                    "  %-15s  mean=%.6f  std=%.6f  n=%d",
                    cat, stats_dict["mean_attention"],
                    stats_dict["std_attention"], stats_dict["n_samples"],
                )

    logger.info("\nSTATISTICAL COMPARISONS:")
    for comp_name, comp_result in comparison_summary.get("comparisons", {}).items():
        p_str = f"p={comp_result['p_value']:.4f}" if comp_result.get("p_value") is not None else "p=N/A"
        logger.info(
            "  %-35s  diff=%.6f  %s  (n=%d)",
            comp_name,
            comp_result.get("mean_diff", 0),
            p_str,
            comp_result.get("n_samples", 0),
        )

    internal_keys = [k for k in comparison_summary if k.endswith("_internal")]
    for ik in internal_keys:
        logger.info("\n%s:", ik.upper().replace("_", " "))
        for comp_name, comp_result in comparison_summary[ik].items():
            p_str = f"p={comp_result['p_value']:.4f}" if comp_result.get("p_value") is not None else "p=N/A"
            logger.info(
                "  %-30s  diff=%.6f  %s",
                comp_name, comp_result.get("mean_diff", 0), p_str,
            )

    if base_top_heads:
        logger.info("\nTOP STRUCTURAL HEADS (base model):")
        for h in base_top_heads[:10]:
            logger.info(
                "  layer=%2d  head=%2d  cat=%-12s  score=%.4f",
                h["layer"], h["head"], h["category"], h["aggregate_score"],
            )

    if ft_top_heads:
        logger.info("\nTOP STRUCTURAL HEADS (finetuned model):")
        for h in ft_top_heads[:10]:
            logger.info(
                "  layer=%2d  head=%2d  cat=%-12s  score=%.4f",
                h["layer"], h["head"], h["category"], h["aggregate_score"],
            )

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output = {
        "config": config,
        "num_samples_base": len(base_results["per_sample"]),
        "num_samples_finetuned": len(ft_results["per_sample"]),
        "comparison_summary": comparison_summary,
        "per_layer_attention": {
            "base_model": base_per_layer,
            "finetuned_model": ft_per_layer,
        },
        "top_structural_heads": {
            "base_model": base_top_heads,
            "finetuned_model": ft_top_heads,
        },
        "base_per_sample": base_results["per_sample"],
        "finetuned_per_sample": ft_results["per_sample"],
    }

    output_path = os.path.join(args.output_dir, "mechanistic_probing_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("\nFull results saved to %s", output_path)

    # Also save a compact summary
    compact = {
        "config": config,
        "comparison_summary": comparison_summary,
        "top_structural_heads": {
            "base_model": base_top_heads[:10],
            "finetuned_model": ft_top_heads[:10],
        },
    }
    compact_path = os.path.join(args.output_dir, "summary.json")
    with open(compact_path, "w") as f:
        json.dump(compact, f, indent=2, default=str)
    logger.info("Compact summary saved to %s", compact_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
