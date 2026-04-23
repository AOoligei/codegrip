"""
Evaluate file-level bug localization on GREPO benchmark.
Computes Hit@K for file-level predictions.

Usage:
    # Zero-shot with local model
    python src/eval/eval_grepo_file_level.py \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --test_data data/grepo_text/grepo_test.jsonl \
        --output_dir experiments/zeroshot_qwen25_7b

    # With custom prompt template
    python src/eval/eval_grepo_file_level.py \
        --model_path /path/to/model \
        --test_data data/grepo_text/grepo_test.jsonl \
        --prompt_template configs/prompt_zeroshot.txt
"""

import os
import json
import argparse
import time
from typing import List, Dict, Set
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_hit_at_k(predicted_files: List[str], ground_truth_files: Set[str], k: int) -> float:
    """Compute Hit@K: fraction of ground truth files in top-K predictions."""
    if not ground_truth_files:
        return 0.0
    top_k = set(predicted_files[:k])
    hits = len(top_k & ground_truth_files)
    return hits / len(ground_truth_files)


def parse_predicted_files(response: str) -> List[str]:
    """Extract file paths from model response.

    Handles various output formats:
    - Numbered lists: "1. path/to/file.py"
    - Bullet lists: "- path/to/file.py"
    - Plain list: "path/to/file.py"
    - Code blocks with file paths
    """
    files = []
    seen = set()

    for line in response.split("\n"):
        line = line.strip()
        # Skip empty lines
        if not line:
            continue

        # Remove numbering/bullets
        line = line.lstrip("0123456789.-) ")
        line = line.strip("`\"'")
        line = line.strip()

        # Check if it looks like a file path
        if "/" in line and line.endswith(".py"):
            # Extract the path part (might have explanation after)
            path = line.split()[0] if " " in line else line
            path = path.strip("`\"',:;")
            if path.endswith(".py") and path not in seen:
                files.append(path)
                seen.add(path)

    return files


def build_zeroshot_prompt(issue_text: str, repo: str) -> str:
    """Build zero-shot prompt for file-level bug localization."""
    return f"""You are a software engineer analyzing a bug report for the Python repository "{repo}".

Given the following bug report, predict which Python files (.py) in the repository need to be modified to fix this bug. List the most likely files in order of relevance.

Bug Report:
{issue_text}

List the Python file paths that most likely need to be modified to fix this bug. Output ONLY the file paths, one per line, most relevant first. Do not include explanations.

Files:"""


def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Run single inference with the model."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response


def evaluate(
    model,
    tokenizer,
    test_data: List[Dict],
    output_dir: str,
    max_examples: int = -1,
    k_values: List[int] = None,
):
    """Run evaluation on test set."""
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    os.makedirs(output_dir, exist_ok=True)

    results = []
    per_repo_metrics = {}

    total = len(test_data) if max_examples < 0 else min(max_examples, len(test_data))
    print(f"Evaluating {total} examples...")

    for idx, example in enumerate(test_data[:total]):
        if idx % 20 == 0:
            print(f"  [{idx}/{total}] Processing {example['repo']} #{example['issue_id']}...")

        # Build prompt (use override if available, e.g., filetree mode)
        prompt = example.get("_prompt_override") or build_zeroshot_prompt(example["issue_text"], example["repo"])

        # Run inference
        t0 = time.time()
        response = run_inference(model, tokenizer, prompt)
        elapsed = time.time() - t0

        # Parse predictions
        predicted_files = parse_predicted_files(response)

        # Ground truth
        gt_files = set(example["changed_py_files"])

        # Compute metrics
        metrics = {}
        for k in k_values:
            metrics[f"hit@{k}"] = compute_hit_at_k(predicted_files, gt_files, k)

        # Store result
        result = {
            "repo": example["repo"],
            "issue_id": example["issue_id"],
            "ground_truth": list(gt_files),
            "predicted": predicted_files[:20],
            "metrics": metrics,
            "inference_time": elapsed,
            "response": response[:500],
        }
        results.append(result)

        # Accumulate per-repo metrics
        repo = example["repo"]
        if repo not in per_repo_metrics:
            per_repo_metrics[repo] = {f"hit@{k}": [] for k in k_values}
            per_repo_metrics[repo]["count"] = 0
        per_repo_metrics[repo]["count"] += 1
        for k in k_values:
            per_repo_metrics[repo][f"hit@{k}"].append(metrics[f"hit@{k}"])

    # Aggregate metrics
    overall = {f"hit@{k}": [] for k in k_values}
    for r in results:
        for k in k_values:
            overall[f"hit@{k}"].append(r["metrics"][f"hit@{k}"])

    avg_overall = {k: sum(v) / len(v) * 100 for k, v in overall.items() if v}
    avg_per_repo = {}
    for repo, m in per_repo_metrics.items():
        avg_per_repo[repo] = {
            k: sum(v) / len(v) * 100 for k, v in m.items() if isinstance(v, list) and v
        }
        avg_per_repo[repo]["count"] = m["count"]

    # Print results
    print("\n" + "=" * 60)
    print("OVERALL RESULTS (file-level)")
    print("=" * 60)
    for k in k_values:
        print(f"  Hit@{k}: {avg_overall.get(f'hit@{k}', 0):.2f}%")

    print("\nPER-REPO RESULTS:")
    header = "Repo".ljust(15) + "".join(f"Hit@{k}".rjust(8) for k in k_values) + "  Count"
    print(header)
    print("-" * len(header))
    for repo in sorted(avg_per_repo.keys()):
        m = avg_per_repo[repo]
        line = repo.ljust(15) + "".join(f"{m.get(f'hit@{k}', 0):7.2f}%" for k in k_values)
        line += f"  {m['count']:5d}"
        print(line)

    # Save results
    with open(os.path.join(output_dir, "predictions.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "overall": avg_overall,
        "per_repo": avg_per_repo,
        "config": {
            "model": str(model.name_or_path) if hasattr(model, "name_or_path") else "unknown",
            "total_examples": total,
            "k_values": k_values,
        },
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return summary


def build_filetree_prompt(issue_text: str, repo: str, py_files: List[str]) -> str:
    """Build prompt with file tree context (matches SFT v1_filetree format)."""
    file_list = "\n".join(py_files[:300])
    return (
        f'You are analyzing a bug report for the Python repository "{repo}".\n\n'
        f"The repository contains the following Python files:\n{file_list}\n\n"
        f"Bug Report:\n{issue_text}\n\n"
        f"From the file list above, identify which Python files need to be modified "
        f"to fix this bug. Output file paths only, one per line, most relevant first."
    )


def build_graph_prompt(
    issue_text: str,
    repo: str,
    py_files: List[str],
    cochange_pairs: List = None,
    import_edges: List = None,
    max_files: int = 200,
    max_cochange: int = 20,
    max_imports: int = 20,
) -> str:
    """Build graph-conditioned prompt (matches SFT v3_graph format).

    Order: bug report + instruction FIRST, structural context MIDDLE, file list LAST.
    """
    import os as _os

    parts = []

    # Co-change clusters
    if cochange_pairs:
        cc_lines = []
        for item in cochange_pairs[:max_cochange]:
            if len(item) == 3:
                fa, fb, cnt = item
            else:
                fa, fb = item[:2]
                cnt = 2
            if cnt >= 2:
                cc_lines.append(f"  {fa} <-> {fb} ({cnt} times)")
        if cc_lines:
            parts.append(
                "Files frequently modified together (co-change history):\n"
                + "\n".join(cc_lines)
            )

    # Import relationships
    if import_edges:
        import_lines = [f"  {a} -> {b}" for a, b in import_edges[:max_imports]]
        if import_lines:
            parts.append(
                "Import dependencies (A -> B means A imports B):\n"
                + "\n".join(import_lines)
            )

    # Test-source mappings
    source_files = {}
    test_files_list = []
    for f in py_files:
        basename = _os.path.basename(f)
        if basename.startswith("test_") or "/tests/" in f or "/test/" in f:
            test_files_list.append(f)
        else:
            source_files[basename] = f
    tm_lines = []
    for tf in test_files_list[:100]:
        basename = _os.path.basename(tf)
        if basename.startswith("test_"):
            source_name = basename[5:]
            if source_name in source_files:
                tm_lines.append(f"  {tf} tests {source_files[source_name]}")
    if tm_lines:
        parts.append("Test-source file mappings:\n" + "\n".join(tm_lines[:10]))

    # File list (LAST — expendable)
    parts.append("Python files in this repository:\n" + "\n".join(py_files[:max_files]))

    structural_context = "\n\n".join(parts)

    return (
        f'You are analyzing a bug report for the Python repository "{repo}". '
        f"Identify which Python files need to be modified to fix this bug.\n\n"
        f"Bug Report:\n{issue_text}\n\n"
        f"{structural_context}\n\n"
        f"Output file paths only, one per line, most relevant first."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to base model")
    parser.add_argument("--lora_path", default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--test_data", required=True, help="Path to test JSONL")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--device", default="cuda:2", help="GPU device")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--prompt_mode", default="zeroshot",
                        choices=["zeroshot", "filetree", "graph"],
                        help="Prompt mode: zeroshot, filetree, or graph (with structural context)")
    parser.add_argument("--file_tree_dir", default=None,
                        help="Directory with per-repo file tree JSONs")
    parser.add_argument("--dep_graph_dir", default=None,
                        help="Directory with per-repo dep graph JSONs (for graph mode)")
    parser.add_argument("--train_data", default=None,
                        help="Training data JSONL for co-change computation (for graph mode)")
    args = parser.parse_args()

    # Load test data
    print(f"Loading test data from {args.test_data}...")
    with open(args.test_data) as f:
        test_data = [json.loads(l) for l in f]
    print(f"  Loaded {len(test_data)} examples")

    # Filter to examples with Python file changes
    test_data = [d for d in test_data if d["changed_py_files"]]
    print(f"  After filtering (has Python files): {len(test_data)} examples")

    # Load file trees if needed
    file_trees = {}
    if args.prompt_mode in ("filetree", "graph"):
        ft_dir = args.file_tree_dir or "data/file_trees"
        print(f"Loading file trees from {ft_dir}...")
        for fname in os.listdir(ft_dir):
            if fname.endswith(".json"):
                repo = fname.replace(".json", "")
                with open(os.path.join(ft_dir, fname)) as f:
                    file_trees[repo] = json.load(f)
        print(f"  Loaded trees for {len(file_trees)} repos")

    # Load structural data for graph mode
    cochange_index = {}
    import_index = {}
    if args.prompt_mode == "graph":
        # Build co-change index from training data
        train_path = args.train_data or "data/grepo_text/grepo_train.jsonl"
        print(f"Building co-change index from {train_path}...")
        from collections import defaultdict, Counter
        repo_pairs = defaultdict(Counter)
        with open(train_path) as f:
            for line in f:
                d = json.loads(line)
                files = sorted(d.get('changed_py_files', []))
                for i, fa in enumerate(files):
                    for fb in files[i+1:]:
                        repo_pairs[d['repo']][(fa, fb)] += 1
        for repo, pairs in repo_pairs.items():
            cochange_index[repo] = sorted(pairs.items(), key=lambda x: -x[1])
            cochange_index[repo] = [(fa, fb, cnt) for (fa, fb), cnt in cochange_index[repo]]
        print(f"  Co-change data for {len(cochange_index)} repos")

        # Build import index
        dep_dir = args.dep_graph_dir or "data/dep_graphs"
        print(f"Building import index from {dep_dir}...")
        for fname in os.listdir(dep_dir):
            if not fname.endswith("_rels.json"):
                continue
            repo = fname.replace("_rels.json", "")
            with open(os.path.join(dep_dir, fname)) as f:
                rels = json.load(f)
            edges = []
            for importer, imported_list in rels.get('file_imports', {}).items():
                for imported in imported_list:
                    if importer.endswith('.py') and imported.endswith('.py'):
                        edges.append((importer, imported))
            # Prioritize by PR frequency
            tree = file_trees.get(repo, {})
            pr_counts = tree.get("file_to_pr_count", {})
            edges.sort(key=lambda e: -(pr_counts.get(e[0], 0) + pr_counts.get(e[1], 0)))
            import_index[repo] = edges
        print(f"  Import data for {len(import_index)} repos")

    # Load model
    print(f"Loading model from {args.model_path}...")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype_map[args.dtype],
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter if provided
    if args.lora_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path)

    model.eval()
    print(f"  Model loaded on {args.device}")

    # Override prompt builder based on mode
    if args.prompt_mode == "filetree":
        for item in test_data:
            repo = item["repo"]
            tree = file_trees.get(repo, {})
            py_files = tree.get("py_files", [])
            item["_prompt_override"] = build_filetree_prompt(
                item["issue_text"], repo, py_files
            )
    elif args.prompt_mode == "graph":
        for item in test_data:
            repo = item["repo"]
            tree = file_trees.get(repo, {})
            py_files = tree.get("py_files", [])
            cc_pairs = cochange_index.get(repo, [])
            imp_edges = import_index.get(repo, [])
            item["_prompt_override"] = build_graph_prompt(
                item["issue_text"], repo, py_files,
                cochange_pairs=cc_pairs,
                import_edges=imp_edges,
            )

    # Run evaluation
    evaluate(model, tokenizer, test_data, args.output_dir, max_examples=args.max_examples)


if __name__ == "__main__":
    main()
