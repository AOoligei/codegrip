"""
Attention Probe: Direct evidence that the path-only reranker ignores code tokens.

Loads the path-only reranker (Qwen2.5-7B + LoRA from rankft_runB_graph) in 4-bit,
constructs code-centric prompts (path + code), extracts last-layer attention at the
final token position, and reports what fraction of attention goes to each token type
(path, code, issue, template).

Usage:
    python scripts/attention_probe.py --gpu_id 0
"""

import argparse
import json
import os
import sys
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Deterministic
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ============================================================
# Config
# ============================================================
DEFAULT_MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
DEFAULT_LORA_PATH = "/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/best"
DEFAULT_TEST_DATA = "data/grepo_text/grepo_test.jsonl"
DEFAULT_CANDIDATES = "data/rankft/merged_bm25_exp6_candidates.jsonl"
DEFAULT_REPO_DIR = "data/repos"

# Prompt template (from train_rankft_code_centric.py)
PROMPT_PREFIX = (
    "Given the bug report, analyze the code and determine if this file "
    "likely needs modification.\n\n"
    "Bug Report: "
)
PROMPT_MID_FILE = "\n\nFile: "
PROMPT_MID_CODE = "\n\nCode (key sections):\n"
PROMPT_SUFFIX = (
    "\n\nBased on the code content and structure, is this file likely to need "
    "modification?\nAnswer:"
)


# ============================================================
# Code extraction (from train_rankft_code_centric.py)
# ============================================================
import ast

def _extract_signatures(source: str) -> List[str]:
    sigs = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return sigs
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            args_str = ast.get_source_segment(source, node.args)
            if args_str is None:
                lines = source.split('\n')
                if node.lineno <= len(lines):
                    sigs.append(lines[node.lineno - 1].strip())
            else:
                sigs.append(f"{prefix} {node.name}({args_str}):")
        elif isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                seg = ast.get_source_segment(source, base)
                if seg:
                    bases.append(seg)
            base_str = f"({', '.join(bases)})" if bases else ""
            sigs.append(f"class {node.name}{base_str}:")
    return sigs


def extract_code_content(repo_dir: str, repo: str, filepath: str,
                         head_lines: int = 50, max_chars: int = 1500) -> str:
    full_path = os.path.join(repo_dir, repo, filepath)
    try:
        with open(full_path, 'r', errors='replace') as f:
            full_source = f.read()
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return "# (file not available)"

    lines = full_source.split('\n')
    head = '\n'.join(lines[:head_lines])

    sigs = _extract_signatures(full_source)
    extra_sigs = [s.strip() for s in sigs if s.strip() not in head]

    if extra_sigs:
        sig_block = "\n# ... (signatures from rest of file)\n" + '\n'.join(extra_sigs)
        content = head + sig_block
    else:
        content = head

    if len(content) > max_chars:
        content = content[:max_chars] + "\n# ... (truncated)"

    return content


# ============================================================
# Build prompt with known segment boundaries
# ============================================================

def build_prompt_with_segments(
    issue_text: str,
    candidate_path: str,
    code_content: str,
    tokenizer,
    max_seq_length: int = 1024,
) -> Tuple[str, Dict[str, Tuple[int, int]]]:
    """Build prompt and return (prompt_str, token_span_dict).

    token_span_dict maps token type -> (start_token_idx, end_token_idx) in the
    tokenized prompt (exclusive end).

    We build the prompt piece by piece, tokenize each piece, and record spans.
    """
    # Pieces in order
    pieces = [
        ("template", PROMPT_PREFIX),
        ("issue", issue_text),
        ("template", PROMPT_MID_FILE),
        ("path", candidate_path),
        ("template", PROMPT_MID_CODE),
        ("code", code_content),
        ("template", PROMPT_SUFFIX),
    ]

    # Build full prompt string
    full_prompt = "".join(p[1] for p in pieces)

    # Tokenize the full prompt
    full_ids = tokenizer.encode(full_prompt, add_special_tokens=False)

    # Truncate code if needed (same logic as train_rankft_code_centric.py)
    if len(full_ids) > max_seq_length:
        # Measure non-code token cost
        non_code = PROMPT_PREFIX + issue_text + PROMPT_MID_FILE + candidate_path + PROMPT_MID_CODE + PROMPT_SUFFIX
        non_code_len = len(tokenizer.encode(non_code, add_special_tokens=False))
        code_budget = max_seq_length - non_code_len - 5
        if code_budget <= 0:
            code_content = "# (truncated)"
        else:
            code_ids = tokenizer.encode(code_content, add_special_tokens=False)
            if len(code_ids) > code_budget:
                code_content = tokenizer.decode(code_ids[:code_budget])

        # Rebuild pieces with truncated code
        pieces = [
            ("template", PROMPT_PREFIX),
            ("issue", issue_text),
            ("template", PROMPT_MID_FILE),
            ("path", candidate_path),
            ("template", PROMPT_MID_CODE),
            ("code", code_content),
            ("template", PROMPT_SUFFIX),
        ]
        full_prompt = "".join(p[1] for p in pieces)
        full_ids = tokenizer.encode(full_prompt, add_special_tokens=False)

    # Now assign each token to a type.
    # Strategy: tokenize each piece individually (without special tokens),
    # accumulate, and assign labels. The concatenation of piece tokenizations
    # may differ from tokenizing the whole string due to tokenizer boundary
    # effects. So we use a character-offset approach instead.

    # Character offset -> token type
    char_offsets = []
    offset = 0
    for ttype, text in pieces:
        char_offsets.append((offset, offset + len(text), ttype))
        offset += len(text)

    # Get character span for each token using offset_mapping
    encoding = tokenizer(full_prompt, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"]

    # Assign each token a type based on its character span
    token_types = []
    for tok_start, tok_end in offset_mapping:
        # Find which piece this token's midpoint falls in
        mid = (tok_start + tok_end) / 2.0 if tok_end > tok_start else tok_start
        assigned = "template"
        for seg_start, seg_end, ttype in char_offsets:
            if seg_start <= mid < seg_end:
                assigned = ttype
                break
        token_types.append(assigned)

    return full_prompt, token_ids, token_types


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Attention probe for CodeGRIP reranker")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--lora_path", default=DEFAULT_LORA_PATH)
    parser.add_argument("--test_data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--repo_dir", default=DEFAULT_REPO_DIR)
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--output_json", default="results/attention_probe_results.json")
    args = parser.parse_args()

    # Resolve relative paths from project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    device = f"cuda:{args.gpu_id}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda:0"  # logical device after CUDA_VISIBLE_DEVICES

    # ---- Load test data ----
    print("Loading test data...")
    test_data = []
    with open(args.test_data) as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"  Total test examples: {len(test_data)}")

    # ---- Load candidates ----
    print("Loading candidates...")
    candidates_by_key = {}
    with open(args.candidates) as f:
        for line in f:
            d = json.loads(line)
            key = (d["repo"], d["issue_id"])
            candidates_by_key[key] = d["candidates"]

    # ---- Sample examples ----
    # For each example, pick one GT file (positive) and one non-GT file (negative)
    # to get a mix. We just need forward passes, label doesn't matter.
    random.seed(42)
    sampled = random.sample(test_data, min(args.num_examples, len(test_data)))
    print(f"  Sampled {len(sampled)} examples for analysis")

    # ---- Load model in 4-bit ----
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model in 4-bit from {args.model_path}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )

    print(f"Loading LoRA from {args.lora_path}...")
    model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=False)
    model.eval()

    # ---- Run attention analysis ----
    print("\nRunning attention analysis...")

    # Per-example results
    all_fractions = []  # list of dicts: {type: fraction}
    skipped = 0

    for i, example in enumerate(sampled):
        repo = example["repo"]
        issue_id = example["issue_id"]
        issue_text = example["issue_text"]
        gt_files = set(example["changed_py_files"])

        key = (repo, issue_id)
        cands = candidates_by_key.get(key, [])
        if not cands:
            skipped += 1
            continue

        # Pick one candidate: prefer a GT file if in candidate list, else first
        chosen = None
        for c in cands:
            if c in gt_files:
                chosen = c
                break
        if chosen is None:
            chosen = cands[0]

        # Extract code
        code_content = extract_code_content(args.repo_dir, repo, chosen)

        # Build prompt and get token types
        prompt_str, token_ids, token_types = build_prompt_with_segments(
            issue_text, chosen, code_content, tokenizer, args.max_seq_length
        )

        # Sanity: skip if too few tokens
        if len(token_ids) < 5:
            skipped += 1
            continue

        # Forward pass with attention output
        input_ids = torch.tensor([token_ids], device=device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_attentions=True,
                use_cache=False,
            )

        # Extract last layer attention: shape (1, num_heads, seq_len, seq_len)
        last_layer_attn = outputs.attentions[-1]  # (1, H, S, S)
        # Average across heads, take the last token's attention over all positions
        # Shape: (S,) after mean over heads and selecting last token row
        attn_last_token = last_layer_attn[0, :, -1, :].mean(dim=0)  # (S,)
        attn_last_token = attn_last_token.float().cpu().numpy()

        # Skip if NaN (can happen with quantized models)
        if np.any(np.isnan(attn_last_token)):
            skipped += 1
            del outputs, last_layer_attn, attn_last_token, input_ids
            torch.cuda.empty_cache()
            continue

        # Compute fraction per token type
        type_attn = defaultdict(float)
        type_count = defaultdict(int)
        for idx, ttype in enumerate(token_types):
            if idx < len(attn_last_token):
                type_attn[ttype] += attn_last_token[idx]
                type_count[ttype] += 1

        # Normalize (attention should already sum to ~1, but just in case)
        total_attn = sum(type_attn.values())
        fractions = {}
        for ttype in ["issue", "path", "code", "template"]:
            fractions[ttype] = float(type_attn[ttype] / total_attn) if total_attn > 0 else 0.0
            fractions[f"{ttype}_ntokens"] = type_count[ttype]

        all_fractions.append(fractions)

        # Free GPU memory
        del outputs, last_layer_attn, attn_last_token, input_ids
        torch.cuda.empty_cache()

        if (i + 1) % 10 == 0:
            # Running stats
            running = {t: np.mean([f[t] for f in all_fractions]) for t in ["issue", "path", "code", "template"]}
            print(f"  [{i+1}/{len(sampled)}] "
                  f"path={running['path']:.4f} code={running['code']:.4f} "
                  f"issue={running['issue']:.4f} template={running['template']:.4f}")

    # ---- Compute statistics ----
    print(f"\nDone. Processed {len(all_fractions)} examples, skipped {skipped}.")

    if not all_fractions:
        print("ERROR: No examples processed successfully.")
        return

    stats = {}
    for ttype in ["path", "code", "issue", "template"]:
        vals = [f[ttype] for f in all_fractions]
        stats[ttype] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
        }
        ntoks = [f[f"{ttype}_ntokens"] for f in all_fractions]
        stats[ttype]["avg_ntokens"] = float(np.mean(ntoks))

    # Path vs code attention ratio
    path_vals = np.array([f["path"] for f in all_fractions])
    code_vals = np.array([f["code"] for f in all_fractions])
    # Avoid division by zero
    ratio_vals = path_vals / np.maximum(code_vals, 1e-10)
    stats["path_to_code_ratio"] = {
        "mean": float(np.mean(ratio_vals)),
        "std": float(np.std(ratio_vals)),
        "median": float(np.median(ratio_vals)),
    }

    # ---- Print results ----
    print("\n" + "=" * 70)
    print("ATTENTION DISTRIBUTION AT LAST TOKEN (before Yes/No prediction)")
    print("Model: path-only reranker (rankft_runB_graph/best)")
    print("Prompt: code-centric (path + code content)")
    print(f"Examples: {len(all_fractions)}")
    print("=" * 70)

    header = f"{'Token Type':<12} {'Attn Fraction':>15} {'Avg #Tokens':>12}"
    print(header)
    print("-" * len(header))
    for ttype in ["path", "code", "issue", "template"]:
        s = stats[ttype]
        print(f"{ttype:<12} {s['mean']:.4f} +/- {s['std']:.4f}   {s['avg_ntokens']:>8.1f}")

    print()
    r = stats["path_to_code_ratio"]
    print(f"Path/Code attention ratio: {r['mean']:.2f} +/- {r['std']:.2f} "
          f"(median: {r['median']:.2f})")

    # Per-token attention (normalized by token count)
    print("\nPer-token attention (fraction / #tokens, x1000):")
    for ttype in ["path", "code", "issue", "template"]:
        s = stats[ttype]
        if s["avg_ntokens"] > 0:
            per_tok = s["mean"] / s["avg_ntokens"] * 1000
            print(f"  {ttype:<12} {per_tok:.4f}")

    # ---- Save results ----
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    results = {
        "config": {
            "model_path": args.model_path,
            "lora_path": args.lora_path,
            "num_examples": len(all_fractions),
            "num_skipped": skipped,
            "max_seq_length": args.max_seq_length,
        },
        "attention_distribution": stats,
        "per_example": all_fractions,
    }
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
