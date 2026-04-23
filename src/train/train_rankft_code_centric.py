"""
RankFT Code-Centric: Reranker that uses BOTH real paths AND code content.

Key difference from code-residual:
  - code-residual: anonymized paths + code -> tests if code alone has signal
  - code-centric:  REAL paths + code     -> tests if code ADDS signal on top of paths

This is the "truly strong code-reading baseline" requested by reviewers.

Minimal diff from train_rankft.py:
  1. New prompt template encouraging code analysis
  2. Always reads code content from data/repos/
  3. Extracts first 50 lines + function/class signatures (AST-based)
  4. Keeps real paths (no anonymization)

Usage:
    python src/train/train_rankft_code_centric.py \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path experiments/exp1_sft_only/stage2_sft/final \
        --train_data data/grepo_text/grepo_train.jsonl \
        --bm25_candidates data/rankft/grepo_train_bm25_top500.jsonl \
        --repo_dir data/repos \
        --output_dir /data/chenlibin/grepo_agent_experiments/code_centric_scorer \
        --device cuda:0
"""

import ast
import hashlib
import os
import json
import argparse
import random
import math
import re
import time
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

try:
    from rankft_training_utils import save_final_and_best_adapters
except ImportError:
    from src.train.rankft_training_utils import save_final_and_best_adapters

# Deterministic seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================
# Prompt template — Code-Centric variant
# Real paths + code content. Prompt encourages code reading.
# ============================================================

PROMPT_TEMPLATE = (
    "Given the bug report, analyze the code and determine if this file "
    "likely needs modification.\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Code (key sections):\n{code_content}\n\n"
    "Based on the code content and structure, is this file likely to need "
    "modification?\nAnswer:"
)


def build_prompt(issue_text: str, candidate_path: str, code_content: str,
                 tokenizer=None, max_seq_length: int = 1024) -> str:
    """Build the scoring prompt, ensuring Answer: suffix is never truncated."""
    # Build the non-code parts first to measure their token cost
    suffix = "Based on the code content and structure, is this file likely to need modification?\nAnswer:"
    prefix = (
        f"Given the bug report, analyze the code and determine if this file "
        f"likely needs modification.\n\n"
        f"Bug Report: {issue_text}\n\n"
        f"File: {candidate_path}\n\n"
        f"Code (key sections):\n"
    )
    if tokenizer is not None:
        # Reserve tokens for prefix + suffix + some margin
        prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
        suffix_tokens = len(tokenizer.encode(suffix, add_special_tokens=False))
        code_budget = max_seq_length - prefix_tokens - suffix_tokens - 10  # margin
        if code_budget <= 0:
            # Issue text is too long; truncate code to minimal placeholder
            code_content = "# (truncated due to long issue)"
        else:
            code_tokens = tokenizer.encode(code_content, add_special_tokens=False)
            if len(code_tokens) > code_budget:
                code_content = tokenizer.decode(code_tokens[:code_budget])

    return f"{prefix}{code_content}\n\n{suffix}"


# ============================================================
# Code extraction: first N lines + function/class signatures
# ============================================================

def _extract_signatures(source: str) -> List[str]:
    """Extract function and class definition signatures using AST.

    Returns a list of signature strings like:
      "def foo(x, y) -> int:"
      "class Bar(Base):"
    """
    sigs = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return sigs
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Reconstruct signature from the source line
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            args_str = ast.get_source_segment(source, node.args)
            if args_str is None:
                # Fallback: get the first line of the function def
                lines = source.split('\n')
                if node.lineno <= len(lines):
                    sig_line = lines[node.lineno - 1].strip()
                    sigs.append(sig_line)
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


def extract_code_content(
    repo_dir: str,
    repo: str,
    filepath: str,
    head_lines: int = 50,
    max_chars: int = 1500,
) -> str:
    """Extract code content: first N lines + function/class signatures.

    Strategy:
      1. Read first `head_lines` lines (imports, module docstring, early defs)
      2. Parse full file with AST to get function/class signatures
      3. Append signatures not already in the head portion
      4. Truncate to max_chars (~375 tokens at 4 chars/token)

    Falls back to head-only if AST parsing fails.
    """
    full_path = os.path.join(repo_dir, repo, filepath)
    try:
        with open(full_path, 'r', errors='replace') as f:
            full_source = f.read()
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return "# (file not available)"

    lines = full_source.split('\n')
    head = '\n'.join(lines[:head_lines])

    # Extract signatures not already visible in head
    sigs = _extract_signatures(full_source)
    # Filter out signatures that appear (by name) in the head portion
    extra_sigs = []
    for sig in sigs:
        # Check if the function/class name is already in the head
        # Use the full sig line as dedup key
        if sig.strip() not in head:
            extra_sigs.append(sig.strip())

    if extra_sigs:
        sig_block = "\n# ... (signatures from rest of file)\n" + '\n'.join(extra_sigs)
        content = head + sig_block
    else:
        content = head

    # Truncate to max_chars
    if len(content) > max_chars:
        content = content[:max_chars] + "\n# ... (truncated)"

    return content


# ============================================================
# Negative mining (reused from train_rankft.py)
# ============================================================

# Import the reusable components from the main training script.
# We import selectively to keep this file self-contained for the
# parts that differ, while reusing the large negative sampler.
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rankft import (
    NegativeSampler,
    build_cochange_index,
    build_import_adjacency,
    load_file_trees,
    load_train_data,
    load_bm25_candidates,
    get_yes_no_token_ids,
    compute_scores,
)


# ============================================================
# Training loop
# ============================================================

def train(args):
    """Main training routine — code-centric variant."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Re-seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Save config
    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    device = args.device

    # ---- Load tokenizer ----
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    yes_id, no_id = get_yes_no_token_ids(tokenizer)

    # ---- Load base model ----
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    # ---- Attach or load LoRA ----
    if args.lora_path:
        print(f"Loading existing LoRA adapter from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=True)
    else:
        print(f"Creating fresh LoRA adapter (rank={args.lora_rank})...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.train()

    # ---- Load training data ----
    print("Loading training data...")
    train_data = load_train_data(args.train_data)

    print("Loading BM25 candidates...")
    bm25_candidates = load_bm25_candidates(args.bm25_candidates)

    print("Building graph indexes...")
    cochange_index = {}
    if args.train_data_for_cochange:
        cochange_index = build_cochange_index(args.train_data_for_cochange)
        print(f"  Co-change index for {len(cochange_index)} repos")
    elif args.train_data:
        cochange_index = build_cochange_index(args.train_data)
        print(f"  Co-change index for {len(cochange_index)} repos")

    import_index = {}
    if args.dep_graph_dir:
        import_index = build_import_adjacency(args.dep_graph_dir)
        print(f"  Import index for {len(import_index)} repos")

    print("Loading file trees...")
    file_tree_dir = args.file_tree_dir if args.file_tree_dir else "data/file_trees"
    repo_files = load_file_trees(file_tree_dir)
    print(f"  File trees for {len(repo_files)} repos")

    # Create negative sampler (reused from train_rankft.py)
    neg_sampler = NegativeSampler(
        bm25_candidates=bm25_candidates,
        cochange_index=cochange_index,
        import_index=import_index,
        repo_files=repo_files,
        neg_bm25_ratio=args.neg_bm25_ratio,
        neg_graph_ratio=args.neg_graph_ratio,
        neg_random_ratio=args.neg_random_ratio,
    )

    # ---- Optimizer and scheduler ----
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    total_examples = len(train_data)
    steps_per_epoch = math.ceil(total_examples / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * 0.05)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Training loop ----
    print(f"\n{'='*60}")
    print(f"RankFT Code-Centric Training")
    print(f"  Examples: {total_examples}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Negatives per positive: {args.num_negatives}")
    print(f"  Neg mix: BM25={args.neg_bm25_ratio} "
          f"Graph={args.neg_graph_ratio} Random={args.neg_random_ratio}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}")
    print(f"  LR: {args.learning_rate}, Warmup: {warmup_steps}")
    print(f"  Max seq length: {args.max_seq_length}")
    print(f"  Code: head_lines={args.code_head_lines}, max_chars={args.code_max_chars}")
    print(f"  Repo dir: {args.repo_dir}")
    print(f"{'='*60}\n")

    global_step = 0
    accumulated_loss = 0.0
    log_losses = []
    best_loss = float("inf")
    start_time = time.time()

    # Diagnostics
    diag_path = os.path.join(args.output_dir, "training_diagnostics.jsonl")
    diag_file = open(diag_path, "w")
    diag_pos_scores: List[float] = []
    diag_neg_scores: List[float] = []
    diag_neg_type_scores: Dict[str, List[float]] = defaultdict(list)

    # Track code extraction stats
    code_stats = {"available": 0, "missing": 0, "with_sigs": 0}

    for epoch in range(args.num_epochs):
        indices = list(range(total_examples))
        random.shuffle(indices)
        optimizer.zero_grad()

        accum_count = 0  # tracks actual backward passes (not skipped examples)
        for ex_idx, data_idx in enumerate(indices):
            example = train_data[data_idx]
            repo = example["repo"]
            issue_id = example["issue_id"]
            issue_text = example["issue_text"]
            gt_files_list = sorted(example["changed_py_files"])  # sorted for determinism
            gt_files = set(gt_files_list)

            # Pick one positive file
            positive_file = random.choice(gt_files_list)

            # Sample negatives
            negatives, neg_types = neg_sampler.sample(
                repo=repo,
                issue_id=issue_id,
                gt_files=gt_files,
                num_negatives=args.num_negatives,
            )

            if len(negatives) == 0:
                continue

            # Build prompts with code content
            candidates = [positive_file] + negatives
            prompts = []
            for cand in candidates:
                code = extract_code_content(
                    args.repo_dir, repo, cand,
                    head_lines=args.code_head_lines,
                    max_chars=args.code_max_chars,
                )
                # Track stats
                if "file not available" in code:
                    code_stats["missing"] += 1
                else:
                    code_stats["available"] += 1
                    if "signatures from rest" in code:
                        code_stats["with_sigs"] += 1

                prompts.append(build_prompt(issue_text, cand, code,
                                           tokenizer=tokenizer,
                                           max_seq_length=args.max_seq_length))

            # Forward pass
            try:
                scores = compute_scores(
                    model, tokenizer, prompts,
                    yes_id, no_id,
                    args.max_seq_length, device,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at example {ex_idx}, skipping. "
                          f"Group size: {len(prompts)}")
                    torch.cuda.empty_cache()
                    continue
                raise

            # Listwise loss
            log_probs = F.log_softmax(scores, dim=0)
            loss = -log_probs[0]

            # Diagnostics
            with torch.no_grad():
                s = scores.detach().float()
                diag_pos_scores.append(s[0].item())
                for ni, ns in enumerate(s[1:].tolist()):
                    diag_neg_scores.append(ns)
                    if ni < len(neg_types):
                        diag_neg_type_scores[neg_types[ni]].append(ns)

            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()
            accum_count += 1

            # Gradient accumulation step
            if accum_count % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                step_loss = accumulated_loss
                log_losses.append(step_loss)
                accumulated_loss = 0.0

                # Write diagnostics
                if diag_pos_scores:
                    pos_arr = np.array(diag_pos_scores)
                    neg_arr = np.array(diag_neg_scores) if diag_neg_scores else np.array([0.0])
                    diag_record = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": step_loss,
                        "pos_score_mean": float(pos_arr.mean()),
                        "neg_score_mean": float(neg_arr.mean()),
                        "score_gap": float(pos_arr.mean() - neg_arr.mean()),
                    }
                    for nt in sorted(diag_neg_type_scores.keys()):
                        nt_scores = diag_neg_type_scores[nt]
                        diag_record[f"neg_{nt}_score_mean"] = float(np.mean(nt_scores))
                        diag_record[f"neg_{nt}_count"] = len(nt_scores)
                    diag_file.write(json.dumps(diag_record) + "\n")
                    diag_file.flush()

                diag_pos_scores.clear()
                diag_neg_scores.clear()
                diag_neg_type_scores.clear()

                # Logging
                if global_step % args.logging_steps == 0:
                    avg_recent = np.mean(log_losses[-args.logging_steps:])
                    elapsed = time.time() - start_time
                    current_lr = scheduler.get_last_lr()[0]
                    print(
                        f"  [Epoch {epoch+1}/{args.num_epochs}] "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {step_loss:.4f} (avg: {avg_recent:.4f}) | "
                        f"LR: {current_lr:.2e} | "
                        f"Time: {elapsed:.0f}s"
                    )

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    print(f"  Saving checkpoint to {ckpt_dir}...")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

                    state = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "loss_history": log_losses,
                        "best_loss": best_loss,
                        "code_stats": code_stats,
                    }
                    with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
                        json.dump(state, f, indent=2)

                    recent_avg = np.mean(log_losses[-min(50, len(log_losses)):])
                    if recent_avg < best_loss:
                        best_loss = recent_avg
                        best_dir = os.path.join(args.output_dir, "best")
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
                        print(f"    New best (avg loss: {best_loss:.4f})")

                # Code stats every 100 steps
                if global_step % 100 == 0:
                    total_code = code_stats["available"] + code_stats["missing"]
                    if total_code > 0:
                        pct_avail = 100.0 * code_stats["available"] / total_code
                        pct_sigs = 100.0 * code_stats["with_sigs"] / max(1, code_stats["available"])
                        print(f"  [Code stats @ step {global_step}] "
                              f"available={pct_avail:.1f}%, "
                              f"with_extra_sigs={pct_sigs:.1f}%")

        # Handle remaining accumulated gradients
        if accum_count % args.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        print(f"\n  Epoch {epoch+1} complete. "
              f"Avg loss: {np.mean(log_losses[-steps_per_epoch:]):.4f}")

    # ---- Close diagnostics ----
    diag_file.close()

    # ---- Final best check ----
    recent_avg = np.mean(log_losses[-min(50, len(log_losses)):])
    if recent_avg < best_loss:
        best_loss = recent_avg
        best_dir = os.path.join(args.output_dir, "best")
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        print(f"  End-of-training best update (avg loss: {best_loss:.4f})")

    # ---- Save final model ----
    print(f"\nSaving final model under {args.output_dir}...")
    final_dir, best_dir, best_backfilled = save_final_and_best_adapters(
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
    )
    if best_backfilled:
        print(f"  Backfilled missing best adapter at {best_dir}")

    # Save loss history and code stats
    with open(os.path.join(args.output_dir, "loss_history.json"), "w") as f:
        json.dump({
            "losses": log_losses,
            "total_steps": global_step,
            "code_stats": code_stats,
        }, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"  Total steps: {global_step}")
    print(f"  Final avg loss: {np.mean(log_losses[-50:]):.4f}")
    print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Code stats: {code_stats}")
    print(f"  Output: {args.output_dir}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RankFT Code-Centric: reranker with real paths + code content"
    )

    # Model
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lora_path", default=None)

    # Data
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--bm25_candidates", required=True)
    parser.add_argument("--dep_graph_dir", default="data/dep_graphs")
    parser.add_argument("--train_data_for_cochange", default=None)
    parser.add_argument("--file_tree_dir", default="data/file_trees")

    # Code content
    parser.add_argument("--repo_dir", default="data/repos",
                        help="Directory containing repo snapshots")
    parser.add_argument("--code_head_lines", type=int, default=50,
                        help="Number of head lines to read from each file")
    parser.add_argument("--code_max_chars", type=int, default=1500,
                        help="Max chars of code content (~375 tokens)")

    # Output
    parser.add_argument("--output_dir", required=True)

    # Hardware
    parser.add_argument("--device", default="cuda:0")

    # Negative mining
    parser.add_argument("--num_negatives", type=int, default=4,
                        help="Negatives per positive (reduced from 16 for memory)")
    parser.add_argument("--neg_bm25_ratio", type=float, default=0.5)
    parser.add_argument("--neg_graph_ratio", type=float, default=0.25)
    parser.add_argument("--neg_random_ratio", type=float, default=0.25)

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Max tokens per prompt (longer to fit code)")

    # LoRA config
    parser.add_argument("--lora_rank", type=int, default=32)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Validate negative ratios
    ratio_sum = args.neg_bm25_ratio + args.neg_graph_ratio + args.neg_random_ratio
    if abs(ratio_sum - 1.0) > 0.01:
        parser.error(f"Negative ratios must sum to 1.0, got {ratio_sum:.2f}")

    train(args)


if __name__ == "__main__":
    main()
