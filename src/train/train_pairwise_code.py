"""
Pairwise Code-Contrastive Reranker for bug localization.

Key insight: Pointwise yes/no classification lets models shortcut via paths.
Pairwise comparison forces code-level discrimination.

Training format:
  Given a bug report and two candidate files (with anonymized paths),
  predict which file is more likely to contain the bug.

  Prompt:
    "Given the bug report, which file is more likely to need modification?
     Bug Report: {issue_text}
     File A:
     {code_a}
     File B:
     {code_b}
     Answer: File"

  Label: "A" if first file is positive, "B" otherwise.
  Loss: cross-entropy on logits for tokens "A" vs "B".

Inference:
  For each candidate, compute pairwise win rate against K random opponents.
  Rank by win rate. This gives a code-based ranking without path access.
"""

import os
import json
import argparse
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Deterministic
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================
# Prompt template — Pairwise comparison, NO path info
# ============================================================

PAIRWISE_PROMPT = (
    "Given the bug report, which file is more likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File A:\n{code_a}\n\n"
    "File B:\n{code_b}\n\n"
    "Answer: File"
)

# Max tokens reserved for the prompt frame (everything except code_a and code_b)
# We'll dynamically truncate code to ensure "Answer: File" is never cut off
_PROMPT_FRAME_TOKENS = 150  # conservative estimate for the frame

# Code loading
_REPO_BASE_DIR = "data/repos"
_CODE_MAX_LINES = 50


def _find_repo_dir(repo_name):
    candidates = [
        os.path.join(_REPO_BASE_DIR, repo_name),
        os.path.join(_REPO_BASE_DIR, repo_name.replace('/', '__')),
        os.path.join(_REPO_BASE_DIR, repo_name.replace('/', '_')),
        os.path.join(_REPO_BASE_DIR, repo_name.split('/')[-1]),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return ""


def read_file_content(repo_dir, file_path, max_lines=50):
    full_path = os.path.join(repo_dir, file_path)
    try:
        with open(full_path, 'r', errors='ignore') as f:
            lines = f.readlines()[:max_lines]
        return ''.join(lines)[:2000]
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return '# (file content unavailable)'


def build_pairwise_prompt(issue_text, code_a, code_b, tokenizer=None, max_seq_length=768):
    """Build prompt, truncating code to ensure 'Answer: File' survives."""
    issue_trunc = issue_text[:800]
    # If tokenizer provided, dynamically truncate code
    if tokenizer is not None:
        # Budget for each code block = (max_seq - frame - issue) / 2
        frame_and_issue = tokenizer(
            f"Given the bug report, which file is more likely to need modification?\n\nBug Report: {issue_trunc}\n\nFile A:\n\n\nFile B:\n\n\nAnswer: File",
            add_special_tokens=True
        )
        frame_tokens = len(frame_and_issue['input_ids'])
        budget_per_code = max(50, (max_seq_length - frame_tokens) // 2)

        # Truncate each code block to fit budget
        code_a_tok = tokenizer.encode(code_a, add_special_tokens=False)[:budget_per_code]
        code_b_tok = tokenizer.encode(code_b, add_special_tokens=False)[:budget_per_code]
        code_a = tokenizer.decode(code_a_tok)
        code_b = tokenizer.decode(code_b_tok)

    return PAIRWISE_PROMPT.format(
        issue_text=issue_trunc,
        code_a=code_a,
        code_b=code_b,
    )


def load_training_data(train_path, bm25_path):
    """Load training examples with BM25 candidates as negatives."""
    train_data = [json.loads(l) for l in open(train_path)]
    bm25_index = {}
    for l in open(bm25_path):
        d = json.loads(l)
        key = (d['repo'], d['issue_id'])
        cands = d.get('candidates', d.get('bm25_candidates', []))
        bm25_index[key] = cands
    return train_data, bm25_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--bm25_candidates", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--pairs_per_example", type=int, default=4,
                        help="Number of (pos, neg) pairs per training example")
    parser.add_argument("--max_seq_length", type=int, default=768)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--code_max_lines", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo_dir", default="data/repos")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    global _REPO_BASE_DIR, _CODE_MAX_LINES
    _REPO_BASE_DIR = args.repo_dir
    _CODE_MAX_LINES = args.code_max_lines

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load model
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )

    # Apply LoRA
    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
        )
        model = get_peft_model(model, lora_config)

    model.train()
    model.gradient_checkpointing_enable()

    # Get token IDs for "A" and "B"
    # The prompt ends with "Answer: File" so the next token should be " A" or " B" (with space)
    # Try both with and without space, use whichever the tokenizer produces
    test_prompt = "Answer: File"
    test_a = "Answer: File A"
    test_b = "Answer: File B"
    base_ids = tokenizer.encode(test_prompt, add_special_tokens=False)
    a_full_ids = tokenizer.encode(test_a, add_special_tokens=False)
    b_full_ids = tokenizer.encode(test_b, add_special_tokens=False)
    # The label token is the last token that differs
    a_id = a_full_ids[-1]
    b_id = b_full_ids[-1]
    # Verify they're different
    assert a_id != b_id, f"A and B tokens are the same: {a_id}"
    print(f"Token IDs: A={a_id} ('{tokenizer.decode([a_id])}'), B={b_id} ('{tokenizer.decode([b_id])}')")

    # Load data
    print("Loading training data...")
    train_data, bm25_index = load_training_data(args.train_data, args.bm25_candidates)

    # Filter to examples with GT in candidates
    valid_examples = []
    for ex in train_data:
        key = (ex['repo'], ex['issue_id'])
        if key not in bm25_index:
            continue
        gt_files = set(ex.get('changed_py_files', []))
        if not gt_files:
            continue
        candidates = bm25_index[key]
        gt_in_cands = gt_files & set(candidates)
        neg_cands = [c for c in candidates if c not in gt_files]
        if gt_in_cands and neg_cands:
            valid_examples.append({
                'repo': ex['repo'],
                'issue_id': ex['issue_id'],
                'issue_text': ex['issue_text'],
                'gt_files': list(gt_in_cands),
                'neg_files': neg_cands[:20],  # top-20 negatives
            })

    print(f"Valid training examples: {len(valid_examples)}")

    # Build training pairs
    print("Building pairwise training pairs...")
    all_pairs = []
    for ex in valid_examples:
        repo_dir = _find_repo_dir(ex['repo'])
        if not repo_dir:
            continue

        for _ in range(args.pairs_per_example):
            pos_file = random.choice(ex['gt_files'])
            neg_file = random.choice(ex['neg_files'])

            code_pos = read_file_content(repo_dir, pos_file, _CODE_MAX_LINES)
            code_neg = read_file_content(repo_dir, neg_file, _CODE_MAX_LINES)

            # Randomly assign pos to A or B (prevent position bias)
            if random.random() < 0.5:
                prompt = build_pairwise_prompt(ex['issue_text'], code_pos, code_neg,
                                               tokenizer=tokenizer, max_seq_length=args.max_seq_length)
                label_id = a_id  # correct answer is A
            else:
                prompt = build_pairwise_prompt(ex['issue_text'], code_neg, code_pos,
                                               tokenizer=tokenizer, max_seq_length=args.max_seq_length)
                label_id = b_id  # correct answer is B

            all_pairs.append({
                'prompt': prompt,
                'label_id': label_id,
                'repo': ex['repo'],
            })

    random.shuffle(all_pairs)
    print(f"Total training pairs: {len(all_pairs)}")

    # Training
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=0.01
    )

    total_steps = (len(all_pairs) * args.num_epochs) // args.gradient_accumulation_steps
    print(f"Total steps: {total_steps}")

    # Diagnostics
    diag_file = os.path.join(args.output_dir, "training_diagnostics.jsonl")
    diag_f = open(diag_file, 'w')

    global_step = 0
    log_losses = []
    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(args.num_epochs):
        random.shuffle(all_pairs)
        accum_loss = 0.0

        for i, pair in enumerate(all_pairs):
            # Tokenize
            encoding = tokenizer(
                pair['prompt'],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq_length,
                padding=False,
            ).to(args.device)

            # Forward pass
            try:
                outputs = model(**encoding)
                logits = outputs.logits[0, -1]  # last token logits

                # Loss: cross-entropy between A and B logits
                target = torch.tensor([0 if pair['label_id'] == a_id else 1],
                                      device=args.device)
                ab_logits = torch.stack([logits[a_id], logits[b_id]]).unsqueeze(0)
                loss = F.cross_entropy(ab_logits, target)
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                accum_loss += loss.item()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    print(f"  OOM at pair {i}, skipping")
                    continue
                raise

            # Step
            if (i + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                step_loss = accum_loss
                log_losses.append(step_loss)
                accum_loss = 0.0

                if global_step % args.logging_steps == 0:
                    elapsed = time.time() - start_time
                    avg_loss = np.mean(log_losses[-50:])
                    print(f"  [Epoch {epoch+1}/{args.num_epochs}] Step {global_step}/{total_steps} | "
                          f"Loss: {step_loss:.4f} (avg: {avg_loss:.4f}) | "
                          f"Time: {elapsed:.0f}s", flush=True)

                    diag = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": step_loss,
                        "avg_loss": avg_loss,
                        "elapsed": elapsed,
                    }
                    diag_f.write(json.dumps(diag) + '\n')
                    diag_f.flush()

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"  Saved checkpoint to {ckpt_dir}")

                # Update best
                recent_avg = np.mean(log_losses[-min(50, len(log_losses)):])
                if recent_avg < best_loss:
                    best_loss = recent_avg
                    best_dir = os.path.join(args.output_dir, "best")
                    model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)

    # Save final
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    diag_f.close()

    print(f"\nTraining complete. {global_step} steps, final loss: {log_losses[-1]:.4f}")
    print(f"Best: {best_dir}, Final: {final_dir}")


if __name__ == "__main__":
    main()
