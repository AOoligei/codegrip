#!/usr/bin/env python3
"""
Train a function-level code expert (FCE).

Key insight: code understanding should help at FUNCTION level, not file level.
At file level, path dominates. At function level (scoped within correct file),
path no longer disambiguates — the model must read code.

Training data: (issue, file_path, function_name, function_body, label)
- Positive: GT function in GT file
- Negatives: other functions in same file (path constant, code varies)

This is cleaner than file-level training: negatives share the file path,
so the model MUST use code content to distinguish.

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/train_function_code_expert.py \
        --gpu_id 0 \
        --output_dir /data/chenlibin/grepo_agent_experiments/fce/expert \
        --epochs 2 --lr 5e-5
"""

import argparse
import ast
import json
import os
import random
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
# Use the richer labels file with per-file qualified positive_functions
TRAIN_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_train_function_labels.jsonl"
BM25_TRAIN_PATH = "/home/chenlibin/grepo_agent/data/rankft/grepo_train_bm25_top500.jsonl"
REPO_DIR = "/home/chenlibin/grepo_agent/data/repos"

# Function-level prompt WITH code body
FCE_PROMPT = (
    "Given the bug report, is this function likely to need modification? "
    "Read the function body carefully.\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {file_path}\n"
    "Function: {function_name}\n"
    "Code:\n{function_body}\n\n"
    "Answer:"
)


def extract_functions_from_file(repo, file_path, max_lines=40):
    """Extract (name, qualified_name, body, lineno) tuples from a Python file.

    Qualified name includes class scope (e.g., "MyClass.my_method") to
    disambiguate when multiple classes define methods with the same name.
    """
    full_path = os.path.join(REPO_DIR, repo, file_path)
    if not os.path.isfile(full_path):
        return []
    try:
        with open(full_path, "r", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception:
        return []

    lines = source.splitlines()
    funcs = []

    def walk(node, scope_prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                new_prefix = f"{scope_prefix}{child.name}." if scope_prefix else f"{child.name}."
                walk(child, new_prefix)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual_name = f"{scope_prefix}{child.name}"
                start = child.lineno - 1
                end = min(start + max_lines, len(lines))
                body = "\n".join(lines[start:end])
                funcs.append({
                    "name": child.name,
                    "qual_name": qual_name,
                    "body": body,
                    "lineno": child.lineno,
                })
                # Recurse into nested defs (but keep them scoped under parent func)
                walk(child, f"{qual_name}.")
            else:
                walk(child, scope_prefix)

    walk(tree)
    return funcs


def truncate_prompt_safely(prompt, tokenizer, max_seq_length):
    """Truncate prompt from middle, preserving Answer: suffix."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_seq_length:
        return prompt
    suffix = "\n\nAnswer:"
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    keep = max_seq_length - len(suffix_ids) - 1
    return tokenizer.decode(ids[:keep] + suffix_ids, skip_special_tokens=True)


def compute_listwise_loss(model, tokenizer, prompts, yes_id, no_id,
                          device, max_seq_length):
    """Listwise loss: positive at index 0, CE with target=0."""
    prompts = [truncate_prompt_safely(p, tokenizer, max_seq_length) for p in prompts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_seq_length,
                       padding_side="left").to(device)
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    yes_logits = logits[:, yes_id].float()
    no_logits = logits[:, no_id].float()
    scores = yes_logits - no_logits
    target = torch.zeros(1, dtype=torch.long, device=device)
    return torch.nn.functional.cross_entropy(scores.unsqueeze(0), target)


def build_training_examples(args):
    """Build function-level training examples from GREPO train.

    For each training issue with changed_functions:
      For each (gt_file, gt_func):
        positive = (gt_file, gt_func, its_body)
        same-file negatives = other functions from same gt_file (path-matched)
        cross-file negatives = functions from other files in BM25 candidates
    """
    print("Loading training data...")
    train_data = []
    with open(TRAIN_PATH) as f:
        for line in f:
            train_data.append(json.loads(line))

    bm25_data = {}
    with open(BM25_TRAIN_PATH) as f:
        for line in f:
            rec = json.loads(line)
            bm25_data[(rec["repo"], str(rec["issue_id"]))] = rec

    print(f"  {len(train_data)} issues, {len(bm25_data)} BM25 candidates")

    examples = []
    stats = {
        "no_changed_funcs": 0,
        "no_gt_file": 0,
        "no_bm25": 0,
        "no_funcs_in_file": 0,
        "gt_func_not_found": 0,
        "too_few_negs": 0,
        "valid": 0,
    }

    for rec in train_data:
        repo = rec["repo"]
        issue_id = str(rec["issue_id"])
        issue_text = rec["issue_text"]
        gt_files = rec.get("changed_py_files", rec.get("changed_files", []))
        cfd = rec.get("changed_functions_detailed", {})

        if not cfd or not gt_files:
            stats["no_changed_funcs"] += 1
            continue

        key = (repo, issue_id)
        if key not in bm25_data:
            stats["no_bm25"] += 1
            continue

        candidates = bm25_data[key].get("candidates",
                                         bm25_data[key].get("bm25_candidates", []))

        # Use per-file qualified positive_functions (Class.method form)
        for gt_file in gt_files:
            file_info = cfd.get(gt_file, {})
            positive_qual_names = file_info.get("positive_functions", [])
            if not positive_qual_names:
                # Module-level edit or no positive functions known
                stats["no_changed_funcs"] += 1
                continue

            file_funcs = extract_functions_from_file(
                repo, gt_file, max_lines=args.func_max_lines)
            if not file_funcs:
                stats["no_funcs_in_file"] += 1
                continue

            # Build qual_name -> function dict for fast lookup
            qual_name_to_func = {f["qual_name"]: f for f in file_funcs}

            # Find GT functions by qualified name
            gt_in_file = []
            for pqn in positive_qual_names:
                if pqn in qual_name_to_func:
                    gt_in_file.append(qual_name_to_func[pqn])

            if not gt_in_file:
                stats["gt_func_not_found"] += 1
                continue

            # Local GT qual names for negative filtering
            local_gt_qual_names = {f["qual_name"] for f in gt_in_file}

            # For each GT function in this file, build a training example
            for gt_func in gt_in_file:
                positive = gt_func

                # Same-file negatives: exclude THIS file's GT funcs only.
                # Other files' GT funcs (with same name) are NOT excluded because
                # they don't exist in this file; the name collision is fine here.
                same_file_negs = [f for f in file_funcs
                                  if f["qual_name"] not in local_gt_qual_names]

                # Cross-file negatives: from other BM25 candidates
                cross_file_negs = []
                for cand_file in candidates[:20]:
                    if cand_file in gt_files:
                        continue
                    cand_funcs = extract_functions_from_file(
                        repo, cand_file, max_lines=args.func_max_lines)
                    cross_file_negs.extend(
                        [(cand_file, f) for f in cand_funcs]
                    )
                    if len(cross_file_negs) >= args.num_cross_negs * 3:
                        break

                # Enforce exact listwise structure: 1 + num_same + num_cross
                # (otherwise CE denominator varies, poisoning learning)
                if len(same_file_negs) < args.num_same_file_negs:
                    stats["too_few_negs"] += 1
                    continue
                if len(cross_file_negs) < args.num_cross_negs:
                    stats["too_few_negs"] += 1
                    continue

                sampled_same = random.sample(same_file_negs, args.num_same_file_negs)
                sampled_cross = random.sample(cross_file_negs, args.num_cross_negs)

                examples.append({
                    "repo": repo,
                    "issue_id": issue_id,
                    "issue_text": issue_text,
                    "gt_file": gt_file,
                    "positive": positive,
                    "same_file_negs": sampled_same,
                    "cross_file_negs": [
                        {"file": cf, "func": f} for cf, f in sampled_cross
                    ],
                })
                stats["valid"] += 1

    print(f"\n=== Build stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return examples


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    if args.prebuilt_data:
        print(f"Loading prebuilt data from {args.prebuilt_data}")
        examples = []
        with open(args.prebuilt_data) as f:
            for line in f:
                examples.append(json.loads(line))
    else:
        examples = build_training_examples(args)
        with open(os.path.join(args.output_dir, "train_examples.jsonl"), "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
    print(f"\n{len(examples)} function-level training examples")

    print("\nLoading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": device},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()
    model.print_trainable_parameters()

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    accum_steps = args.gradient_accumulation_steps

    print(f"\nTraining: {args.epochs} epochs, lr={args.lr}, accum={accum_steps}")

    global_step = 0
    log_file = open(os.path.join(args.output_dir, "training_log.jsonl"), "w")
    start_time = time.time()

    for epoch in range(args.epochs):
        indices = list(range(len(examples)))
        random.shuffle(indices)
        epoch_loss = 0
        epoch_examples = 0
        accum_count = 0
        optimizer.zero_grad()

        for ex_idx, data_idx in enumerate(indices):
            ex = examples[data_idx]
            issue = ex["issue_text"]
            gt_file = ex["gt_file"]
            positive = ex["positive"]

            # Build prompts: positive first, then same-file negs, then cross-file negs
            items = [(gt_file, positive)]
            for f in ex["same_file_negs"]:
                items.append((gt_file, f))
            for cf in ex["cross_file_negs"]:
                items.append((cf["file"], cf["func"]))

            # Use qualified name in prompt to disambiguate class methods
            prompts = [
                FCE_PROMPT.format(
                    issue_text=issue,
                    file_path=fp,
                    function_name=func.get("qual_name", func["name"]),
                    function_body=func["body"][:800],
                )
                for fp, func in items
            ]

            try:
                loss = compute_listwise_loss(
                    model, tokenizer, prompts, yes_id, no_id,
                    device, args.max_seq_length)
                loss = loss / accum_steps
                loss.backward()
                epoch_loss += loss.item() * accum_steps
                epoch_examples += 1
                accum_count += 1
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    accum_count = 0
                    continue
                raise

            if accum_count >= accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                accum_count = 0

            if (ex_idx + 1) % 100 == 0:
                avg_loss = epoch_loss / max(1, epoch_examples)
                elapsed = time.time() - start_time
                print(f"  [{epoch+1}/{args.epochs}] [{ex_idx+1}/{len(indices)}] "
                      f"loss={avg_loss:.4f} ({elapsed:.0f}s)")
                log_file.write(json.dumps({
                    "epoch": epoch, "step": ex_idx + 1,
                    "global_step": global_step,
                    "avg_loss": avg_loss, "elapsed": elapsed,
                }) + "\n")
                log_file.flush()

        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = epoch_loss / max(1, epoch_examples)
        print(f"  Epoch {epoch+1} done: avg_loss={avg_loss:.4f}, "
              f"examples={epoch_examples}")

    save_dir = os.path.join(args.output_dir, "best")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    log_file.close()
    print(f"\nModel saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_same_file_negs", type=int, default=4)
    parser.add_argument("--num_cross_negs", type=int, default=4)
    parser.add_argument("--func_max_lines", type=int, default=40)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=1536)
    parser.add_argument("--prebuilt_data", type=str, default=None,
                        help="Skip data building, load from pre-built JSONL")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
