#!/usr/bin/env python3
"""
Group-DRO robust training for path-only reranker.

Makes the reranker robust to path perturbations WITHOUT needing code content.
For each example, generates clean + shuffled views and optimizes worst-group loss.

Loss = max(L_clean, L_shuffle_fn, L_shuffle_dir) + lambda * L_consistency
L_consistency = KL(softmax(scores_clean) || softmax(scores_perturbed))

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/train_group_dro.py \
        --gpu_id 0 \
        --output_dir /data/chenlibin/grepo_agent_experiments/group_dro \
        --epochs 2 --lr 5e-5
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
TRAIN_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_train.jsonl"
BM25_TRAIN_PATH = "/home/chenlibin/grepo_agent/data/rankft/grepo_train_bm25_top500.jsonl"

PROMPT_TEMPLATE = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)


def shuffle_filenames(paths):
    """Shuffle filenames within each directory."""
    dir_to_files = defaultdict(list)
    for p in paths:
        parts = p.rsplit("/", 1)
        if len(parts) == 2:
            dir_to_files[parts[0]].append(parts[1])
        else:
            dir_to_files[""].append(parts[0])
    mapping = {}
    for dir_path, filenames in dir_to_files.items():
        shuffled = filenames.copy()
        random.shuffle(shuffled)
        for orig, new in zip(filenames, shuffled):
            o = f"{dir_path}/{orig}" if dir_path else orig
            n = f"{dir_path}/{new}" if dir_path else new
            mapping[o] = n
    return mapping


def shuffle_dirs(paths):
    """Keep filenames, randomly shuffle directory assignments."""
    dirs = list(set(os.path.dirname(p) for p in paths if "/" in p))
    if len(dirs) < 2:
        return {p: p for p in paths}
    shuffled_dirs = dirs.copy()
    random.shuffle(shuffled_dirs)
    dir_map = dict(zip(dirs, shuffled_dirs))
    mapping = {}
    for p in paths:
        d = os.path.dirname(p)
        f = os.path.basename(p)
        new_d = dir_map.get(d, d)
        mapping[p] = f"{new_d}/{f}" if new_d else f
    return mapping


def apply_mapping(paths, mapping):
    return [mapping.get(p, p) for p in paths]


def compute_scores(model, tokenizer, prompts, yes_id, no_id,
                   device, max_seq_length):
    """Compute Yes-No logit scores for a batch of prompts."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_seq_length,
                       padding_side="left").to(device)
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    scores = logits[:, yes_id].float() - logits[:, no_id].float()
    return scores


def listwise_ce(scores):
    """Listwise CE loss. Index 0 is positive."""
    target = torch.zeros(1, dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores.unsqueeze(0), target)


def load_data():
    train_data = []
    with open(TRAIN_PATH) as f:
        for line in f:
            train_data.append(json.loads(line))
    bm25_data = {}
    with open(BM25_TRAIN_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            bm25_data[key] = rec
    return train_data, bm25_data


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    print("Loading data...")
    train_data, bm25_data = load_data()
    print(f"  {len(train_data)} train, {len(bm25_data)} candidates")

    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config,
        device_map={"": device}, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.train()

    lora_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    accum_steps = args.gradient_accumulation_steps

    print(f"Training: {args.epochs} epochs, lr={args.lr}, neg={args.num_negatives}, "
          f"lambda_cons={args.lambda_consistency}")

    log_file = open(os.path.join(args.output_dir, "training_log.jsonl"), "w")
    start_time = time.time()
    global_step = 0

    for epoch in range(args.epochs):
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        epoch_loss = 0
        epoch_examples = 0
        optimizer.zero_grad()
        accum_count = 0
        group_losses = {"clean": [], "shuffle_fn": [], "shuffle_dir": []}

        for ex_idx, data_idx in enumerate(indices):
            example = train_data[data_idx]
            repo = example["repo"]
            issue_text = example["issue_text"]
            gt_files = set(example.get("changed_py_files",
                                        example.get("changed_files", [])))
            if not gt_files:
                continue

            key = (repo, str(example["issue_id"]))
            if key not in bm25_data:
                continue

            positive_file = random.choice(sorted(gt_files))
            bm25_cands = bm25_data[key].get("candidates",
                                              bm25_data[key].get("bm25_candidates", []))
            neg_pool = [c for c in bm25_cands if c not in gt_files]
            if len(neg_pool) < args.num_negatives:
                continue
            negatives = random.sample(neg_pool, args.num_negatives)

            all_files = [positive_file] + negatives

            # Truncate issue
            issue_ids = tokenizer.encode(issue_text, add_special_tokens=False)
            if len(issue_ids) > 450:
                issue_text_trunc = tokenizer.decode(issue_ids[:450],
                                                     skip_special_tokens=True)
            else:
                issue_text_trunc = issue_text

            # Build perturbation mappings from FULL candidate pool (not just sampled)
            all_pool = bm25_cands[:200]
            fn_mapping = shuffle_filenames(all_pool)
            dir_mapping = shuffle_dirs(all_pool)

            # Generate views for sampled files
            clean_prompts = [PROMPT_TEMPLATE.format(
                issue_text=issue_text_trunc, candidate_path=f) for f in all_files]

            fn_files = apply_mapping(all_files, fn_mapping)
            fn_prompts = [PROMPT_TEMPLATE.format(
                issue_text=issue_text_trunc, candidate_path=f) for f in fn_files]

            dir_files = apply_mapping(all_files, dir_mapping)
            dir_prompts = [PROMPT_TEMPLATE.format(
                issue_text=issue_text_trunc, candidate_path=f) for f in dir_files]

            try:
                # Compute clean scores and get detached distribution for consistency
                scores_clean = compute_scores(model, tokenizer, clean_prompts,
                                               yes_id, no_id, device, 512)
                loss_clean = listwise_ce(scores_clean)
                p_clean = F.softmax(scores_clean.detach(), dim=0)

                # Pick ONE random perturbation per example (saves memory)
                if random.random() < 0.5:
                    perturb_prompts = fn_prompts
                    perturb_name = "shuffle_fn"
                else:
                    perturb_prompts = dir_prompts
                    perturb_name = "shuffle_dir"

                scores_perturb = compute_scores(model, tokenizer, perturb_prompts,
                                                 yes_id, no_id, device, 512)
                loss_perturb = listwise_ce(scores_perturb)

                # Worst-case loss over clean and perturbed
                loss_dro = torch.max(loss_clean, loss_perturb)

                # Consistency: KL(perturbed || clean)
                log_p_perturb = F.log_softmax(scores_perturb, dim=0)
                loss_cons = F.kl_div(log_p_perturb, p_clean, reduction="sum")

                loss = loss_dro + args.lambda_consistency * loss_cons
                loss = loss / accum_steps
                loss.backward()

                epoch_loss += loss.item() * accum_steps
                epoch_examples += 1
                accum_count += 1

                group_losses["clean"].append(loss_clean.item())
                group_losses[perturb_name].append(loss_perturb.item())

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

            if (ex_idx + 1) % 200 == 0:
                avg_loss = epoch_loss / max(1, epoch_examples)
                elapsed = time.time() - start_time
                avg_clean = np.mean(group_losses["clean"][-200:]) if group_losses["clean"] else 0
                avg_fn = np.mean(group_losses["shuffle_fn"][-200:]) if group_losses["shuffle_fn"] else 0
                avg_dir = np.mean(group_losses["shuffle_dir"][-200:]) if group_losses["shuffle_dir"] else 0
                print(f"  [{epoch+1}/{args.epochs}] [{ex_idx+1}] "
                      f"loss={avg_loss:.4f} clean={avg_clean:.3f} "
                      f"fn={avg_fn:.3f} dir={avg_dir:.3f} ({elapsed:.0f}s)")
                log_file.write(json.dumps({
                    "epoch": epoch, "step": ex_idx + 1,
                    "avg_loss": avg_loss,
                    "clean": avg_clean, "shuffle_fn": avg_fn, "shuffle_dir": avg_dir,
                }) + "\n")
                log_file.flush()

        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = epoch_loss / max(1, epoch_examples)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, examples={epoch_examples}")

    save_dir = os.path.join(args.output_dir, "best")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    log_file.close()
    print(f"\nSaved to {save_dir}")
    print(f"Total time: {time.time() - start_time:.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_negatives", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lambda_consistency", type=float, default=0.1)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
