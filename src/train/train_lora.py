"""
LoRA fine-tuning for file-level bug localization.

Fine-tunes a code LLM with LoRA on GREPO SFT data.
Supports Qwen2.5-Coder-7B-Instruct and similar models.

Usage:
    python src/train/train_lora.py \
        --model_path /path/to/Qwen2.5-Coder-7B-Instruct \
        --train_data data/sft/sft_v1_filetree.jsonl \
        --output_dir experiments/lora_v1_filetree \
        --epochs 3 --lr 2e-4 --lora_rank 32
"""

import os
import json
import argparse
import random
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

random.seed(42)
torch.manual_seed(42)


def load_sft_data(path: str, max_examples: int = -1) -> Dataset:
    """Load SFT JSONL data into HuggingFace Dataset."""
    examples = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            examples.append(item)

    if max_examples > 0:
        random.shuffle(examples)
        examples = examples[:max_examples]

    # Convert to Dataset format expected by SFTTrainer
    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])

    # Data
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--max_seq_length", type=int, default=4096)

    # LoRA config
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"])

    # Training
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)

    # Hardware
    parser.add_argument("--device", default="cuda:1")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model from {args.model_path}...")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype_map[args.dtype],
        device_map=args.device,
        trust_remote_code=True,
    )

    # Configure LoRA
    print(f"Configuring LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    print(f"Loading training data from {args.train_data}...")
    train_dataset = load_sft_data(args.train_data, max_examples=args.max_examples)
    print(f"  Training examples: {len(train_dataset)}")

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        max_length=args.max_seq_length,
        dataset_text_field=None,  # We use messages format
        report_to="none",
        seed=42,
        dataloader_pin_memory=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    print(f"Saving final model to {final_dir}...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
