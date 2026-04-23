"""
CodeGRIP: Two-stage training for graph-internalized bug localization.

Stage 1 (GSP): Graph Structure Pre-training with structural tasks
Stage 2 (SFT): Supervised fine-tuning on bug localization task

Usage:
    # Full two-stage training
    python src/train/train_codegrip.py \
        --model_path /path/to/Qwen2.5-Coder-7B-Instruct \
        --gsp_data data/gsp/gsp_all.jsonl \
        --sft_data data/sft/sft_v1_filetree.jsonl \
        --output_dir experiments/codegrip_v1 \
        --device cuda:2

    # Stage 1 only (GSP pre-training)
    python src/train/train_codegrip.py \
        --model_path /path/to/Qwen2.5-Coder-7B-Instruct \
        --gsp_data data/gsp/gsp_all.jsonl \
        --output_dir experiments/codegrip_gsp_only \
        --skip_sft --device cuda:2

    # Stage 2 only (SFT from pre-trained LoRA)
    python src/train/train_codegrip.py \
        --model_path /path/to/Qwen2.5-Coder-7B-Instruct \
        --sft_data data/sft/sft_v1_filetree.jsonl \
        --output_dir experiments/codegrip_sft_only \
        --skip_gsp --device cuda:2
"""

import os
import json
import argparse
import random
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig

random.seed(42)
torch.manual_seed(42)


def load_jsonl_data(path: str, max_examples: int = -1) -> Dataset:
    """Load JSONL data into HuggingFace Dataset."""
    examples = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            examples.append(item)

    if max_examples > 0:
        random.shuffle(examples)
        examples = examples[:max_examples]

    return Dataset.from_list(examples)


def create_lora_model(
    model_path: str,
    dtype: torch.dtype,
    device: str,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: List[str] = None,
):
    """Load base model and attach fresh LoRA adapter."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def run_training_stage(
    model,
    tokenizer,
    dataset: Dataset,
    output_dir: str,
    stage_name: str,
    epochs: int = 2,
    lr: float = 2e-4,
    batch_size: int = 1,
    grad_accum: int = 8,
    max_seq_length: int = 4096,
    warmup_ratio: float = 0.05,
    dtype: str = "bfloat16",
    save_steps: int = 200,
    logging_steps: int = 10,
):
    """Run one training stage (GSP or SFT)."""
    print(f"\n{'='*60}")
    print(f"Stage: {stage_name}")
    print(f"  Examples: {len(dataset)}")
    print(f"  Epochs: {epochs}, LR: {lr}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=dtype == "bfloat16",
        fp16=dtype == "float16",
        max_length=max_seq_length,
        dataset_text_field=None,
        report_to="none",
        seed=42,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save stage checkpoint
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved {stage_name} model to {final_dir}")

    return model


def main():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--device", default="cuda:2")

    # Data
    parser.add_argument("--gsp_data", default=None, help="GSP pre-training data")
    parser.add_argument("--sft_data", default=None, help="SFT fine-tuning data")
    parser.add_argument("--max_gsp_examples", type=int, default=-1)
    parser.add_argument("--max_sft_examples", type=int, default=-1)
    parser.add_argument("--max_seq_length", type=int, default=4096)

    # LoRA config
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"])

    # Training hyperparams
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gsp_epochs", type=int, default=2)
    parser.add_argument("--gsp_lr", type=float, default=2e-4)
    parser.add_argument("--sft_epochs", type=int, default=3)
    parser.add_argument("--sft_lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)

    # Stage control
    parser.add_argument("--skip_gsp", action="store_true", help="Skip GSP stage")
    parser.add_argument("--skip_sft", action="store_true", help="Skip SFT stage")
    parser.add_argument("--gsp_checkpoint", default=None,
                        help="Resume SFT from existing GSP checkpoint")
    parser.add_argument("--adapter_composition", action="store_true",
                        help="Keep GSP adapter frozen (no merge), add SFT adapter on top")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # Load tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ============================================================
    # Stage 1: Graph Structure Pre-training (GSP)
    # ============================================================
    if not args.skip_gsp:
        if not args.gsp_data:
            raise ValueError("--gsp_data required when GSP is not skipped")

        print(f"\nLoading GSP data from {args.gsp_data}...")
        gsp_dataset = load_jsonl_data(args.gsp_data, max_examples=args.max_gsp_examples)
        print(f"  GSP examples: {len(gsp_dataset)}")

        # Create fresh LoRA model
        print(f"\nLoading base model for GSP...")
        model = create_lora_model(
            args.model_path, dtype, args.device,
            lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, target_modules=args.target_modules,
        )

        gsp_output = os.path.join(args.output_dir, "stage1_gsp")
        model = run_training_stage(
            model, tokenizer, gsp_dataset, gsp_output,
            stage_name="GSP (Graph Structure Pre-training)",
            epochs=args.gsp_epochs, lr=args.gsp_lr,
            batch_size=args.batch_size,
            grad_accum=args.gradient_accumulation_steps,
            max_seq_length=args.max_seq_length,
            dtype=args.dtype, save_steps=args.save_steps,
            logging_steps=args.logging_steps,
        )

        gsp_adapter_path = os.path.join(gsp_output, "final")
    else:
        model = None
        gsp_adapter_path = args.gsp_checkpoint

    # ============================================================
    # Stage 2: SFT Fine-tuning
    # ============================================================
    if not args.skip_sft:
        if not args.sft_data:
            raise ValueError("--sft_data required when SFT is not skipped")

        print(f"\nLoading SFT data from {args.sft_data}...")
        sft_dataset = load_jsonl_data(args.sft_data, max_examples=args.max_sft_examples)
        print(f"  SFT examples: {len(sft_dataset)}")

        if model is None:
            # Need to load model fresh (either with or without GSP adapter)
            print(f"\nLoading base model for SFT...")
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )

            if gsp_adapter_path:
                if args.adapter_composition:
                    # Warm-start: load GSP adapter and continue training on SFT data
                    # (no merge - just fine-tune the same adapter on the new task)
                    print(f"Loading GSP adapter from {gsp_adapter_path} (warm-start mode)...")
                    model = PeftModel.from_pretrained(base_model, gsp_adapter_path, is_trainable=True)
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    model.enable_input_require_grads()
                    model.print_trainable_parameters()
                    print("  Warm-start: continuing GSP adapter training on SFT data")
                else:
                    # Load GSP adapter then merge, then create new adapter for SFT
                    print(f"Loading GSP adapter from {gsp_adapter_path}...")
                    model = PeftModel.from_pretrained(base_model, gsp_adapter_path)
                    model = model.merge_and_unload()
                    model = model.cuda()
                    print("Merged GSP adapter into base model")

                    # Create new LoRA adapter for SFT (on top of merged GSP)
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=args.lora_rank,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout,
                        target_modules=args.target_modules,
                        bias="none",
                    )
                    model = get_peft_model(model, lora_config)
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    model.enable_input_require_grads()
                    model.print_trainable_parameters()
            else:
                # Fresh LoRA for SFT only (no GSP)
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=args.target_modules,
                    bias="none",
                )
                model = get_peft_model(base_model, lora_config)
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.enable_input_require_grads()
                model.print_trainable_parameters()
        else:
            # Continue from GSP stage - save GSP adapter, reload base + merged for SFT
            print("\nSaving GSP adapter and reloading for SFT stage...")
            gsp_adapter_path = os.path.join(args.output_dir, "stage1_gsp", "final")

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

            # Reload base model fresh
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            # Load and merge GSP adapter
            model = PeftModel.from_pretrained(base_model, gsp_adapter_path)
            model = model.merge_and_unload()
            model = model.cuda()
            print("  Merged GSP adapter into base model")

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.target_modules,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
            model.print_trainable_parameters()

        sft_output = os.path.join(args.output_dir, "stage2_sft")
        model = run_training_stage(
            model, tokenizer, sft_dataset, sft_output,
            stage_name="SFT (Bug Localization Fine-tuning)",
            epochs=args.sft_epochs, lr=args.sft_lr,
            batch_size=args.batch_size,
            grad_accum=args.gradient_accumulation_steps,
            max_seq_length=args.max_seq_length,
            dtype=args.dtype, save_steps=args.save_steps,
            logging_steps=args.logging_steps,
        )

    print("\n" + "=" * 60)
    print("CodeGRIP training complete!")
    print(f"Results in: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
