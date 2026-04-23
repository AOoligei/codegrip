"""
Experiment 1A: Long-Context Code-Centric Reranker.

Thin wrapper around train_rankft_code_centric.py with longer context defaults:
  - max_seq_length:  1024 -> 4096
  - code_head_lines:   50 -> 300
  - code_max_chars:  1500 -> 12000  (~3000 tokens)
  - num_negatives:      4 -> 2      (memory constraint with 4x longer seqs)
  - gradient_accumulation_steps: 16 -> 32  (compensate fewer negatives)

Hypothesis: Providing ~3000 tokens of code (vs ~375) gives the reranker
enough context to reason about file relevance from code structure, not
just path names.

Usage:
    python src/train/train_rankft_code_longctx.py \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path experiments/exp1_sft_only/stage2_sft/final \
        --train_data data/grepo_text/grepo_train.jsonl \
        --bm25_candidates data/rankft/grepo_train_bm25_top500.jsonl \
        --repo_dir data/repos \
        --output_dir /data/chenlibin/grepo_agent_experiments/exp1a_code_longctx \
        --device cuda:0
"""

import sys
import os

# Ensure sibling modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_rankft_code_centric import main as _original_main
import argparse


def main():
    """Run code-centric training with long-context defaults.

    We monkey-patch argparse defaults before calling the original main().
    This avoids duplicating any logic.
    """
    # Store original parse_args
    _original_parse_args = argparse.ArgumentParser.parse_args

    def patched_parse_args(self, args=None, namespace=None):
        """Override specific defaults for long-context experiment."""
        # Apply our default overrides before parsing
        for action in self._actions:
            if action.dest == "max_seq_length":
                action.default = 4096
            elif action.dest == "code_head_lines":
                action.default = 300
            elif action.dest == "code_max_chars":
                action.default = 12000
            elif action.dest == "num_negatives":
                action.default = 2
            elif action.dest == "gradient_accumulation_steps":
                action.default = 32
        return _original_parse_args(self, args=args, namespace=namespace)

    # Patch, call, restore
    argparse.ArgumentParser.parse_args = patched_parse_args
    try:
        _original_main()
    finally:
        argparse.ArgumentParser.parse_args = _original_parse_args


if __name__ == "__main__":
    main()
