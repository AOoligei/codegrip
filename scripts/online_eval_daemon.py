#!/usr/bin/env python3
"""
Online eval daemon: watches for new checkpoints, evals each one,
keeps only the best. Runs alongside training on a shared GPU.

Usage:
    python online_eval_daemon.py \
        --exp_dir experiments/swebench_adapted_delex50 \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --test_data data/swebench_lite/swebench_lite_test.jsonl \
        --candidates data/rankft/swebench_test_bm25_top500.jsonl \
        --gpu_id 0 \
        --top_k 50 --max_seq_length 1024 \
        --poll_interval 60
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time


def get_checkpoints(exp_dir):
    """Return sorted list of checkpoint dirs by step number."""
    ckpts = []
    for d in glob.glob(os.path.join(exp_dir, "checkpoint-*")):
        try:
            step = int(os.path.basename(d).split("-")[1])
            if os.path.exists(os.path.join(d, "adapter_config.json")):
                ckpts.append((step, d))
        except (ValueError, IndexError):
            pass
    return sorted(ckpts)


def eval_checkpoint(model_path, lora_path, test_data, candidates, gpu_id,
                    output_dir, top_k, max_seq_length):
    """Run 4-bit eval, return H@1 or None on failure."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable, "scripts/eval_rankft_4bit.py",
        "--model_path", model_path,
        "--lora_path", lora_path,
        "--test_data", test_data,
        "--bm25_candidates", candidates,
        "--output_dir", output_dir,
        "--gpu_id", str(gpu_id),
        "--top_k", str(top_k),
        "--max_seq_length", str(max_seq_length),
        "--score_batch_size", "1",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Override gpu_id to 0 since CUDA_VISIBLE_DEVICES maps it
    cmd[cmd.index("--gpu_id") + 1] = "0"
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"  Eval failed: {result.stderr[-200:]}", flush=True)
            return None
    except subprocess.TimeoutExpired:
        print("  Eval timed out", flush=True)
        return None

    summary_path = os.path.join(output_dir, "summary.json")
    if not os.path.exists(summary_path):
        return None
    try:
        with open(summary_path) as f:
            data = json.load(f)
        return data["overall"]["hit@1"]
    except (json.JSONDecodeError, KeyError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidates2", default=None, help="Second candidate pool (e.g. tricked)")
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--poll_interval", type=int, default=30)
    parser.add_argument("--keep_all", action="store_true", help="Don't delete non-best checkpoints")
    args = parser.parse_args()

    best_h1 = -1.0
    best_step = -1
    best_h1_c2 = -1.0
    evaled_steps = set()

    print(f"Online eval daemon started for {args.exp_dir}", flush=True)
    print(f"  GPU: {args.gpu_id}, poll: {args.poll_interval}s", flush=True)
    print(f"  Candidates: {args.candidates}", flush=True)
    if args.candidates2:
        print(f"  Candidates2: {args.candidates2}", flush=True)

    while True:
        ckpts = get_checkpoints(args.exp_dir)
        new_ckpts = [(s, d) for s, d in ckpts if s not in evaled_steps]

        for step, ckpt_dir in new_ckpts:
            print(f"\n[{time.strftime('%H:%M:%S')}] Evaluating checkpoint-{step}...", flush=True)
            
            out_dir = os.path.join(args.exp_dir, f"eval_step{step}")
            h1 = eval_checkpoint(
                args.model_path, ckpt_dir, args.test_data, args.candidates,
                args.gpu_id, out_dir, args.top_k, args.max_seq_length
            )
            
            h1_c2 = None
            if args.candidates2:
                out_dir2 = os.path.join(args.exp_dir, f"eval_step{step}_c2")
                h1_c2 = eval_checkpoint(
                    args.model_path, ckpt_dir, args.test_data, args.candidates2,
                    args.gpu_id, out_dir2, args.top_k, args.max_seq_length
                )

            evaled_steps.add(step)

            if h1 is not None:
                improved = h1 > best_h1
                marker = " *** NEW BEST ***" if improved else ""
                print(f"  step {step}: H@1={h1:.2f}%{' C2=' + f'{h1_c2:.2f}%' if h1_c2 else ''}{marker}", flush=True)
                
                if improved:
                    # Save as best
                    best_dir = os.path.join(args.exp_dir, "online_best")
                    if os.path.exists(best_dir):
                        shutil.rmtree(best_dir)
                    shutil.copytree(ckpt_dir, best_dir)
                    best_h1 = h1
                    best_step = step
                    if h1_c2 is not None:
                        best_h1_c2 = h1_c2
                    # Save best info
                    with open(os.path.join(args.exp_dir, "online_best_info.json"), "w") as f:
                        json.dump({"step": step, "h1": h1, "h1_c2": h1_c2}, f, indent=2)

                if not args.keep_all and not improved:
                    # Delete non-best checkpoint to save disk
                    shutil.rmtree(ckpt_dir)
                    shutil.rmtree(out_dir, ignore_errors=True)
                    if args.candidates2:
                        shutil.rmtree(out_dir2, ignore_errors=True)
                    print(f"  Deleted checkpoint-{step} (not best)", flush=True)
            else:
                print(f"  step {step}: eval failed", flush=True)

        # Check if training is done
        final_dir = os.path.join(args.exp_dir, "final")
        if os.path.exists(os.path.join(final_dir, "adapter_config.json")):
            # Eval final too
            if "final" not in evaled_steps:
                print(f"\n[{time.strftime('%H:%M:%S')}] Training done. Evaluating final...", flush=True)
                out_dir = os.path.join(args.exp_dir, "eval_final")
                h1 = eval_checkpoint(
                    args.model_path, final_dir, args.test_data, args.candidates,
                    args.gpu_id, out_dir, args.top_k, args.max_seq_length
                )
                h1_c2 = None
                if args.candidates2:
                    out_dir2 = os.path.join(args.exp_dir, "eval_final_c2")
                    h1_c2 = eval_checkpoint(
                        args.model_path, final_dir, args.test_data, args.candidates2,
                        args.gpu_id, out_dir2, args.top_k, args.max_seq_length
                    )
                evaled_steps.add("final")
                if h1 is not None:
                    marker = " *** NEW BEST ***" if h1 > best_h1 else ""
                    print(f"  final: H@1={h1:.2f}%{' C2=' + f'{h1_c2:.2f}%' if h1_c2 else ''}{marker}", flush=True)
                    if h1 > best_h1:
                        best_dir = os.path.join(args.exp_dir, "online_best")
                        if os.path.exists(best_dir):
                            shutil.rmtree(best_dir)
                        shutil.copytree(final_dir, best_dir)
                        best_h1 = h1
                        best_step = "final"
                        with open(os.path.join(args.exp_dir, "online_best_info.json"), "w") as f:
                            json.dump({"step": "final", "h1": h1, "h1_c2": h1_c2}, f, indent=2)

            # Also eval best/ if it exists
            best_ckpt = os.path.join(args.exp_dir, "best")
            if os.path.exists(os.path.join(best_ckpt, "adapter_config.json")) and "best" not in evaled_steps:
                print(f"Evaluating best/...", flush=True)
                out_dir = os.path.join(args.exp_dir, "eval_best")
                h1 = eval_checkpoint(
                    args.model_path, best_ckpt, args.test_data, args.candidates,
                    args.gpu_id, out_dir, args.top_k, args.max_seq_length
                )
                evaled_steps.add("best")
                if h1 is not None:
                    marker = " *** NEW BEST ***" if h1 > best_h1 else ""
                    print(f"  best/: H@1={h1:.2f}%{marker}", flush=True)
                    if h1 > best_h1:
                        best_h1 = h1
                        best_step = "best"

            print(f"\n{'='*50}", flush=True)
            print(f"DONE: {args.exp_dir}", flush=True)
            print(f"Best: step={best_step}, H@1={best_h1:.2f}%", flush=True)
            if best_h1_c2 > 0:
                print(f"Best C2: H@1={best_h1_c2:.2f}%", flush=True)
            print(f"{'='*50}", flush=True)
            break

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
