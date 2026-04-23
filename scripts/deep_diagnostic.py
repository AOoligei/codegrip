"""
Deep diagnostic for RankFT cross-encoder reranker.

Prints intermediate variables for a few examples to understand WHY
the trained model ranks files differently from the base model.

Key things to inspect:
1. Per-file Yes/No logits and score (Yes - No)
2. Top-5 predicted tokens at the answer position
3. Ranking comparison: trained vs base model
4. Whether GT files are promoted, and by how much
5. What kinds of files get high/low scores (pattern analysis)
"""

import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Config
# ============================================================
MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
LORA_PATH = "/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/best"
TEST_DATA = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_test.jsonl"
CANDIDATES = "/home/chenlibin/grepo_agent/data/rankft/exp6_expanded_candidates.jsonl"
DEVICE_TRAINED = "cuda:0"  # Trained model GPU (physical GPU set via CUDA_VISIBLE_DEVICES)
DEVICE_BASE = "cuda:1"     # Base model GPU
N_EXAMPLES = 5
MAX_SEQ_LEN = 512

PROMPT_TEMPLATE = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)


def load_data():
    """Load test data and candidates."""
    test_data = {}
    with open(TEST_DATA) as f:
        for line in f:
            item = json.loads(line)
            key = f"{item['repo']}_{item['issue_id']}"
            test_data[key] = item

    candidates = {}
    with open(CANDIDATES) as f:
        for line in f:
            item = json.loads(line)
            key = f"{item['repo']}_{item['issue_id']}"
            candidates[key] = item.get("candidates", [])

    return test_data, candidates


def select_diverse_examples(test_data, candidates):
    """Select diverse examples: easy hit, hard miss, multi-file, single-file."""
    examples = []
    keys = list(test_data.keys())

    # Categorize
    single_file = []
    multi_file = []
    many_candidates = []

    for key in keys:
        item = test_data[key]
        cands = candidates.get(key, [])
        gt = set(item.get("changed_py_files", []))
        if not gt or not cands:
            continue
        gt_in_cands = gt & set(cands)
        info = {"key": key, "gt": gt, "gt_in_cands": gt_in_cands, "n_cands": len(cands)}

        if len(gt) == 1:
            single_file.append(info)
        else:
            multi_file.append(info)
        if len(cands) > 20:
            many_candidates.append(info)

    # Pick: 1 single-file with GT in candidates, 1 single-file GT NOT in candidates,
    # 1 multi-file with partial GT, 1 with many candidates, 1 random
    selected = []

    # Single file, GT present
    for info in single_file:
        if info["gt_in_cands"]:
            selected.append(info)
            break

    # Single file, GT absent
    for info in single_file:
        if not info["gt_in_cands"]:
            selected.append(info)
            break

    # Multi-file
    for info in multi_file:
        if info["gt_in_cands"] and len(info["gt_in_cands"]) < len(info["gt"]):
            selected.append(info)
            break

    # Many candidates
    for info in sorted(many_candidates, key=lambda x: -x["n_cands"]):
        if info["key"] not in {s["key"] for s in selected}:
            selected.append(info)
            break

    # Random different one
    import random
    random.seed(42)
    remaining = [k for k in keys if k not in {s["key"] for s in selected}]
    if remaining:
        rk = random.choice(remaining)
        item = test_data[rk]
        gt = set(item.get("changed_py_files", []))
        cands = candidates.get(rk, [])
        if gt and cands:
            selected.append({"key": rk, "gt": gt, "gt_in_cands": gt & set(cands), "n_cands": len(cands)})

    return selected[:N_EXAMPLES]


@torch.no_grad()
def score_single(model, tokenizer, prompt, yes_id, no_id):
    """Score a single prompt, returning detailed logit info."""
    enc = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    last_logits = outputs.logits[0, -1]  # logits at last position

    yes_logit = last_logits[yes_id].item()
    no_logit = last_logits[no_id].item()
    score = yes_logit - no_logit

    # Top-10 predicted tokens
    top_vals, top_ids = torch.topk(last_logits, 10)
    top_tokens = []
    for val, tid in zip(top_vals.tolist(), top_ids.tolist()):
        token_str = tokenizer.decode([tid])
        top_tokens.append((token_str, val))

    # Softmax probability of Yes vs No
    yn_logits = torch.tensor([yes_logit, no_logit])
    yn_probs = torch.softmax(yn_logits, dim=0)
    p_yes = yn_probs[0].item()

    return {
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "score": score,
        "p_yes": p_yes,
        "top_tokens": top_tokens,
        "seq_len": input_ids.shape[1],
    }


def analyze_example(model_trained, model_base, tokenizer, yes_id, no_id,
                    test_item, cands, example_info):
    """Deep analysis of one example."""
    gt = example_info["gt"]
    gt_in_cands = example_info["gt_in_cands"]

    issue_text = test_item["issue_text"]
    repo = test_item["repo"]
    issue_id = test_item["issue_id"]

    print(f"\n{'='*80}")
    print(f"EXAMPLE: {repo} #{issue_id}")
    print(f"{'='*80}")
    print(f"  GT files: {gt}")
    print(f"  GT in candidates: {gt_in_cands}")
    print(f"  Total candidates: {len(cands)}")

    # Show issue text (first 300 chars)
    issue_short = issue_text[:500].replace('\n', ' | ')
    print(f"  Issue (truncated): {issue_short}...")

    # Score all candidates with BOTH models
    results_trained = []
    results_base = []

    for i, cand in enumerate(cands):
        prompt = PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=cand)

        r_trained = score_single(model_trained, tokenizer, prompt, yes_id, no_id)
        r_base = score_single(model_base, tokenizer, prompt, yes_id, no_id)

        is_gt = cand in gt
        results_trained.append({"file": cand, "is_gt": is_gt, **r_trained, "model": "trained"})
        results_base.append({"file": cand, "is_gt": is_gt, **r_base, "model": "base"})

        # Progress for large candidate sets
        if (i + 1) % 10 == 0:
            print(f"    Scored {i+1}/{len(cands)} candidates...", end='\r')

    print(f"    Scored {len(cands)}/{len(cands)} candidates.     ")

    # Sort by score
    results_trained.sort(key=lambda x: -x["score"])
    results_base.sort(key=lambda x: -x["score"])

    # Build rank maps
    rank_trained = {r["file"]: i+1 for i, r in enumerate(results_trained)}
    rank_base = {r["file"]: i+1 for i, r in enumerate(results_base)}

    # ---- Print trained model top-10 ----
    print(f"\n  --- TRAINED MODEL (Run B) Top-10 ---")
    print(f"  {'Rank':>4} {'GT?':>3} {'Score':>8} {'P(Yes)':>7} {'Yes':>8} {'No':>8} {'SeqLen':>6} {'BaseRank':>8}  File")
    for i, r in enumerate(results_trained[:10]):
        gt_mark = "***" if r["is_gt"] else "   "
        base_r = rank_base[r["file"]]
        delta = base_r - (i+1)
        delta_str = f"({base_r})" if delta == 0 else f"({base_r}, {'+'if delta>0 else ''}{delta})"
        print(f"  {i+1:>4} {gt_mark} {r['score']:>8.3f} {r['p_yes']:>7.3f} {r['yes_logit']:>8.3f} {r['no_logit']:>8.3f} {r['seq_len']:>6} {delta_str:>12}  {r['file']}")

    # ---- Print base model top-10 ----
    print(f"\n  --- BASE MODEL (Qwen zero-shot) Top-10 ---")
    print(f"  {'Rank':>4} {'GT?':>3} {'Score':>8} {'P(Yes)':>7} {'Yes':>8} {'No':>8} {'SeqLen':>6} {'TrainRank':>10}  File")
    for i, r in enumerate(results_base[:10]):
        gt_mark = "***" if r["is_gt"] else "   "
        train_r = rank_trained[r["file"]]
        print(f"  {i+1:>4} {gt_mark} {r['score']:>8.3f} {r['p_yes']:>7.3f} {r['yes_logit']:>8.3f} {r['no_logit']:>8.3f} {r['seq_len']:>6} {'('+str(train_r)+')':>10}  {r['file']}")

    # ---- GT file analysis ----
    print(f"\n  --- GT FILE RANKINGS ---")
    for gt_file in sorted(gt):
        in_cands = gt_file in rank_trained
        if in_cands:
            tr = rank_trained[gt_file]
            br = rank_base[gt_file]
            # Find the detailed scores
            tr_info = next(r for r in results_trained if r["file"] == gt_file)
            br_info = next(r for r in results_base if r["file"] == gt_file)
            print(f"    {gt_file}")
            print(f"      Trained: rank={tr:>3}, score={tr_info['score']:>8.3f}, P(Yes)={tr_info['p_yes']:.3f}")
            print(f"      Base:    rank={br:>3}, score={br_info['score']:>8.3f}, P(Yes)={br_info['p_yes']:.3f}")
            print(f"      Rank improvement: {br - tr:>+d} positions")
        else:
            print(f"    {gt_file}  -- NOT IN CANDIDATES")

    # ---- Token analysis for GT files ----
    if gt_in_cands:
        print(f"\n  --- TOP PREDICTED TOKENS AT ANSWER POSITION (GT files) ---")
        for gt_file in sorted(gt_in_cands):
            tr_info = next(r for r in results_trained if r["file"] == gt_file)
            br_info = next(r for r in results_base if r["file"] == gt_file)
            print(f"    {gt_file}:")
            print(f"      Trained top-5: {[(t, f'{v:.2f}') for t, v in tr_info['top_tokens'][:5]]}")
            print(f"      Base top-5:    {[(t, f'{v:.2f}') for t, v in br_info['top_tokens'][:5]]}")

    # ---- Score distribution analysis ----
    trained_scores = [r["score"] for r in results_trained]
    base_scores = [r["score"] for r in results_base]
    gt_trained_scores = [r["score"] for r in results_trained if r["is_gt"]]
    gt_base_scores = [r["score"] for r in results_base if r["is_gt"]]
    non_gt_trained = [r["score"] for r in results_trained if not r["is_gt"]]
    non_gt_base = [r["score"] for r in results_base if not r["is_gt"]]

    print(f"\n  --- SCORE DISTRIBUTION ---")
    print(f"    Trained - All: mean={np.mean(trained_scores):.3f}, std={np.std(trained_scores):.3f}, "
          f"min={min(trained_scores):.3f}, max={max(trained_scores):.3f}")
    if gt_trained_scores:
        print(f"    Trained - GT:  mean={np.mean(gt_trained_scores):.3f}")
    if non_gt_trained:
        print(f"    Trained - Neg: mean={np.mean(non_gt_trained):.3f}")
    if gt_trained_scores and non_gt_trained:
        margin = np.mean(gt_trained_scores) - np.mean(non_gt_trained)
        print(f"    Trained - GT-Neg margin: {margin:+.3f}")

    print(f"    Base    - All: mean={np.mean(base_scores):.3f}, std={np.std(base_scores):.3f}")
    if gt_base_scores:
        print(f"    Base    - GT:  mean={np.mean(gt_base_scores):.3f}")
    if non_gt_base:
        print(f"    Base    - Neg: mean={np.mean(non_gt_base):.3f}")
    if gt_base_scores and non_gt_base:
        margin = np.mean(gt_base_scores) - np.mean(non_gt_base)
        print(f"    Base    - GT-Neg margin: {margin:+.3f}")

    # ---- Files most boosted by training ----
    rank_changes = []
    for f in cands:
        if f in rank_trained and f in rank_base:
            rank_changes.append((f, rank_base[f] - rank_trained[f], f in gt))
    rank_changes.sort(key=lambda x: -x[1])

    print(f"\n  --- FILES MOST BOOSTED BY TRAINING (rank improvement) ---")
    for f, delta, is_gt in rank_changes[:5]:
        gt_mark = " [GT]" if is_gt else ""
        tr = rank_trained[f]
        br = rank_base[f]
        print(f"    {f}{gt_mark}: base_rank={br} -> trained_rank={tr} (improved {delta} positions)")

    print(f"\n  --- FILES MOST DEMOTED BY TRAINING ---")
    for f, delta, is_gt in rank_changes[-5:]:
        gt_mark = " [GT]" if is_gt else ""
        tr = rank_trained[f]
        br = rank_base[f]
        print(f"    {f}{gt_mark}: base_rank={br} -> trained_rank={tr} (dropped {-delta} positions)")


def main():
    print("=" * 80)
    print("DEEP DIAGNOSTIC: RankFT Cross-Encoder Reranker")
    print("=" * 80)
    print(f"  Model: {MODEL_PATH}")
    print(f"  LoRA:  {LORA_PATH}")
    print(f"  Trained on: {DEVICE_TRAINED}, Base on: {DEVICE_BASE}")

    # Load data
    print("\nLoading data...")
    test_data, candidates = load_data()
    print(f"  {len(test_data)} test items, {len(candidates)} candidate sets")

    # Select examples
    examples = select_diverse_examples(test_data, candidates)
    print(f"\nSelected {len(examples)} diverse examples:")
    for ex in examples:
        print(f"  {ex['key']}: {len(ex['gt'])} GT files, {ex['n_cands']} candidates, "
              f"{len(ex['gt_in_cands'])} GT in candidates")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    yes_id, no_id = yes_ids[0], no_ids[0]
    print(f"  Yes token: id={yes_id}, decoded='{tokenizer.decode([yes_id])}'")
    print(f"  No token:  id={no_id}, decoded='{tokenizer.decode([no_id])}'")

    # Load trained model (base + LoRA) on GPU 5
    print(f"\nLoading trained model on {DEVICE_TRAINED}...")
    model_trained = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE_TRAINED, trust_remote_code=True,
    )
    model_trained = PeftModel.from_pretrained(model_trained, LORA_PATH)
    model_trained.eval()
    print("  Trained model loaded.")

    # Load base model on GPU 6
    print(f"\nLoading base model on {DEVICE_BASE}...")
    model_base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE_BASE, trust_remote_code=True,
    )
    model_base.eval()
    print("  Base model loaded.")

    # Analyze each example
    for example_info in examples:
        key = example_info["key"]
        test_item = test_data[key]
        cands = candidates.get(key, [])
        analyze_example(model_trained, model_base, tokenizer, yes_id, no_id,
                        test_item, cands, example_info)

    print(f"\n{'='*80}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
