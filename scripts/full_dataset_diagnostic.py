"""
Full-dataset diagnostic for RankFT cross-encoder.

Runs on ALL test examples and produces aggregate statistics:
1. Score distribution: GT vs non-GT files (histogram-style)
2. Rank improvement distribution: how many positions GT files move
3. Error categorization: WHY do failures happen?
4. Per-repo breakdown of model effectiveness
5. Calibration analysis: P(Yes) vs actual positive rate
6. Candidate coverage analysis
"""

import json
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

torch.manual_seed(42)
np.random.seed(42)

MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
LORA_PATH = "/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/best"
TEST_DATA = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_test.jsonl"
CANDIDATES = "/home/chenlibin/grepo_agent/data/rankft/exp6_expanded_candidates.jsonl"
DEVICE = "cuda:0"
MAX_SEQ_LEN = 512
BATCH_SIZE = 16

PROMPT_TEMPLATE = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)


def load_data():
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


def get_yes_no_ids(tokenizer):
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    return yes_ids[0], no_ids[0]


@torch.no_grad()
def score_batch(model, tokenizer, prompts, yes_id, no_id, device):
    """Score a batch of prompts, return list of (yes_logit, no_logit, score)."""
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_SEQ_LEN,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            results = []
            for p in prompts:
                enc1 = tokenizer([p], return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
                ids = enc1["input_ids"].to(device)
                mask = enc1["attention_mask"].to(device)
                out = model(input_ids=ids, attention_mask=mask)
                y = out.logits[0, -1, yes_id].item()
                n = out.logits[0, -1, no_id].item()
                results.append((y, n, y - n))
            return results
        raise

    logits = outputs.logits
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(logits.size(0), device=device)
    last_logits = logits[batch_idx, seq_lengths]

    yes_logits = last_logits[:, yes_id].cpu().tolist()
    no_logits = last_logits[:, no_id].cpu().tolist()

    return [(y, n, y - n) for y, n in zip(yes_logits, no_logits)]


def score_all_candidates(model, tokenizer, issue_text, cands, yes_id, no_id, device):
    """Score all candidates for one example."""
    prompts = [PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=c) for c in cands]
    all_scores = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i+BATCH_SIZE]
        all_scores.extend(score_batch(model, tokenizer, batch, yes_id, no_id, device))
    return all_scores


def main():
    print("=" * 80)
    print("FULL-DATASET DIAGNOSTIC: RankFT Cross-Encoder")
    print("=" * 80)

    # Load data
    test_data, candidates = load_data()
    keys = [k for k in candidates if k in test_data and test_data[k].get("changed_py_files")]
    print(f"Examples with candidates AND GT: {len(keys)}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    yes_id, no_id = get_yes_no_ids(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    print("Model loaded.")

    # ========== Aggregate collectors ==========
    all_gt_scores = []          # scores of GT files
    all_neg_scores = []         # scores of non-GT files
    all_gt_ranks = []           # rank of each GT file
    all_gt_p_yes = []           # P(Yes) for GT files
    all_neg_p_yes = []          # P(Yes) for non-GT files
    rank_improvements = []      # (base_rank - trained_rank) for GT files
    # Note: we don't have base model here, so rank_improvement = input_order_rank - reranked_rank

    # Error categorization
    error_cats = defaultdict(int)
    # Categories: gt_not_in_cands, gt_rank1, gt_top5, gt_top10, gt_below10

    # Per-repo stats
    repo_stats = defaultdict(lambda: {
        "total": 0, "gt_in_cands": 0, "hit1": 0, "hit5": 0, "hit10": 0,
        "gt_scores": [], "neg_scores": [], "gt_ranks": [],
    })

    # Score distribution bins
    score_bins_gt = defaultdict(int)    # binned GT scores
    score_bins_neg = defaultdict(int)   # binned non-GT scores

    # Calibration buckets: group by P(Yes) range, count actual positives
    calib_buckets = defaultdict(lambda: {"total": 0, "positive": 0})

    total = len(keys)
    import time
    start = time.time()

    for idx, key in enumerate(keys):
        item = test_data[key]
        repo = item["repo"]
        issue_text = item["issue_text"]
        gt = set(item["changed_py_files"])
        cands = candidates[key]

        if not cands:
            continue

        gt_in_cands = gt & set(cands)
        repo_stats[repo]["total"] += 1

        if not gt_in_cands:
            error_cats["gt_not_in_cands"] += 1
            # Still score to get neg distribution
            scores = score_all_candidates(model, tokenizer, issue_text, cands, yes_id, no_id, DEVICE)
            for (y, n, s) in scores:
                all_neg_scores.append(s)
                p_yes = 1.0 / (1.0 + np.exp(-s))  # sigmoid
                all_neg_p_yes.append(p_yes)
                bucket = int(p_yes * 10) / 10  # 0.0, 0.1, ..., 0.9
                calib_buckets[bucket]["total"] += 1
                # bin
                bin_key = round(s * 2) / 2  # 0.5-wide bins
                score_bins_neg[bin_key] += 1
        else:
            repo_stats[repo]["gt_in_cands"] += 1
            scores = score_all_candidates(model, tokenizer, issue_text, cands, yes_id, no_id, DEVICE)

            # Build scored list
            scored = [(cands[i], scores[i]) for i in range(len(cands))]
            scored.sort(key=lambda x: -x[1][2])  # sort by score descending (x[1] = (y,n,s))
            reranked = [c for c, _ in scored]

            # Compute metrics
            for i, (cand, (y, n, s)) in enumerate(scored):
                is_gt = cand in gt
                p_yes = 1.0 / (1.0 + np.exp(-s))
                bucket = int(p_yes * 10) / 10
                calib_buckets[bucket]["total"] += 1

                if is_gt:
                    all_gt_scores.append(s)
                    all_gt_p_yes.append(p_yes)
                    all_gt_ranks.append(i + 1)
                    calib_buckets[bucket]["positive"] += 1
                    repo_stats[repo]["gt_scores"].append(s)
                    repo_stats[repo]["gt_ranks"].append(i + 1)
                    bin_key = round(s * 2) / 2
                    score_bins_gt[bin_key] += 1

                    # Original rank (position in input candidate list)
                    orig_rank = cands.index(cand) + 1
                    rank_improvements.append(orig_rank - (i + 1))
                else:
                    all_neg_scores.append(s)
                    all_neg_p_yes.append(p_yes)
                    repo_stats[repo]["neg_scores"].append(s)
                    bin_key = round(s * 2) / 2
                    score_bins_neg[bin_key] += 1

            # Hit metrics
            best_gt_rank = min(reranked.index(f) + 1 for f in gt_in_cands)
            if best_gt_rank == 1:
                error_cats["gt_rank1"] += 1
                repo_stats[repo]["hit1"] += 1
            elif best_gt_rank <= 5:
                error_cats["gt_top5_not1"] += 1
            elif best_gt_rank <= 10:
                error_cats["gt_top10_not5"] += 1
            else:
                error_cats["gt_below10"] += 1

            if best_gt_rank <= 5:
                repo_stats[repo]["hit5"] += 1
            if best_gt_rank <= 10:
                repo_stats[repo]["hit10"] += 1

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed
            eta = (total - idx - 1) / rate
            print(f"  [{idx+1}/{total}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    elapsed = time.time() - start
    print(f"\nScoring complete: {total} examples in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # ========== Print Results ==========
    print(f"\n{'='*80}")
    print("1. ERROR CATEGORIZATION")
    print(f"{'='*80}")
    total_examples = sum(error_cats.values())
    for cat in ["gt_rank1", "gt_top5_not1", "gt_top10_not5", "gt_below10", "gt_not_in_cands"]:
        count = error_cats[cat]
        pct = count / total_examples * 100 if total_examples else 0
        print(f"  {cat:<20}: {count:>5} ({pct:>5.1f}%)")

    print(f"\n{'='*80}")
    print("2. SCORE DISTRIBUTION (GT vs Non-GT)")
    print(f"{'='*80}")
    if all_gt_scores:
        print(f"  GT files:     mean={np.mean(all_gt_scores):.3f}, std={np.std(all_gt_scores):.3f}, "
              f"median={np.median(all_gt_scores):.3f}, min={min(all_gt_scores):.3f}, max={max(all_gt_scores):.3f}")
    print(f"  Non-GT files: mean={np.mean(all_neg_scores):.3f}, std={np.std(all_neg_scores):.3f}, "
          f"median={np.median(all_neg_scores):.3f}, min={min(all_neg_scores):.3f}, max={max(all_neg_scores):.3f}")
    if all_gt_scores:
        margin = np.mean(all_gt_scores) - np.mean(all_neg_scores)
        print(f"  Mean margin (GT - Neg): {margin:+.3f}")
        # Overlap: what fraction of negatives score higher than the average GT?
        gt_mean = np.mean(all_gt_scores)
        neg_above_gt = sum(1 for s in all_neg_scores if s > gt_mean)
        print(f"  Non-GT files scoring above GT mean: {neg_above_gt}/{len(all_neg_scores)} "
              f"({neg_above_gt/len(all_neg_scores)*100:.1f}%)")

    # Histogram (text-based)
    print(f"\n  Score histogram (0.5-wide bins):")
    all_bins = sorted(set(list(score_bins_gt.keys()) + list(score_bins_neg.keys())))
    for b in all_bins:
        gt_c = score_bins_gt.get(b, 0)
        neg_c = score_bins_neg.get(b, 0)
        gt_bar = "#" * min(gt_c // 5, 40)
        neg_bar = "." * min(neg_c // 50, 40)
        if gt_c > 0 or neg_c > 0:
            print(f"    [{b:>5.1f}] GT:{gt_c:>5} {gt_bar}")
            print(f"           Neg:{neg_c:>5} {neg_bar}")

    print(f"\n{'='*80}")
    print("3. GT FILE RANK DISTRIBUTION")
    print(f"{'='*80}")
    if all_gt_ranks:
        print(f"  Total GT files in candidates: {len(all_gt_ranks)}")
        print(f"  Mean rank: {np.mean(all_gt_ranks):.1f}, Median: {np.median(all_gt_ranks):.1f}")
        for k in [1, 3, 5, 10, 20]:
            in_topk = sum(1 for r in all_gt_ranks if r <= k)
            print(f"  GT in top-{k:>2}: {in_topk}/{len(all_gt_ranks)} ({in_topk/len(all_gt_ranks)*100:.1f}%)")

    print(f"\n{'='*80}")
    print("4. RANK IMPROVEMENT (original order -> reranked)")
    print(f"{'='*80}")
    if rank_improvements:
        improved = sum(1 for r in rank_improvements if r > 0)
        same = sum(1 for r in rank_improvements if r == 0)
        worsened = sum(1 for r in rank_improvements if r < 0)
        print(f"  Improved: {improved}/{len(rank_improvements)} ({improved/len(rank_improvements)*100:.1f}%)")
        print(f"  Same:     {same}/{len(rank_improvements)} ({same/len(rank_improvements)*100:.1f}%)")
        print(f"  Worsened: {worsened}/{len(rank_improvements)} ({worsened/len(rank_improvements)*100:.1f}%)")
        print(f"  Mean improvement: {np.mean(rank_improvements):+.1f} positions")
        print(f"  Median improvement: {np.median(rank_improvements):+.1f} positions")
        # Distribution
        for bucket in [(-100, -10), (-10, -5), (-5, -1), (-1, 0), (0, 1), (1, 5), (5, 10), (10, 100)]:
            lo, hi = bucket
            count = sum(1 for r in rank_improvements if lo < r <= hi)
            if count > 0:
                print(f"    ({lo:>4}, {hi:>4}]: {count}")

    print(f"\n{'='*80}")
    print("5. CALIBRATION ANALYSIS")
    print(f"{'='*80}")
    print(f"  {'P(Yes) range':<15} {'Total':>8} {'Positive':>8} {'Actual %':>8}")
    for bucket in sorted(calib_buckets.keys()):
        b = calib_buckets[bucket]
        actual_pct = b["positive"] / b["total"] * 100 if b["total"] else 0
        print(f"  [{bucket:.1f}, {bucket+0.1:.1f})    {b['total']:>8} {b['positive']:>8} {actual_pct:>7.1f}%")

    print(f"\n{'='*80}")
    print("6. PER-REPO BREAKDOWN (top-20 by size)")
    print(f"{'='*80}")
    sorted_repos = sorted(repo_stats.items(), key=lambda x: -x[1]["total"])
    print(f"  {'Repo':<25} {'N':>4} {'GTin':>4} {'H@1%':>6} {'H@5%':>6} {'H@10%':>6} "
          f"{'GT_score':>8} {'Neg_score':>9} {'Margin':>7} {'MedRank':>7}")
    for repo, st in sorted_repos[:25]:
        h1 = st["hit1"] / st["gt_in_cands"] * 100 if st["gt_in_cands"] else 0
        h5 = st["hit5"] / st["gt_in_cands"] * 100 if st["gt_in_cands"] else 0
        h10 = st["hit10"] / st["gt_in_cands"] * 100 if st["gt_in_cands"] else 0
        gt_s = np.mean(st["gt_scores"]) if st["gt_scores"] else float('nan')
        neg_s = np.mean(st["neg_scores"]) if st["neg_scores"] else float('nan')
        margin = gt_s - neg_s if st["gt_scores"] and st["neg_scores"] else float('nan')
        med_rank = np.median(st["gt_ranks"]) if st["gt_ranks"] else float('nan')
        print(f"  {repo:<25} {st['total']:>4} {st['gt_in_cands']:>4} {h1:>5.1f}% {h5:>5.1f}% {h10:>5.1f}% "
              f"{gt_s:>8.2f} {neg_s:>9.2f} {margin:>+7.2f} {med_rank:>7.1f}")

    # Save detailed results
    output = {
        "error_cats": dict(error_cats),
        "score_stats": {
            "gt_mean": float(np.mean(all_gt_scores)) if all_gt_scores else None,
            "gt_std": float(np.std(all_gt_scores)) if all_gt_scores else None,
            "neg_mean": float(np.mean(all_neg_scores)),
            "neg_std": float(np.std(all_neg_scores)),
            "margin": float(np.mean(all_gt_scores) - np.mean(all_neg_scores)) if all_gt_scores else None,
        },
        "rank_stats": {
            "mean_rank": float(np.mean(all_gt_ranks)) if all_gt_ranks else None,
            "median_rank": float(np.median(all_gt_ranks)) if all_gt_ranks else None,
            "mean_improvement": float(np.mean(rank_improvements)) if rank_improvements else None,
            "pct_improved": float(sum(1 for r in rank_improvements if r > 0) / len(rank_improvements) * 100) if rank_improvements else None,
        },
        "calibration": {str(k): v for k, v in sorted(calib_buckets.items())},
    }
    out_path = "/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/full_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
