"""
Advanced Debug V2: Deeper analyses.

1. Cross-issue transfer: Does the same file get high scores for unrelated issues?
   (Tests if model is memorizing file popularity vs understanding issues)
2. Training signal leakage: Score overlap between train and test repos
3. Listwise margin analysis: Gap between rank-1 and rank-2 scores
4. Loss landscape: Compute NLL loss on Yes/No for GT vs Neg files
5. Token-level probing: What does the model predict at different positions?
6. Co-change pattern detection: Does the model score test files higher for source files?
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
def score_prompt(model, tokenizer, prompt, yes_id, no_id, device):
    enc = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    out = model(input_ids=ids, attention_mask=mask)
    y = out.logits[0, -1, yes_id].item()
    n = out.logits[0, -1, no_id].item()
    return y - n, y, n


# ============================================================
# 1. CROSS-ISSUE TRANSFER TEST
# ============================================================
def cross_issue_test(model, tokenizer, yes_id, no_id, device, test_data, candidates):
    """
    Test if the model memorizes "popular files" vs understanding issues.

    For each GT file, score it against:
    a) Its correct issue text
    b) An unrelated issue from the SAME repo
    c) An unrelated issue from a DIFFERENT repo

    If (a) >> (b) >> (c), the model uses issue text properly.
    If (a) ≈ (b) >> (c), the model just memorizes per-repo file popularity.
    If (a) ≈ (b) ≈ (c), the model memorizes global file popularity.
    """
    print(f"\n{'='*80}")
    print("1. CROSS-ISSUE TRANSFER TEST")
    print(f"{'='*80}")
    print("Does the model use issue text or just memorize file popularity?\n")

    # Group examples by repo
    repo_examples = defaultdict(list)
    for key in candidates:
        if key not in test_data:
            continue
        item = test_data[key]
        gt = set(item.get("changed_py_files", []))
        cands = candidates[key]
        gt_in = gt & set(cands)
        if gt_in:
            repo_examples[item["repo"]].append((key, item, cands, gt_in))

    # Pick repos with multiple examples
    results = []
    tested = 0
    for repo in sorted(repo_examples.keys()):
        examples = repo_examples[repo]
        if len(examples) < 2:
            continue
        if tested >= 10:
            break

        # Take first 2 examples from this repo
        key1, item1, cands1, gt_in1 = examples[0]
        key2, item2, cands2, gt_in2 = examples[1]

        gt_file1 = list(gt_in1)[0]
        gt_file2 = list(gt_in2)[0]

        # Also find a different repo
        other_repo = None
        other_issue = None
        for r in repo_examples:
            if r != repo:
                other_repo = r
                other_issue = repo_examples[r][0][1]["issue_text"]
                break
        if other_issue is None:
            continue

        # Score GT file 1 against: correct issue, same-repo-other issue, other-repo issue
        s_correct, _, _ = score_prompt(model, tokenizer,
            PROMPT_TEMPLATE.format(issue_text=item1["issue_text"], candidate_path=gt_file1),
            yes_id, no_id, device)
        s_same_repo, _, _ = score_prompt(model, tokenizer,
            PROMPT_TEMPLATE.format(issue_text=item2["issue_text"], candidate_path=gt_file1),
            yes_id, no_id, device)
        s_other_repo, _, _ = score_prompt(model, tokenizer,
            PROMPT_TEMPLATE.format(issue_text=other_issue, candidate_path=gt_file1),
            yes_id, no_id, device)

        print(f"  {repo} #{item1['issue_id']} | GT: {gt_file1}")
        print(f"    Correct issue:      score={s_correct:>7.3f}")
        print(f"    Same-repo-other:    score={s_same_repo:>7.3f} (delta={s_same_repo-s_correct:+.3f})")
        print(f"    Other-repo issue:   score={s_other_repo:>7.3f} (delta={s_other_repo-s_correct:+.3f})")

        results.append({
            "repo": repo, "file": gt_file1,
            "correct": s_correct, "same_repo": s_same_repo, "other_repo": s_other_repo,
        })
        tested += 1

    # Aggregate
    if results:
        correct_scores = [r["correct"] for r in results]
        same_repo_scores = [r["same_repo"] for r in results]
        other_repo_scores = [r["other_repo"] for r in results]
        print(f"\n  AGGREGATE ({len(results)} examples):")
        print(f"    Correct issue:    mean={np.mean(correct_scores):.3f}")
        print(f"    Same-repo-other:  mean={np.mean(same_repo_scores):.3f} (delta={np.mean(same_repo_scores)-np.mean(correct_scores):+.3f})")
        print(f"    Other-repo issue: mean={np.mean(other_repo_scores):.3f} (delta={np.mean(other_repo_scores)-np.mean(correct_scores):+.3f})")

        # How many times correct > same_repo > other_repo?
        proper_ordering = sum(1 for r in results if r["correct"] > r["same_repo"] > r["other_repo"])
        partial_ordering = sum(1 for r in results if r["correct"] > r["same_repo"])
        print(f"    Correct > Same-repo > Other-repo: {proper_ordering}/{len(results)} ({proper_ordering/len(results)*100:.0f}%)")
        print(f"    Correct > Same-repo: {partial_ordering}/{len(results)} ({partial_ordering/len(results)*100:.0f}%)")

    return results


# ============================================================
# 2. CO-CHANGE PATTERN DETECTION
# ============================================================
def cochange_pattern_test(model, tokenizer, yes_id, no_id, device, test_data, candidates):
    """
    Does the model score test files higher when the source file is the GT?
    Test if the model learned co-change patterns (_test.py <-> source.py).
    """
    print(f"\n{'='*80}")
    print("2. CO-CHANGE PATTERN DETECTION")
    print(f"{'='*80}")
    print("Does the model learn test<->source co-change patterns?\n")

    test_source_pairs = []  # (key, test_file, source_file, score_test, score_source)

    for key in list(candidates.keys())[:200]:  # Sample 200
        if key not in test_data:
            continue
        item = test_data[key]
        gt = set(item.get("changed_py_files", []))
        cands = candidates[key]
        gt_in = gt & set(cands)
        if not gt_in:
            continue

        issue_text = item["issue_text"]

        for gt_file in gt_in:
            # Check if this is a test file
            if "_test.py" in gt_file or "test_" in gt_file.split("/")[-1]:
                # This is a test file, look for corresponding source
                possible_source = gt_file.replace("_test.py", ".py").replace("test_", "")
                if possible_source in cands:
                    s_test, _, _ = score_prompt(model, tokenizer,
                        PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=gt_file),
                        yes_id, no_id, device)
                    s_source, _, _ = score_prompt(model, tokenizer,
                        PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=possible_source),
                        yes_id, no_id, device)
                    test_source_pairs.append({
                        "key": key, "test_file": gt_file, "source_file": possible_source,
                        "score_test": s_test, "score_source": s_source,
                        "source_is_gt": possible_source in gt,
                    })
            else:
                # Source file, look for test
                possible_test = gt_file.replace(".py", "_test.py")
                if possible_test in cands:
                    s_source, _, _ = score_prompt(model, tokenizer,
                        PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=gt_file),
                        yes_id, no_id, device)
                    s_test, _, _ = score_prompt(model, tokenizer,
                        PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=possible_test),
                        yes_id, no_id, device)
                    test_source_pairs.append({
                        "key": key, "test_file": possible_test, "source_file": gt_file,
                        "score_test": s_test, "score_source": s_source,
                        "source_is_gt": True,
                    })

    if test_source_pairs:
        print(f"  Found {len(test_source_pairs)} test-source pairs")

        # When source is GT, how does test file score compare?
        gt_source_pairs = [p for p in test_source_pairs if p["source_is_gt"]]
        if gt_source_pairs:
            test_scores = [p["score_test"] for p in gt_source_pairs]
            source_scores = [p["score_source"] for p in gt_source_pairs]
            print(f"\n  When SOURCE is GT ({len(gt_source_pairs)} pairs):")
            print(f"    Source (GT) mean score: {np.mean(source_scores):.3f}")
            print(f"    Test file mean score:   {np.mean(test_scores):.3f}")
            print(f"    Test also scored high (>0): {sum(1 for s in test_scores if s > 0)}/{len(test_scores)}")

        # Sample output
        for p in test_source_pairs[:5]:
            gt_mark = " [GT]" if p["source_is_gt"] else ""
            print(f"\n    {p['source_file']}{gt_mark}: score={p['score_source']:.3f}")
            print(f"    {p['test_file']}: score={p['score_test']:.3f}")
    else:
        print("  No test-source pairs found in sample.")


# ============================================================
# 3. LISTWISE MARGIN ANALYSIS
# ============================================================
def margin_analysis(model, tokenizer, yes_id, no_id, device, test_data, candidates):
    """
    Analyze the gap between top-1 and top-2 scores.

    High margin = confident prediction
    Low margin = uncertain, close call

    Correlate margin with correctness.
    """
    print(f"\n{'='*80}")
    print("3. LISTWISE MARGIN ANALYSIS")
    print(f"{'='*80}")
    print("Is the model more confident when it's correct?\n")

    margins_correct = []   # top-1 is GT
    margins_incorrect = [] # top-1 is NOT GT

    count = 0
    for key in list(candidates.keys())[:100]:  # Sample 100
        if key not in test_data:
            continue
        item = test_data[key]
        gt = set(item.get("changed_py_files", []))
        cands = candidates[key]
        gt_in = gt & set(cands)
        if not gt_in or len(cands) < 5:
            continue

        issue_text = item["issue_text"]

        # Score top-5 candidates
        scores = []
        for c in cands[:10]:
            s, _, _ = score_prompt(model, tokenizer,
                PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=c),
                yes_id, no_id, device)
            scores.append((c, s))

        scores.sort(key=lambda x: -x[1])
        top1_file, top1_score = scores[0]
        top2_file, top2_score = scores[1]
        margin = top1_score - top2_score

        is_correct = top1_file in gt
        if is_correct:
            margins_correct.append(margin)
        else:
            margins_incorrect.append(margin)

        count += 1
        if count % 20 == 0:
            print(f"  Processed {count}...", end='\r')

    print(f"  Processed {count} examples.      ")

    if margins_correct and margins_incorrect:
        print(f"\n  Correct top-1 ({len(margins_correct)} examples):")
        print(f"    Mean margin: {np.mean(margins_correct):.3f}")
        print(f"    Median margin: {np.median(margins_correct):.3f}")
        print(f"  Incorrect top-1 ({len(margins_incorrect)} examples):")
        print(f"    Mean margin: {np.mean(margins_incorrect):.3f}")
        print(f"    Median margin: {np.median(margins_incorrect):.3f}")
        print(f"\n  High-confidence correct (margin>2): {sum(1 for m in margins_correct if m > 2)}/{len(margins_correct)}")
        print(f"  High-confidence wrong (margin>2): {sum(1 for m in margins_incorrect if m > 2)}/{len(margins_incorrect)}")
        print(f"  Low-confidence correct (margin<0.5): {sum(1 for m in margins_correct if m < 0.5)}/{len(margins_correct)}")
        print(f"  Low-confidence wrong (margin<0.5): {sum(1 for m in margins_incorrect if m < 0.5)}/{len(margins_incorrect)}")


# ============================================================
# 4. FILE TYPE BIAS ANALYSIS
# ============================================================
def file_type_bias(model, tokenizer, yes_id, no_id, device, test_data, candidates):
    """
    Analyze model bias toward certain file types/patterns.
    Does the model have a prior that __init__.py or test files are more likely?
    """
    print(f"\n{'='*80}")
    print("4. FILE TYPE BIAS ANALYSIS")
    print(f"{'='*80}")
    print("Does the model have biases toward certain file patterns?\n")

    # Use a neutral issue text
    neutral_issue = "Fix a bug in the application."

    file_patterns = [
        "src/module/main.py",
        "src/module/__init__.py",
        "src/module/utils.py",
        "tests/test_main.py",
        "tests/test_module.py",
        "setup.py",
        "src/module/core.py",
        "src/module/config.py",
        "docs/conf.py",
        "examples/demo.py",
        "src/module/models.py",
        "src/module/views.py",
        "src/module/api.py",
        "src/module/exceptions.py",
    ]

    print(f"  Scores for synthetic file paths (neutral issue text):")
    scores = []
    for fp in file_patterns:
        s, y, n = score_prompt(model, tokenizer,
            PROMPT_TEMPLATE.format(issue_text=neutral_issue, candidate_path=fp),
            yes_id, no_id, device)
        p_yes = 1.0 / (1.0 + np.exp(-s))
        scores.append((fp, s, p_yes))
        print(f"    {fp:<45} score={s:>7.3f}  P(Yes)={p_yes:.3f}")

    # Compare init vs non-init
    init_scores = [s for f, s, _ in scores if "__init__" in f]
    test_scores = [s for f, s, _ in scores if "test" in f]
    other_scores = [s for f, s, _ in scores if "__init__" not in f and "test" not in f]

    print(f"\n  Average by type:")
    if init_scores:
        print(f"    __init__.py: {np.mean(init_scores):.3f}")
    if test_scores:
        print(f"    test files:  {np.mean(test_scores):.3f}")
    if other_scores:
        print(f"    other files: {np.mean(other_scores):.3f}")


# ============================================================
# 5. LOSS ANALYSIS: NLL on Yes/No
# ============================================================
@torch.no_grad()
def compute_nll(model, tokenizer, prompt, yes_id, no_id, device):
    """Compute negative log-likelihood of 'Yes' and 'No' tokens."""
    enc = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    out = model(input_ids=ids, attention_mask=mask)
    logits = out.logits[0, -1]  # last position
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    nll_yes = -log_probs[yes_id].item()
    nll_no = -log_probs[no_id].item()
    return nll_yes, nll_no


def loss_analysis(model, tokenizer, yes_id, no_id, device, test_data, candidates):
    """Compute NLL loss for GT (should predict Yes) and Neg (should predict No) files."""
    print(f"\n{'='*80}")
    print("5. LOSS ANALYSIS (NLL)")
    print(f"{'='*80}")
    print("How confident is the model's Yes/No prediction?\n")

    gt_nll_yes = []
    gt_nll_no = []
    neg_nll_yes = []
    neg_nll_no = []

    count = 0
    for key in list(candidates.keys())[:50]:
        if key not in test_data:
            continue
        item = test_data[key]
        gt = set(item.get("changed_py_files", []))
        cands = candidates[key]
        gt_in = gt & set(cands)
        if not gt_in:
            continue

        issue_text = item["issue_text"]

        # Score GT files
        for gf in list(gt_in)[:2]:
            prompt = PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=gf)
            nll_y, nll_n = compute_nll(model, tokenizer, prompt, yes_id, no_id, device)
            gt_nll_yes.append(nll_y)
            gt_nll_no.append(nll_n)

        # Score 2 negative files
        negs = [c for c in cands if c not in gt][:2]
        for nf in negs:
            prompt = PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=nf)
            nll_y, nll_n = compute_nll(model, tokenizer, prompt, yes_id, no_id, device)
            neg_nll_yes.append(nll_y)
            neg_nll_no.append(nll_n)

        count += 1

    print(f"  Analyzed {count} examples")
    print(f"\n  GT files (should predict 'Yes'):")
    print(f"    NLL(Yes): {np.mean(gt_nll_yes):.3f} ± {np.std(gt_nll_yes):.3f}")
    print(f"    NLL(No):  {np.mean(gt_nll_no):.3f} ± {np.std(gt_nll_no):.3f}")
    print(f"  Neg files (should predict 'No'):")
    print(f"    NLL(Yes): {np.mean(neg_nll_yes):.3f} ± {np.std(neg_nll_yes):.3f}")
    print(f"    NLL(No):  {np.mean(neg_nll_no):.3f} ± {np.std(neg_nll_no):.3f}")

    # Ideal: GT NLL(Yes) low, GT NLL(No) high; Neg NLL(Yes) high, Neg NLL(No) low
    print(f"\n  Cross-entropy loss if using model as classifier:")
    gt_loss = np.mean(gt_nll_yes)  # Loss on GT (target=Yes)
    neg_loss = np.mean(neg_nll_no)  # Loss on Neg (target=No)
    print(f"    GT loss (predict Yes):  {gt_loss:.3f}")
    print(f"    Neg loss (predict No):  {neg_loss:.3f}")
    print(f"    Average:                {(gt_loss + neg_loss) / 2:.3f}")


def main():
    print("=" * 80)
    print("ADVANCED DEBUG V2: Deeper Behavioral Analysis")
    print("=" * 80)

    test_data, candidates = load_data()

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
    print("Model loaded.\n")

    # Run all analyses
    cross_issue_test(model, tokenizer, yes_id, no_id, DEVICE, test_data, candidates)
    cochange_pattern_test(model, tokenizer, yes_id, no_id, DEVICE, test_data, candidates)
    margin_analysis(model, tokenizer, yes_id, no_id, DEVICE, test_data, candidates)
    file_type_bias(model, tokenizer, yes_id, no_id, DEVICE, test_data, candidates)
    loss_analysis(model, tokenizer, yes_id, no_id, DEVICE, test_data, candidates)

    print(f"\n{'='*80}")
    print("ADVANCED DEBUG V2 COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
