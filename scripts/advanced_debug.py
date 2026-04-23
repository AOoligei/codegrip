"""
Advanced debugging tools for RankFT cross-encoder.

Paper-level analyses:
1. Gradient attribution: which input tokens most influence Yes/No prediction
2. Attention analysis: cross-attention between issue tokens and file path tokens
3. LoRA weight analysis: which layers changed most, where is graph knowledge stored
4. Embedding space analysis: representations of GT vs non-GT files
5. Score sensitivity: how does score change with path perturbations
"""

import json
import os
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
OUTPUT_DIR = "/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/advanced_debug"

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


def pick_examples(test_data, candidates, n=3):
    """Pick examples where GT is in candidates."""
    selected = []
    for key in candidates:
        if key not in test_data:
            continue
        item = test_data[key]
        gt = set(item.get("changed_py_files", []))
        cands = candidates[key]
        gt_in = gt & set(cands)
        if gt_in and len(cands) >= 10:
            selected.append((key, item, cands, gt, gt_in))
            if len(selected) >= n:
                break
    return selected


# ============================================================
# 1. GRADIENT ATTRIBUTION
# ============================================================
def gradient_attribution(model, tokenizer, prompt, yes_id, no_id, device):
    """
    Compute input-gradient attribution: which tokens most influence score.
    Uses grad of (yes_logit - no_logit) w.r.t. input embeddings.
    """
    model.zero_grad()

    enc = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Get embeddings and enable gradient
    embeddings = model.get_input_embeddings()(input_ids)
    embeddings = embeddings.detach().requires_grad_(True)

    # Forward with embeddings
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    last_logits = outputs.logits[0, -1]
    score = last_logits[yes_id] - last_logits[no_id]

    score.backward()

    # Gradient * embedding (element-wise), then L2 norm per token
    grad = embeddings.grad[0]  # (seq_len, hidden_dim)
    attr = (grad * embeddings[0].detach()).float().norm(dim=-1)  # (seq_len,)
    attr = attr.cpu().numpy()

    # Normalize
    attr = attr / (attr.max() + 1e-10)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

    return tokens, attr, score.item()


def analyze_gradient_attribution(model, tokenizer, yes_id, no_id, device, examples):
    """Run gradient attribution on selected examples."""
    print(f"\n{'='*80}")
    print("GRADIENT ATTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    print("Shows which input tokens most influence the Yes-No score.\n")

    results = []

    for key, item, cands, gt, gt_in in examples:
        repo = item["repo"]
        issue_id = item["issue_id"]
        issue_text = item["issue_text"]

        # Pick one GT file and one non-GT file
        gt_file = list(gt_in)[0]
        non_gt_file = next(c for c in cands if c not in gt)

        for label, fpath in [("GT", gt_file), ("NEG", non_gt_file)]:
            prompt = PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=fpath)
            tokens, attr, score = gradient_attribution(model, tokenizer, prompt, yes_id, no_id, device)

            print(f"  {repo}#{issue_id} | {label}: {fpath} | Score={score:.3f}")

            # Find the file path tokens and issue tokens
            # Locate "File:" and "Bug Report:" in tokens
            token_strs = [tokenizer.decode([tid]) for tid in tokenizer.encode(prompt, add_special_tokens=False)]

            # Top-10 most attributed tokens
            top_idx = np.argsort(attr)[-15:][::-1]
            print(f"    Top-15 attributed tokens:")
            for ti in top_idx:
                if ti < len(tokens):
                    print(f"      [{ti:>3}] attr={attr[ti]:.3f} token='{tokens[ti]}'")

            # Aggregate attribution by region
            prompt_text = prompt
            # Find regions
            bug_report_start = prompt_text.find("Bug Report:")
            file_start = prompt_text.find("\nFile:")
            answer_start = prompt_text.find("\nAnswer:")

            # Map char positions to token positions (approximate)
            # Build cumulative char length
            cum_len = 0
            token_char_pos = []
            for t in tokens:
                token_char_pos.append(cum_len)
                cum_len += len(t.replace("Ġ", " ").replace("▁", " "))

            region_attr = {"preamble": 0, "issue_text": 0, "file_path": 0, "answer": 0}
            region_count = {"preamble": 0, "issue_text": 0, "file_path": 0, "answer": 0}
            for i, (tok, a) in enumerate(zip(tokens, attr)):
                # Rough region assignment based on token index
                if i < len(tokens) * 0.1:
                    region = "preamble"
                elif i < len(tokens) * 0.85:
                    region = "issue_text"
                elif i < len(tokens) * 0.95:
                    region = "file_path"
                else:
                    region = "answer"
                region_attr[region] += a
                region_count[region] += 1

            print(f"    Region attribution (mean):")
            for region in ["preamble", "issue_text", "file_path", "answer"]:
                mean_a = region_attr[region] / max(region_count[region], 1)
                print(f"      {region:<15}: mean_attr={mean_a:.4f} (sum={region_attr[region]:.3f}, n={region_count[region]})")

            results.append({
                "key": key, "label": label, "file": fpath, "score": score,
                "top_tokens": [(int(ti), float(attr[ti]), tokens[ti]) for ti in top_idx if ti < len(tokens)],
            })
            print()

    return results


# ============================================================
# 2. ATTENTION ANALYSIS
# ============================================================
def attention_analysis(model, tokenizer, prompt, device):
    """Extract attention patterns from the last layer."""
    enc = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_attentions=True)

    # Last layer attention: (1, num_heads, seq_len, seq_len)
    if outputs.attentions is None:
        return None, None
    last_attn = outputs.attentions[-1][0]  # (num_heads, seq_len, seq_len)
    # Average over heads
    avg_attn = last_attn.mean(dim=0)  # (seq_len, seq_len)
    # Last token's attention to all other tokens
    last_token_attn = avg_attn[-1].float().cpu().numpy()  # (seq_len,)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

    return tokens, last_token_attn


def analyze_attention(model, tokenizer, device, examples):
    """Analyze where the model looks when making the Yes/No decision."""
    print(f"\n{'='*80}")
    print("ATTENTION ANALYSIS (Last Layer, Last Token -> All)")
    print(f"{'='*80}")
    print("Where does the model look when deciding Yes/No?\n")

    for key, item, cands, gt, gt_in in examples[:2]:
        repo = item["repo"]
        issue_id = item["issue_id"]
        issue_text = item["issue_text"]
        gt_file = list(gt_in)[0]
        non_gt_file = next(c for c in cands if c not in gt)

        for label, fpath in [("GT", gt_file), ("NEG", non_gt_file)]:
            prompt = PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=fpath)
            tokens, attn = attention_analysis(model, tokenizer, prompt, device)

            if tokens is None:
                print(f"  {repo}#{issue_id} | {label}: Attention not available (sdpa backend, skipping)")
                continue

            print(f"  {repo}#{issue_id} | {label}: {fpath}")

            # Top-10 most attended tokens
            top_idx = np.argsort(attn)[-10:][::-1]
            print(f"    Top-10 attended tokens (by last token):")
            for ti in top_idx:
                if ti < len(tokens):
                    print(f"      [{ti:>3}] attn={attn[ti]:.4f} token='{tokens[ti]}'")

            # Aggregate attention by region (same rough split)
            n = len(tokens)
            regions = {
                "preamble (0-10%)": attn[:int(n*0.1)].sum(),
                "issue (10-85%)": attn[int(n*0.1):int(n*0.85)].sum(),
                "file_path (85-95%)": attn[int(n*0.85):int(n*0.95)].sum(),
                "answer (95-100%)": attn[int(n*0.95):].sum(),
            }
            print(f"    Region attention mass:")
            for region, mass in regions.items():
                print(f"      {region:<25}: {mass:.4f}")
            print()


# ============================================================
# 3. LoRA WEIGHT ANALYSIS
# ============================================================
def analyze_lora_weights(model):
    """Analyze LoRA adapter weights: which layers have the largest deltas."""
    print(f"\n{'='*80}")
    print("LoRA WEIGHT ANALYSIS")
    print(f"{'='*80}")
    print("Which layers have the largest LoRA contributions?\n")

    lora_info = []
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            norm = param.data.float().norm().item()
            shape = tuple(param.shape)
            lora_info.append((name, norm, shape, param.data.float().abs().mean().item()))

    # Sort by norm
    lora_info.sort(key=lambda x: -x[1])

    print(f"  Total LoRA parameters: {len(lora_info)}")
    total_params = sum(p.numel() for n, p in model.named_parameters() if "lora" in n.lower())
    print(f"  Total LoRA param count: {total_params:,}")

    print(f"\n  Top-20 LoRA matrices by Frobenius norm:")
    print(f"  {'Name':<70} {'Norm':>8} {'Mean|W|':>8} {'Shape'}")
    for name, norm, shape, mean_abs in lora_info[:20]:
        # Shorten name
        short = name.replace("base_model.model.model.", "").replace(".default", "")
        print(f"  {short:<70} {norm:>8.4f} {mean_abs:>8.6f} {shape}")

    # Aggregate by layer
    layer_norms = defaultdict(float)
    for name, norm, shape, mean_abs in lora_info:
        # Extract layer number
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                layer_num = int(parts[i+1])
                layer_norms[layer_num] += norm
                break

    if layer_norms:
        print(f"\n  LoRA norm by transformer layer:")
        for layer in sorted(layer_norms.keys()):
            bar = "#" * int(layer_norms[layer] / max(layer_norms.values()) * 40)
            print(f"    Layer {layer:>2}: {layer_norms[layer]:>8.4f} {bar}")

    # Aggregate by module type (q_proj, k_proj, v_proj, o_proj, etc.)
    module_norms = defaultdict(float)
    module_counts = defaultdict(int)
    for name, norm, shape, mean_abs in lora_info:
        for module_type in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            if module_type in name:
                module_norms[module_type] += norm
                module_counts[module_type] += 1
                break

    print(f"\n  LoRA norm by module type:")
    for mod in sorted(module_norms.keys(), key=lambda x: -module_norms[x]):
        avg = module_norms[mod] / max(module_counts[mod], 1)
        print(f"    {mod:<12}: total_norm={module_norms[mod]:>8.4f}, count={module_counts[mod]:>3}, avg={avg:.4f}")


# ============================================================
# 4. EMBEDDING SPACE ANALYSIS
# ============================================================
@torch.no_grad()
def get_hidden_states(model, tokenizer, prompt, device):
    """Get the hidden state at the last token position (before LM head)."""
    enc = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                   output_hidden_states=True)
    # Last layer, last token
    last_hidden = outputs.hidden_states[-1][0, -1]  # (hidden_dim,)
    return last_hidden.float().cpu().numpy()


def analyze_embeddings(model, tokenizer, device, examples):
    """Analyze if GT and non-GT files separate in embedding space."""
    print(f"\n{'='*80}")
    print("EMBEDDING SPACE ANALYSIS")
    print(f"{'='*80}")
    print("Do GT and non-GT files separate in the model's representation space?\n")

    for key, item, cands, gt, gt_in in examples[:2]:
        repo = item["repo"]
        issue_id = item["issue_id"]
        issue_text = item["issue_text"]

        gt_embeddings = []
        neg_embeddings = []

        for cand in cands[:30]:  # Limit for speed
            prompt = PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=cand)
            h = get_hidden_states(model, tokenizer, prompt, device)
            if cand in gt:
                gt_embeddings.append(h)
            else:
                neg_embeddings.append(h)

        if not gt_embeddings:
            print(f"  {repo}#{issue_id}: no GT files in top-30 candidates")
            continue

        gt_emb = np.stack(gt_embeddings)
        neg_emb = np.stack(neg_embeddings)

        # Cosine similarity analysis
        gt_centroid = gt_emb.mean(axis=0)
        neg_centroid = neg_emb.mean(axis=0)

        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

        gt_gt_sim = np.mean([cosine_sim(e, gt_centroid) for e in gt_emb])
        neg_gt_sim = np.mean([cosine_sim(e, gt_centroid) for e in neg_emb])
        gt_neg_sim = np.mean([cosine_sim(e, neg_centroid) for e in gt_emb])
        neg_neg_sim = np.mean([cosine_sim(e, neg_centroid) for e in neg_emb])

        # L2 distance
        gt_gt_dist = np.mean([np.linalg.norm(e - gt_centroid) for e in gt_emb])
        neg_gt_dist = np.mean([np.linalg.norm(e - gt_centroid) for e in neg_emb])

        print(f"  {repo}#{issue_id} ({len(gt_emb)} GT, {len(neg_emb)} Neg in top-30):")
        print(f"    Cosine sim to GT centroid:  GT={gt_gt_sim:.4f}, Neg={neg_gt_sim:.4f}, gap={gt_gt_sim-neg_gt_sim:+.4f}")
        print(f"    Cosine sim to Neg centroid: GT={gt_neg_sim:.4f}, Neg={neg_neg_sim:.4f}")
        print(f"    L2 dist to GT centroid:     GT={gt_gt_dist:.4f}, Neg={neg_gt_dist:.4f}")

        # Norm analysis
        gt_norms = np.linalg.norm(gt_emb, axis=1)
        neg_norms = np.linalg.norm(neg_emb, axis=1)
        print(f"    Hidden state norms:         GT={gt_norms.mean():.2f}±{gt_norms.std():.2f}, "
              f"Neg={neg_norms.mean():.2f}±{neg_norms.std():.2f}")
        print()


# ============================================================
# 5. SCORE SENSITIVITY (PATH PERTURBATION)
# ============================================================
@torch.no_grad()
def score_prompt(model, tokenizer, prompt, yes_id, no_id, device):
    enc = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    input_ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    out = model(input_ids=input_ids, attention_mask=mask)
    y = out.logits[0, -1, yes_id].item()
    n = out.logits[0, -1, no_id].item()
    return y - n


def analyze_sensitivity(model, tokenizer, yes_id, no_id, device, examples):
    """Test how sensitive the model is to path perturbations."""
    print(f"\n{'='*80}")
    print("SCORE SENSITIVITY TO PATH PERTURBATIONS")
    print(f"{'='*80}")
    print("How does the score change when we modify the file path?\n")

    for key, item, cands, gt, gt_in in examples[:2]:
        repo = item["repo"]
        issue_id = item["issue_id"]
        issue_text = item["issue_text"]
        gt_file = list(gt_in)[0]

        print(f"  {repo}#{issue_id} | GT file: {gt_file}")

        # Original score
        prompt = PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=gt_file)
        orig_score = score_prompt(model, tokenizer, prompt, yes_id, no_id, device)
        print(f"    Original score: {orig_score:.3f}")

        # Perturbation 1: Remove directory (just basename)
        basename = os.path.basename(gt_file)
        prompt1 = PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=basename)
        s1 = score_prompt(model, tokenizer, prompt1, yes_id, no_id, device)
        print(f"    Basename only ({basename}): {s1:.3f} (delta={s1-orig_score:+.3f})")

        # Perturbation 2: Shuffle directory components
        parts = gt_file.split("/")
        if len(parts) > 2:
            shuffled = "/".join(reversed(parts[:-1])) + "/" + parts[-1]
            prompt2 = PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=shuffled)
            s2 = score_prompt(model, tokenizer, prompt2, yes_id, no_id, device)
            print(f"    Reversed dirs ({shuffled}): {s2:.3f} (delta={s2-orig_score:+.3f})")

        # Perturbation 3: Wrong file extension
        wrong_ext = gt_file.replace(".py", ".js")
        prompt3 = PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=wrong_ext)
        s3 = score_prompt(model, tokenizer, prompt3, yes_id, no_id, device)
        print(f"    Wrong ext ({wrong_ext}): {s3:.3f} (delta={s3-orig_score:+.3f})")

        # Perturbation 4: Random file from same repo
        random_file = next(c for c in cands if c not in gt)
        prompt4 = PROMPT_TEMPLATE.format(issue_text=issue_text, candidate_path=random_file)
        s4 = score_prompt(model, tokenizer, prompt4, yes_id, no_id, device)
        print(f"    Random neg ({random_file}): {s4:.3f} (delta={s4-orig_score:+.3f})")

        # Perturbation 5: Empty issue text
        prompt5 = PROMPT_TEMPLATE.format(issue_text="", candidate_path=gt_file)
        s5 = score_prompt(model, tokenizer, prompt5, yes_id, no_id, device)
        print(f"    Empty issue text: {s5:.3f} (delta={s5-orig_score:+.3f})")

        # Perturbation 6: Unrelated issue text
        # Use issue text from a different example
        other_keys = [k for k in examples if k[0] != key]
        if other_keys:
            other_issue = other_keys[0][1]["issue_text"]
            prompt6 = PROMPT_TEMPLATE.format(issue_text=other_issue, candidate_path=gt_file)
            s6 = score_prompt(model, tokenizer, prompt6, yes_id, no_id, device)
            print(f"    Wrong issue text: {s6:.3f} (delta={s6-orig_score:+.3f})")

        print()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("ADVANCED DEBUGGING: RankFT Cross-Encoder")
    print("=" * 80)

    # Load data
    test_data, candidates = load_data()

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
    print("Model loaded.\n")

    # Pick examples
    examples = pick_examples(test_data, candidates, n=3)
    print(f"Selected {len(examples)} examples for analysis.\n")

    # Run all analyses
    # 3. LoRA weight analysis (no forward pass needed, fast)
    analyze_lora_weights(model)

    # 1. Gradient attribution
    analyze_gradient_attribution(model, tokenizer, yes_id, no_id, DEVICE, examples)

    # 2. Attention analysis
    analyze_attention(model, tokenizer, DEVICE, examples)

    # 4. Embedding space
    analyze_embeddings(model, tokenizer, DEVICE, examples)

    # 5. Sensitivity
    analyze_sensitivity(model, tokenizer, yes_id, no_id, DEVICE, examples)

    print(f"\n{'='*80}")
    print("ADVANCED DEBUGGING COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
