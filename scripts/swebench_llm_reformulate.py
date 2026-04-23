#!/usr/bin/env python3
"""
LLM-based query reformulation for SWE-bench BM25 retrieval.

Key insight from LocAgent: asking an LLM to extract relevant file names,
class names, and function names from issue text dramatically improves
BM25 retrieval (+36% in their paper).

This script uses Qwen2.5-7B-Instruct to reformulate issue queries.
"""
import os
import json
import re
import argparse
import time
from collections import defaultdict
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Prompts for query reformulation
# ============================================================

REFORMULATE_PROMPT = """You are an expert Python developer analyzing a GitHub issue report.

Given the issue text below, extract the following information:

1. **File paths**: Any file paths mentioned or strongly implied (e.g., django/db/models.py)
2. **Module names**: Python module references (e.g., django.db.models)
3. **Class names**: Relevant class names mentioned
4. **Function names**: Relevant function/method names mentioned
5. **Key terms**: Technical terms that would help find the buggy file

Output ONLY a comma-separated list of search keywords. Include file paths, module paths, class names, function names, and key technical terms. No explanations.

Issue:
{issue_text}

Keywords:"""


REFORMULATE_PROMPT_V2 = """Given this GitHub issue, which Python source files likely need to be modified to fix it? List the most likely file paths, one per line. If unsure about exact paths, list module.class patterns.

Issue:
{issue_text}

Most likely files to modify:"""


def load_model(model_path, gpu_id=0):
    """Load model for inference."""
    device = f"cuda:{gpu_id}"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def generate_keywords(model, tokenizer, issue_text, device, max_new_tokens=200):
    """Generate search keywords from issue text."""
    prompt = REFORMULATE_PROMPT.format(issue_text=issue_text[:2000])  # Truncate long issues

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=3072).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
    )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def parse_keywords(response):
    """Parse LLM response into keyword list."""
    # Split by commas, newlines, semicolons
    parts = re.split(r'[,;\n]', response)
    keywords = []
    for part in parts:
        part = part.strip().strip('- ').strip()
        if part and len(part) > 1:
            keywords.append(part)
    return keywords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/data/shuyang/models/Qwen2.5-7B-Instruct')
    parser.add_argument('--test_data', default='data/swebench_lite/swebench_lite_test.jsonl')
    parser.add_argument('--output', default='data/rankft/swebench_llm_keywords.jsonl')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer, device = load_model(args.model_path, args.gpu_id)
    print(f"  Model loaded on {device}")

    examples = []
    with open(args.test_data) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"  {len(examples)} examples")

    results = []
    t0 = time.time()
    for i, ex in enumerate(examples):
        issue_text = ex.get('issue_text', ex.get('problem_statement', ''))
        key = ex.get('issue_id', ex.get('instance_id', ''))

        response = generate_keywords(model, tokenizer, issue_text, device)
        keywords = parse_keywords(response)

        results.append({
            'issue_id': key,
            'repo': ex.get('repo', ''),
            'keywords': keywords,
            'raw_response': response,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(examples) - i - 1)
            print(f"  [{i+1}/{len(examples)}] ETA: {eta:.0f}s | Keywords: {keywords[:5]}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"\nSaved {len(results)} reformulated queries to {args.output}")

    # Quick stats
    avg_kw = sum(len(r['keywords']) for r in results) / len(results)
    print(f"  Avg keywords per issue: {avg_kw:.1f}")


if __name__ == '__main__':
    main()
