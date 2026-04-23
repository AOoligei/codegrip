#!/usr/bin/env python3
"""Extract attention weights over path tokens vs code tokens."""
import argparse
import json
import os
import re
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def is_path_token(token_str, path_pattern=re.compile(r'[/\\._]|src|lib|utils|test|config|main|init|setup|py|js|ts')):
    """Heuristic: is this token part of a file path?"""
    return bool(path_pattern.search(token_str.lower()))

def analyze_attention(model_path, lora_path, test_data_path, candidates_path, 
                      output_path, gpu_id=0, n_examples=50, max_seq_length=512):
    device = f"cuda:{gpu_id}"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map=device, trust_remote_code=True
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    
    # Load data
    test_data = [json.loads(l) for l in open(test_data_path)][:n_examples]
    cand_data = {}
    for line in open(candidates_path):
        d = json.loads(line)
        key = (d['repo'], str(d['issue_id']))
        cand_data[key] = d
    
    results = []
    
    for idx, item in enumerate(test_data):
        key = (item['repo'], str(item['issue_id']))
        if key not in cand_data:
            continue
        
        candidates = cand_data[key].get('candidates', cand_data[key].get('bm25_candidates', []))
        gt_files = set(item.get('changed_py_files', []))
        
        # Take first GT file and first non-GT file
        pos_file = list(gt_files & set(candidates))
        if not pos_file:
            continue
        pos_file = pos_file[0]
        
        # Construct input (same as eval script)
        issue_text = item['issue_text'][:500]
        prompt = f"Bug report: {issue_text}\n\nCandidate file: {pos_file}\n\nIs this file relevant? Answer Yes or No."
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=max_seq_length, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        # Analyze last layer attention
        # Shape: (1, num_heads, seq_len, seq_len)
        last_attn = outputs.attentions[-1][0]  # (num_heads, seq_len, seq_len)
        avg_attn = last_attn.mean(dim=0)  # (seq_len, seq_len)
        
        # Get token strings
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Classify each token
        path_mask = torch.tensor([is_path_token(t) for t in tokens], device=device)
        
        # Attention from the LAST token (where prediction happens) to all other tokens
        last_token_attn = avg_attn[-1]  # (seq_len,)
        
        path_attn = last_token_attn[path_mask].sum().item()
        non_path_attn = last_token_attn[~path_mask].sum().item()
        n_path = path_mask.sum().item()
        n_non_path = (~path_mask).sum().item()
        
        results.append({
            'idx': idx,
            'issue_id': str(item['issue_id']),
            'n_tokens': len(tokens),
            'n_path_tokens': n_path,
            'n_non_path_tokens': n_non_path,
            'path_attn_total': path_attn,
            'non_path_attn_total': non_path_attn,
            'path_attn_per_token': path_attn / max(n_path, 1),
            'non_path_attn_per_token': non_path_attn / max(n_non_path, 1),
        })
        
        if (idx + 1) % 10 == 0:
            avg_ratio = np.mean([r['path_attn_per_token'] / max(r['non_path_attn_per_token'], 1e-8) for r in results])
            print(f"  [{idx+1}/{len(test_data)}] Avg path/non-path attn ratio: {avg_ratio:.2f}x")
    
    # Summary
    path_attns = [r['path_attn_per_token'] for r in results]
    non_path_attns = [r['non_path_attn_per_token'] for r in results]
    ratios = [p / max(np, 1e-8) for p, np in zip(path_attns, non_path_attns)]
    
    summary = {
        'n_examples': len(results),
        'mean_path_attn_per_token': float(np.mean(path_attns)),
        'mean_non_path_attn_per_token': float(np.mean(non_path_attns)),
        'mean_ratio': float(np.mean(ratios)),
        'median_ratio': float(np.median(ratios)),
        'results': results
    }
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Attention Analysis ===")
    print(f"Examples: {len(results)}")
    print(f"Path attn per token: {np.mean(path_attns):.4f}")
    print(f"Non-path attn per token: {np.mean(non_path_attns):.4f}")
    print(f"Ratio (path/non-path): {np.mean(ratios):.2f}x")
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--lora_path', default=None)
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--n_examples', type=int, default=50)
    args = parser.parse_args()
    analyze_attention(args.model_path, args.lora_path, args.test_data, args.candidates, 
                      args.output, args.gpu_id, args.n_examples)
