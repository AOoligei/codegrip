#!/usr/bin/env python3
"""Function-level pointwise scorer for codeaware LoRA on SweRank function corpus.

Reads SweRank retriever output (JSONL, one row per SWE-bench instance with
a 'docs' field = top-K function IDs from SweRankEmbed-Large). For each row:
  - Use `problem_statement` as issue text
  - Find corpus for `swe-bench-lite-function_<instance_id>` under dataset_dir
  - Score each doc (func_id + body) pointwise via (yes - no) logit diff
  - Write SweRank-compatible rerank_<top_k>_llm_gen_num.json per instance
"""
import os, sys, json, argparse, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

torch.manual_seed(42); np.random.seed(42)

PROMPT_FUNC = ("Given the bug report, is this function likely to need modification?\n\n"
               "Bug Report: {issue}\n\n"
               "Function: {func_id}\n\n"
               "Code:\n{body}\n\n"
               "Answer:")


def build_ids_preserve_suffix(prompt, tok, max_len, suffix="\n\nAnswer:"):
    suf_ids = tok.encode(suffix, add_special_tokens=False)
    if prompt.endswith(suffix):
        body_ids = tok.encode(prompt[:-len(suffix)], add_special_tokens=False)
    else:
        body_ids = tok.encode(prompt, add_special_tokens=False)
    keep = max_len - len(suf_ids)
    if keep <= 0:
        return suf_ids[-max_len:]
    return body_ids[:keep] + suf_ids


@torch.no_grad()
def score_batch(model, tok, prompts, yes_id, no_id, max_len, dev, bs=8):
    batch_ids = [build_ids_preserve_suffix(p, tok, max_len) for p in prompts]
    pad_id = tok.pad_token_id
    scores = []
    for i in range(0, len(batch_ids), bs):
        chunk = batch_ids[i:i+bs]
        max_in = max(len(x) for x in chunk)
        padded = [[pad_id] * (max_in - len(x)) + x for x in chunk]
        attn = [[0] * (max_in - len(x)) + [1] * len(x) for x in chunk]
        input_ids = torch.tensor(padded, device=dev)
        attention_mask = torch.tensor(attn, device=dev)
        out = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        s = (out[:, yes_id].float() - out[:, no_id].float()).cpu().numpy()
        scores.extend(s.tolist())
    return scores


def load_corpus(corpus_path):
    """Return dict mapping func_id to body text (strips leading _id first line)."""
    out = {}
    for line in open(corpus_path):
        d = json.loads(line)
        fid = d["_id"]
        text = d.get("text", "")
        if text.startswith(fid + "\n"):
            body = text[len(fid) + 1:]
        else:
            body = text
        out[fid] = body
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lora_path", default=None)
    ap.add_argument("--dataset_dir", required=True,
                    help="Directory of SweRank per-instance datasets")
    ap.add_argument("--dataset", default="swe-bench-lite",
                    help="Prefix for per-instance dirs (dataset-function_<iid>)")
    ap.add_argument("--retriever_results_jsonl", required=True,
                    help="SweRank retriever JSONL (each row has 'docs' field)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_type", default="codeaware-func")
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--max_seq_length", type=int, default=1536)
    ap.add_argument("--max_body_chars", type=int, default=3000)
    ap.add_argument("--max_issue_chars", type=int, default=1500)
    ap.add_argument("--score_batch_size", type=int, default=4)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--use_4bit", action="store_true")
    args = ap.parse_args()

    out_root = os.path.join(args.output_dir, args.data_type)
    os.makedirs(out_root, exist_ok=True)
    dev = f"cuda:{args.gpu_id}"

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if args.use_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16)
        m = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                  quantization_config=bnb,
                                                  device_map={"": dev},
                                                  trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16)
    else:
        m = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                  torch_dtype=torch.bfloat16,
                                                  device_map={"": dev},
                                                  trust_remote_code=True)
    if args.lora_path:
        m = PeftModel.from_pretrained(m, args.lora_path)
    m.eval()
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    quant_str = "4bit" if args.use_4bit else "bf16"
    print(f"Loaded {quant_str} model. yes_id={yes_id}, no_id={no_id}", flush=True)

    # Load retriever JSONL — each row is one SWE-bench instance w/ 'docs' field
    rows = [json.loads(l) for l in open(args.retriever_results_jsonl)]
    print(f"Loaded {len(rows)} retriever rows from {args.retriever_results_jsonl}", flush=True)

    n_done = 0; n_skip = 0; start = time.time()
    for row in rows:
        iid = row.get("instance_id")
        if not iid: n_skip += 1; continue
        docs = row.get("docs") or []
        if not docs: n_skip += 1; continue
        issue_text = row.get("problem_statement", "")

        inst_dir_name = f"{args.dataset}-function_{iid}"
        inst_path = os.path.join(args.dataset_dir, inst_dir_name)
        corpus_path = os.path.join(inst_path, "corpus.jsonl")
        if not os.path.isfile(corpus_path):
            n_skip += 1; continue

        corpus = load_corpus(corpus_path)

        per_inst_out = os.path.join(out_root, inst_dir_name)
        os.makedirs(per_inst_out, exist_ok=True)

        cand_ids = docs[:args.top_k]
        issue = issue_text[:args.max_issue_chars]
        prompts = []
        valid_ids = []
        for fid in cand_ids:
            body = corpus.get(fid, "")[:args.max_body_chars]
            if not body: continue
            prompts.append(PROMPT_FUNC.format(issue=issue, func_id=fid, body=body))
            valid_ids.append(fid)
        if not prompts:
            n_skip += 1; continue

        scores = score_batch(m, tok, prompts, yes_id, no_id,
                             args.max_seq_length, dev, args.score_batch_size)
        ranked = sorted(zip(valid_ids, scores), key=lambda x: -x[1])

        result_per_q = {iid: {fid: float(score) for fid, score in ranked}}
        out_file = os.path.join(per_inst_out, f"rerank_{args.top_k}_llm_gen_num.json")
        with open(out_file, "w") as f:
            json.dump(result_per_q, f)

        n_done += 1
        if n_done % 20 == 0:
            print(f"  [{n_done}/{len(rows)}] {time.time()-start:.0f}s", flush=True)

    print(f"\n=== DONE n={n_done} skip={n_skip} ===", flush=True)
    print(f"Output: {out_root}", flush=True)


if __name__ == "__main__":
    main()
