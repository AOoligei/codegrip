#!/usr/bin/env python3
"""
Ensemble reranker + e5 under PathSwap.

Path reranker collapses on hashed paths (~1%);
e5 on file content (which is unchanged on disk) should retain code signal.
RRF/zsum should let e5 carry when reranker fails.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
import random; random.seed(42); np.random.seed(42); torch.manual_seed(42)

MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"
LORA_PATH = "/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/best"
E5_BASE = "/data/chenlibin/models/models--intfloat--e5-large-v2/snapshots"
TEST_PATH = "/home/chenlibin/grepo_agent/data/pathswap/grepo_test_pathswap.jsonl"
BM25_PATH = "/home/chenlibin/grepo_agent/data/pathswap/merged_bm25_exp6_candidates_pathswap.jsonl"
RENAME_MAP_PATH = "/home/chenlibin/grepo_agent/data/pathswap/rename_maps.json"
REPO_DIR = "/home/chenlibin/grepo_agent/data/repos"

PROMPT = ("Given the bug report, is this file likely to need modification?\n\n"
          "Bug Report: {issue_text}\n\nFile: {file}\n\nAnswer:")


def find_e5_path():
    for d in os.listdir(E5_BASE):
        p = os.path.join(E5_BASE, d)
        if os.path.isfile(os.path.join(p, "config.json")):
            return p
    raise RuntimeError("e5 not found")


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def e5_encode(m, tok, texts, dev, prefix="passage: ", bs=8):
    embs = []
    for i in range(0, len(texts), bs):
        batch = [prefix + t for t in texts[i:i+bs]]
        inp = tok(batch, max_length=512, padding=True, truncation=True, return_tensors='pt').to(dev)
        with torch.no_grad():
            out = m(**inp)
        e = average_pool(out.last_hidden_state, inp['attention_mask'])
        e = F.normalize(e, p=2, dim=1)
        embs.append(e.cpu().float().numpy())
    return np.concatenate(embs, axis=0)


def reranker_score_batch(m, tok, prompts, yid, nid, dev, bs=8):
    out = []
    for i in range(0, len(prompts), bs):
        b = prompts[i:i+bs]
        inp = tok(b, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024, padding_side="left").to(dev)
        with torch.no_grad():
            logits = m(**inp).logits[:, -1, :]
        s = (logits[:, yid].float() - logits[:, nid].float()).cpu().numpy()
        out.extend(s.tolist())
    return out


def rrf(scores_list, k=60):
    N = len(scores_list[0])
    ranks = []
    for s in scores_list:
        order = np.argsort(-np.array(s))
        rank = np.empty(N)
        for r, idx in enumerate(order):
            rank[idx] = r + 1
        ranks.append(rank)
    return [sum(1/(k + r[i]) for r in ranks) for i in range(N)]


def zscore(x):
    x = np.array(x); s = x.std()
    return (x - x.mean()) / (s + 1e-9)


def read_head(repo, fpath, n=50):
    full = os.path.join(REPO_DIR, repo, fpath)
    if not os.path.isfile(full):
        return f"file: {fpath}"
    try:
        with open(full, "r", errors="replace") as f:
            return f"file: {fpath}\n" + "".join(f.readlines()[:n])[:1000]
    except Exception:
        return f"file: {fpath}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--max_examples", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dev = f"cuda:{args.gpu_id}"

    test = [json.loads(l) for l in open(TEST_PATH)]
    if args.max_examples: test = test[:args.max_examples]
    bm25 = {}
    with open(BM25_PATH) as f:
        for l in f:
            r = json.loads(l); bm25[(r["repo"], r["issue_id"])] = r

    # Reverse rename map: hashed -> original (for reading code from disk)
    rename_maps = json.load(open(RENAME_MAP_PATH))
    reverse_maps = {}
    for repo, m in rename_maps.items():
        reverse_maps[repo] = {v: k for k, v in m.items()}

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    rr_tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if rr_tok.pad_token is None: rr_tok.pad_token = rr_tok.eos_token
    rr_tok.padding_side = "left"
    print("Loading reranker...", flush=True)
    rr_m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
            device_map={"": dev}, trust_remote_code=True, torch_dtype=torch.bfloat16)
    rr_m = PeftModel.from_pretrained(rr_m, LORA_PATH); rr_m.eval()
    yid = rr_tok.encode("Yes", add_special_tokens=False)[0]
    nid = rr_tok.encode("No", add_special_tokens=False)[0]

    print("Loading e5...", flush=True)
    e5_p = find_e5_path()
    e5_tok = AutoTokenizer.from_pretrained(e5_p)
    e5_m = AutoModel.from_pretrained(e5_p, torch_dtype=torch.float16).to(dev).eval()

    records = []; start = time.time()
    for i, rec in enumerate(test):
        repo = rec.get("repo", "")
        issue = rec["issue_text"][:1500]
        gt = set(rec.get("changed_py_files", rec.get("changed_files", [])))
        if not gt: continue
        key = (repo, rec.get("issue_id"))
        if key not in bm25: continue
        cands = bm25[key].get("candidates", bm25[key].get("bm25_candidates", []))[:args.K]
        if not cands: continue
        for g in gt:
            if g not in cands: cands.append(g)

        # Reranker scores (hashed paths, since test data is PathSwap)
        prompts = [PROMPT.format(issue_text=issue, file=c) for c in cands]
        s_path = reranker_score_batch(rr_m, rr_tok, prompts, yid, nid, dev)

        # E5 scores: read code from ORIGINAL files (use reverse map)
        rev = reverse_maps.get(repo, {})
        cand_texts = []
        for c in cands:
            orig = rev.get(c, c)  # if not in map, try as-is
            cand_texts.append(read_head(repo, orig, 50))
        cand_emb = e5_encode(e5_m, e5_tok, cand_texts, dev, prefix="passage: ")
        issue_emb = e5_encode(e5_m, e5_tok, [issue], dev, prefix="query: ")
        s_code = (cand_emb @ issue_emb[0]).tolist()

        records.append({"repo": repo, "gt": list(gt), "cands": cands,
                        "s_path": s_path, "s_code": s_code})
        if (i+1) % 50 == 0:
            print(f"  [{i+1}] ({time.time()-start:.0f}s)", flush=True)

    print(f"\n=== PathSwap Ensemble (n={len(records)}) ===")
    strategies = {
        "path_only":  lambda r: r["s_path"],
        "code_only":  lambda r: r["s_code"],
        "rrf":        lambda r: rrf([r["s_path"], r["s_code"]]),
        "zsum_eq":    lambda r: list(zscore(r["s_path"]) + zscore(r["s_code"])),
        "zsum_p03":   lambda r: list(0.3*zscore(r["s_path"]) + 0.7*zscore(r["s_code"])),
        "zsum_p05":   lambda r: list(0.5*zscore(r["s_path"]) + 0.5*zscore(r["s_code"])),
        "zsum_p07":   lambda r: list(0.7*zscore(r["s_path"]) + 0.3*zscore(r["s_code"])),
    }
    results = {}
    for name, fn in strategies.items():
        hits = []
        for r in records:
            scored = fn(r)
            top = r["cands"][int(np.argmax(scored))]
            hits.append(1.0 if top in set(r["gt"]) else 0.0)
        acc = float(np.mean(hits)*100)
        results[name] = acc
        print(f"  {name:15s}: R@1={acc:.2f}%")

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"n": len(records), "strategies": results}, f, indent=2)


if __name__ == "__main__":
    main()
