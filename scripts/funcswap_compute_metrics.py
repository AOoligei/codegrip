#!/usr/bin/env python3
"""Compute file / module / function Acc@k for our reranker output on a SweRank
function-level dataset. Mirrors SweRank's `cal_metrics_w_dataset` semantics
(refactored_eval_localization.py:333). Self-contained so we don't depend on
the SweRank package being importable.

Usage:
  python funcswap_compute_metrics.py \
      --rerank_root /data/.../func_codeaware_eval_funcswap/codeaware-func \
      --dataset_dir /data/.../SweRank_perturbed_funcswap \
      --prefix swe-bench-lite-function_ \
      --out /data/.../func_codeaware_eval_funcswap/metrics.json

Acc@k semantics (matches SweRank):
  - For each instance, gt_labels is a fixed-length [max_k] vector with the
    first len(gt) positions set to 1 (the rest 0).
  - pred_labels is a [max_k] vector with 1 at every position where the
    predicted doc-id is in the gt set.
  - Acc@k = mean over instances of [sum(pred_labels[:k]) == sum(gt_labels[:k])]
    i.e. an instance counts as a hit only if (number of correct preds in top-k)
    equals (number of GT capped at k). For instances with a single GT this
    reduces to "is the GT in top-k".
"""
import argparse
import collections
import json
import os
import re
import sys


def load_qrels_for_instance(inst_dir, instance_id):
    """Returns dict instance_id -> {doc_id: 1} matching BEIR's loader output."""
    qrel_path = os.path.join(inst_dir, "qrels", "test.tsv")
    out = {}
    if not os.path.isfile(qrel_path):
        return {instance_id: {}}
    with open(qrel_path) as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, cid, score = parts[0], parts[1], int(parts[2])
            if score <= 0:
                continue
            out.setdefault(qid, {})[cid] = score
    if instance_id not in out:
        out[instance_id] = {}
    return out


def gt_at_level(qrel_dict, instance_id, level):
    """Mirrors SweRank refactored_eval_localization.py:380-387."""
    out = []
    for func in set(qrel_dict.get(instance_id, {}).keys()):
        if level == "file":
            fn = func.split(".py")[0] + ".py"
            if fn not in out:
                out.append(fn)
        elif level == "module":
            fn = func.split(".py/")[0] + ".py"
            mname = func.split(".py/")[-1].split("/")[0]
            mid = f"{fn}:{mname}"
            if mid not in out:
                out.append(mid)
        elif level == "function":
            fle, func_n = func.split(".py/")
            if func_n.endswith(".__init__"):
                func_n = func_n[: (len(func_n) - len(".__init__"))]
            fn = f"{fle}.py:{func_n.strip('/').replace('/', '.')}"
            if fn not in out:
                out.append(fn)
    return out


def pred_at_level(pred_funcs, level):
    """Mirrors SweRank refactored_eval_localization.py:393-416."""
    if level == "file":
        out = []
        seen = set()
        for vv in pred_funcs:
            m = re.match(r"(.+\.py)(/.*)?", vv)
            if m:
                if m.group(1) not in seen:
                    out.append(m.group(1))
                    seen.add(m.group(1))
        return out
    elif level == "module":
        out = []
        for pl in pred_funcs:
            if ".py/" not in pl:
                continue
            fle, func_n = pl.split(".py/", 1)
            module_name = f"{fle}.py:{func_n.strip('/').split('/')[0]}"
            if module_name not in out:
                out.append(module_name)
        return out
    elif level == "function":
        out = []
        for pl in pred_funcs:
            if ".py/" not in pl:
                continue
            fle, func_n = pl.split(".py/", 1)
            if func_n.endswith(".__init__"):
                func_n = func_n[: (len(func_n) - len(".__init__"))]
            out.append(f"{fle}.py:{func_n.strip('/').replace('/', '.')}")
        return out


def acc_at_k(_pred_labels, _gt_labels, k):
    """SweRank acc_at_k: hit if sum(pred[:k]) == sum(gt[:k])."""
    n = 0
    hits = 0
    for pl, gl in zip(_pred_labels, _gt_labels):
        relevant = sum(pl[:k])
        total = sum(gl[:k])
        if relevant == total:
            hits += 1
        n += 1
    return hits / n if n else 0.0


def recall_at_k(_pred_labels, _gt_labels, k):
    """recall = sum(pred[:k]) / sum(gt) per instance, averaged."""
    vals = []
    for pl, gl in zip(_pred_labels, _gt_labels):
        relevant = sum(pl[:k])
        total = sum(gl)
        vals.append(relevant / total if total else 0.0)
    return sum(vals) / len(vals) if vals else 0.0


def evaluate(rerank_root, dataset_dir, prefix, k_lists):
    """Returns dict {level: {f'Acc@{k}': value, f'Recall@{k}': value}}."""
    levels = list(k_lists.keys())
    max_k_per_level = {lvl: max(k_lists[lvl]) for lvl in levels}

    inst_dirs = sorted(
        d for d in os.listdir(dataset_dir)
        if d.startswith(prefix) and os.path.isdir(os.path.join(dataset_dir, d))
    )
    print(f"Found {len(inst_dirs)} instance dirs under {dataset_dir}", flush=True)

    # Collect per-level (gt_labels, pred_labels) lists
    per_level = {lvl: {"gt": [], "pred": [], "n_inst": 0} for lvl in levels}
    n_skip_no_rerank = 0
    n_skip_no_gt = 0
    n_total = 0

    for inst in inst_dirs:
        instance_id = inst[len(prefix):]
        rerank_path = os.path.join(rerank_root, inst, "rerank_100_llm_gen_num.json")
        if not os.path.isfile(rerank_path):
            n_skip_no_rerank += 1
            continue
        rerank = json.load(open(rerank_path))
        if instance_id not in rerank:
            n_skip_no_rerank += 1
            continue
        # rerank[instance_id] is dict doc_id -> score; rank desc.
        scored = rerank[instance_id]
        ranked_docs = [d for d, _ in sorted(scored.items(), key=lambda x: -x[1])]

        qrel = load_qrels_for_instance(os.path.join(dataset_dir, inst), instance_id)
        if not qrel.get(instance_id):
            n_skip_no_gt += 1
            continue
        n_total += 1

        for lvl in levels:
            max_k = max_k_per_level[lvl]
            gt_list = gt_at_level(qrel, instance_id, lvl)
            pred_list = pred_at_level(ranked_docs, lvl)[:max_k]

            gt_labels = [0] * max_k
            pred_labels = [0] * max_k
            for i in range(min(len(gt_list), max_k)):
                gt_labels[i] = 1
            for i, p in enumerate(pred_list):
                if p in gt_list:
                    pred_labels[i] = 1
            per_level[lvl]["gt"].append(gt_labels)
            per_level[lvl]["pred"].append(pred_labels)
            per_level[lvl]["n_inst"] += 1

    # Compute metrics
    results = {}
    for lvl in levels:
        results[lvl] = {"n": per_level[lvl]["n_inst"]}
        for k in k_lists[lvl]:
            results[lvl][f"Acc@{k}"] = round(
                acc_at_k(per_level[lvl]["pred"], per_level[lvl]["gt"], k) * 100, 2
            )
            results[lvl][f"Recall@{k}"] = round(
                recall_at_k(per_level[lvl]["pred"], per_level[lvl]["gt"], k) * 100, 2
            )
    results["_meta"] = {
        "rerank_root": rerank_root,
        "dataset_dir": dataset_dir,
        "n_instances_found": len(inst_dirs),
        "n_evaluated": n_total,
        "skip_no_rerank": n_skip_no_rerank,
        "skip_no_gt": n_skip_no_gt,
    }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank_root", required=True,
                    help="Dir containing per-instance subdirs with rerank_100_llm_gen_num.json")
    ap.add_argument("--dataset_dir", required=True,
                    help="Dir containing per-instance subdirs with qrels/test.tsv")
    ap.add_argument("--prefix", default="swe-bench-lite-function_")
    ap.add_argument("--out", required=True, help="Output JSON path for metrics")
    args = ap.parse_args()

    k_lists = {
        "file": [1, 3, 5, 10],
        "module": [5, 10],
        "function": [1, 5, 10],
    }
    results = evaluate(args.rerank_root, args.dataset_dir, args.prefix, k_lists)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
