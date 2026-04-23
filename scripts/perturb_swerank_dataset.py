#!/usr/bin/env python3
"""Apply SHA-256 path hashing to SweRank's BEIR datasets.

For each swe-bench-lite-function_<instance> directory:
  - corpus.jsonl: _id is "<filepath>/<funcname>"; hash the filepath part.
    Also rewrite the first line of `text` (which mirrors _id).
  - qrels/test.tsv: rewrite corpus-id with the same hashed prefix.
  - queries.jsonl: untouched (bug-report text).

Output mirrors the directory name with suffix `_pathhash`.
Hash is deterministic per path component (matches eval_ranking_metrics.py hash_path).
"""
import argparse, json, hashlib, os, glob, re


def hash_filepath(filepath: str) -> str:
    """Hash each path component; preserve .py suffix on the leaf."""
    parts = []
    for x in filepath.split("/"):
        if not x: continue
        h = hashlib.sha256(x.encode()).hexdigest()[:16]
        if x.endswith(".py"):
            parts.append(f"m_{h}.py")
        else:
            parts.append(f"d_{h}")
    return "/".join(parts)


def split_id(item_id: str):
    """Split '<filepath>/<funcname>' into (filepath, funcname).
    Filepath is everything up to and including the last `.py/` boundary."""
    # Find the .py/ boundary
    m = re.search(r"\.py/", item_id)
    if m is None:
        # No function part; entire id is filepath
        return item_id, ""
    end = m.end()  # index just after '.py/'
    filepath = item_id[:end - 1]  # excl trailing slash
    funcname = item_id[end:]
    return filepath, funcname


def perturb_id(item_id: str) -> str:
    fp, fn = split_id(item_id)
    new_fp = hash_filepath(fp)
    if fn:
        return f"{new_fp}/{fn}"
    return new_fp


def perturb_corpus_line(obj: dict) -> dict:
    old_id = obj["_id"]
    new_id = perturb_id(old_id)
    obj["_id"] = new_id
    # Rewrite first line of text if it equals old_id
    text = obj.get("text", "")
    lines = text.split("\n", 1)
    if lines and lines[0] == old_id:
        obj["text"] = new_id + ("\n" + lines[1] if len(lines) > 1 else "")
    return obj


def perturb_qrels(infile: str, outfile: str) -> int:
    n = 0
    with open(infile) as f, open(outfile, "w") as g:
        header = f.readline()
        g.write(header)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3: continue
            q, c, s = parts[0], parts[1], parts[2]
            new_c = perturb_id(c)
            g.write(f"{q}\t{new_c}\t{s}\n")
            n += 1
    return n


def perturb_corpus(infile: str, outfile: str) -> int:
    n = 0
    seen = set()
    with open(infile) as f, open(outfile, "w") as g:
        for line in f:
            obj = json.loads(line)
            obj = perturb_corpus_line(obj)
            if obj["_id"] in seen:
                # Hash collision: warn (rare with 16-char prefix but possible)
                print(f"WARN: collision on {obj['_id']}", flush=True)
                continue
            seen.add(obj["_id"])
            g.write(json.dumps(obj) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True, help="dir containing swe-bench-lite-function_* subdirs")
    ap.add_argument("--prefix", default="swe-bench-lite-function_")
    ap.add_argument("--suffix", default="_pathhash")
    ap.add_argument("--out_root", required=True, help="separate output root dir (not same parent as src_dir)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    src_dirs = sorted(glob.glob(os.path.join(args.src_dir, args.prefix + "*")))
    src_dirs = [d for d in src_dirs if os.path.isdir(d) and not d.endswith(args.suffix)]
    if args.limit: src_dirs = src_dirs[:args.limit]
    print(f"Found {len(src_dirs)} instance dirs", flush=True)

    n_corpus, n_qrels = 0, 0
    for src in src_dirs:
        # Output dir: same parent, name with suffix appended
        dst = os.path.join(args.out_root, os.path.basename(src))
        os.makedirs(os.path.join(dst, "qrels"), exist_ok=True)

        # Copy queries.jsonl unchanged
        with open(os.path.join(src, "queries.jsonl")) as f, open(os.path.join(dst, "queries.jsonl"), "w") as g:
            g.write(f.read())

        # Perturb corpus
        n = perturb_corpus(os.path.join(src, "corpus.jsonl"),
                            os.path.join(dst, "corpus.jsonl"))
        n_corpus += n

        # Perturb qrels
        n = perturb_qrels(os.path.join(src, "qrels", "test.tsv"),
                           os.path.join(dst, "qrels", "test.tsv"))
        n_qrels += n

    print(f"Done. {len(src_dirs)} instances, {n_corpus} corpus rows, {n_qrels} qrels rows.")


if __name__ == "__main__":
    main()
