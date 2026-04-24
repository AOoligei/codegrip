# CodeGRIP

File-level bug localization via path-augmented reranking on top of BM25.

**Paper**: *The Oracle Fallacy in Code Bug Localization: When Better Candidate Recall Produces Worse Results* (under review, NeurIPS 2026). LaTeX sources in [`paper/`](paper/).

## Contents

```
src/train/           LoRA training (file-level + function-level)
scripts/             evaluation + data generation
  eval_codeaware_4bit.py         file-level pointwise eval
  score_codeaware_function.py    function-level pointwise eval
  build_swebench_pathswap.py     SHA-256 PathSwap generator
  build_swebench_verified_*.py   Verified split BM25 + PathSwap
  cleanlite_build_subset.py      module-token leakage cleaner
configs/             QLoRA configs (Qwen2.5 7B / 14B)
paper/               main.tex
```

## Setup

```bash
conda create -n codegrip python=3.10 -y && conda activate codegrip
pip install torch==2.6.0 transformers==4.51.3 peft==0.15.1 bitsandbytes==0.45.4 \
    rank_bm25 datasets tqdm numpy
```

Base model: `Qwen/Qwen2.5-7B-Instruct` (or 14B).

## Recipes

**Train file-level codeaware reranker (best: 14B, aug=0.5):**
```bash
python src/train/train_rankft.py \
  --model_path Qwen/Qwen2.5-14B-Instruct \
  --train_data data/rankft/clean_train_combined_v2.jsonl \
  --repo_dir data/unified_repos \
  --output_dir experiments/scale14b_aug05 \
  --include_code --code_max_lines 50 \
  --path_augment_fraction 0.5 \
  --num_negatives 8 --learning_rate 5e-5 --num_epochs 2 \
  --batch_size 1 --gradient_accumulation_steps 16 \
  --max_seq_length 768 --lora_rank 32 --seed 42
```

**Eval on SWE-bench Lite (normal + SHA-256 PathSwap):**
```bash
bash scripts/eval_2lora_pathswap_pair.sh <LORA_PATH> <TAG> <GPU>
```

**Train function-level reranker:**
```bash
python src/train/train_rankft_function.py \
  --train_pairs data/func_gt/train_pairs.jsonl \
  --corpus_dir data/func_corpus \
  --output_dir experiments/func_codeaware \
  --path_augment_fraction 0.5 --seed 42
```

## Data

All sources are public. Derived artifacts (BM25 pools, PathSwap variants, per-commit Verified index, function-level corpus) regenerate via our scripts in ~1-2 h.

### 1. Base datasets

**SWE-bench Lite / Verified / train** (public, via HuggingFace):
```python
from datasets import load_dataset
load_dataset("princeton-nlp/SWE-bench_Lite",      split="test")   # 300
load_dataset("princeton-nlp/SWE-bench_Verified",  split="test")   # 500
load_dataset("princeton-nlp/SWE-bench",           split="train")  # for training pairs
```

**GREPO** (public, PKU MuLab): https://github.com/qingpingmo/GREPO  
Dataset on ModelScope: https://modelscope.cn/datasets/qingpingmomo/Grepo  
We use the train split (7883 issues) from `data/grepo_text/grepo_train.jsonl` in their release.

**Source repositories** (75 repos for GREPO + 12 for SWE-bench): `git clone` from GitHub per the repo list in `data/repos/.manifest.txt` (regenerated below).

### 2. Derived artifacts (regenerate from scripts)

```bash
# 1. BM25 top-500 candidates for SWE-bench train (used by Run 1/2 training)
python scripts/build_swebench_bm25.py --split train --top_k 500
# 2. BM25 top-500 for SWE-bench Lite / Verified test
python scripts/build_swebench_bm25.py --split test --benchmark lite
python scripts/build_swebench_verified_bm25_strict.py  # per-commit, Acc@100 = 85%
# 3. SHA-256 PathSwap test variants
python scripts/build_swebench_pathswap.py --benchmark lite
python scripts/build_swebench_verified_pathswap.py
# 4. Combined training pool (SWE + GREPO, 5916 pairs)
python scripts/build_clean_train_combined.py \
  --swe_train data/rankft/clean_swe_train.jsonl \
  --grepo_train data/grepo_text/grepo_train.jsonl \
  --out data/rankft/clean_train_combined_v2.jsonl
# 5. Function-level GT + corpus + train pairs (for function-level LoRA)
python scripts/build_func_gt_swe.py
python scripts/build_func_gt_grepo.py
python scripts/build_func_corpus_per_commit.py --num_workers 8
python scripts/codegrip_func_mine_hard_negatives.py \
  --gt_files data/func_gt/func_gt_swe.jsonl data/func_gt/func_gt_grepo.jsonl \
  --corpus_dir data/func_corpus --output data/func_gt/train_pairs.jsonl \
  --drop_log data/func_gt/drop.log
```

All scripts use `seed=42` and are deterministic.

## Headline numbers

On BM25 top-100 file pool (partial R@1):

| Model | Lite | Lite PathSwap | Δ_rel | Verified | Verified PathSwap | Δ_rel |
|---|---|---|---|---|---|---|
| 14B codeaware (aug=0.5) | **60.33** | **53.67** | **-11.0%** | **48.93** | **46.67** | **-4.6%** |
| &nbsp;&nbsp;+ body-cleanfull§ | 56.67 | 53.33 | **-5.9%** | TODO | TODO | TODO |
| 14B path-only (aug=0.0) | 56.33 | 39.33 | -30.2% | 48.62 | 34.18 | -29.7% |
| 7B Run 2 (combined+aug) | 56.67 | 47.33 | -16.5% | 48.13 | 45.02 | -6.5% |
| SweRankLLM-Small (7B, same pool) | 47.93 | — | — | — | — | — |

§ body-cleanfull: the 50-line code snippet has repo-package tokens (astropy/django/sympy/…, 16 total) replaced by consistent hashes, isolating true path effect from body content mismatching the hashed path.

## TODO (pending experiments)

- [ ] **File-level cleanfull on SWE-bench Verified** (body-leakage-controlled Δ% on n=500). Lite done: Δ_rel refines from $-$11.0% to $-$5.9% after cleaning. Script: [`scripts/eval_codeaware_4bit_cleanfull.py`](scripts/eval_codeaware_4bit_cleanfull.py).
- [ ] **Outline-prompt variant of codeaware**: replace first-50-lines with AST-rendered class/def signatures + decorators + first docstring line. Script ready: [`scripts/eval_codeaware_4bit_outline.py`](scripts/eval_codeaware_4bit_outline.py). Quick eval first; if ≥ 60.33 Lite R@1 then train new LoRA with outline prompt.
- [ ] **SweRank native-pipeline reproduction**: current retriever file-level Acc@100 is 37.23% vs ≥78.10% expected; root cause not localized within budget. Documented as paper limitation §codeaware:swerank_repro.
- [ ] **Significance for function-level cleanfull**: n=274 + current Δ=$-$14% is not statistically significant (McNemar p=0.44). Need Verified function pool to get n~500 — blocked on SweRank retriever reproduction above.

## License

MIT.
