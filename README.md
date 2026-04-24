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

## Progress log (2026-04-24)

Work since the collaborator suggested replacing first-N-lines with AST outline:

- Trained and evaluated two 14B QLoRA rerankers (Qwen2.5-14B) with same Run 2 recipe/data, changing only aug fraction: 14B aug=0.5 reaches Lite 60.33 / Verified 48.93, PathSwap Δ_rel $-$11.0% / $-$4.6%; same-base 14B aug=0 collapses $-$30.2% / $-$29.7% — so scale alone does not fix path shortcuts.
- Ran the full SweRank native pipeline (Embed-Large + LLM-Small) under our SHA-256 PathSwap: 20.8% → 21.17% (no collapse). Documented, with the caveat that our absolute reproduction of the paper's 78.10% is 57pp off and the root cause could not be localized within budget.
- Ran a body-cleanfull control for file-level eval (hash 16 repo-package tokens in the 50-line code snippet): Lite Δ_rel refines from $-$11.0% to $-$5.9% once incidental body/path token mismatches are removed.
- Ran a 5-step function-level cleaning progression on the SweRank function pool (n=274, our function-level Codeaware LoRA): normal 19.34 → 17.88 → 16.79 → 16.60 → 15.69; monotonic trend, none of the pairwise differences is statistically significant (McNemar $p>0.25$). Placed in the appendix as diagnostic with a strong caveat because it sits on top of the unreproduced SweRank pipeline.
- Paper rewriting with three chained codex reviews (5/10 → 6/10 → 7/10) plus one cold review (6/10), followed by ~15 independent cold-start codex proofread rounds fixing numerical, cross-reference, unit (pp vs %), and terminology issues throughout the draft. Added a main-result figure summarizing the 6-bar / 2-benchmark comparison.

Scripts added this week:

- [`scripts/eval_codeaware_4bit_cleanfull.py`](scripts/eval_codeaware_4bit_cleanfull.py): file-level eval with repo-package token hashing in the code snippet.
- [`scripts/eval_codeaware_4bit_outline.py`](scripts/eval_codeaware_4bit_outline.py): file-level eval with AST outline replacing first-N-lines (not yet run end-to-end; collaborator is running this track).
- [`scripts/cleanlite_build_subset.py`](scripts/cleanlite_build_subset.py): import / docstring / module-path cleaner for function-level corpus.

## TODO

Experiments we have not yet run but that would strengthen the paper:

- **Single-knob 2×2 ablation** (fixed data + fixed base model, vary only `aug` ∈ {0, 0.5} and `code-in-prompt` ∈ {no, yes}, 3 seeds each). Currently Run 1 vs Codeaware toggles both knobs simultaneously, so we present the comparison as a recipe rather than isolating either knob. Cost ~15-30 GPU-hours.
- **3-seed error bars for 14B Codeaware headline rows**. At n=300 / n=500 single-seed, the 0.31pp Verified gain of 14B Codeaware over 14B no-aug is within seed noise.
- **File-level body-cleanfull on SWE-bench Verified**. Lite refines Δ_rel from $-$11.0% to $-$5.9%; Verified is left as a TODO (script ready, just needs a GPU pass).
- **Function-level cleaning progression with larger n**. Currently n=274 is too small for significance; blocked on resolving the SweRank retriever reproduction issue above, or on re-running function-level eval on a wider pool.
- **SweRank native-pipeline reproduction**. We obtain 20.8% strict file Acc@1 vs the paper's 78.10% under the official scripts and paper-pinned environment, a 57pp gap we could not localize. Ruled out: sequence-length truncation, obvious cmdline divergence, prefix/tokenizer mismatch. Likely remaining candidates: model-revision / checkpoint-weights specifics, or a subtle eval-protocol difference we haven't identified.
- **Outline-based codeaware** (collaborator track). Replace the 50-line first-N snippet with a compact AST outline (class / def signatures, decorators, first-docstring-line). Quick: apply to the current 14B Codeaware LoRA at inference time and check whether Lite R@1 holds. If it does (and especially if Δ_rel under PathSwap tightens), train a new LoRA with outline prompt.

## License

MIT.
