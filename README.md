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

Training + eval data released separately (too large for git).

**HuggingFace:** `huggingface.co/datasets/AOoligei/codegrip` *(pending upload)*

Expected layout after download:
```
data/rankft/
  clean_swe_train.jsonl             1870 SWE-bench-train pairs
  clean_train_combined_v2.jsonl     5916 SWE + GREPO combined (Run 2)
data/swebench_lite/
  swebench_lite_test.jsonl          300 Lite instances
  swebench_lite_test_pathswap.jsonl SHA-256 perturbed variant
  swebench_bm25_final_top500.jsonl  BM25 candidate pool
  pathswap_alias_map.json           original -> hashed path map
data/swebench_verified/
  swebench_verified_test.jsonl      500 Verified
  swebench_bm25_strict.jsonl        per-commit BM25 (85% Acc@100)
```

## Headline numbers

On BM25 top-100 file pool (partial R@1):

| Model | Lite | Lite PathSwap | Δ_rel | Verified | Verified PathSwap | Δ_rel |
|---|---|---|---|---|---|---|
| 14B codeaware (aug=0.5) | **60.33** | **53.67** | **-11.0%** | **48.93** | **46.67** | **-4.6%** |
| 14B path-only (aug=0.0) | 56.33 | 39.33 | -30.2% | 48.62 | 34.18 | -29.7% |
| 7B Run 2 (combined+aug) | 56.67 | 47.33 | -16.5% | 48.13 | 45.02 | -6.5% |
| SweRankLLM-Small (7B, same pool) | 47.93 | — | — | — | — | — |

## License

MIT.
