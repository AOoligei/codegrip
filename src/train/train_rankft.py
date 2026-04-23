"""
RankFT: Ranking Fine-Tuning for file-level bug localization reranking.

Trains a pointwise cross-encoder with listwise (group softmax) loss.
For each training group: 1 positive + M negative (issue, file_path) pairs.
Score = logit("Yes") - logit("No") at the answer position.
Loss = -log(exp(s_pos) / sum(exp(s_i) for i in group)).

Negative mining strategies:
  - BM25-hard: top-K BM25 candidates that are NOT ground truth
  - Graph-hard: import neighbors / co-change neighbors that are NOT GT
  - Random: random files from the same repo

Usage:
    python src/train/train_rankft.py \
        --model_path /path/to/Qwen2.5-7B \
        --train_data data/grepo_text/grepo_train.jsonl \
        --bm25_candidates data/bm25_candidates/train_bm25_top500.jsonl \
        --dep_graph_dir data/dep_graphs \
        --train_data_for_cochange data/grepo_text/grepo_train.jsonl \
        --output_dir experiments/rankft_v1 \
        --device cuda:0

    # Resume from existing SFT LoRA:
    python src/train/train_rankft.py \
        --model_path /path/to/Qwen2.5-7B \
        --lora_path experiments/sft_v1/final \
        --train_data data/grepo_text/grepo_train.jsonl \
        --bm25_candidates data/bm25_candidates/train_bm25_top500.jsonl \
        --output_dir experiments/rankft_from_sft \
        --device cuda:0
"""

import hashlib
import os
import json
import argparse
import random
import math
import re
import time
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

try:
    from rankft_training_utils import save_final_and_best_adapters
except ImportError:
    from src.train.rankft_training_utils import save_final_and_best_adapters

# Deterministic seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================
# Prompt template
# ============================================================

PROMPT_TEMPLATE = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)


def build_prompt(issue_text: str, candidate_path: str) -> str:
    """Build the scoring prompt for a single (issue, file) pair."""
    return PROMPT_TEMPLATE.format(
        issue_text=issue_text,
        candidate_path=candidate_path,
    )


# ============================================================
# Path-invariant augmentation: prompts with code content
# ============================================================

PROMPT_WITH_CODE_TEMPLATE = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Code:\n{code_content}\n\n"
    "Answer:"
)


# Missing-code tripwire (per-process counter; checked in training loop)
CODE_READ_STATS = {"total": 0, "missing": 0}


def _read_code_content(repo_dir: str, repo: str, filepath: str, max_lines: int) -> str:
    """Read first N lines of a source file from repo snapshot.
    Increments CODE_READ_STATS for tripwire monitoring.
    """
    full_path = os.path.join(repo_dir, repo, filepath)
    CODE_READ_STATS["total"] += 1
    try:
        with open(full_path, 'r', errors='replace') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip())
            return '\n'.join(lines)
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        CODE_READ_STATS["missing"] += 1
        return "# (file not available)"


def build_prompt_with_code(issue_text: str, candidate_path: str,
                           code_content: str) -> str:
    """Build prompt with both path and code content."""
    return PROMPT_WITH_CODE_TEMPLATE.format(
        issue_text=issue_text,
        candidate_path=candidate_path,
        code_content=code_content[:1500],  # ~370 tokens; max_seq_length truncation handles overflow
    )


# ============================================================
# On-the-fly delexicalization for path debiasing
# ============================================================

def _stable_hash(s: str, length: int = 6) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:length]


def _get_issue_tokens(issue_text: str) -> set:
    tokens = set()
    for word in re.findall(r'[a-zA-Z_]\w{2,}', issue_text.lower()):
        tokens.add(word)
    for p in re.findall(r'[\w./]+\.py', issue_text):
        for part in p.replace('.py', '').split('/'):
            for sub in part.split('_'):
                if len(sub) >= 3:
                    tokens.add(sub.lower())
    return tokens


def _delexicalize_path(path: str, issue_tokens: set) -> str:
    parts = path.split('/')
    new_parts = []
    for part in parts:
        if part.endswith('.py'):
            base = part[:-3]
            sub_parts = base.split('_')
            new_sub = [_stable_hash(s) if (s.lower() in issue_tokens and len(s) >= 3) else s for s in sub_parts]
            new_parts.append('_'.join(new_sub) + '.py')
        else:
            sub_parts = part.split('_')
            new_sub = [_stable_hash(s) if (s.lower() in issue_tokens and len(s) >= 3) else s for s in sub_parts]
            new_parts.append('_'.join(new_sub))
    return '/'.join(new_parts)


# ============================================================
# Negative mining
# ============================================================

def build_cochange_index(
    train_data_path: str,
    min_cochange: int = 1,
) -> Dict[str, Dict[str, Set[str]]]:
    """Build per-repo co-change adjacency from training PR data.

    Returns {repo: {file: {neighbor_files}}}.
    """
    repo_pairs: Dict[str, Counter] = defaultdict(Counter)
    with open(train_data_path) as f:
        for line in f:
            item = json.loads(line)
            repo = item["repo"]
            files = sorted(item.get("changed_py_files", []))
            for i, fa in enumerate(files):
                for fb in files[i + 1:]:
                    repo_pairs[repo][(fa, fb)] += 1

    index: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for repo, pairs in repo_pairs.items():
        for (fa, fb), count in pairs.items():
            if count >= min_cochange:
                index[repo][fa].add(fb)
                index[repo][fb].add(fa)
    return index


def build_import_adjacency(
    dep_graph_dir: str,
) -> Dict[str, Dict[str, Set[str]]]:
    """Build per-repo import adjacency from dependency graphs.

    Returns {repo: {file: {neighbor_files}}}.
    """
    result: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    if not os.path.isdir(dep_graph_dir):
        print(f"  Warning: dep_graph_dir not found: {dep_graph_dir}")
        return result
    for fname in os.listdir(dep_graph_dir):
        if not fname.endswith("_rels.json"):
            continue
        repo = fname.replace("_rels.json", "")
        with open(os.path.join(dep_graph_dir, fname)) as f:
            rels = json.load(f)
        for importer, imported_list in rels.get("file_imports", {}).items():
            for imported in imported_list:
                if importer.endswith(".py") and imported.endswith(".py"):
                    result[repo][importer].add(imported)
                    result[repo][imported].add(importer)
    return result


def load_file_trees(file_tree_dir: str) -> Dict[str, List[str]]:
    """Load per-repo Python file lists from file tree JSONs.

    Returns {repo: [py_file_paths]}.
    """
    repo_files: Dict[str, List[str]] = {}
    if not os.path.isdir(file_tree_dir):
        print(f"  Warning: file_tree_dir not found: {file_tree_dir}")
        return repo_files
    for fname in os.listdir(file_tree_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(file_tree_dir, fname)) as f:
            tree = json.load(f)
        repo = tree.get("repo", fname.replace(".json", ""))
        repo_files[repo] = tree.get("py_files", [])
    return repo_files


def _path_component_distance(a: str, b: str) -> int:
    """Edit distance between path components (directories + filename)."""
    parts_a = a.split("/")
    parts_b = b.split("/")
    m, n = len(parts_a), len(parts_b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cost = 0 if parts_a[i - 1] == parts_b[j - 1] else 1
            dp[j], prev = min(dp[j] + 1, dp[j - 1] + 1, prev + cost), dp[j]
    return dp[n]


class NegativeSampler:
    """Sample negatives for a given (repo, issue) with configurable mix.

    Supports 7 negative source types:
      - bm25: BM25-hard (textually similar but wrong)
      - dense: dense-retriever-hard (e.g. E5 top-K candidates, wrong ones)
      - graph: co-change + import neighbors of GT (structurally related)
      - samedir: files in same directory as GT
      - pathdist: files with smallest path-component edit distance to GT
      - treeneighbor: files in sibling directories of GT
      - random: uniform random from repo
    """

    def __init__(
        self,
        bm25_candidates: Dict[str, Dict],
        cochange_index: Dict[str, Dict[str, Set[str]]],
        import_index: Dict[str, Dict[str, Set[str]]],
        repo_files: Dict[str, List[str]],
        neg_bm25_ratio: float = 0.5,
        neg_dense_ratio: float = 0.0,
        neg_graph_ratio: float = 0.25,
        neg_random_ratio: float = 0.25,
        neg_samedir_ratio: float = 0.0,
        neg_pathdist_ratio: float = 0.0,
        neg_treeneighbor_ratio: float = 0.0,
        dense_candidates: Optional[Dict[str, Dict]] = None,
    ):
        self.bm25_candidates = bm25_candidates
        self.dense_candidates = dense_candidates or {}
        self.cochange_index = cochange_index
        self.import_index = import_index
        self.repo_files = repo_files
        self.neg_bm25_ratio = neg_bm25_ratio
        self.neg_dense_ratio = neg_dense_ratio
        self.neg_graph_ratio = neg_graph_ratio
        self.neg_random_ratio = neg_random_ratio
        self.neg_samedir_ratio = neg_samedir_ratio
        self.neg_pathdist_ratio = neg_pathdist_ratio
        self.neg_treeneighbor_ratio = neg_treeneighbor_ratio

        # Pre-build directory index per repo for samedir/treeneighbor
        self._dir_index: Dict[str, Dict[str, List[str]]] = {}
        for repo, files in repo_files.items():
            dir_map: Dict[str, List[str]] = defaultdict(list)
            for fpath in files:
                dir_map[os.path.dirname(fpath)].append(fpath)
            self._dir_index[repo] = dict(dir_map)

    def _sample_source(
        self,
        pool: List[str],
        n_want: int,
        selected: Set[str],
        result: List[str],
    ) -> None:
        """Add up to n_want files from pool to result, skipping already selected."""
        count = 0
        for f in pool:
            if count >= n_want:
                break
            if f not in selected:
                selected.add(f)
                result.append(f)
                count += 1

    def sample(
        self,
        repo: str,
        issue_id,
        gt_files: Set[str],
        num_negatives: int,
    ) -> Tuple[List[str], List[str]]:
        """Sample num_negatives negative files for a given example.

        Returns (file_paths, neg_types) where neg_types[i] is one of
        'bm25', 'dense', 'graph', 'samedir', 'pathdist', 'treeneighbor', 'random', 'pad'.
        """
        n_bm25 = int(round(num_negatives * self.neg_bm25_ratio))
        n_dense = int(round(num_negatives * self.neg_dense_ratio))
        n_graph = int(round(num_negatives * self.neg_graph_ratio))
        n_samedir = int(round(num_negatives * self.neg_samedir_ratio))
        n_pathdist = int(round(num_negatives * self.neg_pathdist_ratio))
        n_treeneighbor = int(round(num_negatives * self.neg_treeneighbor_ratio))
        n_random = num_negatives - n_bm25 - n_dense - n_graph - n_samedir - n_pathdist - n_treeneighbor

        selected: Set[str] = set()
        result: List[str] = []
        neg_types: List[str] = []
        all_files = self.repo_files.get(repo, [])

        def _sample_source_typed(pool, n_want, neg_type):
            count = 0
            for f in pool:
                if count >= n_want:
                    break
                if f not in selected:
                    selected.add(f)
                    result.append(f)
                    neg_types.append(neg_type)
                    count += 1

        # 1) BM25-hard negatives
        bm25_key = f"{repo}_{issue_id}"
        bm25_cands = self.bm25_candidates.get(bm25_key, {}).get("candidates", [])
        bm25_negs = [c for c in bm25_cands if c not in gt_files]
        _sample_source_typed(bm25_negs[:n_bm25 + 10], n_bm25, "bm25")

        # 1b) Dense-retriever-hard negatives (e.g. E5 top-K, excluding GT)
        if n_dense > 0:
            dense_key = f"{repo}_{issue_id}"
            dense_cands = self.dense_candidates.get(dense_key, {}).get("candidates", [])
            dense_negs = [c for c in dense_cands if c not in gt_files]
            _sample_source_typed(dense_negs[:n_dense + 10], n_dense, "dense")

        # 2) Graph-hard negatives (import + co-change neighbors of GT, minus GT)
        if n_graph > 0:
            graph_pool: Set[str] = set()
            repo_cochange = self.cochange_index.get(repo, {})
            repo_imports = self.import_index.get(repo, {})
            for gt_f in gt_files:
                graph_pool.update(repo_cochange.get(gt_f, set()))
                graph_pool.update(repo_imports.get(gt_f, set()))
            graph_pool -= gt_files
            graph_list = list(graph_pool)
            random.shuffle(graph_list)
            _sample_source_typed(graph_list, n_graph, "graph")

        # 3) Same-directory negatives
        if n_samedir > 0:
            dir_index = self._dir_index.get(repo, {})
            samedir_pool: Set[str] = set()
            for gt_f in gt_files:
                gt_dir = os.path.dirname(gt_f)
                samedir_pool.update(dir_index.get(gt_dir, []))
            samedir_pool -= gt_files
            samedir_list = list(samedir_pool)
            random.shuffle(samedir_list)
            _sample_source_typed(samedir_list, n_samedir, "samedir")

        # 4) Path-edit-distance negatives (closest path components to GT)
        if n_pathdist > 0:
            non_gt = [f for f in all_files if f not in gt_files and f not in selected]
            gt_list = list(gt_files)
            # Score each candidate by min distance to any GT file
            scored = []
            for f in non_gt:
                min_dist = min(_path_component_distance(f, g) for g in gt_list)
                scored.append((min_dist, f))
            scored.sort(key=lambda x: x[0])
            pathdist_list = [f for _, f in scored]
            _sample_source_typed(pathdist_list, n_pathdist, "pathdist")

        # 5) Tree-neighbor negatives (sibling directories of GT)
        if n_treeneighbor > 0:
            dir_index = self._dir_index.get(repo, {})
            treeneighbor_pool: Set[str] = set()
            for gt_f in gt_files:
                gt_dir = os.path.dirname(gt_f)
                parent_dir = os.path.dirname(gt_dir)
                # Find sibling directories (children of parent)
                for d, files_in_d in dir_index.items():
                    if d != gt_dir and os.path.dirname(d) == parent_dir:
                        treeneighbor_pool.update(files_in_d)
            treeneighbor_pool -= gt_files
            treeneighbor_list = list(treeneighbor_pool)
            random.shuffle(treeneighbor_list)
            _sample_source_typed(treeneighbor_list, n_treeneighbor, "treeneighbor")

        # 6) Random negatives from same repo
        random_pool = [f for f in all_files if f not in gt_files and f not in selected]
        random.shuffle(random_pool)
        n_still_needed = num_negatives - len(result)
        _sample_source_typed(random_pool, max(n_random, n_still_needed), "random")

        # If we still don't have enough (small repo), pad with whatever is available
        if len(result) < num_negatives:
            remaining_pool = [f for f in all_files if f not in gt_files and f not in selected]
            random.shuffle(remaining_pool)
            _sample_source_typed(remaining_pool, num_negatives - len(result), "random")

        # Final fallback: duplicate if repo is very small
        while len(result) < num_negatives and len(result) > 0:
            idx = random.randint(0, len(result) - 1)
            result.append(result[idx])
            neg_types.append("pad")

        return result[:num_negatives], neg_types[:num_negatives]


# ============================================================
# Data loading
# ============================================================

def load_train_data(path: str) -> List[Dict]:
    """Load GREPO training JSONL."""
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if item.get("changed_py_files"):
                data.append(item)
    print(f"  Loaded {len(data)} training examples from {path}")
    return data


def load_bm25_candidates(path: str) -> Dict[str, Dict]:
    """Load precomputed BM25 top-K candidates.

    Expected format per line:
        {"repo": ..., "issue_id": ..., "candidates": [file1, file2, ...]}

    Returns {repo_issueId: {"candidates": [...]}}.
    """
    result = {}
    if not os.path.isfile(path):
        print(f"  Warning: BM25 candidates file not found: {path}")
        return result
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            key = f"{item['repo']}_{item['issue_id']}"
            result[key] = item
    print(f"  Loaded BM25 candidates for {len(result)} examples from {path}")
    return result


# ============================================================
# Score extraction
# ============================================================

def get_yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    """Get token IDs for 'Yes' and 'No'."""
    # Try common tokenizations
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    # Take the first token if multi-token
    yes_id = yes_ids[0]
    no_id = no_ids[0]
    print(f"  Yes token ID: {yes_id} (decoded: '{tokenizer.decode([yes_id])}')")
    print(f"  No token ID: {no_id} (decoded: '{tokenizer.decode([no_id])}')")
    return yes_id, no_id


def compute_scores(
    model,
    tokenizer,
    prompts: List[str],
    yes_id: int,
    no_id: int,
    max_seq_length: int,
    device: str,
    mini_batch_size: int = 4,
) -> torch.Tensor:
    """Forward pass: compute score = logit(Yes) - logit(No) for each prompt.

    Processes prompts in mini-batches to avoid OOM.
    Returns tensor of shape (len(prompts),) on the model's device.
    """
    all_scores = []

    for i in range(0, len(prompts), mini_batch_size):
        batch_prompts = prompts[i : i + mini_batch_size]

        encodings = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, seq_len, vocab_size)

        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.size(0), device=device)
        last_logits = logits[batch_indices, seq_lengths]  # (batch, vocab_size)

        scores = last_logits[:, yes_id] - last_logits[:, no_id]  # (batch,)
        all_scores.append(scores)

        del outputs, logits, input_ids, attention_mask
        torch.cuda.empty_cache()

    return torch.cat(all_scores, dim=0)


# ============================================================
# Training loop
# ============================================================

def train(args):
    """Main training routine."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Re-seed with args.seed so different seeds produce different runs
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Save config
    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    device = args.device

    # ---- Load tokenizer ----
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for batch scoring with causal LM

    yes_id, no_id = get_yes_no_token_ids(tokenizer)

    # ---- Load base model ----
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    # ---- Attach or load LoRA ----
    if args.lora_path:
        print(f"Loading existing LoRA adapter from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=True)
    else:
        print(f"Creating fresh LoRA adapter (rank={args.lora_rank})...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,  # alpha = 2 * rank (convention: 64 for rank 32)
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.train()

    # ---- Load training data ----
    print("Loading training data...")
    train_data = load_train_data(args.train_data)

    print("Loading BM25 candidates...")
    bm25_candidates = load_bm25_candidates(args.bm25_candidates)

    dense_candidates = {}
    if args.dense_candidates:
        print("Loading dense retriever candidates...")
        dense_candidates = load_bm25_candidates(args.dense_candidates)
        print(f"  Dense candidates for {len(dense_candidates)} examples")

    print("Building graph indexes...")
    cochange_index = {}
    if args.train_data_for_cochange:
        cochange_index = build_cochange_index(args.train_data_for_cochange)
        print(f"  Co-change index for {len(cochange_index)} repos")
    elif args.train_data:
        cochange_index = build_cochange_index(args.train_data)
        print(f"  Co-change index for {len(cochange_index)} repos")

    import_index = {}
    if args.dep_graph_dir:
        import_index = build_import_adjacency(args.dep_graph_dir)
        print(f"  Import index for {len(import_index)} repos")

    print("Loading file trees...")
    file_tree_dir = args.file_tree_dir if args.file_tree_dir else "data/file_trees"
    repo_files = load_file_trees(file_tree_dir)
    print(f"  File trees for {len(repo_files)} repos")

    # Create negative sampler
    neg_sampler = NegativeSampler(
        bm25_candidates=bm25_candidates,
        cochange_index=cochange_index,
        import_index=import_index,
        repo_files=repo_files,
        neg_bm25_ratio=args.neg_bm25_ratio,
        neg_dense_ratio=args.neg_dense_ratio,
        neg_graph_ratio=args.neg_graph_ratio,
        neg_random_ratio=args.neg_random_ratio,
        neg_samedir_ratio=args.neg_samedir_ratio,
        neg_pathdist_ratio=args.neg_pathdist_ratio,
        neg_treeneighbor_ratio=args.neg_treeneighbor_ratio,
        dense_candidates=dense_candidates,
    )

    # ---- Optimizer and scheduler ----
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    total_examples = len(train_data)
    steps_per_epoch = math.ceil(total_examples / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * 0.05)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Training loop ----
    print(f"\n{'='*60}")
    print(f"RankFT Training")
    print(f"  Examples: {total_examples}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Negatives per positive: {args.num_negatives}")
    print(f"  Neg mix: BM25={args.neg_bm25_ratio} Dense={args.neg_dense_ratio} "
          f"Graph={args.neg_graph_ratio} Random={args.neg_random_ratio}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}")
    print(f"  LR: {args.learning_rate}, Warmup: {warmup_steps}")
    print(f"  Max seq length: {args.max_seq_length}")
    if args.delex_fraction > 0:
        print(f"  Delex fraction: {args.delex_fraction} (path debiasing ON)")
    print(f"{'='*60}\n")

    global_step = 0
    accumulated_loss = 0.0
    log_losses = []
    best_loss = float("inf")
    resume_from_epoch = 0
    resume_from_example = 0
    start_time = time.time()

    # --- Resume from checkpoint ---
    if getattr(args, 'resume', False):
        # Find latest checkpoint
        import glob as _glob
        ckpt_dirs = sorted(
            [d for d in _glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
             if os.path.isfile(os.path.join(d, "training_state.json"))],
            key=lambda d: int(os.path.basename(d).split("-")[1])
        )
        if ckpt_dirs:
            latest_ckpt = ckpt_dirs[-1]
            print(f"\n  Resuming from {latest_ckpt}")
            # Load training state
            with open(os.path.join(latest_ckpt, "training_state.json")) as f:
                state = json.load(f)
            global_step = state["global_step"]
            log_losses = state.get("loss_history", [])
            best_loss = state.get("best_loss", float("inf"))
            resume_from_epoch = state["epoch"]
            resume_from_example = (global_step % steps_per_epoch) * args.gradient_accumulation_steps
            # Load adapter weights (PeftModel imported at top of file)
            model = PeftModel.from_pretrained(
                model.base_model.model if hasattr(model, 'base_model') else model,
                latest_ckpt, is_trainable=True
            )
            model.to(args.device)
            # Rebuild optimizer for new model params
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.learning_rate, weight_decay=0.01,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            # Load optimizer/scheduler state
            opt_path = os.path.join(latest_ckpt, "optimizer_scheduler.pt")
            if os.path.exists(opt_path):
                opt_state = torch.load(opt_path, map_location=args.device)
                optimizer.load_state_dict(opt_state["optimizer"])
                scheduler.load_state_dict(opt_state["scheduler"])
                print(f"  Restored optimizer/scheduler state")
            print(f"  Resumed: step={global_step}, epoch={resume_from_epoch}, "
                  f"best_loss={best_loss:.4f}, {len(log_losses)} loss entries")
        else:
            print("  --resume specified but no checkpoints found, starting fresh")

    # --- Diagnostics ---
    diag_path = os.path.join(args.output_dir, "training_diagnostics.jsonl")
    diag_file = open(diag_path, "a" if global_step > 0 else "w")
    # Accumulators for per-step (across gradient accumulation) diagnostics
    diag_pos_scores: List[float] = []
    diag_neg_scores: List[float] = []
    diag_neg_type_scores: Dict[str, List[float]] = defaultdict(list)
    diag_neg_type_losses: Dict[str, List[float]] = defaultdict(list)

    for epoch in range(args.num_epochs):
        # Skip completed epochs when resuming
        if epoch < resume_from_epoch:
            continue

        # Shuffle training data each epoch
        indices = list(range(total_examples))
        random.shuffle(indices)

        optimizer.zero_grad()

        for ex_idx, data_idx in enumerate(indices):
            # Skip examples already processed when resuming
            if epoch == resume_from_epoch and ex_idx < resume_from_example:
                continue

            example = train_data[data_idx]
            repo = example["repo"]
            issue_id = example["issue_id"]
            issue_text = example["issue_text"]
            gt_files = set(example["changed_py_files"])

            # Pick one positive file (if multiple GT, randomly choose one)
            positive_file = random.choice(list(gt_files))

            # Sample negatives
            negatives, neg_types = neg_sampler.sample(
                repo=repo,
                issue_id=issue_id,
                gt_files=gt_files,
                num_negatives=args.num_negatives,
            )

            if len(negatives) == 0:
                continue

            # Build prompts: positive first, then negatives
            candidates = [positive_file] + negatives

            # On-the-fly delexicalization for path debiasing
            if args.delex_fraction > 0 and random.random() < args.delex_fraction:
                issue_tokens = _get_issue_tokens(issue_text)
                candidates = [_delexicalize_path(c, issue_tokens) for c in candidates]

            # Prompt builder: code is included whenever --include_code is set OR
            # path_augment_fraction > 0 (latter implies code-aware training).
            include_code = args.include_code or (args.path_augment_fraction > 0)
            # Always consume one random.random() for the shuffle decision so the
            # RNG sequence is identical across augment=0 and augment=0.5 ablations.
            _shuffle_roll = random.random()
            shuffle_now = (args.path_augment_fraction > 0
                           and _shuffle_roll < args.path_augment_fraction)
            if include_code:
                if shuffle_now:
                    # Shuffle paths among candidates; code stays with original file
                    display_paths = list(candidates)
                    random.shuffle(display_paths)
                    prompts = []
                    for j, original_file in enumerate(candidates):
                        code = _read_code_content(args.repo_dir, repo,
                                                  original_file, args.code_max_lines)
                        prompts.append(build_prompt_with_code(
                            issue_text, display_paths[j], code))
                else:
                    # Code-aware prompt with original (non-shuffled) paths
                    prompts = []
                    for cand in candidates:
                        code = _read_code_content(args.repo_dir, repo,
                                                  cand, args.code_max_lines)
                        prompts.append(build_prompt_with_code(issue_text, cand, code))
            else:
                # Original path-only prompt (no code)
                prompts = [build_prompt(issue_text, cand) for cand in candidates]

            # Forward pass and compute scores
            try:
                scores = compute_scores(
                    model, tokenizer, prompts,
                    yes_id, no_id,
                    args.max_seq_length, device,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at example {ex_idx}, skipping. "
                          f"Group size: {len(prompts)}")
                    torch.cuda.empty_cache()
                    continue
                raise

            # Listwise loss: -log_softmax(scores)[0]
            # scores[0] is the positive, rest are negatives
            log_probs = F.log_softmax(scores, dim=0)
            loss = -log_probs[0]

            # --- Collect diagnostics (detached, no grad) ---
            with torch.no_grad():
                s = scores.detach().float()
                diag_pos_scores.append(s[0].item())
                neg_s = s[1:]  # negative scores
                for ni, ns in enumerate(neg_s.tolist()):
                    diag_neg_scores.append(ns)
                    if ni < len(neg_types):
                        nt = neg_types[ni]
                        diag_neg_type_scores[nt].append(ns)
                # Per-negative-type contribution: -log_softmax value for each neg
                neg_log_probs = log_probs[1:]
                for ni, nlp in enumerate(neg_log_probs.tolist()):
                    if ni < len(neg_types):
                        # Use the negative of log_prob as a proxy for how hard
                        # this negative is (higher = model gives it more mass)
                        diag_neg_type_losses[neg_types[ni]].append(-nlp)

            # Scale by gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            accumulated_loss += loss.item()

            # Gradient accumulation step
            if (ex_idx + 1) % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                step_loss = accumulated_loss
                log_losses.append(step_loss)
                accumulated_loss = 0.0

                # --- Write per-step diagnostics record ---
                if diag_pos_scores:
                    pos_arr = np.array(diag_pos_scores)
                    neg_arr = np.array(diag_neg_scores) if diag_neg_scores else np.array([0.0])
                    diag_record = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": step_loss,
                        "pos_score_mean": float(pos_arr.mean()),
                        "neg_score_mean": float(neg_arr.mean()),
                        "score_gap": float(pos_arr.mean() - neg_arr.mean()),
                        "pos_score_min": float(pos_arr.min()),
                        "pos_score_max": float(pos_arr.max()),
                        "neg_score_min": float(neg_arr.min()),
                        "neg_score_max": float(neg_arr.max()),
                    }
                    # Per-negative-type breakdown
                    for nt in sorted(diag_neg_type_scores.keys()):
                        nt_scores = diag_neg_type_scores[nt]
                        nt_losses = diag_neg_type_losses.get(nt, [])
                        diag_record[f"neg_{nt}_score_mean"] = float(np.mean(nt_scores))
                        diag_record[f"neg_{nt}_count"] = len(nt_scores)
                        if nt_losses:
                            diag_record[f"neg_{nt}_loss_mean"] = float(np.mean(nt_losses))
                    diag_file.write(json.dumps(diag_record) + "\n")
                    diag_file.flush()

                # Reset diagnostics accumulators
                diag_pos_scores.clear()
                diag_neg_scores.clear()
                diag_neg_type_scores.clear()
                diag_neg_type_losses.clear()

                # --- Score distribution summary every 50 steps ---
                if global_step % 50 == 0:
                    # Read last 50 records from diagnostics
                    recent_pos = []
                    recent_neg = []
                    recent_gap = []
                    recent_neg_by_type: Dict[str, List[float]] = defaultdict(list)
                    diag_file.flush()
                    try:
                        with open(diag_path) as _df:
                            lines = _df.readlines()
                        for line in lines[-50:]:
                            rec = json.loads(line)
                            recent_pos.append(rec["pos_score_mean"])
                            recent_neg.append(rec["neg_score_mean"])
                            recent_gap.append(rec["score_gap"])
                            for k, v in rec.items():
                                if k.startswith("neg_") and k.endswith("_score_mean"):
                                    nt_name = k[4:-11]  # strip neg_ and _score_mean
                                    recent_neg_by_type[nt_name].append(v)
                    except Exception:
                        pass
                    if recent_pos:
                        print(f"  [Diagnostics @ step {global_step}]")
                        print(f"    Pos score:  mean={np.mean(recent_pos):.3f}  "
                              f"min={np.min(recent_pos):.3f}  max={np.max(recent_pos):.3f}")
                        print(f"    Neg score:  mean={np.mean(recent_neg):.3f}  "
                              f"min={np.min(recent_neg):.3f}  max={np.max(recent_neg):.3f}")
                        print(f"    Score gap:  mean={np.mean(recent_gap):.3f}")
                        for nt_name in sorted(recent_neg_by_type.keys()):
                            nt_vals = recent_neg_by_type[nt_name]
                            print(f"    Neg[{nt_name}] score: mean={np.mean(nt_vals):.3f}")

                # Logging
                if global_step % args.logging_steps == 0:
                    avg_recent = np.mean(log_losses[-args.logging_steps:])
                    elapsed = time.time() - start_time
                    current_lr = scheduler.get_last_lr()[0]
                    miss_frac = (CODE_READ_STATS["missing"] / max(1, CODE_READ_STATS["total"]))
                    print(
                        f"  [Epoch {epoch+1}/{args.num_epochs}] "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {step_loss:.4f} (avg: {avg_recent:.4f}) | "
                        f"LR: {current_lr:.2e} | "
                        f"Time: {elapsed:.0f}s | "
                        f"CodeMiss: {CODE_READ_STATS['missing']}/{CODE_READ_STATS['total']} ({miss_frac*100:.1f}%)"
                    )
                    # Tripwire: if code reads are silently failing >5%, abort.
                    # Fires whenever code is included (augment OR --include_code).
                    if ((args.path_augment_fraction > 0 or args.include_code)
                            and CODE_READ_STATS["total"] >= 200
                            and miss_frac > 0.05):
                        raise RuntimeError(
                            f"Missing-code rate {miss_frac*100:.1f}% > 5% threshold. "
                            f"Likely repo-naming mismatch (repo_dir={args.repo_dir}). "
                            f"Aborting to avoid training on placeholder code."
                        )

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    print(f"  Saving checkpoint to {ckpt_dir}...")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

                    # Save training state (including optimizer/scheduler for resume)
                    state = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "loss_history": log_losses,
                        "best_loss": best_loss,
                    }
                    with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
                        json.dump(state, f, indent=2)
                    torch.save({
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }, os.path.join(ckpt_dir, "optimizer_scheduler.pt"))

                    # Track best
                    recent_avg = np.mean(log_losses[-min(50, len(log_losses)):])
                    if recent_avg < best_loss:
                        best_loss = recent_avg
                        best_dir = os.path.join(args.output_dir, "best")
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
                        print(f"    New best (avg loss: {best_loss:.4f})")

        # Handle remaining accumulated gradients at end of epoch
        if (ex_idx + 1) % args.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        print(f"\n  Epoch {epoch+1} complete. "
              f"Avg loss: {np.mean(log_losses[-steps_per_epoch:]):.4f}")

    # ---- Close diagnostics ----
    diag_file.close()
    print(f"\n  Training diagnostics saved to {diag_path}")

    # ---- Final best check (save_steps may have missed the true best) ----
    recent_avg = np.mean(log_losses[-min(50, len(log_losses)):])
    if recent_avg < best_loss:
        best_loss = recent_avg
        best_dir = os.path.join(args.output_dir, "best")
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        print(f"  End-of-training best update (avg loss: {best_loss:.4f})")

    # ---- Save final model ----
    print(f"\nSaving final model under {args.output_dir}...")
    final_dir, best_dir, best_backfilled = save_final_and_best_adapters(
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
    )
    if best_backfilled:
        print(f"  Backfilled missing best adapter at {best_dir}")

    # Save loss history
    with open(os.path.join(args.output_dir, "loss_history.json"), "w") as f:
        json.dump({"losses": log_losses, "total_steps": global_step}, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"  Total steps: {global_step}")
    print(f"  Final avg loss: {np.mean(log_losses[-50:]):.4f}")
    print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Output: {args.output_dir}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RankFT: Ranking Fine-Tuning for bug localization reranking"
    )

    # Model
    parser.add_argument("--model_path", required=True,
                        help="Path to base model (e.g., Qwen2.5-7B)")
    parser.add_argument("--lora_path", default=None,
                        help="Optional: start from existing LoRA checkpoint")

    # Data
    parser.add_argument("--train_data", required=True,
                        help="GREPO training data JSONL")
    parser.add_argument("--bm25_candidates", required=True,
                        help="Precomputed BM25 top-500 per example (JSONL)")
    parser.add_argument("--dense_candidates", default=None,
                        help="Precomputed dense retriever (e.g. E5) top-500 "
                             "per example (JSONL). Used for mixed-pool training.")
    parser.add_argument("--dep_graph_dir", default="data/dep_graphs",
                        help="Directory with per-repo dependency graph JSONs")
    parser.add_argument("--train_data_for_cochange", default=None,
                        help="Training data for building co-change index "
                             "(defaults to --train_data)")
    parser.add_argument("--file_tree_dir", default="data/file_trees",
                        help="Directory with per-repo file tree JSONs "
                             "(for random negative pool)")

    # Output
    parser.add_argument("--output_dir", required=True,
                        help="Where to save checkpoints")

    # Hardware
    parser.add_argument("--device", default="cuda:0",
                        help="CUDA device")

    # Negative mining
    parser.add_argument("--num_negatives", type=int, default=32,
                        help="Number of negatives per positive (M)")
    parser.add_argument("--neg_bm25_ratio", type=float, default=0.5,
                        help="Fraction of BM25-hard negatives")
    parser.add_argument("--neg_dense_ratio", type=float, default=0.0,
                        help="Fraction of dense-retriever-hard negatives "
                             "(requires --dense_candidates)")
    parser.add_argument("--neg_graph_ratio", type=float, default=0.25,
                        help="Fraction of graph-hard negatives")
    parser.add_argument("--neg_random_ratio", type=float, default=0.25,
                        help="Fraction of random negatives")
    parser.add_argument("--neg_samedir_ratio", type=float, default=0.0,
                        help="Fraction of same-directory negatives")
    parser.add_argument("--neg_pathdist_ratio", type=float, default=0.0,
                        help="Fraction of path-edit-distance negatives")
    parser.add_argument("--neg_treeneighbor_ratio", type=float, default=0.0,
                        help="Fraction of tree-neighbor (sibling dir) negatives")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Micro-batch size (groups per step, typically 1)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Max tokens per (issue, file) prompt")

    # LoRA config
    parser.add_argument("--lora_rank", type=int, default=32)

    # Resume
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint in output_dir")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--delex_fraction", type=float, default=0.0,
                        help="Fraction of examples to delexicalize (0=none, 0.5=half). "
                             "Hashes path tokens that overlap with issue text to break "
                             "lexical shortcuts and force non-lexical learning.")
    parser.add_argument("--path_augment_fraction", type=float, default=0.0,
                        help="Fraction of examples where paths are shuffled among candidates "
                             "(0=none, 0.5=half). Requires --repo_dir and --code_max_lines. "
                             "Forces model to use code content when paths are unreliable.")
    parser.add_argument("--code_max_lines", type=int, default=30,
                        help="Max lines of code to include when path_augment_fraction > 0")
    parser.add_argument("--repo_dir", default="data/repos",
                        help="Directory containing repo snapshots (for code content)")
    parser.add_argument("--include_code", action="store_true",
                        help="Always include code in the prompt, even when path_augment_fraction=0. "
                             "Use this for ablations comparing 'code+shuffle' vs 'code only'.")

    args = parser.parse_args()

    # Validate negative ratios
    ratio_sum = (args.neg_bm25_ratio + args.neg_dense_ratio + args.neg_graph_ratio
                 + args.neg_random_ratio + args.neg_samedir_ratio + args.neg_pathdist_ratio
                 + args.neg_treeneighbor_ratio)
    if abs(ratio_sum - 1.0) > 0.01:
        parser.error(
            f"Negative ratios must sum to 1.0, got {ratio_sum:.2f} "
            f"(bm25={args.neg_bm25_ratio}, dense={args.neg_dense_ratio}, "
            f"graph={args.neg_graph_ratio}, samedir={args.neg_samedir_ratio}, "
            f"pathdist={args.neg_pathdist_ratio}, treeneighbor={args.neg_treeneighbor_ratio}, "
            f"random={args.neg_random_ratio})"
        )

    # Validate dense_candidates requirement
    if args.neg_dense_ratio > 0 and not args.dense_candidates:
        parser.error(
            f"--neg_dense_ratio={args.neg_dense_ratio} requires --dense_candidates"
        )

    train(args)


if __name__ == "__main__":
    main()
