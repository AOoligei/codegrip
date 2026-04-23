"""
Decomposed Reranker Training: Code-Residual Model
Trains the "code residual" branch of a decomposed path-prior + code-residual reranker.
Key difference from standard train_rankft.py:
- Paths are ANONYMIZED (replaced with generic tokens)
- Code content is INCLUDED in the prompt
- Same-directory negatives are used to force content-based discrimination
"""

import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import everything from the base training script
from src.train.train_rankft import *

# Override the prompt template for code-residual training
CODE_RESIDUAL_PROMPT = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Code:\n{code_content}\n\n"
    "Answer:"
)

PATH_ONLY_PROMPT = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)


def read_file_content(repo_dir: str, file_path: str, max_lines: int = 100) -> str:
    """Read first max_lines of a file from repo."""
    full_path = os.path.join(repo_dir, file_path)
    try:
        with open(full_path, 'r', errors='ignore') as f:
            lines = f.readlines()[:max_lines]
        return ''.join(lines)
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return '# (file content unavailable)'


def anonymize_path(path: str, idx: int = 0) -> str:
    """Replace file path with anonymous token."""
    return f"file_{idx:04d}.py"


def build_decomposed_prompt(issue_text: str, candidate_path: str, 
                            code_content: str = "", anonymize: bool = True,
                            file_idx: int = 0) -> str:
    """Build prompt for decomposed training."""
    if anonymize:
        display_path = anonymize_path(candidate_path, file_idx)
    else:
        display_path = candidate_path
    
    if code_content.strip():
        return CODE_RESIDUAL_PROMPT.format(
            issue_text=issue_text,
            candidate_path=display_path,
            code_content=code_content[:2000],  # Truncate to ~2000 chars
        )
    else:
        return PATH_ONLY_PROMPT.format(
            issue_text=issue_text,
            candidate_path=display_path,
        )


if __name__ == "__main__":
    import argparse
    
    # Parse args same as base, but add decomposed-specific flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--bm25_candidates", required=True)
    parser.add_argument("--file_tree_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--repo_dir", default="data/repos", help="Directory containing repo checkouts")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_negatives", type=int, default=16)
    parser.add_argument("--neg_bm25_ratio", type=float, default=0.25)
    parser.add_argument("--neg_dense_ratio", type=float, default=0.0)
    parser.add_argument("--neg_graph_ratio", type=float, default=0.0)
    parser.add_argument("--neg_random_ratio", type=float, default=0.0)
    parser.add_argument("--neg_samedir_ratio", type=float, default=0.50)
    parser.add_argument("--neg_pathdist_ratio", type=float, default=0.25)
    parser.add_argument("--neg_treeneighbor_ratio", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--delex_fraction", type=float, default=0.0)
    parser.add_argument("--anonymize_paths", action="store_true", default=True,
                        help="Anonymize file paths (default: True for code-residual training)")
    parser.add_argument("--no_anonymize_paths", action="store_false", dest="anonymize_paths")
    parser.add_argument("--include_code", action="store_true", default=True,
                        help="Include code content in prompt")
    parser.add_argument("--no_include_code", action="store_false", dest="include_code")
    parser.add_argument("--code_max_lines", type=int, default=100)
    parser.add_argument("--resume", action="store_true", default=False)
    
    args = parser.parse_args()
    
    # Monkey-patch the build_prompt function
    import src.train.train_rankft as base_module
    
    # Store repo_dir for code loading
    _repo_dir = args.repo_dir
    _include_code = args.include_code
    _anonymize = args.anonymize_paths
    _code_max_lines = args.code_max_lines
    _file_counter = [0]
    
    # Map repo names to repo directories
    _repo_map = {}
    if os.path.isdir(_repo_dir):
        for d in os.listdir(_repo_dir):
            full = os.path.join(_repo_dir, d)
            if os.path.isdir(full):
                # Map both "astropy" and "astropy__astropy" style names
                _repo_map[d] = full
                _repo_map[d.replace('_', '/')] = full  # astropy/astropy -> astropy_astropy
    
    def _find_repo_path(repo_name: str) -> str:
        """Find repo directory for a given repo name."""
        # Try exact match
        if repo_name in _repo_map:
            return _repo_map[repo_name]
        # Try with __ separator (e.g., "astropy/astropy" -> "astropy__astropy")
        clean = repo_name.replace('/', '__').replace('\\', '__')
        if clean in _repo_map:
            return _repo_map[clean]
        # Try slug
        slug = repo_name.replace('/', '_').replace('\\', '_')
        if slug in _repo_map:
            return _repo_map[slug]
        return ""
    
    _original_build_prompt = base_module.build_prompt
    
    def _decomposed_build_prompt(issue_text: str, candidate_path: str) -> str:
        """Override build_prompt to include code content and anonymize paths."""
        _file_counter[0] += 1
        
        code = ""
        if _include_code:
            # Try to read code content
            # We need repo context - use a module-level variable
            repo_path = getattr(_decomposed_build_prompt, '_current_repo_path', '')
            if repo_path:
                code = read_file_content(repo_path, candidate_path, _code_max_lines)
        
        return build_decomposed_prompt(
            issue_text, candidate_path, code,
            anonymize=_anonymize, file_idx=_file_counter[0]
        )
    
    # Patch the module
    base_module.build_prompt = _decomposed_build_prompt
    
    # Also need to patch the training loop to set current repo
    # We'll do this by wrapping the train function
    _original_train = base_module.train
    
    def _patched_train(args_inner):
        """Wrap train to inject repo path into build_prompt."""
        # The train function iterates over examples which have 'repo' field
        # We need to intercept each example to set the repo path
        # This is done by patching the NegativeSampler's sample method
        
        original_sample = base_module.NegativeSampler._sample_negatives
        
        def _patched_sample(self, repo, issue_id, gt_files, num_negatives):
            # Set repo path for code loading
            repo_path = _find_repo_path(repo)
            _decomposed_build_prompt._current_repo_path = repo_path
            return original_sample(self, repo, issue_id, gt_files, num_negatives)
        
        base_module.NegativeSampler._sample_negatives = _patched_sample
        
        return _original_train(args_inner)
    
    # Run training with patches
    print(f"=== Decomposed Reranker Training ===")
    print(f"  Anonymize paths: {_anonymize}")
    print(f"  Include code: {_include_code}")
    print(f"  Code max lines: {_code_max_lines}")
    print(f"  Repo dir: {_repo_dir}")
    print(f"  Found {len(_repo_map)} repo directories")
    print()
    
    _patched_train(args)
