import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/mnt/chenlibin/.cache/huggingface"

DEFAULT_MODEL = "/mnt/chenlibin/models/SciThinker-30B"
DEFAULT_OUTPUT = "/mnt/chenlibin/baselines/results/scithinker_codegrip_ideas.txt"


def parse_args():
    p = argparse.ArgumentParser(description="Run SciThinker on CodeGRIP project.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--title", default="Paths over Code: What Cross-Encoder Rerankers Actually Learn for Bug Localization")
    p.add_argument(
        "--abstract",
        default=(
            "Cross-encoder rerankers for bug localization don't read code — they read file paths. "
            "We show that a Qwen2.5-7B reranker retains 95% of its accuracy from paths alone; "
            "removing code content costs only 4 percentage points while anonymizing paths collapses "
            "accuracy by 95%. This path prior is universal across model families: Llama-3.1-8B shows "
            "75% accuracy loss from filename shuffling (vs. 73% for Qwen2.5-7B) and 92% loss from "
            "directory shuffling (vs. 46% for Qwen). The path prior functions as an implicit structural "
            "proxy, predicting import edges at AUC 0.73 and co-change edges at AUC 0.67. This explains "
            "a sharp empirical asymmetry: among seven tested graph injection points, only candidate-side "
            "expansion works (+2-6pp) while all six scorer-side methods fail — because paths already "
            "encode the structural information graphs provide. We propose path-aware hard negative mining "
            "to force content-based learning, using same-directory negatives that neutralize the path "
            "shortcut. Our 7B system achieves 54.67% file-level Acc@1 on SWE-bench Lite using only "
            "1,874 training examples."
        ),
    )
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--prompt-output", default=None, help="Optional path to save the final prompt.")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


PROJECT_CONTEXT = """
## Project: CodeGRIP — File-Level Bug Localization

### System
Pipeline: BM25 retriever (top-500) → graph expansion (~200 candidates via co-change + import edges) → cross-encoder reranker (7B LLM + LoRA, scores each file against bug report).
Trained on GREPO benchmark: 7,883 train / 1,704 test examples across 86 Python repositories.
Also evaluated on SWE-bench Lite: 300 test instances, 1,874 training examples.

### Core Finding: Path Prior
The reranker derives ~95% of accuracy from file paths, not code content.
Perturbation evidence (Qwen2.5-7B / Llama-3.1-8B):
- Path anonymization: -95% / pending
- Filename shuffle: -73% / -75%
- Directory shuffle: -46% / -92%
- Flatten directories: -23% / -98%
- Module name removal: 0% / -10%
- Add code content to scorer: -4pp / N/A

Path similarity predicts import edges (AUC 0.73) and co-change edges (AUC 0.67), making paths an implicit structural proxy.

### 7 Injection Points Tested
Only candidate-side graph expansion works (+2-6pp). All 6 scorer-side methods fail:
SFT prompt context (±0.2%), graph hard negatives (+0.7% n.s.), code content in scorer (-4pp), feature fusion (-0.02pp), score propagation (-0.4 to -14pp), mixed-pool training (negligible).
Explanation: paths already encode structure → explicit graph at scorer is redundant.

### What Didn't Work (Important)
- "Oracle Fallacy" (7pp gap between candidate pools): turned out to be checkpoint selection artifact. Under dense checkpointing (save_steps=10 vs 200), gap is <1.2pp (mean 0.49pp across 23 checkpoints).
- Delex50 (hash path tokens during training): only +0.2pp improvement with proper checkpoint selection, not the +10pp initially thought.
- SWE-bench oracle fallacy replication: 0 discordant pairs, no gap at all.

### Current Experiments Running
- Path-aware hard negative mining: 50% same-directory negatives + 25% path-distance negatives. Training step 700/790. Hypothesis: forces content-based learning by neutralizing path shortcuts.
- Attention analysis: extracting attention weights over path vs non-path tokens across 3 model families.
- Cross-LLM validation with Qwen3-8B (perturbation experiments running).

### Competitive Landscape
SWE-bench Lite file-level Acc@1:
- SweRankLLM-Large (32B, 67K data): 83.21%
- LocAgent + Claude-3.5: 77.74%
- SweRankEmbed-Large (7B, 67K data): 72.63%
- LocAgent + Qwen2.5-7B (ft): 70.80%
- Ours (7B, 1.8K data): 54.67%
No prior work has specifically studied file paths as shortcut features in code models.
No method uses GNNs — LocAgent uses graph for LLM agent navigation, not neural message passing.

### Available Assets
- Models: Qwen2.5-{0.5B,1.5B,3B,7B,14B}-Instruct, Llama-3.1-8B, Qwen3-8B
- 8x RTX 4090 (shared), 2x RTX 4090 (campus server)
- LocAgent codebase available for integration
- PathSwap-GREPO counterfactual benchmark (swaps paths between files to test path reliance)

### Key Bottlenecks
1. Path-aware neg mining may not improve R@1 enough to be a compelling "fix"
2. SWE-bench gap to SOTA is large (54.67% vs 83.21%) — we position as analytical paper, not SOTA paper
3. Need the path prior to be actionable, not just diagnostic
4. Paper framing: "path prior" is novel but needs memorable framing for NeurIPS

### Target
NeurIPS 2026 (deadline ~May 2026). Paper rewritten with path prior as main thesis. Title candidates:
- "Paths over Code: What Cross-Encoder Rerankers Actually Learn for Bug Localization"
- "Do Bug Localization Models Actually Read Code?"
"""


def build_user_prompt(title: str, abstract: str) -> str:
    parts = [
        "You are a knowledgeable and insightful AI researcher.",
        "You are given a seed paper and additional structured context about an ongoing research project.",
        "Your task is to propose one follow-up research idea that is both scientifically strong and realistically actionable given the current project state.",
        "",
        "Seed paper:",
        f"Title: {title}",
        f"Abstract: {abstract}",
        "",
        "Structured project context:",
        PROJECT_CONTEXT.strip(),
        "",
        "Instructions:",
        "1. Use the project context seriously; do not ignore failed directions, current bottlenecks, or available assets.",
        "2. Prefer an idea that can materially improve the paper's novelty or 'wow factor' without requiring a totally new project.",
        "3. Do not simply restate the current paper. Propose a follow-up that is adjacent, stronger, and feasible.",
        "4. Avoid including specific numerical results in the abstract.",
        "5. Keep the abstract moderate in length, like a normal ML paper abstract.",
        "",
        "Output format (strict, no extra text):",
        "Title: <your proposed paper title>",
        "Abstract: <your proposed abstract>",
    ]
    return "\n".join(parts)


def main():
    args = parse_args()

    print("Loading...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Loaded.")

    user_prompt = build_user_prompt(args.title, args.abstract)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.",
        },
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    if args.prompt_output:
        Path(args.prompt_output).write_text(text)

    inputs = tokenizer(text, return_tensors="pt")
    input_device = next((p.device for p in model.parameters() if p.device.type != "meta"), torch.device("cpu"))
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    print(f"Input length: {inputs['input_ids'].shape[1]} tokens")
    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    print("\n=== RESPONSE ===")
    print(response)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(response)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
