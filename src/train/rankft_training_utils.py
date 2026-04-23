from __future__ import annotations

from pathlib import Path


def save_final_and_best_adapters(model, tokenizer, output_dir: str) -> tuple[str, str, bool]:
    base_dir = Path(output_dir)
    final_dir = base_dir / "final"
    best_dir = base_dir / "best"

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    best_missing = not best_dir.exists() or not any(best_dir.iterdir())
    if best_missing:
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)

    return str(final_dir), str(best_dir), best_missing
