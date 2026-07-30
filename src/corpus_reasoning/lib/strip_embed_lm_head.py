"""Strip embed_tokens / lm_head from PEFT adapter checkpoints in-place.

`resize_token_embeddings(<vocab>+2)` for the doc-start/doc-end tokens makes
PEFT auto-promote embed_tokens and lm_head into modules_to_save, so every
saved adapter_model.safetensors contains the full (re-trained) embedding
matrix. For gradient-norm sparsity analysis we only need the LoRA matrices,
so we drop those large modules from each per-step checkpoint.

Usage:
    python -m scripts.lib.strip_embed_lm_head <output_dir>

Strips every <output_dir>/checkpoint-*/adapter_model.safetensors. Leaves the
root-level adapter_model.safetensors untouched so the final trained
embeddings are preserved once for the run.
"""

import sys
from pathlib import Path

from safetensors.torch import load_file, save_file


_DROP_SUBSTRINGS = ("embed_tokens", "lm_head")


def strip_file(path: Path) -> tuple[int, int, int]:
    sd = load_file(str(path))
    before = sum(t.numel() * t.element_size() for t in sd.values())
    kept = {k: v for k, v in sd.items()
            if not any(s in k for s in _DROP_SUBSTRINGS)}
    dropped = len(sd) - len(kept)
    if dropped == 0:
        return 0, before, before
    save_file(kept, str(path))
    after = sum(t.numel() * t.element_size() for t in kept.values())
    return dropped, before, after


def main(output_dir: str) -> None:
    root = Path(output_dir)
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")
    total_before = 0
    total_after = 0
    n_stripped = 0
    for ckpt in sorted(root.glob("checkpoint-*")):
        adapter = ckpt / "adapter_model.safetensors"
        if not adapter.exists():
            continue
        dropped, before, after = strip_file(adapter)
        if dropped:
            n_stripped += 1
            total_before += before
            total_after += after
            print(f"  {ckpt.name}: dropped {dropped} tensors  "
                  f"{before/1e6:.0f}MB -> {after/1e6:.0f}MB")
        else:
            print(f"  {ckpt.name}: nothing to strip (already clean)")
    if n_stripped:
        saved = (total_before - total_after) / 1e9
        print(f"stripped {n_stripped} checkpoints, freed {saved:.2f} GB")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m scripts.lib.strip_embed_lm_head <output_dir>")
    main(sys.argv[1])
