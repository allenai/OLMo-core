"""
Fold a LoRA training checkpoint back into a plain one, so it can be evaluated.

**This step is mandatory and its omission is silent.** olmo-eval builds the model from the
``MultimodalLMConfig`` under ``config.json``, which — by design — knows nothing about LoRA,
and then calls ``load_model_and_optim_state``. That function documents its own behaviour:
"if you have keys in the checkpoint dict that are not present in the current state of the
model ... those keys won't be loaded." So pointing olmo-eval at an unmerged LoRA checkpoint
does not raise. It loads the *frozen base weights*, drops the adapters on the floor, and
reports the score of the model you started from. Run this first, always.

Usage::

    # merge, writing a sibling directory that is interchangeable with a full-finetune run
    python src/scripts/merge_lora_checkpoint.py <run>/step2000 <run>/step2000-merged

    # just ask whether a checkpoint still has unmerged adapters (exit 1 if it does)
    python src/scripts/merge_lora_checkpoint.py --check <run>/step2000

CPU only, non-distributed. Peak memory is roughly the size of the model in the
checkpoint's dtype (~16 GB for Molmo2-4B in fp32).

The optimizer state is deliberately **not** carried over: a merged checkpoint has no
adapters to optimize, so it is an evaluation artifact, not a resumable training state.
Resume from the original.
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

from olmo_core.distributed.checkpoint import (
    get_checkpoint_metadata,
    load_keys,
    save_state_dict,
)
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)

MODEL_PREFIX = "model."
CHECKPOINT_SUBDIR = "model_and_optim"


def _model_keys(step_dir: Path) -> list:
    metadata = get_checkpoint_metadata(step_dir / CHECKPOINT_SUBDIR)
    return sorted(k for k in metadata.state_dict_metadata if k.startswith(MODEL_PREFIX))


def _adapter_prefixes(keys) -> list:
    """The ``model.<...>`` prefixes that carry both halves of an adapter."""
    a = {k[: -len(".lora_A")] for k in keys if k.endswith(".lora_A")}
    b = {k[: -len(".lora_B")] for k in keys if k.endswith(".lora_B")}
    if a != b:
        raise RuntimeError(
            f"Unpaired LoRA adapters in checkpoint: A-only={sorted(a - b)}, B-only={sorted(b - a)}"
        )
    return sorted(a)


def _read_alpha(step_dir: Path) -> Optional[float]:
    """``alpha`` from the run's saved config. The rank is recoverable from ``lora_A``'s
    shape, but the scaling also needs ``alpha``, which only the config records."""
    config_path = step_dir / "config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return None
    lora = (config.get("train_module") or {}).get("lora")
    if isinstance(lora, dict) and "alpha" in lora:
        return float(lora["alpha"])
    return None


def check(step_dir: Path) -> int:
    keys = _model_keys(step_dir)
    prefixes = _adapter_prefixes(keys)
    if prefixes:
        print(
            f"UNMERGED: {step_dir} carries {len(prefixes)} LoRA adapter(s). "
            "Evaluating it directly would silently score the base model.\n"
            f"Run: python src/scripts/merge_lora_checkpoint.py {step_dir} {step_dir}-merged"
        )
        return 1
    print(f"OK: {step_dir} has no LoRA adapters; it can be evaluated as-is.")
    return 0


def merge(src: Path, dst: Path, *, alpha: Optional[float], save_overwrite: bool) -> None:
    keys = _model_keys(src)
    prefixes = _adapter_prefixes(keys)
    if not prefixes:
        raise RuntimeError(
            f"{src} contains no LoRA adapters — nothing to merge. If this is a "
            "full-finetune checkpoint, evaluate it directly."
        )

    if alpha is None:
        alpha = _read_alpha(src)
    if alpha is None:
        raise RuntimeError(
            f"Could not read 'train_module.lora.alpha' from {src / 'config.json'}; pass "
            "--alpha explicitly. Guessing it would scale every adapter wrongly and the "
            "result would look like a merely mediocre model, not a broken one."
        )

    log.info("Loading %d model tensors from %s ...", len(keys), src)
    tensors: Dict[str, torch.Tensor] = dict(zip(keys, load_keys(src / CHECKPOINT_SUBDIR, keys)))

    merged = 0
    for prefix in prefixes:
        weight_key = f"{prefix}.weight"
        if weight_key not in tensors:
            raise RuntimeError(f"Adapter at '{prefix}' has no base weight '{weight_key}'")
        lora_a = tensors.pop(f"{prefix}.lora_A")
        lora_b = tensors.pop(f"{prefix}.lora_B")
        rank = lora_a.shape[0]
        weight = tensors[weight_key]
        delta = (lora_b.float() @ lora_a.float()) * (alpha / rank)
        if delta.shape != weight.shape:
            raise RuntimeError(
                f"Adapter at '{prefix}' produces a {tuple(delta.shape)} delta for a "
                f"{tuple(weight.shape)} weight"
            )
        tensors[weight_key] = (weight.float() + delta).to(weight.dtype)
        merged += 1

    assert not any(k.endswith((".lora_A", ".lora_B")) for k in tensors)
    log.info("Merged %d adapters at alpha=%.4g; writing %s ...", merged, alpha, dst)
    save_state_dict(dst / CHECKPOINT_SUBDIR, tensors, save_overwrite=save_overwrite)

    # olmo-eval reads the model architecture out of `config.json` beside `model_and_optim`.
    src_config = src / "config.json"
    if src_config.is_file():
        shutil.copy2(src_config, dst / "config.json")
    else:
        log.warning("No config.json in %s; olmo-eval will not be able to build the model", src)

    log.info("Done. Evaluate %s exactly as you would a full-finetune checkpoint.", dst)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("src", type=Path, help="Source step directory (contains model_and_optim/)")
    parser.add_argument("dst", type=Path, nargs="?", help="Destination step directory")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report whether SRC still has unmerged adapters and exit (1 if it does)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="LoRA alpha; defaults to train_module.lora.alpha in SRC/config.json",
    )
    parser.add_argument("--save-overwrite", action="store_true")
    args = parser.parse_args()

    prepare_cli_environment()

    if args.check:
        return check(args.src)
    if args.dst is None:
        parser.error("DST is required unless --check is passed")
    merge(args.src, args.dst, alpha=args.alpha, save_overwrite=args.save_overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
