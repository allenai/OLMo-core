"""Discover and convert StateBench ladder checkpoints to HuggingFace format.

Walks a checkpoint root (by default the StateBench model-ladder directory), finds every
saved olmo-core checkpoint (``.../stepN/model_and_optim/.metadata``), and converts each
one to HF format in a sister directory named ``stepN-hf`` next to the original. Already
converted checkpoints are skipped, so the script can be re-run as more training runs
finish. Conversions are written to ``stepN-hf.tmp`` and renamed on success, so an
interrupted run never leaves a directory that looks converted but isn't.

Every StateBench model variant routes through the HF ``olmo3_5_hybrid`` format (the
gated-attention/peri-norm/embed-scale backbone is not expressible as ``Olmo2Config``/
``Olmo3Config``), so validating and loading the converted checkpoints requires a
``transformers`` build that provides that architecture, e.g.::

    pip install 'transformers @ git+https://github.com/jopetty/transformers.git@olmo-3.5-hybrid-state-bench'

(This branch extends yashassamaga/transformers@olmo-3.5-hybrid with support for the
homogeneous layer_types the pure-transformer and pure-GDN StateBench variants need.)

Typical usage, on a weka-mounted GPU machine (the olmo-core Beaker image). Installs
land in the container and die with the session, so run this once per session::

    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv pip install --python "$(which python)" -e '.[fla]'
    uv pip install --python "$(which python)" --no-deps -e /weka/oe-training-default/tf-fork

then::

    python src/scripts/train/ladder/convert_state_bench_checkpoints.py --dry-run
    python src/scripts/train/ladder/convert_state_bench_checkpoints.py
"""

import argparse
import json
import logging
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

from olmo_core.config import DType
from olmo_core.nn.hf import convert_checkpoint_to_hf
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_ROOT = "/weka/oe-training-default/ai2-llm/model-ladders/state-bench"

_STEP_DIR_RE = re.compile(r"^step(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(DEFAULT_CHECKPOINT_ROOT),
        help=f"Checkpoint root to search (default: {DEFAULT_CHECKPOINT_ROOT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List which checkpoints would be converted or skipped, then exit.",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=1,
        help=(
            "Skip checkpoints below this step (default: 1, which skips the step0 "
            "initialization checkpoints). Set to 0 to convert those too."
        ),
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Convert only the highest-numbered step of each run instead of every checkpoint.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip logit-parity validation of each converted checkpoint.",
    )
    parser.add_argument(
        "--dtype",
        type=DType,
        choices=[DType.float32, DType.bfloat16],
        default=DType.bfloat16,
        help="Dtype the converted weights are saved in (default: bfloat16).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for conversion and validation (default: cuda when available).",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help=(
            "Value for max_position_embeddings. By default it is read from the checkpoint's "
            "saved instance-source config."
        ),
    )
    return parser.parse_args()


def find_checkpoint_dirs(root: Path) -> list[Path]:
    """Return every saved checkpoint step directory under ``root``, sorted by path."""
    step_dirs = []
    for metadata in root.rglob("model_and_optim/.metadata"):
        step_dir = metadata.parent.parent
        if _STEP_DIR_RE.match(step_dir.name):
            step_dirs.append(step_dir)
        else:
            log.warning(f"Ignoring checkpoint in unrecognized directory layout: {step_dir}")
    return sorted(step_dirs)


def keep_final_steps(step_dirs: list[Path]) -> list[Path]:
    """Keep only the highest-numbered step directory of each run."""
    final_by_run: dict[Path, Path] = {}
    for step_dir in step_dirs:
        step = int(_STEP_DIR_RE.match(step_dir.name).group(1))  # type: ignore[union-attr]
        current = final_by_run.get(step_dir.parent)
        if current is None or step > int(_STEP_DIR_RE.match(current.name).group(1)):  # type: ignore[union-attr]
            final_by_run[step_dir.parent] = step_dir
    return sorted(final_by_run.values())


def _find_first(config: Any, key: str) -> Any:
    """Depth-first search a nested config dict for the first non-null value under ``key``."""
    if isinstance(config, dict):
        value = config.get(key)
        if value is not None:
            return value
        for child in config.values():
            found = _find_first(child, key)
            if found is not None:
                return found
    elif isinstance(config, list):
        for child in config:
            found = _find_first(child, key)
            if found is not None:
                return found
    return None


def load_experiment_config(step_dir: Path) -> dict[str, Any]:
    config_path = step_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"No config.json in {step_dir}")
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    if "model" not in config:
        raise ValueError(f"{config_path} is not an olmo-core experiment config")
    return config


def convert_one(step_dir: Path, args: argparse.Namespace) -> None:
    config = load_experiment_config(step_dir)

    # Ladder configs don't have a top-level dataset/tokenizer entry; the tokenizer config
    # is recorded on the instance sources' token sources.
    tokenizer_config = _find_first(
        {k: config[k] for k in ("data_loader", "instance_sources") if k in config}, "tokenizer"
    )
    if not isinstance(tokenizer_config, dict):
        raise ValueError(f"Could not find a tokenizer config in {step_dir / 'config.json'}")

    max_sequence_length = args.max_sequence_length or _find_first(
        config.get("instance_sources"), "sequence_length"
    )
    if not isinstance(max_sequence_length, int):
        raise ValueError(
            f"Could not determine the sequence length from {step_dir / 'config.json'}; "
            "pass --max-sequence-length"
        )

    output_dir = step_dir.parent / f"{step_dir.name}-hf"
    tmp_dir = step_dir.parent / f"{step_dir.name}-hf.tmp"
    if tmp_dir.exists():
        log.warning(f"Removing stale partial conversion at {tmp_dir}")
        shutil.rmtree(tmp_dir)

    device = torch.device(args.device)
    convert_checkpoint_to_hf(
        original_checkpoint_path=step_dir,
        output_path=tmp_dir,
        transformer_config_dict=config["model"],
        tokenizer_config_dict=tokenizer_config,
        dtype=args.dtype,
        max_sequence_length=max_sequence_length,
        validate=not args.skip_validation,
        device=device,
        validation_device=device,
    )
    tmp_dir.rename(output_dir)
    log.info(f"Converted {step_dir} -> {output_dir}")


def main() -> int:
    args = parse_args()
    prepare_cli_environment()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"checkpoint root does not exist: {root}")

    step_dirs = find_checkpoint_dirs(root)
    step_dirs = [
        d
        for d in step_dirs
        if int(_STEP_DIR_RE.match(d.name).group(1)) >= args.min_step  # type: ignore[union-attr]
    ]
    if args.final_only:
        step_dirs = keep_final_steps(step_dirs)
    if not step_dirs:
        raise SystemExit(f"no checkpoints found under {root}")

    pending = [d for d in step_dirs if not (d.parent / f"{d.name}-hf").is_dir()]
    skipped = len(step_dirs) - len(pending)
    log.info(
        f"Found {len(step_dirs)} checkpoint(s) under {root}: "
        f"{skipped} already converted, {len(pending)} to convert"
    )

    if args.dry_run:
        for step_dir in step_dirs:
            state = "skip (already converted)" if step_dir not in pending else "convert"
            print(f"{state}: {step_dir.relative_to(root)}")
        return 0

    failures: list[Path] = []
    for i, step_dir in enumerate(pending, start=1):
        log.info(f"[{i}/{len(pending)}] Converting {step_dir.relative_to(root)}")
        try:
            convert_one(step_dir, args)
        except Exception:
            log.error(f"Conversion failed for {step_dir}:\n{traceback.format_exc()}")
            failures.append(step_dir)

    if failures:
        log.error(f"{len(failures)}/{len(pending)} conversion(s) failed:")
        for step_dir in failures:
            log.error(f"  {step_dir}")
        return 1
    log.info(f"All {len(pending)} conversion(s) succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
