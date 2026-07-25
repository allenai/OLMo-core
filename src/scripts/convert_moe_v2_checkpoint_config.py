"""
Migrate a checkpoint's ``config.json`` off the legacy ``MoEFusedV2*`` / ``MoEV2*`` /
``olmo_core.nn.moe.v2.*`` class paths onto the canonical ``OLMoDDP*`` (``olmo_core.nn.ddp.*``) ones.

The fused MoE-v2 stack was promoted to the ``OLMoDDP*`` names, and the old names/module paths (kept
for a while as aliases and re-export shims) have been removed. Model and optimizer *weights* are
unaffected — the rename preserved the same class objects, so parameter names are identical — but a
``config.json`` serialized under the old names records stale ``_CLASS_`` values that no longer
resolve. This script rewrites just those ``_CLASS_`` strings in place so the checkpoint loads again.

Usage::

    # Rewrite one or more checkpoints (a checkpoint dir or a config.json, local or remote).
    python -m scripts.convert_moe_v2_checkpoint_config /path/to/checkpoint/step10000
    python -m scripts.convert_moe_v2_checkpoint_config s3://bucket/run/step10000/config.json

    # Preview the changes without writing.
    python -m scripts.convert_moe_v2_checkpoint_config --dry-run /path/to/checkpoint/step10000

    # Write the migrated config to a different location instead of in place.
    python -m scripts.convert_moe_v2_checkpoint_config \
        --output /tmp/config.json /path/to/checkpoint/step10000
"""

import json
import logging
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple

import click

from olmo_core.config import Config
from olmo_core.io import (
    file_exists,
    get_bytes_range,
    get_file_size,
    is_url,
    join_path,
    normalize_path,
    upload,
)
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)

CONFIG_FILENAME = "config.json"

#: Maps every legacy ``_CLASS_`` value to its canonical replacement. Covers both the renamed-symbol
#: aliases (``MoEFusedV2*`` / ``MoEV2*``) and the relocated-but-canonically-named entries under the
#: former ``olmo_core.nn.moe.v2.*`` / ``moe_train_module`` module paths.
CLASS_PATH_REWRITES: Dict[str, str] = {
    # Model config (same module, renamed symbol).
    "olmo_core.nn.transformer.config.MoEFusedV2TransformerConfig": (
        "olmo_core.nn.transformer.config.OLMoDDPModelConfig"
    ),
    # Block config/class (renamed symbol and/or relocated module path).
    "olmo_core.nn.moe.v2.block.MoEFusedV2TransformerBlockConfig": (
        "olmo_core.nn.ddp.block.OLMoDDPTransformerBlockConfig"
    ),
    "olmo_core.nn.moe.v2.block.MoEFusedV2TransformerBlock": (
        "olmo_core.nn.ddp.block.OLMoDDPTransformerBlock"
    ),
    "olmo_core.nn.moe.v2.block.OLMoDDPTransformerBlockConfig": (
        "olmo_core.nn.ddp.block.OLMoDDPTransformerBlockConfig"
    ),
    "olmo_core.nn.moe.v2.block.OLMoDDPTransformerBlock": (
        "olmo_core.nn.ddp.block.OLMoDDPTransformerBlock"
    ),
    # Model class (renamed symbol and/or relocated module path).
    "olmo_core.nn.moe.v2.model.MoEFusedV2Transformer": "olmo_core.nn.ddp.model.OLMoDDPModel",
    "olmo_core.nn.moe.v2.model.OLMoDDPModel": "olmo_core.nn.ddp.model.OLMoDDPModel",
    # Optimizer config/class (same module, renamed symbol).
    "olmo_core.optim.moe_optimizer.MoEFusedV2OptimizerConfig": (
        "olmo_core.optim.moe_optimizer.OLMoDDPOptimizerConfig"
    ),
    "olmo_core.optim.moe_optimizer.MoEFusedV2Optimizer": (
        "olmo_core.optim.moe_optimizer.OLMoDDPOptimizer"
    ),
    # Train-module config (same module, renamed symbol).
    "olmo_core.train.train_module.transformer.config.MoEV2TransformerTrainModuleConfig": (
        "olmo_core.train.train_module.transformer.config.OLMoDDPTrainModuleConfig"
    ),
    # Train-module class (renamed symbol and relocated module path).
    "olmo_core.train.train_module.transformer.moe_train_module.MoEV2TransformerTrainModule": (
        "olmo_core.train.train_module.transformer.ddp_train_module.OLMoDDPTrainModule"
    ),
}


def rewrite_config_dict(data: Any, _path: str = "") -> Tuple[Any, List[Tuple[str, str, str]]]:
    """
    Recursively rewrite legacy ``_CLASS_`` values in a config dictionary.

    :param data: The parsed config (a ``dict``, ``list``, or scalar).

    :returns: A ``(new_data, changes)`` pair, where ``changes`` is a list of
        ``(json_path, old_class, new_class)`` tuples describing each rewrite.
    """
    changes: List[Tuple[str, str, str]] = []
    if isinstance(data, dict):
        new: Dict[str, Any] = {}
        for key, value in data.items():
            child_path = f"{_path}.{key}" if _path else key
            if (
                key == Config.CLASS_NAME_FIELD
                and isinstance(value, str)
                and value in CLASS_PATH_REWRITES
            ):
                new_value = CLASS_PATH_REWRITES[value]
                changes.append((child_path, value, new_value))
                new[key] = new_value
            else:
                new[key], child_changes = rewrite_config_dict(value, child_path)
                changes.extend(child_changes)
        return new, changes
    if isinstance(data, list):
        new_list: List[Any] = []
        for idx, item in enumerate(data):
            new_item, child_changes = rewrite_config_dict(item, f"{_path}[{idx}]")
            new_list.append(new_item)
            changes.extend(child_changes)
        return new_list, changes
    return data, changes


def _resolve_config_path(path: str) -> str:
    """Resolve a checkpoint dir or config file to the ``config.json`` path to rewrite."""
    path = normalize_path(path)
    if path.endswith(".json"):
        return path
    candidate = str(join_path(path, CONFIG_FILENAME))
    if not file_exists(candidate):
        raise FileNotFoundError(
            f"No '{CONFIG_FILENAME}' found under '{path}'. Pass a checkpoint directory that "
            f"contains one, or the path to the config file directly."
        )
    return candidate


def _read_json(path: str) -> Dict[str, Any]:
    raw = get_bytes_range(path, 0, get_file_size(path))
    return json.loads(raw.decode("utf-8"))


def _write_json(data: Dict[str, Any], target: str, *, save_overwrite: bool) -> None:
    serialized = json.dumps(data, indent=2)
    if is_url(target):
        with TemporaryDirectory() as tmp_dir:
            local = join_path(tmp_dir, CONFIG_FILENAME)
            with open(local, "w") as f:
                f.write(serialized)
            upload(local, target, save_overwrite=save_overwrite)
    else:
        if file_exists(target) and not save_overwrite:
            raise FileExistsError(
                f"'{target}' already exists. Pass --save-overwrite to overwrite it."
            )
        with open(target, "w") as f:
            f.write(serialized)


def convert_checkpoint_config(
    checkpoint: str,
    *,
    output: str | None = None,
    dry_run: bool = False,
    save_overwrite: bool = True,
) -> int:
    """
    Migrate a single checkpoint's ``config.json``.

    :param checkpoint: A checkpoint directory or ``config.json`` path (local or remote).
    :param output: Where to write the migrated config. Defaults to rewriting in place.
    :param dry_run: If ``True``, log the changes but don't write anything.
    :param save_overwrite: Whether to overwrite the target if it already exists.

    :returns: The number of ``_CLASS_`` values rewritten.
    """
    source = _resolve_config_path(checkpoint)
    log.info(f"Reading config from '{source}'")
    config = _read_json(source)

    new_config, changes = rewrite_config_dict(config)
    if not changes:
        log.info("No legacy MoE-v2 class paths found; nothing to migrate.")
        return 0

    log.info(f"Found {len(changes)} legacy class path(s) to rewrite:")
    for json_path, old, new in changes:
        log.info(f"  {json_path}: {old} -> {new}")

    target = output or source
    if dry_run:
        log.info(f"[dry-run] Would write migrated config to '{target}'")
        return len(changes)

    _write_json(new_config, target, save_overwrite=save_overwrite)
    log.info(f"Wrote migrated config to '{target}'")
    return len(changes)


@click.command()
@click.argument("checkpoints", nargs=-1, required=True, type=str)
@click.option(
    "--output",
    type=str,
    default=None,
    help=(
        "Write the migrated config here instead of in place. Only valid with a single checkpoint."
    ),
)
@click.option("--dry-run", is_flag=True, help="Log the changes without writing anything.")
@click.option(
    "--save-overwrite/--no-save-overwrite",
    default=True,
    help="Whether to overwrite the target config if it already exists (default: overwrite).",
)
def main(checkpoints: Tuple[str, ...], output: str | None, dry_run: bool, save_overwrite: bool):
    """
    Migrate CHECKPOINTS (checkpoint dirs or config.json paths) off the legacy MoE-v2 class paths.
    """
    if output is not None and len(checkpoints) != 1:
        raise click.UsageError("--output can only be used with a single checkpoint.")

    total = 0
    for checkpoint in checkpoints:
        total += convert_checkpoint_config(
            checkpoint,
            output=output,
            dry_run=dry_run,
            save_overwrite=save_overwrite,
        )
    log.info(
        f"Done. Rewrote {total} class path(s) across {len(checkpoints)} checkpoint(s)"
        + (" (dry run)." if dry_run else ".")
    )


if __name__ == "__main__":
    prepare_cli_environment()
    main()
