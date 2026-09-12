"""
Migrate a checkpoint's ``config.json`` off the legacy ``MoEFusedV2*`` / ``MoEV2*`` /
``olmo_core.nn.moe.v2.*`` class paths onto the canonical ``OLMoDDP*`` (``olmo_core.nn.ddp.*``) ones.

The fused MoE-v2 stack was promoted to the ``OLMoDDP*`` names, and the old names/module paths (kept
for a while as aliases and re-export shims) have been removed. Model and optimizer *weights* are
unaffected — the rename preserved the same class objects, so parameter names are identical — but a
``config.json`` serialized under the old names records stale ``_CLASS_`` values that no longer
resolve. This script rewrites just those ``_CLASS_`` strings in place so the checkpoint loads again.

Run it by file path from the repo root::

    # Rewrite one or more checkpoints (a checkpoint dir or a config.json, local or remote).
    python src/scripts/convert_moe_v2_checkpoint_config.py /path/to/checkpoint/step10000
    python src/scripts/convert_moe_v2_checkpoint_config.py s3://bucket/run/step10000/config.json

    # Preview the changes without writing.
    python src/scripts/convert_moe_v2_checkpoint_config.py --dry-run /path/to/checkpoint/step10000

    # Write the migrated config to a different location instead of in place.
    python src/scripts/convert_moe_v2_checkpoint_config.py \
        --output /tmp/config.json /path/to/checkpoint/step10000
"""

import json
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple

import click

# Ensure ``olmo_core`` is importable when this file is run directly from a source checkout: Python
# puts this script's own directory (``src/scripts``) on ``sys.path`` rather than ``src``. This is a
# no-op when the package is installed (``pip install -e .``).
_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from olmo_core.config import Config  # noqa: E402
from olmo_core.io import (  # noqa: E402
    file_exists,
    get_bytes_range,
    get_file_size,
    is_url,
    join_path,
    normalize_path,
    upload,
)
from olmo_core.utils import prepare_cli_environment  # noqa: E402

log = logging.getLogger(__name__)

CONFIG_FILENAME = "config.json"

#: Maps a class's leaf name (the part after the last ``.``) to its canonical fully-qualified
#: ``_CLASS_`` path. Keying on the leaf name — rather than the full module path — makes the rewrite
#: robust to *every* way an old name could have been recorded: the renamed-symbol aliases
#: (``MoEFusedV2*`` / ``MoEV2*``), the former ``olmo_core.nn.moe.v2.*`` / ``moe_train_module`` module
#: paths, and the package-level re-export paths (e.g. ``olmo_core.optim.MoEFusedV2OptimizerConfig``).
#: The canonical names are included too, so a config recorded under a now-removed module path but the
#: canonical name is normalized to the canonical module. These leaf names are unique in the codebase.
_CANONICAL_BY_LEAF_NAME: Dict[str, str] = {
    # Model config.
    "MoEFusedV2TransformerConfig": "olmo_core.nn.transformer.config.OLMoDDPModelConfig",
    "OLMoDDPModelConfig": "olmo_core.nn.transformer.config.OLMoDDPModelConfig",
    # Block config/class.
    "MoEFusedV2TransformerBlockConfig": "olmo_core.nn.ddp.block.OLMoDDPTransformerBlockConfig",
    "OLMoDDPTransformerBlockConfig": "olmo_core.nn.ddp.block.OLMoDDPTransformerBlockConfig",
    "MoEFusedV2TransformerBlock": "olmo_core.nn.ddp.block.OLMoDDPTransformerBlock",
    "OLMoDDPTransformerBlock": "olmo_core.nn.ddp.block.OLMoDDPTransformerBlock",
    # Model class.
    "MoEFusedV2Transformer": "olmo_core.nn.ddp.model.OLMoDDPModel",
    "OLMoDDPModel": "olmo_core.nn.ddp.model.OLMoDDPModel",
    # Optimizer config/class.
    "MoEFusedV2OptimizerConfig": "olmo_core.optim.moe_optimizer.OLMoDDPOptimizerConfig",
    "OLMoDDPOptimizerConfig": "olmo_core.optim.moe_optimizer.OLMoDDPOptimizerConfig",
    "MoEFusedV2Optimizer": "olmo_core.optim.moe_optimizer.OLMoDDPOptimizer",
    "OLMoDDPOptimizer": "olmo_core.optim.moe_optimizer.OLMoDDPOptimizer",
    # Train-module config/class.
    "MoEV2TransformerTrainModuleConfig": (
        "olmo_core.train.train_module.transformer.config.OLMoDDPTrainModuleConfig"
    ),
    "OLMoDDPTrainModuleConfig": (
        "olmo_core.train.train_module.transformer.config.OLMoDDPTrainModuleConfig"
    ),
    "MoEV2TransformerTrainModule": (
        "olmo_core.train.train_module.transformer.ddp_train_module.OLMoDDPTrainModule"
    ),
    "OLMoDDPTrainModule": (
        "olmo_core.train.train_module.transformer.ddp_train_module.OLMoDDPTrainModule"
    ),
    # MoE sub-configs that the deleted moe.v2.block shim re-exported. Their names are unchanged;
    # only the module path needs normalizing off the removed shim to the defining module.
    "MoERouterConfigV2": "olmo_core.nn.moe.v2.router.MoERouterConfigV2",
    "RoutedExpertsConfig": "olmo_core.nn.moe.v2.routed_experts.RoutedExpertsConfig",
    "SharedExpertsConfig": "olmo_core.nn.moe.v2.shared_experts.SharedExpertsConfig",
}


def _canonical_class_path(value: str) -> str | None:
    """
    Return the canonical ``_CLASS_`` path for a legacy/relocated one, or ``None`` to leave it as-is.

    Matches on the leaf class name so any module path an old alias could have been recorded under —
    concrete module, former ``moe.v2.*`` shim, or package-level re-export — maps to the canonical
    fully-qualified path.
    """
    leaf = value.rpartition(".")[2]
    return _CANONICAL_BY_LEAF_NAME.get(leaf)


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
            canonical = (
                _canonical_class_path(value)
                if key == Config.CLASS_NAME_FIELD and isinstance(value, str)
                else None
            )
            if canonical is not None and canonical != value:
                changes.append((child_path, value, canonical))
                new[key] = canonical
            elif canonical is not None:
                new[key] = value
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
