#!/usr/bin/env python3
"""
Convert an OLMo-core checkpoint from the "hacky" GDN branch (tyler/anejs/linear-rnns)
to the current main branch parameter naming.

On the hacky branch, GDN layers used an ``FLABlock`` that stored a wrapper ``FLA`` module
in ``self.fla``, which in turn stored the actual ``GatedDeltaNet`` in ``self.inner``.
The GatedDeltaNet implementation also used different parameter names (``q_proj`` vs ``w_q``).

Hacky branch parameter paths (GDN layers)::

    blocks.{i}.fla.inner.q_proj.weight     ->  blocks.{i}.attention.w_q.weight
    blocks.{i}.fla.inner.k_proj.weight     ->  blocks.{i}.attention.w_k.weight
    blocks.{i}.fla.inner.v_proj.weight     ->  blocks.{i}.attention.w_v.weight
    blocks.{i}.fla.inner.a_proj.weight     ->  blocks.{i}.attention.w_a.weight
    blocks.{i}.fla.inner.b_proj.weight     ->  blocks.{i}.attention.w_b.weight
    blocks.{i}.fla.inner.g_proj.weight     ->  blocks.{i}.attention.w_g.weight
    blocks.{i}.fla.inner.o_proj.weight     ->  blocks.{i}.attention.w_out.weight
    blocks.{i}.fla.inner.A_log            ->  blocks.{i}.attention.A_log
    blocks.{i}.fla.inner.dt_bias          ->  blocks.{i}.attention.dt_bias
    blocks.{i}.fla.inner.q_conv1d.weight  ->  blocks.{i}.attention.q_conv1d.weight
    blocks.{i}.fla.inner.k_conv1d.weight  ->  blocks.{i}.attention.k_conv1d.weight
    blocks.{i}.fla.inner.v_conv1d.weight  ->  blocks.{i}.attention.v_conv1d.weight
    blocks.{i}.fla.inner.o_norm.weight    ->  blocks.{i}.attention.o_norm.weight
    blocks.{i}.fla_norm.weight            ->  blocks.{i}.attention_norm.weight

Attention layers are left untouched (they already use ``blocks.{i}.attention.*``
and ``blocks.{i}.attention_norm.*``).

Usage::

    uv run python src/scripts/convert_hacky_gdn_checkpoint.py \\
        --input /path/to/old/step1000 \\
        --output /path/to/new/step1000

The script expects the standard OLMo-core checkpoint layout::

    step1000/
    ├── model_and_optim/
    │   ├── .metadata
    │   └── ...
    ├── train/
    │   └── rank0.pt
    └── .metadata.json

It copies everything except ``model_and_optim/`` verbatim, then writes a corrected
``model_and_optim/`` directory.
"""

import json
import logging
import os
import re
import shutil
from pathlib import Path

import click

from olmo_core.distributed.checkpoint import save_state_dict
from olmo_core.distributed.checkpoint.filesystem import RemoteFileSystemReader
from olmo_core.io import file_exists, normalize_path
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Key renaming rules
# ---------------------------------------------------------------------------

# Regex for GDN parameters nested under ``fla.inner.``.
# Group 1: prefix up to and including the block index  (e.g. "model.blocks.5")
# Group 2: the parameter suffix after ``fla.inner.``   (e.g. "q_proj.weight")
# Uses ``.*`` prefix to also match optimizer state keys like
# ``optim.state.blocks.0.fla.inner.q_proj.weight.exp_avg``.
_FLA_INNER_RE = re.compile(r"^(.*blocks\.\d+)\.fla\.inner\.(.+)$")

# Regex for the FLA layer norm (``fla_norm`` -> ``attention_norm``).
_FLA_NORM_RE = re.compile(r"^(.*blocks\.\d+)\.fla_norm\.(.+)$")

# Mapping from hacky-branch linear projection names to main-branch names.
_PROJ_RENAME = {
    "q_proj": "w_q",
    "k_proj": "w_k",
    "v_proj": "w_v",
    "a_proj": "w_a",
    "b_proj": "w_b",
    "g_proj": "w_g",
    "o_proj": "w_out",
}


def _rename_proj(suffix: str) -> str:
    """Rename the leading projection name in *suffix* if it matches."""
    for old, new in _PROJ_RENAME.items():
        if suffix == old or suffix.startswith(old + "."):
            return new + suffix[len(old) :]
    return suffix


def _resolve_model_and_optim_dir(checkpoint_dir: str) -> str:
    """Return the ``model_and_optim`` sub-directory of *checkpoint_dir*."""
    checkpoint_dir = normalize_path(checkpoint_dir)

    metadata_path = checkpoint_dir.rstrip("/") + "/.metadata"
    if file_exists(metadata_path):
        return checkpoint_dir

    sub = checkpoint_dir.rstrip("/") + "/model_and_optim"
    if file_exists(sub + "/.metadata"):
        return sub

    return checkpoint_dir


def _load_full_state_dict(dir: str) -> dict:
    """Load all keys from a (possibly sharded) checkpoint into a flat dict on CPU."""
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    state_dict: dict = {}
    reader = RemoteFileSystemReader(dir)
    _load_state_dict(
        state_dict,
        storage_reader=reader,
        planner=_EmptyStateDictLoadPlanner(
            keys=list(reader.read_metadata().state_dict_metadata.keys())
        ),
        no_dist=True,
    )
    return state_dict


def rename_keys(state_dict: dict) -> dict:
    """Rename checkpoint keys from the hacky GDN layout to the main layout.

    Two transformations are applied:

    1. ``blocks.{i}.fla.inner.<param>`` -> ``blocks.{i}.attention.<param>``
       with projection renames (``q_proj`` -> ``w_q``, etc.).
    2. ``blocks.{i}.fla_norm.<param>`` -> ``blocks.{i}.attention_norm.<param>``

    Keys that don't match either pattern are copied through unchanged.
    If a value is itself a dict (nested checkpoint format), the function
    recurses into it.
    """
    new_sd: dict = {}
    renamed = 0

    for key, value in state_dict.items():
        # Recurse into nested dicts (e.g. checkpoint with "model" and "optim" sub-dicts).
        if isinstance(value, dict):
            new_sd[key] = rename_keys(value)
            continue

        # 1. GDN parameters: fla.inner.* -> attention.*
        m = _FLA_INNER_RE.match(key)
        if m:
            prefix, suffix = m.group(1), m.group(2)
            suffix = _rename_proj(suffix)
            new_key = f"{prefix}.attention.{suffix}"
            log.info("  %s  ->  %s", key, new_key)
            new_sd[new_key] = value
            renamed += 1
            continue

        # 2. FLA layer norm: fla_norm.* -> attention_norm.*
        m = _FLA_NORM_RE.match(key)
        if m:
            prefix, suffix = m.group(1), m.group(2)
            new_key = f"{prefix}.attention_norm.{suffix}"
            log.info("  %s  ->  %s", key, new_key)
            new_sd[new_key] = value
            renamed += 1
            continue

        # No match - keep as-is.
        new_sd[key] = value

    log.info("Renamed %d keys total", renamed)
    return new_sd


def convert_config(config: dict) -> dict:
    """Convert a hacky-branch ``config.json`` to the main-branch format.

    The old config stores the block as a single :class:`TransformerBlockConfig`
    with ``name="fla_hybrid"``, an ``fla`` field (:class:`FLAConfig`), and
    ``fla_hybrid_attention_indices`` listing which layers use attention.

    The new config uses a dict of named blocks (``"gdn"`` and ``"attn"``) with
    a ``block_pattern`` that cycles over layers.
    """
    config = json.loads(json.dumps(config))  # deep copy
    model = config.get("model")
    if model is None:
        log.warning("No 'model' key in config.json, skipping config conversion.")
        return config

    old_block = model.get("block", {})

    # Only convert if this is actually the old fla_hybrid format.
    if old_block.get("name") != "fla_hybrid" or "fla" not in old_block:
        log.info("Config does not use old fla_hybrid format, keeping as-is.")
        return config

    n_layers = model["n_layers"]

    # ---------------------------------------------------------------
    # Build the GDN sequence mixer config from the old ``fla`` field.
    # ---------------------------------------------------------------
    old_fla = old_block["fla"]
    old_fla_kwargs = old_fla.get("fla_layer_kwargs", {})

    from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig

    gdn_mixer = GatedDeltaNetConfig(
        n_heads=old_block["attention"]["n_heads"],
        head_dim=old_fla_kwargs.get("head_dim"),
        allow_neg_eigval=old_fla_kwargs.get("allow_neg_eigval", True),
        dtype=old_fla.get("dtype", "float32"),
    ).as_config_dict()

    # ---------------------------------------------------------------
    # Build the attention sequence mixer from the old ``attention`` field.
    # ---------------------------------------------------------------
    attn_mixer = old_block["attention"]  # already in the correct format

    # ---------------------------------------------------------------
    # Shared block fields (layer_norm, feed_forward).
    # ---------------------------------------------------------------
    shared_fields = {}
    for field_name in ("layer_norm", "feed_forward", "feed_forward_moe", "dropout"):
        if field_name in old_block:
            shared_fields[field_name] = old_block[field_name]

    # ---------------------------------------------------------------
    # Build the two named block configs.
    # ---------------------------------------------------------------
    from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerBlockType

    # GDN block uses the default (pre-norm) block type.
    gdn_block = {
        "sequence_mixer": gdn_mixer,
        **shared_fields,
        "name": TransformerBlockType.default.value,
        "_CLASS_": "olmo_core.nn.transformer.config.TransformerBlockConfig",
    }

    # Attention block: determine block type from the original.
    # The old "fla_hybrid" type means the attention layers used reordered_norm.
    attn_block_type = TransformerBlockType.reordered_norm.value
    attn_block = {
        "sequence_mixer": attn_mixer,
        **shared_fields,
        "name": attn_block_type,
        "_CLASS_": "olmo_core.nn.transformer.config.TransformerBlockConfig",
    }

    # ---------------------------------------------------------------
    # Derive block_pattern from fla_hybrid_attention_indices.
    # ---------------------------------------------------------------
    attn_indices = set(old_block.get("fla_hybrid_attention_indices", []))
    raw_pattern = ["attn" if i in attn_indices else "gdn" for i in range(n_layers)]

    # Try to find the shortest repeating cycle.
    block_pattern = raw_pattern
    for cycle_len in range(1, n_layers + 1):
        if n_layers % cycle_len != 0:
            continue
        candidate = raw_pattern[:cycle_len]
        if candidate * (n_layers // cycle_len) == raw_pattern:
            block_pattern = candidate
            break

    # ---------------------------------------------------------------
    # Replace the model config fields.
    # ---------------------------------------------------------------
    model["block"] = {"gdn": gdn_block, "attn": attn_block}
    model["block_pattern"] = block_pattern

    # Remove stale fields that don't exist in the new config.
    model.pop("block_overrides", None)

    log.info(
        "Converted config: block_pattern=%s (cycle length %d for %d layers)",
        block_pattern,
        len(block_pattern),
        n_layers,
    )
    return config


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    help="Path to the source checkpoint directory (e.g. /checkpoints/step1000).",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    help="Path to write the converted checkpoint.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite the output directory if it already exists.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the key renames without writing anything.",
)
def main(input_path: str, output_path: str, overwrite: bool, dry_run: bool) -> None:
    """Convert a hacky-GDN checkpoint to the main GDN parameter naming."""
    input_path = normalize_path(input_path)
    output_path = normalize_path(output_path)

    model_and_optim_dir = _resolve_model_and_optim_dir(input_path)
    log.info("Loading checkpoint from: %s", model_and_optim_dir)

    # ------------------------------------------------------------------
    # 1. Load the full (unsharded) state dict.
    # ------------------------------------------------------------------
    state_dict = _load_full_state_dict(model_and_optim_dir)
    log.info("Loaded %d keys", len(state_dict))

    # ------------------------------------------------------------------
    # 2. Rename keys.
    # ------------------------------------------------------------------
    state_dict = rename_keys(state_dict)

    if dry_run:
        log.info("Dry run -- not writing output.")
        return

    # ------------------------------------------------------------------
    # 3. Copy non-model_and_optim files (train state, metadata, config, etc.).
    # ------------------------------------------------------------------
    output_dir = Path(output_path)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory '{output_dir}' already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    # Copy everything except model_and_optim/
    input_dir = Path(input_path)
    if input_dir.is_dir():

        def _ignore_model_and_optim(directory: str, contents: list[str]) -> list[str]:
            if os.path.basename(directory) == os.path.basename(str(input_dir)):
                return [c for c in contents if c == "model_and_optim"]
            return []

        shutil.copytree(str(input_dir), str(output_dir), ignore=_ignore_model_and_optim)
        log.info("Copied auxiliary checkpoint files to %s", output_dir)

    # ------------------------------------------------------------------
    # 4. Convert config.json to the new format.
    # ------------------------------------------------------------------
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            old_config = json.load(f)
        new_config = convert_config(old_config)
        with open(config_path, "w") as f:
            json.dump(new_config, f, indent=2)
            f.write("\n")
        log.info("Converted config.json")

    # ------------------------------------------------------------------
    # 5. Write the corrected model_and_optim checkpoint.
    # ------------------------------------------------------------------
    new_model_and_optim_dir = str(output_dir / "model_and_optim")
    log.info("Saving corrected checkpoint to: %s", new_model_and_optim_dir)
    save_state_dict(
        new_model_and_optim_dir,
        state_dict,
        save_overwrite=True,
    )
    log.info("Done! Converted checkpoint written to: %s", output_path)


if __name__ == "__main__":
    prepare_cli_environment()
    main()
