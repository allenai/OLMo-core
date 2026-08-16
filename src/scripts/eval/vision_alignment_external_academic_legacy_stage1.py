"""Evaluate the canonical legacy Stage-1 endpoint on the frozen academic panel.

This is a checkpoint-policy wrapper around :mod:`vision_alignment_external_academic`.
It intentionally reuses that module's immutable manifest loader, prompts, native inference,
task scorers, row schema, and aggregate rederivation without changing the frozen evaluator.
The only new behavior is strict admission and provenance for the completed pre-alignment
``s002`` Stage-1 checkpoint.

The resulting receipt is a descriptive historical-reference artifact.  In particular, the
manifest's image-overlap inventory describes the later vision-alignment training union, not
the legacy Stage-1 training mixture.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import stat
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import vision_alignment_external_academic as academic
from torch.distributed.checkpoint import FileSystemReader

from olmo_core.data.multimodal.vision_alignment_sources import (
    load_pinned_vision_alignment_tokenizer,
)
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
from olmo_core.train import prepare_training_environment, teardown_training_environment

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
RECEIPT_FORMAT = "vision_alignment_external_academic_legacy_stage1_receipt"
PROTOCOL_NAME = "vision-alignment-external-academic-legacy-stage1-ep8-v1"

EXPECTED_MANIFEST = Path(
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/"
    "joint-v1-external-academic-v1/manifest-6ff70cf8.json"
)
EXPECTED_MANIFEST_SHA256 = "e9f875766c6080a8c5451e827fdfdf7e42eda95ee5377b12c8d8885f13733dac"
EXPECTED_MANIFEST_CONTENT_SHA256 = (
    "a3f8a2ff7f06746ac82647ddd90337e549e5b68e7f432f98f2284f5e2f431255"
)
EXPECTED_MANIFEST_GIT = {
    "revision": "6ff70cf8f79f6841153d6e3cb4479096a234d467",
    "dirty": False,
}

EXPECTED_CHECKPOINT = Path(
    "/weka/oe-training-default/rustin/experiments/vision-moe/checkpoints/"
    "s002-stage1-corrected-clean-32k-b300-20260807/step32000"
)
EXPECTED_CONFIG_SHA256 = "15b655200cb4a9b01c778139e4fe0e4b73a6b4b959f29b0afefe78625ee2e78c"
EXPECTED_CHECKPOINT_MARKER_SHA256 = (
    "77dfdeec42fe7990f4b3b9c4eeecd480edcf5066c110603b115920af38423d03"
)
EXPECTED_DCP_METADATA_SHA256 = "aa8e4979a3cde21736c2c635c52bf03291fdfd051d2cf012de844a9cf40f7624"
EXPECTED_STATE_FILE_COUNT = 257
EXPECTED_STATE_BYTES = 387_025_521_198
EXPECTED_STATE_FILE_INVENTORY_SHA256 = (
    "e558282e2612262555005d85a109904853e40458dd4650f5a74ec6b8dbd66f87"
)
EXPECTED_DISTCP_SHARD_COUNT = 256
EXPECTED_DISTCP_SHARD_BYTES = 387_021_304_242
EXPECTED_ROOT_FILE_COUNT = 275
EXPECTED_ROOT_BYTES = 387_025_813_619
EXPECTED_ROOT_FILE_INVENTORY_SHA256 = (
    "a31372199edc243871ffdb216cc9db10f9d20dfc7acc59fdf05df11e1929b679"
)
EXPECTED_TRAIN_STATE_COUNT = 16
EXPECTED_TRAIN_STATE_BYTES = 264_320
EXPECTED_TRAIN_STATE_INVENTORY_SHA256 = (
    "6876401107e548fb8cf84a20d9f39a7bdaba80e65d8c3e5ef4ca3510fc49d961"
)
EXPECTED_TRAIN_STATE_PROJECTIONS_SHA256 = (
    "7845cf4ab4c6cb9241adf52f6111edc5c2e1948532aa7aacaeea0884829b9d5c"
)

EXPECTED_DCP_KEY_COUNT = 3_274
EXPECTED_DCP_KEYS_SHA256 = "bf034000ed35312a9efe64d1932585f645f60408c3fa7a499d5d167357b49afa"
EXPECTED_MODEL_KEY_COUNT = 818
EXPECTED_MODEL_KEYS_SHA256 = "31c5f41638dc9e789a314d8499718e2e484270676637c5c91ff8cfa3f850b43a"
EXPECTED_OPTIMIZER_KEY_COUNT = 2_454
EXPECTED_AUXILIARY_DCP_KEYS = ["__moe_skip_step_grad_norms", "__moe_skip_step_losses"]

EXPECTED_TRAINING_GIT_REVISION = "586b25bb0d10263877c7ff39d315e0d359aa0806"
EXPECTED_RUN_NAME = "s002-stage1-corrected-clean-32k-b300-20260807"
EXPECTED_WANDB_RUN_ID = "bkw8jsyr"
EXPECTED_GLOBAL_STEP = 32_000
EXPECTED_GLOBAL_TRAIN_TOKENS = 10_485_760_000
EXPECTED_TRAIN_WORLD_SIZE = 16

TOKENIZER_IDENTIFIER = "allenai/dolma2-tokenizer"
TOKENIZER_REVISION = "5292e5d6c0f40b67cc765fe41bec991cf4345b5c"
TOKENIZER_FINGERPRINT = "8fec2af8c372f4c72a1a665ad8e70517625f94f041dbfcb7db4932071380f9a7"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _checkpoint_identity_scope() -> dict[str, str]:
    return {
        "distcp_shards": (
            "relative path and byte size only; individual DCP shard contents are not hashed"
        ),
        "dcp_metadata": "full-file SHA-256",
        "config_and_marker": "full-file SHA-256",
        "trainer_states": "full-file SHA-256 for every rank",
        "stability": "opening and closing identities must be exactly equal",
    }


def _expected_trainer_state_summary() -> dict[str, Any]:
    return {
        "rank_count": EXPECTED_TRAIN_STATE_COUNT,
        "ranks": list(range(EXPECTED_TRAIN_STATE_COUNT)),
        "global_step": EXPECTED_GLOBAL_STEP,
        "max_steps": EXPECTED_GLOBAL_STEP,
        "global_train_tokens_seen": EXPECTED_GLOBAL_TRAIN_TOKENS,
        "world_size": EXPECTED_TRAIN_WORLD_SIZE,
        "data_parallel_world_size": EXPECTED_TRAIN_WORLD_SIZE,
        "batches_processed": EXPECTED_GLOBAL_STEP,
        "wandb_run_id": EXPECTED_WANDB_RUN_ID,
        "wandb_step": EXPECTED_GLOBAL_STEP,
        "wandb_name": EXPECTED_RUN_NAME,
        "wandb_project": "molmo2-stage1",
        "rank_projections_sha256": EXPECTED_TRAIN_STATE_PROJECTIONS_SHA256,
    }


def _expected_dcp_key_projection() -> dict[str, Any]:
    return {
        "dcp_state_key_count": EXPECTED_DCP_KEY_COUNT,
        "dcp_state_keys_sha256": EXPECTED_DCP_KEYS_SHA256,
        "model_tensor_key_count": EXPECTED_MODEL_KEY_COUNT,
        "model_tensor_keys_sha256": EXPECTED_MODEL_KEYS_SHA256,
        "optimizer_state_key_count": EXPECTED_OPTIMIZER_KEY_COUNT,
        "auxiliary_keys": EXPECTED_AUXILIARY_DCP_KEYS,
    }


def _implementation_identity() -> dict[str, Any]:
    """Identify both the frozen evaluator and this checkpoint-policy wrapper."""
    repo_root = Path(__file__).resolve().parents[3]
    files = {
        "wrapper": academic._repo_relative_file_identity(Path(__file__), repo_root),
    }
    return {
        "frozen_evaluator": academic._implementation_identity(),
        "wrapper_files": files,
        "wrapper_files_sha256": academic._canonical_sha256(files),
    }


def _relative_file_inventory(root: Path) -> list[dict[str, Any]]:
    """Inventory every checkpoint file by relative path and byte size without shard hashing."""
    root = root.expanduser().resolve()
    if root != EXPECTED_CHECKPOINT.resolve() or not root.is_dir():
        raise ValueError("Legacy Stage-1 root is not the exact canonical checkpoint")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"Legacy Stage-1 checkpoint contains a symlink: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise ValueError(f"Legacy Stage-1 checkpoint contains an invalid file: {path}")
        rows.append({"name": path.relative_to(root).as_posix(), "bytes": metadata.st_size})
    return rows


def _trainer_state_inventory(checkpoint: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Hash and project all 16 trusted trainer-state files after identity admission."""
    train_dir = checkpoint / "train"
    paths = sorted(
        train_dir.glob("rank*.pt"),
        key=lambda path: int(path.stem.removeprefix("rank")),
    )
    rows = []
    for path in paths:
        rank_match = re.fullmatch(r"rank([0-9]+)\.pt", path.name)
        if rank_match is None:
            raise ValueError(f"Unexpected trainer-state filename {path.name!r}")
        identity = academic._file_identity(path)
        rows.append(
            {
                "name": path.relative_to(checkpoint).as_posix(),
                "bytes": identity["bytes"],
                "sha256": identity["sha256"],
            }
        )
    if (
        len(rows) != EXPECTED_TRAIN_STATE_COUNT
        or sum(row["bytes"] for row in rows) != EXPECTED_TRAIN_STATE_BYTES
        or academic._canonical_sha256(rows) != EXPECTED_TRAIN_STATE_INVENTORY_SHA256
    ):
        raise ValueError("Legacy Stage-1 trainer-state file identities differ")

    projections = []
    for rank, path in enumerate(paths):
        # These are fixed, fully hashed files from the user's canonical checkpoint.  The
        # full-file admission above happens before allowing the trusted PyTorch pickle load.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise TypeError(f"Trainer state rank {rank} is not a mapping")
        data_loader = state.get("data_loader")
        callbacks = state.get("callbacks")
        wandb = callbacks.get("wandb") if isinstance(callbacks, dict) else None
        if not isinstance(data_loader, dict) or not isinstance(wandb, dict):
            raise ValueError(f"Trainer state rank {rank} lacks data-loader or W&B provenance")
        projections.append(
            {
                "rank": rank,
                "global_step": state.get("global_step"),
                "max_steps": state.get("max_steps"),
                "global_train_tokens_seen": state.get("global_train_tokens_seen"),
                "world_size": state.get("world_size"),
                "data_parallel_rank": data_loader.get("packing_state", {}).get("dp_rank"),
                "data_parallel_world_size": data_loader.get("packing_state", {}).get(
                    "dp_world_size"
                ),
                "batches_processed": data_loader.get("batches_processed"),
                "wandb_run_id": wandb.get("run_id"),
                "wandb_step": wandb.get("step"),
                "wandb_name": wandb.get("name"),
                "wandb_project": wandb.get("project"),
            }
        )
    expected_projection = {
        "global_step": EXPECTED_GLOBAL_STEP,
        "max_steps": EXPECTED_GLOBAL_STEP,
        "global_train_tokens_seen": EXPECTED_GLOBAL_TRAIN_TOKENS,
        "world_size": EXPECTED_TRAIN_WORLD_SIZE,
        "data_parallel_world_size": EXPECTED_TRAIN_WORLD_SIZE,
        "batches_processed": EXPECTED_GLOBAL_STEP,
        "wandb_run_id": EXPECTED_WANDB_RUN_ID,
        "wandb_step": EXPECTED_GLOBAL_STEP,
        "wandb_name": EXPECTED_RUN_NAME,
        "wandb_project": "molmo2-stage1",
    }
    for rank, projection in enumerate(projections):
        if projection["rank"] != rank or projection["data_parallel_rank"] != rank:
            raise ValueError("Legacy Stage-1 trainer-state rank coverage differs")
        if any(projection[field] != value for field, value in expected_projection.items()):
            raise ValueError(f"Legacy Stage-1 trainer-state rank {rank} progress differs")
    summary = {
        "rank_count": len(projections),
        "ranks": [projection["rank"] for projection in projections],
        **expected_projection,
        "rank_projections_sha256": academic._canonical_sha256(projections),
    }
    if summary != _expected_trainer_state_summary():
        raise ValueError("Legacy Stage-1 trainer-state summary differs")
    return rows, summary


def _dcp_key_projection(state_dir: Path) -> dict[str, Any]:
    """Read the fully hashed DCP metadata and project its model-load key coverage."""
    metadata = FileSystemReader(state_dir).read_metadata()
    keys = sorted(metadata.state_dict_metadata)
    model_keys = [key for key in keys if key.endswith(".main")]
    optimizer_keys = [key for key in keys if key.endswith((".exp_avg", ".exp_avg_sq", ".step"))]
    auxiliary_keys = [
        key for key in keys if key not in set(model_keys) and key not in set(optimizer_keys)
    ]
    projection = {
        "dcp_state_key_count": len(keys),
        "dcp_state_keys_sha256": academic._canonical_sha256(keys),
        "model_tensor_key_count": len(model_keys),
        "model_tensor_keys_sha256": academic._canonical_sha256(model_keys),
        "optimizer_state_key_count": len(optimizer_keys),
        "auxiliary_keys": auxiliary_keys,
    }
    if projection != _expected_dcp_key_projection():
        raise ValueError("Legacy Stage-1 DCP state-key projection differs")
    return projection


def _checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    """Bind the exact legacy checkpoint without implying byte hashes for 387 GB of shards."""
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint != EXPECTED_CHECKPOINT.resolve():
        raise ValueError("Only the canonical completed legacy Stage-1 endpoint is admitted")
    config_path = checkpoint / "config.json"
    base = academic._checkpoint_identity(checkpoint, config_path)
    root_inventory = _relative_file_inventory(checkpoint)
    trainer_states, trainer_summary = _trainer_state_inventory(checkpoint)
    shard_rows = [row for row in base["state_file_inventory"] if row["name"].endswith(".distcp")]
    if (
        base["config"]["sha256"] != EXPECTED_CONFIG_SHA256
        or base["checkpoint_marker"]["sha256"] != EXPECTED_CHECKPOINT_MARKER_SHA256
        or base["dcp_metadata"]["sha256"] != EXPECTED_DCP_METADATA_SHA256
        or base["state_file_count"] != EXPECTED_STATE_FILE_COUNT
        or base["state_bytes"] != EXPECTED_STATE_BYTES
        or base["state_file_inventory_sha256"] != EXPECTED_STATE_FILE_INVENTORY_SHA256
        or len(shard_rows) != EXPECTED_DISTCP_SHARD_COUNT
        or sum(row["bytes"] for row in shard_rows) != EXPECTED_DISTCP_SHARD_BYTES
        or len(root_inventory) != EXPECTED_ROOT_FILE_COUNT
        or sum(row["bytes"] for row in root_inventory) != EXPECTED_ROOT_BYTES
        or academic._canonical_sha256(root_inventory) != EXPECTED_ROOT_FILE_INVENTORY_SHA256
    ):
        raise ValueError("Legacy Stage-1 checkpoint identity differs")
    return {
        **base,
        "identity_scope": _checkpoint_identity_scope(),
        "distcp_shard_count": len(shard_rows),
        "distcp_shard_bytes": sum(row["bytes"] for row in shard_rows),
        "root_file_inventory": root_inventory,
        "root_file_inventory_sha256": academic._canonical_sha256(root_inventory),
        "root_file_count": len(root_inventory),
        "root_bytes": sum(row["bytes"] for row in root_inventory),
        "trainer_state_inventory": trainer_states,
        "trainer_state_inventory_sha256": academic._canonical_sha256(trainer_states),
        "trainer_state_summary": trainer_summary,
        "dcp_key_projection": _dcp_key_projection(Path(base["state_dir"])),
    }


def _legacy_lineage(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Project the exact legacy training lineage from its fully hashed config."""
    launch = raw_config.get("launch")
    trainer = raw_config.get("trainer")
    dataset = raw_config.get("dataset")
    model = raw_config.get("model")
    if (
        not isinstance(launch, dict)
        or not isinstance(trainer, dict)
        or not isinstance(dataset, dict)
        or not isinstance(model, dict)
    ):
        raise ValueError("Legacy Stage-1 config structure differs")
    git = launch.get("git")
    callbacks = trainer.get("callbacks")
    wandb = callbacks.get("wandb") if isinstance(callbacks, dict) else None
    max_duration = trainer.get("max_duration")
    if (
        not isinstance(git, dict)
        or not isinstance(wandb, dict)
        or not isinstance(max_duration, dict)
    ):
        raise ValueError("Legacy Stage-1 config lineage is incomplete")
    token_ids = dataset.get("token_ids")
    if not isinstance(token_ids, dict):
        raise ValueError("Legacy Stage-1 config token IDs are missing")
    lineage = {
        "role": "historical_full_fixed_stage1_reference",
        "training_git_revision": git.get("ref"),
        "run_name": wandb.get("name"),
        "wandb": {
            "project": wandb.get("project"),
            "group": wandb.get("group"),
            "run_id": wandb.get("run_id"),
        },
        "base_checkpoint_declared_by_config": raw_config.get("base_checkpoint"),
        "maximum_steps": max_duration.get("value"),
        "global_batch_size_tokens": raw_config.get("global_batch_size"),
        "training_tokenizer_provenance": {
            "identifier": raw_config.get("tokenizer_id"),
            "revision": None,
            "fingerprint": None,
            "completeness": (
                "legacy config records the tokenizer identifier but does not pin its revision "
                "or file fingerprint"
            ),
        },
        "model_source": {
            "molmo2_config_model_id": raw_config.get("molmo2_config_model_id"),
            "molmo2_config_revision": raw_config.get("molmo2_config_revision"),
            "vision_model_id": raw_config.get("vision_model_id"),
            "vision_revision": raw_config.get("vision_revision"),
            "vision_fingerprint": raw_config.get("vision_fingerprint"),
        },
        "config_subtree_sha256": {
            "model": academic._canonical_sha256(model),
            "language_model": academic._canonical_sha256(model.get("lm")),
            "vision": academic._canonical_sha256(model.get("vision")),
            "connector": academic._canonical_sha256(model.get("connector")),
            "dataset_token_ids": academic._canonical_sha256(token_ids),
        },
    }
    expected = {
        "training_git_revision": EXPECTED_TRAINING_GIT_REVISION,
        "run_name": EXPECTED_RUN_NAME,
        "wandb": {
            "project": "molmo2-stage1",
            "group": "s002-stage1-corrected-32k",
            "run_id": EXPECTED_WANDB_RUN_ID,
        },
        "maximum_steps": EXPECTED_GLOBAL_STEP,
        "global_batch_size_tokens": 327_680,
    }
    if any(lineage[field] != value for field, value in expected.items()):
        raise ValueError("Legacy Stage-1 config run lineage differs")
    if lineage["config_subtree_sha256"] != {
        "model": "92e744bab90fa2c63db7a2e73f6f2eb1bc28668dedeff1651b3753bd0cd0b4ad",
        "language_model": "2967410d31ea00c434a9b9e10e8e26d491bb6d4c1c1ca17b6db88b4c4ae6276f",
        "vision": "2df39473ff2dfe7ce8b31b64ddac63eb1d270d7ec9cb8cdf49794712e4db9eb7",
        "connector": "23f5af11291b358560f417267cd8b43f7b0de9a60a00320d2d8b8aa32c74127a",
        "dataset_token_ids": academic.EXPECTED_MOLMO2_TOKEN_IDS_SHA256,
    }:
        raise ValueError("Legacy Stage-1 config model or token-ID identity differs")
    return lineage


def _evaluation_tokenizer(raw_config: Mapping[str, Any], cache_dir: str) -> tuple[Any, Any]:
    model = raw_config.get("model")
    if (
        raw_config.get("tokenizer_id") != TOKENIZER_IDENTIFIER
        or not isinstance(model, dict)
        or not isinstance(model.get("lm"), dict)
        or type(model["lm"].get("vocab_size")) is not int
    ):
        raise ValueError("Legacy Stage-1 tokenizer or model-vocabulary config differs")
    tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
        identifier=TOKENIZER_IDENTIFIER,
        revision=TOKENIZER_REVISION,
        expected_fingerprint=TOKENIZER_FINGERPRINT,
        cache_dir=cache_dir,
        model_vocab_size=model["lm"]["vocab_size"],
    )
    config_token_ids = raw_config["dataset"]["token_ids"]
    if (
        tokenizer.eos_token_id != academic.EXPECTED_EOS_TOKEN_ID
        or tokenizer.pad_token_id != academic.EXPECTED_PAD_TOKEN_ID
        or token_ids.as_config_dict() != config_token_ids
        or academic._canonical_sha256(token_ids.as_config_dict())
        != academic.EXPECTED_MOLMO2_TOKEN_IDS_SHA256
        or raw_config["model"].get("image_patch_token_id") != token_ids.im_patch_id
        or raw_config.get("collator", {}).get("pad_token_id") != tokenizer.pad_token_id
    ):
        raise ValueError("Legacy Stage-1 evaluation tokenizer token identities differ")
    return tokenizer, token_ids


def _tokenizer_payload(tokenizer: Any, token_ids: Any) -> dict[str, Any]:
    return {
        "usage": "evaluation_only_exact_pin",
        "identifier": TOKENIZER_IDENTIFIER,
        "revision": TOKENIZER_REVISION,
        "fingerprint": TOKENIZER_FINGERPRINT,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "token_ids": token_ids.as_config_dict(),
        "token_ids_sha256": academic._canonical_sha256(token_ids.as_config_dict()),
        "historical_training_revision_was_pinned": False,
    }


def _manifest_reference(manifest: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    reference = {
        **identity,
        "content_sha256": manifest["content_sha256"],
        "partial": manifest["selection"]["partial"],
        "panel_status": manifest["selection"]["panel_status"],
        "builder_git": manifest["git"],
    }
    if (
        reference["path"] != str(EXPECTED_MANIFEST.resolve())
        or reference["sha256"] != EXPECTED_MANIFEST_SHA256
        or reference["content_sha256"] != EXPECTED_MANIFEST_CONTENT_SHA256
        or reference["builder_git"] != EXPECTED_MANIFEST_GIT
    ):
        raise ValueError("Legacy comparison requires the exact frozen academic manifest")
    return reference


def _interpretation_limits() -> dict[str, Any]:
    return {
        "descriptive_only": True,
        "official_leaderboard_submission": False,
        "historical_reference_role": "old full fixed Stage-1 before vision alignment",
        "image_overlap_field_semantics": (
            "alignment_train_image_overlap and shuffled_alignment_train_image_overlap refer "
            "only to the later vision-alignment train-union inventory"
        ),
        "legacy_stage1_contamination_claim": (
            "none; the manifest does not inventory the legacy Stage-1 training mixture"
        ),
        "training_tokenizer_provenance": (
            "the old training config pins only the identifier; this receipt separately pins "
            "the tokenizer revision and fingerprint used for evaluation"
        ),
        "inherited_protocol_claim_scope": (
            "the unchanged frozen protocol's 'joint step selection' claim records the panel's "
            "original purpose; this receipt uses it only as a historical Stage-1 reference"
        ),
        "legacy_sequence_cap_comparability": (
            "all 9,216 control prompts are exactly rederived, have maximum input length 1,246, "
            "and require at most 1,259 tokens including the frozen generation allowance; none "
            "exceeds the legacy 2,560-token training cap"
        ),
    }


def _validate_legacy_sequence_cap(tasks: Mapping[str, Any], tokenizer: Any) -> None:
    """Rederive every prompt length and prove the old 2,560-token cap is non-confounding."""
    input_lengths = []
    required_lengths = []
    for task in academic.DEFAULT_TASKS:
        for row in tasks[task]["examples"]:
            example = academic._receipt_example_from_row(task, row)
            grid = row["image_grid_signature"]
            image_ids = academic.build_image_token_ids(*grid)
            prompt = (
                academic._build_mc_prompt(example.question, example.options)
                if example.options
                else academic._free_answer_prompt(example.question)
            )
            expected_input = len(
                academic.document_prompt_ids(tokenizer, prompt, image_ids=image_ids)
            )
            for control in academic.CONTROLS:
                if row["controls"][control]["input_tokens"] != expected_input:
                    raise ValueError(
                        f"Legacy Stage-1 {task}/{example.example_id}/{control} input-token "
                        "count was not rederived"
                    )
                input_lengths.append(expected_input)
                required_lengths.append(
                    expected_input + (academic.DEFAULT_MAX_NEW_TOKENS if not example.options else 0)
                )
    if (
        len(input_lengths) != 9_216
        or max(input_lengths) != 1_246
        or max(required_lengths) != 1_259
        or any(length > 2_560 for length in required_lengths)
    ):
        raise ValueError("Legacy Stage-1 sequence-cap comparability audit differs")


def _model_load_payload(
    train_module: Any,
    state_dir: Path,
    *,
    checkpoint_load_threads: int,
) -> dict[str, Any]:
    metadata = FileSystemReader(state_dir).read_metadata()
    checkpoint_keys = set(metadata.state_dict_metadata)
    resolved_rows = []
    missing = []
    for model_part in train_module.model_parts:
        for name, parameter in model_part.named_parameters():
            checkpoint_key = train_module._resolve_model_checkpoint_key(name, checkpoint_keys)
            if checkpoint_key is None:
                missing.append(name)
            else:
                resolved_rows.append(
                    {
                        "model_parameter": name,
                        "checkpoint_key": checkpoint_key,
                        "local_numel": parameter.numel(),
                    }
                )
    resolved_keys = sorted(row["checkpoint_key"] for row in resolved_rows)
    model_keys = sorted(key for key in checkpoint_keys if key.endswith(".main"))
    if missing or len(resolved_keys) != len(set(resolved_keys)) or resolved_keys != model_keys:
        raise RuntimeError(
            "Legacy Stage-1 eval load did not resolve every and only model tensor key; "
            f"missing={missing[:8]}"
        )
    local = {
        "rank": dist.get_rank(),
        "resolved_model_parameter_count": len(resolved_rows),
        "resolved_checkpoint_keys_sha256": academic._canonical_sha256(resolved_keys),
        "remaining_meta_parameter_count": sum(
            int(parameter.is_meta)
            for model_part in train_module.model_parts
            for parameter in model_part.parameters()
        ),
    }
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    expected_local = {
        "resolved_model_parameter_count": EXPECTED_MODEL_KEY_COUNT,
        "resolved_checkpoint_keys_sha256": EXPECTED_MODEL_KEYS_SHA256,
        "remaining_meta_parameter_count": 0,
    }
    if [row["rank"] for row in gathered] != list(range(academic.EP_DEGREE)) or any(
        any(row[field] != value for field, value in expected_local.items()) for row in gathered
    ):
        raise RuntimeError("Legacy Stage-1 model-load coverage differs across EP ranks")
    return {
        "checkpoint_kind": "multimodal_stage1",
        "api": "MultimodalOLMoDDPTrainModule.load_state_dict_direct",
        "state_dir": str(state_dir),
        "eval_only": True,
        "load_optimizer_state": False,
        "process_group": "WORLD",
        "world_size": academic.EP_DEGREE,
        "ep_degree": academic.EP_DEGREE,
        "expert_parallel_path": ExpertParallelPath.sync_1d.value,
        "checkpoint_load_threads": checkpoint_load_threads,
        "coverage": expected_local,
        "all_rank_completion": gathered,
    }


def _validate_checkpoint_payload(value: Any) -> dict[str, Any]:
    fields = (
        "checkpoint",
        "config",
        "checkpoint_marker",
        "state_dir",
        "dcp_metadata",
        "state_file_inventory",
        "state_file_inventory_sha256",
        "state_file_count",
        "state_bytes",
        "identity_scope",
        "distcp_shard_count",
        "distcp_shard_bytes",
        "root_file_inventory",
        "root_file_inventory_sha256",
        "root_file_count",
        "root_bytes",
        "trainer_state_inventory",
        "trainer_state_inventory_sha256",
        "trainer_state_summary",
        "dcp_key_projection",
    )
    checkpoint = academic._exact_mapping(value, fields, name="legacy Stage-1 checkpoint")
    config = academic._exact_mapping(
        checkpoint["config"], ("path", "bytes", "sha256"), name="legacy Stage-1 config"
    )
    marker = academic._exact_mapping(
        checkpoint["checkpoint_marker"],
        ("path", "bytes", "sha256"),
        name="legacy Stage-1 checkpoint marker",
    )
    dcp_metadata = academic._exact_mapping(
        checkpoint["dcp_metadata"],
        ("path", "bytes", "sha256"),
        name="legacy Stage-1 DCP metadata",
    )
    expected_root = EXPECTED_CHECKPOINT.resolve()
    expected_state_dir = expected_root / "model_and_optim"
    if (
        checkpoint["checkpoint"] != str(expected_root)
        or config
        != {
            "path": str(expected_root / "config.json"),
            "bytes": 28_061,
            "sha256": EXPECTED_CONFIG_SHA256,
        }
        or marker
        != {
            "path": str(expected_root / ".metadata.json"),
            "bytes": 40,
            "sha256": EXPECTED_CHECKPOINT_MARKER_SHA256,
        }
        or checkpoint["state_dir"] != str(expected_state_dir)
        or dcp_metadata
        != {
            "path": str(expected_state_dir / ".metadata"),
            "bytes": 4_216_956,
            "sha256": EXPECTED_DCP_METADATA_SHA256,
        }
        or checkpoint["state_file_count"] != EXPECTED_STATE_FILE_COUNT
        or checkpoint["state_bytes"] != EXPECTED_STATE_BYTES
        or checkpoint["state_file_inventory_sha256"] != EXPECTED_STATE_FILE_INVENTORY_SHA256
        or checkpoint["distcp_shard_count"] != EXPECTED_DISTCP_SHARD_COUNT
        or checkpoint["distcp_shard_bytes"] != EXPECTED_DISTCP_SHARD_BYTES
        or checkpoint["root_file_count"] != EXPECTED_ROOT_FILE_COUNT
        or checkpoint["root_bytes"] != EXPECTED_ROOT_BYTES
        or checkpoint["root_file_inventory_sha256"] != EXPECTED_ROOT_FILE_INVENTORY_SHA256
        or checkpoint["trainer_state_inventory_sha256"] != EXPECTED_TRAIN_STATE_INVENTORY_SHA256
        or checkpoint["identity_scope"] != _checkpoint_identity_scope()
        or checkpoint["trainer_state_summary"] != _expected_trainer_state_summary()
        or checkpoint["dcp_key_projection"] != _expected_dcp_key_projection()
    ):
        raise ValueError("Legacy Stage-1 serialized checkpoint identity differs")
    for inventory, expected_count, expected_bytes, expected_sha in (
        (
            checkpoint["state_file_inventory"],
            EXPECTED_STATE_FILE_COUNT,
            EXPECTED_STATE_BYTES,
            EXPECTED_STATE_FILE_INVENTORY_SHA256,
        ),
        (
            checkpoint["root_file_inventory"],
            EXPECTED_ROOT_FILE_COUNT,
            EXPECTED_ROOT_BYTES,
            EXPECTED_ROOT_FILE_INVENTORY_SHA256,
        ),
    ):
        if (
            not isinstance(inventory, list)
            or len(inventory) != expected_count
            or any(
                not isinstance(row, dict)
                or set(row) != {"name", "bytes"}
                or not isinstance(row["name"], str)
                or not row["name"]
                or type(row["bytes"]) is not int
                or row["bytes"] <= 0
                for row in inventory
            )
            or sum(row.get("bytes", 0) for row in inventory) != expected_bytes
            or academic._canonical_sha256(inventory) != expected_sha
        ):
            raise ValueError("Legacy Stage-1 serialized file inventory differs")
    trainer_states = checkpoint["trainer_state_inventory"]
    if (
        not isinstance(trainer_states, list)
        or len(trainer_states) != EXPECTED_TRAIN_STATE_COUNT
        or any(
            not isinstance(row, dict)
            or set(row) != {"name", "bytes", "sha256"}
            or row["name"] != f"train/rank{rank}.pt"
            or type(row["bytes"]) is not int
            or row["bytes"] <= 0
            or not isinstance(row["sha256"], str)
            or _SHA256_RE.fullmatch(row["sha256"]) is None
            for rank, row in enumerate(trainer_states)
        )
        or sum(row.get("bytes", 0) for row in trainer_states) != EXPECTED_TRAIN_STATE_BYTES
        or academic._canonical_sha256(trainer_states) != EXPECTED_TRAIN_STATE_INVENTORY_SHA256
    ):
        raise ValueError("Legacy Stage-1 serialized trainer states differ")
    return checkpoint


def _validate_model_load(value: Any, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    load = academic._exact_mapping(
        value,
        (
            "checkpoint_kind",
            "api",
            "state_dir",
            "eval_only",
            "load_optimizer_state",
            "process_group",
            "world_size",
            "ep_degree",
            "expert_parallel_path",
            "checkpoint_load_threads",
            "coverage",
            "all_rank_completion",
        ),
        name="legacy Stage-1 model load",
    )
    expected_coverage = {
        "resolved_model_parameter_count": EXPECTED_MODEL_KEY_COUNT,
        "resolved_checkpoint_keys_sha256": EXPECTED_MODEL_KEYS_SHA256,
        "remaining_meta_parameter_count": 0,
    }
    coverage = academic._exact_mapping(
        load["coverage"], tuple(expected_coverage), name="legacy Stage-1 model-load coverage"
    )
    if (
        load["checkpoint_kind"] != "multimodal_stage1"
        or load["api"] != "MultimodalOLMoDDPTrainModule.load_state_dict_direct"
        or load["state_dir"] != checkpoint["state_dir"]
        or load["eval_only"] is not True
        or load["load_optimizer_state"] is not False
        or load["process_group"] != "WORLD"
        or load["world_size"] != academic.EP_DEGREE
        or load["ep_degree"] != academic.EP_DEGREE
        or load["expert_parallel_path"] != ExpertParallelPath.sync_1d.value
        or type(load["checkpoint_load_threads"]) is not int
        or load["checkpoint_load_threads"] <= 0
        or coverage != expected_coverage
    ):
        raise ValueError("Legacy Stage-1 model-load declaration differs")
    completions = load["all_rank_completion"]
    if (
        not isinstance(completions, list)
        or len(completions) != academic.EP_DEGREE
        or any(
            not isinstance(row, dict) or set(row) != {"rank", *expected_coverage}
            for row in completions
        )
        or [row.get("rank") for row in completions] != list(range(academic.EP_DEGREE))
        or any(
            {field: row.get(field) for field in expected_coverage} != expected_coverage
            for row in completions
        )
    ):
        raise ValueError("Legacy Stage-1 all-rank load completion differs")
    return load


def _load_manifest(
    reference: Any, *, verify_live_sources: bool
) -> tuple[dict[str, Any], dict[str, dict[str, academic.AcademicExample]] | None]:
    value = academic._exact_mapping(
        reference,
        ("path", "bytes", "sha256", "content_sha256", "partial", "panel_status", "builder_git"),
        name="legacy Stage-1 receipt manifest reference",
    )
    base_reference = {
        field: value[field]
        for field in ("path", "bytes", "sha256", "content_sha256", "partial", "panel_status")
    }
    manifest, loaded = academic._load_receipt_manifest(
        base_reference,
        verify_live_sources=verify_live_sources,
    )
    _manifest_reference(manifest, academic._file_identity(Path(value["path"])))
    if value["builder_git"] != EXPECTED_MANIFEST_GIT:
        raise ValueError("Legacy Stage-1 receipt manifest builder Git differs")
    return manifest, loaded


def _validate_receipt_payload(
    receipt: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    loaded: Mapping[str, Mapping[str, academic.AcademicExample]] | None,
    tokenizer: Any,
    token_ids: Any,
) -> dict[str, Any]:
    value = academic._exact_mapping(
        receipt,
        (
            "schema_version",
            "format",
            "protocol_name",
            "created_at",
            "launch_git",
            "implementation",
            "manifest",
            "checkpoint",
            "legacy_stage1_lineage",
            "model_load",
            "artifact_policy",
            "interpretation_limits",
            "tokenizer",
            "protocol",
            "tasks",
            "content_sha256",
        ),
        name="legacy Stage-1 external academic receipt",
    )
    academic._verify_content_sha256(value, name="legacy Stage-1 external academic receipt")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["format"] != RECEIPT_FORMAT
        or value["protocol_name"] != PROTOCOL_NAME
    ):
        raise ValueError("Legacy Stage-1 external academic envelope differs")
    academic._validate_timestamp(value["created_at"], name="legacy Stage-1 receipt created_at")
    launch_git = academic._exact_mapping(
        value["launch_git"], ("revision", "dirty"), name="legacy Stage-1 launch Git"
    )
    academic._validate_git_identity(launch_git)
    if value["implementation"] != _implementation_identity():
        raise ValueError("Legacy Stage-1 evaluator implementation differs")
    manifest_reference = _manifest_reference(
        manifest,
        academic._file_identity(Path(value["manifest"]["path"])),
    )
    if value["manifest"] != manifest_reference:
        raise ValueError("Legacy Stage-1 receipt manifest reference differs")
    checkpoint = _validate_checkpoint_payload(value["checkpoint"])
    raw_config = academic._load_json_strict(Path(checkpoint["config"]["path"]))
    if value["legacy_stage1_lineage"] != _legacy_lineage(raw_config):
        raise ValueError("Legacy Stage-1 receipt lineage differs")
    _validate_model_load(value["model_load"], checkpoint)
    if value["artifact_policy"] != {
        "descriptive_only": True,
        "promotion_eligible": False,
        "historical_reference_comparison_evidence": True,
    }:
        raise ValueError("Legacy Stage-1 receipt artifact policy differs")
    if value["interpretation_limits"] != _interpretation_limits():
        raise ValueError("Legacy Stage-1 receipt interpretation limits differ")
    if value["tokenizer"] != _tokenizer_payload(tokenizer, token_ids):
        raise ValueError("Legacy Stage-1 receipt evaluation tokenizer differs")
    if value["protocol"] != academic._protocol_payload(manifest):
        raise ValueError("Legacy Stage-1 receipt frozen benchmark protocol differs")
    academic._validate_receipt_tasks(
        value["tasks"],
        manifest=manifest,
        loaded=loaded,
        tokenizer=tokenizer,
        text_vocab_size=min(token_ids.image_token_ids),
    )
    _validate_legacy_sequence_cap(value["tasks"], tokenizer)
    return value


def validate_legacy_stage1_receipt(
    path: str | Path,
    expected_sha256: str,
    *,
    verify_live: bool = True,
    hf_cache: str | None = None,
) -> dict[str, Any]:
    """Strictly reload and rederive one legacy Stage-1 academic receipt.

    :param path: Canonical receipt path.
    :param expected_sha256: Independently supplied raw receipt SHA-256.
    :param verify_live: Rehash live benchmark sources and checkpoint identities.
    :param hf_cache: Optional local Hugging Face cache containing the pinned tokenizer.
    :returns: The validated receipt.
    """
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError("Expected receipt SHA-256 must be lowercase hex")
    receipt_path = Path(path).expanduser().resolve()
    if academic._file_identity(receipt_path)["sha256"] != expected_sha256:
        raise ValueError("Legacy Stage-1 receipt raw SHA-256 differs")
    receipt = academic._load_json_strict(receipt_path)
    manifest, loaded = _load_manifest(receipt.get("manifest"), verify_live_sources=verify_live)
    checkpoint_payload = _validate_checkpoint_payload(receipt.get("checkpoint"))
    raw_config = academic._load_json_strict(Path(checkpoint_payload["config"]["path"]))
    cache_dir = hf_cache or raw_config.get("hf_cache_dir")
    if not isinstance(cache_dir, str) or not cache_dir:
        raise ValueError("A local tokenizer cache is required")
    tokenizer, token_ids = _evaluation_tokenizer(raw_config, cache_dir)
    _validate_receipt_payload(
        receipt,
        manifest=manifest,
        loaded=loaded,
        tokenizer=tokenizer,
        token_ids=token_ids,
    )
    if verify_live and _checkpoint_identity(EXPECTED_CHECKPOINT) != checkpoint_payload:
        raise ValueError("Legacy Stage-1 live checkpoint identity differs")
    return receipt


def _validate_confirmatory_manifest(manifest: Mapping[str, Any]) -> None:
    selection = manifest["selection"]
    if (
        selection.get("panel_status") != "confirmatory"
        or selection.get("tasks") != list(academic.DEFAULT_TASKS)
        or selection.get("seed") != academic.DEFAULT_SELECTION_SEED
        or selection.get("examples_per_task_limit") != academic.DEFAULT_EXAMPLES_PER_TASK
        or selection.get("partial") is not True
    ):
        raise ValueError("Legacy comparison requires the frozen 512-per-task panel")


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    academic._validate_runtime_args(args)
    manifest, loaded, manifest_identity = academic._validate_manifest_and_load_examples(
        Path(args.manifest)
    )
    _validate_confirmatory_manifest(manifest)
    manifest_reference = _manifest_reference(manifest, manifest_identity)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    opening_checkpoint = _checkpoint_identity(checkpoint)
    raw_config = academic._load_json_strict(checkpoint / "config.json")
    lineage = _legacy_lineage(raw_config)
    launch_git = academic._git_revision()
    academic._validate_git_identity(launch_git)
    implementation = _implementation_identity()

    cache_dir = args.hf_cache or raw_config.get("hf_cache_dir")
    if not isinstance(cache_dir, str) or not cache_dir:
        raise ValueError("A local tokenizer cache is required")
    if args.hf_cache:
        os.environ["HF_HOME"] = str(Path(args.hf_cache).expanduser().resolve())
    os.environ.setdefault("OLMO_USE_OWN_SYMM_MEM", "1")
    os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1")
    os.environ.setdefault("OLMO_OWN_SYMM_PREWARM", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    prepare_training_environment()
    try:
        model, module_config, checkpoint_kind = academic._build_model_and_module_config(
            raw_config,
            ep_degree=academic.EP_DEGREE,
            max_sequence_length=args.max_sequence_length,
            rank_batch_size=args.max_sequence_length,
            ep_path=ExpertParallelPath.sync_1d,
        )
        if checkpoint_kind != "multimodal_stage1":
            raise ValueError("The legacy reference is not a native multimodal Stage-1 checkpoint")
        train_module = module_config.build(model, eval_only=True)
        state_dir = Path(academic._checkpoint_state_dir(checkpoint)).resolve()
        train_module.load_state_dict_direct(
            state_dir,
            process_group=dist.group.WORLD,
            thread_count=args.checkpoint_load_threads,
            load_optim_state=False,
        )
        academic._set_model_parts_eval(train_module)
        model_load = _model_load_payload(
            train_module,
            state_dir,
            checkpoint_load_threads=args.checkpoint_load_threads,
        )
        tokenizer, token_ids = _evaluation_tokenizer(raw_config, cache_dir)
        if (
            academic._answer_token_coverage(loaded, tokenizer)
            != academic.EXPECTED_ANSWER_TOKEN_COVERAGE
        ):
            raise ValueError("Frozen answer-token coverage differs for the evaluation tokenizer")
        inference = academic._NativeAcademicInference(
            train_module,
            tokenizer,
            token_ids,
            max_sequence_length=args.max_sequence_length,
            max_crops=args.max_crops,
            max_new_tokens=args.max_new_tokens,
            sequence_bucket_size=args.sequence_bucket_size,
        )
        tasks = academic._evaluate_manifest(inference, manifest, loaded)

        (
            closing_manifest,
            closing_loaded,
            closing_manifest_identity,
        ) = academic._validate_manifest_and_load_examples(Path(args.manifest))
        if (
            closing_manifest != manifest
            or closing_manifest_identity != manifest_identity
            or {task: tuple(examples) for task, examples in closing_loaded.items()}
            != {task: tuple(examples) for task, examples in loaded.items()}
        ):
            raise ValueError("Manifest, sources, or selected images changed during evaluation")
        if _checkpoint_identity(checkpoint) != opening_checkpoint:
            raise ValueError("Legacy Stage-1 checkpoint identity changed during evaluation")
        if _implementation_identity() != implementation:
            raise ValueError("Legacy Stage-1 evaluator implementation changed during evaluation")
        closing_git = academic._git_revision()
        academic._validate_git_identity(closing_git)
        if closing_git != launch_git:
            raise ValueError("Launch Git identity changed during legacy Stage-1 evaluation")

        payload = {
            "schema_version": SCHEMA_VERSION,
            "format": RECEIPT_FORMAT,
            "protocol_name": PROTOCOL_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "launch_git": launch_git,
            "implementation": implementation,
            "manifest": manifest_reference,
            "checkpoint": opening_checkpoint,
            "legacy_stage1_lineage": lineage,
            "model_load": model_load,
            "artifact_policy": {
                "descriptive_only": True,
                "promotion_eligible": False,
                "historical_reference_comparison_evidence": True,
            },
            "interpretation_limits": _interpretation_limits(),
            "tokenizer": _tokenizer_payload(tokenizer, token_ids),
            "protocol": academic._protocol_payload(manifest),
            "tasks": tasks,
        }
        receipt = academic._attach_content_sha256(payload)
        _validate_receipt_payload(
            receipt,
            manifest=manifest,
            loaded=loaded,
            tokenizer=tokenizer,
            token_ids=token_ids,
        )
        publication: list[Any] = [None]
        if get_rank() == 0:
            try:
                output = Path(args.output)
                academic._write_json_no_overwrite(output, receipt)
                raw_sha256 = academic._sha256_file_stable(
                    academic._artifact_path(output, name="receipt")
                )
                validate_legacy_stage1_receipt(
                    output,
                    raw_sha256,
                    verify_live=False,
                    hf_cache=cache_dir,
                )
                publication[0] = {"ok": True, "sha256": raw_sha256}
            except Exception as error:  # noqa: BLE001 - propagate rank-zero persistence failure.
                publication[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(publication, src=0)
        result = publication[0]
        if not isinstance(result, Mapping) or result.get("ok") is not True:
            detail = result.get("error") if isinstance(result, Mapping) else repr(result)
            raise RuntimeError(f"Could not persist legacy Stage-1 academic receipt: {detail}")
        if get_rank() == 0:
            log.info(
                "Wrote legacy Stage-1 academic receipt %s (sha256=%s)",
                args.output,
                result["sha256"],
            )
        return receipt
    finally:
        teardown_training_environment()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate the exact legacy Stage-1 endpoint on the frozen panel."
    )
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--hf-cache")
    evaluate.add_argument(
        "--max-sequence-length", type=int, default=academic.DEFAULT_MAX_SEQUENCE_LENGTH
    )
    evaluate.add_argument("--max-crops", type=int, default=academic.DEFAULT_MAX_CROPS)
    evaluate.add_argument("--max-new-tokens", type=int, default=academic.DEFAULT_MAX_NEW_TOKENS)
    evaluate.add_argument(
        "--sequence-bucket-size", type=int, default=academic.DEFAULT_SEQUENCE_BUCKET_SIZE
    )
    evaluate.add_argument("--checkpoint-load-threads", type=int, default=8)

    validate = commands.add_parser("validate-receipt", help="Strictly rederive one receipt.")
    validate.add_argument("--receipt", required=True)
    validate.add_argument("--expected-sha256", required=True)
    validate.add_argument("--hf-cache")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parser().parse_args()
    if args.command == "evaluate":
        _evaluate(args)
        return
    if args.command == "validate-receipt":
        validate_legacy_stage1_receipt(
            args.receipt,
            args.expected_sha256,
            verify_live=True,
            hf_cache=args.hf_cache,
        )
        log.info("Validated legacy Stage-1 external-academic receipt %s", args.receipt)
        return
    raise AssertionError(f"Unknown command {args.command!r}")


if __name__ == "__main__":
    main()
