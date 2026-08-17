"""Evaluate the two canonical vision-alignment phase-boundary checkpoints.

This is a checkpoint-policy wrapper around :mod:`vision_alignment_external_academic`.
It reuses that frozen evaluator's manifest loader, prompts, native inference, task scorers,
row schema, and aggregate validation unchanged.  The wrapper admits only the permanent bridge
step500 checkpoint immediately before vision-tower unfreezing and the permanent perception
treatment step4000 checkpoint immediately before joint language-model unfreezing.

Each receipt binds the exact phase configuration, trainer states, approved promotion evidence,
full SHA-256 DCP inventory, downstream transition configuration, and all-rank native load
coverage.  The result is descriptive phase-boundary evidence, not promotion or causal evidence.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import vision_alignment_external_academic as academic
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from olmo_core.data.multimodal.vision_alignment_sources import (
    load_pinned_vision_alignment_tokenizer,
)
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
from olmo_core.train import prepare_training_environment, teardown_training_environment

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
RECEIPT_FORMAT = "vision_alignment_external_academic_phase_boundary_receipt"
PROTOCOL_NAME = "vision-alignment-external-academic-phase-boundary-ep8-v1"

EXPECTED_FROZEN_EVALUATOR_SHA256 = (
    "29a2c0fa37993e211ef634914aa740119178be3770cc1c178688ca9986e441f0"
)
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

TOKENIZER_IDENTIFIER = "allenai/dolma2-tokenizer"
TOKENIZER_REVISION = "5292e5d6c0f40b67cc765fe41bec991cf4345b5c"
TOKENIZER_FINGERPRINT = "8fec2af8c372f4c72a1a665ad8e70517625f94f041dbfcb7db4932071380f9a7"
IMAGE_TOKEN_ROWS = [100278, 100279, 100280, 100281, 100282, 100283]
EMPTY_LIST_SHA256 = "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"

ROOT = Path("/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment")
BRIDGE_CHECKPOINT = ROOT / "checkpoints/vision-alignment-bridge-real-v1/step500"
PERCEPTION_CHECKPOINT = ROOT / "checkpoints/vision-alignment-perception-treatment-v1/step4000"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _candidate(
    *,
    checkpoint: Path,
    step: int,
    phase: str,
    lineage_id: str,
    config_sha256: str,
    checkpoint_identity_sha256: str,
    dcp_metadata_sha256: str,
    full_state_inventory_sha256: str,
    data_contract_sha256: str,
    trainable_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": config_sha256,
        "checkpoint_identity_sha256": checkpoint_identity_sha256,
        "checkpoint_marker_sha256": (
            "77dfdeec42fe7990f4b3b9c4eeecd480edcf5066c110603b115920af38423d03"
        ),
        "data_contract_sha256": data_contract_sha256,
        "dcp_metadata_sha256": dcp_metadata_sha256,
        "global_step": step,
        "image_embedding_rows": IMAGE_TOKEN_ROWS,
        "lineage_id": lineage_id,
        "phase": phase,
        "state_file_inventory_sha256": full_state_inventory_sha256,
        "trainable_contract_sha256": trainable_contract_sha256,
        "vocab_size": 100352,
    }


BOUNDARIES: dict[str, dict[str, Any]] = {
    "bridge_step500": {
        "key": "bridge_step500",
        "role": "bridge_pre_vision_tower_unfreeze",
        "transition": "bridge_to_perception_treatment",
        "checkpoint": BRIDGE_CHECKPOINT,
        "step": 500,
        "phase": "bridge",
        "next_phase": "perception",
        "config_bytes": 31_762,
        "config_sha256": "41df40c299f4f3101c3ef58d657d99fb624194beaee7321ea456727212be1dad",
        "marker_sha256": "77dfdeec42fe7990f4b3b9c4eeecd480edcf5066c110603b115920af38423d03",
        "dcp_metadata_bytes": 415_630,
        "dcp_metadata_sha256": ("8338c070ea2c6ada63a838e73b880ea0716f48df5de8b7f936c50d570229d342"),
        "state_file_count": 257,
        "state_bytes": 67_072_480_922,
        "state_file_inventory_sha256": (
            "14d9e963f73800cfdbf317b6b35d04eb92a3360b837edd436db407acc124ef96"
        ),
        "distcp_shard_count": 256,
        "distcp_shard_bytes": 67_072_065_292,
        "root_file_count": 275,
        "root_bytes": 67_072_778_516,
        "root_file_inventory_sha256": (
            "d59b32379bfb4fe3c24a6d6fe52bd92b5e23858e5de91eb1b62c0e761737eac6"
        ),
        "trainer_state_count": 16,
        "trainer_state_bytes": 265_792,
        "trainer_state_inventory_sha256": (
            "581f42c68e1dc692559d319c6081cd19af7ccfce45d8599f92266e66b97a9d83"
        ),
        "trainer_projection_sha256": (
            "45e2c69b0f2a1932a210e4fad829e0767385c37f95e06cb32811bc5022a1ef38"
        ),
        "global_train_tokens": 163_840_000,
        "wandb_run_id": "f2rdhz4y",
        "run_name": "vision-alignment-bridge-real-v1",
        "training_git_revision": "9aa6d27e3f76f373c7d6cd9bc47c91b3f0f3e571",
        "current_freeze": [
            "vision.*",
            "lm.embedding_norm.*",
            "lm.blocks.*",
            "lm.lm_head.*",
        ],
        "next_freeze": [
            "lm.embedding_norm.*",
            "lm.blocks.*",
            "lm.lm_head.*",
        ],
        "newly_trainable": ["vision.*"],
        "still_frozen": [
            "lm.embedding_norm.*",
            "lm.blocks.*",
            "lm.lm_head.*",
        ],
        "trainability_arm": None,
        "next_trainability_arm": "treatment",
        "lineage_id": "vision-alignment-bridge-real-v1",
        "data_contract_sha256": (
            "fbaebbc192458ee3e517d747937b1ad3caa08e1ba7612e7b2fbb11357b76f4e7"
        ),
        "trainable_contract_sha256": (
            "644f7a582939d22ab7288ddd8b05f046635fee678ede5642c8947631ba3ab1bd"
        ),
        "source_audit_path": ROOT / "artifacts/bridge-source-audit-v1.json",
        "source_audit_bytes": 155_369,
        "source_audit_sha256": ("b26c9a877ef6444a157828a231b1a15edb87d2c261e0d4a3e4e5c31c63cbe06c"),
        "source_audit_fingerprint": (
            "6d49234bb6233e9ce6a4becec04de6c068f3cb3fd1cfddb71a219b88fce4b3a0"
        ),
        "phase_data_provenance": None,
        "approval_gate_path": ROOT / "evals/bridge-real-v1-promotion-v1/parent-gate-v2.json",
        "approval_gate_bytes": 1_552,
        "approval_gate_sha256": (
            "e6dea8f8f1fd52c2b008e5460854169a893a814bd19da77b1567330116282b6a"
        ),
        "approval_gate_version": 2,
        "approval_gate_approved_at": "2026-08-12T20:38:20Z",
        "promotion_bundle_path": ROOT / "evals/bridge-real-v1-promotion-v1/promotion-bundle.json",
        "promotion_bundle_bytes": 5_383,
        "promotion_bundle_sha256": (
            "efaecfbce9020e83b5c8eea52ea1a47fb581861bddc85c8273fefb4815f7f977"
        ),
        "promotion_bundle_content_sha256": (
            "502a2ba400c11b5decd67cfdc43a044daa45dab5118f0b85c3e90d44c037a5d4"
        ),
        "promotion_bundle_format": "vision_alignment_bridge_promotion_bundle",
        "full_dcp_evidence_path": (ROOT / "evals/bridge-real-v1-matched-wrong-v3/step500.json"),
        "full_dcp_evidence_bytes": 926_349,
        "full_dcp_evidence_sha256": (
            "28e4f9b5122250bd851781a879c75c52b67d6b578760afc21ac1f5d665c4430c"
        ),
        "full_dcp_evidence_content_sha256": None,
        "full_dcp_selector": ("checkpoint",),
        "full_dcp_inventory_sha256": (
            "1b6e76a9ed2055c3681d52da827033f1dcf1ceca5f31e2a78aac9b009fbd3c5e"
        ),
        "full_checkpoint_identity_sha256": (
            "671c3b0034ee73f0ed74a99e24a9970673db1ec2a9b9c14d8f0facadb6b54e9e"
        ),
        "next_config_path": PERCEPTION_CHECKPOINT / "config.json",
        "next_config_bytes": 31_638,
        "next_config_sha256": ("6e6da90df7048d74fe611c45032b8c7b5c9846725a2029492b82353589ceca23"),
        "next_lineage_id": "vision-alignment-perception-treatment-v1",
        "next_run_name": "vision-alignment-perception-treatment-v1",
        "next_max_steps": 4_000,
        "next_sequence_length": 2_560,
        "dcp_projection": {
            "dcp_state_key_count": 856,
            "dcp_state_keys_sha256": (
                "462f3b905d71e7199c1df0815217ca2a6fa6de5bee0dafab044537d72cc98b57"
            ),
            "optimizer_main_key_count": 12,
            "optimizer_main_keys_sha256": (
                "53d025b46c17057f7f17491ed2574d5f8d330f0676d1f03043612aa41eb40d08"
            ),
            "frozen_model_key_count": 806,
            "frozen_model_keys_sha256": (
                "6c87c9ef4d2b1ac137a6a05b2dbd0cad64f7d49f6415d9827fede806cb1adbc0"
            ),
            "model_tensor_key_count": 818,
            "model_tensor_keys_sha256": (
                "6f9078544595a32d5b763793d61dd64a1dc4f317804078aa12071edfe4fb6156"
            ),
            "optimizer_moment_key_count": 36,
            "optimizer_moment_keys_sha256": (
                "c38fa2c336526ad22a88fe2c88d2519c8e4b6ab8bdff2a00794f5867ddd08294"
            ),
            "persistent_buffer_key_count": 0,
            "persistent_buffer_keys_sha256": EMPTY_LIST_SHA256,
            "auxiliary_keys": ["__moe_skip_step_grad_norms", "__moe_skip_step_losses"],
        },
        "load_coverage": {
            "checkpoint_key_count": 856,
            "complete": True,
            "eval_state_key_count": 12,
            "frozen_state_key_count": 806,
            "load_completed": True,
            "model_parameter_assignments_sha256": (
                "c6518b1a0c42dd2863cc92228bbd6f66410f0cd7ba2c996c9a033dd154c933e7"
            ),
            "model_parameter_checkpoint_key_count": 818,
            "model_parameter_checkpoint_keys_sha256": (
                "6f9078544595a32d5b763793d61dd64a1dc4f317804078aa12071edfe4fb6156"
            ),
            "model_parameter_count": 818,
            "persistent_buffer_count": 0,
            "persistent_buffer_keys_sha256": EMPTY_LIST_SHA256,
            "prepared_load_key_count": 818,
            "shadowed_frozen_key_count": 0,
            "shadowed_frozen_keys_sha256": EMPTY_LIST_SHA256,
            "unused_model_bearing_key_count": 0,
            "sha256": "54ef913c8091906c59222a67e273bda5dacf32c1436db977b409d649dfff1f4d",
        },
    },
    "perception_step4000": {
        "key": "perception_step4000",
        "role": "perception_treatment_pre_joint_language_model_unfreeze",
        "transition": "perception_treatment_to_joint",
        "checkpoint": PERCEPTION_CHECKPOINT,
        "step": 4_000,
        "phase": "perception",
        "next_phase": "joint",
        "config_bytes": 31_638,
        "config_sha256": "6e6da90df7048d74fe611c45032b8c7b5c9846725a2029492b82353589ceca23",
        "marker_sha256": "77dfdeec42fe7990f4b3b9c4eeecd480edcf5066c110603b115920af38423d03",
        "dcp_metadata_bytes": 2_163_479,
        "dcp_metadata_sha256": ("07d61828ec819cf478e934dd1615a894972f069c021a3da1e3fa39bf6f399ba8"),
        "state_file_count": 257,
        "state_bytes": 70_913_692_393,
        "state_file_inventory_sha256": (
            "9bccb9a66afde8e8e6de383fb97afabc2dbb1b923dd0d6188164838e21a9fd73"
        ),
        "distcp_shard_count": 256,
        "distcp_shard_bytes": 70_911_528_914,
        "root_file_count": 275,
        "root_bytes": 70_914_003_559,
        "root_file_inventory_sha256": (
            "b8ff1559757de45435c14def19cf451a6027aa5842476336aaaae4209812cb4f"
        ),
        "trainer_state_count": 16,
        "trainer_state_bytes": 279_488,
        "trainer_state_inventory_sha256": (
            "ffd1a61462728b1d1bff411161a51517f4a0016cea9003568c513bccf5c7614e"
        ),
        "trainer_projection_sha256": (
            "f7e20233169236ec15d2f2ed36e0f607e3dd2863b2ca85652cd05c68c36ccc0d"
        ),
        "global_train_tokens": 1_310_720_000,
        "wandb_run_id": "4eggnrzc",
        "run_name": "vision-alignment-perception-treatment-v1",
        "training_git_revision": "d8ec4f57cf026424ccd13f20452365b6b1df34e5",
        "current_freeze": [
            "lm.embedding_norm.*",
            "lm.blocks.*",
            "lm.lm_head.*",
        ],
        "next_freeze": ["lm.lm_head.w_out.weight"],
        "newly_trainable": [
            "lm.embedding_norm.*",
            "lm.blocks.*",
            "lm.lm_head.norm.weight",
        ],
        "still_frozen": ["lm.lm_head.w_out.weight"],
        "trainability_arm": "treatment",
        "next_trainability_arm": "treatment",
        "lineage_id": "vision-alignment-perception-treatment-v1",
        "data_contract_sha256": (
            "1116f73987da8c94fb8158a2b4e38629fc18cd3227d2c88d5cadeafeadbfd916"
        ),
        "trainable_contract_sha256": (
            "b8721acb806dbf023f1554917a82df4c31d61eb38a172ebf59ea6241b203fa8e"
        ),
        "source_audit_path": ROOT / "artifacts/perception-source-audit-v2.json",
        "source_audit_bytes": 577_325,
        "source_audit_sha256": ("d2a36bcf4d08208d34b3295014c953e8923d8b7d9d885dee9569c6633a87cfcc"),
        "source_audit_fingerprint": (
            "2e9bf63765c674b6a1161cd12313580e29dac028f990bfd9d304a00d844d5d3b"
        ),
        "phase_data_provenance": {
            "path": (
                ROOT / "artifacts/perception-provenance-v2/"
                "vision-alignment-perception-provenance.json"
            ),
            "bytes": 27_059,
            "sha256": "73cb3920676db5e16d789f7257800dcb44b2553b6463cff81beb740213d921e2",
            "content_sha256": ("c29f98a501d10c1ac50542729bcefa9a6b5ef444d6ca8cef22cf52b9dd69d9b3"),
        },
        "approval_gate_path": (
            ROOT / "evals/perception-v1-promotion-v1/" "perception-parent-gate-v3-bfaa56036.json"
        ),
        "approval_gate_bytes": 1_524,
        "approval_gate_sha256": (
            "6f110f00becd2f6360fcb0dd8f85fd78e4bcba787087ef44f3159c5f8d486316"
        ),
        "approval_gate_version": 3,
        "approval_gate_approved_at": "2026-08-13T21:47:16Z",
        "promotion_bundle_path": (
            ROOT / "evals/perception-v1-promotion-v1/promotion-bundle-bfaa56036.json"
        ),
        "promotion_bundle_bytes": 7_700,
        "promotion_bundle_sha256": (
            "6d06a99fb9cbe5e6941689d413c8f832d8c222b28063551a2923000950cadfff"
        ),
        "promotion_bundle_content_sha256": (
            "776cb9a4577c385a60c02f6596a39872763bbe6a8064c8a08f894bc79e38e22d"
        ),
        "promotion_bundle_format": "vision_alignment_perception_promotion_bundle",
        "full_dcp_evidence_path": (
            ROOT / "evals/perception-v1-promotion-v1/" "counterfactual-outcome-bfaa56036.json"
        ),
        "full_dcp_evidence_bytes": 20_220_027,
        "full_dcp_evidence_sha256": (
            "b62ebe1e90a12d5204972e5697cebd65a6484a52e7a806d3d2a0be742d92a6a8"
        ),
        "full_dcp_evidence_content_sha256": (
            "2352ef4cdb63172e850a9c8a67f8f97114a4ca32ecb1e8ff69fae3d77cf0ae3a"
        ),
        "full_dcp_selector": ("checkpoints", "treatment", "step4000"),
        "full_dcp_inventory_sha256": (
            "4aa8389fe8e1725468d9bd7a175dfc158c01f0d85b0b76a4000fb382c64c1b06"
        ),
        "full_checkpoint_identity_sha256": (
            "10b81d98490a0ba5e9e209422db235b64b43c16187091ada5674b8079c51848f"
        ),
        "next_config_path": ROOT / "checkpoints/vision-alignment-joint-v1/step4000/config.json",
        "next_config_bytes": 33_727,
        "next_config_sha256": ("64b302865831b5aaf11e86e142a85b3467a06b93d6c214fb67f7f94a45c4ddc8"),
        "next_lineage_id": "vision-alignment-joint-v1",
        "next_run_name": "vision-alignment-joint-v1",
        "next_max_steps": 16_000,
        "next_sequence_length": 8_192,
        "dcp_projection": {
            "dcp_state_key_count": 2_065,
            "dcp_state_keys_sha256": (
                "f60d4f245f14882010ce8c1944b35ac51bf4e7e83afdae917dbac424aa5475fb"
            ),
            "optimizer_main_key_count": 415,
            "optimizer_main_keys_sha256": (
                "aabf923fb261f20cbbdd11f78c3456eab4e81a1a2df6044f0cc95c16e01fd3c6"
            ),
            "frozen_model_key_count": 403,
            "frozen_model_keys_sha256": (
                "4c2435aff95a7719d686c8994ca382e3a4cef7e0e4c517f320354546a0d0d9be"
            ),
            "model_tensor_key_count": 818,
            "model_tensor_keys_sha256": (
                "b8c3e151779a80844c3714a9e1b95f73f649f61d4bd03968ec9cda0c23522c2f"
            ),
            "optimizer_moment_key_count": 1_245,
            "optimizer_moment_keys_sha256": (
                "27aa9f869cee04568385a4378e27999078bb9cd4f4a2c8ecb135e1ee046075ed"
            ),
            "persistent_buffer_key_count": 0,
            "persistent_buffer_keys_sha256": EMPTY_LIST_SHA256,
            "auxiliary_keys": ["__moe_skip_step_grad_norms", "__moe_skip_step_losses"],
        },
        "load_coverage": {
            "checkpoint_key_count": 2_065,
            "complete": True,
            "eval_state_key_count": 415,
            # The frozen-load surface is determined by the current academic eval topology,
            # not only by stable ``frozen_model.*`` keys in the checkpoint.  The evaluator
            # freezes ``vision.*``, so the multimodal loader adds all 403 vision ``*.main``
            # keys to the 403 stable frozen LM keys.  These 403 vision keys overlap the 415-key
            # eval state, leaving the prepared union at exactly 818 model tensors.
            "frozen_state_key_count": 806,
            "load_completed": True,
            "model_parameter_assignments_sha256": (
                "cc29d766e9364fc47d8f7ed105c4e5c8a212fa07536a2a21a42b0ae6bc58eb25"
            ),
            "model_parameter_checkpoint_key_count": 818,
            "model_parameter_checkpoint_keys_sha256": (
                "b8c3e151779a80844c3714a9e1b95f73f649f61d4bd03968ec9cda0c23522c2f"
            ),
            "model_parameter_count": 818,
            "persistent_buffer_count": 0,
            "persistent_buffer_keys_sha256": EMPTY_LIST_SHA256,
            "prepared_load_key_count": 818,
            "shadowed_frozen_key_count": 0,
            "shadowed_frozen_keys_sha256": EMPTY_LIST_SHA256,
            "unused_model_bearing_key_count": 0,
            "sha256": "6ec333ddba34dce5dd512448a1c235883ee9739c311ec505275144efe44e47c2",
        },
    },
}

for _boundary in BOUNDARIES.values():
    _boundary["candidate"] = _candidate(
        checkpoint=_boundary["checkpoint"],
        step=_boundary["step"],
        phase=_boundary["phase"],
        lineage_id=_boundary["lineage_id"],
        config_sha256=_boundary["config_sha256"],
        checkpoint_identity_sha256=_boundary["full_checkpoint_identity_sha256"],
        dcp_metadata_sha256=_boundary["dcp_metadata_sha256"],
        full_state_inventory_sha256=_boundary["full_dcp_inventory_sha256"],
        data_contract_sha256=_boundary["data_contract_sha256"],
        trainable_contract_sha256=_boundary["trainable_contract_sha256"],
    )


def _boundary_spec(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    matches = [spec for spec in BOUNDARIES.values() if spec["checkpoint"].resolve() == checkpoint]
    if len(matches) != 1:
        raise ValueError(
            "Phase-boundary academic evaluation admits only canonical bridge step500 or "
            "perception-treatment step4000"
        )
    return matches[0]


def _checkpoint_identity_scope() -> dict[str, str]:
    return {
        "distcp_shards": "full-file SHA-256 from approved evidence, verified pre and post",
        "dcp_metadata": "full-file SHA-256 plus strict state-key projection",
        "config_and_marker": "full-file SHA-256",
        "trainer_states": "full-file SHA-256 for every rank plus safe progress projection",
        "root_inventory": "relative path and byte size for every checkpoint file",
        "stability": "opening and closing identities and full DCP hashes must remain exact",
    }


def _implementation_identity() -> dict[str, Any]:
    """Identify the frozen evaluator and this phase-boundary policy wrapper."""
    frozen = academic._implementation_identity()
    evaluator = frozen.get("files", {}).get("evaluator", {})
    if evaluator.get("sha256") != EXPECTED_FROZEN_EVALUATOR_SHA256:
        raise ValueError("Frozen external-academic evaluator implementation differs")
    repo_root = Path(__file__).resolve().parents[3]
    files = {"wrapper": academic._repo_relative_file_identity(Path(__file__), repo_root)}
    return {
        "frozen_evaluator": frozen,
        "wrapper_files": files,
        "wrapper_files_sha256": academic._canonical_sha256(files),
    }


def _relative_file_inventory(root: Path) -> list[dict[str, Any]]:
    root = root.expanduser().resolve()
    _boundary_spec(root)
    rows: list[dict[str, Any]] = []
    directories: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        metadata = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"Phase-boundary checkpoint contains a symlink: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            directories.append(relative)
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise ValueError(f"Phase-boundary checkpoint contains an invalid file: {path}")
        rows.append({"name": relative, "bytes": metadata.st_size})
    if directories != ["model_and_optim", "train"]:
        raise ValueError("Phase-boundary checkpoint directory surface differs")
    return rows


def _expected_trainer_state_summary(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rank_count": spec["trainer_state_count"],
        "ranks": list(range(spec["trainer_state_count"])),
        "global_step": spec["step"],
        "max_steps": spec["step"],
        "global_train_tokens_seen": spec["global_train_tokens"],
        "world_size": 16,
        "data_parallel_world_size": 16,
        "batches_processed": spec["step"],
        "rank0_wandb_run_id": spec["wandb_run_id"],
        "nonzero_rank_wandb_run_ids": [None],
        "wandb_step": spec["step"],
        "wandb_name": spec["run_name"],
        "wandb_project": "vision-alignment",
        "rank_projections_sha256": spec["trainer_projection_sha256"],
    }


def _trainer_state_inventory(
    checkpoint: Path, spec: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_dir = checkpoint / "train"
    paths = sorted(train_dir.glob("rank*.pt"), key=lambda path: int(path.stem[4:]))
    rows = []
    for path in paths:
        identity = academic._file_identity(path)
        rows.append(
            {
                "name": path.relative_to(checkpoint).as_posix(),
                "bytes": identity["bytes"],
                "sha256": identity["sha256"],
            }
        )
    if (
        len(rows) != spec["trainer_state_count"]
        or sum(row["bytes"] for row in rows) != spec["trainer_state_bytes"]
        or academic._canonical_sha256(rows) != spec["trainer_state_inventory_sha256"]
    ):
        raise ValueError("Phase-boundary trainer-state file identities differ")

    projections = []
    for rank, path in enumerate(paths):
        # Every pickle is admitted by its frozen full-file SHA-256 before deserialization.
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
    common = {
        "global_step": spec["step"],
        "max_steps": spec["step"],
        "global_train_tokens_seen": spec["global_train_tokens"],
        "world_size": 16,
        "data_parallel_world_size": 16,
        "batches_processed": spec["step"],
        "wandb_step": spec["step"],
        "wandb_name": spec["run_name"],
        "wandb_project": "vision-alignment",
    }
    for rank, projection in enumerate(projections):
        if projection["rank"] != rank or projection["data_parallel_rank"] != rank:
            raise ValueError("Phase-boundary trainer-state rank coverage differs")
        if any(projection[field] != value for field, value in common.items()):
            raise ValueError(f"Phase-boundary trainer-state rank {rank} progress differs")
        expected_run_id = spec["wandb_run_id"] if rank == 0 else None
        if projection["wandb_run_id"] != expected_run_id:
            raise ValueError(f"Phase-boundary trainer-state rank {rank} W&B identity differs")
    summary = {
        "rank_count": len(projections),
        "ranks": [projection["rank"] for projection in projections],
        **common,
        "rank0_wandb_run_id": projections[0]["wandb_run_id"],
        "nonzero_rank_wandb_run_ids": sorted(
            {projection["wandb_run_id"] for projection in projections[1:]}, key=str
        ),
        "rank_projections_sha256": academic._canonical_sha256(projections),
    }
    if summary != _expected_trainer_state_summary(spec):
        raise ValueError("Phase-boundary trainer-state summary differs")
    return rows, summary


def _dcp_key_projection(state_dir: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    metadata = FileSystemReader(state_dir).read_metadata()
    keys = sorted(metadata.state_dict_metadata)
    main = [key for key in keys if key.endswith(".main")]
    frozen = [key for key in keys if key.startswith("frozen_model.")]
    moments = [key for key in keys if key.endswith((".exp_avg", ".exp_avg_sq", ".step"))]
    buffers = [key for key in keys if key.startswith("model_buffer.")]
    classified = set(main) | set(frozen) | set(moments) | set(buffers)
    projection = {
        "dcp_state_key_count": len(keys),
        "dcp_state_keys_sha256": academic._canonical_sha256(keys),
        "optimizer_main_key_count": len(main),
        "optimizer_main_keys_sha256": academic._canonical_sha256(main),
        "frozen_model_key_count": len(frozen),
        "frozen_model_keys_sha256": academic._canonical_sha256(frozen),
        "model_tensor_key_count": len(main) + len(frozen),
        "model_tensor_keys_sha256": academic._canonical_sha256(sorted(main + frozen)),
        "optimizer_moment_key_count": len(moments),
        "optimizer_moment_keys_sha256": academic._canonical_sha256(moments),
        "persistent_buffer_key_count": len(buffers),
        "persistent_buffer_keys_sha256": academic._canonical_sha256(buffers),
        "auxiliary_keys": [key for key in keys if key not in classified],
    }
    if projection != spec["dcp_projection"]:
        raise ValueError("Phase-boundary DCP state-key projection differs")
    return projection


def _file_reference(path: Path, *, size: int, sha256: str, name: str) -> dict[str, Any]:
    identity = academic._file_identity(path)
    expected = {"path": str(path.resolve()), "bytes": size, "sha256": sha256}
    if identity != expected:
        raise ValueError(f"{name} raw identity differs")
    return identity


def _checkpoint_full_identity(
    spec: Mapping[str, Any], *, include_inventory: bool
) -> dict[str, Any]:
    evidence_path = Path(spec["full_dcp_evidence_path"])
    _file_reference(
        evidence_path,
        size=spec["full_dcp_evidence_bytes"],
        sha256=spec["full_dcp_evidence_sha256"],
        name="full-DCP evidence",
    )
    evidence = academic._load_json_strict(evidence_path)
    content_sha = spec["full_dcp_evidence_content_sha256"]
    if content_sha is not None:
        academic._verify_content_sha256(evidence, name="full-DCP evidence")
        if evidence.get("content_sha256") != content_sha:
            raise ValueError("Full-DCP evidence content identity differs")
    value: Any = evidence
    for field in spec["full_dcp_selector"]:
        if not isinstance(value, dict) or field not in value:
            raise ValueError("Full-DCP evidence checkpoint selector differs")
        value = value[field]
    fields = (
        "root",
        "state_dir",
        "config_sha256",
        "checkpoint_marker_sha256",
        "dcp_metadata_sha256",
        "state_file_hash_algorithm",
        "state_file_inventory_sha256",
        "state_file_inventory",
        "identity_sha256",
    )
    identity = academic._exact_mapping(value, fields, name="full-DCP checkpoint identity")
    inventory = identity["state_file_inventory"]
    if not isinstance(inventory, list) or len(inventory) != spec["state_file_count"]:
        raise ValueError("Full-DCP evidence inventory count differs")
    expected_paths = []
    for index, row in enumerate(inventory):
        item = academic._exact_mapping(
            row, ("path", "size", "sha256"), name=f"full-DCP inventory row {index}"
        )
        relative = Path(str(item["path"]))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[:1] != ("model_and_optim",)
            or type(item["size"]) is not int
            or item["size"] <= 0
            or not isinstance(item["sha256"], str)
            or _SHA256_RE.fullmatch(item["sha256"]) is None
        ):
            raise ValueError("Full-DCP evidence inventory row differs")
        expected_paths.append(relative.as_posix())
    if (
        expected_paths != sorted(expected_paths)
        or len(set(expected_paths)) != len(expected_paths)
        or academic._canonical_sha256(inventory) != identity["state_file_inventory_sha256"]
        or academic._canonical_sha256(
            {
                field: field_value
                for field, field_value in identity.items()
                if field != "identity_sha256"
            }
        )
        != identity["identity_sha256"]
    ):
        raise ValueError("Full-DCP evidence identity is not canonical")
    summary = {
        "root": identity["root"],
        "state_dir": identity["state_dir"],
        "config_sha256": identity["config_sha256"],
        "checkpoint_marker_sha256": identity["checkpoint_marker_sha256"],
        "dcp_metadata_sha256": identity["dcp_metadata_sha256"],
        "state_file_hash_algorithm": identity["state_file_hash_algorithm"],
        "state_file_count": len(inventory),
        "state_file_inventory_sha256": identity["state_file_inventory_sha256"],
        "identity_sha256": identity["identity_sha256"],
    }
    expected_summary = {
        "root": str(spec["checkpoint"]),
        "state_dir": str(spec["checkpoint"] / "model_and_optim"),
        "config_sha256": spec["config_sha256"],
        "checkpoint_marker_sha256": spec["marker_sha256"],
        "dcp_metadata_sha256": spec["dcp_metadata_sha256"],
        "state_file_hash_algorithm": "sha256",
        "state_file_count": spec["state_file_count"],
        "state_file_inventory_sha256": spec["full_dcp_inventory_sha256"],
        "identity_sha256": spec["full_checkpoint_identity_sha256"],
    }
    if summary != expected_summary:
        raise ValueError("Full-DCP evidence checkpoint identity differs")
    return dict(identity) if include_inventory else summary


def _checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    spec = _boundary_spec(checkpoint)
    base = academic._checkpoint_identity(checkpoint, checkpoint / "config.json")
    root_inventory = _relative_file_inventory(checkpoint)
    trainer_states, trainer_summary = _trainer_state_inventory(checkpoint, spec)
    shards = [row for row in base["state_file_inventory"] if row["name"].endswith(".distcp")]
    if (
        base["config"]
        != {
            "path": str(checkpoint / "config.json"),
            "bytes": spec["config_bytes"],
            "sha256": spec["config_sha256"],
        }
        or base["checkpoint_marker"]
        != {
            "path": str(checkpoint / ".metadata.json"),
            "bytes": 40,
            "sha256": spec["marker_sha256"],
        }
        or base["dcp_metadata"]
        != {
            "path": str(checkpoint / "model_and_optim/.metadata"),
            "bytes": spec["dcp_metadata_bytes"],
            "sha256": spec["dcp_metadata_sha256"],
        }
        or base["state_file_count"] != spec["state_file_count"]
        or base["state_bytes"] != spec["state_bytes"]
        or base["state_file_inventory_sha256"] != spec["state_file_inventory_sha256"]
        or len(shards) != spec["distcp_shard_count"]
        or sum(row["bytes"] for row in shards) != spec["distcp_shard_bytes"]
        or len(root_inventory) != spec["root_file_count"]
        or sum(row["bytes"] for row in root_inventory) != spec["root_bytes"]
        or academic._canonical_sha256(root_inventory) != spec["root_file_inventory_sha256"]
    ):
        raise ValueError("Phase-boundary checkpoint identity differs")
    return {
        **base,
        "boundary_key": spec["key"],
        "identity_scope": _checkpoint_identity_scope(),
        "distcp_shard_count": len(shards),
        "distcp_shard_bytes": sum(row["bytes"] for row in shards),
        "root_file_inventory": root_inventory,
        "root_file_inventory_sha256": academic._canonical_sha256(root_inventory),
        "root_file_count": len(root_inventory),
        "root_bytes": sum(row["bytes"] for row in root_inventory),
        "trainer_state_inventory": trainer_states,
        "trainer_state_inventory_sha256": academic._canonical_sha256(trainer_states),
        "trainer_state_summary": trainer_summary,
        "dcp_key_projection": _dcp_key_projection(Path(base["state_dir"]), spec),
        "full_dcp_identity": _checkpoint_full_identity(spec, include_inventory=False),
    }


def _provenance_payload(spec: Mapping[str, Any]) -> dict[str, Any]:
    gate_path = Path(spec["approval_gate_path"])
    gate_ref = _file_reference(
        gate_path,
        size=spec["approval_gate_bytes"],
        sha256=spec["approval_gate_sha256"],
        name="approval gate",
    )
    gate = academic._load_json_strict(gate_path)
    if (
        gate.get("format") != "vision_alignment_parent_gate"
        or gate.get("version") != spec["approval_gate_version"]
        or gate.get("status") != "approved"
        or gate.get("approved_by") != "rustins"
        or gate.get("approved_at") != spec["approval_gate_approved_at"]
        or gate.get("checkpoint") != str(spec["checkpoint"])
        or gate.get("global_step") != spec["step"]
        or gate.get("phase") != spec["phase"]
        or gate.get("checkpoint_config_sha256") != spec["config_sha256"]
        or gate.get("checkpoint_identity_sha256") != spec["full_checkpoint_identity_sha256"]
        or gate.get("data_contract_sha256") != spec["data_contract_sha256"]
        or gate.get("trainable_contract_sha256") != spec["trainable_contract_sha256"]
        or gate.get("promotion_bundle_path") != str(spec["promotion_bundle_path"])
        or gate.get("promotion_bundle_sha256") != spec["promotion_bundle_sha256"]
        or gate.get("metrics_artifact_sha256") != spec["promotion_bundle_sha256"]
    ):
        raise ValueError("Approved phase-boundary gate semantics differ")

    bundle_path = Path(spec["promotion_bundle_path"])
    bundle_ref = _file_reference(
        bundle_path,
        size=spec["promotion_bundle_bytes"],
        sha256=spec["promotion_bundle_sha256"],
        name="promotion bundle",
    )
    bundle = academic._load_json_strict(bundle_path)
    academic._verify_content_sha256(bundle, name="phase-boundary promotion bundle")
    if (
        bundle.get("format") != spec["promotion_bundle_format"]
        or bundle.get("version") != 1
        or bundle.get("status") != "ready_for_human_approval"
        or bundle.get("content_sha256") != spec["promotion_bundle_content_sha256"]
        or bundle.get("candidate") != spec["candidate"]
    ):
        raise ValueError("Phase-boundary promotion bundle semantics differ")
    evidence_reference = (
        bundle.get("receipts", {}).get("matched_wrong", {}).get("bridge_step500")
        if spec["key"] == "bridge_step500"
        else bundle.get("receipts", {}).get("counterfactual_outcome")
    )
    if evidence_reference != {
        "path": str(spec["full_dcp_evidence_path"]),
        "sha256": spec["full_dcp_evidence_sha256"],
    }:
        raise ValueError("Promotion bundle full-DCP evidence reference differs")

    evidence_ref = _file_reference(
        Path(spec["full_dcp_evidence_path"]),
        size=spec["full_dcp_evidence_bytes"],
        sha256=spec["full_dcp_evidence_sha256"],
        name="full-DCP evidence",
    )
    evidence_ref["content_sha256"] = spec["full_dcp_evidence_content_sha256"]

    source_path = Path(spec["source_audit_path"])
    source_ref = _file_reference(
        source_path,
        size=spec["source_audit_bytes"],
        sha256=spec["source_audit_sha256"],
        name="source audit",
    )
    source = academic._load_json_strict(source_path)
    if (
        source.get("fingerprint") != spec["source_audit_fingerprint"]
        or source.get("phase") != spec["phase"]
        or source.get("status") != "ok"
    ):
        raise ValueError("Phase-boundary source audit differs")
    source_ref["fingerprint"] = spec["source_audit_fingerprint"]

    phase_data: dict[str, Any] | None = None
    phase_data_spec = spec["phase_data_provenance"]
    if phase_data_spec is not None:
        phase_data_path = Path(phase_data_spec["path"])
        phase_data = _file_reference(
            phase_data_path,
            size=phase_data_spec["bytes"],
            sha256=phase_data_spec["sha256"],
            name="phase data provenance",
        )
        payload = academic._load_json_strict(phase_data_path)
        academic._verify_content_sha256(payload, name="phase data provenance")
        if (
            payload.get("content_sha256") != phase_data_spec["content_sha256"]
            or payload.get("phase") != spec["phase"]
            or payload.get("status") != "verified"
        ):
            raise ValueError("Phase data provenance semantics differ")
        phase_data["content_sha256"] = phase_data_spec["content_sha256"]

    return {
        "approval_gate": gate_ref,
        "promotion_bundle": {
            **bundle_ref,
            "content_sha256": spec["promotion_bundle_content_sha256"],
        },
        "full_dcp_evidence": evidence_ref,
        "full_dcp_checkpoint_identity": _checkpoint_full_identity(spec, include_inventory=False),
        "source_audit": source_ref,
        "phase_data_provenance": phase_data,
    }


def _phase_boundary_payload(
    raw_config: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    vision_alignment = raw_config.get("vision_alignment")
    artifacts = raw_config.get("artifacts")
    initialization = raw_config.get("initialization")
    launch = raw_config.get("launch")
    trainer = raw_config.get("trainer")
    train_module = raw_config.get("train_module")
    data = raw_config.get("data")
    if any(
        not isinstance(value, dict)
        for value in (
            vision_alignment,
            artifacts,
            initialization,
            launch,
            trainer,
            train_module,
            data,
        )
    ):
        raise ValueError("Phase-boundary checkpoint config structure differs")
    assert isinstance(vision_alignment, dict)
    assert isinstance(artifacts, dict)
    assert isinstance(initialization, dict)
    assert isinstance(launch, dict)
    assert isinstance(trainer, dict)
    assert isinstance(train_module, dict)
    assert isinstance(data, dict)
    current_max = trainer.get("max_duration")
    current_mixture = data.get("mixture")
    launch_git = launch.get("git")
    if (
        not isinstance(current_max, dict)
        or not isinstance(current_mixture, dict)
        or not isinstance(launch_git, dict)
    ):
        raise ValueError("Phase-boundary checkpoint config lineage is incomplete")

    next_path = Path(spec["next_config_path"])
    next_identity = _file_reference(
        next_path,
        size=spec["next_config_bytes"],
        sha256=spec["next_config_sha256"],
        name="next-phase config",
    )
    next_config = academic._load_json_strict(next_path)
    next_va = next_config.get("vision_alignment")
    next_init = next_config.get("initialization")
    next_trainer = next_config.get("trainer")
    next_train_module = next_config.get("train_module")
    next_data = next_config.get("data")
    next_launch = next_config.get("launch")
    if any(
        not isinstance(value, dict)
        for value in (
            next_va,
            next_init,
            next_trainer,
            next_train_module,
            next_data,
            next_launch,
        )
    ):
        raise ValueError("Next-phase config structure differs")
    assert isinstance(next_va, dict)
    assert isinstance(next_init, dict)
    assert isinstance(next_trainer, dict)
    assert isinstance(next_train_module, dict)
    assert isinstance(next_data, dict)
    next_max = next_trainer.get("max_duration")
    next_mixture = next_data.get("mixture")
    if not isinstance(next_max, dict) or not isinstance(next_mixture, dict):
        raise ValueError("Next-phase config duration or mixture differs")

    expected_current_initialization: dict[str, Any]
    if spec["key"] == "bridge_step500":
        expected_current_initialization = {
            "mode": "bare",
            "_CLASS_": "__main__.InitializationConfig",
        }
    else:
        expected_current_initialization = {
            "mode": "checkpoint",
            "checkpoint": str(BRIDGE_CHECKPOINT),
            "expected_parent_phase": "bridge",
            "parent_config_sha256": BOUNDARIES["bridge_step500"]["config_sha256"],
            "parent_gate_path": str(BOUNDARIES["bridge_step500"]["approval_gate_path"]),
            "parent_gate_sha256": BOUNDARIES["bridge_step500"]["approval_gate_sha256"],
            "_CLASS_": "__main__.InitializationConfig",
        }
    expected_next_initialization = {
        "mode": "checkpoint",
        "checkpoint": str(spec["checkpoint"]),
        "expected_parent_phase": spec["phase"],
        "parent_config_sha256": spec["config_sha256"],
        "parent_gate_path": str(spec["approval_gate_path"]),
        "parent_gate_sha256": spec["approval_gate_sha256"],
        "_CLASS_": "__main__.InitializationConfig",
    }
    next_save_folder = Path(spec["next_config_path"]).parents[1]
    if (
        raw_config.get("phase") != spec["phase"]
        or raw_config.get("required_run_name") != spec["run_name"]
        or raw_config.get("perception_trainability_arm") != spec["trainability_arm"]
        or vision_alignment.get("phase") != spec["phase"]
        or vision_alignment.get("lineage_id") != spec["lineage_id"]
        or vision_alignment.get("data_contract_sha256") != spec["data_contract_sha256"]
        or vision_alignment.get("trainable_contract_sha256") != spec["trainable_contract_sha256"]
        or launch_git.get("ref") != spec["training_git_revision"]
        or trainer.get("save_folder") != str(spec["checkpoint"].parent)
        or current_max.get("value") != spec["step"]
        or current_max.get("unit") != "steps"
        or train_module.get("freeze_params") != spec["current_freeze"]
        or train_module.get("train_embedding_rows") != IMAGE_TOKEN_ROWS
        or train_module.get("max_sequence_length") != 2_560
        or data.get("sequence_length") != 2_560
        or data.get("max_crops") != 8
        or data.get("message_format") != "document"
        or current_mixture.get("phase") != spec["phase"]
        or data.get("source_audit_path") != str(spec["source_audit_path"])
        or data.get("source_audit_fingerprint") != spec["source_audit_fingerprint"]
        or initialization != expected_current_initialization
        or artifacts.get("tokenizer_id") != TOKENIZER_IDENTIFIER
        or artifacts.get("tokenizer_revision") != TOKENIZER_REVISION
        or artifacts.get("tokenizer_fingerprint") != TOKENIZER_FINGERPRINT
        or next_config.get("phase") != spec["next_phase"]
        or next_config.get("required_run_name") != spec["next_run_name"]
        or next_config.get("perception_trainability_arm") != spec["next_trainability_arm"]
        or next_va.get("phase") != spec["next_phase"]
        or next_va.get("lineage_id") != spec["next_lineage_id"]
        or next_init != expected_next_initialization
        or next_trainer.get("save_folder") != str(next_save_folder)
        or next_trainer.get("load_path") != str(spec["checkpoint"])
        or next_trainer.get("load_strategy") != "always"
        or next_trainer.get("load_optim_state") is not False
        or next_trainer.get("load_trainer_state") is not False
        or next_max.get("value") != spec["next_max_steps"]
        or next_max.get("unit") != "steps"
        or next_train_module.get("freeze_params") != spec["next_freeze"]
        or next_train_module.get("train_embedding_rows") != IMAGE_TOKEN_ROWS
        or next_train_module.get("max_sequence_length") != spec["next_sequence_length"]
        or next_data.get("sequence_length") != spec["next_sequence_length"]
        or next_data.get("max_crops") != 8
        or next_data.get("message_format") != "document"
        or next_mixture.get("phase") != spec["next_phase"]
    ):
        raise ValueError("Phase-boundary lineage or freeze semantics differ")
    if spec["key"] == "bridge_step500":
        if (
            trainer.get("load_path") is not None
            or vision_alignment.get("parent_checkpoint") is not None
        ):
            raise ValueError("Bridge boundary unexpectedly declares a phase parent")
    elif (
        trainer.get("load_path") != str(BRIDGE_CHECKPOINT)
        or trainer.get("load_strategy") != "always"
        or trainer.get("load_optim_state") is not False
        or trainer.get("load_trainer_state") is not False
        or vision_alignment.get("parent_checkpoint") != str(BRIDGE_CHECKPOINT)
        or vision_alignment.get("parent_config_sha256")
        != BOUNDARIES["bridge_step500"]["config_sha256"]
        or vision_alignment.get("parent_gate_sha256")
        != BOUNDARIES["bridge_step500"]["approval_gate_sha256"]
        or data.get("perception_provenance_path") != str(spec["phase_data_provenance"]["path"])
        or data.get("perception_provenance_sha256") != spec["phase_data_provenance"]["sha256"]
    ):
        raise ValueError("Perception-treatment parent lineage differs")

    return {
        "boundary_key": spec["key"],
        "role": spec["role"],
        "transition": spec["transition"],
        "terminal_phase_checkpoint": True,
        "completed_phase": {
            "phase": spec["phase"],
            "lineage_id": spec["lineage_id"],
            "global_step": spec["step"],
            "maximum_steps": spec["step"],
            "training_git_revision": spec["training_git_revision"],
            "run_name": spec["run_name"],
            "trainability_arm": spec["trainability_arm"],
            "freeze_params": list(spec["current_freeze"]),
            "train_embedding_rows": IMAGE_TOKEN_ROWS,
            "sequence_length": 2_560,
            "data_contract_sha256": spec["data_contract_sha256"],
            "trainable_contract_sha256": spec["trainable_contract_sha256"],
            "initialization": expected_current_initialization,
        },
        "next_phase": {
            "phase": spec["next_phase"],
            "lineage_id": spec["next_lineage_id"],
            "config": next_identity,
            "loads_exact_boundary_checkpoint": True,
            "load_optimizer_state": False,
            "load_trainer_state": False,
            "freeze_params": list(spec["next_freeze"]),
            "train_embedding_rows": IMAGE_TOKEN_ROWS,
            "sequence_length": spec["next_sequence_length"],
        },
        "freeze_transition": {
            "newly_trainable": list(spec["newly_trainable"]),
            "still_frozen": list(spec["still_frozen"]),
            "exact_config_semantics": True,
        },
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
        raise ValueError("Phase-boundary evaluation requires the exact frozen academic manifest")
    return reference


def _evaluation_tokenizer(
    raw_config: Mapping[str, Any], cache_override: str | None
) -> tuple[Any, Any]:
    artifacts = raw_config.get("artifacts")
    model = raw_config.get("model")
    collator = raw_config.get("collator")
    if (
        not isinstance(artifacts, dict)
        or artifacts.get("tokenizer_id") != TOKENIZER_IDENTIFIER
        or artifacts.get("tokenizer_revision") != TOKENIZER_REVISION
        or artifacts.get("tokenizer_fingerprint") != TOKENIZER_FINGERPRINT
        or not isinstance(model, dict)
        or not isinstance(model.get("lm"), dict)
        or type(model["lm"].get("vocab_size")) is not int
        or not isinstance(collator, dict)
    ):
        raise ValueError("Phase-boundary tokenizer or vocabulary config differs")
    cache_dir = cache_override or artifacts.get("hf_cache_dir")
    if not isinstance(cache_dir, str) or not cache_dir:
        raise ValueError("A local tokenizer cache is required")
    tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
        identifier=TOKENIZER_IDENTIFIER,
        revision=TOKENIZER_REVISION,
        expected_fingerprint=TOKENIZER_FINGERPRINT,
        cache_dir=cache_dir,
        model_vocab_size=model["lm"]["vocab_size"],
    )
    if (
        tokenizer.eos_token_id != academic.EXPECTED_EOS_TOKEN_ID
        or tokenizer.pad_token_id != academic.EXPECTED_PAD_TOKEN_ID
        or academic._canonical_sha256(token_ids.as_config_dict())
        != academic.EXPECTED_MOLMO2_TOKEN_IDS_SHA256
        or model.get("image_patch_token_id") != token_ids.im_patch_id
        or collator.get("pad_token_id") != tokenizer.pad_token_id
    ):
        raise ValueError("Phase-boundary evaluation tokenizer token identities differ")
    return tokenizer, token_ids


def _tokenizer_payload(tokenizer: Any, token_ids: Any) -> dict[str, Any]:
    return {
        "usage": "training_and_evaluation_exact_pin",
        "identifier": TOKENIZER_IDENTIFIER,
        "revision": TOKENIZER_REVISION,
        "fingerprint": TOKENIZER_FINGERPRINT,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "token_ids": token_ids.as_config_dict(),
        "token_ids_sha256": academic._canonical_sha256(token_ids.as_config_dict()),
    }


def _interpretation_limits(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "descriptive_only": True,
        "official_leaderboard_submission": False,
        "boundary_role": spec["role"],
        "causal_phase_effect_claim": "none; adjacent phases change trainability and data",
        "image_overlap_field_semantics": (
            "alignment_train_image_overlap and shuffled_alignment_train_image_overlap refer to "
            "the frozen full vision-alignment train-union inventory, not phase-specific exposure"
        ),
        "phase_specific_contamination_claim": "none",
        "inherited_protocol_claim_scope": (
            "the unchanged frozen protocol's joint-step-selection claim records the panel's "
            "original purpose; this receipt uses it only for phase-boundary diagnostics"
        ),
        "training_sequence_cap_comparability": (
            "all 9,216 control prompts are exactly rederived, have maximum input length 1,246, "
            "and require at most 1,259 tokens including generation; neither exceeds the 2,560-"
            "token bridge/perception training cap"
        ),
    }


def _validate_sequence_cap(tasks: Mapping[str, Any], tokenizer: Any) -> None:
    input_lengths = []
    required_lengths = []
    for task in academic.DEFAULT_TASKS:
        for row in tasks[task]["examples"]:
            example = academic._receipt_example_from_row(task, row)
            image_ids = academic.build_image_token_ids(*row["image_grid_signature"])
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
                        f"Phase-boundary {task}/{example.example_id}/{control} input-token "
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
        raise ValueError("Phase-boundary sequence-cap comparability audit differs")


def _native_checkpoint_load_coverage(train_module: Any, state_dir: Path) -> dict[str, Any]:
    metadata = FileSystemReader(state_dir).read_metadata()
    checkpoint_keys = set(metadata.state_dict_metadata)
    required_methods = (
        "_get_model_state_dict_for_eval_load",
        "_resolve_model_checkpoint_key",
        "_frozen_checkpoint_model_param_state_dict_for_load",
        "_frozen_checkpoint_param_state_dict_for_load",
        "_persistent_model_buffer_state_dict",
    )
    methods: dict[str, Callable[..., Any]] = {}
    for name in required_methods:
        method = getattr(train_module, name, None)
        if not callable(method):
            raise TypeError("Native checkpoint load lacks a required complete-coverage API")
        methods[name] = method
    eval_state = methods["_get_model_state_dict_for_eval_load"](metadata)
    frozen_parameters = methods["_frozen_checkpoint_model_param_state_dict_for_load"](
        checkpoint_keys
    )
    frozen_tensors = methods["_frozen_checkpoint_param_state_dict_for_load"](checkpoint_keys)
    persistent_buffers = methods["_persistent_model_buffer_state_dict"]()
    if set(frozen_parameters) != set(frozen_tensors):
        raise RuntimeError("Native frozen-parameter and frozen-tensor load keys differ")
    if missing_buffers := sorted(set(persistent_buffers) - checkpoint_keys):
        raise RuntimeError(f"Native checkpoint is missing model buffers: {missing_buffers[:10]}")
    for label, state in (
        ("eval", eval_state),
        ("frozen", frozen_tensors),
        ("buffer", persistent_buffers),
    ):
        for key, target in state.items():
            tensor_metadata = metadata.state_dict_metadata.get(key)
            if not isinstance(tensor_metadata, TensorStorageMetadata):
                raise TypeError(f"Native {label} target {key!r} lacks tensor metadata")
            if tuple(target.size()) != tuple(tensor_metadata.size):
                raise RuntimeError(f"Native {label} target {key!r} shape differs")
            if target.numel() != math.prod(int(size) for size in tensor_metadata.size):
                raise RuntimeError(f"Native {label} target {key!r} numel differs")

    model_parts = getattr(train_module, "model_parts", None)
    if not isinstance(model_parts, Sequence) or not model_parts:
        raise RuntimeError("Native checkpoint load does not expose model_parts")
    frozen_by_parameter: dict[int, list[str]] = {}
    for key, parameter in frozen_parameters.items():
        frozen_by_parameter.setdefault(id(parameter), []).append(key)
    parameter_names: dict[int, list[str]] = {}
    parameter_by_id: dict[int, Any] = {}
    for part_index, model_part in enumerate(model_parts):
        for name, parameter in model_part.named_parameters():
            parameter_names.setdefault(id(parameter), []).append(f"part{part_index}.{name}")
            parameter_by_id[id(parameter)] = parameter
    if not parameter_by_id:
        raise RuntimeError("Native checkpoint load model has no parameters")
    if orphaned := sorted(
        key for key, parameter in frozen_parameters.items() if id(parameter) not in parameter_by_id
    ):
        raise RuntimeError(f"Native frozen targets are absent from model_parts: {orphaned[:10]}")

    covered_keys: set[str] = set()
    parameter_ids_by_key: dict[str, set[int]] = {}
    assignments: list[dict[str, Any]] = []
    missing_parameters: list[str] = []
    resolver = methods["_resolve_model_checkpoint_key"]
    for parameter_id, parameter in parameter_by_id.items():
        names = parameter_names[parameter_id]
        resolved = {
            key
            for name in names
            if (key := resolver(name.split(".", 1)[1], checkpoint_keys)) is not None
        } & set(eval_state)
        frozen = set(frozen_by_parameter.get(id(parameter), ()))
        authoritative = resolved | frozen
        if not authoritative:
            missing_parameters.extend(names)
            continue
        if len(authoritative) != 1:
            raise RuntimeError(
                f"Native model parameter {names} resolves ambiguously to {sorted(authoritative)}"
            )
        key = next(iter(authoritative))
        parameter_ids_by_key.setdefault(key, set()).add(parameter_id)
        covered_keys.add(key)
        assignments.append(
            {"parameter_names": sorted(names), "checkpoint_keys": sorted(authoritative)}
        )
    if missing_parameters:
        raise RuntimeError(f"Native checkpoint misses model parameters: {missing_parameters[:10]}")
    if multiply_mapped := sorted(
        key for key, parameter_ids in parameter_ids_by_key.items() if len(parameter_ids) > 1
    ):
        raise RuntimeError(f"Native keys map to multiple parameters: {multiply_mapped[:10]}")
    prepared = set(eval_state) | set(frozen_tensors)
    if unused_prepared := sorted(prepared - covered_keys):
        raise RuntimeError(f"Native prepared model keys are unused: {unused_prepared[:10]}")

    def model_bearing(key: str) -> bool:
        return key.endswith(".main") or key.startswith(("frozen_model.", "model_buffer.", "model."))

    def logical_name(key: str) -> str:
        if key.startswith("frozen_model."):
            return key.removeprefix("frozen_model.")
        if key.endswith(".main"):
            return key.removesuffix(".main").removeprefix("module.")
        return key

    consumed = covered_keys | set(persistent_buffers)
    unused_model = {key for key in checkpoint_keys - consumed if model_bearing(key)}
    main_names = {logical_name(key) for key in covered_keys if key.endswith(".main")}
    shadowed = {
        key
        for key in unused_model
        if key.startswith("frozen_model.") and logical_name(key) in main_names
    }
    if unexpected := sorted(unused_model - shadowed):
        raise RuntimeError(f"Native checkpoint has unused model keys: {unexpected[:10]}")
    report = {
        "complete": True,
        "checkpoint_key_count": len(checkpoint_keys),
        "model_parameter_count": len(parameter_by_id),
        "model_parameter_checkpoint_key_count": len(covered_keys),
        "model_parameter_checkpoint_keys_sha256": academic._canonical_sha256(sorted(covered_keys)),
        "model_parameter_assignments_sha256": academic._canonical_sha256(
            sorted(assignments, key=lambda assignment: assignment["parameter_names"])
        ),
        "eval_state_key_count": len(eval_state),
        "frozen_state_key_count": len(frozen_parameters),
        "persistent_buffer_count": len(persistent_buffers),
        "persistent_buffer_keys_sha256": academic._canonical_sha256(sorted(persistent_buffers)),
        "shadowed_frozen_key_count": len(shadowed),
        "shadowed_frozen_keys_sha256": academic._canonical_sha256(sorted(shadowed)),
        "unused_model_bearing_key_count": 0,
        "prepared_load_key_count": len(prepared | set(persistent_buffers)),
    }
    report["sha256"] = academic._canonical_sha256(report)
    return report


def _native_checkpoint_load_coverage_distributed(
    train_module: Any, state_dir: Path
) -> dict[str, Any]:
    try:
        local: dict[str, Any] = {
            "ok": True,
            "report": _native_checkpoint_load_coverage(train_module, state_dir),
        }
    except Exception as error:  # noqa: BLE001 - propagate every rank's failure.
        local = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    failures = [
        f"rank {rank}: {packet.get('error')}"
        if isinstance(packet, Mapping)
        else f"rank {rank}: malformed report"
        for rank, packet in enumerate(gathered)
        if not isinstance(packet, Mapping) or packet.get("ok") is not True
    ]
    if failures:
        raise RuntimeError(f"Native checkpoint load coverage failed: {failures}")
    reports = [packet["report"] for packet in gathered]
    if any(report != reports[0] for report in reports[1:]):
        raise RuntimeError("Native checkpoint load coverage differs across ranks")
    return reports[0]


def _model_load_payload(
    coverage: dict[str, Any],
    train_module: Any,
    state_dir: Path,
    spec: Mapping[str, Any],
    *,
    checkpoint_load_threads: int,
) -> dict[str, Any]:
    coverage["load_completed"] = True
    coverage["sha256"] = academic._canonical_sha256(
        {field: value for field, value in coverage.items() if field != "sha256"}
    )
    expected_coverage = spec["load_coverage"]
    if coverage != expected_coverage:
        differences = []
        for field in sorted(set(coverage) | set(expected_coverage)):
            expected = repr(expected_coverage[field]) if field in expected_coverage else "<missing>"
            actual = repr(coverage[field]) if field in coverage else "<missing>"
            if expected != actual:
                differences.append(f"{field}: expected={expected}, actual={actual}")
        raise RuntimeError(
            "Phase-boundary native checkpoint-load coverage differs; " + "; ".join(differences)
        )
    local = {
        "rank": dist.get_rank(),
        "coverage_sha256": coverage["sha256"],
        "remaining_meta_parameter_count": sum(
            int(parameter.is_meta)
            for model_part in train_module.model_parts
            for parameter in model_part.parameters()
        ),
    }
    completions: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(completions, local)
    expected_completion = [
        {
            "rank": rank,
            "coverage_sha256": coverage["sha256"],
            "remaining_meta_parameter_count": 0,
        }
        for rank in range(academic.EP_DEGREE)
    ]
    if completions != expected_completion:
        raise RuntimeError("Phase-boundary model load did not complete identically on all ranks")
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
        "coverage": coverage,
        "all_rank_completion": completions,
    }


def _validate_live_full_dcp(spec: Mapping[str, Any], *, hash_workers: int) -> None:
    if hash_workers <= 0:
        raise ValueError("Checkpoint hash workers must be positive")
    identity = _checkpoint_full_identity(spec, include_inventory=True)
    inventory = identity["state_file_inventory"]
    root = Path(identity["root"]).resolve()
    state_dir = Path(identity["state_dir"]).resolve()
    paths = [root / row["path"] for row in inventory]
    if sorted(state_dir.iterdir()) != sorted(paths):
        raise ValueError("Live full-DCP directory entries differ from approved evidence")
    before = {path: path.stat() for path in paths}
    if any(not stat.S_ISREG(metadata.st_mode) for metadata in before.values()):
        raise ValueError("Live full-DCP inventory contains a non-regular file")
    with ThreadPoolExecutor(max_workers=min(hash_workers, len(paths))) as executor:
        hashes = list(executor.map(academic._sha256_file_stable, paths))
    actual = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": digest,
        }
        for path, digest in zip(paths, hashes, strict=True)
    ]
    if actual != inventory or sorted(state_dir.iterdir()) != sorted(paths):
        raise ValueError("Live full-DCP content differs from approved evidence")
    for path, metadata in before.items():
        current = path.stat()
        if (
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
            current.st_ino,
            current.st_dev,
        ) != (
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
            metadata.st_ino,
            metadata.st_dev,
        ):
            raise ValueError("Live full-DCP content changed during verification")


def _validate_live_full_dcp_distributed(
    spec: Mapping[str, Any], *, hash_workers: int, phase: str
) -> None:
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            log.info("Verifying all full-DCP hashes (%s) for %s", phase, spec["checkpoint"])
            _validate_live_full_dcp(spec, hash_workers=hash_workers)
            packet[0] = {"ok": True}
        except Exception as error:  # noqa: BLE001 - broadcast rank-zero verification failure.
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Full-DCP {phase} verification failed: {detail}")
    dist.barrier()


def _validate_checkpoint_payload(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
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
        "boundary_key",
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
        "full_dcp_identity",
    )
    checkpoint = academic._exact_mapping(value, fields, name="phase-boundary checkpoint")
    spec = _boundary_spec(Path(str(checkpoint["checkpoint"])))
    if (
        checkpoint["boundary_key"] != spec["key"]
        or checkpoint["config"]
        != {
            "path": str(spec["checkpoint"] / "config.json"),
            "bytes": spec["config_bytes"],
            "sha256": spec["config_sha256"],
        }
        or checkpoint["checkpoint_marker"]
        != {
            "path": str(spec["checkpoint"] / ".metadata.json"),
            "bytes": 40,
            "sha256": spec["marker_sha256"],
        }
        or checkpoint["state_dir"] != str(spec["checkpoint"] / "model_and_optim")
        or checkpoint["dcp_metadata"]
        != {
            "path": str(spec["checkpoint"] / "model_and_optim/.metadata"),
            "bytes": spec["dcp_metadata_bytes"],
            "sha256": spec["dcp_metadata_sha256"],
        }
        or checkpoint["state_file_count"] != spec["state_file_count"]
        or checkpoint["state_bytes"] != spec["state_bytes"]
        or checkpoint["state_file_inventory_sha256"] != spec["state_file_inventory_sha256"]
        or checkpoint["identity_scope"] != _checkpoint_identity_scope()
        or checkpoint["distcp_shard_count"] != spec["distcp_shard_count"]
        or checkpoint["distcp_shard_bytes"] != spec["distcp_shard_bytes"]
        or checkpoint["root_file_count"] != spec["root_file_count"]
        or checkpoint["root_bytes"] != spec["root_bytes"]
        or checkpoint["root_file_inventory_sha256"] != spec["root_file_inventory_sha256"]
        or checkpoint["trainer_state_inventory_sha256"] != spec["trainer_state_inventory_sha256"]
        or checkpoint["trainer_state_summary"] != _expected_trainer_state_summary(spec)
        or checkpoint["dcp_key_projection"] != spec["dcp_projection"]
        or checkpoint["full_dcp_identity"]
        != _checkpoint_full_identity(spec, include_inventory=False)
    ):
        raise ValueError("Serialized phase-boundary checkpoint identity differs")
    for inventory, expected_count, expected_bytes, expected_sha in (
        (
            checkpoint["state_file_inventory"],
            spec["state_file_count"],
            spec["state_bytes"],
            spec["state_file_inventory_sha256"],
        ),
        (
            checkpoint["root_file_inventory"],
            spec["root_file_count"],
            spec["root_bytes"],
            spec["root_file_inventory_sha256"],
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
            or sum(row["bytes"] for row in inventory) != expected_bytes
            or academic._canonical_sha256(inventory) != expected_sha
        ):
            raise ValueError("Serialized phase-boundary file inventory differs")
    trainers = checkpoint["trainer_state_inventory"]
    if (
        not isinstance(trainers, list)
        or len(trainers) != spec["trainer_state_count"]
        or any(
            not isinstance(row, dict)
            or set(row) != {"name", "bytes", "sha256"}
            or row["name"] != f"train/rank{rank}.pt"
            or type(row["bytes"]) is not int
            or row["bytes"] <= 0
            or not isinstance(row["sha256"], str)
            or _SHA256_RE.fullmatch(row["sha256"]) is None
            for rank, row in enumerate(trainers)
        )
        or sum(row["bytes"] for row in trainers) != spec["trainer_state_bytes"]
        or academic._canonical_sha256(trainers) != spec["trainer_state_inventory_sha256"]
    ):
        raise ValueError("Serialized phase-boundary trainer-state identity differs")
    return checkpoint, spec


def _validate_model_load(
    value: Any, checkpoint: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    fields = (
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
    )
    load = academic._exact_mapping(value, fields, name="phase-boundary model load")
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
        or load["coverage"] != spec["load_coverage"]
    ):
        raise ValueError("Phase-boundary model-load declaration differs")
    expected_completions = [
        {
            "rank": rank,
            "coverage_sha256": spec["load_coverage"]["sha256"],
            "remaining_meta_parameter_count": 0,
        }
        for rank in range(academic.EP_DEGREE)
    ]
    if load["all_rank_completion"] != expected_completions:
        raise ValueError("Phase-boundary all-rank load completion differs")
    return load


def _load_manifest(
    reference: Any, *, verify_live_sources: bool
) -> tuple[dict[str, Any], dict[str, dict[str, academic.AcademicExample]] | None]:
    value = academic._exact_mapping(
        reference,
        ("path", "bytes", "sha256", "content_sha256", "partial", "panel_status", "builder_git"),
        name="phase-boundary manifest reference",
    )
    base = {
        field: value[field]
        for field in ("path", "bytes", "sha256", "content_sha256", "partial", "panel_status")
    }
    manifest, loaded = academic._load_receipt_manifest(
        base, verify_live_sources=verify_live_sources
    )
    if value != _manifest_reference(manifest, academic._file_identity(Path(value["path"]))):
        raise ValueError("Phase-boundary receipt manifest reference differs")
    return manifest, loaded


def _artifact_policy() -> dict[str, bool]:
    return {
        "descriptive_only": True,
        "promotion_eligible": False,
        "phase_boundary_comparison_evidence": True,
        "checkpoint_selection_evidence": False,
        "causal_phase_effect_evidence": False,
    }


def _validate_receipt_payload(
    receipt: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    loaded: Mapping[str, Mapping[str, academic.AcademicExample]] | None,
    tokenizer: Any,
    token_ids: Any,
) -> dict[str, Any]:
    fields = (
        "schema_version",
        "format",
        "protocol_name",
        "created_at",
        "launch_git",
        "implementation",
        "manifest",
        "checkpoint",
        "phase_boundary",
        "provenance",
        "model_load",
        "artifact_policy",
        "interpretation_limits",
        "tokenizer",
        "protocol",
        "tasks",
        "content_sha256",
    )
    value = academic._exact_mapping(receipt, fields, name="phase-boundary academic receipt")
    academic._verify_content_sha256(value, name="phase-boundary academic receipt")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["format"] != RECEIPT_FORMAT
        or value["protocol_name"] != PROTOCOL_NAME
    ):
        raise ValueError("Phase-boundary academic receipt envelope differs")
    academic._validate_timestamp(value["created_at"], name="phase-boundary receipt created_at")
    launch_git = academic._exact_mapping(
        value["launch_git"], ("revision", "dirty"), name="phase-boundary launch Git"
    )
    academic._validate_git_identity(launch_git)
    if value["implementation"] != _implementation_identity():
        raise ValueError("Phase-boundary evaluator implementation differs")
    if value["manifest"] != _manifest_reference(
        manifest, academic._file_identity(Path(value["manifest"]["path"]))
    ):
        raise ValueError("Phase-boundary receipt manifest differs")
    checkpoint, spec = _validate_checkpoint_payload(value["checkpoint"])
    raw_config = academic._load_json_strict(Path(checkpoint["config"]["path"]))
    if value["phase_boundary"] != _phase_boundary_payload(raw_config, spec):
        raise ValueError("Phase-boundary receipt lineage differs")
    if value["provenance"] != _provenance_payload(spec):
        raise ValueError("Phase-boundary receipt provenance differs")
    _validate_model_load(value["model_load"], checkpoint, spec)
    if value["artifact_policy"] != _artifact_policy():
        raise ValueError("Phase-boundary receipt artifact policy differs")
    if value["interpretation_limits"] != _interpretation_limits(spec):
        raise ValueError("Phase-boundary receipt interpretation limits differ")
    if value["tokenizer"] != _tokenizer_payload(tokenizer, token_ids):
        raise ValueError("Phase-boundary receipt tokenizer differs")
    if value["protocol"] != academic._protocol_payload(manifest):
        raise ValueError("Phase-boundary receipt frozen benchmark protocol differs")
    academic._validate_receipt_tasks(
        value["tasks"],
        manifest=manifest,
        loaded=loaded,
        tokenizer=tokenizer,
        text_vocab_size=min(token_ids.image_token_ids),
    )
    _validate_sequence_cap(value["tasks"], tokenizer)
    return value


def validate_phase_boundary_receipt(
    path: str | Path,
    expected_sha256: str,
    *,
    verify_live: bool = True,
    hf_cache: str | None = None,
    checkpoint_hash_workers: int = 16,
) -> dict[str, Any]:
    """Strictly reload and rederive one phase-boundary academic receipt.

    :param path: Receipt path.
    :param expected_sha256: Independently supplied raw receipt SHA-256.
    :param verify_live: Rehash benchmark sources and every live DCP shard when true.
    :param hf_cache: Optional local cache containing the pinned tokenizer.
    :param checkpoint_hash_workers: Parallel workers for live DCP SHA-256 verification.
    :returns: The validated receipt.
    """
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError("Expected receipt SHA-256 must be lowercase hex")
    receipt_path = Path(path).expanduser().resolve()
    if academic._file_identity(receipt_path)["sha256"] != expected_sha256:
        raise ValueError("Phase-boundary receipt raw SHA-256 differs")
    receipt = academic._load_json_strict(receipt_path)
    manifest, loaded = _load_manifest(receipt.get("manifest"), verify_live_sources=verify_live)
    checkpoint, spec = _validate_checkpoint_payload(receipt.get("checkpoint"))
    raw_config = academic._load_json_strict(Path(checkpoint["config"]["path"]))
    tokenizer, token_ids = _evaluation_tokenizer(raw_config, hf_cache)
    _validate_receipt_payload(
        receipt,
        manifest=manifest,
        loaded=loaded,
        tokenizer=tokenizer,
        token_ids=token_ids,
    )
    if verify_live:
        if _checkpoint_identity(Path(checkpoint["checkpoint"])) != checkpoint:
            raise ValueError("Live phase-boundary checkpoint identity differs")
        _validate_live_full_dcp(spec, hash_workers=checkpoint_hash_workers)
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
        raise ValueError("Phase-boundary comparison requires the frozen 512-per-task panel")


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    academic._validate_runtime_args(args)
    if args.checkpoint_hash_workers <= 0:
        raise ValueError("--checkpoint-hash-workers must be positive")
    manifest, loaded, manifest_identity = academic._validate_manifest_and_load_examples(
        Path(args.manifest)
    )
    _validate_confirmatory_manifest(manifest)
    manifest_reference = _manifest_reference(manifest, manifest_identity)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    spec = _boundary_spec(checkpoint_path)
    opening_checkpoint = _checkpoint_identity(checkpoint_path)
    raw_config = academic._load_json_strict(checkpoint_path / "config.json")
    phase_boundary = _phase_boundary_payload(raw_config, spec)
    provenance = _provenance_payload(spec)
    launch_git = academic._git_revision()
    academic._validate_git_identity(launch_git)
    implementation = _implementation_identity()

    artifacts = raw_config.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Phase-boundary config lacks tokenizer artifacts")
    cache_dir = args.hf_cache or artifacts.get("hf_cache_dir")
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
        _validate_live_full_dcp_distributed(
            spec, hash_workers=args.checkpoint_hash_workers, phase="opening"
        )
        model, module_config, checkpoint_kind = academic._build_model_and_module_config(
            raw_config,
            ep_degree=academic.EP_DEGREE,
            max_sequence_length=args.max_sequence_length,
            rank_batch_size=args.max_sequence_length,
            ep_path=ExpertParallelPath.sync_1d,
        )
        if checkpoint_kind != "multimodal_stage1":
            raise ValueError("Phase boundary is not a native multimodal checkpoint")
        train_module = module_config.build(model, eval_only=True)
        state_dir = Path(academic._checkpoint_state_dir(checkpoint_path)).resolve()
        load_coverage = _native_checkpoint_load_coverage_distributed(train_module, state_dir)
        train_module.load_state_dict_direct(
            state_dir,
            process_group=dist.group.WORLD,
            thread_count=args.checkpoint_load_threads,
            load_optim_state=False,
        )
        academic._set_model_parts_eval(train_module)
        model_load = _model_load_payload(
            load_coverage,
            train_module,
            state_dir,
            spec,
            checkpoint_load_threads=args.checkpoint_load_threads,
        )
        tokenizer, token_ids = _evaluation_tokenizer(raw_config, cache_dir)
        if (
            academic._answer_token_coverage(loaded, tokenizer)
            != academic.EXPECTED_ANSWER_TOKEN_COVERAGE
        ):
            raise ValueError("Frozen answer-token coverage differs")
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
        if _checkpoint_identity(checkpoint_path) != opening_checkpoint:
            raise ValueError("Phase-boundary checkpoint identity changed during evaluation")
        if _phase_boundary_payload(raw_config, spec) != phase_boundary:
            raise ValueError("Phase-boundary lineage changed during evaluation")
        if _provenance_payload(spec) != provenance:
            raise ValueError("Phase-boundary provenance changed during evaluation")
        _validate_live_full_dcp_distributed(
            spec, hash_workers=args.checkpoint_hash_workers, phase="closing"
        )
        if _implementation_identity() != implementation:
            raise ValueError("Phase-boundary evaluator implementation changed")
        closing_git = academic._git_revision()
        academic._validate_git_identity(closing_git)
        if closing_git != launch_git:
            raise ValueError("Launch Git identity changed during evaluation")

        payload = {
            "schema_version": SCHEMA_VERSION,
            "format": RECEIPT_FORMAT,
            "protocol_name": PROTOCOL_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "launch_git": launch_git,
            "implementation": implementation,
            "manifest": manifest_reference,
            "checkpoint": opening_checkpoint,
            "phase_boundary": phase_boundary,
            "provenance": provenance,
            "model_load": model_load,
            "artifact_policy": _artifact_policy(),
            "interpretation_limits": _interpretation_limits(spec),
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
                validate_phase_boundary_receipt(
                    output,
                    raw_sha256,
                    verify_live=False,
                    hf_cache=cache_dir,
                )
                publication[0] = {"ok": True, "sha256": raw_sha256}
            except Exception as error:  # noqa: BLE001 - broadcast persistence failure.
                publication[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(publication, src=0)
        result = publication[0]
        if not isinstance(result, Mapping) or result.get("ok") is not True:
            detail = result.get("error") if isinstance(result, Mapping) else repr(result)
            raise RuntimeError(f"Could not persist phase-boundary receipt: {detail}")
        if get_rank() == 0:
            log.info("Wrote phase-boundary receipt %s (sha256=%s)", args.output, result["sha256"])
        return receipt
    finally:
        teardown_training_environment()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate one exact phase boundary.")
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
    evaluate.add_argument("--checkpoint-hash-workers", type=int, default=16)

    validate = commands.add_parser("validate-receipt", help="Strictly rederive one receipt.")
    validate.add_argument("--receipt", required=True)
    validate.add_argument("--expected-sha256", required=True)
    validate.add_argument("--hf-cache")
    validate.add_argument("--checkpoint-hash-workers", type=int, default=16)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parser().parse_args()
    if args.command == "evaluate":
        _evaluate(args)
        return
    if args.command == "validate-receipt":
        validate_phase_boundary_receipt(
            args.receipt,
            args.expected_sha256,
            verify_live=True,
            hf_cache=args.hf_cache,
            checkpoint_hash_workers=args.checkpoint_hash_workers,
        )
        log.info("Validated phase-boundary external-academic receipt %s", args.receipt)
        return
    raise AssertionError(f"Unknown command {args.command!r}")


if __name__ == "__main__":
    main()
