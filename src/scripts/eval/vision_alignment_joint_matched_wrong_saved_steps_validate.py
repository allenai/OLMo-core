"""Validate exact joint saved-endpoint matched/wrong evaluator receipts.

This is the independent receipt contract for the descriptive saved-endpoint evaluator.  It
deliberately reuses the reviewed V1 row, metric, pairing, tokenizer, projection, and native-text
validators while defining a separate V2 envelope for steps 12000, 14400, and 16000.  In
particular, step14400 remains a retained ephemeral checkpoint; validation never relabels it as a
permanent endpoint.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_local_module(name: str, filename: str) -> ModuleType:
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load saved-endpoint validation dependency {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_v1 = _load_local_module(
    "_vision_alignment_joint_matched_wrong_v1_for_saved_steps_validation",
    "vision_alignment_joint_matched_wrong_compare.py",
)

FORMAT = "vision_alignment_joint_matched_wrong_receipt"
VERSION = 2
STATUS = "valid"
STEPS = (12000, 14400, 16000)
ENDPOINT_CONTRACT = "vision-alignment-joint-saved-endpoints-v1"
ENDPOINT_SET_VERSION = "joint-saved-endpoints-v1"
PROTOCOL_NAME = "vision-alignment-joint-native-matched-wrong-saved-endpoints-v2"
PRIMARY_STATISTIC = "all-response wrong-minus-correct CE gap at one exact saved checkpoint"

EXPECTED_CHECKPOINT_BASE = Path(
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/checkpoints/"
    "vision-alignment-joint-v1"
)
EXPECTED_PAIRING_MANIFEST_PATH = Path(
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/"
    "joint-v1-matched-wrong-v1/pairing-manifest-12f5623a.json"
)
EXPECTED_PAIRING_MANIFEST_SHA256 = (
    "24a768b89ca0b73386362c3aa6db9afb27b36318e159766ac3a0cb62cf978739"
)
EXPECTED_PAIRING_MANIFEST_CONTENT_SHA256 = (
    "f021873ba46192f62eab44b27d9f198db5b36081b548326b1547b25020b457de"
)
EXPECTED_V1_COMPARATOR_SHA256 = "56b3526293332b70adf73977d2edbf050a6d507b2ff7b9ef9a3ddc2bd75f53e1"
EXPECTED_SHARED_IMPLEMENTATION_SHA256 = {
    "scoring_engine_": "3daf99a9996de0c3deaf62653dc43fa47eaca915785bda0b94cc1d897b25e058",
    "perception_helper_": "8389ef3c8ce4f06b6caa2376e12c1e0a09067cc0f26677d840fe9bb470a4bc64",
    "bridge_helper_": "fb7c7192e8cf92ccba83cbde51b3c7b8a82d37ca89377a894f27ff37ba9ebbdf",
    "pairing_implementation_": ("dcd2b7c5538ca11eb1bfcafd2fbbbe028a037519266f050602d73db9e76cd33f"),
    "training_contract_": "b08d291a699008d549842552c438aa1560ac7143923575eade21c2ce155d3698",
}
EXPECTED_MODEL_AND_OPTIM_BYTES = 384_970_228_158
EXPECTED_TRAINER_STATE_BYTES = 281_984
EXPECTED_CONFIG_BYTES = 33_727
EXPECTED_TRAINER_ERRORS = (2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
EXPECTED_WANDB_RUN_ID = "4gxnu6we"
EXPECTED_STORAGE_CLASS = {
    12000: "scheduled_permanent",
    14400: "retained_ephemeral",
    16000: "scheduled_permanent",
}
EXPECTED_MARKER = {
    12000: {"ephemeral": False, "version": "2.5.0"},
    14400: {"ephemeral": True, "version": "2.5.0"},
    16000: {"ephemeral": False, "version": "2.5.0"},
}
EXPECTED_MARKER_SHA256 = {
    12000: "77dfdeec42fe7990f4b3b9c4eeecd480edcf5066c110603b115920af38423d03",
    14400: "3c4b070a507487454f081c1bc4eac4a68ffa3b2eeec46b892efb5f0f6400762e",
    16000: "77dfdeec42fe7990f4b3b9c4eeecd480edcf5066c110603b115920af38423d03",
}
EXPECTED_DCP_METADATA_SHA256 = {
    12000: "44cc94aa5b69bb774e45561062476d4e97a3d6ef3ff6e5ab40f53591a42a651f",
    14400: "b074eeeef5ff87635495853be028b3720ce33798e69f77858cdad2302f02497b",
    16000: "a377447e5cea89c8d204df5a3d95810bd860bd6111d55dbb52bbe951aa6f4ff2",
}
EXPECTED_CHECKPOINT_BYTES = {
    12000: 384_970_543_909,
    14400: 384_970_543_908,
    16000: 384_970_543_909,
}

RECEIPT_FIELDS = frozenset(set(_v1.RECEIPT_FIELDS) | {"endpoint"})
ENDPOINT_FIELDS = frozenset(
    {
        "contract",
        "admissible_steps",
        "step",
        "storage_class",
        "nearest_step_substitution",
    }
)
CHECKPOINT_FIELDS = frozenset(
    {
        "root",
        "state_dir",
        "config_sha256",
        "checkpoint_marker_sha256",
        "dcp_metadata_sha256",
        "state_file_hash_algorithm",
        "state_file_inventory_sha256",
        "state_file_inventory",
        "model_and_optim_identity_sha256",
        "checkpoint_step",
        "permanent",
        "checkpoint_marker",
        "trainer_state_rank_count",
        "trainer_state_file_inventory",
        "trainer_state_file_inventory_sha256",
        "trainer_state_summary",
        "trainer_state_total_data_errors_by_rank",
        "trainer_state_total_data_errors_sum",
        "identity_sha256",
    }
)
TRAINER_SUMMARY_FIELDS = frozenset(
    {
        "global_step",
        "global_train_tokens_seen",
        "max_steps",
        "world_size",
        "batches_processed",
        "consecutive_data_errors",
        "wandb_run_id",
        "wandb_name",
    }
)
PRODUCER_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "validator_path",
        "validator_sha256",
        "scoring_engine_path",
        "scoring_engine_sha256",
        "perception_helper_path",
        "perception_helper_sha256",
        "bridge_helper_path",
        "bridge_helper_sha256",
        "pairing_implementation_path",
        "pairing_implementation_sha256",
        "training_contract_path",
        "training_contract_sha256",
    }
)
V1_ARTIFACT_POLICY_FIELDS = frozenset(
    {
        "output_overwrite_enabled",
        "pairing_manifest_requires_sha256_pin",
        "all_pairings_rehashed",
        "all_pairings_deterministically_rebuilt",
        "checkpoint_private_snapshot",
        "checkpoint_post_identity_rehashed",
        "native_sources_full_hash_pre_and_post",
        "descriptive_only",
        "promotion_eligible",
    }
)
ARTIFACT_POLICY_FIELDS = frozenset(
    set(V1_ARTIFACT_POLICY_FIELDS)
    | {
        "checkpoint_source_marker_preserved",
        "retained_ephemeral_not_promoted_to_permanent",
        "nearest_step_substitution_allowed",
    }
)
V1_PROTOCOL_FIELDS = frozenset(
    {
        "name",
        "descriptive_only",
        "promotion_eligible",
        "primary_statistic",
        "per_checkpoint_statistic",
        "response_logits_materialized",
        "sources",
        "blank_sources",
        "native_source",
        "visual_split",
        "visual_population",
        "examples_per_visual_source",
        "native_population",
        "native_examples",
        "native_filtered_indices",
        "pairing_seed",
        "pairing_sha256",
        "pairing_rule",
        "recipient_replay",
        "blank_rule",
        "ce_definition",
        "native_dual_denominator",
        "sequence_length",
        "rank_batch_instances",
        "global_batch_instances",
        "nodes",
        "world_size",
        "local_world_size",
        "ep_degree",
        "dp_process_group_size",
        "training_beaker_image",
        "training_git_ref",
        "checkpoint_config_sha256",
        "projection_raw_sha256",
        "source_audit_fingerprint",
        "native_holdout_fingerprint",
        "native_row_provenance_sha256",
        "native_identity",
    }
)
PROTOCOL_FIELDS = frozenset(
    set(V1_PROTOCOL_FIELDS)
    | {
        "endpoint_set_version",
        "admissible_steps",
        "evaluated_step",
        "nearest_step_substitution",
    }
)


def _expected_state_paths() -> set[str]:
    paths = {"model_and_optim/.metadata"}
    paths.update(
        f"model_and_optim/__{writer}_{shard}.distcp" for writer in range(16) for shard in range(16)
    )
    return paths


def _validate_endpoint(value: Any, *, step: int, name: str) -> Mapping[str, Any]:
    endpoint = _v1._exact(value, ENDPOINT_FIELDS, name=name)
    if (
        endpoint["contract"] != ENDPOINT_CONTRACT
        or endpoint["admissible_steps"] != list(STEPS)
        or endpoint["step"] != step
        or endpoint["storage_class"] != EXPECTED_STORAGE_CLASS[step]
        or endpoint["nearest_step_substitution"] is not False
    ):
        raise ValueError(f"{name} differs from the frozen saved-endpoint contract")
    return endpoint


def _validate_checkpoint_identity(
    value: Any,
    *,
    config: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    name: str,
    verify_live_files: bool,
) -> Mapping[str, Any]:
    identity = _v1._exact(value, CHECKPOINT_FIELDS, name=name)
    step = config["step"]
    if step not in STEPS:
        raise ValueError(f"{name} step is not an admissible saved endpoint")
    if identity["state_file_hash_algorithm"] != "sha256":
        raise ValueError(f"{name} uses an unsupported hash algorithm")
    if identity["config_sha256"] != config["sha256"]:
        raise ValueError(f"{name} config identity differs")
    for field in (
        "config_sha256",
        "checkpoint_marker_sha256",
        "dcp_metadata_sha256",
        "state_file_inventory_sha256",
        "model_and_optim_identity_sha256",
        "trainer_state_file_inventory_sha256",
        "identity_sha256",
    ):
        if not _v1._is_sha256(identity[field]):
            raise ValueError(f"{name} {field} is invalid")

    root = _v1._direct_existing_path(Path(str(identity["root"])), name=f"{name} root")
    state_dir = _v1._direct_existing_path(
        Path(str(identity["state_dir"])), name=f"{name} state directory"
    )
    train_dir = _v1._direct_existing_path(root / "train", name=f"{name} trainer-state directory")
    marker_expected = EXPECTED_MARKER[step]
    permanent_expected = not marker_expected["ephemeral"]
    if (
        root != EXPECTED_CHECKPOINT_BASE / f"step{step}"
        or state_dir != root / "model_and_optim"
        or identity["checkpoint_step"] != step
        or identity["permanent"] is not permanent_expected
        or identity["checkpoint_marker"] != marker_expected
        or endpoint["storage_class"] != EXPECTED_STORAGE_CLASS[step]
        or (endpoint["storage_class"] == "scheduled_permanent") is not permanent_expected
    ):
        raise ValueError(f"{name} storage class, root, or marker differs")
    config_path = _v1._direct_existing_path(Path(str(config["path"])), name=f"{name} config")
    if config_path != root / "config.json":
        raise ValueError(f"{name} config is not rooted in the checkpoint")
    if sorted(path.name for path in root.iterdir()) != [
        ".metadata.json",
        "config.json",
        "model_and_optim",
        "train",
    ]:
        raise ValueError(f"{name} root entries differ from the exact 275-file endpoint")
    marker, marker_sha = _v1._load_json_bytes(root / ".metadata.json", name=f"{name} marker")
    if (
        marker != marker_expected
        or marker != identity["checkpoint_marker"]
        or marker_sha != EXPECTED_MARKER_SHA256[step]
        or marker_sha != identity["checkpoint_marker_sha256"]
    ):
        raise ValueError(f"{name} source storage marker differs")

    inventory = _v1._validate_inventory(
        identity["state_file_inventory"],
        root=root,
        directory=state_dir,
        name=f"{name} state inventory",
        verify_live_files=verify_live_files,
    )
    if (
        len(inventory) != 257
        or [str(item["path"]) for item in inventory] != sorted(_expected_state_paths())
        or sum(int(item["size"]) for item in inventory) != EXPECTED_MODEL_AND_OPTIM_BYTES
        or identity["state_file_inventory_sha256"] != _v1._canonical_sha256(inventory)
    ):
        raise ValueError(f"{name} does not bind the exact 257-file model/optimizer inventory")

    trainer_inventory = _v1._validate_inventory(
        identity["trainer_state_file_inventory"],
        root=root,
        directory=train_dir,
        name=f"{name} trainer-state inventory",
        verify_live_files=verify_live_files,
    )
    expected_trainer_paths = [f"train/rank{rank}.pt" for rank in range(16)]
    if (
        identity["trainer_state_rank_count"] != 16
        or [item["path"] for item in trainer_inventory] != expected_trainer_paths
        or sum(int(item["size"]) for item in trainer_inventory) != EXPECTED_TRAINER_STATE_BYTES
        or identity["trainer_state_file_inventory_sha256"]
        != _v1._canonical_sha256(trainer_inventory)
    ):
        raise ValueError(f"{name} trainer-state inventory differs")

    summary = _v1._exact(
        identity["trainer_state_summary"],
        TRAINER_SUMMARY_FIELDS,
        name=f"{name} trainer-state summary",
    )
    if (
        summary["global_step"] != step
        or summary["global_train_tokens_seen"] != step * 1_048_576
        or summary["max_steps"] != 16_000
        or summary["world_size"] != 16
        or summary["batches_processed"] != step
        or summary["consecutive_data_errors"] != 0
        or summary["wandb_run_id"] != EXPECTED_WANDB_RUN_ID
        or summary["wandb_name"] != "vision-alignment-joint-v1"
    ):
        raise ValueError(f"{name} trainer-state progress differs")
    errors = identity["trainer_state_total_data_errors_by_rank"]
    if errors != list(EXPECTED_TRAINER_ERRORS) or identity[
        "trainer_state_total_data_errors_sum"
    ] != sum(EXPECTED_TRAINER_ERRORS):
        raise ValueError(f"{name} trainer-state data-error panel differs")

    model_fields = (
        "root",
        "state_dir",
        "config_sha256",
        "checkpoint_marker_sha256",
        "dcp_metadata_sha256",
        "state_file_hash_algorithm",
        "state_file_inventory_sha256",
        "state_file_inventory",
    )
    model_identity = {field: identity[field] for field in model_fields}
    if identity["model_and_optim_identity_sha256"] != _v1._canonical_sha256(model_identity):
        raise ValueError(f"{name} model/optimizer identity differs")
    dcp_records = [item for item in inventory if item["path"] == "model_and_optim/.metadata"]
    if (
        len(dcp_records) != 1
        or dcp_records[0]["size"] != 4_208_572
        or dcp_records[0]["sha256"] != EXPECTED_DCP_METADATA_SHA256[step]
        or identity["dcp_metadata_sha256"] != EXPECTED_DCP_METADATA_SHA256[step]
    ):
        raise ValueError(f"{name} DCP metadata identity differs")
    unsigned = dict(identity)
    digest = unsigned.pop("identity_sha256")
    if digest != _v1._canonical_sha256(unsigned):
        raise ValueError(f"{name} identity SHA-256 differs")
    # The exact declared inventory is 257 DCP entries + 16 trainer states + config + marker.
    declared_checkpoint_bytes = (
        sum(int(item["size"]) for item in inventory)
        + sum(int(item["size"]) for item in trainer_inventory)
        + config_path.stat().st_size
        + (root / ".metadata.json").stat().st_size
    )
    if (
        len(inventory) + len(trainer_inventory) + 2 != 275
        or config_path.stat().st_size != EXPECTED_CONFIG_BYTES
        or declared_checkpoint_bytes != EXPECTED_CHECKPOINT_BYTES[step]
    ):
        raise ValueError(f"{name} complete 275-file inventory differs")
    return identity


def _validate_producer(value: Any, *, name: str) -> Mapping[str, Any]:
    producer = _v1._exact(value, PRODUCER_FIELDS, name=name)
    pairing_source = inspect.getsourcefile(_v1.validate_matched_wrong_image_pairing)
    if pairing_source is None:
        raise RuntimeError("Could not locate live matched-wrong pairing implementation")
    live = {
        "": Path(__file__).with_name("vision_alignment_joint_matched_wrong_saved_steps.py"),
        "validator_": Path(__file__),
        "scoring_engine_": Path(__file__).with_name("vision_alignment_joint_matched_wrong.py"),
        "perception_helper_": Path(__file__).with_name(
            "vision_alignment_perception_matched_wrong.py"
        ),
        "bridge_helper_": Path(__file__).with_name("vision_alignment_matched_wrong.py"),
        "pairing_implementation_": Path(pairing_source),
        "training_contract_": Path(__file__).resolve().parents[1] / "train" / "Vision-Alignment.py",
    }
    for prefix, live_path in live.items():
        _v1._validate_implementation_ref(
            {"path": producer[f"{prefix}path"], "sha256": producer[f"{prefix}sha256"]},
            live_path=live_path,
            name=f"{name} {prefix or 'evaluator'}",
        )
        expected_sha256 = EXPECTED_SHARED_IMPLEMENTATION_SHA256.get(prefix)
        if expected_sha256 is not None and producer[f"{prefix}sha256"] != expected_sha256:
            raise ValueError(f"{name} {prefix} bytes differ from the frozen V1 implementation")
    comparator_path = Path(str(_v1.__file__)).resolve()
    if _v1._sha256_file(comparator_path) != EXPECTED_V1_COMPARATOR_SHA256:
        raise ValueError(f"{name} validator dependency differs from the frozen V1 comparator")
    return producer


def _validate_artifact_policy(value: Any, *, name: str) -> Mapping[str, Any]:
    policy = _v1._exact(value, ARTIFACT_POLICY_FIELDS, name=name)
    _v1._validate_semantic_policy(
        {field: policy[field] for field in V1_ARTIFACT_POLICY_FIELDS}, name=name
    )
    if (
        policy["checkpoint_source_marker_preserved"] is not True
        or policy["retained_ephemeral_not_promoted_to_permanent"] is not True
        or policy["nearest_step_substitution_allowed"] is not False
    ):
        raise ValueError(f"{name} differs from the saved-endpoint immutability policy")
    return policy


def _validate_protocol(value: Any, *, step: int, name: str) -> Mapping[str, Any]:
    protocol = _v1._exact(value, PROTOCOL_FIELDS, name=name)
    if (
        protocol["name"] != PROTOCOL_NAME
        or protocol["primary_statistic"] != PRIMARY_STATISTIC
        or protocol["endpoint_set_version"] != ENDPOINT_SET_VERSION
        or protocol["admissible_steps"] != list(STEPS)
        or protocol["evaluated_step"] != step
        or protocol["nearest_step_substitution"] is not False
    ):
        raise ValueError(f"{name} differs from the saved-endpoint protocol")
    legacy = {field: protocol[field] for field in V1_PROTOCOL_FIELDS}
    legacy["name"] = _v1.EVALUATOR_PROTOCOL_NAME
    legacy[
        "primary_statistic"
    ] = "paired source-balanced change in wrong-minus-correct CE from step4000 to step8000"
    _v1._validate_protocol(legacy, name=f"{name} shared V1 scoring contract")
    return protocol


def _load_evaluator_receipt(
    path_value: str | Path,
    *,
    expected_sha256: str,
    step: int,
    verify_live_checkpoint: bool,
) -> dict[str, Any]:
    if step not in STEPS:
        raise ValueError(f"Evaluator receipt step must be one of {STEPS}")
    if not _v1._is_sha256(expected_sha256):
        raise ValueError(f"step{step} expected receipt SHA-256 must be lowercase hex")
    path = _v1._direct_existing_path(Path(path_value), name=f"step{step} evaluator receipt")
    payload, raw_sha = _v1._load_json_bytes(
        path,
        expected_sha256=expected_sha256,
        name=f"step{step} evaluator receipt",
    )
    receipt = _v1._exact(payload, RECEIPT_FIELDS, name=f"step{step} evaluator receipt")
    if receipt["format"] != FORMAT or receipt["version"] != VERSION or receipt["status"] != STATUS:
        raise ValueError(f"step{step} evaluator receipt identity or validity differs")
    _v1._timestamp(receipt["created_at"], name=f"step{step} evaluator receipt created_at")
    _v1._validate_content_sha256(receipt, name=f"step{step} evaluator receipt")

    endpoint = _validate_endpoint(receipt["endpoint"], step=step, name=f"step{step} endpoint")
    config = _v1._validate_checkpoint_config(
        receipt["checkpoint_config"], step=step, name=f"step{step} checkpoint config"
    )
    checkpoint = _validate_checkpoint_identity(
        receipt["checkpoint"],
        config=config,
        endpoint=endpoint,
        name=f"step{step} checkpoint identity",
        verify_live_files=verify_live_checkpoint,
    )
    load_coverage = _v1._validate_load_coverage(
        receipt["load_coverage"], name=f"step{step} load coverage"
    )
    _validate_producer(receipt["producer"], name=f"step{step} producer")
    _v1._validate_git(receipt["git"], name=f"step{step} git")
    projection = _v1._validate_projection(receipt["projection"], name=f"step{step} projection")
    source_audit = _v1._validate_source_audit(
        receipt["source_audit"], name=f"step{step} source audit"
    )
    _v1._validate_registry_domains(
        projection, source_audit, name=f"step{step} projection/source-audit"
    )
    tokenizer = _v1._validate_tokenizer(receipt["tokenizer"], name=f"step{step} tokenizer")
    protocol = _validate_protocol(receipt["protocol"], step=step, name=f"step{step} protocol")
    _validate_artifact_policy(receipt["artifact_policy"], name=f"step{step} artifact policy")
    if protocol["descriptive_only"] is not True or protocol["promotion_eligible"] is not False:
        raise ValueError(f"step{step} protocol does not lock descriptive-only semantics")

    manifest_ref, manifest, pairings = _v1._validate_pairing_manifest(
        receipt["pairing_manifest"],
        receipt_protocol=protocol,
        receipt_checkpoint_config=config,
        receipt_projection=projection,
        receipt_source_audit=source_audit,
        receipt_tokenizer=tokenizer,
        name=f"step{step} pairing manifest",
    )
    if (
        Path(str(manifest_ref["path"])) != EXPECTED_PAIRING_MANIFEST_PATH
        or manifest_ref["sha256"] != EXPECTED_PAIRING_MANIFEST_SHA256
        or manifest_ref["content_sha256"] != EXPECTED_PAIRING_MANIFEST_CONTENT_SHA256
    ):
        raise ValueError(f"step{step} pairing manifest is not the exact shared V1 artifact")
    visual_results = _v1._require_mapping(
        receipt["visual_results"], name=f"step{step} visual results"
    )
    if set(visual_results) != set(_v1.SOURCE_NAMES):
        raise ValueError(f"step{step} visual results lack the exact source set")
    visual_rows: dict[str, list[dict[str, Any]]] = {}
    for source in _v1.SOURCE_NAMES:
        visual_rows[source] = _v1._validate_visual_rows(
            visual_results[source],
            source=source,
            pairing=pairings[source],
            name=f"step{step} {source} visual result",
        )

    blank_results = _v1._require_mapping(receipt["blank_results"], name=f"step{step} blank results")
    if set(blank_results) != set(_v1.BLANK_SOURCE_NAMES):
        raise ValueError(f"step{step} blank results lack the exact caption/transcript set")
    blank_rows: dict[str, list[dict[str, Any]]] = {}
    for source in _v1.BLANK_SOURCE_NAMES:
        blank_rows[source] = _v1._validate_blank_rows(
            blank_results[source],
            source=source,
            visual_rows=visual_rows[source],
            pairing_sha256=_v1.matched_wrong_image_pairing_sha256(pairings[source]),
            pairing_coverage=pairings[source]["coverage"],
            name=f"step{step} {source} blank result",
        )

    (
        native_rows,
        native_order_sha256,
        native_provenance_sha256,
        native_identity_sha256,
    ) = _v1._validate_native_rows(receipt["native_result"], name=f"step{step} native result")
    native_identity = _v1._validate_native_identity(
        protocol["native_identity"], name=f"step{step} native identity"
    )
    if (
        native_identity_sha256 != _v1._canonical_sha256(native_identity)
        or native_provenance_sha256 != native_identity["row_provenance_sha256"]
        or protocol["native_row_provenance_sha256"] != native_provenance_sha256
        or native_order_sha256 != native_identity["manifest_order_sha256"]
    ):
        raise ValueError(f"step{step} native result does not bind the protocol identity")
    return {
        "input": {"path": str(path), "sha256": raw_sha},
        "receipt": receipt,
        "endpoint": endpoint,
        "checkpoint": checkpoint,
        "checkpoint_config": config,
        "load_coverage": load_coverage,
        "manifest_ref": manifest_ref,
        "manifest": manifest,
        "pairings": pairings,
        "visual_rows": visual_rows,
        "blank_rows": blank_rows,
        "native_rows": native_rows,
    }


def validate_evaluator_receipt(
    path: str | Path,
    expected_sha256: str,
    step: int,
    verify_live_checkpoint: bool = False,
) -> None:
    """Validate one exact V2 saved-endpoint evaluator receipt.

    The evaluator calls this hook on scratch receipt bytes before immutable publication.  It may
    defer the second live checkpoint read there because it has just authenticated the endpoint;
    independent post-publication audits can set ``verify_live_checkpoint=True``.
    """
    _load_evaluator_receipt(
        path,
        expected_sha256=expected_sha256,
        step=step,
        verify_live_checkpoint=verify_live_checkpoint,
    )
