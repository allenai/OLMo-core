"""Contracts and comparison helpers for dense SSMax multimodal evaluation.

This module deliberately has no dependency on ``lmms-eval``.  The executable runner keeps
that optional dependency at the edge, while checkpoint identity checks, task filtering, and
paired result comparison remain unit-testable in the base OLMo-core environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from olmo_core.eval.vision_alignment_ssmax_bridge import SSMaxBridgeEvidenceError
from olmo_core.eval.vision_alignment_ssmax_bridge import (
    checkpoint_identity as full_checkpoint_identity,
)
from olmo_core.nn.attention import AttentionConfig, GatedDeltaNetConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.vision import MultimodalLMConfig

SSMAX_VARIANTS = ("ssmax_head_qknorm", "ssmax_no_qknorm")
SSMAX_DOWNSTREAM_TASKS = (
    "ssmax_blink_jigsaw",
    "ssmax_mathvista_geometry_mc",
)

LMMS_EVAL_REVISION = "cb45ac4d4a667ea5ef89c7a148bff69b3489b981"
BLINK_DATASET_REVISION = "a3666eb249237ba3d5eca8db21176cc47967e040"
MATHVISTA_DATASET_REVISION = "2b6ad69445fbb5695c9b165475e8decdbeb97747"
BLINK_JIGSAW_EXAMPLES = 150
MATHVISTA_GEOMETRY_MC_EXAMPLES = 203
TRAJECTORY_BOOTSTRAP_SAMPLES = 10_000
TRAJECTORY_BOOTSTRAP_SEED = 6_198
TRAJECTORY_PRIMARY_PHASE = "joint"
TRAJECTORY_PRIMARY_STEP = 16_000
TRAJECTORY_TASK_EQUIVALENCE_MARGIN = 0.03
TRAJECTORY_MACRO_EQUIVALENCE_MARGIN = 0.02

_SHA256 = re.compile(r"[0-9a-f]{64}")


class SSMaxDownstreamContractError(ValueError):
    """Raised when a checkpoint, result, or paired-evaluation contract differs."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of ``path`` without following an implicit fallback."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value with deterministic serialization."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SSMaxDownstreamContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def checkpoint_root(path: str | Path) -> Path:
    """Resolve a checkpoint root whether the caller names it or ``model_and_optim``."""
    resolved = Path(path).resolve()
    if resolved.name == "model_and_optim":
        resolved = resolved.parent
    return resolved


@dataclass(frozen=True)
class CheckpointIdentity:
    """Full byte identity and semantic fields for one evaluation checkpoint."""

    path: str
    model_variant: str
    phase: str
    global_step: int
    config_sha256: str
    marker_sha256: str
    dcp_metadata_sha256: str
    state_file_count: int
    state_file_inventory_sha256: str
    trainer_state_count: int
    trainer_state_inventory_sha256: str
    identity_sha256: str

    @property
    def checkpoint(self) -> str:
        """Return the native checkpoint path used by :class:`Checkpointer`."""
        return self.path

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def verify_checkpoint_identity(
    checkpoint: str | Path,
    *,
    expected_model_variant: str,
    expected_phase: str,
    expected_global_step: int,
    expected_config_sha256: str,
    expected_marker_sha256: str,
    expected_dcp_metadata_sha256: str,
    expected_checkpoint_identity_sha256: str,
    hash_workers: int = 8,
) -> tuple[CheckpointIdentity, dict[str, Any]]:
    """Verify every byte of a permanent checkpoint before loading model tensors.

    The identity uses the bridge manifest contract: every DCP state file and every trainer-rank
    state file is hashed, and trainer step/world-size receipts are checked.  Pinning only DCP
    metadata is insufficient because a modified ``.distcp`` shard can retain valid metadata.

    :returns: The verified identity and parsed ``config.json``.
    """
    if expected_model_variant not in SSMAX_VARIANTS:
        raise SSMaxDownstreamContractError(
            f"expected_model_variant must be one of {SSMAX_VARIANTS}, got "
            f"{expected_model_variant!r}"
        )
    if expected_phase not in {"bridge", "perception", "joint"}:
        raise SSMaxDownstreamContractError(f"unsupported expected phase {expected_phase!r}")
    if (
        isinstance(expected_global_step, bool)
        or not isinstance(expected_global_step, int)
        or expected_global_step < 0
    ):
        raise SSMaxDownstreamContractError("expected global step must be a non-negative integer")
    if hash_workers <= 0:
        raise SSMaxDownstreamContractError("hash_workers must be positive")
    expected_hashes = {
        "config_sha256": _require_sha256(expected_config_sha256, name="config SHA-256"),
        "marker_sha256": _require_sha256(expected_marker_sha256, name="marker SHA-256"),
        "dcp_metadata_sha256": _require_sha256(
            expected_dcp_metadata_sha256, name="DCP metadata SHA-256"
        ),
        "identity_sha256": _require_sha256(
            expected_checkpoint_identity_sha256, name="checkpoint identity SHA-256"
        ),
    }

    root = checkpoint_root(checkpoint)
    try:
        full_identity = full_checkpoint_identity(root, workers=hash_workers)
    except (SSMaxBridgeEvidenceError, OSError, RuntimeError, ValueError) as error:
        raise SSMaxDownstreamContractError(str(error)) from error
    for name, expected in expected_hashes.items():
        actual = full_identity.get(name)
        if actual != expected:
            raise SSMaxDownstreamContractError(
                f"{name} mismatch for {root}: expected {expected}, got {actual}"
            )
    if full_identity.get("global_step") != expected_global_step:
        raise SSMaxDownstreamContractError(
            "checkpoint global step differs: "
            f"expected {expected_global_step}, got {full_identity.get('global_step')!r}"
        )

    with (root / "config.json").open() as file_handle:
        raw_config = json.load(file_handle)
    if raw_config.get("model_variant") != expected_model_variant:
        raise SSMaxDownstreamContractError(
            "checkpoint model_variant differs: "
            f"expected {expected_model_variant!r}, got {raw_config.get('model_variant')!r}"
        )
    if raw_config.get("phase") != expected_phase:
        raise SSMaxDownstreamContractError(
            f"checkpoint phase differs: expected {expected_phase!r}, got {raw_config.get('phase')!r}"
        )
    alignment = raw_config.get("vision_alignment")
    if not isinstance(alignment, dict):
        raise SSMaxDownstreamContractError("checkpoint lacks vision_alignment metadata")
    if alignment.get("model_variant") != expected_model_variant:
        raise SSMaxDownstreamContractError("vision_alignment.model_variant differs")
    if alignment.get("phase") != expected_phase:
        raise SSMaxDownstreamContractError("vision_alignment.phase differs")

    identity = CheckpointIdentity(
        path=str(root),
        model_variant=expected_model_variant,
        phase=expected_phase,
        global_step=expected_global_step,
        config_sha256=expected_hashes["config_sha256"],
        marker_sha256=expected_hashes["marker_sha256"],
        dcp_metadata_sha256=expected_hashes["dcp_metadata_sha256"],
        state_file_count=int(full_identity["state_file_count"]),
        state_file_inventory_sha256=str(full_identity["state_file_inventory_sha256"]),
        trainer_state_count=int(full_identity["trainer_state_count"]),
        trainer_state_inventory_sha256=str(full_identity["trainer_state_inventory_sha256"]),
        identity_sha256=expected_hashes["identity_sha256"],
    )
    return identity, raw_config


def validate_ssmax_model_config(
    model_config: MultimodalLMConfig,
    *,
    expected_model_variant: str,
) -> None:
    """Reject a serialized model that is not one of the reviewed 1.4B SSMax hybrids."""
    if expected_model_variant not in SSMAX_VARIANTS:
        raise SSMaxDownstreamContractError(f"unknown SSMax variant {expected_model_variant!r}")
    lm_config = model_config.lm
    if not isinstance(lm_config, TransformerConfig):
        raise SSMaxDownstreamContractError("dense SSMax evaluation requires TransformerConfig")
    if (
        lm_config.d_model != 1280
        or lm_config.n_layers != 20
        or lm_config.vocab_size != 100_352
        or not isinstance(lm_config.block.sequence_mixer, GatedDeltaNetConfig)
    ):
        raise SSMaxDownstreamContractError("model is not the reviewed 1.4B SSMax hybrid")

    expected_qk_norm = expected_model_variant == "ssmax_head_qknorm"
    attention_layers = []
    for layer_index, block_config in enumerate(lm_config.resolved_block_configs):
        mixer = block_config.sequence_mixer
        if isinstance(mixer, AttentionConfig):
            attention_layers.append(layer_index)
            if not mixer.scalable_softmax:
                raise SSMaxDownstreamContractError(
                    f"Scalable-Softmax is disabled in attention layer {layer_index}"
                )
            has_qk_norm = mixer.qk_norm is not None and mixer.use_head_qk_norm is True
            if has_qk_norm != expected_qk_norm:
                raise SSMaxDownstreamContractError(
                    f"QK-norm differs in layer {layer_index}: expected {expected_qk_norm}, "
                    f"got {has_qk_norm}"
                )
        elif not isinstance(mixer, GatedDeltaNetConfig):
            raise SSMaxDownstreamContractError(
                f"unsupported sequence mixer in layer {layer_index}: {type(mixer).__name__}"
            )
    if attention_layers != [4, 9, 14, 19]:
        raise SSMaxDownstreamContractError(
            f"SSMax attention layers differ: expected [4, 9, 14, 19], got {attention_layers}"
        )


def is_mathvista_geometry_mc(document: Mapping[str, Any]) -> bool:
    """Return whether a MathVista row is in the exact efficient geometry-MC slice."""
    metadata = document.get("metadata")
    return bool(
        isinstance(metadata, Mapping)
        and metadata.get("task") == "geometry problem solving"
        and document.get("question_type") == "multi_choice"
    )


def task_definition_inventory(root: str | Path) -> dict[str, Any]:
    """Fingerprint the checked-in task definitions used by the downstream runner."""
    task_root = Path(root)
    names = ("blink_jigsaw.yaml", "mathvista_geometry_mc.yaml", "utils.py")
    files = []
    for name in names:
        path = task_root / name
        if not path.is_file():
            raise SSMaxDownstreamContractError(f"missing downstream task definition {path}")
        files.append({"path": name, "sha256": sha256_file(path)})
    return {"files": files, "sha256": canonical_sha256(files)}


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SSMaxDownstreamContractError(f"{name} must be an object")
    return value


def _sample_outcomes(payload: Mapping[str, Any], task: str) -> dict[str, dict[str, Any]]:
    lmms_eval = _require_mapping(payload.get("lmms_eval"), name="lmms_eval")
    samples_by_task = _require_mapping(lmms_eval.get("samples"), name="lmms_eval.samples")
    samples = samples_by_task.get(task)
    if not isinstance(samples, list) or not samples:
        raise SSMaxDownstreamContractError(f"missing complete samples for {task}")

    outcomes: dict[str, dict[str, Any]] = {}
    for sample in samples:
        row = _require_mapping(sample, name=f"{task} sample")
        if task == "ssmax_blink_jigsaw":
            metric = _require_mapping(row.get("blink_acc"), name="blink_acc")
            example_id = metric.get("id")
            target = metric.get("gt_content")
            prediction = metric.get("pred_parsed")
            correct = metric.get("is_correct")
            num_choices = metric.get("num_choices")
        elif task == "ssmax_mathvista_geometry_mc":
            metric = _require_mapping(
                row.get("mathvista_geometry_mc_acc"), name="mathvista_geometry_mc_acc"
            )
            example_id = metric.get("question_id")
            target = metric.get("answer")
            prediction = metric.get("raw_response")
            correct = metric.get("true_false")
            num_choices = metric.get("num_choices")
        else:
            raise SSMaxDownstreamContractError(f"unsupported downstream task {task!r}")

        if not isinstance(example_id, (str, int)):
            raise SSMaxDownstreamContractError(f"{task} sample has no stable ID")
        if not isinstance(correct, bool):
            raise SSMaxDownstreamContractError(f"{task} sample correctness is not boolean")
        if not isinstance(num_choices, int) or isinstance(num_choices, bool) or num_choices < 2:
            raise SSMaxDownstreamContractError(f"{task} sample has invalid num_choices")
        normalized_prediction = prediction.strip().upper() if isinstance(prediction, str) else ""
        prediction_valid = bool(
            len(normalized_prediction) == 1
            and "A" <= normalized_prediction <= chr(ord("A") + num_choices - 1)
        )
        key = str(example_id)
        if key in outcomes:
            raise SSMaxDownstreamContractError(f"duplicate {task} sample ID {key!r}")
        outcomes[key] = {
            "target": target,
            "prediction": normalized_prediction,
            "prediction_valid": prediction_valid,
            "correct": correct,
            "num_choices": num_choices,
        }
    return outcomes


def _validate_result(payload: Mapping[str, Any], *, expected_variant: str) -> None:
    if payload.get("schema_version") != 1:
        raise SSMaxDownstreamContractError("unsupported downstream result schema")
    identity = _require_mapping(payload.get("checkpoint_identity"), name="checkpoint_identity")
    if identity.get("model_variant") != expected_variant:
        raise SSMaxDownstreamContractError(
            f"result variant differs: expected {expected_variant!r}, "
            f"got {identity.get('model_variant')!r}"
        )
    if identity.get("phase") not in {"bridge", "perception", "joint"}:
        raise SSMaxDownstreamContractError("result checkpoint phase is unsupported")
    global_step = identity.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise SSMaxDownstreamContractError("result checkpoint global step is invalid")
    for field in (
        "config_sha256",
        "marker_sha256",
        "dcp_metadata_sha256",
        "state_file_inventory_sha256",
        "trainer_state_inventory_sha256",
        "identity_sha256",
    ):
        _require_sha256(identity.get(field), name=f"checkpoint identity {field}")
    for field in ("state_file_count", "trainer_state_count"):
        count = identity.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise SSMaxDownstreamContractError(f"checkpoint identity {field} must be positive")
    protocol = _require_mapping(payload.get("protocol"), name="protocol")
    if protocol.get("tasks") != list(SSMAX_DOWNSTREAM_TASKS):
        raise SSMaxDownstreamContractError("result does not contain the exact fast task pair")
    if protocol.get("partial") is not False or protocol.get("limit") is not None:
        raise SSMaxDownstreamContractError("partial downstream results cannot be compared")
    expected = {
        "lmms_eval_revision": LMMS_EVAL_REVISION,
        "blink_dataset_revision": BLINK_DATASET_REVISION,
        "mathvista_dataset_revision": MATHVISTA_DATASET_REVISION,
        "dataset_auth": None,
        "response_mode": "valid_choice_letter_logits",
        "prompt_layout": "document",
        "crop_budget_mode": "shared_total",
        "max_sequence_length": 8192,
        "max_crops_total": 8,
        "sequence_bucket_size": 128,
        "world_size": 1,
        "checkpoint_format": "native_olmo_core_dcp",
        "checkpoint_conversion": None,
        "checkpoint_identity_semantics": "vision_alignment_ssmax_bridge.checkpoint_identity",
        "mathvista_scoring": "local_valid_letter_choice_string_equal",
        "external_judge": None,
        "generation": "single_forward_valid_option_letter_logits",
    }
    for name, value in expected.items():
        if protocol.get(name) != value:
            raise SSMaxDownstreamContractError(
                f"result protocol {name} differs: expected {value!r}, got {protocol.get(name)!r}"
            )
    _require_sha256(protocol.get("task_definition_sha256"), name="task definition SHA-256")
    load = _require_mapping(protocol.get("checkpoint_load"), name="checkpoint_load")
    expected_load = {
        "strict_model_state": True,
        "load_optimizer_state": False,
        "load_trainer_state": False,
        "state_file_count": identity["state_file_count"],
        "state_file_inventory_sha256": identity["state_file_inventory_sha256"],
        "trainer_state_count": identity["trainer_state_count"],
        "trainer_state_inventory_sha256": identity["trainer_state_inventory_sha256"],
    }
    if dict(load) != expected_load:
        raise SSMaxDownstreamContractError("result checkpoint_load receipt differs from identity")


def _mcnemar_exact(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def _task_comparison(
    left: Mapping[str, dict[str, Any]],
    right: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    if set(left) != set(right):
        raise SSMaxDownstreamContractError("paired task sample IDs differ between model arms")
    both_correct = left_only = right_only = both_wrong = 0
    left_invalid = right_invalid = 0
    chance_sum = 0.0
    left_predictions: dict[str, int] = {}
    right_predictions: dict[str, int] = {}
    source_projection = []
    for example_id in sorted(left):
        left_row = left[example_id]
        right_row = right[example_id]
        if (
            left_row["target"] != right_row["target"]
            or left_row["num_choices"] != right_row["num_choices"]
        ):
            raise SSMaxDownstreamContractError(
                f"paired source row differs for sample {example_id!r}"
            )
        source_projection.append(
            {
                "id": example_id,
                "target": left_row["target"],
                "num_choices": left_row["num_choices"],
            }
        )
        chance_sum += 1.0 / left_row["num_choices"]
        left_predictions[str(left_row["prediction"])] = (
            left_predictions.get(str(left_row["prediction"]), 0) + 1
        )
        right_predictions[str(right_row["prediction"])] = (
            right_predictions.get(str(right_row["prediction"]), 0) + 1
        )
        left_invalid += int(not left_row["prediction_valid"])
        right_invalid += int(not right_row["prediction_valid"])
        pair = (left_row["correct"], right_row["correct"])
        if pair == (True, True):
            both_correct += 1
        elif pair == (True, False):
            left_only += 1
        elif pair == (False, True):
            right_only += 1
        else:
            both_wrong += 1

    count = len(left)
    left_correct = both_correct + left_only
    right_correct = both_correct + right_only
    return {
        "samples": count,
        "source_projection_sha256": canonical_sha256(source_projection),
        "chance_accuracy": chance_sum / count,
        "left": {
            "correct": left_correct,
            "accuracy": left_correct / count,
            "prediction_histogram": dict(sorted(left_predictions.items())),
            "max_prediction_share": max(left_predictions.values()) / count,
            "invalid_prediction_count": left_invalid,
            "invalid_prediction_share": left_invalid / count,
        },
        "right": {
            "correct": right_correct,
            "accuracy": right_correct / count,
            "prediction_histogram": dict(sorted(right_predictions.items())),
            "max_prediction_share": max(right_predictions.values()) / count,
            "invalid_prediction_count": right_invalid,
            "invalid_prediction_share": right_invalid / count,
        },
        "accuracy_difference_left_minus_right": (left_correct - right_correct) / count,
        "paired_outcomes": {
            "both_correct": both_correct,
            "left_only_correct": left_only,
            "right_only_correct": right_only,
            "both_wrong": both_wrong,
        },
        "mcnemar_exact_two_sided_p": _mcnemar_exact(left_only, right_only),
    }


def compare_downstream_results(
    left_payload: Mapping[str, Any],
    right_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare QK-norm and no-QK-norm results on exactly matched samples.

    The macro score weights BLINK Jigsaw and MathVista geometry-MC equally.  It is a
    descriptive ranking signal, not a significance gate.
    """
    _validate_result(left_payload, expected_variant="ssmax_head_qknorm")
    _validate_result(right_payload, expected_variant="ssmax_no_qknorm")
    left_protocol = _require_mapping(left_payload["protocol"], name="left protocol")
    right_protocol = _require_mapping(right_payload["protocol"], name="right protocol")
    left_identity = _require_mapping(
        left_payload["checkpoint_identity"], name="left checkpoint identity"
    )
    right_identity = _require_mapping(
        right_payload["checkpoint_identity"], name="right checkpoint identity"
    )
    for field in ("phase", "global_step"):
        if left_identity.get(field) != right_identity.get(field):
            raise SSMaxDownstreamContractError(
                f"paired checkpoint point differs in {field}: "
                f"{left_identity.get(field)!r} != {right_identity.get(field)!r}"
            )
    for field in (
        "task_definition_sha256",
        "max_sequence_length",
        "max_crops_total",
        "sequence_bucket_size",
    ):
        if left_protocol.get(field) != right_protocol.get(field):
            raise SSMaxDownstreamContractError(f"paired result protocol differs in {field}")

    tasks: dict[str, Any] = {}
    for task in SSMAX_DOWNSTREAM_TASKS:
        tasks[task] = _task_comparison(
            _sample_outcomes(left_payload, task),
            _sample_outcomes(right_payload, task),
        )
    expected_counts = {
        "ssmax_blink_jigsaw": BLINK_JIGSAW_EXAMPLES,
        "ssmax_mathvista_geometry_mc": MATHVISTA_GEOMETRY_MC_EXAMPLES,
    }
    for task, expected_count in expected_counts.items():
        if tasks[task]["samples"] != expected_count:
            raise SSMaxDownstreamContractError(
                f"{task} coverage differs: expected {expected_count}, "
                f"got {tasks[task]['samples']}"
            )

    left_macro = sum(task["left"]["accuracy"] for task in tasks.values()) / len(tasks)
    right_macro = sum(task["right"]["accuracy"] for task in tasks.values()) / len(tasks)
    if left_macro > right_macro:
        observed_point_ranking = "ssmax_head_qknorm"
    elif right_macro > left_macro:
        observed_point_ranking = "ssmax_no_qknorm"
    else:
        observed_point_ranking = "tie"
    task_directions = {
        task: (
            "ssmax_head_qknorm"
            if result["accuracy_difference_left_minus_right"] > 0
            else (
                "ssmax_no_qknorm" if result["accuracy_difference_left_minus_right"] < 0 else "tie"
            )
        )
        for task, result in tasks.items()
    }
    non_tie_directions = {value for value in task_directions.values() if value != "tie"}
    return {
        "format": "vision_alignment_ssmax_downstream_comparison",
        "version": 1,
        "interpretation": "descriptive_paired_fast_signal",
        "checkpoint_point": {
            "phase": left_identity["phase"],
            "global_step": left_identity["global_step"],
        },
        "checkpoint_identities": {
            "ssmax_head_qknorm": dict(left_identity),
            "ssmax_no_qknorm": dict(right_identity),
        },
        "tasks": tasks,
        "macro_accuracy": {
            "ssmax_head_qknorm": left_macro,
            "ssmax_no_qknorm": right_macro,
            "difference_qknorm_minus_no_qknorm": left_macro - right_macro,
        },
        "observed_point_ranking": observed_point_ranking,
        "inference": {
            "conclusion": "inconclusive",
            "predeclared_winner_rule": None,
            "task_directions": task_directions,
            "task_direction_consistent": len(non_tie_directions) <= 1,
            "paired_mcnemar_exact_two_sided_p": {
                task: result["mcnemar_exact_two_sided_p"] for task, result in tasks.items()
            },
            "reason": (
                "No cross-task superiority rule was preregistered. Point estimates and paired "
                "task tests are descriptive and cannot establish molmofiability or adaptation."
            ),
        },
        "caveat": (
            "This two-task fast suite reports observed downstream outcomes only; it does not "
            "prove a QK-norm mechanism or substitute for the full Molmo image evaluation suite."
        ),
    }


def load_and_compare_results(
    left_path: str | Path,
    right_path: str | Path,
    *,
    expected_left_sha256: str,
    expected_right_sha256: str,
) -> dict[str, Any]:
    """Load two exact result files, verify their hashes, and compare them."""
    paths = (Path(left_path), Path(right_path))
    expected = (
        _require_sha256(expected_left_sha256, name="left result SHA-256"),
        _require_sha256(expected_right_sha256, name="right result SHA-256"),
    )
    payloads = []
    for path, digest in zip(paths, expected):
        if sha256_file(path) != digest:
            raise SSMaxDownstreamContractError(f"result SHA-256 mismatch for {path}")
        with path.open() as file_handle:
            payloads.append(json.load(file_handle))
    comparison = compare_downstream_results(payloads[0], payloads[1])
    comparison["inputs"] = {
        "ssmax_head_qknorm": {"path": str(paths[0]), "sha256": expected[0]},
        "ssmax_no_qknorm": {"path": str(paths[1]), "sha256": expected[1]},
    }
    return comparison


def _bootstrap_seed(label: str) -> int:
    raw = hashlib.sha256(f"{TRAJECTORY_BOOTSTRAP_SEED}:{label}".encode()).digest()
    return int.from_bytes(raw[:8], "big")


def _row_paired_bootstrap_means(values: list[int], *, label: str) -> list[float]:
    if not values:
        raise SSMaxDownstreamContractError("cannot bootstrap an empty paired row vector")
    rng = random.Random(_bootstrap_seed(label))
    count = len(values)
    return [
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(TRAJECTORY_BOOTSTRAP_SAMPLES)
    ]


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _bootstrap_interval(distribution: list[float]) -> dict[str, Any]:
    ordered = sorted(distribution)
    return {
        "confidence_level": 0.95,
        "lower": _percentile(ordered, 0.025),
        "upper": _percentile(ordered, 0.975),
        "samples": TRAJECTORY_BOOTSTRAP_SAMPLES,
        "root_seed": TRAJECTORY_BOOTSTRAP_SEED,
        "method": "row_paired_nonparametric_percentile_python_mt19937_v1",
    }


def _trajectory_task(
    baseline_qknorm: Mapping[str, dict[str, Any]],
    baseline_no_qknorm: Mapping[str, dict[str, Any]],
    candidate_qknorm: Mapping[str, dict[str, Any]],
    candidate_no_qknorm: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[int]]:
    row_sets = [
        set(rows)
        for rows in (
            baseline_qknorm,
            baseline_no_qknorm,
            candidate_qknorm,
            candidate_no_qknorm,
        )
    ]
    if any(row_ids != row_sets[0] for row_ids in row_sets[1:]):
        raise SSMaxDownstreamContractError(
            "trajectory sample IDs differ across baseline/candidate model arms"
        )

    did_rows = []
    source_projection = []
    baseline_qknorm_correct = baseline_no_qknorm_correct = 0
    candidate_qknorm_correct = candidate_no_qknorm_correct = 0
    for example_id in sorted(row_sets[0]):
        rows = (
            baseline_qknorm[example_id],
            baseline_no_qknorm[example_id],
            candidate_qknorm[example_id],
            candidate_no_qknorm[example_id],
        )
        target = rows[0]["target"]
        num_choices = rows[0]["num_choices"]
        if any(row["target"] != target or row["num_choices"] != num_choices for row in rows[1:]):
            raise SSMaxDownstreamContractError(
                f"trajectory source row differs for sample {example_id!r}"
            )
        source_projection.append({"id": example_id, "target": target, "num_choices": num_choices})
        baseline_qk = int(rows[0]["correct"])
        baseline_no_qk = int(rows[1]["correct"])
        candidate_qk = int(rows[2]["correct"])
        candidate_no_qk = int(rows[3]["correct"])
        baseline_qknorm_correct += baseline_qk
        baseline_no_qknorm_correct += baseline_no_qk
        candidate_qknorm_correct += candidate_qk
        candidate_no_qknorm_correct += candidate_no_qk
        did_rows.append((candidate_qk - baseline_qk) - (candidate_no_qk - baseline_no_qk))

    count = len(did_rows)
    baseline_qknorm_accuracy = baseline_qknorm_correct / count
    baseline_no_qknorm_accuracy = baseline_no_qknorm_correct / count
    candidate_qknorm_accuracy = candidate_qknorm_correct / count
    candidate_no_qknorm_accuracy = candidate_no_qknorm_correct / count
    qknorm_gain = candidate_qknorm_accuracy - baseline_qknorm_accuracy
    no_qknorm_gain = candidate_no_qknorm_accuracy - baseline_no_qknorm_accuracy
    did = qknorm_gain - no_qknorm_gain
    if not math.isclose(did, sum(did_rows) / count, rel_tol=0.0, abs_tol=1e-15):
        raise AssertionError("row-level and aggregate DID calculations differ")
    return (
        {
            "samples": count,
            "source_projection_sha256": canonical_sha256(source_projection),
            "baseline_accuracy": {
                "ssmax_head_qknorm": baseline_qknorm_accuracy,
                "ssmax_no_qknorm": baseline_no_qknorm_accuracy,
            },
            "candidate_accuracy": {
                "ssmax_head_qknorm": candidate_qknorm_accuracy,
                "ssmax_no_qknorm": candidate_no_qknorm_accuracy,
            },
            "accuracy_gain_from_step0": {
                "ssmax_head_qknorm": qknorm_gain,
                "ssmax_no_qknorm": no_qknorm_gain,
            },
            "gain_difference_qknorm_minus_no_qknorm": did,
        },
        did_rows,
    )


def compare_downstream_trajectory(
    baseline_qknorm_payload: Mapping[str, Any],
    baseline_no_qknorm_payload: Mapping[str, Any],
    candidate_qknorm_payload: Mapping[str, Any],
    candidate_no_qknorm_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare step0-normalized adaptation gains for one exact candidate checkpoint pair.

    The bootstrap resamples matched source rows jointly across both model arms and both time
    points.  BLINK and MathVista are resampled independently, then weighted equally for the macro
    interval.  The 10,000-draw seed and percentile algorithm are fixed before results exist.
    """
    baseline_absolute = compare_downstream_results(
        baseline_qknorm_payload, baseline_no_qknorm_payload
    )
    candidate_absolute = compare_downstream_results(
        candidate_qknorm_payload, candidate_no_qknorm_payload
    )
    baseline_point = baseline_absolute["checkpoint_point"]
    candidate_point = candidate_absolute["checkpoint_point"]
    if baseline_point != {"phase": "bridge", "global_step": 0}:
        raise SSMaxDownstreamContractError(
            "trajectory baseline must be the paired assembled bridge step0"
        )
    if candidate_point == baseline_point:
        raise SSMaxDownstreamContractError("trajectory candidate must differ from bridge step0")

    baseline_protocol = _require_mapping(
        baseline_qknorm_payload["protocol"], name="baseline protocol"
    )
    candidate_protocol = _require_mapping(
        candidate_qknorm_payload["protocol"], name="candidate protocol"
    )
    for field in (
        "task_definition_sha256",
        "max_sequence_length",
        "max_crops_total",
        "sequence_bucket_size",
        "response_mode",
        "prompt_layout",
        "crop_budget_mode",
    ):
        if baseline_protocol.get(field) != candidate_protocol.get(field):
            raise SSMaxDownstreamContractError(
                f"trajectory result protocol differs across time in {field}"
            )

    tasks: dict[str, Any] = {}
    distributions: dict[str, list[float]] = {}
    for task in SSMAX_DOWNSTREAM_TASKS:
        details, did_rows = _trajectory_task(
            _sample_outcomes(baseline_qknorm_payload, task),
            _sample_outcomes(baseline_no_qknorm_payload, task),
            _sample_outcomes(candidate_qknorm_payload, task),
            _sample_outcomes(candidate_no_qknorm_payload, task),
        )
        distribution = _row_paired_bootstrap_means(did_rows, label=task)
        details["gain_difference_bootstrap_ci"] = _bootstrap_interval(distribution)
        tasks[task] = details
        distributions[task] = distribution

    macro_did = sum(
        result["gain_difference_qknorm_minus_no_qknorm"] for result in tasks.values()
    ) / len(tasks)
    macro_distribution = [
        sum(distributions[task][index] for task in SSMAX_DOWNSTREAM_TASKS)
        / len(SSMAX_DOWNSTREAM_TASKS)
        for index in range(TRAJECTORY_BOOTSTRAP_SAMPLES)
    ]
    macro_interval = _bootstrap_interval(macro_distribution)
    task_dids = [
        tasks[task]["gain_difference_qknorm_minus_no_qknorm"] for task in SSMAX_DOWNSTREAM_TASKS
    ]
    if all(value > 0 for value in task_dids) and macro_interval["lower"] > 0:
        conclusion = "directional_signal_ssmax_head_qknorm"
    elif all(value < 0 for value in task_dids) and macro_interval["upper"] < 0:
        conclusion = "directional_signal_ssmax_no_qknorm"
    elif (
        macro_interval["lower"] >= -TRAJECTORY_MACRO_EQUIVALENCE_MARGIN
        and macro_interval["upper"] <= TRAJECTORY_MACRO_EQUIVALENCE_MARGIN
        and all(
            tasks[task]["gain_difference_bootstrap_ci"]["lower"]
            >= -TRAJECTORY_TASK_EQUIVALENCE_MARGIN
            and tasks[task]["gain_difference_bootstrap_ci"]["upper"]
            <= TRAJECTORY_TASK_EQUIVALENCE_MARGIN
            for task in SSMAX_DOWNSTREAM_TASKS
        )
    ):
        conclusion = "practical_equivalence_fast_suite"
    else:
        conclusion = "inconclusive"

    if macro_did > 0:
        observed_direction = "ssmax_head_qknorm"
    elif macro_did < 0:
        observed_direction = "ssmax_no_qknorm"
    else:
        observed_direction = "tie"
    return {
        "format": "vision_alignment_ssmax_downstream_trajectory_comparison",
        "version": 1,
        "baseline_point": baseline_point,
        "candidate_point": candidate_point,
        "absolute_same_step": {
            "baseline": baseline_absolute,
            "candidate": candidate_absolute,
        },
        "tasks": tasks,
        "equal_task_macro": {
            "gain_difference_qknorm_minus_no_qknorm": macro_did,
            "gain_difference_bootstrap_ci": macro_interval,
            "observed_point_direction": observed_direction,
        },
        "inference": {
            "conclusion": conclusion,
            "primary_endpoint": {
                "phase": TRAJECTORY_PRIMARY_PHASE,
                "global_step": TRAJECTORY_PRIMARY_STEP,
                "eligible": candidate_point
                == {"phase": TRAJECTORY_PRIMARY_PHASE, "global_step": TRAJECTORY_PRIMARY_STEP},
            },
            "predeclared_decision_rule": (
                "Superiority requires both task DIDs to have the same strict sign and the "
                "equal-task macro 95% row-paired bootstrap interval to exclude zero in that "
                "direction. Practical equivalence requires both task intervals to lie within "
                "+/-0.03 and the macro interval within +/-0.02; otherwise inconclusive."
            ),
            "equivalence_margins": {
                "per_task_accuracy": TRAJECTORY_TASK_EQUIVALENCE_MARGIN,
                "equal_task_macro_accuracy": TRAJECTORY_MACRO_EQUIVALENCE_MARGIN,
            },
            "criterion_satisfied": conclusion != "inconclusive",
            "scope": "conditional_pilot_fast_suite_adaptation_signal",
            "requires_external_veto_clearance": [
                "all_phase_hard_invariants",
                "native_text_noninferiority",
                "generation_answer_distribution",
            ],
        },
        "caveat": (
            "The DID separates fast-suite gain from assembled step0 capability, but it remains "
            "an exploratory two-task signal and does not prove a QK-norm mechanism."
        ),
    }


def load_and_compare_trajectory(
    baseline_qknorm_path: str | Path,
    baseline_no_qknorm_path: str | Path,
    candidate_qknorm_path: str | Path,
    candidate_no_qknorm_path: str | Path,
    *,
    expected_baseline_qknorm_sha256: str,
    expected_baseline_no_qknorm_sha256: str,
    expected_candidate_qknorm_sha256: str,
    expected_candidate_no_qknorm_sha256: str,
) -> dict[str, Any]:
    """Hash-pin and compare the exact four results required for one trajectory DID."""
    entries = (
        (
            "baseline_qknorm",
            Path(baseline_qknorm_path),
            expected_baseline_qknorm_sha256,
        ),
        (
            "baseline_no_qknorm",
            Path(baseline_no_qknorm_path),
            expected_baseline_no_qknorm_sha256,
        ),
        (
            "candidate_qknorm",
            Path(candidate_qknorm_path),
            expected_candidate_qknorm_sha256,
        ),
        (
            "candidate_no_qknorm",
            Path(candidate_no_qknorm_path),
            expected_candidate_no_qknorm_sha256,
        ),
    )
    payloads = []
    inputs = {}
    for name, path, expected_digest in entries:
        digest = _require_sha256(expected_digest, name=f"{name} result SHA-256")
        if sha256_file(path) != digest:
            raise SSMaxDownstreamContractError(f"result SHA-256 mismatch for {path}")
        with path.open() as file_handle:
            payloads.append(json.load(file_handle))
        inputs[name] = {"path": str(path), "sha256": digest}
    comparison = compare_downstream_trajectory(*payloads)
    comparison["inputs"] = inputs
    return comparison
