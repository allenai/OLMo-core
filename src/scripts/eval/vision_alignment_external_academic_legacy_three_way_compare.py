"""Strictly compare legacy Stage-1, VA-12k, and VA-16k academic receipts on CPU.

Each input is first admitted by its public evaluator validator.  This comparator then requires
one shared frozen panel, tokenizer, prompt/control protocol, task selection, row ordering, and
control-input tokenization across all three checkpoints.  It reuses the frozen two-way
comparator's paired image-cluster bootstrap implementation and reports all checkpoint scores,
the two paired vision-alignment deltas against legacy Stage-1, and the retained VA-16k versus
VA-12k delta.

The output is descriptive historical-comparison evidence only.  In particular, the manifest's
alignment-overlap flags describe the vision-alignment training union, not the legacy Stage-1
training mixture, and therefore are not a legacy-data contamination screen.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import vision_alignment_external_academic_compare as frozen

FORMAT = "vision_alignment_external_academic_legacy_three_way_comparison"
SCHEMA_VERSION = 1
PROTOCOL_NAME = "vision-alignment-external-academic-legacy-va12k-va16k-comparison-v1"

LEGACY_KEY = "legacy_stage1_step32000"
STEP12_KEY = "vision_alignment_step12000"
STEP16_KEY = "vision_alignment_step16000"
INPUT_KEYS = (LEGACY_KEY, STEP12_KEY, STEP16_KEY)
MODEL_OUTPUT_KEYS = ("legacy_stage1_step32000", "step12000", "step16000")
CURRENT_STEPS = (12_000, 16_000)
CONTROLS = frozen.CONTROLS
DEFAULT_BOOTSTRAP_SEED = frozen.DEFAULT_BOOTSTRAP_SEED
DEFAULT_BOOTSTRAP_SAMPLES = frozen.DEFAULT_BOOTSTRAP_SAMPLES
CONFIDENCE_LEVEL = frozen.CONFIDENCE_LEVEL

EXPECTED_CURRENT_FORMAT = frozen.EXPECTED_INPUT_FORMAT
EXPECTED_CURRENT_PROTOCOL = frozen.EXPECTED_INPUT_PROTOCOL
EXPECTED_LEGACY_FORMAT = "vision_alignment_external_academic_legacy_stage1_receipt"
EXPECTED_LEGACY_PROTOCOL = "vision-alignment-external-academic-legacy-stage1-ep8-v1"
EXPECTED_LEGACY_STEP = 32_000

_TASK_METRICS = frozen._TASK_METRICS
_BINARY_METRIC_TASKS = frozen._BINARY_METRIC_TASKS
_FREE_ANSWER_TASKS = frozen._FREE_ANSWER_TASKS
_STATIC_ROW_FIELDS = frozen._STATIC_ROW_FIELDS
_CONTROL_INPUT_FIELDS = (
    "input_tokens",
    "image_grid_signature",
    "image_token_count",
    "image_token_ids_sha256",
)
_REFERENCE_FIELDS = {"path", "bytes", "sha256", "content_sha256"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_CURRENT_EVALUATOR: ModuleType | None = None
_LEGACY_EVALUATOR: ModuleType | None = None


def _load_module(filename: str, name: str) -> ModuleType:
    path = Path(__file__).resolve().with_name(filename)
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load comparator dependency {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_current_evaluator() -> ModuleType:
    global _CURRENT_EVALUATOR
    if _CURRENT_EVALUATOR is None:
        module = _load_module(
            "vision_alignment_external_academic.py",
            "_vision_alignment_external_academic_for_legacy_three_way_comparator",
        )
        if not callable(getattr(module, "validate_external_academic_receipt", None)):
            raise RuntimeError("Current academic evaluator lacks its public receipt validator")
        _CURRENT_EVALUATOR = module
    return _CURRENT_EVALUATOR


def _load_legacy_evaluator() -> ModuleType:
    global _LEGACY_EVALUATOR
    if _LEGACY_EVALUATOR is None:
        module = _load_module(
            "vision_alignment_external_academic_legacy_stage1.py",
            "_vision_alignment_external_academic_legacy_for_three_way_comparator",
        )
        if not callable(getattr(module, "validate_legacy_stage1_receipt", None)):
            raise RuntimeError("Legacy academic evaluator lacks its public receipt validator")
        _LEGACY_EVALUATOR = module
    return _LEGACY_EVALUATOR


def _repo_file_identity(path: Path, *, name: str) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo_root).as_posix()
    except ValueError as error:
        raise ValueError(f"{name} is outside its repository") from error
    identity = frozen._file_identity(resolved, name=name)
    return {
        "repo_relative_path": relative,
        "bytes": identity["bytes"],
        "sha256": identity["sha256"],
    }


def _implementation_identity() -> dict[str, Any]:
    directory = Path(__file__).resolve().parent
    files = {
        "comparator": _repo_file_identity(Path(__file__), name="three-way comparator"),
        "frozen_two_way_comparator": _repo_file_identity(
            directory / "vision_alignment_external_academic_compare.py",
            name="frozen two-way comparator",
        ),
        "current_evaluator": _repo_file_identity(
            directory / "vision_alignment_external_academic.py",
            name="current academic evaluator",
        ),
        "legacy_evaluator_wrapper": _repo_file_identity(
            directory / "vision_alignment_external_academic_legacy_stage1.py",
            name="legacy academic evaluator wrapper",
        ),
    }
    return {"files": files, "files_sha256": frozen._canonical_sha256(files)}


def _input_reference(
    path: Path, expected_sha256: str, receipt: Mapping[str, Any], *, name: str
) -> dict[str, Any]:
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError(f"Expected {name} SHA-256 must be lowercase hex")
    identity = frozen._file_identity(path, name=name)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{name} raw SHA-256 differs")
    content_sha256 = receipt.get("content_sha256")
    if not isinstance(content_sha256, str) or _SHA256_RE.fullmatch(content_sha256) is None:
        raise ValueError(f"{name} content SHA-256 is invalid")
    return {**identity, "content_sha256": content_sha256}


def _validate_reference(reference: Mapping[str, Any], *, name: str) -> tuple[str, str]:
    if set(reference) != _REFERENCE_FIELDS:
        raise ValueError(f"{name} input reference fields differ")
    path = reference.get("path")
    sha256 = reference.get("sha256")
    if not isinstance(path, str) or not isinstance(sha256, str):
        raise TypeError(f"{name} input path or SHA-256 is invalid")
    identity = frozen._file_identity(Path(path), name=name)
    if any(identity[field] != reference[field] for field in ("path", "bytes", "sha256")):
        raise ValueError(f"{name} raw identity differs")
    return path, sha256


def _load_validated_input(
    key: str,
    reference: Mapping[str, Any],
    *,
    verify_live: bool,
    hf_cache: str | None,
) -> dict[str, Any]:
    path, sha256 = _validate_reference(reference, name=key)
    if key == LEGACY_KEY:
        validator = _load_legacy_evaluator().validate_legacy_stage1_receipt
        receipt = validator(path, sha256, verify_live=verify_live, hf_cache=hf_cache)
    elif key in (STEP12_KEY, STEP16_KEY):
        validator = _load_current_evaluator().validate_external_academic_receipt
        receipt = validator(path, sha256, verify_live=verify_live)
    else:
        raise ValueError(f"Unknown comparison input {key!r}")
    if not isinstance(receipt, dict):
        raise TypeError(f"Public validator returned a non-object for {key}")
    if receipt.get("content_sha256") != reference.get("content_sha256"):
        raise ValueError(f"{key} semantic identity differs")
    return receipt


def _normalized_manifest(receipt: Mapping[str, Any]) -> dict[str, Any]:
    manifest = receipt.get("manifest")
    if not isinstance(manifest, Mapping):
        raise TypeError("Academic receipt manifest reference is invalid")
    fields = ("path", "bytes", "sha256", "content_sha256", "partial", "panel_status")
    if any(field not in manifest for field in fields):
        raise ValueError("Academic receipt manifest reference is incomplete")
    return {field: manifest[field] for field in fields}


def _normalized_tokenizer(receipt: Mapping[str, Any]) -> dict[str, Any]:
    tokenizer = receipt.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise TypeError("Academic receipt tokenizer is invalid")
    fields = (
        "identifier",
        "revision",
        "fingerprint",
        "eos_token_id",
        "pad_token_id",
        "token_ids",
        "token_ids_sha256",
    )
    if any(field not in tokenizer for field in fields):
        raise ValueError("Academic receipt tokenizer identity is incomplete")
    normalized = {field: tokenizer[field] for field in fields}
    if (
        not isinstance(normalized["identifier"], str)
        or not isinstance(normalized["revision"], str)
        or not isinstance(normalized["fingerprint"], str)
        or _SHA256_RE.fullmatch(str(normalized["fingerprint"])) is None
        or _SHA256_RE.fullmatch(str(normalized["token_ids_sha256"])) is None
        or not isinstance(normalized["token_ids"], dict)
        or frozen._canonical_sha256(normalized["token_ids"]) != normalized["token_ids_sha256"]
    ):
        raise ValueError("Academic receipt tokenizer identity is invalid")
    return normalized


def _current_step(receipt: Mapping[str, Any]) -> int:
    prior = receipt.get("prior_matched_wrong_v2")
    checkpoint = receipt.get("checkpoint")
    if not isinstance(prior, Mapping) or not isinstance(checkpoint, Mapping):
        raise TypeError("Current academic receipt lacks checkpoint provenance")
    step = prior.get("step")
    path = checkpoint.get("checkpoint")
    if type(step) is not int or step not in CURRENT_STEPS or not isinstance(path, str):
        raise ValueError("Current academic receipt checkpoint step is invalid")
    if Path(path).name != f"step{step}":
        raise ValueError("Current academic receipt checkpoint path and step differ")
    return step


def _legacy_step(receipt: Mapping[str, Any]) -> int:
    checkpoint = receipt.get("checkpoint")
    lineage = receipt.get("legacy_stage1_lineage")
    if not isinstance(checkpoint, Mapping) or not isinstance(lineage, Mapping):
        raise TypeError("Legacy academic receipt lacks checkpoint lineage")
    path = checkpoint.get("checkpoint")
    maximum_steps = lineage.get("maximum_steps")
    if (
        not isinstance(path, str)
        or Path(path).name != f"step{EXPECTED_LEGACY_STEP}"
        or maximum_steps != EXPECTED_LEGACY_STEP
    ):
        raise ValueError("Legacy academic receipt is not the Stage-1 step-32000 endpoint")
    return EXPECTED_LEGACY_STEP


def _validate_artifact_policies(receipts: Mapping[str, Mapping[str, Any]]) -> None:
    current_policy = {
        "descriptive_only": True,
        "promotion_eligible": False,
        "checkpoint_selection_evidence": True,
    }
    legacy_policy = {
        "descriptive_only": True,
        "promotion_eligible": False,
        "historical_reference_comparison_evidence": True,
    }
    if receipts[LEGACY_KEY].get("artifact_policy") != legacy_policy:
        raise ValueError("Legacy academic input must remain descriptive-only evidence")
    if any(
        receipts[key].get("artifact_policy") != current_policy for key in (STEP12_KEY, STEP16_KEY)
    ):
        raise ValueError("Vision-alignment academic inputs must remain descriptive-only evidence")


def _validate_inputs(receipts: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    if tuple(receipts) != INPUT_KEYS:
        raise ValueError("Comparison requires ordered legacy, VA-12k, and VA-16k inputs")
    legacy_receipt = receipts[LEGACY_KEY]
    step12 = receipts[STEP12_KEY]
    step16 = receipts[STEP16_KEY]
    if (
        legacy_receipt.get("schema_version") != 1
        or legacy_receipt.get("format") != EXPECTED_LEGACY_FORMAT
        or legacy_receipt.get("protocol_name") != EXPECTED_LEGACY_PROTOCOL
        or _legacy_step(legacy_receipt) != EXPECTED_LEGACY_STEP
    ):
        raise ValueError("Legacy academic input protocol identity differs")
    for expected_step, receipt in zip(CURRENT_STEPS, (step12, step16)):
        if (
            receipt.get("schema_version") != 1
            or receipt.get("format") != EXPECTED_CURRENT_FORMAT
            or receipt.get("protocol_name") != EXPECTED_CURRENT_PROTOCOL
            or _current_step(receipt) != expected_step
        ):
            raise ValueError(f"VA-{expected_step // 1000}k academic input identity differs")
    for field in ("schema_version", "format", "protocol_name", "git", "implementation"):
        if step12.get(field) != step16.get(field):
            raise ValueError(f"Vision-alignment academic receipt {field} differs")
    legacy_implementation = legacy_receipt.get("implementation")
    if not isinstance(legacy_implementation, Mapping) or legacy_implementation.get(
        "frozen_evaluator"
    ) != step12.get("implementation"):
        raise ValueError("Legacy wrapper and current receipts use different frozen evaluators")
    legacy_manifest = legacy_receipt.get("manifest")
    if not isinstance(legacy_manifest, Mapping) or legacy_manifest.get("builder_git") != step12.get(
        "git"
    ):
        raise ValueError("Legacy and current receipts identify different manifest-builder Git")
    _validate_artifact_policies(receipts)

    manifests = [_normalized_manifest(receipts[key]) for key in INPUT_KEYS]
    if any(value != manifests[0] for value in manifests[1:]):
        raise ValueError("Academic manifest identity differs across all three checkpoints")
    tokenizers = [_normalized_tokenizer(receipts[key]) for key in INPUT_KEYS]
    if any(value != tokenizers[0] for value in tokenizers[1:]):
        raise ValueError("Evaluation tokenizer identity differs across all three checkpoints")
    protocols = [receipts[key].get("protocol") for key in INPUT_KEYS]
    if not isinstance(protocols[0], Mapping) or any(
        value != protocols[0] for value in protocols[1:]
    ):
        raise ValueError("Prompt/control protocol identity differs across all three checkpoints")
    task_order = protocols[0].get("tasks")
    if not isinstance(task_order, list) or tuple(task_order) != tuple(_TASK_METRICS):
        raise ValueError("Academic task order differs from the reviewed panel")
    if protocols[0].get("controls") != list(CONTROLS) or not isinstance(
        protocols[0].get("prompt"), Mapping
    ):
        raise ValueError("Academic prompt/control protocol is incomplete")
    for key in INPUT_KEYS:
        tasks = receipts[key].get("tasks")
        if not isinstance(tasks, dict) or set(tasks) != set(task_order):
            raise ValueError(f"{key} task coverage differs from the reviewed panel")
    return tuple(task_order)


def _number(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    output = float(value)
    if not math.isfinite(output) or not 0.0 <= output <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return output


RowTriplet = tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]


def _paired_rows(receipts: Mapping[str, Mapping[str, Any]], task: str) -> list[RowTriplet]:
    payloads = [receipts[key]["tasks"][task] for key in INPUT_KEYS]
    expected_metric = _TASK_METRICS[task]
    for key, payload in zip(INPUT_KEYS, payloads):
        if not isinstance(payload, Mapping) or payload.get("metric") != expected_metric:
            raise ValueError(f"{key} task {task!r} metric differs")
        if any(
            payload.get(field) != payloads[0].get(field)
            for field in ("source", "selection_count", "selection_sha256")
        ):
            raise ValueError(f"Task {task!r} selection identity differs across checkpoints")
    row_lists = [payload.get("examples") for payload in payloads]
    if any(not isinstance(rows, list) or not rows for rows in row_lists):
        raise ValueError(f"Task {task!r} lacks complete example rows")
    expected_count = payloads[0].get("selection_count")
    if any(len(rows) != expected_count for rows in row_lists if isinstance(rows, list)):
        raise ValueError(f"Task {task!r} example coverage differs")

    triplets: list[RowTriplet] = []
    for index, rows in enumerate(zip(*row_lists)):
        if len(rows) != len(INPUT_KEYS) or any(not isinstance(row, Mapping) for row in rows):
            raise TypeError(f"Task {task!r} row {index} is invalid")
        legacy_row, step12_row, step16_row = rows
        if any(
            row.get(field) != legacy_row.get(field)
            for row in (step12_row, step16_row)
            for field in _STATIC_ROW_FIELDS
        ):
            raise ValueError(f"Task {task!r} row {index} identity or ordering differs")
        for image_field in ("image_sha256", "shuffled_image_sha256"):
            value = legacy_row.get(image_field)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"Task {task!r} row {index} image identity is invalid")
        for key, row in zip(INPUT_KEYS, rows):
            controls = row.get("controls")
            if not isinstance(controls, dict) or set(controls) != set(CONTROLS):
                raise ValueError(f"{key} task {task!r} row {index} controls differ")
            for control in CONTROLS:
                output = controls[control]
                if not isinstance(output, Mapping):
                    raise TypeError(f"{key} task {task!r}/{index}/{control} is invalid")
                _number(output.get("score"), name=f"{key} {task}/{index}/{control} score")
                if any(
                    output.get(field) != legacy_row["controls"][control].get(field)
                    for field in _CONTROL_INPUT_FIELDS
                ):
                    raise ValueError(
                        f"Task {task!r} row {index}/{control} prompt/control input identity differs"
                    )
        tokenization = legacy_row["controls"]
        for control in CONTROLS:
            output = tokenization[control]
            if (
                type(output.get("input_tokens")) is not int
                or output["input_tokens"] <= 0
                or not isinstance(output.get("image_grid_signature"), list)
                or len(output["image_grid_signature"]) != 4
                or any(
                    type(value) is not int or value <= 0 for value in output["image_grid_signature"]
                )
                or type(output.get("image_token_count")) is not int
                or output["image_token_count"] <= 0
                or not isinstance(output.get("image_token_ids_sha256"), str)
                or _SHA256_RE.fullmatch(output["image_token_ids_sha256"]) is None
            ):
                raise ValueError(f"Task {task!r} row {index}/{control} tokenization is invalid")
        triplets.append((legacy_row, step12_row, step16_row))
    return triplets


def _score(row: Mapping[str, Any], control: str) -> float:
    return float(row["controls"][control]["score"])


def _vectors(
    values: Sequence[Sequence[float]], *, model_suffix: str = ""
) -> dict[str, Sequence[float]]:
    legacy_values, step12_values, step16_values = values
    return {
        f"{MODEL_OUTPUT_KEYS[0]}{model_suffix}": legacy_values,
        f"{MODEL_OUTPUT_KEYS[1]}{model_suffix}": step12_values,
        f"{MODEL_OUTPUT_KEYS[2]}{model_suffix}": step16_values,
        "paired_delta_12000_minus_legacy_stage1": [
            current - old for old, current in zip(legacy_values, step12_values)
        ],
        "paired_delta_16000_minus_legacy_stage1": [
            current - old for old, current in zip(legacy_values, step16_values)
        ],
        "paired_delta_16000_minus_12000": [
            right - left for left, right in zip(step12_values, step16_values)
        ],
    }


def _mcnemar_pair(
    first: Sequence[float],
    second: Sequence[float],
    *,
    first_name: str,
    second_name: str,
    binary_metric: bool,
) -> dict[str, Any]:
    if len(first) != len(second) or not first:
        raise ValueError("McNemar inputs must be non-empty and paired")
    first_correct = [score == 1.0 for score in first]
    second_correct = [score == 1.0 for score in second]
    both = sum(left and right for left, right in zip(first_correct, second_correct))
    first_only = sum(left and not right for left, right in zip(first_correct, second_correct))
    second_only = sum(not left and right for left, right in zip(first_correct, second_correct))
    neither = len(first) - both - first_only - second_only
    discordant = first_only + second_only
    if discordant:
        tail = sum(
            math.comb(discordant, value) for value in range(min(first_only, second_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "outcome": "per-example score equals exactly 1.0",
        "interpretation": (
            "standard_binary_metric_mcnemar"
            if binary_metric
            else "dichotomized_perfect_score_mcnemar_diagnostic"
        ),
        "examples": len(first),
        "both_correct": both,
        f"{first_name}_only": first_only,
        f"{second_name}_only": second_only,
        "neither_correct": neither,
        "discordant": discordant,
        "exact_two_sided_p_value": float(p_value),
    }


def _mcnemar_all(values: Sequence[Sequence[float]], *, task: str) -> dict[str, Any]:
    legacy_values, step12_values, step16_values = values
    kwargs = {"binary_metric": task in _BINARY_METRIC_TASKS}
    return {
        "step12000_vs_legacy_stage1": _mcnemar_pair(
            legacy_values,
            step12_values,
            first_name="legacy_stage1_step32000",
            second_name="step12000",
            **kwargs,
        ),
        "step16000_vs_legacy_stage1": _mcnemar_pair(
            legacy_values,
            step16_values,
            first_name="legacy_stage1_step32000",
            second_name="step16000",
            **kwargs,
        ),
        "step16000_vs_step12000": _mcnemar_pair(
            step12_values,
            step16_values,
            first_name="step12000",
            second_name="step16000",
            **kwargs,
        ),
    }


def _select_panel(
    pairs: Sequence[RowTriplet], *, exact_byte_nonoverlap: bool
) -> tuple[list[RowTriplet], list[RowTriplet]]:
    recipients = [
        pair
        for pair in pairs
        if not exact_byte_nonoverlap or not pair[0]["alignment_train_image_overlap"]
    ]
    shuffled = [
        pair
        for pair in recipients
        if not exact_byte_nonoverlap or not pair[0]["shuffled_alignment_train_image_overlap"]
    ]
    return recipients, shuffled


def _panel(
    pairs: Sequence[RowTriplet],
    *,
    task: str,
    label: str,
    seed: int,
    samples: int,
    exact_byte_nonoverlap: bool,
) -> dict[str, Any]:
    recipients, shuffled = _select_panel(pairs, exact_byte_nonoverlap=exact_byte_nonoverlap)
    if not recipients or not shuffled:
        raise ValueError(f"Task {task!r}/{label} has no exact-byte-nonoverlap examples")

    def summarize(
        selected: Sequence[RowTriplet],
        values: Sequence[Sequence[float]],
        suffix: str,
        *,
        model_suffix: str = "",
    ) -> dict[str, dict[str, Any]]:
        return frozen._cluster_bootstrap(
            _vectors(values, model_suffix=model_suffix),
            [str(pair[0]["image_sha256"]) for pair in selected],
            seed=frozen._derived_seed(seed, f"{task}:{label}:{suffix}"),
            samples=samples,
        )

    correct_values = [
        [_score(pair[index], "correct") for pair in recipients] for index in range(len(INPUT_KEYS))
    ]
    correct = summarize(recipients, correct_values, "correct")
    image_use: dict[str, Any] = {}
    for control, selected in (("shuffled", shuffled), ("blank", recipients)):
        gap_values = [
            [_score(pair[index], "correct") - _score(pair[index], control) for pair in selected]
            for index in range(len(INPUT_KEYS))
        ]
        image_use[control] = summarize(
            selected,
            gap_values,
            control,
            model_suffix="_correct_minus_control",
        )
        image_use[control]["eligibility"] = (
            "recipient_and_shuffled_donor_exact_byte_nonoverlap"
            if exact_byte_nonoverlap and control == "shuffled"
            else (
                "recipient_exact_byte_nonoverlap"
                if exact_byte_nonoverlap
                else "all_frozen_examples"
            )
        )
    generation_cap: dict[str, Any] | None = None
    if task in _FREE_ANSWER_TASKS:
        generation_cap = {}
        for control, selected in (
            ("correct", recipients),
            ("shuffled", shuffled),
            ("blank", recipients),
        ):
            rates: list[float] = []
            control_payload: dict[str, Any] = {}
            for index, name in enumerate(MODEL_OUTPUT_KEYS):
                capped = [
                    pair[index]["controls"][control].get("stop_reason") == "max_tokens"
                    for pair in selected
                ]
                rate = float(sum(capped) / len(capped))
                rates.append(rate)
                control_payload[name] = {
                    "examples": len(capped),
                    "max_tokens": sum(capped),
                    "rate": rate,
                }
            control_payload.update(
                {
                    "rate_delta_12000_minus_legacy_stage1": rates[1] - rates[0],
                    "rate_delta_16000_minus_legacy_stage1": rates[2] - rates[0],
                    "rate_delta_16000_minus_12000": rates[2] - rates[1],
                }
            )
            generation_cap[control] = control_payload
    return {
        "eligibility": (
            "exact_encoded_image_byte_nonoverlap"
            if exact_byte_nonoverlap
            else "all_frozen_examples"
        ),
        "correct_accuracy": {
            **correct,
            "mcnemar_exact_score_one": _mcnemar_all(correct_values, task=task),
        },
        "image_use": image_use,
        "generation_cap": generation_cap,
    }


def _task_comparison(
    pairs: Sequence[RowTriplet], *, task: str, seed: int, samples: int
) -> dict[str, Any]:
    strata = sorted(
        {str(pair[0]["stratum"]) for pair in pairs if pair[0].get("stratum") is not None}
    )

    def panel(selected: Sequence[RowTriplet], label: str, exact: bool) -> dict[str, Any]:
        return _panel(
            selected,
            task=task,
            label=label,
            seed=seed,
            samples=samples,
            exact_byte_nonoverlap=exact,
        )

    return {
        "metric": _TASK_METRICS[task],
        "metric_scale": "higher_is_better_accuracy_in_[0,1]",
        "example_order_sha256": frozen._canonical_sha256([pair[0]["example_id"] for pair in pairs]),
        "control_input_identity_sha256": frozen._canonical_sha256(
            [
                {
                    "example_id": pair[0]["example_id"],
                    "controls": {
                        control: {
                            field: pair[0]["controls"][control][field]
                            for field in _CONTROL_INPUT_FIELDS
                        }
                        for control in CONTROLS
                    },
                }
                for pair in pairs
            ]
        ),
        "all_examples": panel(pairs, "all", False),
        "exact_byte_nonoverlap": panel(pairs, "exact_byte_nonoverlap", True),
        "strata": {
            stratum: {
                "all_examples": panel(
                    [pair for pair in pairs if pair[0]["stratum"] == stratum],
                    f"stratum:{stratum}:all",
                    False,
                ),
                "exact_byte_nonoverlap": panel(
                    [pair for pair in pairs if pair[0]["stratum"] == stratum],
                    f"stratum:{stratum}:exact_byte_nonoverlap",
                    True,
                ),
            }
            for stratum in strata
        },
    }


def _macro_comparison(
    paired_by_task: Mapping[str, Sequence[RowTriplet]],
    *,
    seed: int,
    samples: int,
    exact_byte_nonoverlap: bool,
) -> dict[str, Any]:
    metric_names = (
        "correct_legacy_stage1_step32000",
        "correct_step12000",
        "correct_step16000",
        "correct_delta_12000_minus_legacy_stage1",
        "correct_delta_16000_minus_legacy_stage1",
        "correct_delta_16000_minus_12000",
        "correct_minus_shuffled_legacy_stage1_step32000",
        "correct_minus_shuffled_step12000",
        "correct_minus_shuffled_step16000",
        "correct_minus_shuffled_delta_12000_minus_legacy_stage1",
        "correct_minus_shuffled_delta_16000_minus_legacy_stage1",
        "correct_minus_shuffled_delta_16000_minus_12000",
        "correct_minus_blank_legacy_stage1_step32000",
        "correct_minus_blank_step12000",
        "correct_minus_blank_step16000",
        "correct_minus_blank_delta_12000_minus_legacy_stage1",
        "correct_minus_blank_delta_16000_minus_legacy_stage1",
        "correct_minus_blank_delta_16000_minus_12000",
    )
    task_vectors: dict[str, dict[str, tuple[list[float], list[str]]]] = {
        name: {} for name in metric_names
    }

    def add_vectors(
        task: str,
        selected: Sequence[RowTriplet],
        *,
        prefix: str,
        value: Callable[[Mapping[str, Any]], float],
    ) -> None:
        if not selected:
            return
        clusters = [str(pair[0]["image_sha256"]) for pair in selected]
        values = [[value(pair[index]) for pair in selected] for index in range(len(INPUT_KEYS))]
        legacy_values, step12_values, step16_values = values
        task_vectors[f"{prefix}_legacy_stage1_step32000"][task] = (legacy_values, clusters)
        task_vectors[f"{prefix}_step12000"][task] = (step12_values, clusters)
        task_vectors[f"{prefix}_step16000"][task] = (step16_values, clusters)
        task_vectors[f"{prefix}_delta_12000_minus_legacy_stage1"][task] = (
            [right - left for left, right in zip(legacy_values, step12_values)],
            clusters,
        )
        task_vectors[f"{prefix}_delta_16000_minus_legacy_stage1"][task] = (
            [right - left for left, right in zip(legacy_values, step16_values)],
            clusters,
        )
        task_vectors[f"{prefix}_delta_16000_minus_12000"][task] = (
            [right - left for left, right in zip(step12_values, step16_values)],
            clusters,
        )

    for task, pairs in paired_by_task.items():
        recipients, shuffled = _select_panel(pairs, exact_byte_nonoverlap=exact_byte_nonoverlap)
        add_vectors(task, recipients, prefix="correct", value=lambda row: _score(row, "correct"))
        add_vectors(
            task,
            shuffled,
            prefix="correct_minus_shuffled",
            value=lambda row: _score(row, "correct") - _score(row, "shuffled"),
        )
        add_vectors(
            task,
            recipients,
            prefix="correct_minus_blank",
            value=lambda row: _score(row, "correct") - _score(row, "blank"),
        )
    label_prefix = "exact_byte_nonoverlap" if exact_byte_nonoverlap else "all_examples"
    return {
        "metric_compatibility": (
            "Only per-example higher-is-better scores bounded to [0,1] are included; task "
            "means receive equal weight despite different benchmark metrics."
        ),
        "statistics": {
            name: frozen._macro_stat(
                vectors,
                seed=seed,
                samples=samples,
                label=f"{label_prefix}:{name}",
            )
            for name, vectors in task_vectors.items()
        },
    }


def _shared_identity(
    receipts: Mapping[str, Mapping[str, Any]],
    task_order: Sequence[str],
    tasks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    protocol = receipts[LEGACY_KEY]["protocol"]
    manifest = _normalized_manifest(receipts[LEGACY_KEY])
    tokenizer = _normalized_tokenizer(receipts[LEGACY_KEY])
    return {
        "manifest": manifest,
        "tokenizer": tokenizer,
        "protocol": protocol,
        "protocol_sha256": frozen._canonical_sha256(protocol),
        "prompt_sha256": frozen._canonical_sha256(protocol["prompt"]),
        "controls_sha256": frozen._canonical_sha256(protocol["controls"]),
        "task_order": list(task_order),
        "example_order_sha256": {task: tasks[task]["example_order_sha256"] for task in task_order},
        "control_input_identity_sha256": {
            task: tasks[task]["control_input_identity_sha256"] for task in task_order
        },
        "evaluator_identities": {
            LEGACY_KEY: receipts[LEGACY_KEY]["implementation"],
            STEP12_KEY: receipts[STEP12_KEY]["implementation"],
            STEP16_KEY: receipts[STEP16_KEY]["implementation"],
        },
    }


def _build_comparison_receipt(
    receipts: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
    created_at: str,
) -> dict[str, Any]:
    if type(bootstrap_seed) is not int:
        raise TypeError("Bootstrap seed must be an integer")
    if type(bootstrap_samples) is not int or bootstrap_samples <= 0:
        raise ValueError("Bootstrap samples must be a positive integer")
    frozen._validate_timestamp(created_at, name="three-way comparison created_at")
    if tuple(references) != INPUT_KEYS:
        raise ValueError("Three-way comparison input references differ")
    if any(set(references[key]) != _REFERENCE_FIELDS for key in INPUT_KEYS):
        raise ValueError("Three-way comparison input reference fields differ")
    task_order = _validate_inputs(receipts)
    paired_by_task = {task: _paired_rows(receipts, task) for task in task_order}
    tasks = {
        task: _task_comparison(
            pairs,
            task=task,
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        )
        for task, pairs in paired_by_task.items()
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "format": FORMAT,
        "protocol_name": PROTOCOL_NAME,
        "created_at": created_at,
        "implementation": _implementation_identity(),
        "inputs": {key: dict(references[key]) for key in INPUT_KEYS},
        "shared_evaluation_identity": _shared_identity(receipts, task_order, tasks),
        "bootstrap": {
            "method": "paired percentile cluster bootstrap",
            "confidence_level": CONFIDENCE_LEVEL,
            "cluster_unit": "recipient encoded-image-byte SHA-256 within task",
            "prng": "numpy.default_rng(PCG64)",
            "percentile_interpolation": "linear",
            "seed": bootstrap_seed,
            "samples": bootstrap_samples,
            "implementation": "frozen vision_alignment_external_academic_compare.py helpers",
        },
        "directions": [
            "step12000_minus_legacy_stage1_step32000",
            "step16000_minus_legacy_stage1_step32000",
            "step16000_minus_step12000",
        ],
        "tasks": tasks,
        "equal_task_macro": {
            "all_examples": _macro_comparison(
                paired_by_task,
                seed=bootstrap_seed,
                samples=bootstrap_samples,
                exact_byte_nonoverlap=False,
            ),
            "exact_byte_nonoverlap": _macro_comparison(
                paired_by_task,
                seed=bootstrap_seed,
                samples=bootstrap_samples,
                exact_byte_nonoverlap=True,
            ),
        },
        "interpretation_limits": {
            "overlap_inventory_scope": (
                "alignment_train_image_overlap flags refer only to the later vision-alignment "
                "training union and are not a legacy Stage-1 training-contamination screen"
            ),
            "benchmark_scope": (
                "sampled local validation projections; descriptive evidence, not official "
                "leaderboard submissions"
            ),
            "historical_tokenizer_scope": (
                "the exact tokenizer pin is the shared evaluation tokenizer; the legacy "
                "training config did not record its revision or fingerprint"
            ),
        },
        "policy": {
            "conclusion": "descriptive_only",
            "promotion_eligible": False,
            "automatic_checkpoint_selection": False,
            "promotion_decision": None,
            "intended_use": (
                "human-reviewed historical comparison of the legacy Stage-1 endpoint with "
                "vision-alignment checkpoints"
            ),
        },
    }
    return frozen._attach_content_sha256(payload)


def _validated_inputs_from_references(
    references: Any,
    *,
    verify_live: bool,
    hf_cache: str | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Mapping[str, Any]]]:
    if not isinstance(references, dict) or tuple(references) != INPUT_KEYS:
        raise ValueError("Three-way comparison input ordering differs")
    receipts: dict[str, dict[str, Any]] = {}
    normalized: dict[str, Mapping[str, Any]] = {}
    for key in INPUT_KEYS:
        reference = references[key]
        if not isinstance(reference, Mapping):
            raise TypeError(f"Comparison {key} input reference is invalid")
        receipts[key] = _load_validated_input(
            key,
            reference,
            verify_live=verify_live,
            hf_cache=hf_cache,
        )
        normalized[key] = reference
    return receipts, normalized


def validate_academic_legacy_three_way_comparison_receipt(
    path: str | Path,
    expected_sha256: str,
    *,
    verify_live_inputs: bool = True,
    hf_cache: str | None = None,
) -> dict[str, Any]:
    """Strictly reload and fully rederive a legacy/VA-12k/VA-16k comparison receipt.

    :param path: Canonical three-way comparison receipt path.
    :param expected_sha256: Independently supplied raw comparison SHA-256.
    :param verify_live_inputs: Ask all public input validators to rehash live provenance.
    :param hf_cache: Optional local cache used by the legacy receipt's tokenizer validator.
    :returns: The validated three-way comparison receipt.
    """
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError("Expected comparison SHA-256 must be lowercase hex")
    receipt, identity = frozen._load_json(Path(path), name="legacy three-way comparison receipt")
    if identity["sha256"] != expected_sha256:
        raise ValueError("Legacy three-way comparison raw SHA-256 differs")
    frozen._verify_content_sha256(receipt, name="legacy three-way comparison receipt")
    expected_fields = {
        "schema_version",
        "format",
        "protocol_name",
        "created_at",
        "implementation",
        "inputs",
        "shared_evaluation_identity",
        "bootstrap",
        "directions",
        "tasks",
        "equal_task_macro",
        "interpretation_limits",
        "policy",
        "content_sha256",
    }
    if set(receipt) != expected_fields:
        raise ValueError("Legacy three-way comparison receipt fields differ")
    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or receipt["format"] != FORMAT
        or receipt["protocol_name"] != PROTOCOL_NAME
        or receipt["implementation"] != _implementation_identity()
    ):
        raise ValueError("Legacy three-way comparison implementation or protocol differs")
    bootstrap = receipt.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise TypeError("Legacy three-way comparison bootstrap is invalid")
    seed = bootstrap.get("seed")
    samples = bootstrap.get("samples")
    if type(seed) is not int or type(samples) is not int or samples <= 0:
        raise ValueError("Legacy three-way comparison bootstrap differs")
    receipts, references = _validated_inputs_from_references(
        receipt["inputs"],
        verify_live=verify_live_inputs,
        hf_cache=hf_cache,
    )
    expected = _build_comparison_receipt(
        receipts,
        references,
        bootstrap_seed=seed,
        bootstrap_samples=samples,
        created_at=receipt["created_at"],
    )
    if receipt != expected:
        raise ValueError("Legacy three-way comparison does not equal its full rederivation")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-stage1-receipt", required=True)
    parser.add_argument("--legacy-stage1-sha256", required=True)
    parser.add_argument("--step12000-receipt", required=True)
    parser.add_argument("--step12000-sha256", required=True)
    parser.add_argument("--step16000-receipt", required=True)
    parser.add_argument("--step16000-sha256", required=True)
    parser.add_argument("--hf-cache")
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    """Validate three inputs, build the descriptive comparison, and publish it once."""
    args = _parser().parse_args()
    output = frozen._direct_path(
        Path(args.output), name="legacy three-way comparison output", allow_missing_leaf=True
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite canonical artifact {output}")
    paths = {
        LEGACY_KEY: Path(args.legacy_stage1_receipt),
        STEP12_KEY: Path(args.step12000_receipt),
        STEP16_KEY: Path(args.step16000_receipt),
    }
    expected_hashes = {
        LEGACY_KEY: args.legacy_stage1_sha256,
        STEP12_KEY: args.step12000_sha256,
        STEP16_KEY: args.step16000_sha256,
    }
    receipts: dict[str, dict[str, Any]] = {}
    for key in INPUT_KEYS:
        if key == LEGACY_KEY:
            receipt = _load_legacy_evaluator().validate_legacy_stage1_receipt(
                paths[key],
                expected_hashes[key],
                verify_live=True,
                hf_cache=args.hf_cache,
            )
        else:
            receipt = _load_current_evaluator().validate_external_academic_receipt(
                paths[key], expected_hashes[key], verify_live=True
            )
        if not isinstance(receipt, dict):
            raise TypeError(f"Public validator returned a non-object for {key}")
        receipts[key] = receipt
    references = {
        key: _input_reference(
            paths[key], expected_hashes[key], receipts[key], name=f"{key} receipt"
        )
        for key in INPUT_KEYS
    }
    comparison = _build_comparison_receipt(
        receipts,
        references,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    validated, normalized = _validated_inputs_from_references(
        references, verify_live=True, hf_cache=args.hf_cache
    )
    rederived = _build_comparison_receipt(
        validated,
        normalized,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        created_at=comparison["created_at"],
    )
    if rederived != comparison:
        raise ValueError("Legacy three-way comparison changed before publication")
    frozen._write_json_no_overwrite(output, comparison)
    raw_sha256 = frozen._file_identity(output, name="legacy three-way comparison receipt")["sha256"]
    validate_academic_legacy_three_way_comparison_receipt(
        output,
        raw_sha256,
        verify_live_inputs=False,
        hf_cache=args.hf_cache,
    )
    print(f"wrote {output} sha256={raw_sha256}")


if __name__ == "__main__":
    main()
