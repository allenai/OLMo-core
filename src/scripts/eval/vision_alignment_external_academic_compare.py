"""Strictly compare step-12000 and step-16000 external-academic receipts on CPU.

The input receipts remain descriptive checkpoint-selection evidence.  This comparator invokes
their public validator, requires one shared frozen panel and protocol, rederives paired task and
image-control statistics, and writes a canonical, write-once comparison receipt.  It never loads
a model and never emits an automatic checkpoint promotion decision.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import sys
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

FORMAT = "vision_alignment_external_academic_comparison"
SCHEMA_VERSION = 1
PROTOCOL_NAME = "vision-alignment-external-academic-step12000-step16000-v1"
EXPECTED_INPUT_FORMAT = "vision_alignment_external_academic_receipt"
EXPECTED_INPUT_PROTOCOL = "vision-alignment-external-academic-ep8-v1"
STEPS = (12_000, 16_000)
STEP_KEYS = ("step12000", "step16000")
CONTROLS = ("correct", "shuffled", "blank")
DEFAULT_BOOTSTRAP_SEED = 26_081_600
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TASK_METRICS = {
    "vqav2": "vqa_accuracy",
    "textvqa": "vqa_accuracy",
    "docvqa": "anls",
    "chartqa": "relaxed_accuracy",
    "ai2d": "multiple_choice_accuracy",
    "a_okvqa_mc": "multiple_choice_accuracy",
}
_BINARY_METRIC_TASKS = frozenset(("chartqa", "ai2d", "a_okvqa_mc"))
_FREE_ANSWER_TASKS = frozenset(("vqav2", "textvqa", "docvqa", "chartqa"))
_STATIC_ROW_FIELDS = (
    "example_id",
    "source_position",
    "annotation_sha256",
    "image_sha256",
    "image_grid_signature",
    "image_token_count",
    "alignment_train_image_overlap",
    "shuffled_donor_id",
    "shuffled_image_sha256",
    "shuffled_image_grid_signature",
    "shuffled_alignment_train_image_overlap",
    "question",
    "gold_answers",
    "options",
    "gold_answer_index",
    "stratum",
)
_ACADEMIC_EVALUATOR: ModuleType | None = None


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _attach_content_sha256(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "content_sha256" in payload:
        raise ValueError("Payload already contains content_sha256")
    output = dict(payload)
    output["content_sha256"] = _canonical_sha256(output)
    return output


def _verify_content_sha256(payload: Mapping[str, Any], *, name: str) -> None:
    content_sha256 = payload.get("content_sha256")
    if not isinstance(content_sha256, str) or _SHA256_RE.fullmatch(content_sha256) is None:
        raise ValueError(f"{name} has an invalid content SHA-256")
    unsigned = dict(payload)
    del unsigned["content_sha256"]
    if _canonical_sha256(unsigned) != content_sha256:
        raise ValueError(f"{name} content SHA-256 differs")


def _validate_timestamp(value: Any, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} is not valid ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a UTC offset")


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"JSON repeats key {key!r}")
        output[key] = value
    return output


def _direct_path(path: Path, *, name: str, allow_missing_leaf: bool = False) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    components = (*reversed(absolute.parents), absolute)
    for index, component in enumerate(components):
        if component == Path(component.anchor):
            continue
        try:
            info = component.lstat()
        except FileNotFoundError:
            if allow_missing_leaf and index == len(components) - 1:
                continue
            if allow_missing_leaf:
                continue
            raise ValueError(f"{name} component is unavailable: {component}") from None
        except OSError as error:
            raise ValueError(f"{name} component is unavailable: {component}: {error}") from error
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{name} contains a symlinked component: {component}")
    return absolute


def _read_regular_file(path: Path, *, name: str) -> tuple[bytes, dict[str, Any]]:
    path = _direct_path(path, name=name)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{name} is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"Could not read {name} from {path}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    def signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    current = path.lstat()
    if signature(before) != signature(after) or signature(before) != signature(current):
        raise ValueError(f"{name} changed while it was read")
    raw = b"".join(chunks)
    return raw, {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _file_identity(path: Path, *, name: str) -> dict[str, Any]:
    return _read_regular_file(path, name=name)[1]


def _load_json(path: Path, *, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, identity = _read_regular_file(path, name=name)

    def reject_constant(value: str) -> Any:
        raise ValueError(f"{name} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_json_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not parse {name}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value, identity


def _write_json_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish canonical JSON while refusing existing paths and symlinks."""
    path = _direct_path(path, name="comparison output", allow_missing_leaf=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite canonical artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _direct_path(path, name="comparison output", allow_missing_leaf=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o644,
        )
        raw = _canonical_bytes(payload) + b"\n"
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(f"Refusing to overwrite canonical artifact {path}") from None
        directory_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _implementation_identity() -> dict[str, Any]:
    path = Path(__file__).resolve()
    repo_root = path.parents[3]
    try:
        relative = path.relative_to(repo_root).as_posix()
    except ValueError as error:
        raise ValueError("Comparator implementation is outside its repository") from error
    identity = _file_identity(path, name="comparator implementation")
    return {
        "repo_relative_path": relative,
        "bytes": identity["bytes"],
        "sha256": identity["sha256"],
    }


def _load_academic_evaluator() -> ModuleType:
    global _ACADEMIC_EVALUATOR
    if _ACADEMIC_EVALUATOR is not None:
        return _ACADEMIC_EVALUATOR
    path = Path(__file__).resolve().with_name("vision_alignment_external_academic.py")
    name = "_vision_alignment_external_academic_for_comparator"
    cached = sys.modules.get(name)
    if cached is not None:
        _ACADEMIC_EVALUATOR = cached
        return cached
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load external-academic validator {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "validate_external_academic_receipt", None)):
        raise RuntimeError("External-academic evaluator lacks its public receipt validator")
    _ACADEMIC_EVALUATOR = module
    return module


def _input_reference(
    path: Path, expected_sha256: str, receipt: Mapping[str, Any]
) -> dict[str, Any]:
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError("Expected input SHA-256 must be lowercase hex")
    identity = _file_identity(path, name="external-academic receipt")
    if identity["sha256"] != expected_sha256:
        raise ValueError("External-academic receipt raw SHA-256 differs")
    content_sha256 = receipt.get("content_sha256")
    if not isinstance(content_sha256, str) or _SHA256_RE.fullmatch(content_sha256) is None:
        raise ValueError("External-academic receipt content SHA-256 is invalid")
    return {**identity, "content_sha256": content_sha256}


def _load_validated_input(reference: Mapping[str, Any], *, verify_live: bool) -> dict[str, Any]:
    if set(reference) != {"path", "bytes", "sha256", "content_sha256"}:
        raise ValueError("Comparison input reference fields differ")
    path_value = reference.get("path")
    sha256 = reference.get("sha256")
    if not isinstance(path_value, str) or not isinstance(sha256, str):
        raise TypeError("Comparison input path or SHA-256 is invalid")
    identity = _file_identity(Path(path_value), name="external-academic receipt")
    if any(identity[field] != reference[field] for field in ("path", "bytes", "sha256")):
        raise ValueError("External-academic receipt raw identity differs")
    evaluator = _load_academic_evaluator()
    receipt = evaluator.validate_external_academic_receipt(
        path_value,
        sha256,
        verify_live=verify_live,
    )
    if not isinstance(receipt, dict):
        raise TypeError("Public external-academic receipt validator returned a non-object")
    if receipt.get("content_sha256") != reference.get("content_sha256"):
        raise ValueError("External-academic receipt semantic identity differs")
    return receipt


def _checkpoint_step(receipt: Mapping[str, Any]) -> int:
    prior = receipt.get("prior_matched_wrong_v2")
    checkpoint = receipt.get("checkpoint")
    if not isinstance(prior, Mapping) or not isinstance(checkpoint, Mapping):
        raise TypeError("External-academic receipt lacks checkpoint provenance")
    step = prior.get("step")
    checkpoint_path = checkpoint.get("checkpoint")
    if type(step) is not int or step not in STEPS or not isinstance(checkpoint_path, str):
        raise ValueError("External-academic receipt checkpoint step is invalid")
    if Path(checkpoint_path).name != f"step{step}":
        raise ValueError("External-academic receipt checkpoint path and step differ")
    return step


def _validate_input_pair(receipts: Mapping[int, Mapping[str, Any]]) -> tuple[str, ...]:
    if set(receipts) != set(STEPS):
        raise ValueError("Comparison requires exactly steps 12000 and 16000")
    for step in STEPS:
        if _checkpoint_step(receipts[step]) != step:
            raise ValueError(f"step{step} input contains a different checkpoint")
    first, second = (receipts[step] for step in STEPS)
    if (
        first.get("schema_version") != 1
        or first.get("format") != EXPECTED_INPUT_FORMAT
        or first.get("protocol_name") != EXPECTED_INPUT_PROTOCOL
    ):
        raise ValueError("External-academic input protocol identity differs")
    shared_fields = ("schema_version", "format", "protocol_name", "git", "implementation")
    for field in shared_fields:
        if first.get(field) != second.get(field):
            raise ValueError(f"External-academic receipt {field} differs across checkpoints")
    for field in ("manifest", "tokenizer", "protocol"):
        if first.get(field) != second.get(field):
            raise ValueError(f"External-academic receipt shared {field} differs")
    expected_policy = {
        "descriptive_only": True,
        "promotion_eligible": False,
        "checkpoint_selection_evidence": True,
    }
    if any(receipt.get("artifact_policy") != expected_policy for receipt in receipts.values()):
        raise ValueError("External-academic inputs must remain descriptive-only evidence")
    protocol = first.get("protocol")
    if not isinstance(protocol, Mapping):
        raise TypeError("External-academic protocol is invalid")
    task_order = protocol.get("tasks")
    if not isinstance(task_order, list) or tuple(task_order) != tuple(_TASK_METRICS):
        raise ValueError("External-academic task order differs from the reviewed panel")
    for step, receipt in receipts.items():
        tasks = receipt.get("tasks")
        if not isinstance(tasks, dict) or set(tasks) != set(task_order):
            raise ValueError(f"step{step} receipt task coverage differs from its protocol")
    return tuple(task_order)


def _number(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    output = float(value)
    if not math.isfinite(output) or not 0.0 <= output <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return output


def _paired_rows(
    receipts: Mapping[int, Mapping[str, Any]], task: str
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    task_payloads = [receipts[step]["tasks"][task] for step in STEPS]
    expected_metric = _TASK_METRICS[task]
    for step, payload in zip(STEPS, task_payloads):
        if not isinstance(payload, Mapping) or payload.get("metric") != expected_metric:
            raise ValueError(f"step{step} task {task!r} metric is not the reviewed accuracy scale")
        if (
            payload.get("source") != task_payloads[0].get("source")
            or payload.get("selection_count") != task_payloads[0].get("selection_count")
            or payload.get("selection_sha256") != task_payloads[0].get("selection_sha256")
        ):
            raise ValueError(f"Task {task!r} selection identity differs across checkpoints")
    row_lists = [payload.get("examples") for payload in task_payloads]
    if any(not isinstance(rows, list) or not rows for rows in row_lists):
        raise ValueError(f"Task {task!r} lacks complete example rows")
    first_rows, second_rows = row_lists
    assert isinstance(first_rows, list) and isinstance(second_rows, list)
    if len(first_rows) != len(second_rows) or len(first_rows) != task_payloads[0].get(
        "selection_count"
    ):
        raise ValueError(f"Task {task!r} example coverage differs")
    pairs = []
    for index, (first, second) in enumerate(zip(first_rows, second_rows)):
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            raise TypeError(f"Task {task!r} row {index} is invalid")
        if any(first.get(field) != second.get(field) for field in _STATIC_ROW_FIELDS):
            raise ValueError(f"Task {task!r} row {index} identity or ordering differs")
        image_sha256 = first.get("image_sha256")
        donor_sha256 = first.get("shuffled_image_sha256")
        if (
            not isinstance(image_sha256, str)
            or _SHA256_RE.fullmatch(image_sha256) is None
            or not isinstance(donor_sha256, str)
            or _SHA256_RE.fullmatch(donor_sha256) is None
        ):
            raise ValueError(f"Task {task!r} row {index} image identity is invalid")
        for step, row in zip(STEPS, (first, second)):
            controls = row.get("controls")
            if not isinstance(controls, dict) or set(controls) != set(CONTROLS):
                raise ValueError(f"step{step} task {task!r} row {index} controls differ")
            for control in CONTROLS:
                output = controls[control]
                if not isinstance(output, Mapping):
                    raise TypeError(f"step{step} task {task!r}/{index}/{control} is invalid")
                _number(output.get("score"), name=f"step{step} {task}/{index}/{control} score")
                shared_tokenization_fields = (
                    "input_tokens",
                    "image_grid_signature",
                    "image_token_count",
                    "image_token_ids_sha256",
                )
                if any(
                    first["controls"][control].get(field) != second["controls"][control].get(field)
                    for field in shared_tokenization_fields
                ):
                    raise ValueError(
                        f"Task {task!r} row {index}/{control} input tokenization differs"
                    )
                tokenization = first["controls"][control]
                if (
                    type(tokenization.get("input_tokens")) is not int
                    or tokenization["input_tokens"] <= 0
                    or not isinstance(tokenization.get("image_grid_signature"), list)
                    or len(tokenization["image_grid_signature"]) != 4
                    or any(
                        type(value) is not int or value <= 0
                        for value in tokenization["image_grid_signature"]
                    )
                    or type(tokenization.get("image_token_count")) is not int
                    or tokenization["image_token_count"] <= 0
                    or not isinstance(tokenization.get("image_token_ids_sha256"), str)
                    or _SHA256_RE.fullmatch(tokenization["image_token_ids_sha256"]) is None
                ):
                    raise ValueError(
                        f"Task {task!r} row {index}/{control} tokenization identity is invalid"
                    )
        pairs.append((first, second))
    return pairs


def _derived_seed(seed: int, label: str) -> int:
    raw = hashlib.sha256(f"{seed}\0{label}".encode()).digest()
    return int.from_bytes(raw[:8], "big")


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 0:
        raise ValueError("Cannot take a percentile of no values")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _cluster_bootstrap_components(
    vectors: Mapping[str, Sequence[float]],
    clusters: Sequence[str],
    *,
    seed: int,
    samples: int,
) -> tuple[dict[str, float], dict[str, list[float]], int]:
    if samples <= 0:
        raise ValueError("Bootstrap samples must be positive")
    if not vectors or any(len(values) != len(clusters) for values in vectors.values()):
        raise ValueError("Bootstrap vectors and cluster labels differ")
    if not clusters:
        raise ValueError("Bootstrap requires at least one paired example")
    ordered_clusters = list(dict.fromkeys(clusters))
    cluster_index = {cluster: index for index, cluster in enumerate(ordered_clusters)}
    counts = np.zeros(len(ordered_clusters), dtype=np.int64)
    sums = {name: np.zeros(len(ordered_clusters), dtype=np.float64) for name in vectors}
    for position, cluster in enumerate(clusters):
        index = cluster_index[cluster]
        counts[index] += 1
        for name, values in vectors.items():
            sums[name][index] += float(values[position])
    generator = np.random.default_rng(seed)
    replicate_arrays = {name: np.empty(samples, dtype=np.float64) for name in vectors}
    chunk_size = 512
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        sampled = generator.integers(
            0,
            len(ordered_clusters),
            size=(stop - start, len(ordered_clusters)),
        )
        denominators = counts[sampled].sum(axis=1)
        for name in vectors:
            replicate_arrays[name][start:stop] = sums[name][sampled].sum(axis=1) / denominators
    means = {name: float(sum(values) / len(values)) for name, values in vectors.items()}
    replicates = {name: values.tolist() for name, values in replicate_arrays.items()}
    return means, replicates, len(ordered_clusters)


def _cluster_bootstrap(
    vectors: Mapping[str, Sequence[float]],
    clusters: Sequence[str],
    *,
    seed: int,
    samples: int,
) -> dict[str, dict[str, Any]]:
    means, replicates, cluster_count = _cluster_bootstrap_components(
        vectors,
        clusters,
        seed=seed,
        samples=samples,
    )
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    output: dict[str, dict[str, Any]] = {}
    for name, values in vectors.items():
        sorted_replicates = sorted(replicates[name])
        output[name] = {
            "examples": len(values),
            "clusters": cluster_count,
            "cluster_unit": "recipient_image_sha256",
            "mean": means[name],
            "ci95": {
                "low": _percentile(sorted_replicates, alpha),
                "high": _percentile(sorted_replicates, 1.0 - alpha),
            },
        }
    return output


def _mcnemar(
    first_scores: Sequence[float], second_scores: Sequence[float], *, binary_metric: bool
) -> dict[str, Any]:
    if len(first_scores) != len(second_scores) or not first_scores:
        raise ValueError("McNemar inputs must be non-empty and paired")
    first_correct = [score == 1.0 for score in first_scores]
    second_correct = [score == 1.0 for score in second_scores]
    both = sum(left and right for left, right in zip(first_correct, second_correct))
    first_only = sum(left and not right for left, right in zip(first_correct, second_correct))
    second_only = sum(not left and right for left, right in zip(first_correct, second_correct))
    neither = len(first_correct) - both - first_only - second_only
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
        "examples": len(first_scores),
        "both_correct": both,
        "step12000_only": first_only,
        "step16000_only": second_only,
        "neither_correct": neither,
        "discordant": discordant,
        "exact_two_sided_p_value": float(p_value),
    }


def _score(row: Mapping[str, Any], control: str) -> float:
    return float(row["controls"][control]["score"])


def _panel(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    task: str,
    label: str,
    seed: int,
    samples: int,
    exact_byte_nonoverlap: bool,
) -> dict[str, Any]:
    recipient_pairs = [
        pair
        for pair in pairs
        if not exact_byte_nonoverlap or not pair[0]["alignment_train_image_overlap"]
    ]
    shuffled_pairs = [
        pair
        for pair in recipient_pairs
        if not exact_byte_nonoverlap or not pair[0]["shuffled_alignment_train_image_overlap"]
    ]
    if not recipient_pairs or not shuffled_pairs:
        raise ValueError(f"Task {task!r}/{label} has no exact-byte-nonoverlap paired examples")

    def summarize(
        selected: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
        vectors: Mapping[str, Sequence[float]],
        suffix: str,
    ) -> dict[str, dict[str, Any]]:
        return _cluster_bootstrap(
            vectors,
            [str(first["image_sha256"]) for first, _ in selected],
            seed=_derived_seed(seed, f"{task}:{label}:{suffix}"),
            samples=samples,
        )

    first_correct = [_score(first, "correct") for first, _ in recipient_pairs]
    second_correct = [_score(second, "correct") for _, second in recipient_pairs]
    correct = summarize(
        recipient_pairs,
        {
            "step12000": first_correct,
            "step16000": second_correct,
            "paired_delta_16000_minus_12000": [
                second - first for first, second in zip(first_correct, second_correct)
            ],
        },
        "correct",
    )
    image_use: dict[str, Any] = {}
    for control, selected in (("shuffled", shuffled_pairs), ("blank", recipient_pairs)):
        first_gaps = [_score(first, "correct") - _score(first, control) for first, _ in selected]
        second_gaps = [
            _score(second, "correct") - _score(second, control) for _, second in selected
        ]
        image_use[control] = summarize(
            selected,
            {
                "step12000_correct_minus_control": first_gaps,
                "step16000_correct_minus_control": second_gaps,
                "paired_delta_16000_minus_12000": [
                    second - first for first, second in zip(first_gaps, second_gaps)
                ],
            },
            control,
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
            ("correct", recipient_pairs),
            ("shuffled", shuffled_pairs),
            ("blank", recipient_pairs),
        ):
            control_payload = {}
            for step_index, step in enumerate(STEPS):
                capped = [
                    pair[step_index]["controls"][control].get("stop_reason") == "max_tokens"
                    for pair in selected
                ]
                control_payload[f"step{step}"] = {
                    "examples": len(capped),
                    "max_tokens": sum(capped),
                    "rate": float(sum(capped) / len(capped)),
                }
            control_payload["rate_delta_16000_minus_12000"] = (
                control_payload["step16000"]["rate"] - control_payload["step12000"]["rate"]
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
            "mcnemar_exact_score_one": _mcnemar(
                first_correct,
                second_correct,
                binary_metric=task in _BINARY_METRIC_TASKS,
            ),
        },
        "image_use": image_use,
        "generation_cap": generation_cap,
    }


def _task_comparison(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    task: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    strata = sorted(
        {str(first["stratum"]) for first, _ in pairs if first.get("stratum") is not None}
    )
    return {
        "metric": _TASK_METRICS[task],
        "metric_scale": "higher_is_better_accuracy_in_[0,1]",
        "example_order_sha256": _canonical_sha256([first["example_id"] for first, _ in pairs]),
        "all_examples": _panel(
            pairs,
            task=task,
            label="all",
            seed=seed,
            samples=samples,
            exact_byte_nonoverlap=False,
        ),
        "exact_byte_nonoverlap": _panel(
            pairs,
            task=task,
            label="exact_byte_nonoverlap",
            seed=seed,
            samples=samples,
            exact_byte_nonoverlap=True,
        ),
        "strata": {
            stratum: {
                "all_examples": _panel(
                    [pair for pair in pairs if pair[0]["stratum"] == stratum],
                    task=task,
                    label=f"stratum:{stratum}:all",
                    seed=seed,
                    samples=samples,
                    exact_byte_nonoverlap=False,
                ),
                "exact_byte_nonoverlap": _panel(
                    [pair for pair in pairs if pair[0]["stratum"] == stratum],
                    task=task,
                    label=f"stratum:{stratum}:exact_byte_nonoverlap",
                    seed=seed,
                    samples=samples,
                    exact_byte_nonoverlap=True,
                ),
            }
            for stratum in strata
        },
    }


def _macro_stat(
    task_vectors: Mapping[str, tuple[Sequence[float], Sequence[str]]],
    *,
    seed: int,
    samples: int,
    label: str,
) -> dict[str, Any]:
    included = tuple(task for task in _TASK_METRICS if task in task_vectors)
    if not included:
        raise ValueError(f"Equal-task macro {label!r} has no compatible tasks")
    task_replicates: dict[str, list[float]] = {}
    task_means: dict[str, float] = {}
    for task in included:
        values, clusters = task_vectors[task]
        means, replicates, _ = _cluster_bootstrap_components(
            {"value": values},
            clusters,
            seed=_derived_seed(seed, f"macro:{label}:{task}"),
            samples=samples,
        )
        task_means[task] = means["value"]
        task_replicates[task] = replicates["value"]
    macro_replicates = sorted(
        sum(task_replicates[task][index] for task in included) / len(included)
        for index in range(samples)
    )
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return {
        "included_tasks": list(included),
        "excluded_tasks": [task for task in _TASK_METRICS if task not in included],
        "task_weighting": "equal_task",
        "mean": float(sum(task_means.values()) / len(task_means)),
        "ci95": {
            "low": _percentile(macro_replicates, alpha),
            "high": _percentile(macro_replicates, 1.0 - alpha),
        },
    }


def _macro_comparison(
    paired_by_task: Mapping[str, Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]],
    *,
    seed: int,
    samples: int,
    exact_byte_nonoverlap: bool,
) -> dict[str, Any]:
    metric_vectors: dict[str, dict[str, tuple[list[float], list[str]]]] = {
        name: {}
        for name in (
            "correct_step12000",
            "correct_step16000",
            "correct_delta_16000_minus_12000",
            "correct_minus_shuffled_step12000",
            "correct_minus_shuffled_step16000",
            "correct_minus_shuffled_delta_16000_minus_12000",
            "correct_minus_blank_step12000",
            "correct_minus_blank_step16000",
            "correct_minus_blank_delta_16000_minus_12000",
        )
    }
    for task, pairs in paired_by_task.items():
        recipient_pairs = [
            pair
            for pair in pairs
            if not exact_byte_nonoverlap or not pair[0]["alignment_train_image_overlap"]
        ]
        shuffled_pairs = [
            pair
            for pair in recipient_pairs
            if not exact_byte_nonoverlap or not pair[0]["shuffled_alignment_train_image_overlap"]
        ]
        if recipient_pairs:
            clusters = [str(first["image_sha256"]) for first, _ in recipient_pairs]
            first = [_score(row, "correct") for row, _ in recipient_pairs]
            second = [_score(row, "correct") for _, row in recipient_pairs]
            metric_vectors["correct_step12000"][task] = (first, clusters)
            metric_vectors["correct_step16000"][task] = (second, clusters)
            metric_vectors["correct_delta_16000_minus_12000"][task] = (
                [right - left for left, right in zip(first, second)],
                clusters,
            )
            for control in ("blank",):
                first_gap = [
                    _score(row, "correct") - _score(row, control) for row, _ in recipient_pairs
                ]
                second_gap = [
                    _score(row, "correct") - _score(row, control) for _, row in recipient_pairs
                ]
                metric_vectors[f"correct_minus_{control}_step12000"][task] = (
                    first_gap,
                    clusters,
                )
                metric_vectors[f"correct_minus_{control}_step16000"][task] = (
                    second_gap,
                    clusters,
                )
                metric_vectors[f"correct_minus_{control}_delta_16000_minus_12000"][task] = (
                    [right - left for left, right in zip(first_gap, second_gap)],
                    clusters,
                )
        if shuffled_pairs:
            clusters = [str(first["image_sha256"]) for first, _ in shuffled_pairs]
            first_gap = [
                _score(row, "correct") - _score(row, "shuffled") for row, _ in shuffled_pairs
            ]
            second_gap = [
                _score(row, "correct") - _score(row, "shuffled") for _, row in shuffled_pairs
            ]
            metric_vectors["correct_minus_shuffled_step12000"][task] = (first_gap, clusters)
            metric_vectors["correct_minus_shuffled_step16000"][task] = (second_gap, clusters)
            metric_vectors["correct_minus_shuffled_delta_16000_minus_12000"][task] = (
                [right - left for left, right in zip(first_gap, second_gap)],
                clusters,
            )
    label_prefix = "exact_byte_nonoverlap" if exact_byte_nonoverlap else "all_examples"
    return {
        "metric_compatibility": (
            "Only per-example higher-is-better scores bounded to [0,1] are included; task "
            "means receive equal weight despite different benchmark metrics."
        ),
        "statistics": {
            name: _macro_stat(
                vectors,
                seed=seed,
                samples=samples,
                label=f"{label_prefix}:{name}",
            )
            for name, vectors in metric_vectors.items()
        },
    }


def _build_comparison_receipt(
    receipts: Mapping[int, Mapping[str, Any]],
    references: Mapping[int, Mapping[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
    created_at: str,
) -> dict[str, Any]:
    if type(bootstrap_seed) is not int:
        raise TypeError("Bootstrap seed must be an integer")
    if type(bootstrap_samples) is not int or bootstrap_samples <= 0:
        raise ValueError("Bootstrap samples must be a positive integer")
    _validate_timestamp(created_at, name="comparison created_at")
    if set(references) != set(STEPS):
        raise ValueError("Comparison input references differ")
    for step in STEPS:
        if set(references[step]) != {"path", "bytes", "sha256", "content_sha256"}:
            raise ValueError(f"step{step} comparison input reference fields differ")
    task_order = _validate_input_pair(receipts)
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
    first = receipts[12_000]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "format": FORMAT,
        "protocol_name": PROTOCOL_NAME,
        "created_at": created_at,
        "implementation": _implementation_identity(),
        "inputs": {f"step{step}": dict(references[step]) for step in STEPS},
        "shared_identity": {
            "git": first["git"],
            "evaluator_implementation": first["implementation"],
            "manifest": first["manifest"],
            "tokenizer": first["tokenizer"],
            "protocol": first["protocol"],
            "task_order": list(task_order),
            "example_order_sha256": {
                task: tasks[task]["example_order_sha256"] for task in task_order
            },
        },
        "bootstrap": {
            "method": "paired percentile cluster bootstrap",
            "confidence_level": CONFIDENCE_LEVEL,
            "cluster_unit": "recipient encoded-image-byte SHA-256 within task",
            "prng": "numpy.default_rng(PCG64)",
            "percentile_interpolation": "linear",
            "seed": bootstrap_seed,
            "samples": bootstrap_samples,
        },
        "direction": "step16000_minus_step12000",
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
        "policy": {
            "conclusion": "descriptive_only",
            "promotion_eligible": False,
            "automatic_checkpoint_selection": False,
            "promotion_decision": None,
            "intended_use": "human-reviewed evidence for choosing the Stage-1 mid-training seed",
        },
    }
    return _attach_content_sha256(payload)


def _validated_inputs_from_references(
    references: Mapping[str, Any], *, verify_live: bool
) -> tuple[dict[int, dict[str, Any]], dict[int, Mapping[str, Any]]]:
    if not isinstance(references, dict) or tuple(references) != STEP_KEYS:
        raise ValueError("Comparison input ordering differs")
    receipts: dict[int, dict[str, Any]] = {}
    normalized: dict[int, Mapping[str, Any]] = {}
    for step, key in zip(STEPS, STEP_KEYS):
        reference = references[key]
        if not isinstance(reference, Mapping):
            raise TypeError(f"Comparison {key} input reference is invalid")
        receipts[step] = _load_validated_input(reference, verify_live=verify_live)
        normalized[step] = reference
    return receipts, normalized


def validate_academic_comparison_receipt(
    path: str | Path,
    expected_sha256: str,
    *,
    verify_live_inputs: bool = True,
) -> dict[str, Any]:
    """Strictly reload and fully rederive an academic checkpoint-comparison receipt.

    :param path: Canonical comparison receipt path.
    :param expected_sha256: Independently supplied raw comparison SHA-256.
    :param verify_live_inputs: Ask each input's public validator to rehash live provenance.
    :returns: The validated comparison receipt.
    """
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError("Expected comparison SHA-256 must be lowercase hex")
    receipt, identity = _load_json(Path(path), name="academic comparison receipt")
    if identity["sha256"] != expected_sha256:
        raise ValueError("Academic comparison receipt raw SHA-256 differs")
    _verify_content_sha256(receipt, name="academic comparison receipt")
    expected_fields = {
        "schema_version",
        "format",
        "protocol_name",
        "created_at",
        "implementation",
        "inputs",
        "shared_identity",
        "bootstrap",
        "direction",
        "tasks",
        "equal_task_macro",
        "policy",
        "content_sha256",
    }
    if set(receipt) != expected_fields:
        raise ValueError("Academic comparison receipt fields differ")
    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or receipt["format"] != FORMAT
        or receipt["protocol_name"] != PROTOCOL_NAME
        or receipt["implementation"] != _implementation_identity()
    ):
        raise ValueError("Academic comparison receipt implementation or protocol differs")
    bootstrap = receipt.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise TypeError("Academic comparison bootstrap configuration is invalid")
    seed = bootstrap.get("seed")
    samples = bootstrap.get("samples")
    if type(seed) is not int or type(samples) is not int or samples <= 0:
        raise ValueError("Academic comparison bootstrap configuration differs")
    receipts, references = _validated_inputs_from_references(
        receipt["inputs"], verify_live=verify_live_inputs
    )
    expected = _build_comparison_receipt(
        receipts,
        references,
        bootstrap_seed=seed,
        bootstrap_samples=samples,
        created_at=receipt["created_at"],
    )
    if receipt != expected:
        raise ValueError("Academic comparison receipt does not equal its full rederivation")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step12000-receipt", required=True)
    parser.add_argument("--step12000-sha256", required=True)
    parser.add_argument("--step16000-receipt", required=True)
    parser.add_argument("--step16000-sha256", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    """Validate both inputs, construct the comparison, and publish it exactly once."""
    args = _parser().parse_args()
    output = _direct_path(Path(args.output), name="comparison output", allow_missing_leaf=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite canonical artifact {output}")
    paths = {
        12_000: Path(args.step12000_receipt),
        16_000: Path(args.step16000_receipt),
    }
    expected_hashes = {
        12_000: args.step12000_sha256,
        16_000: args.step16000_sha256,
    }
    evaluator = _load_academic_evaluator()
    receipts = {
        step: evaluator.validate_external_academic_receipt(
            path,
            expected_hashes[step],
            verify_live=True,
        )
        for step, path in paths.items()
    }
    references = {
        step: _input_reference(paths[step], expected_hashes[step], receipts[step]) for step in STEPS
    }
    comparison = _build_comparison_receipt(
        receipts,
        references,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    # Full prepublication rederivation: invoke both public input validators again and rebuild
    # every comparison field from the immutable raw references.
    validated_receipts = {
        step: _load_validated_input(references[step], verify_live=True) for step in STEPS
    }
    rederived = _build_comparison_receipt(
        validated_receipts,
        references,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        created_at=comparison["created_at"],
    )
    if rederived != comparison:
        raise ValueError("Academic comparison changed during prepublication rederivation")
    _write_json_no_overwrite(output, comparison)
    raw_sha256 = _file_identity(output, name="academic comparison receipt")["sha256"]
    validate_academic_comparison_receipt(output, raw_sha256, verify_live_inputs=False)
    print(f"wrote {output} sha256={raw_sha256}")


if __name__ == "__main__":
    main()
