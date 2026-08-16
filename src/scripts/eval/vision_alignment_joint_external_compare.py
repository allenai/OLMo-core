"""Strictly compare the joint-v1 step-12000/16000 external diagnostics on CPU.

This script never imports or loads a model. It authenticates four pinned JSON result files,
validates the complete MMMU-Pro and OLMES-fast protocols and coverage, and emits descriptive
step-16000-minus-step-12000 deltas. It deliberately does not select or promote a checkpoint.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import stat
import string
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORMAT = "vision_alignment_joint_external_comparison"
VERSION = 1
PROTOCOL_NAME = "vision-alignment-joint-v1-external-step12000-step16000-v1"
EXPECTED_GIT_REF = "41860467465842c0ebdc3b8ec6438bb05e476b5f"
EXPECTED_CONFIG_SHA256 = "64b302865831b5aaf11e86e142a85b3467a06b93d6c214fb67f7f94a45c4ddc8"
EXPECTED_CHECKPOINT_PROVENANCE = {
    12_000: {
        "receipt_path": (
            "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/"
            "evals/joint-v1-matched-wrong-v1/step12000-41860467.json"
        ),
        "receipt_sha256": "c7f960975ade934ecf8c9c0c7f39f417d1e1d983d9cbbe1d7f56afef3f00ce64",
        "receipt_content_sha256": (
            "89b76b6b246530267229a5934d6d37c137381b9e1fb12e724f4406aded9d006f"
        ),
        "checkpoint_identity_sha256": (
            "92997888dc8e0f6c57bb6502a2bde580a4afdfe1aba48e481fc82e124d5ff883"
        ),
        "model_and_optim_identity_sha256": (
            "d39222c547cc98596b66637b483c4d15ee62b495fc47bc2b9e9c24edfff00244"
        ),
        "state_file_inventory_sha256": (
            "c6da4882ecc18c73f187af75aeb16c83ec4a61890918dccc5cf3d5fbc184ca19"
        ),
        "trainer_state_file_inventory_sha256": (
            "0889424d5d288526ad1a46d048ca3f524d451589de81afff3282e05956a6f1bc"
        ),
        "checkpoint_marker_sha256": (
            "77dfdeec42fe7990f4b3b9c4eeecd480edcf5066c110603b115920af38423d03"
        ),
        "dcp_metadata_sha256": ("44cc94aa5b69bb774e45561062476d4e97a3d6ef3ff6e5ab40f53591a42a651f"),
    },
    16_000: {
        "receipt_path": (
            "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/"
            "evals/joint-v1-matched-wrong-v1/step16000-41860467.json"
        ),
        "receipt_sha256": "c9307835a3597331add4a800a8a4baa7f2dd4df89f6a44b348766899782b5ccc",
        "receipt_content_sha256": (
            "87bf5af3ca45ad8a2cdcef18df558cbf584ffabe9995b0a92ebb36c0937f3c6d"
        ),
        "checkpoint_identity_sha256": (
            "735bcd883b82fe91e49a7d99190e2a6a4c6160adbebcd796ec9eb2cb9fc27ac1"
        ),
        "model_and_optim_identity_sha256": (
            "7a6f67866128489750d1f64ae7f3e22c630106386030113a6d22925efa6cbd3b"
        ),
        "state_file_inventory_sha256": (
            "536a476ee1084a4e9d04904663141453375cc818e0cffbc53175bf02eb09d55c"
        ),
        "trainer_state_file_inventory_sha256": (
            "d3ce3649af4656b6beaf00cb2ef4f62e121f71d0a04d1edfd1288547070ce371"
        ),
        "checkpoint_marker_sha256": (
            "77dfdeec42fe7990f4b3b9c4eeecd480edcf5066c110603b115920af38423d03"
        ),
        "dcp_metadata_sha256": ("a377447e5cea89c8d204df5a3d95810bd860bd6111d55dbb52bbe951aa6f4ff2"),
    },
}
EXPECTED_V2_LOAD_COVERAGE = {
    "checkpoint_key_count": 3_271,
    "complete": True,
    "eval_state_key_count": 817,
    "frozen_state_key_count": 1,
    "load_completed": True,
    "model_parameter_assignments_sha256": (
        "c7e20cfc124dacc8e2e232526d8e132c460a7fc4b0893615fbd82559f7915b3b"
    ),
    "model_parameter_checkpoint_key_count": 818,
    "model_parameter_checkpoint_keys_sha256": (
        "6c2f287f3afd0234aa506d01c49ebe97eaee6fb6ca16b415ceb7e35affffb5e1"
    ),
    "model_parameter_count": 818,
    "persistent_buffer_count": 0,
    "persistent_buffer_keys_sha256": (
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    ),
    "prepared_load_key_count": 818,
    "sha256": "20b60642a931549f95deadf9c9ab8ac7a679d71ddbeac8af24bd50ef07746132",
    "shadowed_frozen_key_count": 0,
    "shadowed_frozen_keys_sha256": (
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    ),
    "unused_model_bearing_key_count": 0,
}
EXPECTED_EVALUATOR_SOURCES = {
    "mmmu_pro": {
        "filename": "s002_mmmu_pro.py",
        "sha256": "6e633242958424a4e904df77478e2b173d3b4243fb77ca41d07b9cbcb056a807",
    },
    "olmes_fast": {
        "filename": "s002_downstream.py",
        "sha256": "d7e9838b4905831f085a844333e427653a16634f8a7124bcd7fa8ea15b07dee7",
    },
}
STEPS = (12_000, 16_000)
STEP_KEYS = ("step12000", "step16000")
SUITES = ("mmmu_pro", "olmes_fast")
EXPECTED_CHECKPOINTS = {
    step: (
        "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/"
        f"checkpoints/vision-alignment-joint-v1/step{step}"
    )
    for step in STEPS
}
MMMU_TASKS = ("mmmu_pro_vision", "mmmu_pro_standard")
MMMU_SAMPLES_PER_TASK = 1_730
EXPECTED_MMMU_DOC_HASH = "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b"
_MMMU_TASK_CONFIG_FIELDS = frozenset(
    {
        "dataset_name",
        "dataset_path",
        "description",
        "doc_to_target",
        "doc_to_text",
        "doc_to_visual",
        "fewshot_delimiter",
        "full_docs",
        "generation_kwargs",
        "lmms_eval_specific_kwargs",
        "metadata",
        "metric_list",
        "num_fewshot",
        "output_type",
        "process_results",
        "process_results_use_image",
        "repeats",
        "score_key",
        "should_decontaminate",
        "tag",
        "target_delimiter",
        "task",
        "test_split",
    }
)
OLMES_COVERAGE = {
    "arc_challenge_test_bpb_5shot": (147, 7),
    "arc_challenge_test_mc_5shot_fast": (147, 10),
    "arc_easy_test_bpb_5shot": (297, 15),
    "arc_easy_test_mc_5shot_fast": (297, 19),
    "hellaswag_bpb_5shot": (125, 8),
    "mmlu_humanities_test_bpb_5shot": (589, 148),
    "mmlu_humanities_test_mc_5shot_fast": (589, 148),
    "mmlu_other_test_bpb_5shot": (406, 58),
    "mmlu_other_test_mc_5shot_fast": (406, 82),
    "mmlu_social_sciences_test_bpb_5shot": (385, 33),
    "mmlu_social_sciences_test_mc_5shot_fast": (385, 77),
    "mmlu_stem_test_bpb_5shot": (378, 42),
    "mmlu_stem_test_mc_5shot_fast": (378, 54),
    "basic_skills_arithmetic_rc_5shot": (1_309, 21),
    "basic_skills_coding_rc_5shot": (633, 20),
    "basic_skills_common_knowledge_rc_5shot": (621, 10),
    "basic_skills_logical_reasoning_rc_5shot": (559, 18),
    "basic_skills_pattern_rc_5shot": (268, 5),
    "basic_skills_string_operations_rc_5shot": (748, 12),
    "codex_humaneval_gold_bpb_3shot": (21, 3),
    "codex_mbpp_gold_bpb_3shot": (63, 7),
    "minerva_math_500_gold_bpb_0shot": (63, 16),
    "mt_mbpp_cpp_gold_bpb_3shot": (63, 8),
    "mt_mbpp_java_gold_bpb_3shot": (63, 8),
    "mt_mbpp_rust_gold_bpb_3shot": (63, 7),
    "copycolors_10way_fast": (13, 1),
}
OLMES_TASKS = tuple(OLMES_COVERAGE)
OLMES_CATEGORY_TASKS = {
    "language_reasoning": (
        "arc_challenge_test_bpb_5shot",
        "arc_challenge_test_mc_5shot_fast",
        "arc_easy_test_bpb_5shot",
        "arc_easy_test_mc_5shot_fast",
        "hellaswag_bpb_5shot",
    ),
    "language_mmlu": (
        "mmlu_humanities_test_bpb_5shot",
        "mmlu_humanities_test_mc_5shot_fast",
        "mmlu_other_test_bpb_5shot",
        "mmlu_other_test_mc_5shot_fast",
        "mmlu_social_sciences_test_bpb_5shot",
        "mmlu_social_sciences_test_mc_5shot_fast",
        "mmlu_stem_test_bpb_5shot",
        "mmlu_stem_test_mc_5shot_fast",
    ),
    "language_basic_skills": (
        "basic_skills_arithmetic_rc_5shot",
        "basic_skills_coding_rc_5shot",
        "basic_skills_common_knowledge_rc_5shot",
        "basic_skills_logical_reasoning_rc_5shot",
        "basic_skills_pattern_rc_5shot",
        "basic_skills_string_operations_rc_5shot",
    ),
    "language_code_math": (
        "codex_humaneval_gold_bpb_3shot",
        "codex_mbpp_gold_bpb_3shot",
        "minerva_math_500_gold_bpb_0shot",
        "mt_mbpp_cpp_gold_bpb_3shot",
        "mt_mbpp_java_gold_bpb_3shot",
        "mt_mbpp_rust_gold_bpb_3shot",
    ),
    "language_sanity": ("copycolors_10way_fast",),
}
_OLMES_ACCURACY_TASKS = frozenset(
    {
        "arc_challenge_test_mc_5shot_fast",
        "arc_easy_test_mc_5shot_fast",
        "basic_skills_arithmetic_rc_5shot",
        "basic_skills_coding_rc_5shot",
        "basic_skills_common_knowledge_rc_5shot",
        "basic_skills_logical_reasoning_rc_5shot",
        "basic_skills_pattern_rc_5shot",
        "basic_skills_string_operations_rc_5shot",
        "copycolors_10way_fast",
    }
)
_OLMES_LENGTH_NORMALIZED_TASKS = frozenset(
    task for task in OLMES_TASKS if task.startswith("mmlu_") and "_mc_" in task
)
_OLMES_BPB_SUFFIXES = ("BPB v2", "BPB")
_OLMES_ACCURACY_SUFFIXES = (
    "BPB v2",
    "BPB",
    "CE loss v2",
    "CE loss",
    "accuracy v2",
    "accuracy",
    "log soft loss v2",
    "log soft loss",
    "soft loss v2",
    "soft loss",
)
_OLMES_LENGTH_NORMALIZED_SUFFIXES = (
    "BPB v2",
    "BPB",
    "CE loss v2",
    "CE loss",
    "length-normalized accuracy v2",
    "length-normalized accuracy",
    "log soft loss v2",
    "log soft loss",
    "soft loss v2",
    "soft loss",
)
_MMMU_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "created_at",
        "checkpoint",
        "checkpoint_state_dir",
        "config",
        "git",
        "protocol",
        "lmms_eval",
    }
)
_OLMES_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "created_at",
        "checkpoint",
        "checkpoint_state_dir",
        "checkpoint_kind",
        "config",
        "git",
        "protocol",
        "results",
    }
)
_MMMU_PROTOCOL_FIELDS = frozenset(
    {
        "harness",
        "tasks",
        "partial",
        "limit",
        "world_size",
        "ep_degree",
        "expert_parallel_path",
        "logical_eval_replicas",
        "max_sequence_length",
        "max_crops_total",
        "max_crops_per_image",
        "crop_budget_mode",
        "multi_image_text_prefixes",
        "max_new_tokens_override",
        "sequence_bucket_size",
        "attention_backend",
        "prompt_layout",
        "response_separator",
        "response_mode",
        "text_vocab_size",
        "generation",
    }
)
_OLMES_PROTOCOL_FIELDS = frozenset(
    {
        "harness",
        "task_group",
        "tasks",
        "partial",
        "max_batches_per_task",
        "max_sequence_length",
        "rank_batch_size_tokens",
        "world_size",
        "ep_degree",
        "ep_dp_degree",
        "attention_backend",
    }
)
_MMMU_EVAL_FIELDS = frozenset(
    {
        "config",
        "configs",
        "date",
        "efficiency",
        "git_branch",
        "git_hash",
        "group_subtasks",
        "higher_is_better",
        "lmms_eval_version",
        "n-samples",
        "n-shot",
        "results",
        "samples",
        "usage",
        "versions",
    }
)
_MMMU_SAMPLE_FIELDS = frozenset(
    {
        "arguments",
        "doc",
        "doc_hash",
        "doc_id",
        "filtered_resps",
        "mmmu_acc",
        "resps",
        "target",
        "token_counts",
    }
)
_V2_RECEIPT_FIELDS = frozenset(
    {
        "artifact_policy",
        "blank_results",
        "checkpoint",
        "checkpoint_config",
        "content_sha256",
        "created_at",
        "endpoint",
        "format",
        "git",
        "load_coverage",
        "native_result",
        "pairing_manifest",
        "producer",
        "projection",
        "protocol",
        "source_audit",
        "status",
        "tokenizer",
        "version",
        "visual_results",
    }
)
_OUTPUT_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "created_at",
        "producer",
        "inputs",
        "checkpoint_provenance",
        "policy",
        "protocol",
        "coverage",
        "task_comparisons",
        "selection_summary",
        "content_sha256",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for step in STEPS:
        for suite in SUITES:
            flag = suite.replace("_", "-")
            parser.add_argument(f"--step{step}-{flag}", required=True)
            parser.add_argument(f"--expected-step{step}-{flag}-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _serialized_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON repeats key {key!r}")
        result[key] = value
    return result


def _read_regular_file(path_value: str | Path, *, name: str) -> tuple[Path, bytes, str]:
    path = Path(path_value).expanduser().absolute()
    try:
        path_info = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{name} does not exist: {path}") from error
    if stat.S_ISLNK(path_info.st_mode) or not stat.S_ISREG(path_info.st_mode):
        raise ValueError(f"{name} must be a direct regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as file_handle:
            raw = file_handle.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(raw) != after.st_size:
        raise RuntimeError(f"{name} changed while it was read: {path}")
    return path, raw, hashlib.sha256(raw).hexdigest()


def _load_json_source(
    path_value: str | Path, *, expected_sha256: str, name: str
) -> tuple[Path, Mapping[str, Any], str]:
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{name} expected SHA-256 must be 64 lowercase hex characters")
    path, raw, actual_sha256 = _read_regular_file(path_value, name=name)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{name} raw SHA-256 differs: expected {expected_sha256}, got {actual_sha256}"
        )

    def reject_constant(value: str) -> Any:
        raise ValueError(f"{name} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_json_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not decode {name} as strict JSON: {error}") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return path, value, actual_sha256


def _exact(value: Any, fields: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{name} fields differ: missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an array")
    return value


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        suffix = f" and >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be finite{suffix}")
    return result


def _timestamp(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} is not a valid ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return value


def _validate_task_keys(value: Any, tasks: Sequence[str], *, name: str) -> Mapping[str, Any]:
    result = _mapping(value, name=name)
    if set(result) != set(tasks):
        raise ValueError(f"{name} task coverage differs from the exact protocol")
    return result


def _function_repr(value: Any, function_name: str, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or re.fullmatch(rf"<function {re.escape(function_name)} at 0x[0-9a-f]+>", value) is None
    ):
        raise ValueError(f"{name} does not name {function_name}")


def _validate_mmmu_task_config(value: Any, *, task: str, step: int) -> None:
    config = _exact(value, _MMMU_TASK_CONFIG_FIELDS, name=f"step{step} {task} task metadata")
    dataset_name = "vision" if task == "mmmu_pro_vision" else "standard (10 options)"
    fixed = {
        "dataset_name": dataset_name,
        "dataset_path": "MMMU/MMMU_Pro",
        "description": "",
        "doc_to_target": "{{answer}}",
        "fewshot_delimiter": "\n\n",
        "full_docs": False,
        "generation_kwargs": {"max_new_tokens": 256, "until": ["\n\n"]},
        "metadata": {"interleaved_format": False, "version": 0},
        "num_fewshot": 0,
        "output_type": "generate_until",
        "process_results_use_image": False,
        "repeats": 1,
        "score_key": "score",
        "should_decontaminate": False,
        "tag": ["public_eval_qwen3_5_family", "public_eval_gemini3_family"],
        "target_delimiter": " ",
        "task": task,
        "test_split": "test",
    }
    if any(config[field] != expected for field, expected in fixed.items()):
        raise ValueError(f"step{step} {task} dataset/task metadata differs")
    _function_repr(config["doc_to_text"], "mmmu_pro_doc_to_text", name=f"{task} doc_to_text")
    _function_repr(config["doc_to_visual"], "mmmu_pro_doc_to_visual", name=f"{task} doc_to_visual")
    _function_repr(
        config["process_results"],
        "mmmu_pro_process_results",
        name=f"{task} process_results",
    )
    metrics = _sequence(config["metric_list"], name=f"{task} metric metadata")
    if len(metrics) != 1:
        raise ValueError(f"step{step} {task} metric metadata differs")
    metric = _exact(
        metrics[0],
        frozenset({"aggregation", "higher_is_better", "metric"}),
        name=f"{task} metric metadata",
    )
    if metric["metric"] != "mmmu_acc" or metric["higher_is_better"] is not True:
        raise ValueError(f"step{step} {task} metric metadata differs")
    _function_repr(
        metric["aggregation"],
        "mmmu_pro_aggregate_results",
        name=f"{task} metric aggregation",
    )
    specific = _exact(
        config["lmms_eval_specific_kwargs"],
        frozenset({"default", "penguinvl", "qwen3_vl"}),
        name=f"{task} harness-specific metadata",
    )
    if specific["default"] != {
        "pre_prompt": "",
        "post_prompt": "Answer with the option letter from the given choices directly.",
    }:
        raise ValueError(f"step{step} {task} default prompt metadata differs")
    for harness in ("penguinvl", "qwen3_vl"):
        if not isinstance(specific[harness], Mapping):
            raise ValueError(f"step{step} {task} {harness} metadata differs")


def _validate_mmmu_harness_config(value: Any, *, step: int) -> None:
    expected: dict[str, Any] = {
        "batch_size": None,
        "batch_sizes": [],
        "bootstrap_iters": 0,
        "device": None,
        "fewshot_seed": 1234,
        "gen_kwargs": None,
        "limit": None,
        "model": "_Adapter",
        "model_args": "",
        "numpy_seed": 1234,
        "offset": 0,
        "random_seed": 0,
        "torch_seed": 1234,
        "use_cache": None,
    }
    if value != expected:
        raise ValueError(f"step{step} MMMU harness configuration differs")


def _expected_olmes_metric_keys(task: str) -> tuple[str, ...]:
    suffixes: tuple[str, ...]
    if task in _OLMES_LENGTH_NORMALIZED_TASKS:
        suffixes = _OLMES_LENGTH_NORMALIZED_SUFFIXES
    elif task in _OLMES_ACCURACY_TASKS:
        suffixes = _OLMES_ACCURACY_SUFFIXES
    else:
        suffixes = _OLMES_BPB_SUFFIXES
    return tuple(f"{task} ({suffix})" for suffix in suffixes)


def _load_checkpoint_provenance() -> dict[int, dict[str, Any]]:
    provenance: dict[int, dict[str, Any]] = {}
    for step in STEPS:
        checkpoint = EXPECTED_CHECKPOINTS[step]
        config_path, _, config_sha256 = _read_regular_file(
            Path(checkpoint) / "config.json", name=f"step{step} live checkpoint config"
        )
        if config_sha256 != EXPECTED_CONFIG_SHA256:
            raise ValueError(
                f"step{step} live config SHA-256 differs: expected {EXPECTED_CONFIG_SHA256}, "
                f"got {config_sha256}"
            )
        expected = EXPECTED_CHECKPOINT_PROVENANCE[step]
        receipt_path, receipt, receipt_sha256 = _load_json_source(
            expected["receipt_path"],
            expected_sha256=expected["receipt_sha256"],
            name=f"step{step} strict V2 receipt",
        )
        receipt = _exact(receipt, _V2_RECEIPT_FIELDS, name=f"step{step} strict V2 receipt")
        if (
            receipt["format"] != "vision_alignment_joint_matched_wrong_receipt"
            or receipt["version"] != 2
            or receipt["status"] != "valid"
        ):
            raise ValueError(f"step{step} receipt is not the strict valid V2 receipt")
        unsigned_receipt = dict(receipt)
        content_sha256 = unsigned_receipt.pop("content_sha256")
        if content_sha256 != expected[
            "receipt_content_sha256"
        ] or content_sha256 != _canonical_sha256(unsigned_receipt):
            raise ValueError(f"step{step} strict V2 content SHA-256 differs")
        git = _exact(
            receipt["git"],
            frozenset({"dirty", "revision", "status_sha256", "tracked_diff_sha256"}),
            name=f"step{step} strict V2 git",
        )
        empty_sha256 = hashlib.sha256(b"").hexdigest()
        if git != {
            "dirty": False,
            "revision": EXPECTED_GIT_REF,
            "status_sha256": empty_sha256,
            "tracked_diff_sha256": empty_sha256,
        }:
            raise ValueError(f"step{step} strict V2 receipt git provenance differs")
        endpoint = _mapping(receipt["endpoint"], name=f"step{step} endpoint")
        if (
            endpoint.get("contract") != "vision-alignment-joint-saved-endpoints-v1"
            or endpoint.get("step") != step
            or endpoint.get("storage_class") != "scheduled_permanent"
            or endpoint.get("nearest_step_substitution") is not False
            or endpoint.get("admissible_steps") != [12_000, 14_400, 16_000]
        ):
            raise ValueError(f"step{step} strict V2 endpoint identity differs")
        protocol = _mapping(receipt["protocol"], name=f"step{step} strict V2 protocol")
        if (
            protocol.get("name") != "vision-alignment-joint-native-matched-wrong-saved-endpoints-v2"
            or protocol.get("evaluated_step") != step
            or protocol.get("checkpoint_config_sha256") != EXPECTED_CONFIG_SHA256
            or protocol.get("nearest_step_substitution") is not False
            or protocol.get("descriptive_only") is not True
            or protocol.get("promotion_eligible") is not False
        ):
            raise ValueError(f"step{step} strict V2 protocol identity differs")
        checkpoint_config = _mapping(
            receipt["checkpoint_config"], name=f"step{step} checkpoint config receipt"
        )
        if (
            checkpoint_config.get("path") != str(config_path)
            or checkpoint_config.get("sha256") != config_sha256
            or checkpoint_config.get("step") != step
            or checkpoint_config.get("phase") != "joint"
            or checkpoint_config.get("lineage_id") != "vision-alignment-joint-v1"
            or checkpoint_config.get("run_name") != "vision-alignment-joint-v1"
        ):
            raise ValueError(f"step{step} strict V2 checkpoint-config identity differs")
        checkpoint_identity = _mapping(
            receipt["checkpoint"], name=f"step{step} checkpoint identity"
        )
        identity_unsigned = dict(checkpoint_identity)
        identity_sha256 = identity_unsigned.pop("identity_sha256", None)
        if identity_sha256 != _canonical_sha256(identity_unsigned):
            raise ValueError(f"step{step} checkpoint identity self-hash differs")
        expected_identity_fields = {
            "identity_sha256": "checkpoint_identity_sha256",
            "model_and_optim_identity_sha256": "model_and_optim_identity_sha256",
            "state_file_inventory_sha256": "state_file_inventory_sha256",
            "trainer_state_file_inventory_sha256": "trainer_state_file_inventory_sha256",
            "checkpoint_marker_sha256": "checkpoint_marker_sha256",
            "dcp_metadata_sha256": "dcp_metadata_sha256",
        }
        if any(
            checkpoint_identity.get(field) != expected[expected_field]
            for field, expected_field in expected_identity_fields.items()
        ):
            raise ValueError(f"step{step} checkpoint identity differs from its exact pin")
        if (
            checkpoint_identity.get("root") != checkpoint
            or checkpoint_identity.get("state_dir") != f"{checkpoint}/model_and_optim"
            or checkpoint_identity.get("config_sha256") != config_sha256
            or checkpoint_identity.get("checkpoint_step") != step
            or checkpoint_identity.get("permanent") is not True
            or checkpoint_identity.get("checkpoint_marker")
            != {"ephemeral": False, "version": "2.5.0"}
        ):
            raise ValueError(f"step{step} checkpoint endpoint identity differs")
        if _canonical_sha256(
            checkpoint_identity.get("state_file_inventory")
        ) != checkpoint_identity.get("state_file_inventory_sha256") or _canonical_sha256(
            checkpoint_identity.get("trainer_state_file_inventory")
        ) != checkpoint_identity.get(
            "trainer_state_file_inventory_sha256"
        ):
            raise ValueError(f"step{step} checkpoint inventory identity differs")
        trainer_summary = _mapping(
            checkpoint_identity.get("trainer_state_summary"),
            name=f"step{step} trainer-state summary",
        )
        error_panel = _sequence(
            checkpoint_identity.get("trainer_state_total_data_errors_by_rank"),
            name=f"step{step} trainer error panel",
        )
        if (
            trainer_summary.get("global_step") != step
            or trainer_summary.get("batches_processed") != step
            or trainer_summary.get("wandb_name") != "vision-alignment-joint-v1"
            or checkpoint_identity.get("trainer_state_rank_count") != 16
            or len(error_panel) != 16
            or sum(error_panel) != checkpoint_identity.get("trainer_state_total_data_errors_sum")
        ):
            raise ValueError(f"step{step} trainer-state identity differs")
        load_coverage = _mapping(
            receipt["load_coverage"], name=f"step{step} strict V2 load coverage"
        )
        if dict(load_coverage) != EXPECTED_V2_LOAD_COVERAGE:
            raise ValueError(f"step{step} strict V2 load coverage differs")
        unsigned_load = dict(load_coverage)
        load_sha256 = unsigned_load.pop("sha256")
        if load_sha256 != _canonical_sha256(unsigned_load):
            raise ValueError(f"step{step} strict V2 load-coverage SHA-256 differs")
        artifact_policy = _mapping(
            receipt["artifact_policy"], name=f"step{step} strict V2 artifact policy"
        )
        if (
            artifact_policy.get("checkpoint_post_identity_rehashed") is not True
            or artifact_policy.get("output_overwrite_enabled") is not False
            or artifact_policy.get("descriptive_only") is not True
            or artifact_policy.get("promotion_eligible") is not False
        ):
            raise ValueError(f"step{step} strict V2 artifact policy differs")
        provenance[step] = {
            "live_config": {"path": str(config_path), "sha256": config_sha256},
            "strict_v2_receipt": {
                "path": str(receipt_path),
                "sha256": receipt_sha256,
                "content_sha256": content_sha256,
            },
            "checkpoint_identity_sha256": identity_sha256,
            "model_and_optim_identity_sha256": checkpoint_identity[
                "model_and_optim_identity_sha256"
            ],
            "state_file_inventory_sha256": checkpoint_identity["state_file_inventory_sha256"],
            "trainer_state_file_inventory_sha256": checkpoint_identity[
                "trainer_state_file_inventory_sha256"
            ],
            "load_coverage_sha256": load_sha256,
        }
    return provenance


def _validate_common(payload: Mapping[str, Any], *, step: int, suite: str) -> None:
    fields = _MMMU_TOP_FIELDS if suite == "mmmu_pro" else _OLMES_TOP_FIELDS
    _exact(payload, fields, name=f"step{step} {suite} result")
    if payload["schema_version"] != 1:
        raise ValueError(f"step{step} {suite} schema version differs")
    _timestamp(payload["created_at"], name=f"step{step} {suite} created_at")
    checkpoint = EXPECTED_CHECKPOINTS[step]
    if payload["checkpoint"] != checkpoint:
        raise ValueError(f"step{step} {suite} checkpoint path differs")
    if payload["checkpoint_state_dir"] != f"{checkpoint}/model_and_optim":
        raise ValueError(f"step{step} {suite} checkpoint state path differs")
    if payload["config"] != f"{checkpoint}/config.json":
        raise ValueError(f"step{step} {suite} config path differs")
    git = _exact(payload["git"], frozenset({"revision", "dirty"}), name="git provenance")
    if git["revision"] != EXPECTED_GIT_REF or git["dirty"] is not False:
        raise ValueError(f"step{step} {suite} was not produced from the exact clean git ref")
    if suite == "olmes_fast" and payload["checkpoint_kind"] != "multimodal_stage1":
        raise ValueError(f"step{step} OLMES checkpoint kind differs")


def _validate_mmmu_protocol(value: Any, *, step: int) -> None:
    protocol = _exact(value, _MMMU_PROTOCOL_FIELDS, name=f"step{step} MMMU protocol")
    expected = {
        "harness": "lmms-eval",
        "tasks": list(MMMU_TASKS),
        "partial": False,
        "limit": None,
        "world_size": 8,
        "ep_degree": 8,
        "expert_parallel_path": "sync_1d",
        "logical_eval_replicas": 1,
        "max_sequence_length": 8_192,
        "max_crops_total": 8,
        "max_crops_per_image": None,
        "crop_budget_mode": "shared_total",
        "multi_image_text_prefixes": None,
        "max_new_tokens_override": None,
        "sequence_bucket_size": 128,
        "attention_backend": "flex",
        "prompt_layout": "document",
        "response_separator": "single_leading_space",
        "response_mode": "letter_logits",
        "text_vocab_size": 100_278,
        "generation": "single_forward_option_letter_logits",
    }
    if dict(protocol) != expected:
        raise ValueError(f"step{step} MMMU protocol differs from the exact full document gate")


def _validate_mmmu_samples(value: Any, *, task: str, step: int) -> tuple[float, int, str]:
    samples = _sequence(value, name=f"step{step} {task} samples")
    if len(samples) != MMMU_SAMPLES_PER_TASK:
        raise ValueError(f"step{step} {task} does not contain exactly 1730 samples")
    source_projection: list[dict[str, Any]] = []
    document_ids: set[str] = set()
    correct = 0
    for position, raw_row in enumerate(samples):
        row = _exact(
            raw_row,
            _MMMU_SAMPLE_FIELDS,
            name=f"step{step} {task} sample {position}",
        )
        if row["doc_id"] != position:
            raise ValueError(f"step{step} {task} samples are not in exact document order")
        if row["doc_hash"] != EXPECTED_MMMU_DOC_HASH:
            raise ValueError(f"step{step} {task} sample {position} doc hash differs")
        doc = _mapping(row["doc"], name=f"step{step} {task} sample {position} doc")
        document_id = doc.get("id")
        target = row["target"]
        subject = doc.get("subject")
        if (
            not isinstance(document_id, str)
            or not document_id
            or not isinstance(target, str)
            or target not in string.ascii_uppercase[:10]
            or not isinstance(subject, str)
            or not subject
            or doc.get("answer") != target
        ):
            raise ValueError(f"step{step} {task} sample {position} source identity is invalid")
        if document_id in document_ids:
            raise ValueError(f"step{step} {task} repeats document id {document_id!r}")
        document_ids.add(document_id)
        score = _mapping(row["mmmu_acc"], name=f"step{step} {task} sample score")
        if set(score) != {"answer", "id", "parsed_pred", "subject"} or (
            score.get("id") != document_id
            or score.get("answer") != target
            or score.get("subject") != subject
        ):
            raise ValueError(f"step{step} {task} sample {position} score identity differs")
        prediction = score.get("parsed_pred")
        if not isinstance(prediction, str) or prediction not in string.ascii_uppercase[:10]:
            raise ValueError(f"step{step} {task} sample {position} prediction is invalid")
        raw_responses = _sequence(row["resps"], name=f"step{step} {task} raw responses")
        filtered_responses = _sequence(
            row["filtered_resps"], name=f"step{step} {task} filtered responses"
        )
        if (
            len(raw_responses) != 1
            or not isinstance(raw_responses[0], Sequence)
            or isinstance(raw_responses[0], (str, bytes, bytearray))
            or len(raw_responses[0]) != 1
            or len(filtered_responses) != 1
            or filtered_responses[0] != raw_responses[0][0]
            or not isinstance(filtered_responses[0], str)
        ):
            raise ValueError(f"step{step} {task} sample {position} response chain differs")
        raw_prediction = filtered_responses[0]
        if raw_prediction not in string.ascii_uppercase[:10]:
            raise ValueError(f"step{step} {task} sample {position} raw response is not a letter")
        options_value = doc.get("options")
        if not isinstance(options_value, str):
            raise ValueError(f"step{step} {task} sample {position} options metadata differs")
        try:
            options = ast.literal_eval(options_value)
        except (SyntaxError, ValueError) as error:
            raise ValueError(
                f"step{step} {task} sample {position} options metadata is invalid"
            ) from error
        if (
            not isinstance(options, list)
            or len(options) < 2
            or len(options) > len(string.ascii_uppercase)
            or any(not isinstance(option, str) or not option for option in options)
        ):
            raise ValueError(f"step{step} {task} sample {position} options metadata differs")
        valid_option_letters = string.ascii_uppercase[: len(options)]
        if task == "mmmu_pro_vision":
            response_consistent = prediction == raw_prediction
        else:
            response_consistent = prediction in valid_option_letters and (
                raw_prediction not in valid_option_letters or prediction == raw_prediction
            )
        if not response_consistent:
            raise ValueError(f"step{step} {task} sample {position} parsed response differs")
        arguments = _sequence(row["arguments"], name=f"step{step} {task} sample arguments")
        if (
            len(arguments) != 5
            or not isinstance(arguments[0], str)
            or not arguments[0]
            or arguments[1] != {"max_new_tokens": 256, "until": ["\n\n"]}
            or arguments[2] != position
            or arguments[3] != task
            or arguments[4] != "test"
        ):
            raise ValueError(f"step{step} {task} sample {position} task arguments differ")
        if prediction == target:
            correct += 1
        token_counts = _sequence(row["token_counts"], name=f"step{step} {task} sample token counts")
        if len(token_counts) != 1:
            raise ValueError(f"step{step} {task} sample {position} token coverage differs")
        counts = _exact(
            token_counts[0],
            frozenset({"input_tokens", "output_tokens"}),
            name=f"step{step} {task} sample token counts",
        )
        if (
            _integer(counts.get("input_tokens"), name="MMMU input tokens", minimum=1) < 1
            or counts.get("output_tokens") != 1
        ):
            raise ValueError(f"step{step} {task} sample {position} is not letter-logit output")
        source_projection.append(
            {
                "doc_id": position,
                "doc_hash": row["doc_hash"],
                "target": target,
                "arguments": arguments,
                "doc": doc,
            }
        )
    return round(correct / MMMU_SAMPLES_PER_TASK, 5), correct, _canonical_sha256(source_projection)


def _validate_mmmu(
    payload: Mapping[str, Any], *, step: int, source: Mapping[str, str]
) -> dict[str, Any]:
    _validate_common(payload, step=step, suite="mmmu_pro")
    _validate_mmmu_protocol(payload["protocol"], step=step)
    lmms_eval = _exact(payload["lmms_eval"], _MMMU_EVAL_FIELDS, name="lmms-eval result")
    _validate_mmmu_harness_config(lmms_eval["config"], step=step)
    if (
        not isinstance(lmms_eval["date"], str)
        or re.fullmatch(r"[0-9]{8}_[0-9]{6}", lmms_eval["date"]) is None
        or lmms_eval["git_branch"] != "HEAD"
        or lmms_eval["git_hash"] != EXPECTED_GIT_REF[:7]
        or lmms_eval["lmms_eval_version"] != f"HEAD@{EXPECTED_GIT_REF[:7]}"
        or lmms_eval["usage"] != {}
    ):
        raise ValueError(f"step{step} lmms-eval harness metadata differs")
    task_maps = {
        name: _validate_task_keys(lmms_eval[name], MMMU_TASKS, name=f"lmms-eval {name}")
        for name in (
            "configs",
            "group_subtasks",
            "higher_is_better",
            "n-samples",
            "n-shot",
            "results",
            "samples",
            "versions",
        )
    }
    metrics: dict[str, float] = {}
    correct_counts: dict[str, int] = {}
    projection_sha256: dict[str, str] = {}
    for task in MMMU_TASKS:
        _validate_mmmu_task_config(task_maps["configs"][task], task=task, step=step)
        sample_count = _mapping(task_maps["n-samples"][task], name=f"{task} sample count")
        if sample_count != {
            "effective": MMMU_SAMPLES_PER_TASK,
            "original": MMMU_SAMPLES_PER_TASK,
        }:
            raise ValueError(f"step{step} {task} n-samples coverage differs")
        if task_maps["n-shot"][task] != 0 or task_maps["versions"][task] != 0:
            raise ValueError(f"step{step} {task} version/few-shot protocol differs")
        if task_maps["group_subtasks"][task] != []:
            raise ValueError(f"step{step} {task} unexpectedly expands into subtasks")
        if task_maps["higher_is_better"][task] != {"mmmu_acc": True}:
            raise ValueError(f"step{step} {task} metric direction differs")
        aggregate = _exact(
            task_maps["results"][task],
            frozenset(
                {
                    "alias",
                    "mmmu_acc,none",
                    "mmmu_acc_stderr,none",
                    "mmmu_acc_stderr_clt,none",
                    "mmmu_acc_stderr_clustered,none",
                }
            ),
            name=f"step{step} {task} aggregate",
        )
        if aggregate["alias"] != task:
            raise ValueError(f"step{step} {task} aggregate alias differs")
        reported = _finite(
            aggregate["mmmu_acc,none"], name=f"step{step} {task} accuracy", minimum=0
        )
        if reported > 1 or any(
            aggregate[field] != "N/A"
            for field in (
                "mmmu_acc_stderr,none",
                "mmmu_acc_stderr_clt,none",
                "mmmu_acc_stderr_clustered,none",
            )
        ):
            raise ValueError(f"step{step} {task} aggregate metric fields differ")
        rederived, correct, projection = _validate_mmmu_samples(
            task_maps["samples"][task], task=task, step=step
        )
        if reported != rederived:
            raise ValueError(f"step{step} {task} aggregate accuracy differs from its samples")
        metrics[task] = reported
        correct_counts[task] = correct
        projection_sha256[task] = projection
    efficiency = _mapping(lmms_eval["efficiency"], name="lmms-eval efficiency")
    by_task = _validate_task_keys(
        efficiency.get("by_task"), MMMU_TASKS, name="lmms-eval efficiency by task"
    )
    for task in MMMU_TASKS:
        task_efficiency = _mapping(by_task[task], name=f"{task} efficiency")
        if (
            task_efficiency.get("docs") != MMMU_SAMPLES_PER_TASK
            or task_efficiency.get("docs_with_token_counts") != MMMU_SAMPLES_PER_TASK
            or task_efficiency.get("avg_output_tokens_per_sample") != 1
        ):
            raise ValueError(f"step{step} {task} efficiency coverage differs")
    overall = _mapping(efficiency.get("overall"), name="lmms-eval overall efficiency")
    if (
        overall.get("docs") != len(MMMU_TASKS) * MMMU_SAMPLES_PER_TASK
        or overall.get("docs_with_token_counts") != len(MMMU_TASKS) * MMMU_SAMPLES_PER_TASK
        or overall.get("avg_output_tokens_per_sample") != 1
    ):
        raise ValueError(f"step{step} MMMU overall sample coverage differs")
    return {
        "source": dict(source),
        "checkpoint": EXPECTED_CHECKPOINTS[step],
        "metrics": metrics,
        "correct_counts": correct_counts,
        "sample_projection_sha256": projection_sha256,
    }


def _validate_olmes_protocol(value: Any, *, step: int) -> None:
    protocol = _exact(value, _OLMES_PROTOCOL_FIELDS, name=f"step{step} OLMES protocol")
    expected = {
        "harness": "ai2-olmo-eval",
        "task_group": "fast",
        "tasks": list(OLMES_TASKS),
        "partial": False,
        "max_batches_per_task": None,
        "max_sequence_length": 2_048,
        "rank_batch_size_tokens": 8_192,
        "world_size": 8,
        "ep_degree": 8,
        "ep_dp_degree": 1,
        "attention_backend": "flex",
    }
    if dict(protocol) != expected:
        raise ValueError(f"step{step} OLMES protocol differs from the exact full fast gate")


def _olmes_primary_metric(task: str) -> tuple[str, str, str]:
    if task in _OLMES_LENGTH_NORMALIZED_TASKS:
        return f"{task} (length-normalized accuracy v2)", "accuracy", "higher_is_better"
    if task in _OLMES_ACCURACY_TASKS:
        return f"{task} (accuracy v2)", "accuracy", "higher_is_better"
    return f"{task} (BPB v2)", "bits_per_byte", "lower_is_better"


def _validate_olmes(
    payload: Mapping[str, Any], *, step: int, source: Mapping[str, str]
) -> dict[str, Any]:
    _validate_common(payload, step=step, suite="olmes_fast")
    _validate_olmes_protocol(payload["protocol"], step=step)
    results = _validate_task_keys(payload["results"], OLMES_TASKS, name="OLMES results")
    normalized: dict[str, Any] = {}
    for task in OLMES_TASKS:
        result = _exact(
            results[task],
            frozenset(
                {
                    "metrics",
                    "batches_per_ep_dp_rank",
                    "instances_per_ep_dp_rank",
                    "total_batches_per_ep_dp_rank",
                    "elapsed_seconds",
                }
            ),
            name=f"step{step} {task} result",
        )
        expected_instances, expected_batches = OLMES_COVERAGE[task]
        if (
            result["instances_per_ep_dp_rank"] != expected_instances
            or result["batches_per_ep_dp_rank"] != expected_batches
            or result["total_batches_per_ep_dp_rank"] != expected_batches
        ):
            raise ValueError(f"step{step} {task} exact instance/batch coverage differs")
        _finite(result["elapsed_seconds"], name=f"step{step} {task} elapsed", minimum=0)
        raw_metrics = _mapping(result["metrics"], name=f"step{step} {task} metrics")
        expected_metric_keys = _expected_olmes_metric_keys(task)
        if tuple(raw_metrics) != expected_metric_keys:
            raise ValueError(f"step{step} {task} metric-key schema differs")
        metrics: dict[str, float] = {}
        for metric_name, metric_value in raw_metrics.items():
            if not isinstance(metric_name, str) or not metric_name.startswith(f"{task} ("):
                raise ValueError(f"step{step} {task} has an unscoped metric name")
            metrics[metric_name] = _finite(
                metric_value, name=f"step{step} {task} metric {metric_name}"
            )
        primary_name, family, direction = _olmes_primary_metric(task)
        if primary_name not in metrics:
            raise ValueError(f"step{step} {task} is missing primary metric {primary_name!r}")
        normalized[task] = {
            "metrics": metrics,
            "primary_metric": primary_name,
            "metric_family": family,
            "direction": direction,
            "instances": expected_instances,
            "batches": expected_batches,
        }
    return {
        "source": dict(source),
        "checkpoint": EXPECTED_CHECKPOINTS[step],
        "tasks": normalized,
    }


def _producer() -> dict[str, Any]:
    script_path = Path(__file__).absolute()
    _, _, script_sha256 = _read_regular_file(script_path, name="comparator source")
    evaluators: dict[str, dict[str, str]] = {}
    for suite, expected in EXPECTED_EVALUATOR_SOURCES.items():
        path = script_path.with_name(expected["filename"])
        _, _, actual_sha256 = _read_regular_file(path, name=f"{suite} evaluator source")
        if actual_sha256 != expected["sha256"]:
            raise ValueError(
                f"{suite} evaluator source differs from git ref {EXPECTED_GIT_REF}: "
                f"expected {expected['sha256']}, got {actual_sha256}"
            )
        evaluators[suite] = {"path": str(path), "sha256": actual_sha256}
    return {
        "path": str(script_path),
        "sha256": script_sha256,
        "evaluator_sources": evaluators,
    }


def _metric_direction(metric_name: str) -> str:
    if "(accuracy" in metric_name or "(length-normalized accuracy" in metric_name:
        return "higher_is_better"
    if "(BPB" in metric_name or "(CE loss" in metric_name:
        return "lower_is_better"
    if "(soft loss" in metric_name or "(log soft loss" in metric_name:
        return "higher_is_better"
    return "diagnostic_only_unknown_direction"


def _delta(left: float, right: float, *, direction: str) -> dict[str, Any]:
    raw = right - left
    if direction == "higher_is_better":
        oriented: float | None = raw
    elif direction == "lower_is_better":
        oriented = -raw
    else:
        oriented = None
    return {
        "step12000": left,
        "step16000": right,
        "delta_step16000_minus_step12000": raw,
        "direction": direction,
        "oriented_improvement": oriented,
    }


def _task_comparisons(evaluations: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    first_mmmu = evaluations[12_000]["mmmu_pro"]
    second_mmmu = evaluations[16_000]["mmmu_pro"]
    mmmu: dict[str, Any] = {}
    for task in MMMU_TASKS:
        if (
            first_mmmu["sample_projection_sha256"][task]
            != second_mmmu["sample_projection_sha256"][task]
        ):
            raise ValueError(f"MMMU source sample projection differs across checkpoints for {task}")
        metric = _delta(
            first_mmmu["metrics"][task],
            second_mmmu["metrics"][task],
            direction="higher_is_better",
        )
        metric.update({"name": "mmmu_acc,none", "family": "accuracy"})
        mmmu[task] = {
            "category": "vision_mmmu_pro",
            "samples": MMMU_SAMPLES_PER_TASK,
            "sample_projection_sha256": first_mmmu["sample_projection_sha256"][task],
            "correct": {
                "step12000": first_mmmu["correct_counts"][task],
                "step16000": second_mmmu["correct_counts"][task],
            },
            "primary_metric": metric,
            "metric_deltas": {"mmmu_acc,none": dict(metric)},
        }

    first_olmes = evaluations[12_000]["olmes_fast"]
    second_olmes = evaluations[16_000]["olmes_fast"]
    olmes: dict[str, Any] = {}
    category_by_task = {
        task: category for category, tasks in OLMES_CATEGORY_TASKS.items() for task in tasks
    }
    for task in OLMES_TASKS:
        left = first_olmes["tasks"][task]
        right = second_olmes["tasks"][task]
        if (
            set(left["metrics"]) != set(right["metrics"])
            or left["primary_metric"] != right["primary_metric"]
            or left["metric_family"] != right["metric_family"]
            or left["direction"] != right["direction"]
        ):
            raise ValueError(f"OLMES metric schema differs across checkpoints for {task}")
        metric_deltas = {
            name: _delta(
                left["metrics"][name],
                right["metrics"][name],
                direction=_metric_direction(name),
            )
            for name in sorted(left["metrics"])
        }
        primary_name = left["primary_metric"]
        primary = dict(metric_deltas[primary_name])
        primary.update({"name": primary_name, "family": left["metric_family"]})
        olmes[task] = {
            "category": category_by_task[task],
            "instances": left["instances"],
            "batches_per_ep_dp_rank": left["batches"],
            "primary_metric": primary,
            "metric_deltas": metric_deltas,
        }
    return {"mmmu_pro": mmmu, "olmes_fast": olmes}


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty metric collection")
    return math.fsum(values) / len(values)


def _category_summary(
    comparisons: Mapping[str, Mapping[str, Any]], tasks: Sequence[str]
) -> dict[str, Any]:
    primary = [comparisons[task]["primary_metric"] for task in tasks]
    improved = sum(metric["oriented_improvement"] > 0 for metric in primary)
    tied = sum(metric["oriented_improvement"] == 0 for metric in primary)
    families: dict[str, Any] = {}
    for family in sorted({metric["family"] for metric in primary}):
        selected = [metric for metric in primary if metric["family"] == family]
        directions = {metric["direction"] for metric in selected}
        if len(directions) != 1:
            raise ValueError(f"Metric family {family!r} mixes directions")
        families[family] = {
            "task_count": len(selected),
            "direction": next(iter(directions)),
            "mean_step12000": _mean([metric["step12000"] for metric in selected]),
            "mean_step16000": _mean([metric["step16000"] for metric in selected]),
            "mean_delta_step16000_minus_step12000": _mean(
                [metric["delta_step16000_minus_step12000"] for metric in selected]
            ),
            "mean_oriented_improvement": _mean(
                [metric["oriented_improvement"] for metric in selected]
            ),
        }
    return {
        "tasks": list(tasks),
        "task_count": len(tasks),
        "directional_task_counts": {
            "improved_at_step16000": improved,
            "tied": tied,
            "regressed_at_step16000": len(primary) - improved - tied,
        },
        "metric_families": families,
    }


def _selection_summary(task_comparisons: Mapping[str, Any]) -> dict[str, Any]:
    mmmu = task_comparisons["mmmu_pro"]
    olmes = task_comparisons["olmes_fast"]
    categories = {
        "vision_mmmu_pro": _category_summary(mmmu, MMMU_TASKS),
        **{
            category: _category_summary(olmes, tasks)
            for category, tasks in OLMES_CATEGORY_TASKS.items()
        },
        "language_olmes_fast_all": _category_summary(olmes, OLMES_TASKS),
    }
    return {
        "checkpoint_selected": None,
        "automatic_promotion_enabled": False,
        "human_selection_required": True,
        "category_summaries": categories,
        "interpretation": (
            "Positive oriented deltas favor step16000 within each task's primary metric. "
            "Means are reported only within one metric family; accuracy and BPB are never "
            "averaged together, and no decision threshold is applied."
        ),
    }


def _build_receipt(
    evaluations: Mapping[int, Mapping[str, Any]], *, created_at: str | None = None
) -> dict[str, Any]:
    task_comparisons = _task_comparisons(evaluations)
    checkpoint_provenance = _load_checkpoint_provenance()
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    _timestamp(timestamp, name="comparison created_at")
    payload: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "status": "valid_descriptive_comparison",
        "created_at": timestamp,
        "producer": _producer(),
        "inputs": {
            f"step{step}": {suite: dict(evaluations[step][suite]["source"]) for suite in SUITES}
            for step in STEPS
        },
        "checkpoint_provenance": {f"step{step}": checkpoint_provenance[step] for step in STEPS},
        "policy": {
            "descriptive_only": True,
            "automatic_promotion_enabled": False,
            "promotion_decision": None,
            "selection_margins_defined": False,
            "reason": "No user-approved automatic selection rule or margins are defined.",
        },
        "protocol": {
            "name": PROTOCOL_NAME,
            "execution": "standard-library CPU-only; no model or accelerator imports",
            "steps": list(STEPS),
            "delta_convention": "step16000_minus_step12000",
            "expected_git_ref": EXPECTED_GIT_REF,
            "expected_checkpoint_config_sha256": EXPECTED_CONFIG_SHA256,
            "expected_checkpoints": {f"step{step}": EXPECTED_CHECKPOINTS[step] for step in STEPS},
            "input_integrity": "required caller-supplied raw SHA-256 per source JSON",
        },
        "coverage": {
            "mmmu_pro": {
                "tasks": list(MMMU_TASKS),
                "samples_per_task": MMMU_SAMPLES_PER_TASK,
                "sample_projection_sha256": dict(
                    evaluations[12_000]["mmmu_pro"]["sample_projection_sha256"]
                ),
            },
            "olmes_fast": {
                "tasks": list(OLMES_TASKS),
                "instances_and_batches_per_task": {
                    task: {"instances": coverage[0], "batches_per_ep_dp_rank": coverage[1]}
                    for task, coverage in OLMES_COVERAGE.items()
                },
            },
        },
        "task_comparisons": task_comparisons,
        "selection_summary": _selection_summary(task_comparisons),
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    _validate_receipt_structure(payload)
    return payload


def _validate_receipt_structure(value: Any) -> None:
    receipt = _exact(value, _OUTPUT_FIELDS, name="external comparison receipt")
    if (
        receipt["format"] != FORMAT
        or receipt["version"] != VERSION
        or receipt["status"] != "valid_descriptive_comparison"
    ):
        raise ValueError("External comparison receipt identity differs")
    _timestamp(receipt["created_at"], name="external comparison created_at")
    unsigned = dict(receipt)
    digest = unsigned.pop("content_sha256")
    if not _is_sha256(digest) or digest != _canonical_sha256(unsigned):
        raise ValueError("External comparison content SHA-256 differs")
    policy = _exact(
        receipt["policy"],
        frozenset(
            {
                "descriptive_only",
                "automatic_promotion_enabled",
                "promotion_decision",
                "selection_margins_defined",
                "reason",
            }
        ),
        name="external comparison policy",
    )
    if (
        policy["descriptive_only"] is not True
        or policy["automatic_promotion_enabled"] is not False
        or policy["promotion_decision"] is not None
        or policy["selection_margins_defined"] is not False
    ):
        raise ValueError("External comparison violates the descriptive-only policy")
    inputs = _exact(receipt["inputs"], frozenset(STEP_KEYS), name="comparison inputs")
    for step_key in STEP_KEYS:
        suites = _exact(inputs[step_key], frozenset(SUITES), name=f"{step_key} inputs")
        for suite in SUITES:
            reference = _exact(
                suites[suite], frozenset({"path", "sha256"}), name=f"{step_key} {suite} input"
            )
            if not isinstance(reference["path"], str) or not _is_sha256(reference["sha256"]):
                raise ValueError(f"{step_key} {suite} input reference is invalid")
    checkpoint_provenance = _exact(
        receipt["checkpoint_provenance"],
        frozenset(STEP_KEYS),
        name="comparison checkpoint provenance",
    )
    for step_key in STEP_KEYS:
        provenance = _exact(
            checkpoint_provenance[step_key],
            frozenset(
                {
                    "live_config",
                    "strict_v2_receipt",
                    "checkpoint_identity_sha256",
                    "model_and_optim_identity_sha256",
                    "state_file_inventory_sha256",
                    "trainer_state_file_inventory_sha256",
                    "load_coverage_sha256",
                }
            ),
            name=f"{step_key} checkpoint provenance",
        )
        _exact(
            provenance["live_config"],
            frozenset({"path", "sha256"}),
            name=f"{step_key} live config",
        )
        _exact(
            provenance["strict_v2_receipt"],
            frozenset({"path", "sha256", "content_sha256"}),
            name=f"{step_key} strict V2 receipt",
        )
        for field in (
            "checkpoint_identity_sha256",
            "model_and_optim_identity_sha256",
            "state_file_inventory_sha256",
            "trainer_state_file_inventory_sha256",
            "load_coverage_sha256",
        ):
            if not _is_sha256(provenance[field]):
                raise ValueError(f"{step_key} {field} is invalid")
    selection = _mapping(receipt["selection_summary"], name="selection summary")
    if (
        selection.get("checkpoint_selected") is not None
        or selection.get("automatic_promotion_enabled") is not False
        or selection.get("human_selection_required") is not True
    ):
        raise ValueError("Selection summary must not promote a checkpoint")
    coverage = _mapping(receipt["coverage"], name="comparison coverage")
    if coverage.get("mmmu_pro", {}).get("samples_per_task") != MMMU_SAMPLES_PER_TASK:
        raise ValueError("Comparison MMMU coverage differs")
    if coverage.get("olmes_fast", {}).get("tasks") != list(OLMES_TASKS):
        raise ValueError("Comparison OLMES coverage differs")


def _load_evaluations(
    source_paths: Mapping[int, Mapping[str, str | Path]],
    expected_sha256: Mapping[int, Mapping[str, str]],
) -> dict[int, dict[str, Any]]:
    evaluations: dict[int, dict[str, Any]] = {}
    seen_paths: set[Path] = set()
    for step in STEPS:
        evaluations[step] = {}
        for suite in SUITES:
            name = f"step{step} {suite} source"
            path, payload, digest = _load_json_source(
                source_paths[step][suite],
                expected_sha256=expected_sha256[step][suite],
                name=name,
            )
            if path in seen_paths:
                raise ValueError("All four external diagnostic inputs must be distinct files")
            seen_paths.add(path)
            source = {"path": str(path), "sha256": digest}
            evaluations[step][suite] = (
                _validate_mmmu(payload, step=step, source=source)
                if suite == "mmmu_pro"
                else _validate_olmes(payload, step=step, source=source)
            )
    return evaluations


def build_comparison_receipt(
    *,
    step12000_mmmu_pro_path: str | Path,
    step12000_mmmu_pro_sha256: str,
    step12000_olmes_fast_path: str | Path,
    step12000_olmes_fast_sha256: str,
    step16000_mmmu_pro_path: str | Path,
    step16000_mmmu_pro_sha256: str,
    step16000_olmes_fast_path: str | Path,
    step16000_olmes_fast_sha256: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a strict descriptive receipt from four raw-SHA-pinned result files."""
    source_paths: dict[int, dict[str, str | Path]] = {
        12_000: {
            "mmmu_pro": step12000_mmmu_pro_path,
            "olmes_fast": step12000_olmes_fast_path,
        },
        16_000: {
            "mmmu_pro": step16000_mmmu_pro_path,
            "olmes_fast": step16000_olmes_fast_path,
        },
    }
    expected_sha256 = {
        12_000: {
            "mmmu_pro": step12000_mmmu_pro_sha256,
            "olmes_fast": step12000_olmes_fast_sha256,
        },
        16_000: {
            "mmmu_pro": step16000_mmmu_pro_sha256,
            "olmes_fast": step16000_olmes_fast_sha256,
        },
    }
    return _build_receipt(_load_evaluations(source_paths, expected_sha256), created_at=created_at)


def validate_comparison_receipt(value: Any, *, verify_inputs: bool = True) -> None:
    """Validate a comparison receipt and optionally re-open and fully rederive its inputs."""
    _validate_receipt_structure(value)
    if not verify_inputs:
        return
    receipt = _mapping(value, name="external comparison receipt")
    inputs = _mapping(receipt["inputs"], name="comparison inputs")
    source_paths: dict[int, dict[str, str | Path]] = {}
    expected_sha256: dict[int, dict[str, str]] = {}
    for step in STEPS:
        step_inputs = _mapping(inputs[f"step{step}"], name=f"step{step} inputs")
        source_paths[step] = {
            suite: str(_mapping(step_inputs[suite], name=suite)["path"]) for suite in SUITES
        }
        expected_sha256[step] = {
            suite: str(_mapping(step_inputs[suite], name=suite)["sha256"]) for suite in SUITES
        }
    expected = _build_receipt(
        _load_evaluations(source_paths, expected_sha256), created_at=str(receipt["created_at"])
    )
    if _canonical_bytes(receipt) != _canonical_bytes(expected):
        raise ValueError("External comparison differs from complete input rederivation")


def _safe_output_parent(path: Path) -> os.stat_result:
    current = Path(path.anchor)
    for part in path.parent.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir(mode=0o755)
            except FileExistsError:
                pass
            info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Output ancestor must be a direct directory: {current}")
    return path.parent.lstat()


def _write_json_no_overwrite(path_value: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically publish deterministic JSON without replacing an existing path."""
    path = Path(os.path.abspath(Path(path_value).expanduser()))
    if path.suffix != ".json" or path.name in {"", ".", ".."}:
        raise ValueError("External comparison output must be one explicit .json path")
    parent_info = _safe_output_parent(path)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_descriptor = os.open(path.parent, directory_flags)
    temporary_name = f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    temporary_identity: tuple[int, int] | None = None
    try:
        opened_parent = os.fstat(directory_descriptor)
        if (opened_parent.st_dev, opened_parent.st_ino) != (
            parent_info.st_dev,
            parent_info.st_ino,
        ):
            raise RuntimeError("External comparison output parent changed before publication")
        try:
            os.stat(path.name, dir_fd=directory_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"Refusing to overwrite external comparison {path}")
        raw = _serialized_bytes(payload)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temporary_name, flags, 0o644, dir_fd=directory_descriptor)
        try:
            temporary_info = os.fstat(descriptor)
            temporary_identity = (temporary_info.st_dev, temporary_info.st_ino)
            if not stat.S_ISREG(temporary_info.st_mode) or temporary_info.st_nlink != 1:
                raise RuntimeError("External comparison temporary file identity is invalid")
            with os.fdopen(descriptor, "wb", closefd=False) as file_handle:
                file_handle.write(raw)
                file_handle.flush()
                os.fsync(file_handle.fileno())
        finally:
            os.close(descriptor)
        current_temporary = os.stat(
            temporary_name, dir_fd=directory_descriptor, follow_symlinks=False
        )
        if (current_temporary.st_dev, current_temporary.st_ino) != temporary_identity:
            raise RuntimeError("External comparison temporary file ownership changed")
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite external comparison {path}") from error
        published = os.stat(path.name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (published.st_dev, published.st_ino) != temporary_identity or not stat.S_ISREG(
            published.st_mode
        ):
            raise RuntimeError("External comparison publication identity differs")
        os.fsync(directory_descriptor)
    finally:
        if temporary_identity is not None:
            try:
                current = os.stat(
                    temporary_name, dir_fd=directory_descriptor, follow_symlinks=False
                )
            except FileNotFoundError:
                pass
            else:
                if (current.st_dev, current.st_ino) == temporary_identity:
                    os.unlink(temporary_name, dir_fd=directory_descriptor)
        os.close(directory_descriptor)
    closing_parent = path.parent.lstat()
    if (closing_parent.st_dev, closing_parent.st_ino) != (parent_info.st_dev, parent_info.st_ino):
        raise RuntimeError("External comparison output parent changed during publication")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    """Run the strict CPU-only external comparison and publish one immutable receipt."""
    args = _parser().parse_args(argv)
    receipt = build_comparison_receipt(
        step12000_mmmu_pro_path=args.step12000_mmmu_pro,
        step12000_mmmu_pro_sha256=args.expected_step12000_mmmu_pro_sha256,
        step12000_olmes_fast_path=args.step12000_olmes_fast,
        step12000_olmes_fast_sha256=args.expected_step12000_olmes_fast_sha256,
        step16000_mmmu_pro_path=args.step16000_mmmu_pro,
        step16000_mmmu_pro_sha256=args.expected_step16000_mmmu_pro_sha256,
        step16000_olmes_fast_path=args.step16000_olmes_fast,
        step16000_olmes_fast_sha256=args.expected_step16000_olmes_fast_sha256,
    )
    validate_comparison_receipt(receipt, verify_inputs=True)
    output = _write_json_no_overwrite(args.output, receipt)
    _, written, _ = _load_json_source(
        output,
        expected_sha256=hashlib.sha256(_serialized_bytes(receipt)).hexdigest(),
        name="written external comparison",
    )
    validate_comparison_receipt(written, verify_inputs=False)
    if _canonical_bytes(written) != _canonical_bytes(receipt):
        raise RuntimeError("Written external comparison decodes differently")
    print(
        json.dumps(
            {
                "status": "valid_descriptive_comparison",
                "output": str(output),
                "content_sha256": receipt["content_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
