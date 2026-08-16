"""Focused contracts for the joint external diagnostic comparator."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "eval"
        / "vision_alignment_joint_external_compare.py"
    )
    name = "_vision_alignment_joint_external_compare_test_module"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _git() -> dict[str, Any]:
    return {"revision": module.EXPECTED_GIT_REF, "dirty": False}


def _common(step: int) -> dict[str, Any]:
    checkpoint = module.EXPECTED_CHECKPOINTS[step]
    return {
        "schema_version": 1,
        "created_at": "2026-08-16T00:00:00+00:00",
        "checkpoint": checkpoint,
        "checkpoint_state_dir": f"{checkpoint}/model_and_optim",
        "config": f"{checkpoint}/config.json",
        "git": _git(),
    }


def _mmmu_protocol() -> dict[str, Any]:
    return {
        "harness": "lmms-eval",
        "tasks": list(module.MMMU_TASKS),
        "partial": False,
        "limit": None,
        "world_size": 8,
        "ep_degree": 8,
        "expert_parallel_path": "sync_1d",
        "logical_eval_replicas": 1,
        "max_sequence_length": 8192,
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
        "text_vocab_size": 100278,
        "generation": "single_forward_option_letter_logits",
    }


def _mmmu_samples(task: str, correct: int) -> list[dict[str, Any]]:
    rows = []
    for position in range(module.MMMU_SAMPLES_PER_TASK):
        document_id = f"{task}-{position}"
        prediction = "A" if position < correct else "B"
        raw_prediction = prediction
        option_letters = "ABCDEFGHIJ"
        if task == "mmmu_pro_standard" and position == module.MMMU_SAMPLES_PER_TASK - 1:
            raw_prediction = "J"
            option_letters = "AB"
        rows.append(
            {
                "arguments": [
                    "answer with a letter",
                    {"max_new_tokens": 256, "until": ["\n\n"]},
                    position,
                    task,
                    "test",
                ],
                "doc": {
                    "id": document_id,
                    "answer": "A",
                    "subject": "test",
                    "options": str([f"option {letter}" for letter in option_letters]),
                },
                "doc_hash": module.EXPECTED_MMMU_DOC_HASH,
                "doc_id": position,
                "filtered_resps": [raw_prediction],
                "mmmu_acc": {
                    "id": document_id,
                    "answer": "A",
                    "parsed_pred": prediction,
                    "subject": "test",
                },
                "resps": [[raw_prediction]],
                "target": "A",
                "token_counts": [{"input_tokens": 100 + position, "output_tokens": 1}],
            }
        )
    return rows


def _mmmu_task_config(task: str) -> dict[str, Any]:
    return {
        "dataset_name": "vision" if task == "mmmu_pro_vision" else "standard (10 options)",
        "dataset_path": "MMMU/MMMU_Pro",
        "description": "",
        "doc_to_target": "{{answer}}",
        "doc_to_text": "<function mmmu_pro_doc_to_text at 0x1>",
        "doc_to_visual": "<function mmmu_pro_doc_to_visual at 0x2>",
        "fewshot_delimiter": "\n\n",
        "full_docs": False,
        "generation_kwargs": {"max_new_tokens": 256, "until": ["\n\n"]},
        "lmms_eval_specific_kwargs": {
            "default": {
                "pre_prompt": "",
                "post_prompt": "Answer with the option letter from the given choices directly.",
            },
            "penguinvl": {},
            "qwen3_vl": {},
        },
        "metadata": {"interleaved_format": False, "version": 0},
        "metric_list": [
            {
                "aggregation": "<function mmmu_pro_aggregate_results at 0x3>",
                "higher_is_better": True,
                "metric": "mmmu_acc",
            }
        ],
        "num_fewshot": 0,
        "output_type": "generate_until",
        "process_results": "<function mmmu_pro_process_results at 0x4>",
        "process_results_use_image": False,
        "repeats": 1,
        "score_key": "score",
        "should_decontaminate": False,
        "tag": ["public_eval_qwen3_5_family", "public_eval_gemini3_family"],
        "target_delimiter": " ",
        "task": task,
        "test_split": "test",
    }


def _mmmu_payload(step: int) -> dict[str, Any]:
    correct_counts = {
        "mmmu_pro_vision": 865 if step == 12000 else 900,
        "mmmu_pro_standard": 692 if step == 12000 else 675,
    }
    samples = {task: _mmmu_samples(task, correct_counts[task]) for task in module.MMMU_TASKS}
    result = {
        **_common(step),
        "protocol": _mmmu_protocol(),
        "lmms_eval": {
            "config": {
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
            },
            "configs": {task: _mmmu_task_config(task) for task in module.MMMU_TASKS},
            "date": "20260816_010000",
            "efficiency": {
                "by_task": {
                    task: {
                        "docs": module.MMMU_SAMPLES_PER_TASK,
                        "docs_with_token_counts": module.MMMU_SAMPLES_PER_TASK,
                        "avg_output_tokens_per_sample": 1,
                    }
                    for task in module.MMMU_TASKS
                },
                "overall": {
                    "docs": len(module.MMMU_TASKS) * module.MMMU_SAMPLES_PER_TASK,
                    "docs_with_token_counts": (
                        len(module.MMMU_TASKS) * module.MMMU_SAMPLES_PER_TASK
                    ),
                    "avg_output_tokens_per_sample": 1,
                },
            },
            "git_branch": "HEAD",
            "git_hash": module.EXPECTED_GIT_REF[:7],
            "group_subtasks": {task: [] for task in module.MMMU_TASKS},
            "higher_is_better": {task: {"mmmu_acc": True} for task in module.MMMU_TASKS},
            "lmms_eval_version": f"HEAD@{module.EXPECTED_GIT_REF[:7]}",
            "n-samples": {
                task: {
                    "effective": module.MMMU_SAMPLES_PER_TASK,
                    "original": module.MMMU_SAMPLES_PER_TASK,
                }
                for task in module.MMMU_TASKS
            },
            "n-shot": {task: 0 for task in module.MMMU_TASKS},
            "results": {
                task: {
                    "alias": task,
                    "mmmu_acc,none": round(correct_counts[task] / module.MMMU_SAMPLES_PER_TASK, 5),
                    "mmmu_acc_stderr,none": "N/A",
                    "mmmu_acc_stderr_clt,none": "N/A",
                    "mmmu_acc_stderr_clustered,none": "N/A",
                }
                for task in module.MMMU_TASKS
            },
            "samples": samples,
            "usage": {},
            "versions": {task: 0 for task in module.MMMU_TASKS},
        },
    }
    return result


def _olmes_protocol() -> dict[str, Any]:
    return {
        "harness": "ai2-olmo-eval",
        "task_group": "fast",
        "tasks": list(module.OLMES_TASKS),
        "partial": False,
        "max_batches_per_task": None,
        "max_sequence_length": 2048,
        "rank_batch_size_tokens": 8192,
        "world_size": 8,
        "ep_degree": 8,
        "ep_dp_degree": 1,
        "attention_backend": "flex",
    }


def _olmes_payload(step: int) -> dict[str, Any]:
    results = {}
    for index, task in enumerate(module.OLMES_TASKS):
        metrics = {}
        for metric_index, metric_name in enumerate(module._expected_olmes_metric_keys(task)):
            direction = module._metric_direction(metric_name)
            base = (
                0.5 + index / 1000 + metric_index / 10000
                if direction == "higher_is_better"
                else 1.5 + index / 1000 + metric_index / 10000
            )
            value = base
            if step == 16000 and direction == "higher_is_better":
                value += 0.01
            elif step == 16000 and direction == "lower_is_better":
                value -= 0.01
            metrics[metric_name] = value
        instances, batches = module.OLMES_COVERAGE[task]
        results[task] = {
            "metrics": metrics,
            "batches_per_ep_dp_rank": batches,
            "instances_per_ep_dp_rank": instances,
            "total_batches_per_ep_dp_rank": batches,
            "elapsed_seconds": 1.0,
        }
    return {
        **_common(step),
        "checkpoint_kind": "multimodal_stage1",
        "protocol": _olmes_protocol(),
        "results": results,
    }


def _write_source(path: Path, value: dict[str, Any]) -> str:
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _fake_provenance() -> dict[int, dict[str, Any]]:
    return {
        step: {
            "live_config": {
                "path": f"{module.EXPECTED_CHECKPOINTS[step]}/config.json",
                "sha256": module.EXPECTED_CONFIG_SHA256,
            },
            "strict_v2_receipt": {
                "path": f"/receipts/step{step}.json",
                "sha256": f"{step:064x}",
                "content_sha256": f"{step + 1:064x}",
            },
            "checkpoint_identity_sha256": f"{step + 2:064x}",
            "model_and_optim_identity_sha256": f"{step + 3:064x}",
            "state_file_inventory_sha256": f"{step + 4:064x}",
            "trainer_state_file_inventory_sha256": f"{step + 5:064x}",
            "load_coverage_sha256": module.EXPECTED_V2_LOAD_COVERAGE["sha256"],
        }
        for step in module.STEPS
    }


def _use_fake_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_load_checkpoint_provenance", _fake_provenance)


def _strict_v2_receipt(
    *, step: int, checkpoint: str, config_sha256: str, expected: dict[str, str]
) -> dict[str, Any]:
    state_inventory: list[dict[str, Any]] = []
    trainer_inventory: list[dict[str, Any]] = []
    checkpoint_identity = {
        "root": checkpoint,
        "state_dir": f"{checkpoint}/model_and_optim",
        "config_sha256": config_sha256,
        "checkpoint_step": step,
        "permanent": True,
        "checkpoint_marker": {"ephemeral": False, "version": "2.5.0"},
        "checkpoint_marker_sha256": expected["checkpoint_marker_sha256"],
        "dcp_metadata_sha256": expected["dcp_metadata_sha256"],
        "model_and_optim_identity_sha256": expected["model_and_optim_identity_sha256"],
        "state_file_inventory": state_inventory,
        "state_file_inventory_sha256": module._canonical_sha256(state_inventory),
        "trainer_state_file_inventory": trainer_inventory,
        "trainer_state_file_inventory_sha256": module._canonical_sha256(trainer_inventory),
        "trainer_state_rank_count": 16,
        "trainer_state_summary": {
            "global_step": step,
            "batches_processed": step,
            "wandb_name": "vision-alignment-joint-v1",
        },
        "trainer_state_total_data_errors_by_rank": [0] * 16,
        "trainer_state_total_data_errors_sum": 0,
        "state_file_hash_algorithm": "sha256",
    }
    checkpoint_identity["identity_sha256"] = module._canonical_sha256(checkpoint_identity)
    expected.update(
        {
            "checkpoint_identity_sha256": checkpoint_identity["identity_sha256"],
            "state_file_inventory_sha256": checkpoint_identity["state_file_inventory_sha256"],
            "trainer_state_file_inventory_sha256": checkpoint_identity[
                "trainer_state_file_inventory_sha256"
            ],
        }
    )
    receipt = {
        "artifact_policy": {
            "checkpoint_post_identity_rehashed": True,
            "output_overwrite_enabled": False,
            "descriptive_only": True,
            "promotion_eligible": False,
        },
        "blank_results": {},
        "checkpoint": checkpoint_identity,
        "checkpoint_config": {
            "path": f"{checkpoint}/config.json",
            "sha256": config_sha256,
            "step": step,
            "phase": "joint",
            "lineage_id": "vision-alignment-joint-v1",
            "run_name": "vision-alignment-joint-v1",
        },
        "created_at": "2026-08-16T00:00:00+00:00",
        "endpoint": {
            "contract": "vision-alignment-joint-saved-endpoints-v1",
            "step": step,
            "storage_class": "scheduled_permanent",
            "nearest_step_substitution": False,
            "admissible_steps": [12000, 14400, 16000],
        },
        "format": "vision_alignment_joint_matched_wrong_receipt",
        "git": {
            "dirty": False,
            "revision": module.EXPECTED_GIT_REF,
            "status_sha256": hashlib.sha256(b"").hexdigest(),
            "tracked_diff_sha256": hashlib.sha256(b"").hexdigest(),
        },
        "load_coverage": dict(module.EXPECTED_V2_LOAD_COVERAGE),
        "native_result": {},
        "pairing_manifest": {},
        "producer": {},
        "projection": {},
        "protocol": {
            "name": "vision-alignment-joint-native-matched-wrong-saved-endpoints-v2",
            "evaluated_step": step,
            "checkpoint_config_sha256": config_sha256,
            "nearest_step_substitution": False,
            "descriptive_only": True,
            "promotion_eligible": False,
        },
        "source_audit": {},
        "status": "valid",
        "tokenizer": {},
        "version": 2,
        "visual_results": {},
    }
    receipt["content_sha256"] = module._canonical_sha256(receipt)
    return receipt


def _sources(tmp_path: Path) -> dict[str, Any]:
    values = {
        "step12000_mmmu_pro": _mmmu_payload(12000),
        "step12000_olmes_fast": _olmes_payload(12000),
        "step16000_mmmu_pro": _mmmu_payload(16000),
        "step16000_olmes_fast": _olmes_payload(16000),
    }
    arguments: dict[str, Any] = {}
    for name, value in values.items():
        path = tmp_path / f"{name}.json"
        digest = _write_source(path, value)
        arguments[f"{name}_path"] = path
        arguments[f"{name}_sha256"] = digest
    return arguments


def test_complete_comparison_rederives_deltas_and_never_promotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _use_fake_provenance(monkeypatch)
    arguments = _sources(tmp_path)
    receipt = module.build_comparison_receipt(**arguments, created_at="2026-08-16T01:00:00+00:00")

    vision = receipt["task_comparisons"]["mmmu_pro"]["mmmu_pro_vision"]["primary_metric"]
    standard = receipt["task_comparisons"]["mmmu_pro"]["mmmu_pro_standard"]["primary_metric"]
    language = receipt["selection_summary"]["category_summaries"]["language_olmes_fast_all"]
    assert vision["delta_step16000_minus_step12000"] > 0
    assert standard["delta_step16000_minus_step12000"] < 0
    assert language["directional_task_counts"]["improved_at_step16000"] == len(module.OLMES_TASKS)
    assert set(language["metric_families"]) == {"accuracy", "bits_per_byte"}
    assert "unweighted_mean_oriented_improvement" not in language
    assert receipt["policy"]["automatic_promotion_enabled"] is False
    assert receipt["policy"]["promotion_decision"] is None
    assert receipt["selection_summary"]["checkpoint_selected"] is None
    module.validate_comparison_receipt(receipt, verify_inputs=True)


def test_source_sha_partial_and_sample_coverage_are_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _use_fake_provenance(monkeypatch)
    arguments = _sources(tmp_path)
    wrong_sha = dict(arguments)
    wrong_sha["step12000_mmmu_pro_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="raw SHA-256 differs"):
        module.build_comparison_receipt(**wrong_sha)

    wrong_checkpoint = _olmes_payload(16000)
    wrong_checkpoint["checkpoint"] = module.EXPECTED_CHECKPOINTS[12000]
    wrong_checkpoint_path = Path(arguments["step16000_olmes_fast_path"])
    arguments["step16000_olmes_fast_sha256"] = _write_source(
        wrong_checkpoint_path, wrong_checkpoint
    )
    with pytest.raises(ValueError, match="checkpoint path differs"):
        module.build_comparison_receipt(**arguments)

    arguments = _sources(tmp_path)
    wrong_vocab = _mmmu_payload(12000)
    wrong_vocab["protocol"]["text_vocab_size"] = 100280
    wrong_vocab_path = Path(arguments["step12000_mmmu_pro_path"])
    arguments["step12000_mmmu_pro_sha256"] = _write_source(wrong_vocab_path, wrong_vocab)
    with pytest.raises(ValueError, match="full document gate"):
        module.build_comparison_receipt(**arguments)

    arguments = _sources(tmp_path)
    wrong_sample = _mmmu_payload(12000)
    wrong_sample["lmms_eval"]["samples"]["mmmu_pro_vision"][0]["doc"]["answer"] = "B"
    wrong_sample_path = Path(arguments["step12000_mmmu_pro_path"])
    arguments["step12000_mmmu_pro_sha256"] = _write_source(wrong_sample_path, wrong_sample)
    with pytest.raises(ValueError, match="source identity is invalid"):
        module.build_comparison_receipt(**arguments)

    arguments = _sources(tmp_path)
    wrong_metric_schema = _olmes_payload(12000)
    task = module.OLMES_TASKS[0]
    wrong_metric_schema["results"][task]["metrics"].popitem()
    wrong_metric_path = Path(arguments["step12000_olmes_fast_path"])
    arguments["step12000_olmes_fast_sha256"] = _write_source(wrong_metric_path, wrong_metric_schema)
    with pytest.raises(ValueError, match="metric-key schema differs"):
        module.build_comparison_receipt(**arguments)

    arguments = _sources(tmp_path)
    partial = _olmes_payload(12000)
    partial["protocol"]["partial"] = True
    partial_path = Path(arguments["step12000_olmes_fast_path"])
    arguments["step12000_olmes_fast_sha256"] = _write_source(partial_path, partial)
    with pytest.raises(ValueError, match="full fast gate"):
        module.build_comparison_receipt(**arguments)

    arguments = _sources(tmp_path)
    incomplete = _mmmu_payload(16000)
    incomplete["lmms_eval"]["samples"]["mmmu_pro_vision"].pop()
    incomplete_path = Path(arguments["step16000_mmmu_pro_path"])
    arguments["step16000_mmmu_pro_sha256"] = _write_source(incomplete_path, incomplete)
    with pytest.raises(ValueError, match="exactly 1730 samples"):
        module.build_comparison_receipt(**arguments)


def test_live_config_and_strict_v2_receipt_identities_are_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoints: dict[int, str] = {}
    expected_provenance: dict[int, dict[str, str]] = {}
    config_sha256 = ""
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    for step in module.STEPS:
        checkpoint_path = tmp_path / "checkpoints" / f"step{step}"
        checkpoint_path.mkdir(parents=True)
        config_sha256 = _write_source(
            checkpoint_path / "config.json", {"lineage": "joint-v1", "same": True}
        )
        checkpoint = str(checkpoint_path)
        checkpoints[step] = checkpoint
        expected = {
            "model_and_optim_identity_sha256": f"{step + 10:064x}",
            "checkpoint_marker_sha256": f"{step + 11:064x}",
            "dcp_metadata_sha256": f"{step + 12:064x}",
        }
        receipt = _strict_v2_receipt(
            step=step,
            checkpoint=checkpoint,
            config_sha256=config_sha256,
            expected=expected,
        )
        receipt_path = receipts / f"step{step}.json"
        receipt_sha256 = _write_source(receipt_path, receipt)
        expected.update(
            {
                "receipt_path": str(receipt_path),
                "receipt_sha256": receipt_sha256,
                "receipt_content_sha256": receipt["content_sha256"],
            }
        )
        expected_provenance[step] = expected
    monkeypatch.setattr(module, "EXPECTED_CHECKPOINTS", checkpoints)
    monkeypatch.setattr(module, "EXPECTED_CONFIG_SHA256", config_sha256)
    monkeypatch.setattr(module, "EXPECTED_CHECKPOINT_PROVENANCE", expected_provenance)

    provenance = module._load_checkpoint_provenance()
    for step in module.STEPS:
        assert provenance[step]["live_config"]["sha256"] == config_sha256
        assert (
            provenance[step]["checkpoint_identity_sha256"]
            == expected_provenance[step]["checkpoint_identity_sha256"]
        )
        assert (
            provenance[step]["load_coverage_sha256"] == module.EXPECTED_V2_LOAD_COVERAGE["sha256"]
        )

    receipt_path = Path(expected_provenance[12000]["receipt_path"])
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="raw SHA-256 differs"):
        module._load_checkpoint_provenance()


def test_output_is_canonical_and_never_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _use_fake_provenance(monkeypatch)
    receipt = module.build_comparison_receipt(
        **_sources(tmp_path), created_at="2026-08-16T01:00:00+00:00"
    )
    output = tmp_path / "comparison.json"
    module._write_json_no_overwrite(output, receipt)
    decoded = json.loads(output.read_text())
    assert decoded == receipt
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module._write_json_no_overwrite(output, copy.deepcopy(receipt))
    direct_parent = tmp_path / "direct-parent"
    direct_parent.mkdir()
    symlink_parent = tmp_path / "symlink-parent"
    symlink_parent.symlink_to(direct_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="Output ancestor"):
        module._write_json_no_overwrite(symlink_parent / "comparison.json", receipt)


def test_comparator_imports_only_cpu_standard_library():
    path = Path(module.__file__)
    tree = ast.parse(path.read_text())
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint({"torch", "numpy", "olmo_core"})
