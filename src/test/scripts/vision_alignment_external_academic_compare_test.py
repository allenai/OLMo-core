"""Focused contracts for the external-academic checkpoint comparator."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "eval"
        / "vision_alignment_external_academic_compare.py"
    )
    name = "_vision_alignment_external_academic_compare_test_module"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _control(score: float, *, free_answer: bool, capped: bool = False) -> dict[str, Any]:
    output = {
        "score": score,
        "input_tokens": 32,
        "image_grid_signature": [14, 14, 14, 14],
        "image_token_count": 435,
        "image_token_ids_sha256": "9" * 64,
    }
    if free_answer:
        output["stop_reason"] = "max_tokens" if capped else "eos"
    return output


def _row(task: str, position: int, step: int) -> dict[str, Any]:
    scores = {
        12_000: {
            "correct": (1.0, 0.0, 0.0, 1.0),
            "shuffled": (0.0, 0.0, 0.0, 1.0),
            "blank": (0.0, 0.0, 0.0, 0.0),
        },
        16_000: {
            "correct": (1.0, 1.0, 0.0, 0.0),
            "shuffled": (0.0, 0.0, 0.0, 0.0),
            "blank": (1.0, 0.0, 0.0, 0.0),
        },
    }
    strata = {
        "chartqa": ("human", "human", "augmented", "augmented"),
        "ai2d": ("standard", "standard", "transparent", "transparent"),
    }
    free_answer = task in module._FREE_ANSWER_TASKS
    controls = {
        control: _control(
            scores[step][control][position],
            free_answer=free_answer,
            capped=free_answer and control == "correct" and position < (1 if step == 12_000 else 2),
        )
        for control in module.CONTROLS
    }
    return {
        "example_id": f"{task}-{position}",
        "source_position": str(position),
        "annotation_sha256": f"{position + 10:064x}",
        "image_sha256": ("a", "a", "b", "c")[position] * 64,
        "image_grid_signature": [14, 14, 14, 14],
        "image_token_count": 435,
        "alignment_train_image_overlap": position == 2,
        "shuffled_donor_id": f"{task}-{(position + 1) % 4}",
        "shuffled_image_sha256": ("d", "e", "f", "1")[position] * 64,
        "shuffled_image_grid_signature": [14, 14, 14, 14],
        "shuffled_alignment_train_image_overlap": position == 1,
        "question": f"Question {position}?",
        "gold_answers": ["yes"],
        "options": [] if free_answer else ["yes", "no"],
        "gold_answer_index": None if free_answer else 0,
        "stratum": strata.get(task, (None,) * 4)[position],
        "controls": dict(sorted(controls.items())),
    }


def _receipt(step: int) -> dict[str, Any]:
    tasks = {}
    for task, metric in module._TASK_METRICS.items():
        rows = [_row(task, position, step) for position in range(4)]
        tasks[task] = {
            "source": {"task": task, "split": "validation"},
            "selection_count": len(rows),
            "selection_sha256": f"{len(tasks) + 20:064x}",
            "metric": metric,
            "examples": rows,
        }
    return {
        "schema_version": 1,
        "format": "vision_alignment_external_academic_receipt",
        "protocol_name": "vision-alignment-external-academic-ep8-v1",
        "created_at": "2026-08-16T00:00:00+00:00",
        "git": {"revision": "1" * 40, "dirty": False},
        "implementation": {"files_sha256": "2" * 64},
        "manifest": {
            "path": "/manifest.json",
            "bytes": 10,
            "sha256": "3" * 64,
            "content_sha256": "4" * 64,
            "partial": True,
            "panel_status": "confirmatory",
        },
        "checkpoint": {
            "checkpoint": f"/checkpoints/step{step}",
            "state_file_inventory_sha256": f"{step:064x}",
        },
        "prior_matched_wrong_v2": {"step": step},
        "artifact_policy": {
            "descriptive_only": True,
            "promotion_eligible": False,
            "checkpoint_selection_evidence": True,
        },
        "tokenizer": {"fingerprint": "5" * 64},
        "protocol": {
            "tasks": list(module._TASK_METRICS),
            "controls": list(module.CONTROLS),
            "metrics": dict(module._TASK_METRICS),
        },
        # Canonical receipt JSON sorts object keys; scientific task order lives in protocol.tasks.
        "tasks": dict(sorted(tasks.items())),
        "content_sha256": f"{step + 100:064x}",
    }


def _write_inputs(tmp_path: Path, monkeypatch):
    receipts = {step: _receipt(step) for step in module.STEPS}
    paths = {}
    references = {}
    calls = []
    by_path = {}
    for step in module.STEPS:
        path = tmp_path / f"step{step}.json"
        path.write_text(f'{{"step":{step}}}\n', encoding="utf-8")
        paths[step] = path
        by_path[str(path.resolve())] = receipts[step]
        identity = module._file_identity(path, name="test input")
        references[step] = {
            **identity,
            "content_sha256": receipts[step]["content_sha256"],
        }

    def validate(path, expected_sha256, *, verify_live):
        resolved = str(Path(path).resolve())
        calls.append((resolved, expected_sha256, verify_live))
        assert module._file_identity(Path(path), name="test input")["sha256"] == expected_sha256
        return copy.deepcopy(by_path[resolved])

    monkeypatch.setattr(
        module,
        "_ACADEMIC_EVALUATOR",
        SimpleNamespace(validate_external_academic_receipt=validate),
    )
    return receipts, references, calls


def test_build_is_paired_clustered_stratified_and_descriptive(tmp_path, monkeypatch):
    receipts, references, _ = _write_inputs(tmp_path, monkeypatch)
    first = module._build_comparison_receipt(
        receipts,
        references,
        bootstrap_seed=17,
        bootstrap_samples=50,
        created_at="2026-08-16T01:00:00+00:00",
    )
    second = module._build_comparison_receipt(
        receipts,
        references,
        bootstrap_seed=17,
        bootstrap_samples=50,
        created_at="2026-08-16T01:00:00+00:00",
    )
    assert first == second

    task = first["tasks"]["vqav2"]
    correct = task["all_examples"]["correct_accuracy"]
    assert correct["step12000"]["mean"] == 0.5
    assert correct["step16000"]["mean"] == 0.5
    assert correct["paired_delta_16000_minus_12000"]["mean"] == 0.0
    assert correct["step12000"]["clusters"] == 3
    assert correct["mcnemar_exact_score_one"]["step12000_only"] == 1
    assert correct["mcnemar_exact_score_one"]["step16000_only"] == 1
    assert correct["mcnemar_exact_score_one"]["interpretation"].startswith("dichotomized")

    shuffled = task["all_examples"]["image_use"]["shuffled"]
    assert shuffled["step12000_correct_minus_control"]["mean"] == 0.25
    assert shuffled["step16000_correct_minus_control"]["mean"] == 0.5
    assert shuffled["paired_delta_16000_minus_12000"]["mean"] == 0.25
    exact = task["exact_byte_nonoverlap"]
    assert exact["correct_accuracy"]["step12000"]["examples"] == 3
    assert exact["image_use"]["shuffled"]["step12000_correct_minus_control"]["examples"] == 2
    assert task["all_examples"]["generation_cap"]["correct"]["rate_delta_16000_minus_12000"] == 0.25

    assert set(first["tasks"]["chartqa"]["strata"]) == {"human", "augmented"}
    macro = first["equal_task_macro"]["all_examples"]["statistics"]
    assert macro["correct_delta_16000_minus_12000"]["mean"] == 0.0
    assert macro["correct_delta_16000_minus_12000"]["included_tasks"] == list(module._TASK_METRICS)
    assert first["policy"]["promotion_eligible"] is False
    assert first["policy"]["promotion_decision"] is None


def test_cross_checkpoint_manifest_protocol_and_order_drift_are_rejected(tmp_path, monkeypatch):
    receipts, references, _ = _write_inputs(tmp_path, monkeypatch)
    changed = copy.deepcopy(receipts)
    changed[16_000]["manifest"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="shared manifest"):
        module._build_comparison_receipt(
            changed,
            references,
            bootstrap_seed=17,
            bootstrap_samples=2,
            created_at="2026-08-16T01:00:00+00:00",
        )

    changed = copy.deepcopy(receipts)
    changed[16_000]["tasks"]["vqav2"]["examples"][:2] = reversed(
        changed[16_000]["tasks"]["vqav2"]["examples"][:2]
    )
    with pytest.raises(ValueError, match="identity or ordering"):
        module._build_comparison_receipt(
            changed,
            references,
            bootstrap_seed=17,
            bootstrap_samples=2,
            created_at="2026-08-16T01:00:00+00:00",
        )


def test_public_validators_full_rederive_tamper_rejection_and_no_overwrite(tmp_path, monkeypatch):
    receipts, references, calls = _write_inputs(tmp_path, monkeypatch)
    comparison = module._build_comparison_receipt(
        receipts,
        references,
        bootstrap_seed=23,
        bootstrap_samples=5,
        created_at="2026-08-16T01:00:00+00:00",
    )
    output = tmp_path / "comparison.json"
    module._write_json_no_overwrite(output, comparison)
    raw_sha256 = module._file_identity(output, name="test comparison")["sha256"]
    assert (
        module.validate_academic_comparison_receipt(
            output,
            raw_sha256,
            verify_live_inputs=True,
        )
        == comparison
    )
    assert [call[2] for call in calls] == [True, True]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module._write_json_no_overwrite(output, comparison)

    tampered = copy.deepcopy(comparison)
    tampered["tasks"]["vqav2"]["all_examples"]["correct_accuracy"][
        "paired_delta_16000_minus_12000"
    ]["mean"] = 0.125
    del tampered["content_sha256"]
    tampered = module._attach_content_sha256(tampered)
    tampered_path = tmp_path / "tampered.json"
    module._write_json_no_overwrite(tampered_path, tampered)
    tampered_sha256 = module._file_identity(tampered_path, name="tampered comparison")["sha256"]
    with pytest.raises(ValueError, match="full rederivation"):
        module.validate_academic_comparison_receipt(
            tampered_path,
            tampered_sha256,
            verify_live_inputs=False,
        )
