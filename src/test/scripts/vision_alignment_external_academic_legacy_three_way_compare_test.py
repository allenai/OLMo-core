"""Focused contracts for the legacy/VA-12k/VA-16k academic comparator."""

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
        / "vision_alignment_external_academic_legacy_three_way_compare.py"
    )
    sys.path.insert(0, str(path.parent))
    name = "_vision_alignment_external_academic_legacy_three_way_compare_test_module"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _control(score: float, *, free_answer: bool, capped: bool = False) -> dict[str, Any]:
    output: dict[str, Any] = {
        "score": score,
        "input_tokens": 32,
        "image_grid_signature": [14, 14, 14, 14],
        "image_token_count": 435,
        "image_token_ids_sha256": "9" * 64,
    }
    if free_answer:
        output["stop_reason"] = "max_tokens" if capped else "eos"
    return output


def _row(task: str, position: int, model: str) -> dict[str, Any]:
    scores = {
        module.LEGACY_KEY: {
            "correct": (0.0, 0.0, 0.0, 1.0),
            "shuffled": (0.0, 0.0, 0.0, 0.0),
            "blank": (0.0, 0.0, 0.0, 0.0),
        },
        module.STEP12_KEY: {
            "correct": (1.0, 0.0, 0.0, 1.0),
            "shuffled": (0.0, 0.0, 0.0, 1.0),
            "blank": (0.0, 0.0, 0.0, 0.0),
        },
        module.STEP16_KEY: {
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
    cap_count = {module.LEGACY_KEY: 0, module.STEP12_KEY: 1, module.STEP16_KEY: 2}[model]
    controls = {
        control: _control(
            scores[model][control][position],
            free_answer=free_answer,
            capped=free_answer and control == "correct" and position < cap_count,
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
        # Receipt JSON canonicalization sorts these keys, so comparator admission must not
        # mistake object-key order for the scientific control order in protocol.controls.
        "controls": dict(sorted(controls.items())),
    }


def _protocol() -> dict[str, Any]:
    return {
        "tasks": list(module._TASK_METRICS),
        "controls": list(module.CONTROLS),
        "prompt": {
            "free_answer": "Question: {question}\\nAnswer:",
            "multiple_choice": "Question: {question}\\nOptions...",
        },
        "metrics": dict(module._TASK_METRICS),
    }


def _tokenizer(*, legacy: bool) -> dict[str, Any]:
    token_ids = {"im_start_id": 100278, "im_patch_id": 100280}
    payload = {
        "identifier": "allenai/dolma2-tokenizer",
        "revision": "5" * 40,
        "fingerprint": "6" * 64,
        "eos_token_id": 100257,
        "pad_token_id": 100277,
        "token_ids": token_ids,
        "token_ids_sha256": module.frozen._canonical_sha256(token_ids),
    }
    if legacy:
        payload.update(
            {
                "usage": "evaluation_only_exact_pin",
                "historical_training_revision_was_pinned": False,
            }
        )
    return payload


def _receipt(model: str) -> dict[str, Any]:
    tasks: dict[str, dict[str, Any]] = {}
    for task, metric in module._TASK_METRICS.items():
        rows = [_row(task, position, model) for position in range(4)]
        tasks[task] = {
            "source": {"task": task, "split": "validation"},
            "selection_count": len(rows),
            "selection_sha256": f"{len(tasks) + 20:064x}",
            "metric": metric,
            "examples": rows,
        }
    manifest = {
        "path": "/manifest.json",
        "bytes": 10,
        "sha256": "3" * 64,
        "content_sha256": "4" * 64,
        "partial": True,
        "panel_status": "confirmatory",
    }
    common = {
        "schema_version": 1,
        "created_at": "2026-08-16T00:00:00+00:00",
        "manifest": manifest,
        "tokenizer": _tokenizer(legacy=model == module.LEGACY_KEY),
        "protocol": _protocol(),
        "tasks": dict(sorted(tasks.items())),
        "content_sha256": {
            module.LEGACY_KEY: "7" * 64,
            module.STEP12_KEY: "8" * 64,
            module.STEP16_KEY: "9" * 64,
        }[model],
    }
    if model == module.LEGACY_KEY:
        manifest["builder_git"] = {"revision": "a" * 40, "dirty": False}
        return {
            **common,
            "format": module.EXPECTED_LEGACY_FORMAT,
            "protocol_name": module.EXPECTED_LEGACY_PROTOCOL,
            "launch_git": {"revision": "b" * 40, "dirty": False},
            "implementation": {
                "frozen_evaluator": {"files_sha256": "d" * 64},
                "wrapper_files_sha256": "c" * 64,
            },
            "checkpoint": {"checkpoint": "/checkpoints/step32000"},
            "legacy_stage1_lineage": {"maximum_steps": 32_000},
            "artifact_policy": {
                "descriptive_only": True,
                "promotion_eligible": False,
                "historical_reference_comparison_evidence": True,
            },
        }
    step = 12_000 if model == module.STEP12_KEY else 16_000
    return {
        **common,
        "format": module.EXPECTED_CURRENT_FORMAT,
        "protocol_name": module.EXPECTED_CURRENT_PROTOCOL,
        "git": {"revision": "a" * 40, "dirty": False},
        "implementation": {"files_sha256": "d" * 64},
        "checkpoint": {"checkpoint": f"/checkpoints/step{step}"},
        "prior_matched_wrong_v2": {"step": step},
        "artifact_policy": {
            "descriptive_only": True,
            "promotion_eligible": False,
            "checkpoint_selection_evidence": True,
        },
    }


def _write_inputs(tmp_path: Path, monkeypatch):
    receipts = {key: _receipt(key) for key in module.INPUT_KEYS}
    references = {}
    calls: list[tuple[str, bool, str | None]] = []
    by_path = {}
    for position, key in enumerate(module.INPUT_KEYS):
        path = tmp_path / f"input-{position}.json"
        path.write_text(f'{{"input":{position}}}\n', encoding="utf-8")
        resolved = str(path.resolve())
        by_path[resolved] = receipts[key]
        identity = module.frozen._file_identity(path, name="synthetic input")
        references[key] = {**identity, "content_sha256": receipts[key]["content_sha256"]}

    def validate_current(path, expected_sha256, *, verify_live):
        resolved = str(Path(path).resolve())
        assert (
            module.frozen._file_identity(Path(path), name="current input")["sha256"]
            == expected_sha256
        )
        calls.append((resolved, verify_live, None))
        return copy.deepcopy(by_path[resolved])

    def validate_legacy(path, expected_sha256, *, verify_live, hf_cache):
        resolved = str(Path(path).resolve())
        assert (
            module.frozen._file_identity(Path(path), name="legacy input")["sha256"]
            == expected_sha256
        )
        calls.append((resolved, verify_live, hf_cache))
        return copy.deepcopy(by_path[resolved])

    monkeypatch.setattr(
        module,
        "_CURRENT_EVALUATOR",
        SimpleNamespace(validate_external_academic_receipt=validate_current),
    )
    monkeypatch.setattr(
        module,
        "_LEGACY_EVALUATOR",
        SimpleNamespace(validate_legacy_stage1_receipt=validate_legacy),
    )
    return receipts, references, calls


def test_build_reuses_frozen_pairing_and_adds_legacy_deltas(tmp_path, monkeypatch):
    receipts, references, _ = _write_inputs(tmp_path, monkeypatch)
    comparison = module._build_comparison_receipt(
        receipts,
        references,
        bootstrap_seed=17,
        bootstrap_samples=50,
        created_at="2026-08-16T01:00:00+00:00",
    )
    repeated = module._build_comparison_receipt(
        receipts,
        references,
        bootstrap_seed=17,
        bootstrap_samples=50,
        created_at="2026-08-16T01:00:00+00:00",
    )
    assert comparison == repeated

    correct = comparison["tasks"]["vqav2"]["all_examples"]["correct_accuracy"]
    assert correct["legacy_stage1_step32000"]["mean"] == 0.25
    assert correct["step12000"]["mean"] == 0.5
    assert correct["step16000"]["mean"] == 0.5
    assert correct["paired_delta_12000_minus_legacy_stage1"]["mean"] == 0.25
    assert correct["paired_delta_16000_minus_legacy_stage1"]["mean"] == 0.25
    assert correct["paired_delta_16000_minus_12000"]["mean"] == 0.0
    assert correct["legacy_stage1_step32000"]["clusters"] == 3
    mcnemar = correct["mcnemar_exact_score_one"]["step12000_vs_legacy_stage1"]
    assert mcnemar["legacy_stage1_step32000_only"] == 0
    assert mcnemar["step12000_only"] == 1

    current_receipts = {
        12_000: receipts[module.STEP12_KEY],
        16_000: receipts[module.STEP16_KEY],
    }
    current_references = {
        12_000: references[module.STEP12_KEY],
        16_000: references[module.STEP16_KEY],
    }
    frozen_pair = module.frozen._build_comparison_receipt(
        current_receipts,
        current_references,
        bootstrap_seed=17,
        bootstrap_samples=50,
        created_at="2026-08-16T01:00:00+00:00",
    )
    frozen_correct = frozen_pair["tasks"]["vqav2"]["all_examples"]["correct_accuracy"]
    for field in ("step12000", "step16000", "paired_delta_16000_minus_12000"):
        assert correct[field] == frozen_correct[field]
    new_macro = comparison["equal_task_macro"]["all_examples"]["statistics"]
    frozen_macro = frozen_pair["equal_task_macro"]["all_examples"]["statistics"]
    for field in ("correct_step12000", "correct_step16000", "correct_delta_16000_minus_12000"):
        assert new_macro[field] == frozen_macro[field]

    limits = comparison["interpretation_limits"]
    assert "not a legacy Stage-1 training-contamination screen" in limits["overlap_inventory_scope"]
    assert comparison["policy"]["promotion_eligible"] is False
    assert comparison["policy"]["promotion_decision"] is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda receipts: receipts[module.LEGACY_KEY]["manifest"].__setitem__(
                "sha256", "f" * 64
            ),
            "manifest identity",
        ),
        (
            lambda receipts: receipts[module.LEGACY_KEY]["protocol"]["prompt"].__setitem__(
                "free_answer", "changed"
            ),
            "Prompt/control protocol",
        ),
        (
            lambda receipts: receipts[module.STEP12_KEY]["tasks"]["vqav2"]["examples"][0][
                "controls"
            ]["correct"].__setitem__("input_tokens", 33),
            "prompt/control input identity",
        ),
        (
            lambda receipts: receipts[module.STEP16_KEY]["tasks"]["vqav2"]["examples"][
                0
            ].__setitem__("question", "different question"),
            "identity or ordering",
        ),
        (
            lambda receipts: receipts[module.LEGACY_KEY]["implementation"].__setitem__(
                "frozen_evaluator", {"files_sha256": "e" * 64}
            ),
            "different frozen evaluators",
        ),
    ),
)
def test_cross_checkpoint_identity_tampering_is_rejected(tmp_path, monkeypatch, mutation, message):
    receipts, references, _ = _write_inputs(tmp_path, monkeypatch)
    mutation(receipts)
    with pytest.raises(ValueError, match=message):
        module._build_comparison_receipt(
            receipts,
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
    module.frozen._write_json_no_overwrite(output, comparison)
    raw_sha256 = module.frozen._file_identity(output, name="synthetic comparison")["sha256"]
    assert (
        module.validate_academic_legacy_three_way_comparison_receipt(
            output,
            raw_sha256,
            verify_live_inputs=True,
            hf_cache="/synthetic/hf-cache",
        )
        == comparison
    )
    assert [verify_live for _, verify_live, _ in calls] == [True, True, True]
    assert calls[0][2] == "/synthetic/hf-cache"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.frozen._write_json_no_overwrite(output, comparison)

    tampered = copy.deepcopy(comparison)
    tampered["tasks"]["vqav2"]["all_examples"]["correct_accuracy"][
        "paired_delta_12000_minus_legacy_stage1"
    ]["mean"] = 0.125
    del tampered["content_sha256"]
    tampered = module.frozen._attach_content_sha256(tampered)
    tampered_path = tmp_path / "tampered.json"
    module.frozen._write_json_no_overwrite(tampered_path, tampered)
    tampered_sha256 = module.frozen._file_identity(tampered_path, name="tampered comparison")[
        "sha256"
    ]
    with pytest.raises(ValueError, match="full rederivation"):
        module.validate_academic_legacy_three_way_comparison_receipt(
            tampered_path,
            tampered_sha256,
            verify_live_inputs=False,
            hf_cache="/synthetic/hf-cache",
        )
