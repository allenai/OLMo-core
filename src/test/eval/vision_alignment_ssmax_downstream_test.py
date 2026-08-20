"""Tests for the dense SSMax downstream identity and paired comparison contracts."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from olmo_core.eval.vision_alignment_ssmax_bridge import checkpoint_identity
from olmo_core.eval.vision_alignment_ssmax_downstream import (
    BLINK_DATASET_REVISION,
    BLINK_JIGSAW_EXAMPLES,
    LMMS_EVAL_REVISION,
    MATHVISTA_DATASET_REVISION,
    MATHVISTA_GEOMETRY_MC_EXAMPLES,
    SSMAX_DOWNSTREAM_TASKS,
    TRAJECTORY_BOOTSTRAP_SAMPLES,
    TRAJECTORY_PRIMARY_PHASE,
    TRAJECTORY_PRIMARY_STEP,
    SSMaxDownstreamContractError,
    compare_downstream_results,
    compare_downstream_trajectory,
    is_mathvista_geometry_mc,
    task_definition_inventory,
    verify_checkpoint_identity,
)


def _load_task_utils():
    path = (
        Path(__file__).resolve().parents[3]
        / "requirements/lmms-eval-overrides/vision_ssmax_downstream/utils.py"
    )
    return SimpleNamespace(**runpy.run_path(str(path)))


def _checkpoint(tmp_path: Path, *, ephemeral: bool = False) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "step123"
    (root / "model_and_optim").mkdir(parents=True)
    config = {
        "model_variant": "ssmax_head_qknorm",
        "phase": "joint",
        "vision_alignment": {
            "model_variant": "ssmax_head_qknorm",
            "phase": "joint",
        },
        "model": {},
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / ".metadata.json").write_text(
        json.dumps({"ephemeral": False, "version": "test", "global_step": 123})
    )
    (root / "model_and_optim" / ".metadata").write_bytes(b"dcp metadata")
    (root / "model_and_optim" / "__0_0.distcp").write_bytes(b"model state")
    (root / "train").mkdir()
    torch.save({"global_step": 123, "world_size": 1}, root / "train" / "rank0.pt")
    full_identity = checkpoint_identity(root)
    if ephemeral:
        (root / ".metadata.json").write_text(
            json.dumps({"ephemeral": True, "version": "test", "global_step": 123})
        )
    return root, {
        "expected_global_step": full_identity["global_step"],
        "expected_config_sha256": full_identity["config_sha256"],
        "expected_marker_sha256": full_identity["marker_sha256"],
        "expected_dcp_metadata_sha256": full_identity["dcp_metadata_sha256"],
        "expected_checkpoint_identity_sha256": full_identity["identity_sha256"],
    }


def test_checkpoint_identity_accepts_exact_permanent_checkpoint(tmp_path: Path) -> None:
    root, hashes = _checkpoint(tmp_path)
    identity, raw = verify_checkpoint_identity(
        root / "model_and_optim",
        expected_model_variant="ssmax_head_qknorm",
        expected_phase="joint",
        **hashes,
    )
    assert identity.checkpoint == str(root.resolve())
    assert identity.config_sha256 == hashes["expected_config_sha256"]
    assert identity.global_step == 123
    assert identity.state_file_count == 2
    assert identity.trainer_state_count == 1
    assert identity.identity_sha256 == hashes["expected_checkpoint_identity_sha256"]
    assert raw["vision_alignment"]["phase"] == "joint"


def test_checkpoint_identity_rejects_ephemeral_or_changed_files(tmp_path: Path) -> None:
    root, hashes = _checkpoint(tmp_path, ephemeral=True)
    with pytest.raises(SSMaxDownstreamContractError, match="permanent"):
        verify_checkpoint_identity(
            root,
            expected_model_variant="ssmax_head_qknorm",
            expected_phase="joint",
            **hashes,
        )

    root, hashes = _checkpoint(tmp_path / "other")
    (root / "config.json").write_text("{}")
    with pytest.raises(SSMaxDownstreamContractError, match="config_sha256 mismatch"):
        verify_checkpoint_identity(
            root,
            expected_model_variant="ssmax_head_qknorm",
            expected_phase="joint",
            **hashes,
        )

    root, hashes = _checkpoint(tmp_path / "shard-change")
    (root / "model_and_optim" / "__0_0.distcp").write_bytes(b"altered model state")
    with pytest.raises(SSMaxDownstreamContractError, match="identity_sha256 mismatch"):
        verify_checkpoint_identity(
            root,
            expected_model_variant="ssmax_head_qknorm",
            expected_phase="joint",
            **hashes,
        )

    root, hashes = _checkpoint(tmp_path / "trainer-change")
    torch.save(
        {"global_step": 123, "world_size": 1, "altered": True},
        root / "train" / "rank0.pt",
    )
    with pytest.raises(SSMaxDownstreamContractError, match="identity_sha256 mismatch"):
        verify_checkpoint_identity(
            root,
            expected_model_variant="ssmax_head_qknorm",
            expected_phase="joint",
            **hashes,
        )


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        (
            {
                "question_type": "multi_choice",
                "metadata": {"task": "geometry problem solving"},
            },
            True,
        ),
        (
            {"question_type": "free_form", "metadata": {"task": "geometry problem solving"}},
            False,
        ),
        (
            {"question_type": "multi_choice", "metadata": {"task": "figure question answering"}},
            False,
        ),
    ],
)
def test_mathvista_geometry_filter_is_exact(document: dict, expected: bool) -> None:
    assert is_mathvista_geometry_mc(document) is expected


def _protocol() -> dict:
    return {
        "tasks": list(SSMAX_DOWNSTREAM_TASKS),
        "partial": False,
        "limit": None,
        "lmms_eval_revision": LMMS_EVAL_REVISION,
        "blink_dataset_revision": BLINK_DATASET_REVISION,
        "mathvista_dataset_revision": MATHVISTA_DATASET_REVISION,
        "dataset_auth": None,
        "task_definition_sha256": "1" * 64,
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
        "checkpoint_load": {
            "strict_model_state": True,
            "load_optimizer_state": False,
            "load_trainer_state": False,
            "state_file_count": 2,
            "state_file_inventory_sha256": "5" * 64,
            "trainer_state_count": 1,
            "trainer_state_inventory_sha256": "6" * 64,
        },
        "mathvista_scoring": "local_valid_letter_choice_string_equal",
        "external_judge": None,
        "generation": "single_forward_valid_option_letter_logits",
    }


def _payload(
    variant: str,
    *,
    qknorm: bool,
    phase: str = "joint",
    global_step: int = 100,
    blink_correct: int | None = None,
    mathvista_correct: int | None = None,
) -> dict:
    if blink_correct is None:
        blink_correct = 90 if qknorm else 75
    if mathvista_correct is None:
        mathvista_correct = 80 if qknorm else 100
    blink = []
    for index in range(BLINK_JIGSAW_EXAMPLES):
        correct = index < blink_correct
        blink.append(
            {
                "blink_acc": {
                    "id": f"blink-{index}",
                    "gt_content": "A",
                    "pred_parsed": "A" if correct else "B",
                    "is_correct": correct,
                    "num_choices": 3,
                }
            }
        )
    mathvista = []
    for index in range(MATHVISTA_GEOMETRY_MC_EXAMPLES):
        correct = index < mathvista_correct
        mathvista.append(
            {
                "mathvista_geometry_mc_acc": {
                    "question_id": str(index),
                    "answer": "choice A",
                    "prediction": "choice A" if correct else "choice B",
                    "raw_response": "A" if correct else "B",
                    "true_false": correct,
                    "num_choices": 4,
                }
            }
        )
    return {
        "schema_version": 1,
        "checkpoint_identity": {
            "path": f"/checkpoints/{variant}/step{global_step}",
            "model_variant": variant,
            "phase": phase,
            "global_step": global_step,
            "config_sha256": "2" * 64,
            "marker_sha256": "3" * 64,
            "dcp_metadata_sha256": "4" * 64,
            "state_file_count": 2,
            "state_file_inventory_sha256": "5" * 64,
            "trainer_state_count": 1,
            "trainer_state_inventory_sha256": "6" * 64,
            "identity_sha256": "7" * 64,
        },
        "protocol": _protocol(),
        "lmms_eval": {
            "samples": {
                "ssmax_blink_jigsaw": blink,
                "ssmax_mathvista_geometry_mc": mathvista,
            }
        },
    }


def test_paired_comparison_reports_exact_task_outcomes_and_macro_rank() -> None:
    comparison = compare_downstream_results(
        _payload("ssmax_head_qknorm", qknorm=True),
        _payload("ssmax_no_qknorm", qknorm=False),
    )
    blink = comparison["tasks"]["ssmax_blink_jigsaw"]
    assert blink["samples"] == 150
    assert blink["left"]["correct"] == 90
    assert blink["right"]["correct"] == 75
    assert blink["paired_outcomes"]["left_only_correct"] == 15
    assert blink["chance_accuracy"] == pytest.approx(1 / 3)

    mathvista = comparison["tasks"]["ssmax_mathvista_geometry_mc"]
    assert mathvista["samples"] == 203
    assert mathvista["paired_outcomes"]["right_only_correct"] == 20
    assert mathvista["chance_accuracy"] == 0.25
    expected_left = ((90 / 150) + (80 / 203)) / 2
    expected_right = ((75 / 150) + (100 / 203)) / 2
    assert comparison["macro_accuracy"]["ssmax_head_qknorm"] == expected_left
    assert comparison["macro_accuracy"]["ssmax_no_qknorm"] == expected_right
    assert comparison["observed_point_ranking"] == "ssmax_head_qknorm"
    assert comparison["inference"]["conclusion"] == "inconclusive"
    assert comparison["inference"]["task_direction_consistent"] is False
    assert comparison["checkpoint_point"] == {"phase": "joint", "global_step": 100}


def test_paired_comparison_rejects_source_or_protocol_drift() -> None:
    left = _payload("ssmax_head_qknorm", qknorm=True)
    right = _payload("ssmax_no_qknorm", qknorm=False)
    right["lmms_eval"]["samples"]["ssmax_blink_jigsaw"][0]["blink_acc"]["num_choices"] = 4
    with pytest.raises(SSMaxDownstreamContractError, match="paired source row differs"):
        compare_downstream_results(left, right)

    right = _payload("ssmax_no_qknorm", qknorm=False)
    right["protocol"]["max_crops_total"] = 24
    with pytest.raises(SSMaxDownstreamContractError, match="max_crops_total"):
        compare_downstream_results(left, right)

    right = _payload("ssmax_no_qknorm", qknorm=False)
    right["checkpoint_identity"]["global_step"] = 200
    with pytest.raises(SSMaxDownstreamContractError, match="global_step"):
        compare_downstream_results(left, right)


def test_trajectory_did_is_step0_normalized_row_paired_and_deterministic() -> None:
    inputs = (
        _payload("ssmax_head_qknorm", qknorm=False, phase="bridge", global_step=0),
        _payload("ssmax_no_qknorm", qknorm=False, phase="bridge", global_step=0),
        _payload("ssmax_head_qknorm", qknorm=True, phase="perception", global_step=3000),
        _payload("ssmax_no_qknorm", qknorm=False, phase="perception", global_step=3000),
    )
    comparison = compare_downstream_trajectory(*inputs)
    blink = comparison["tasks"]["ssmax_blink_jigsaw"]
    assert blink["accuracy_gain_from_step0"]["ssmax_head_qknorm"] == pytest.approx(0.1)
    assert blink["accuracy_gain_from_step0"]["ssmax_no_qknorm"] == 0.0
    assert blink["gain_difference_qknorm_minus_no_qknorm"] == pytest.approx(0.1)
    mathvista = comparison["tasks"]["ssmax_mathvista_geometry_mc"]
    assert mathvista["gain_difference_qknorm_minus_no_qknorm"] == pytest.approx(-20 / 203)
    expected_macro = (0.1 - 20 / 203) / 2
    assert comparison["equal_task_macro"][
        "gain_difference_qknorm_minus_no_qknorm"
    ] == pytest.approx(expected_macro)
    assert comparison["inference"]["conclusion"] == "inconclusive"
    interval = comparison["equal_task_macro"]["gain_difference_bootstrap_ci"]
    assert interval["samples"] == TRAJECTORY_BOOTSTRAP_SAMPLES == 10_000
    repeated = compare_downstream_trajectory(*inputs)
    assert repeated["equal_task_macro"]["gain_difference_bootstrap_ci"] == interval


def test_trajectory_predeclared_directional_rule_requires_consistency_and_ci() -> None:
    baseline_qknorm = _payload(
        "ssmax_head_qknorm",
        qknorm=False,
        phase="bridge",
        global_step=0,
        blink_correct=0,
        mathvista_correct=0,
    )
    baseline_no_qknorm = _payload(
        "ssmax_no_qknorm",
        qknorm=False,
        phase="bridge",
        global_step=0,
        blink_correct=0,
        mathvista_correct=0,
    )
    candidate_qknorm = _payload(
        "ssmax_head_qknorm",
        qknorm=True,
        phase="bridge",
        global_step=500,
        blink_correct=BLINK_JIGSAW_EXAMPLES,
        mathvista_correct=MATHVISTA_GEOMETRY_MC_EXAMPLES,
    )
    candidate_no_qknorm = _payload(
        "ssmax_no_qknorm",
        qknorm=False,
        phase="bridge",
        global_step=500,
        blink_correct=0,
        mathvista_correct=0,
    )
    comparison = compare_downstream_trajectory(
        baseline_qknorm,
        baseline_no_qknorm,
        candidate_qknorm,
        candidate_no_qknorm,
    )
    assert comparison["equal_task_macro"]["gain_difference_bootstrap_ci"]["lower"] == 1.0
    assert comparison["equal_task_macro"]["gain_difference_bootstrap_ci"]["upper"] == 1.0
    assert comparison["inference"]["conclusion"] == "directional_signal_ssmax_head_qknorm"
    assert comparison["inference"]["criterion_satisfied"] is True


def test_trajectory_predeclared_primary_endpoint_can_conclude_practical_equivalence() -> None:
    baseline_qknorm = _payload("ssmax_head_qknorm", qknorm=False, phase="bridge", global_step=0)
    baseline_no_qknorm = _payload("ssmax_no_qknorm", qknorm=False, phase="bridge", global_step=0)
    candidate_qknorm = _payload(
        "ssmax_head_qknorm",
        qknorm=False,
        phase=TRAJECTORY_PRIMARY_PHASE,
        global_step=TRAJECTORY_PRIMARY_STEP,
    )
    candidate_no_qknorm = _payload(
        "ssmax_no_qknorm",
        qknorm=False,
        phase=TRAJECTORY_PRIMARY_PHASE,
        global_step=TRAJECTORY_PRIMARY_STEP,
    )

    comparison = compare_downstream_trajectory(
        baseline_qknorm,
        baseline_no_qknorm,
        candidate_qknorm,
        candidate_no_qknorm,
    )

    assert comparison["inference"]["conclusion"] == "practical_equivalence_fast_suite"
    assert comparison["inference"]["primary_endpoint"]["eligible"] is True
    assert comparison["inference"]["criterion_satisfied"] is True


def test_trajectory_rejects_non_step0_baseline_and_cross_time_source_drift() -> None:
    baseline_qknorm = _payload("ssmax_head_qknorm", qknorm=False, phase="bridge", global_step=1)
    baseline_no_qknorm = _payload("ssmax_no_qknorm", qknorm=False, phase="bridge", global_step=1)
    candidate_qknorm = _payload("ssmax_head_qknorm", qknorm=True, phase="bridge", global_step=500)
    candidate_no_qknorm = _payload("ssmax_no_qknorm", qknorm=False, phase="bridge", global_step=500)
    with pytest.raises(SSMaxDownstreamContractError, match="bridge step0"):
        compare_downstream_trajectory(
            baseline_qknorm,
            baseline_no_qknorm,
            candidate_qknorm,
            candidate_no_qknorm,
        )

    baseline_qknorm = _payload("ssmax_head_qknorm", qknorm=False, phase="bridge", global_step=0)
    baseline_no_qknorm = _payload("ssmax_no_qknorm", qknorm=False, phase="bridge", global_step=0)
    for payload in (candidate_qknorm, candidate_no_qknorm):
        payload["lmms_eval"]["samples"]["ssmax_blink_jigsaw"][0]["blink_acc"][
            "gt_content"
        ] = "changed-target"
    with pytest.raises(SSMaxDownstreamContractError, match="trajectory source row differs"):
        compare_downstream_trajectory(
            baseline_qknorm,
            baseline_no_qknorm,
            candidate_qknorm,
            candidate_no_qknorm,
        )


def test_checked_in_task_inventory_is_complete_and_revision_pinned() -> None:
    root = (
        Path(__file__).resolve().parents[3]
        / "requirements/lmms-eval-overrides/vision_ssmax_downstream"
    )
    inventory = task_definition_inventory(root)
    assert len(inventory["files"]) == 3
    assert len(inventory["sha256"]) == 64
    assert BLINK_DATASET_REVISION in (root / "blink_jigsaw.yaml").read_text()
    mathvista_yaml = (root / "mathvista_geometry_mc.yaml").read_text()
    utils_source = (root / "utils.py").read_text()
    assert MATHVISTA_DATASET_REVISION in mathvista_yaml
    assert "token: true" not in mathvista_yaml
    assert "llm_as_judge_eval" not in mathvista_yaml
    assert "extract_answer" not in utils_source
    assert "get_chat_response" not in utils_source
    assert "from lmms_eval" not in utils_source
    assert "\nimport lmms_eval" not in utils_source


def test_local_mathvista_prompt_and_scorer_need_no_judge_dependency() -> None:
    module = _load_task_utils()

    document = {
        "pid": "geometry-unit",
        "question": "Which point is left?",
        "query": "saved query",
        "choices": ["P", "Q", "R"],
        "answer": "P",
        "question_type": "multi_choice",
        "answer_type": "text",
        "metadata": {"task": "geometry problem solving"},
    }
    prompt = module.mathvista_doc_to_text(
        document,
        {"shot_type": "solution", "shot": 0, "use_caption": False, "use_ocr": False},
    )
    assert prompt == (
        "Question: Which point is left?\n"
        "Choices:\n"
        "(A) P\n"
        "(B) Q\n"
        "(C) R\n"
        "Hint: Please answer the question and provide the correct option letter, "
        "e.g., A, B, C, D, at the end.\n"
        "Solution:"
    )
    result = module.mathvista_process_results(document, ["A"])
    metric = result["mathvista_geometry_mc_acc"]
    assert metric["prediction"] == "P"
    assert metric["true_false"] is True
    assert module.mathvista_geometry_mc_aggregate_results([metric]) == 1.0


def test_blink_images_are_numeric_contiguous_and_trailing_null_safe() -> None:
    module = _load_task_utils()

    class Image:
        def __init__(self, name: str) -> None:
            self.name = name

        def convert(self, mode: str):
            assert mode == "RGB"
            return self.name

    # Deliberately reverse mapping insertion order: semantic order is the numeric suffix.
    document = {
        "image_4": None,
        "image_3": Image("third"),
        "image_2": Image("second"),
        "image_1": Image("first"),
    }
    assert module.blink_doc_to_visual(document) == ["first", "second", "third"]

    with pytest.raises(ValueError, match="contiguous"):
        module.blink_doc_to_visual({"image_1": Image("first"), "image_3": Image("third")})
    with pytest.raises(ValueError, match="after an empty"):
        module.blink_doc_to_visual(
            {"image_1": Image("first"), "image_2": None, "image_3": Image("third")}
        )
