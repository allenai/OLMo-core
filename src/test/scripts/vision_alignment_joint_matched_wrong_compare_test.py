from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "eval"
        / "vision_alignment_joint_matched_wrong_compare.py"
    )
    name = "_vision_alignment_joint_matched_wrong_compare_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _visual_rows(step: int, source_index: int) -> list[dict[str, Any]]:
    correct_shift = -1.0 if step == 8000 else 0.0
    gap = 2.0 if step == 8000 else 1.0
    rows = []
    for position, weight in enumerate((1.0, 3.0)):
        correct = 2.0 + position + source_index / 10 + correct_shift
        wrong = correct + gap
        rows.append(
            {
                "pairing_position": position,
                "recipient_index": position,
                "donor_index": 1 - position,
                "recipient_content_id": f"{source_index * 4 + position:064x}",
                "donor_content_id": f"{source_index * 4 + 1 - position:064x}",
                "response_tokens": 10 + position,
                "loss_weight": weight,
                "correct_ce": correct,
                "wrong_ce": wrong,
                "wrong_gap": gap,
            }
        )
    return rows


def _blank_rows(visual_rows: list[dict[str, Any]], step: int) -> list[dict[str, Any]]:
    blank_gap = 0.75 if step == 8000 else 0.5
    return [
        {
            "pairing_position": row["pairing_position"],
            "recipient_index": row["recipient_index"],
            "recipient_content_id": row["recipient_content_id"],
            "response_tokens": row["response_tokens"],
            "loss_weight": row["loss_weight"],
            "correct_ce": row["correct_ce"],
            "blank_ce": float(row["correct_ce"]) + blank_gap,
            "blank_gap": blank_gap,
        }
        for row in visual_rows
    ]


def _packet(step: int) -> dict[str, Any]:
    visual_rows = {
        source: _visual_rows(step, source_index)
        for source_index, source in enumerate(module.SOURCE_NAMES)
    }
    native_ce = 2.5 if step == 8000 else 3.0
    sha = f"{step:064x}"
    return {
        "input": {"path": f"/receipts/step{step}.json", "sha256": sha},
        "receipt": {
            "producer": {"name": "evaluator"},
            "git": {"ref": "abc", "dirty": False},
            "artifact_policy": {"descriptive_only": True, "promotion_eligible": False},
            "projection": {"sha256": "1" * 64},
            "source_audit": {"fingerprint": "2" * 64},
            "tokenizer": {"sha256": "3" * 64},
            "protocol": {"name": module.EVALUATOR_PROTOCOL_NAME},
            "pairing_manifest": {"sha256": "4" * 64},
        },
        "checkpoint": {
            "root": f"/checkpoints/step{step}",
            "identity_sha256": f"{step + 1:064x}",
            "trainer_state_summary": {"wandb_run_id": "one-run"},
        },
        "checkpoint_config": {
            "path": f"/checkpoints/step{step}/config.json",
            "sha256": module.EXPECTED_CONFIG_SHA256,
            "phase": "joint",
            "lineage_id": "vision-alignment-joint-v1",
            "run_name": "vision-alignment-joint-v1",
            "step": step,
            "reviewed_profile_path": module.EXPECTED_REVIEWED_PROFILE,
            "reviewed_profile_sha256": module.EXPECTED_REVIEWED_PROFILE_SHA256,
            "reviewed_profile_allowlist_path": "approved.json",
            "reviewed_profile_allowlist_sha256": "5" * 64,
            "training_git_ref": module.EXPECTED_TRAINING_GIT_REF,
            "training_beaker_image": module.EXPECTED_TRAINING_BEAKER_IMAGE,
        },
        "load_coverage": {"shared": True},
        "protocol": {"name": module.EVALUATOR_PROTOCOL_NAME},
        "manifest_ref": {
            "path": "/pairings/manifest.json",
            "sha256": "4" * 64,
            "content_sha256": "6" * 64,
        },
        "manifest": {"shared": True},
        "pairings": {
            source: {
                "rows": [
                    {
                        "index": position,
                        "content_id": visual_rows[source][position]["recipient_content_id"],
                    }
                    for position in range(2)
                ],
                "pairs": [
                    {"recipient": 0, "donor": 1},
                    {"recipient": 1, "donor": 0},
                ],
            }
            for source in module.SOURCE_NAMES
        },
        "visual_rows": visual_rows,
        "blank_rows": {
            source: _blank_rows(visual_rows[source], step) for source in module.BLANK_SOURCE_NAMES
        },
        "native_rows": [
            {
                "evaluation_position": position,
                "dataset_index": position,
                "provenance": {"manifest_index": position},
                "mask_tokens": 8,
                "labeled_tokens": (0 if position in module.NATIVE_FILTERED_INDICES else 8),
                "mask_loss_weight": float(position % 3 + 1),
                "labeled_loss_weight": (
                    0.0 if position in module.NATIVE_FILTERED_INDICES else float(position % 3 + 1)
                ),
                "summed_ce": (
                    0.0
                    if position in module.NATIVE_FILTERED_INDICES
                    else native_ce * float(position % 3 + 1)
                ),
                "filtered": position in module.NATIVE_FILTERED_INDICES,
                "ce": None if position in module.NATIVE_FILTERED_INDICES else native_ce,
            }
            for position in range(module.NATIVE_EXAMPLES)
        ],
        "native_order_sha256": module._canonical_sha256(list(range(module.NATIVE_EXAMPLES))),
        "native_provenance_sha256": module._canonical_sha256(
            [{"manifest_index": position} for position in range(module.NATIVE_EXAMPLES)]
        ),
        "native_identity_sha256": "7" * 64,
    }


def _evaluations() -> dict[int, dict[str, Any]]:
    return {4000: _packet(4000), 8000: _packet(8000)}


def test_build_is_deterministic_paired_equal_source_and_descriptive(monkeypatch):
    monkeypatch.setattr(module, "DEFAULT_BOOTSTRAP_SAMPLES", 40)
    evaluations = _evaluations()
    first = module._build_comparison_receipt(
        evaluations,
        bootstrap_seed=module.DEFAULT_BOOTSTRAP_SEED,
        bootstrap_samples=40,
        created_at="2026-08-15T00:00:00+00:00",
    )
    second = module._build_comparison_receipt(
        evaluations,
        bootstrap_seed=module.DEFAULT_BOOTSTRAP_SEED,
        bootstrap_samples=40,
        created_at="2026-08-15T00:00:00+00:00",
    )
    assert first == second
    assert first["visual"]["equal_source_macro"]["paired_changes"]["gap_change_8000_minus_4000"][
        "mean"
    ] == pytest.approx(1.0)
    assert first["blank"]["equal_source_macro"]["paired_changes"][
        "blank_gap_change_8000_minus_4000"
    ]["mean"] == pytest.approx(0.25)
    assert first["native"]["paired_inline_compatible_ce_reduction_4000_minus_8000"][
        "mean"
    ] == pytest.approx(0.5)
    assert first["native"]["step4000"]["examples"] == 1000
    assert first["policy"]["conclusion"] == "descriptive_only"
    assert first["policy"]["promotion_eligible"] is False
    assert first["policy"]["promotion_decision"] is None
    module.validate_comparison_receipt(first, verify_inputs=False)


def test_cross_checkpoint_pair_drift_is_rejected(monkeypatch):
    monkeypatch.setattr(module, "DEFAULT_BOOTSTRAP_SAMPLES", 10)
    evaluations = _evaluations()
    evaluations[8000]["visual_rows"]["audited_alignment"][0]["donor_index"] = 999
    with pytest.raises(ValueError, match="identity differs"):
        module._build_components(
            evaluations,
            bootstrap_seed=module.DEFAULT_BOOTSTRAP_SEED,
            bootstrap_samples=10,
        )


def test_blank_correct_ce_must_be_the_exact_visual_score():
    visual = _visual_rows(4000, 0)
    raw = {
        "pairing_sha256": "a" * 64,
        "examples": 2,
        "elapsed_seconds": 1.0,
        "population": "matched_eligible_joint_validation_subset",
        "coverage": {"examples": 2},
        "metrics": {},
        "per_example": [
            {
                "pairing_position": position,
                "recipient_index": row["recipient_index"],
                "response_tokens": row["response_tokens"],
                "loss_weight": row["loss_weight"],
                "correct_ce": float(row["correct_ce"]) + (0.1 if position == 0 else 0.0),
                "blank_ce": float(row["correct_ce"]) + 0.5,
                "ce_gap_blank_minus_correct": 0.4 if position == 0 else 0.5,
            }
            for position, row in enumerate(visual)
        ],
    }
    with pytest.raises(ValueError, match="different correct CE"):
        module._validate_blank_rows(
            raw,
            source="pixmo_caption",
            visual_rows=visual,
            pairing_sha256="a" * 64,
            pairing_coverage={"examples": 2},
            name="blank",
        )


def test_native_requires_exact_complete_manifest_order():
    rows = [
        {
            "evaluation_position": position,
            "dataset_index": position,
            "provenance": {"manifest_index": position},
            "mask_tokens": 1,
            "labeled_tokens": 0 if position in module.NATIVE_FILTERED_INDICES else 1,
            "mask_loss_weight": 1.0,
            "labeled_loss_weight": (0.0 if position in module.NATIVE_FILTERED_INDICES else 1.0),
            "summed_ce": 0.0 if position in module.NATIVE_FILTERED_INDICES else 2.0,
            "filtered": position in module.NATIVE_FILTERED_INDICES,
            "ce": None if position in module.NATIVE_FILTERED_INDICES else 2.0,
        }
        for position in range(module.NATIVE_EXAMPLES)
    ]
    rows[7]["dataset_index"] = 8
    raw = {
        "examples": module.NATIVE_EXAMPLES,
        "elapsed_seconds": 1.0,
        "dataset_order_sha256": module._canonical_sha256(list(range(module.NATIVE_EXAMPLES))),
        "row_provenance_sha256": module._canonical_sha256(
            [{"manifest_index": position} for position in range(module.NATIVE_EXAMPLES)]
        ),
        "native_identity_sha256": "7" * 64,
        "metrics": {},
        "per_example": rows,
    }
    with pytest.raises(ValueError, match="manifest order"):
        module._validate_native_rows(raw, name="native")


def test_bootstrap_reuses_indices_across_metrics():
    values = np.asarray([-2.0, -1.0, 1.0, 4.0])
    intervals = module._paired_bootstrap(
        {"one": values, "two": values * 2},
        weights={"one": None, "two": None},
        seed=17,
        samples=200,
    )
    assert intervals["two"]["low"] == pytest.approx(intervals["one"]["low"] * 2)
    assert intervals["two"]["high"] == pytest.approx(intervals["one"]["high"] * 2)


def test_final_tokenizer_and_protocol_schema_are_exact(monkeypatch):
    token_ids = {
        "im_start_id": 100278,
        "im_end_id": 100279,
        "im_patch_id": 100280,
        "im_col_id": 100281,
        "low_res_im_start_id": 100282,
        "image_placeholder_id": 100283,
        "im_end_turn_id": 100265,
        "_CLASS_": "olmo_core.nn.vision.molmo2_tokens.Molmo2TokenIds",
    }
    tokenizer = {
        "id": "allenai/dolma2-tokenizer",
        "revision": "5292e5d6c0f40b67cc765fe41bec991cf4345b5c",
        "fingerprint": "8fec2af8c372f4c72a1a665ad8e70517625f94f041dbfcb7db4932071380f9a7",
        "token_ids": token_ids,
        "token_ids_sha256": module._canonical_sha256(token_ids),
    }
    module._validate_tokenizer(tokenizer, name="tokenizer")
    with pytest.raises(ValueError, match="reviewed joint tokenizer"):
        module._validate_tokenizer(
            {
                **tokenizer,
                "token_ids": {key: value for key, value in token_ids.items() if key != "_CLASS_"},
            },
            name="tokenizer",
        )

    native = {
        "holdout_fingerprint": module.EXPECTED_NATIVE_HOLDOUT_FINGERPRINT,
        "row_provenance_sha256": "8" * 64,
    }
    monkeypatch.setattr(module, "_validate_native_identity", lambda value, *, name: native)
    protocol = {
        "name": module.EVALUATOR_PROTOCOL_NAME,
        "descriptive_only": True,
        "promotion_eligible": False,
        "primary_statistic": (
            "paired source-balanced change in wrong-minus-correct CE from step4000 to step8000"
        ),
        "per_checkpoint_statistic": "all-response loss-weighted scalar CE",
        "response_logits_materialized": False,
        "sources": list(module.SOURCE_NAMES),
        "blank_sources": list(module.BLANK_SOURCE_NAMES),
        "native_source": "native_text_replay",
        "visual_split": "validation",
        "visual_population": "matched_eligible_joint_validation_subset",
        "examples_per_visual_source": 504,
        "native_population": "all holdout windows in exact manifest order",
        "native_examples": module.NATIVE_EXAMPLES,
        "native_filtered_indices": list(module.NATIVE_FILTERED_INDICES),
        "pairing_seed": module.PAIRING_SEED,
        "pairing_sha256": {source: "9" * 64 for source in module.SOURCE_NAMES},
        "pairing_rule": (
            "largest common multiple-of-eight; distinct pinned image content and exact "
            "collated geometry; deterministic explicit unique donors"
        ),
        "recipient_replay": "correct, wrong, and applicable blank forwards share recipients",
        "blank_rule": "zeros_like normalized image tensor; all non-image fields unchanged",
        "ce_definition": (
            "scalar summed CE divided by the one rank-local example's positive labeled loss "
            "weight; no response logits"
        ),
        "native_dual_denominator": (
            "inline CE uses labeled loss weight; training-divisor CE uses all mask loss weight"
        ),
        "sequence_length": 8192,
        "rank_batch_instances": 1,
        "global_batch_instances": 8,
        "nodes": 1,
        "world_size": 8,
        "local_world_size": 8,
        "ep_degree": 8,
        "dp_process_group_size": 8,
        "training_beaker_image": module.EXPECTED_TRAINING_BEAKER_IMAGE,
        "training_git_ref": module.EXPECTED_TRAINING_GIT_REF,
        "checkpoint_config_sha256": module.EXPECTED_CONFIG_SHA256,
        "projection_raw_sha256": module.EXPECTED_PROJECTION_SHA256,
        "source_audit_fingerprint": module.EXPECTED_SOURCE_AUDIT_FINGERPRINT,
        "native_holdout_fingerprint": module.EXPECTED_NATIVE_HOLDOUT_FINGERPRINT,
        "native_row_provenance_sha256": "8" * 64,
        "native_identity": {},
    }
    module._validate_protocol(protocol, name="protocol")
    with pytest.raises(ValueError, match="locked all-response protocol"):
        module._validate_protocol(
            {**protocol, "per_checkpoint_statistic": "changed"}, name="protocol"
        )


def test_projection_and_source_audit_registry_domains_are_distinct_and_exact():
    visual_registry = "ec6d511c5e3797be558cb10aaff680d1e3831078e4402eac109381a442eeea82"
    runtime_registry = "2833734cc14ec38398c35dddba10315e076c25bcfa6b0e90f62c84cd53bfebdc"
    projection = {
        "visual_source_registry_sha256": visual_registry,
        "runtime_registry_sha256": runtime_registry,
    }
    source_audit = {
        "source_registry_sha256": runtime_registry,
        "runtime_registry_sha256": runtime_registry,
    }

    assert visual_registry != runtime_registry
    module._validate_registry_domains(projection, source_audit, name="registries")

    for owner, field in (
        ("projection", "runtime_registry_sha256"),
        ("source_audit", "source_registry_sha256"),
        ("source_audit", "runtime_registry_sha256"),
    ):
        changed_projection = dict(projection)
        changed_source_audit = dict(source_audit)
        target = changed_projection if owner == "projection" else changed_source_audit
        target[field] = "f" * 64
        with pytest.raises(ValueError, match="registries differ"):
            module._validate_registry_domains(
                changed_projection, changed_source_audit, name="registries"
            )


def test_producer_exactly_binds_live_comparator():
    comparator = Path(module.__file__)
    evaluator = comparator.with_name("vision_alignment_joint_matched_wrong.py")
    perception = comparator.with_name("vision_alignment_perception_matched_wrong.py")
    bridge = comparator.with_name("vision_alignment_matched_wrong.py")
    pairing_source = module.inspect.getsourcefile(module.validate_matched_wrong_image_pairing)
    assert pairing_source is not None
    training = comparator.resolve().parents[1] / "train" / "Vision-Alignment.py"
    live = {
        "": evaluator,
        "comparator_": comparator,
        "perception_helper_": perception,
        "bridge_helper_": bridge,
        "pairing_implementation_": Path(pairing_source),
        "training_contract_": training,
    }
    producer = {
        key: value
        for prefix, path in live.items()
        for key, value in (
            (f"{prefix}path", str(path)),
            (f"{prefix}sha256", module._sha256_file(path)),
        )
    }
    module._validate_producer(producer, name="producer")
    with pytest.raises(ValueError, match="live reviewed implementation"):
        module._validate_producer({**producer, "comparator_sha256": "0" * 64}, name="producer")


def test_strict_validator_rederives_and_rejects_aggregate_tamper(monkeypatch):
    monkeypatch.setattr(module, "DEFAULT_BOOTSTRAP_SAMPLES", 20)
    evaluations = _evaluations()
    receipt = module._build_comparison_receipt(
        evaluations,
        bootstrap_seed=module.DEFAULT_BOOTSTRAP_SEED,
        bootstrap_samples=20,
        created_at="2026-08-15T00:00:00+00:00",
    )
    tampered = copy.deepcopy(receipt)
    tampered["visual"]["equal_source_macro"]["paired_changes"]["gap_change_8000_minus_4000"][
        "mean"
    ] = 999.0
    tampered["content_sha256"] = module._canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    monkeypatch.setattr(
        module,
        "_load_evaluator_receipt",
        lambda _path, *, expected_sha256, step, verify_live_checkpoint=True: evaluations[step],
    )
    with pytest.raises(ValueError, match="full input rederivation"):
        module.validate_comparison_receipt(tampered)


def test_atomic_writer_refuses_to_overwrite(tmp_path):
    output = tmp_path / "comparison.json"
    module._write_json_atomic(output, {"first": True})
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module._write_json_atomic(output, {"second": True})
    assert json.loads(output.read_text()) == {"first": True}


def test_atomic_writer_rejects_symlinked_parent(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked component"):
        module._write_json_atomic(linked / "comparison.json", {"invalid": True})
    assert not (real / "comparison.json").exists()


@pytest.mark.parametrize("symlink", [False, True])
def test_atomic_writer_preserves_preexisting_unowned_temp(tmp_path, symlink):
    output = tmp_path / "comparison.json"
    temporary = tmp_path / f".{output.name}.{os.getpid()}.tmp"
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("do-not-delete")
    if symlink:
        temporary.symlink_to(sentinel)
    else:
        temporary.write_text("owned-by-someone-else")
    with pytest.raises(FileExistsError):
        module._write_json_atomic(output, {"invalid": True})
    assert os.path.lexists(temporary)
    assert sentinel.read_text() == "do-not-delete"
    if not symlink:
        assert temporary.read_text() == "owned-by-someone-else"
