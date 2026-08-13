"""Focused contracts for the perception paired-outcome comparator."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).parents[2] / "scripts" / "eval" / "vision_alignment_perception_outcome.py"
    spec = importlib.util.spec_from_file_location("perception_outcome_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(module, *, arm: str, step: int, position: int):
    control = arm == "frozen_vision_control"
    correct = 2.0 - 0.00002 * step - (0.0 if control else 0.04) + position * 0.001
    gap = 0.25 + 0.00001 * step + (0.0 if control else 0.08) + position * 0.0005
    wrong = correct + gap
    return {
        "pairing_position": position,
        "recipient_index": position,
        "donor_index": position + 100,
        "response_tokens": 4 + position,
        "correct_ce": {window: correct for window in module.WINDOWS},
        "wrong_ce": {window: wrong for window in module.WINDOWS},
        "ce_gap_wrong_minus_correct": {window: gap for window in module.WINDOWS},
    }


def _evaluations(module, examples: int = 4):
    protocol = {
        "pairing_seed": 6198,
        "pairing_sha256": {source: source for source in module.PERCEPTION_SOURCE_NAMES},
        "perception_provenance_sha256": "p" * 64,
        "source_audit_fingerprint": "a" * 64,
        "sha256": "e" * 64,
    }
    validation = {"identity": "shared"}
    evaluator = {"identity": "shared"}
    pairings = {source: {"sha256": source} for source in module.PERCEPTION_SOURCE_NAMES}
    pairing_payloads = {source: {"source": source} for source in module.PERCEPTION_SOURCE_NAMES}
    output = {}
    for arm in module.ARMS:
        output[arm] = {}
        frozen = 806 if arm == "frozen_vision_control" else 403
        native = {
            "model_parameter_count": 818,
            "model_parameter_checkpoint_key_count": 818,
            "eval_state_key_count": 818 - frozen,
            "frozen_state_key_count": frozen,
            "persistent_buffer_count": 0,
            "prepared_load_key_count": 818,
            "unused_model_bearing_key_count": 0,
        }
        for step in module.STEPS:
            output[arm][step] = {
                "checkpoint": {
                    "root": f"/{arm}/step{step}",
                    "config_sha256": arm,
                    "identity_sha256": f"{arm}-{step}",
                },
                "protocol": protocol,
                "validation": validation,
                "evaluator": evaluator,
                "evaluator_semantic": evaluator,
                "git": {"identity": "shared"},
                "pairings": pairings,
                "pairing_payloads": pairing_payloads,
                "config": {"arm": arm},
                "normalized_config": b"shared",
                "native_checkpoint_load": native,
                "examples": examples,
                "rows": {
                    source: [
                        _row(module, arm=arm, step=step, position=position)
                        for position in range(examples)
                    ]
                    for source in module.PERCEPTION_SOURCE_NAMES
                },
            }
    return output


def test_components_join_four_receipts_and_compute_positive_did_deterministically():
    module = _load_module()
    evaluations = _evaluations(module)
    kwargs = {
        "profile_pair_ref": {"path": "/pair.json", "sha256": "f" * 64},
        "bootstrap_seed": 17,
        "bootstrap_samples": 200,
    }
    first = module._build_components(evaluations, **kwargs)
    second = module._build_components(evaluations, **kwargs)

    assert first == second
    assert set(first["checkpoints"]) == set(module.ARMS)
    assert first["protocol"]["examples_per_source"] == 4
    assert first["protocol"]["primary_step"] == 4000
    assert first["protocol"]["durability_step"] == 3000
    endpoint = first["summary"]["steps"]["step4000"]["windows"]["all"]
    assert endpoint["did"]["mean"] == pytest.approx(0.08)
    assert endpoint["did"]["ci"]["low"] > 0
    assert endpoint["treatment"]["correct_ce"] < endpoint["control"]["correct_ce"]
    assert endpoint["source_wins"] == {
        "did_positive": 8,
        "treatment_correct_ce_lower": 8,
        "source_count": 8,
    }
    rows = first["sources"][module.PERCEPTION_SOURCE_NAMES[0]]["steps"]["step4000"]["per_example"]
    assert len(rows) == 4
    assert rows[0]["effects"]["gap_improvement_did"]["all"] == pytest.approx(0.08)


def test_join_rejects_any_control_treatment_pairing_drift():
    module = _load_module()
    control = [_row(module, arm="frozen_vision_control", step=4000, position=0)]
    treatment = [_row(module, arm="treatment", step=4000, position=0)]
    treatment[0]["donor_index"] += 1
    with pytest.raises(ValueError, match="pairing or response identity differs"):
        module._join_source_step(
            control,
            treatment,
            source_index=0,
            step_index=1,
            bootstrap_seed=1,
            bootstrap_samples=10,
        )


def test_evaluator_rows_must_equal_the_pinned_pairing_payload():
    module = _load_module()
    rows = [_row(module, arm="treatment", step=4000, position=0)]
    pairing = {"pairs": [{"recipient": 0, "donor": 101}]}
    with pytest.raises(ValueError, match="differ from the pinned pairing"):
        module._validate_rows_against_pairing(rows, pairing, source="test-source")


def test_components_reject_cross_step_response_identity_drift():
    module = _load_module()
    evaluations = _evaluations(module)
    source = module.PERCEPTION_SOURCE_NAMES[0]
    evaluations["treatment"][3000]["rows"][source][0]["response_tokens"] += 1
    with pytest.raises(ValueError, match="Step3000/step4000 pairing or response-token"):
        module._build_components(
            evaluations,
            profile_pair_ref={"path": "/pair.json", "sha256": "f" * 64},
            bootstrap_seed=17,
            bootstrap_samples=20,
        )


def test_components_reject_unreviewed_config_drift():
    module = _load_module()
    evaluations = _evaluations(module)
    evaluations["treatment"][4000]["normalized_config"] = b"unreviewed"
    with pytest.raises(ValueError, match="outside reviewed identity"):
        module._build_components(
            evaluations,
            profile_pair_ref={"path": "/pair.json", "sha256": "f" * 64},
            bootstrap_seed=17,
            bootstrap_samples=20,
        )


def test_completed_runtime_config_identity_rejects_common_unreviewed_drift():
    module = _load_module()
    root = Path(
        "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/checkpoints"
    )
    pair_path = Path(
        "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/artifacts/"
        "perception-profile-pair-v2.json"
    )
    if not pair_path.is_file():
        pytest.skip("Pinned perception artifacts are unavailable")
    pair, _ = module._load_json_bytes(
        pair_path,
        expected_sha256=module.EXPECTED_PROFILE_PAIR_RECEIPT_SHA256,
        name="published profile pair",
    )
    runs = {
        "frozen_vision_control": "vision-alignment-perception-frozen-vision-control-v1",
        "treatment": "vision-alignment-perception-treatment-v1",
    }
    for arm, run in runs.items():
        config_path = root / run / "step4000" / "config.json"
        config, _ = module._load_json_bytes(
            config_path,
            expected_sha256=module.EXPECTED_RUNTIME_CONFIG_SHA256[arm],
            name=f"{arm} completed config",
        )
        module._causal_config_identity(config, arm=arm, profile_pair=pair)
        drifted = copy.deepcopy(config)
        drifted["global_batch_size"] = 1
        drifted["init_seed"] = 42
        with pytest.raises(ValueError, match="realized runtime config"):
            module._causal_config_identity(drifted, arm=arm, profile_pair=pair)


def test_validator_rederives_stored_metrics_and_returns_promotion_policy(monkeypatch):
    module = _load_module()
    evaluations = _evaluations(module)
    profile_ref = {"path": "/pair.json", "sha256": "f" * 64}
    components = module._build_components(
        evaluations,
        profile_pair_ref=profile_ref,
        bootstrap_seed=module.DEFAULT_BOOTSTRAP_SEED,
        bootstrap_samples=module.DEFAULT_BOOTSTRAP_SAMPLES,
    )
    inputs = {
        "profile_pair_receipt": profile_ref,
        "evaluations": {
            arm: {
                f"step{step}": {"path": f"/{arm}-{step}.json", "sha256": "0" * 64}
                for step in module.STEPS
            }
            for arm in module.ARMS
        },
    }
    payload = {
        "format": module.FORMAT,
        "version": module.VERSION,
        "status": "passed",
        "created_at": "2026-08-13T00:00:00+00:00",
        "producer": {"path": str(Path(module.__file__)), "sha256": "1" * 64},
        "inputs": inputs,
        **components,
    }
    payload["content_sha256"] = module._canonical_sha256(payload)
    monkeypatch.setattr(
        module,
        "_load_inputs",
        lambda value, verify_live_inputs: (evaluations, profile_ref),
    )

    normalized = module.validate_outcome_receipt(payload, verify_live_inputs=False)
    assert normalized["checkpoints"] == components["checkpoints"]
    assert normalized["policy_metrics"]["macro"]["did_ci_low"] > 0
    assert set(normalized["policy_metrics"]["sources"]) == set(module.PERCEPTION_SOURCE_NAMES)

    payload["summary"]["steps"]["step4000"]["windows"]["all"]["did"]["mean"] = -1
    payload["content_sha256"] = module._canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="summary differs"):
        module.validate_outcome_receipt(payload, verify_live_inputs=False)


def test_build_rejects_caller_selected_bootstrap_policy(tmp_path):
    module = _load_module()
    with pytest.raises(ValueError, match="locked policy"):
        module.build_outcome_receipt(
            inputs={},
            output=tmp_path / "outcome.json",
            bootstrap_seed=module.DEFAULT_BOOTSTRAP_SEED + 1,
            bootstrap_samples=module.DEFAULT_BOOTSTRAP_SAMPLES,
        )


def test_outcome_version_rejects_json_boolean(monkeypatch):
    module = _load_module()
    receipt = {
        "format": module.FORMAT,
        "version": True,
        "status": "passed",
        "created_at": "2026-08-13T00:00:00+00:00",
        "producer": {},
        "inputs": {},
        "checkpoints": {},
        "protocol": {},
        "sources": {},
        "summary": {},
        "content_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="identity or status"):
        module.validate_outcome_receipt(receipt, verify_live_inputs=False)


def test_implementation_reference_uses_canonical_fallback(tmp_path):
    module = _load_module()
    canonical = tmp_path / "producer.py"
    canonical.write_text("# exact implementation\n")
    digest = module._sha256_file(canonical)
    result = module._implementation_reference(
        path="/expired/gantry/producer.py",
        sha256=digest,
        name="producer",
        canonical_path=canonical,
        verify_live=True,
    )
    assert result == {"basename": "producer.py", "sha256": digest}
