"""CLI tests for perception bundle audit and accountable human approval."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from olmo_core.eval import vision_alignment_perception_promotion as promotion


def _load_module():
    path = Path(__file__).resolve().parents[2] / (
        "scripts/eval/vision_alignment_perception_promotion.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vision_alignment_perception_promotion_test_module", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_approve_writes_exact_immutable_parent_gate_v3(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    checkpoint = tmp_path / "treatment" / "step4000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "vision_alignment": {
                    "recipe_version": 3,
                    "formatter_version": "vision-alignment-native-document-v1",
                }
            }
        )
        + "\n"
    )
    bundle = tmp_path / "perception-promotion-bundle.json"
    bundle.write_text(
        json.dumps({"created_at": "2026-08-13T00:00:00+00:00"}, sort_keys=True) + "\n"
    )
    bundle_sha = promotion.sha256_file(bundle)
    candidate = {
        "checkpoint": str(checkpoint),
        "global_step": 4000,
        "phase": "perception",
        "checkpoint_config_sha256": promotion.sha256_file(checkpoint / "config.json"),
        "checkpoint_identity_sha256": "b" * 64,
        "data_contract_sha256": "c" * 64,
        "trainable_contract_sha256": "d" * 64,
    }
    deviation_sha = {promotion.TREATMENT_GUARD_WAIVER_ID: "e" * 64}
    monkeypatch.setattr(
        module,
        "validate_perception_promotion_bundle",
        lambda value: {"candidate": candidate, "deviation_sha256": deviation_sha},
    )
    output = tmp_path / "parent-gate-v3.json"
    args = [
        "approve",
        f"--bundle={bundle}",
        f"--expected-sha256={bundle_sha}",
        "--approved-by=rustin@example.org",
        "--approved-at=2026-08-13T01:00:00+00:00",
        f"--approve-waiver={promotion.TREATMENT_GUARD_WAIVER_ID}",
        f"--output={output}",
    ]
    module.main(args)
    gate = promotion.load_json(output)
    assert set(gate) == {
        "format",
        "version",
        "status",
        "promotion_kind",
        "promotion_policy",
        "recipe_version",
        "formatter_version",
        "phase",
        "checkpoint",
        "checkpoint_config_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "global_step",
        "metrics_artifact_sha256",
        "promotion_bundle_path",
        "promotion_bundle_sha256",
        "checkpoint_identity_sha256",
        "approved_by",
        "approved_at",
        "waivers",
    }
    assert gate["format"] == "vision_alignment_parent_gate"
    assert gate["version"] == 3
    assert gate["promotion_kind"] == "perception"
    assert gate["promotion_policy"] == promotion.PERCEPTION_PROMOTION_POLICY
    assert gate["phase"] == "perception"
    assert gate["promotion_bundle_path"] == str(bundle.resolve())
    assert gate["promotion_bundle_sha256"] == bundle_sha
    assert gate["metrics_artifact_sha256"] == bundle_sha
    assert gate["waivers"] == [
        {
            "id": promotion.TREATMENT_GUARD_WAIVER_ID,
            "decision": "approved",
            "deviation_sha256": "e" * 64,
        }
    ]

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.main(args)

    missing_waiver = [item for item in args if "--approve-waiver" not in item]
    missing_waiver[-1] = f"--output={tmp_path / 'missing-waiver.json'}"
    with pytest.raises(promotion.PromotionValidationError, match="each locked deviation"):
        module.main(missing_waiver)


def test_approve_rejects_bundle_sha_mismatch(tmp_path: Path) -> None:
    module = _load_module()
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}\n")
    with pytest.raises(promotion.PromotionValidationError, match="raw SHA-256 differs"):
        module.main(
            [
                "approve",
                f"--bundle={bundle}",
                f"--expected-sha256={'0' * 64}",
                "--approved-by=rustin@example.org",
                "--approved-at=2026-08-13T01:00:00+00:00",
                f"--approve-waiver={promotion.TREATMENT_GUARD_WAIVER_ID}",
                f"--output={tmp_path / 'gate.json'}",
            ]
        )
