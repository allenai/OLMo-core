from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from olmo_core.eval import vision_alignment_promotion as promotion


def _load_module():
    path = (
        Path(__file__).resolve().parents[2] / "scripts" / "eval" / "vision_alignment_promotion.py"
    )
    spec = importlib.util.spec_from_file_location("vision_alignment_promotion_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_approve_writes_exact_immutable_v2_gate(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    checkpoint = tmp_path / "bridge" / "step500"
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
    bundle = tmp_path / "promotion-bundle.json"
    bundle.write_text(
        json.dumps({"created_at": "2026-08-12T00:00:00+00:00"}, sort_keys=True) + "\n"
    )
    bundle_sha = promotion.sha256_file(bundle)
    candidate = {
        "checkpoint": str(checkpoint),
        "global_step": 500,
        "phase": "bridge",
        "checkpoint_config_sha256": "a" * 64,
        "checkpoint_identity_sha256": "b" * 64,
        "data_contract_sha256": "c" * 64,
        "trainable_contract_sha256": "d" * 64,
    }
    deviation_sha = {
        promotion.STEP250_WAIVER_ID: "e" * 64,
        promotion.STEP356_WAIVER_ID: "f" * 64,
    }
    monkeypatch.setattr(
        module,
        "validate_promotion_bundle",
        lambda value: {"candidate": candidate, "deviation_sha256": deviation_sha},
    )
    output = tmp_path / "parent-gate-v2.json"
    args = [
        "approve",
        f"--bundle={bundle}",
        f"--expected-sha256={bundle_sha}",
        "--approved-by=rustin@example.org",
        "--approved-at=2026-08-12T01:00:00+00:00",
        f"--approve-waiver={promotion.STEP250_WAIVER_ID}",
        f"--approve-waiver={promotion.STEP356_WAIVER_ID}",
        f"--output={output}",
    ]
    module.main(args)
    gate = promotion.load_json(output)
    assert gate["version"] == 2
    assert gate["promotion_bundle_path"] == str(bundle.resolve())
    assert gate["promotion_bundle_sha256"] == bundle_sha
    assert gate["metrics_artifact_sha256"] == bundle_sha
    assert gate["checkpoint_identity_sha256"] == candidate["checkpoint_identity_sha256"]
    assert gate["approved_by"] == "rustin@example.org"
    assert gate["waivers"] == [
        {
            "id": waiver_id,
            "decision": "approved",
            "deviation_sha256": deviation_sha[waiver_id],
        }
        for waiver_id in sorted(promotion.REQUIRED_WAIVER_IDS)
    ]

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.main(args)

    missing_waiver = [item for item in args if promotion.STEP356_WAIVER_ID not in item]
    missing_waiver[-1] = f"--output={tmp_path / 'missing-waiver.json'}"
    with pytest.raises(promotion.PromotionValidationError, match="each locked deviation"):
        module.main(missing_waiver)
