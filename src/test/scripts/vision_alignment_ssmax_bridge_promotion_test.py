"""CLI regression tests for SSMax bridge promotion and approval."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from olmo_core.eval.vision_alignment_ssmax_bridge import load_json


def _load_module():
    path = Path(__file__).resolve().parents[2] / (
        "scripts/eval/vision_alignment_ssmax_bridge_promotion.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vision_alignment_ssmax_bridge_promotion_test_module", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_approve_does_not_require_created_at_argument(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    report = tmp_path / "promotion.json"
    output = tmp_path / "parent-gate-v4.json"
    expected_sha256 = "a" * 64
    expected_gate = {"format": "vision_alignment_parent_gate", "version": 4}
    calls = []

    def build_parent_gate(**kwargs):
        calls.append(kwargs)
        return expected_gate

    monkeypatch.setattr(module, "build_parent_gate", build_parent_gate)
    module.main(
        [
            "approve",
            f"--report={report}",
            f"--expected-report-sha256={expected_sha256}",
            "--approved-by=rustins",
            "--approved-at=2026-08-21T00:00:00+00:00",
            f"--output={output}",
        ]
    )

    assert calls == [
        {
            "promotion_report_path": report,
            "expected_promotion_report_sha256": expected_sha256,
            "approved_by": "rustins",
            "approved_at": "2026-08-21T00:00:00+00:00",
        }
    ]
    assert load_json(output) == expected_gate
