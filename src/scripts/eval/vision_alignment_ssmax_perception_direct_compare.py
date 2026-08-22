"""Compare two validated direct perception lineages descriptively, without selecting a winner."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from olmo_core.eval.vision_alignment_ssmax_perception_direct import (
    MODEL_VARIANTS,
    SCHEMA_VERSION,
    WINDOWS,
    SSMaxPerceptionDirectEvidenceError,
    canonical_sha256,
    write_json_once,
)
from scripts.eval.vision_alignment_ssmax_perception_direct_promotion import (
    _audit_report,
)

COMPARISON_FORMAT = "vision_alignment_ssmax_perception_direct_model_variant_comparison"
_SHARED_MANIFEST_FIELDS = (
    "training_git",
    "evidence_git",
    "producers",
    "training_recipe",
    "protocol_amendment",
    "perception_provenance",
    "source_audit",
    "source_audit_fingerprint",
    "single_response_projection",
    "attention_probe",
    "text_sentinel",
    "pairings",
    "evaluation",
    "topology",
    "policy",
    "loss_mass_targets",
)
_MACRO_METRICS = (
    "macro_step0_correct_ce",
    "macro_step4000_correct_ce",
    "macro_step0_gap",
    "macro_step3000_gap",
    "macro_step4000_gap",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-promotion-report", type=Path, required=True)
    parser.add_argument("--expected-left-promotion-report-sha256", required=True)
    parser.add_argument("--right-promotion-report", type=Path, required=True)
    parser.add_argument("--expected-right-promotion-report-sha256", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _timestamp(value: Any, *, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise SSMaxPerceptionDirectEvidenceError(f"{name} must be a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SSMaxPerceptionDirectEvidenceError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise SSMaxPerceptionDirectEvidenceError(f"{name} must include a timezone")
    return parsed


def _variant_entry(
    summary: Mapping[str, Any], *, report_path: Path, report_sha256: str
) -> tuple[str, dict[str, Any]]:
    report = summary["report"]
    manifest = summary["manifest"]
    candidate = summary["candidate"]
    variant = str(manifest["model_variant"])
    if report["model_variant"] != variant or report["run_id"] != manifest["run_id"]:
        raise SSMaxPerceptionDirectEvidenceError(
            "Validated report and manifest lineage identities differ"
        )
    windows = report["summary"]["windows"]
    if set(windows) != set(WINDOWS):
        raise SSMaxPerceptionDirectEvidenceError("Direct report window set differs")
    return variant, {
        "promotion_report": {
            "path": str(report_path.expanduser().resolve()),
            "sha256": report_sha256,
            "content_sha256": report["content_sha256"],
        },
        "run_id": report["run_id"],
        "checkpoint_identity_sha256": candidate["identity_sha256"],
        "windows": {window: dict(windows[window]) for window in WINDOWS},
        "attention_trajectory": dict(report["summary"]["attention_trajectory"]),
        "optimizer_guard_trajectory": dict(report["summary"]["optimizer_guard_trajectory"]),
    }


def _shared_protocol_inputs(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {field: manifest[field] for field in _SHARED_MANIFEST_FIELDS}


def build_comparison(
    *,
    left_summary: Mapping[str, Any],
    left_report_path: Path,
    left_report_sha256: str,
    right_summary: Mapping[str, Any],
    right_report_path: Path,
    right_report_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    """Build a non-promotional metric comparison from two fully rebuilt direct reports."""

    comparison_time = _timestamp(created_at, name="comparison created_at")
    left_shared = _shared_protocol_inputs(left_summary["manifest"])
    right_shared = _shared_protocol_inputs(right_summary["manifest"])
    if left_shared != right_shared:
        mismatched = [
            field for field in _SHARED_MANIFEST_FIELDS if left_shared[field] != right_shared[field]
        ]
        raise SSMaxPerceptionDirectEvidenceError(
            f"Cross-model shared protocol inputs differ: {mismatched}"
        )
    entries = dict(
        [
            _variant_entry(
                left_summary,
                report_path=left_report_path,
                report_sha256=left_report_sha256,
            ),
            _variant_entry(
                right_summary,
                report_path=right_report_path,
                report_sha256=right_report_sha256,
            ),
        ]
    )
    if set(entries) != set(MODEL_VARIANTS):
        raise SSMaxPerceptionDirectEvidenceError(
            "Comparison requires exactly one report for each model variant"
        )
    for summary in (left_summary, right_summary):
        report_time = _timestamp(
            summary["report"]["created_at"], name="promotion report created_at"
        )
        if comparison_time < report_time:
            raise SSMaxPerceptionDirectEvidenceError(
                "Comparison predates one of its promotion reports"
            )
    head = entries["ssmax_head_qknorm"]
    no_qk = entries["ssmax_no_qknorm"]

    def difference(window: str, metric: str) -> float:
        return float(head["windows"][window][metric]) - float(no_qk["windows"][window][metric])

    differences = {
        window: {
            "macro_metric_difference": {
                metric: difference(window, metric) for metric in _MACRO_METRICS
            },
            "same_step_metric_difference": {
                "correct_ce": {
                    "0": difference(window, "macro_step0_correct_ce"),
                    "4000": difference(window, "macro_step4000_correct_ce"),
                },
                "visual_gap": {
                    "0": difference(window, "macro_step0_gap"),
                    "3000": difference(window, "macro_step3000_gap"),
                    "4000": difference(window, "macro_step4000_gap"),
                },
            },
            "step0_normalized_adaptation_difference": {
                "correct_ce": {
                    "4000": difference(window, "macro_step4000_correct_ce")
                    - difference(window, "macro_step0_correct_ce"),
                },
                "visual_gap": {
                    "3000": difference(window, "macro_step3000_gap")
                    - difference(window, "macro_step0_gap"),
                    "4000": difference(window, "macro_step4000_gap")
                    - difference(window, "macro_step0_gap"),
                },
            },
        }
        for window in WINDOWS
    }
    result: dict[str, Any] = {
        "format": COMPARISON_FORMAT,
        "version": SCHEMA_VERSION,
        "status": "descriptive_only",
        "decision_scope": "descriptive_non_promotion",
        "created_at": created_at,
        "winner": None,
        "shared_protocol_inputs": left_shared,
        "shared_protocol_inputs_sha256": canonical_sha256(left_shared),
        "model_variants": {variant: entries[variant] for variant in MODEL_VARIANTS},
        "descriptive_difference": {
            "definition": "ssmax_head_qknorm minus ssmax_no_qknorm",
            "step0_normalized_definition": (
                "(ssmax_head_qknorm stepN minus its step0) minus "
                "(ssmax_no_qknorm stepN minus its step0)"
            ),
            "windows": differences,
        },
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable comparison {output}")
    left_path = args.left_promotion_report.expanduser().resolve()
    right_path = args.right_promotion_report.expanduser().resolve()
    left = _audit_report(left_path, args.expected_left_promotion_report_sha256)
    right = _audit_report(right_path, args.expected_right_promotion_report_sha256)
    result = build_comparison(
        left_summary=left,
        left_report_path=left_path,
        left_report_sha256=args.expected_left_promotion_report_sha256,
        right_summary=right,
        right_report_path=right_path,
        right_report_sha256=args.expected_right_promotion_report_sha256,
        created_at=args.created_at,
    )
    write_json_once(output, result)
    print(
        json.dumps(
            {"output": str(output), "content_sha256": result["content_sha256"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
