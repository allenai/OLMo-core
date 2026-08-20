"""Build per-arm SSMax bridge promotion reports and a controlled paired comparison.

``promote`` consumes all seven matched/state and health receipts for one immutable run manifest.
``compare`` consumes the two resulting promotion reports and performs row-paired arm deltas while
proving that step-0 SigLIP, connector, and image-token rows are bit-identical.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from olmo_core.eval.vision_alignment_ssmax_bridge import (
    REQUIRED_STEPS,
    SSMaxBridgeEvidenceError,
    build_pair_comparison,
    build_parent_gate,
    build_promotion_report,
    sha256_file,
    write_json_once,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    promote = subparsers.add_parser("promote")
    promote.add_argument("--manifest", type=Path, required=True)
    promote.add_argument("--expected-manifest-sha256", required=True)
    promote.add_argument("--matched", action="append", default=[], metavar="STEP=PATH")
    promote.add_argument(
        "--expected-matched-sha256", action="append", default=[], metavar="STEP=SHA256"
    )
    promote.add_argument("--health", action="append", default=[], metavar="STEP=PATH")
    promote.add_argument(
        "--expected-health-sha256", action="append", default=[], metavar="STEP=SHA256"
    )
    promote.add_argument("--output", type=Path, required=True)
    promote.add_argument("--created-at")

    approve = subparsers.add_parser("approve")
    approve.add_argument("--report", type=Path, required=True)
    approve.add_argument("--expected-report-sha256", required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--approved-at", required=True)
    approve.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--left-promotion", type=Path, required=True)
    compare.add_argument("--expected-left-promotion-sha256", required=True)
    compare.add_argument("--right-promotion", type=Path, required=True)
    compare.add_argument("--expected-right-promotion-sha256", required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--created-at")
    return parser.parse_args(argv)


def _step_values(values: Sequence[str], *, option: str) -> dict[int, str]:
    output: dict[int, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} must be STEP=VALUE, got {value!r}")
        raw_step, item = value.split("=", 1)
        try:
            step = int(raw_step)
        except ValueError as error:
            raise ValueError(f"{option} step is not an integer: {raw_step!r}") from error
        if step not in REQUIRED_STEPS or step in output or not item:
            raise ValueError(f"{option} has invalid or duplicate step {raw_step!r}")
        output[step] = item
    if set(output) != set(REQUIRED_STEPS):
        raise ValueError(f"{option} must provide exactly steps {list(REQUIRED_STEPS)}")
    return output


def _receipt_references(
    paths: Sequence[str], pins: Sequence[str], *, option: str
) -> dict[int, dict[str, str]]:
    path_values = _step_values(paths, option=option)
    sha_values = _step_values(pins, option=f"--expected-{option.removeprefix('--')}-sha256")
    references = {}
    for step in REQUIRED_STEPS:
        path = Path(path_values[step]).expanduser().resolve()
        expected = sha_values[step]
        if len(expected) != 64 or sha256_file(path) != expected:
            raise SSMaxBridgeEvidenceError(f"{option} step{step} differs from its explicit pin")
        references[step] = {"path": str(path), "sha256": expected}
    return references


def main(argv: Sequence[str] | None = None) -> None:
    """Build an immutable promotion report or controlled two-arm comparison."""

    args = _parse_args(argv)
    created_at = args.created_at or datetime.now(timezone.utc).isoformat()
    if args.command == "promote":
        manifest_path = args.manifest.expanduser().resolve()
        if sha256_file(manifest_path) != args.expected_manifest_sha256:
            raise SSMaxBridgeEvidenceError("Run manifest differs from its explicit pin")
        report = build_promotion_report(
            manifest_path=manifest_path,
            matched_receipts=_receipt_references(
                args.matched,
                args.expected_matched_sha256,
                option="--matched",
            ),
            health_receipts=_receipt_references(
                args.health,
                args.expected_health_sha256,
                option="--health",
            ),
            created_at=created_at,
        )
        write_json_once(args.output, report)
        if report["status"] != "passed":
            raise SystemExit(2)
        return
    if args.command == "approve":
        gate = build_parent_gate(
            promotion_report_path=args.report,
            expected_promotion_report_sha256=args.expected_report_sha256,
            approved_by=args.approved_by,
            approved_at=args.approved_at,
        )
        write_json_once(args.output, gate)
        return

    comparison = build_pair_comparison(
        left_promotion_report={
            "path": str(args.left_promotion.expanduser().resolve()),
            "sha256": args.expected_left_promotion_sha256,
        },
        right_promotion_report={
            "path": str(args.right_promotion.expanduser().resolve()),
            "sha256": args.expected_right_promotion_sha256,
        },
        created_at=created_at,
    )
    write_json_once(args.output, comparison)


if __name__ == "__main__":
    main()
