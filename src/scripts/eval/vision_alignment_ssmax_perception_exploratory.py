"""Audit rejected strict SSMax evidence or issue an exploratory-only version-8 gate."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from olmo_core.eval.vision_alignment_ssmax_perception_direct import write_json_once
from olmo_core.eval.vision_alignment_ssmax_perception_exploratory import (
    audit_strict_report_reference,
    build_parent_gate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit")
    audit.add_argument("--report", type=Path, required=True)
    audit.add_argument("--expected-report-sha256", required=True)

    approve = commands.add_parser("approve")
    approve.add_argument("--report", type=Path, required=True)
    approve.add_argument("--expected-report-sha256", required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--approved-at", required=True)
    approve.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selected immutable exploratory evidence operation."""

    args = _parser().parse_args(argv)
    report_path = args.report.expanduser().resolve()
    if args.command == "audit":
        summary = audit_strict_report_reference(
            {
                "path": str(report_path),
                "sha256": args.expected_report_sha256,
            }
        )
        print(
            f"eligible scope=exploratory_joint_only "
            f"model_variant={summary['manifest']['model_variant']} "
            f"checkpoint={summary['candidate']['path']} "
            f"acknowledged_deviations={len(summary['acknowledged_deviations'])}"
        )
        return

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable exploratory gate {output}")
    gate = build_parent_gate(
        strict_report_path=report_path,
        expected_strict_report_sha256=args.expected_report_sha256,
        approved_by=args.approved_by,
        approved_at=args.approved_at,
    )
    write_json_once(output, gate)


if __name__ == "__main__":
    main()
