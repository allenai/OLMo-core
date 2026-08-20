"""Build or audit a descriptive, non-promotion SSMax joint trajectory report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from olmo_core.eval.vision_alignment_ssmax_joint import (
    REQUIRED_STEPS,
    artifact_reference,
    build_trajectory_report,
    validate_trajectory_report,
    write_json_once,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--evaluation-receipt", action="append", default=[], metavar="STEP=PATH")
    parser.add_argument("--health-receipt", action="append", default=[], metavar="STEP=PATH")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--created-at")
    parser.add_argument("--audit", type=Path)
    return parser.parse_args(argv)


def _references(values: Sequence[str], *, name: str) -> dict[int, dict[str, str]]:
    result = {}
    for value in values:
        raw_step, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError(f"{name} must use STEP=PATH")
        step = int(raw_step)
        if step in result:
            raise ValueError(f"{name} repeats step{step}")
        result[step] = artifact_reference(Path(raw_path))
    if set(result) != set(REQUIRED_STEPS):
        raise ValueError(f"{name} must contain exactly {list(REQUIRED_STEPS)}")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.audit is not None:
        if any((args.manifest, args.output, args.evaluation_receipt, args.health_receipt)):
            raise ValueError("--audit cannot be combined with report creation arguments")
        validate_trajectory_report(args.audit.expanduser().resolve())
        return
    if args.manifest is None or args.output is None:
        raise ValueError("report creation requires --manifest and --output")
    report = build_trajectory_report(
        manifest_path=args.manifest.expanduser().resolve(),
        evaluation_receipts=_references(args.evaluation_receipt, name="evaluation receipts"),
        health_receipts=_references(args.health_receipt, name="health receipts"),
        created_at=args.created_at or datetime.now(timezone.utc).isoformat(),
    )
    write_json_once(args.output.expanduser().resolve(), report)


if __name__ == "__main__":
    main()
