"""Build, audit, or approve a research-only SSMax perception health-waiver gate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from olmo_core.eval.vision_alignment_ssmax_perception_direct import (
    sha256_file,
    write_json_once,
)
from olmo_core.eval.vision_alignment_ssmax_perception_exploratory_waiver import (
    REQUIRED_EVALUATION_STEPS,
    REQUIRED_HEALTH_STEPS,
    SSMaxPerceptionExploratoryWaiverEvidenceError,
    build_evidence_report,
    build_parent_gate,
    validate_evidence_report_reference,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--expected-manifest-sha256", required=True)
    build.add_argument("--evaluation", action="append", default=[], metavar="STEP=PATH")
    build.add_argument(
        "--expected-evaluation-sha256", action="append", default=[], metavar="STEP=SHA256"
    )
    build.add_argument("--health", action="append", default=[], metavar="STEP=PATH")
    build.add_argument(
        "--expected-health-sha256", action="append", default=[], metavar="STEP=SHA256"
    )
    build.add_argument("--created-at")
    build.add_argument("--output", type=Path, required=True)

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


def _values(
    values: Sequence[str], *, option: str, required_steps: tuple[int, ...]
) -> dict[int, str]:
    output: dict[int, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} must be STEP=VALUE, got {value!r}")
        raw_step, item = value.split("=", 1)
        try:
            step = int(raw_step)
        except ValueError as error:
            raise ValueError(f"{option} step is invalid: {raw_step!r}") from error
        if step not in required_steps or step in output or not item:
            raise ValueError(f"{option} selector is invalid or repeated: {raw_step!r}")
        output[step] = item
    if set(output) != set(required_steps):
        raise ValueError(f"{option} must provide exactly steps {list(required_steps)}")
    return output


def _references(
    paths: Sequence[str],
    pins: Sequence[str],
    *,
    option: str,
    required_steps: tuple[int, ...],
) -> dict[int, dict[str, str]]:
    path_values = _values(paths, option=option, required_steps=required_steps)
    pin_values = _values(
        pins,
        option=f"--expected-{option.removeprefix('--')}-sha256",
        required_steps=required_steps,
    )
    output: dict[int, dict[str, str]] = {}
    for step in required_steps:
        path = Path(path_values[step]).expanduser().resolve()
        expected = pin_values[step]
        if len(expected) != 64 or sha256_file(path) != expected:
            raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                f"{option} step{step} differs from its explicit pin"
            )
        output[step] = {"path": str(path), "sha256": expected}
    return output


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selected immutable exploratory waiver operation."""

    args = _parser().parse_args(argv)
    if args.command == "build":
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite immutable evidence report {output}")
        report = build_evidence_report(
            manifest_path=args.manifest.expanduser().resolve(),
            expected_manifest_sha256=args.expected_manifest_sha256,
            evaluation_receipts=_references(
                args.evaluation,
                args.expected_evaluation_sha256,
                option="--evaluation",
                required_steps=REQUIRED_EVALUATION_STEPS,
            ),
            health_receipts=_references(
                args.health,
                args.expected_health_sha256,
                option="--health",
                required_steps=REQUIRED_HEALTH_STEPS,
            ),
            created_at=args.created_at or datetime.now(timezone.utc).isoformat(),
        )
        write_json_once(output, report)
        return

    report_path = args.report.expanduser().resolve()
    if args.command == "audit":
        summary: Mapping[str, object] = validate_evidence_report_reference(
            {"path": str(report_path), "sha256": args.expected_report_sha256}
        )
        manifest = summary["manifest"]
        candidate = summary["candidate"]
        assert isinstance(manifest, Mapping)
        assert isinstance(candidate, Mapping)
        print(
            f"eligible_with_required_waiver model_variant={manifest['model_variant']} "
            f"checkpoint={candidate['path']}"
        )
        return

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable parent gate {output}")
    gate = build_parent_gate(
        evidence_report_path=report_path,
        expected_evidence_report_sha256=args.expected_report_sha256,
        approved_by=args.approved_by,
        approved_at=args.approved_at,
    )
    write_json_once(output, gate)


if __name__ == "__main__":
    main()
