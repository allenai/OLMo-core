"""Build, rebuild-audit, or explicitly approve a direct SSMax perception report."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from olmo_core.eval.vision_alignment_ssmax_perception_direct import (
    REQUIRED_STEPS,
    SSMaxPerceptionDirectEvidenceError,
    build_parent_gate,
    build_promotion_report,
    load_json,
    load_manifest,
    sha256_file,
    validate_promotion_report_reference,
    write_json_once,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    promote = commands.add_parser("promote")
    promote.add_argument("--manifest", type=Path, required=True)
    promote.add_argument("--expected-manifest-sha256", required=True)
    promote.add_argument("--evaluation", action="append", default=[], metavar="STEP=PATH")
    promote.add_argument(
        "--expected-evaluation-sha256", action="append", default=[], metavar="STEP=SHA256"
    )
    promote.add_argument("--health", action="append", default=[], metavar="STEP=PATH")
    promote.add_argument(
        "--expected-health-sha256", action="append", default=[], metavar="STEP=SHA256"
    )
    promote.add_argument("--output", type=Path, required=True)
    promote.add_argument("--created-at")

    audit = commands.add_parser("audit")
    audit.add_argument("--report", type=Path, required=True)
    audit.add_argument("--expected-report-sha256", required=True)

    approve = commands.add_parser("approve")
    approve.add_argument("--report", type=Path, required=True)
    approve.add_argument("--expected-report-sha256", required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--approved-at", required=True)
    approve.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _values(values: Sequence[str], *, option: str) -> dict[int, str]:
    output: dict[int, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} must be STEP=VALUE, got {value!r}")
        raw_step, item = value.split("=", 1)
        try:
            step = int(raw_step)
        except ValueError as error:
            raise ValueError(f"{option} step is invalid: {raw_step!r}") from error
        if step not in REQUIRED_STEPS or step in output or not item:
            raise ValueError(f"{option} selector is invalid or repeated: {raw_step!r}")
        output[step] = item
    if set(output) != set(REQUIRED_STEPS):
        raise ValueError(f"{option} must provide exactly steps {list(REQUIRED_STEPS)}")
    return output


def _references(
    paths: Sequence[str], pins: Sequence[str], *, option: str
) -> dict[int, dict[str, str]]:
    path_values = _values(paths, option=option)
    pin_values = _values(pins, option=f"--expected-{option.removeprefix('--')}-sha256")
    output: dict[int, dict[str, str]] = {}
    for step in REQUIRED_STEPS:
        path = Path(path_values[step]).expanduser().resolve()
        expected = pin_values[step]
        if len(expected) != 64 or sha256_file(path) != expected:
            raise SSMaxPerceptionDirectEvidenceError(
                f"{option} step{step} differs from its explicit pin"
            )
        output[step] = {"path": str(path), "sha256": expected}
    return output


def _audit_report(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    path = path.expanduser().resolve()
    if sha256_file(path) != expected_sha256:
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct promotion report differs from its explicit pin"
        )
    report = load_json(path)
    if not isinstance(report, Mapping):
        raise SSMaxPerceptionDirectEvidenceError("Direct promotion report must be an object")
    manifest_ref = report.get("manifest")
    if not isinstance(manifest_ref, Mapping):
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct promotion report lacks a manifest reference"
        )
    manifest = load_manifest(Path(str(manifest_ref["path"])), verify_live=True)
    run = manifest["run"]
    candidate = run["checkpoints"]["4000"]
    return validate_promotion_report_reference(
        {"path": str(path), "sha256": expected_sha256},
        expected_checkpoint=Path(str(candidate["path"])),
        expected_checkpoint_config_sha256=str(candidate["config_sha256"]),
        expected_model_variant=str(manifest["model_variant"]),
        expected_data_contract_sha256=str(run["data_contract_sha256"]),
        expected_trainable_contract_sha256=str(run["trainable_contract_sha256"]),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "promote":
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite immutable report {output}")
        manifest_path = args.manifest.expanduser().resolve()
        if sha256_file(manifest_path) != args.expected_manifest_sha256:
            raise SSMaxPerceptionDirectEvidenceError(
                "Direct manifest differs from its explicit pin"
            )
        report = build_promotion_report(
            manifest_path=manifest_path,
            evaluation_receipts=_references(
                args.evaluation,
                args.expected_evaluation_sha256,
                option="--evaluation",
            ),
            health_receipts=_references(
                args.health,
                args.expected_health_sha256,
                option="--health",
            ),
            created_at=args.created_at or datetime.now(timezone.utc).isoformat(),
        )
        write_json_once(output, report)
        if report["status"] != "passed":
            raise SystemExit(2)
        return
    if args.command == "audit":
        summary = _audit_report(args.report, args.expected_report_sha256)
        print(
            f"passed model_variant={summary['manifest']['model_variant']} "
            f"checkpoint={summary['candidate']['path']}"
        )
        return
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable parent gate {output}")
    gate = build_parent_gate(
        promotion_report_path=args.report,
        expected_promotion_report_sha256=args.expected_report_sha256,
        approved_by=args.approved_by,
        approved_at=args.approved_at,
    )
    write_json_once(output, gate)


if __name__ == "__main__":
    main()
