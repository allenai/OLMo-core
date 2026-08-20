"""Build, rebuild-audit, or explicitly approve an SSMax perception promotion report."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from olmo_core.eval.vision_alignment_ssmax_perception import (
    ARMS,
    REQUIRED_STEPS,
    TREATMENT_ARM,
    SSMaxPerceptionEvidenceError,
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
    promote.add_argument("--evaluation", action="append", default=[], metavar="ARM:STEP=PATH")
    promote.add_argument(
        "--expected-evaluation-sha256", action="append", default=[], metavar="ARM:STEP=SHA256"
    )
    promote.add_argument("--health", action="append", default=[], metavar="ARM:STEP=PATH")
    promote.add_argument(
        "--expected-health-sha256", action="append", default=[], metavar="ARM:STEP=SHA256"
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


def _values(values: Sequence[str], *, option: str) -> dict[str, dict[int, str]]:
    output: dict[str, dict[int, str]] = {arm: {} for arm in ARMS}
    for value in values:
        if "=" not in value or ":" not in value.split("=", 1)[0]:
            raise ValueError(f"{option} must be ARM:STEP=VALUE, got {value!r}")
        selector, item = value.split("=", 1)
        arm, raw_step = selector.split(":", 1)
        try:
            step = int(raw_step)
        except ValueError as error:
            raise ValueError(f"{option} step is invalid: {raw_step!r}") from error
        if arm not in ARMS or step not in REQUIRED_STEPS or step in output[arm] or not item:
            raise ValueError(f"{option} selector is invalid or repeated: {selector!r}")
        output[arm][step] = item
    if any(set(output[arm]) != set(REQUIRED_STEPS) for arm in ARMS):
        raise ValueError(f"{option} must provide both arms at steps {list(REQUIRED_STEPS)}")
    return output


def _references(
    paths: Sequence[str], pins: Sequence[str], *, option: str
) -> dict[str, dict[int, dict[str, str]]]:
    path_values = _values(paths, option=option)
    pin_values = _values(pins, option=f"--expected-{option.removeprefix('--')}-sha256")
    output: dict[str, dict[int, dict[str, str]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for step in REQUIRED_STEPS:
            path = Path(path_values[arm][step]).expanduser().resolve()
            expected = pin_values[arm][step]
            if len(expected) != 64 or sha256_file(path) != expected:
                raise SSMaxPerceptionEvidenceError(
                    f"{option} {arm}:step{step} differs from its explicit pin"
                )
            output[arm][step] = {"path": str(path), "sha256": expected}
    return output


def _audit_report(path: Path, expected_sha256: str) -> Mapping[str, object]:
    path = path.expanduser().resolve()
    if sha256_file(path) != expected_sha256:
        raise SSMaxPerceptionEvidenceError("Promotion report differs from its explicit pin")
    report = load_json(path)
    if not isinstance(report, Mapping):
        raise SSMaxPerceptionEvidenceError("Promotion report must be an object")
    manifest_ref = report.get("manifest")
    if not isinstance(manifest_ref, Mapping):
        raise SSMaxPerceptionEvidenceError("Promotion report lacks a manifest reference")
    manifest = load_manifest(Path(str(manifest_ref["path"])), verify_live=True)
    treatment = manifest["arms"][TREATMENT_ARM]
    candidate = treatment["checkpoints"]["4000"]
    return validate_promotion_report_reference(
        {"path": str(path), "sha256": expected_sha256},
        expected_checkpoint=Path(str(candidate["path"])),
        expected_checkpoint_config_sha256=str(candidate["config_sha256"]),
        expected_model_variant=str(manifest["model_variant"]),
        expected_data_contract_sha256=str(treatment["data_contract_sha256"]),
        expected_trainable_contract_sha256=str(treatment["trainable_contract_sha256"]),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "promote":
        manifest_path = args.manifest.expanduser().resolve()
        if sha256_file(manifest_path) != args.expected_manifest_sha256:
            raise SSMaxPerceptionEvidenceError("Pair manifest differs from its explicit pin")
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
        write_json_once(args.output, report)
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
    gate = build_parent_gate(
        promotion_report_path=args.report,
        expected_promotion_report_sha256=args.expected_report_sha256,
        approved_by=args.approved_by,
        approved_at=args.approved_at,
    )
    write_json_once(args.output, gate)


if __name__ == "__main__":
    main()
