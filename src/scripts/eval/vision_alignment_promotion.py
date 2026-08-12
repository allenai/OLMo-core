"""Build or audit the immutable Vision Alignment bridge-step500 promotion bundle.

The builder consumes already-produced, raw-SHA-pinned scientific receipts. It never launches
training or creates missing measurements. Existing output paths are always rejected::

    PYTHONPATH=src python src/scripts/eval/vision_alignment_promotion.py build \
      --checkpoint=/path/to/bridge/step500 \
      --frozen-state=/path/to/frozen-state.json \
      --text-retention=/path/to/text-retention.json \
      --cumulative-loss-mass=/path/to/loss-mass.json \
      --optimizer-guard=/path/to/optimizer-guard.json \
      --canary-step250=/path/to/canary-step250-matched.json \
      --bridge-step250=/path/to/bridge-step250-matched.json \
      --bridge-step500=/path/to/bridge-step500-matched.json \
      --independent-step0=/path/to/independent-step0-matched.json \
      --independent-step500=/path/to/independent-step500-matched.json \
      --output=/path/to/promotion-bundle.json

Re-audit every referenced artifact before signing a parent gate::

    PYTHONPATH=src python src/scripts/eval/vision_alignment_promotion.py audit \
      --bundle=/path/to/promotion-bundle.json \
      --expected-sha256=<raw-bundle-sha256>
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from olmo_core.eval.vision_alignment_promotion import (
    REQUIRED_WAIVER_IDS,
    PromotionValidationError,
    build_optimizer_guard_receipt,
    build_promotion_bundle,
    build_text_sentinel,
    candidate_from_matched_receipt,
    load_json,
    sha256_file,
    validate_promotion_bundle,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="Validate receipts and write one immutable bundle.")
    build.add_argument("--checkpoint", type=Path, required=True)
    build.add_argument("--frozen-state", type=Path, required=True)
    build.add_argument("--text-retention", type=Path, required=True)
    build.add_argument("--cumulative-loss-mass", type=Path, required=True)
    build.add_argument("--optimizer-guard", type=Path, required=True)
    build.add_argument("--canary-step250", type=Path, required=True)
    build.add_argument("--bridge-step250", type=Path, required=True)
    build.add_argument("--bridge-step500", type=Path, required=True)
    build.add_argument("--independent-step0", type=Path, required=True)
    build.add_argument("--independent-step500", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--created-at",
        help="Pinned ISO-8601 creation time (defaults to the current UTC time).",
    )

    audit = commands.add_parser("audit", help="Revalidate a bundle and all pinned receipts.")
    audit.add_argument("--bundle", type=Path, required=True)
    audit.add_argument("--expected-sha256", required=True)
    audit.add_argument("--expected-checkpoint", type=Path)
    audit.add_argument("--expected-checkpoint-config-sha256")

    run_health = commands.add_parser(
        "run-health", help="Build the durable optimizer/run-health and step356 guard receipt."
    )
    run_health.add_argument("--checkpoint", type=Path, required=True)
    run_health.add_argument("--matched-step500", type=Path, required=True)
    run_health.add_argument("--expected-matched-step500-sha256", required=True)
    run_health.add_argument("--output-log", type=Path, required=True)
    run_health.add_argument("--expected-output-log-sha256", required=True)
    run_health.add_argument("--output", type=Path, required=True)
    run_health.add_argument("--created-at")

    text_sentinel = commands.add_parser(
        "text-sentinel", help="Build the pinned image-free parent-pretraining sentinel."
    )
    text_sentinel.add_argument("--parent-checkpoint", type=Path, required=True)
    text_sentinel.add_argument("--parent-checkpoint-config-sha256", required=True)
    text_sentinel.add_argument("--parent-data-paths", type=Path, required=True)
    text_sentinel.add_argument("--expected-parent-data-paths-sha256", required=True)
    text_sentinel.add_argument("--sequence-length", type=int, default=256)
    text_sentinel.add_argument("--examples", type=int, default=128)
    text_sentinel.add_argument("--output", type=Path, required=True)

    approve = commands.add_parser(
        "approve", help="Write a strict v2 parent gate from an audited promotion bundle."
    )
    approve.add_argument("--bundle", type=Path, required=True)
    approve.add_argument("--expected-sha256", required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--approved-at", required=True)
    approve.add_argument(
        "--approve-waiver",
        action="append",
        default=[],
        help="Explicitly approve one locked deviation ID; both required IDs must be supplied.",
    )
    approve.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _write_json_once(path: Path, payload: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"Refusing to overwrite immutable promotion artifact {path}"
            ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _print_summary(path: Path, summary: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "path": str(path.expanduser().resolve()),
                "sha256": sha256_file(path.expanduser().resolve()),
                **summary,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the immutable promotion bundle builder or auditor."""
    args = _parse_args(argv)
    if args.command == "build":
        created_at = args.created_at or datetime.now(timezone.utc).isoformat()
        bundle = build_promotion_bundle(
            checkpoint=args.checkpoint,
            frozen_state=args.frozen_state,
            text_retention=args.text_retention,
            cumulative_loss_mass=args.cumulative_loss_mass,
            optimizer_guard=args.optimizer_guard,
            canary_step250=args.canary_step250,
            bridge_step250=args.bridge_step250,
            bridge_step500=args.bridge_step500,
            independent_step0=args.independent_step0,
            independent_step500=args.independent_step500,
            created_at=created_at,
        )
        _write_json_once(args.output, bundle)
        persisted = load_json(args.output.expanduser().resolve())
        if not isinstance(persisted, dict):
            raise PromotionValidationError("Persisted promotion bundle is not an object")
        summary = validate_promotion_bundle(
            persisted, expected_checkpoint=args.checkpoint.expanduser().resolve()
        )
        _print_summary(args.output, summary)
        return

    if args.command == "run-health":
        matched_path = args.matched_step500.expanduser().resolve()
        actual_matched_sha = sha256_file(matched_path)
        if actual_matched_sha != args.expected_matched_step500_sha256:
            raise PromotionValidationError(
                "Primary step500 matched receipt differs from its explicit SHA-256 pin"
            )
        matched = load_json(matched_path)
        if not isinstance(matched, dict):
            raise PromotionValidationError("Primary step500 matched receipt must be an object")
        candidate = candidate_from_matched_receipt(args.checkpoint, matched)
        created_at = args.created_at or datetime.now(timezone.utc).isoformat()
        receipt = build_optimizer_guard_receipt(
            candidate=candidate,
            output_log=args.output_log,
            expected_output_log_sha256=args.expected_output_log_sha256,
            created_at=created_at,
        )
        _write_json_once(args.output, receipt)
        _print_summary(
            args.output,
            {
                "status": receipt["status"],
                "run": receipt["run"],
                "guarded_skips": receipt["guarded_skips"],
            },
        )
        return

    if args.command == "text-sentinel":
        sentinel = build_text_sentinel(
            parent_checkpoint=args.parent_checkpoint,
            parent_checkpoint_config_sha256=args.parent_checkpoint_config_sha256,
            parent_data_paths=args.parent_data_paths,
            expected_parent_data_paths_sha256=args.expected_parent_data_paths_sha256,
            sequence_length=args.sequence_length,
            examples=args.examples,
        )
        _write_json_once(args.output, sentinel)
        _print_summary(
            args.output,
            {
                "status": "passed",
                "examples": sentinel["selection"]["examples"],
                "sequence_length": sentinel["selection"]["sequence_length"],
                "content_sha256": sentinel["content_sha256"],
            },
        )
        return

    if args.command == "approve":
        bundle_path = args.bundle.expanduser().resolve()
        actual_sha = sha256_file(bundle_path)
        if actual_sha != args.expected_sha256:
            raise PromotionValidationError(
                f"Promotion bundle raw SHA-256 differs: expected {args.expected_sha256}, "
                f"got {actual_sha}"
            )
        bundle = load_json(bundle_path)
        if not isinstance(bundle, dict):
            raise PromotionValidationError("Promotion bundle must be an object")
        summary = validate_promotion_bundle(bundle)
        candidate = summary["candidate"]
        if (
            not isinstance(args.approved_by, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@:/+\-]{2,127}", args.approved_by) is None
        ):
            raise PromotionValidationError("--approved-by must be a durable human identity")
        try:
            approved_at = datetime.fromisoformat(args.approved_at.replace("Z", "+00:00"))
            bundle_created_at = datetime.fromisoformat(bundle["created_at"].replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError) as error:
            raise PromotionValidationError(
                "Approval and bundle timestamps must be ISO-8601"
            ) from error
        if (
            approved_at.tzinfo is None
            or approved_at.utcoffset() is None
            or bundle_created_at.tzinfo is None
            or bundle_created_at.utcoffset() is None
            or approved_at < bundle_created_at
        ):
            raise PromotionValidationError("Approval must occur after bundle creation")
        if (
            len(args.approve_waiver) != len(REQUIRED_WAIVER_IDS)
            or set(args.approve_waiver) != REQUIRED_WAIVER_IDS
        ):
            raise PromotionValidationError(
                "Approval requires one explicit --approve-waiver for each locked deviation"
            )
        config = load_json(Path(candidate["checkpoint"]) / "config.json")
        metadata = config.get("vision_alignment") if isinstance(config, dict) else None
        if not isinstance(metadata, dict):
            raise PromotionValidationError("Candidate config lacks vision_alignment metadata")
        gate = {
            "format": "vision_alignment_parent_gate",
            "version": 2,
            "status": "approved",
            "recipe_version": metadata.get("recipe_version"),
            "formatter_version": metadata.get("formatter_version"),
            "phase": candidate["phase"],
            "checkpoint": candidate["checkpoint"],
            "checkpoint_config_sha256": candidate["checkpoint_config_sha256"],
            "data_contract_sha256": candidate["data_contract_sha256"],
            "trainable_contract_sha256": candidate["trainable_contract_sha256"],
            "global_step": candidate["global_step"],
            "metrics_artifact_sha256": actual_sha,
            "promotion_bundle_path": str(bundle_path),
            "promotion_bundle_sha256": actual_sha,
            "checkpoint_identity_sha256": candidate["checkpoint_identity_sha256"],
            "approved_by": args.approved_by,
            "approved_at": args.approved_at,
            "waivers": [
                {
                    "id": waiver_id,
                    "decision": "approved",
                    "deviation_sha256": summary["deviation_sha256"][waiver_id],
                }
                for waiver_id in sorted(REQUIRED_WAIVER_IDS)
            ],
        }
        if (
            not isinstance(gate["recipe_version"], int)
            or not isinstance(gate["formatter_version"], str)
            or not gate["formatter_version"]
        ):
            raise PromotionValidationError("Candidate recipe metadata is incomplete")
        _write_json_once(args.output, gate)
        _print_summary(
            args.output,
            {
                "status": "approved",
                "approved_by": args.approved_by,
                "waiver_ids": sorted(REQUIRED_WAIVER_IDS),
            },
        )
        return

    bundle_path = args.bundle.expanduser().resolve()
    actual_sha = sha256_file(bundle_path)
    if actual_sha != args.expected_sha256:
        raise PromotionValidationError(
            f"Promotion bundle raw SHA-256 differs: expected {args.expected_sha256}, "
            f"got {actual_sha}"
        )
    bundle = load_json(bundle_path)
    if not isinstance(bundle, dict):
        raise PromotionValidationError("Promotion bundle must be an object")
    summary = validate_promotion_bundle(
        bundle,
        expected_checkpoint=(
            args.expected_checkpoint.expanduser().resolve()
            if args.expected_checkpoint is not None
            else None
        ),
        expected_checkpoint_config_sha256=args.expected_checkpoint_config_sha256,
    )
    _print_summary(bundle_path, summary)


if __name__ == "__main__":
    main()
