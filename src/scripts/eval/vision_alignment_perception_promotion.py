"""Build, audit, or explicitly approve a perception-step4000 promotion bundle.

The builder consumes immutable scientific receipts; it never launches evaluation or training.
Every output path is write-once.  The ``approve`` command is reserved for an accountable human
after a separate audit and requires every locked deviation to be named explicitly.
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

from olmo_core.eval.vision_alignment_perception_promotion import (
    PERCEPTION_PROMOTION_POLICY,
    REQUIRED_WAIVER_IDS,
    PromotionValidationError,
    build_perception_promotion_bundle,
    load_json_pinned,
    sha256_file,
    validate_perception_promotion_bundle,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="Validate receipts and write one immutable bundle.")
    build.add_argument("--checkpoint", type=Path, required=True)
    build.add_argument("--comparator-checkpoint", type=Path, required=True)
    build.add_argument("--pair-contract", type=Path, required=True)
    build.add_argument("--counterfactual-outcome", type=Path, required=True)
    build.add_argument("--control-initialization-parity", type=Path, required=True)
    build.add_argument("--treatment-initialization-parity", type=Path, required=True)
    build.add_argument("--control-frozen-state", type=Path, required=True)
    build.add_argument("--treatment-frozen-state", type=Path, required=True)
    build.add_argument("--control-text-retention", type=Path, required=True)
    build.add_argument("--treatment-text-retention", type=Path, required=True)
    build.add_argument("--control-run-health", type=Path, required=True)
    build.add_argument("--treatment-run-health", type=Path, required=True)
    build.add_argument("--loss-mass-pair", type=Path, required=True)
    build.add_argument("--created-at", help="Pinned ISO-8601 time (defaults to current UTC).")
    build.add_argument("--output", type=Path, required=True)

    audit = commands.add_parser("audit", help="Revalidate a bundle and all pinned receipts.")
    audit.add_argument("--bundle", type=Path, required=True)
    audit.add_argument("--expected-sha256", required=True)
    audit.add_argument("--expected-checkpoint", type=Path)
    audit.add_argument("--expected-checkpoint-config-sha256")

    approve = commands.add_parser(
        "approve", help="Write a strict v3 perception parent gate from an audited bundle."
    )
    approve.add_argument("--bundle", type=Path, required=True)
    approve.add_argument("--expected-sha256", required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--approved-at", required=True)
    approve.add_argument(
        "--approve-waiver",
        action="append",
        default=[],
        help="Explicitly approve one locked deviation ID.",
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
    path = path.expanduser().resolve()
    print(
        json.dumps(
            {"path": str(path), "sha256": sha256_file(path), **summary},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def _load_pinned_bundle(path: Path, expected_sha256: str) -> tuple[Path, dict[str, Any]]:
    path = path.expanduser().resolve()
    bundle = load_json_pinned(path, expected_sha256, name="perception promotion bundle")
    if not isinstance(bundle, dict):
        raise PromotionValidationError("Perception promotion bundle must be an object")
    return path, bundle


def main(argv: Sequence[str] | None = None) -> None:
    """Run the immutable perception promotion builder, auditor, or human approval signer."""
    args = _parse_args(argv)
    if args.command == "build":
        created_at = args.created_at or datetime.now(timezone.utc).isoformat()
        bundle = build_perception_promotion_bundle(
            checkpoint=args.checkpoint,
            comparator_checkpoint=args.comparator_checkpoint,
            pair_contract=args.pair_contract,
            counterfactual_outcome=args.counterfactual_outcome,
            control_initialization_parity=args.control_initialization_parity,
            treatment_initialization_parity=args.treatment_initialization_parity,
            control_frozen_state=args.control_frozen_state,
            treatment_frozen_state=args.treatment_frozen_state,
            control_text_retention=args.control_text_retention,
            treatment_text_retention=args.treatment_text_retention,
            control_run_health=args.control_run_health,
            treatment_run_health=args.treatment_run_health,
            loss_mass_pair=args.loss_mass_pair,
            created_at=created_at,
        )
        _write_json_once(args.output, bundle)
        persisted_path = args.output.expanduser().resolve()
        persisted = load_json_pinned(
            persisted_path,
            sha256_file(persisted_path),
            name="persisted perception promotion bundle",
        )
        if not isinstance(persisted, dict):
            raise PromotionValidationError("Persisted perception bundle is not an object")
        summary = validate_perception_promotion_bundle(
            persisted, expected_checkpoint=args.checkpoint.expanduser().resolve()
        )
        _print_summary(args.output, summary)
        return

    bundle_path, bundle = _load_pinned_bundle(args.bundle, args.expected_sha256)
    if args.command == "audit":
        summary = validate_perception_promotion_bundle(
            bundle,
            expected_checkpoint=(
                args.expected_checkpoint.expanduser().resolve()
                if args.expected_checkpoint is not None
                else None
            ),
            expected_checkpoint_config_sha256=args.expected_checkpoint_config_sha256,
        )
        _print_summary(bundle_path, summary)
        return

    summary = validate_perception_promotion_bundle(bundle)
    candidate = summary["candidate"]
    if (
        not isinstance(args.approved_by, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@:/+\-]{2,127}", args.approved_by) is None
    ):
        raise PromotionValidationError("--approved-by must be a durable human identity")
    try:
        approved_at = datetime.fromisoformat(args.approved_at.replace("Z", "+00:00"))
        created_at = datetime.fromisoformat(bundle["created_at"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError) as error:
        raise PromotionValidationError("Approval and bundle times must be ISO-8601") from error
    if (
        approved_at.tzinfo is None
        or approved_at.utcoffset() is None
        or created_at.tzinfo is None
        or created_at.utcoffset() is None
        or approved_at < created_at
    ):
        raise PromotionValidationError("Approval must occur after bundle creation")
    if (
        len(args.approve_waiver) != len(REQUIRED_WAIVER_IDS)
        or set(args.approve_waiver) != REQUIRED_WAIVER_IDS
    ):
        raise PromotionValidationError(
            "Approval requires one explicit --approve-waiver for each locked deviation"
        )
    config = load_json_pinned(
        Path(candidate["checkpoint"]) / "config.json",
        candidate["checkpoint_config_sha256"],
        name="approved candidate config",
    )
    metadata = config.get("vision_alignment") if isinstance(config, dict) else None
    if not isinstance(metadata, dict):
        raise PromotionValidationError("Candidate config lacks vision_alignment metadata")
    gate = {
        "format": "vision_alignment_parent_gate",
        "version": 3,
        "status": "approved",
        "promotion_kind": "perception",
        "promotion_policy": PERCEPTION_PROMOTION_POLICY,
        "recipe_version": metadata.get("recipe_version"),
        "formatter_version": metadata.get("formatter_version"),
        "phase": candidate["phase"],
        "checkpoint": candidate["checkpoint"],
        "checkpoint_config_sha256": candidate["checkpoint_config_sha256"],
        "data_contract_sha256": candidate["data_contract_sha256"],
        "trainable_contract_sha256": candidate["trainable_contract_sha256"],
        "global_step": candidate["global_step"],
        "metrics_artifact_sha256": args.expected_sha256,
        "promotion_bundle_path": str(bundle_path),
        "promotion_bundle_sha256": args.expected_sha256,
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
        type(gate["recipe_version"]) is not int
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


if __name__ == "__main__":
    main()
