"""Validate, merge, and compare fixed-query SSMax attention diagnostics.

Model loading is intentionally kept in the native checkpoint runner: exact text parents require a
strict LM-prefix bridge load, while later Vision Alignment saves require generic HSDP resharding.
Both runners feed the same :class:`SSMaxAttentionDiagnosticsCollector` integration API and emit
bounded rank-local state JSON. This command performs the checkpoint-independent, auditable merge
and comparison steps.

Examples::

    python src/scripts/eval/ssmax_attention_diagnostics.py validate-manifest \
        --manifest probe.json --expected-sha256 SHA256

    python src/scripts/eval/ssmax_attention_diagnostics.py finalize \
        --manifest probe.json --expected-manifest-sha256 SHA256 \
        --state rank0.json --state rank1.json \
        --checkpoint-identity checkpoint-identity.json --output report.json

    python src/scripts/eval/ssmax_attention_diagnostics.py compare \
        --baseline parent-report.json --candidate step500-report.json --output delta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from olmo_core.eval.ssmax_attention_diagnostics import (
    SSMaxAttentionDiagnosticsCollector,
    SSMaxProbeManifest,
    compare_ssmax_attention_reports,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--expected-sha256", required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--manifest", required=True)
    finalize.add_argument("--expected-manifest-sha256", required=True)
    finalize.add_argument("--state", action="append", required=True)
    finalize.add_argument("--checkpoint-identity", required=True)
    finalize.add_argument("--output", required=True)
    finalize.add_argument("--overwrite", action="store_true")

    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--overwrite", action="store_true")
    compare.add_argument("--entropy-drop-threshold", type=float, default=0.10)
    compare.add_argument("--effective-context-fraction-ratio-threshold", type=float, default=0.50)
    compare.add_argument("--absolute-logit-q99-ratio-threshold", type=float, default=2.0)
    compare.add_argument("--q-magnitude-ratio-threshold", type=float, default=2.0)
    return parser.parse_args(argv)


def _load_mapping(path_value: str) -> Mapping[str, Any]:
    path = Path(path_value).expanduser().resolve()
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not decode JSON artifact {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact {path} must contain a mapping")
    return payload


def _write_json(path_value: str, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    path = Path(path_value).expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "validate-manifest":
        manifest = SSMaxProbeManifest.load(
            args.manifest,
            expected_sha256=args.expected_sha256,
            verify_validation_manifest=True,
        )
        print(
            json.dumps(
                {
                    "manifest": str(Path(args.manifest).expanduser().resolve()),
                    "sha256": manifest.sha256,
                    "rows": len(manifest.rows_by_sample_id),
                    "valid": True,
                },
                sort_keys=True,
            )
        )
        return

    if args.command == "finalize":
        manifest = SSMaxProbeManifest.load(
            args.manifest,
            expected_sha256=args.expected_manifest_sha256,
            verify_validation_manifest=True,
        )
        states = [_load_mapping(path) for path in args.state]
        checkpoint_identity = _load_mapping(args.checkpoint_identity)
        report = SSMaxAttentionDiagnosticsCollector.finalize_states(
            manifest,
            states,
            checkpoint_identity=checkpoint_identity,
        )
        _write_json(args.output, report, overwrite=args.overwrite)
        return

    if args.command == "compare":
        comparison = compare_ssmax_attention_reports(
            _load_mapping(args.baseline),
            _load_mapping(args.candidate),
            entropy_drop_threshold=args.entropy_drop_threshold,
            effective_context_fraction_ratio_threshold=(
                args.effective_context_fraction_ratio_threshold
            ),
            absolute_logit_q99_ratio_threshold=args.absolute_logit_q99_ratio_threshold,
            q_magnitude_ratio_threshold=args.q_magnitude_ratio_threshold,
        )
        _write_json(args.output, comparison, overwrite=args.overwrite)
        return
    raise AssertionError(f"Unhandled command {args.command}")


if __name__ == "__main__":
    main()
