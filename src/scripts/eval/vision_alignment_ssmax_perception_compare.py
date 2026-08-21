"""Build a descriptive QK-vs-no-QK perception comparison from rebuilt versioned reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from olmo_core.eval.vision_alignment_ssmax_perception import (
    build_model_variant_comparison,
    validate_model_variant_comparison,
    write_json_once,
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


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = build_model_variant_comparison(
        left_promotion_report={
            "path": str(args.left_promotion_report.expanduser().resolve()),
            "sha256": args.expected_left_promotion_report_sha256,
        },
        right_promotion_report={
            "path": str(args.right_promotion_report.expanduser().resolve()),
            "sha256": args.expected_right_promotion_report_sha256,
        },
        created_at=args.created_at,
        verify_live_checkpoint=True,
    )
    validate_model_variant_comparison(result)
    output = args.output.expanduser().resolve()
    write_json_once(output, result)
    print(
        json.dumps(
            {"output": str(output), "content_sha256": result["content_sha256"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
