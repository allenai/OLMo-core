"""Create a hash-pinned paired comparison of the two SSMax downstream-fast results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from olmo_core.eval.vision_alignment_ssmax_downstream import load_and_compare_results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qknorm-result", required=True)
    parser.add_argument("--expected-qknorm-result-sha256", required=True)
    parser.add_argument("--no-qknorm-result", required=True)
    parser.add_argument("--expected-no-qknorm-result-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    comparison = load_and_compare_results(
        args.qknorm_result,
        args.no_qknorm_result,
        expected_left_sha256=args.expected_qknorm_result_sha256,
        expected_right_sha256=args.expected_no_qknorm_result_sha256,
    )
    comparison["created_at"] = datetime.now(timezone.utc).isoformat()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite downstream comparison {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
