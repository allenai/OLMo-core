"""Create a hash-pinned step0-to-candidate SSMax downstream trajectory comparison."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from olmo_core.eval.vision_alignment_ssmax_downstream import load_and_compare_trajectory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-qknorm-result", required=True)
    parser.add_argument("--expected-baseline-qknorm-sha256", required=True)
    parser.add_argument("--baseline-no-qknorm-result", required=True)
    parser.add_argument("--expected-baseline-no-qknorm-sha256", required=True)
    parser.add_argument("--candidate-qknorm-result", required=True)
    parser.add_argument("--expected-candidate-qknorm-sha256", required=True)
    parser.add_argument("--candidate-no-qknorm-result", required=True)
    parser.add_argument("--expected-candidate-no-qknorm-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    comparison = load_and_compare_trajectory(
        args.baseline_qknorm_result,
        args.baseline_no_qknorm_result,
        args.candidate_qknorm_result,
        args.candidate_no_qknorm_result,
        expected_baseline_qknorm_sha256=args.expected_baseline_qknorm_sha256,
        expected_baseline_no_qknorm_sha256=args.expected_baseline_no_qknorm_sha256,
        expected_candidate_qknorm_sha256=args.expected_candidate_qknorm_sha256,
        expected_candidate_no_qknorm_sha256=args.expected_candidate_no_qknorm_sha256,
    )
    comparison["created_at"] = datetime.now(timezone.utc).isoformat()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite downstream trajectory comparison {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
