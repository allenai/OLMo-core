"""Compare two validated SSMax joint trajectories without declaring a winner."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from olmo_core.eval.vision_alignment_ssmax_joint import (
    compare_trajectory_reports,
    validate_trajectory_report,
    write_json_once,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    left = validate_trajectory_report(args.left.expanduser().resolve())
    right = validate_trajectory_report(args.right.expanduser().resolve())
    result = compare_trajectory_reports(left, right)
    write_json_once(args.output.expanduser().resolve(), result)


if __name__ == "__main__":
    main()
