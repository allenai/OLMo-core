"""Finalize an immutable SSMax joint manifest after all permanent checkpoints exist."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from olmo_core.eval.vision_alignment_ssmax_joint import (
    build_manifest,
    load_manifest_spec,
    write_json_once,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--created-at")
    parser.add_argument("--hash-workers", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    spec_path = args.spec.expanduser().resolve()
    spec = load_manifest_spec(spec_path)
    manifest = build_manifest(
        spec,
        spec_path=spec_path,
        created_at=args.created_at or datetime.now(timezone.utc).isoformat(),
        hash_workers=args.hash_workers,
    )
    write_json_once(args.output.expanduser().resolve(), manifest)


if __name__ == "__main__":
    main()
