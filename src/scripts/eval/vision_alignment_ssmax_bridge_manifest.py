"""Materialize one immutable SSMax bridge run manifest after all saved steps exist.

The checked-in per-arm specification owns every path.  This command does not accept checkpoint,
pairing, recipe, or profile substitutions.  It creates the two fixed matched-wrong pairing files
once (or validates their existing bytes), hashes every DCP shard and trainer-rank state at all
seven preregistered steps, and atomically publishes a self-hashed final manifest.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from olmo_core.eval.vision_alignment_ssmax_bridge import (
    build_manifest,
    load_json,
    load_manifest_spec,
    write_json_once,
)
from olmo_core.eval.vision_alignment_ssmax_data import (
    build_validation_datasets,
    create_or_validate_pairing,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-workers", type=int, default=8)
    parser.add_argument("--created-at")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Build fixed pairings and publish one finalized per-model manifest."""

    args = _parse_args(argv)
    if args.hash_workers <= 0:
        raise ValueError("--hash-workers must be positive")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable manifest {output}")
    spec_path = args.spec.expanduser().resolve()
    spec = load_manifest_spec(spec_path)
    checkpoint_root = Path(str(spec["checkpoint_root"])).expanduser().resolve()
    raw_config = load_json(checkpoint_root / "step0" / "config.json")
    if not isinstance(raw_config, Mapping):
        raise ValueError("step0 config must contain an object")
    if (
        raw_config.get("model_variant") != spec["model_variant"]
        or raw_config.get("required_run_name") != spec["run_name"]
        or raw_config.get("phase") != "bridge"
    ):
        raise ValueError("step0 config does not belong to the specified bridge arm")

    validation_path = Path(str(spec["validation"])).expanduser().resolve()
    validation_sha = raw_config.get("evaluation", {}).get("validation_manifest_sha256")
    if not isinstance(validation_sha, str):
        raise ValueError("step0 config does not pin its validation manifest SHA-256")
    _, _, datasets, content_ids, _ = build_validation_datasets(
        raw_config,
        manifest_path=validation_path,
        manifest_sha256=validation_sha,
    )
    pairing_paths = spec["pairing_paths"]
    if not isinstance(pairing_paths, Mapping):
        raise ValueError("Manifest spec pairing_paths must be an object")
    evaluation = spec["evaluation"]
    pairing_references = {
        source: create_or_validate_pairing(
            datasets[source],
            path=Path(str(pairing_paths[source])),
            examples=int(evaluation["examples_per_source"]),
            seed=int(evaluation["pairing_seed"]),
            content_ids=content_ids,
        )
        for source in evaluation["sources"]
    }
    created_at = args.created_at or datetime.now(timezone.utc).isoformat()
    manifest = build_manifest(
        spec,
        spec_path=spec_path,
        pairing_references=pairing_references,
        created_at=created_at,
        hash_workers=args.hash_workers,
    )
    write_json_once(output, manifest)


if __name__ == "__main__":
    main()
