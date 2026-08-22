"""Create fixed pairings and finalize one immutable direct SSMax perception manifest.

The concrete reviewed specification selects exactly one vision-unfrozen run.  Finalization is
refused until that run has permanent checkpoints at every protocol step.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from olmo_core.data.multimodal.ssmax_single_response import (
    SSMAX_SINGLE_RESPONSE_PROJECTION_SEED,
    SSMaxSingleResponseDataset,
)
from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    build_selected_perception_dataset,
    load_perception_provenance_manifest,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    load_pinned_vision_alignment_tokenizer,
)
from olmo_core.eval.vision_alignment_ssmax_data import create_or_validate_pairing
from olmo_core.eval.vision_alignment_ssmax_perception_direct import (
    REQUIRED_STEPS,
    SOURCES,
    build_manifest,
    load_json,
    load_manifest_spec,
    write_json_once,
)
from scripts.eval import vision_alignment_ssmax_perception as paired_runner


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-workers", type=int, default=8)
    parser.add_argument("--created-at")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.hash_workers <= 0:
        raise ValueError("--hash-workers must be positive")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable manifest {output}")
    spec = load_manifest_spec(args.spec.expanduser().resolve())
    checkpoint_root = Path(str(spec["checkpoint_root"])).expanduser().resolve()
    for step in REQUIRED_STEPS:
        if not (checkpoint_root / f"step{step}").is_dir():
            raise FileNotFoundError(
                f"Completed direct perception checkpoint step{step} is required before finalization"
            )
    raw_config = load_json(checkpoint_root / "step0" / "config.json")
    if not isinstance(raw_config, Mapping):
        raise TypeError("Direct step0 config must contain an object")
    artifacts = raw_config["artifacts"]
    tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
        identifier=artifacts["tokenizer_id"],
        revision=artifacts["tokenizer_revision"],
        expected_fingerprint=artifacts["tokenizer_fingerprint"],
        cache_dir=artifacts["hf_cache_dir"],
    )
    provenance = load_perception_provenance_manifest(
        spec["perception_provenance"],
        expected_sha256=raw_config["data"]["perception_provenance_sha256"],
        verify_finevision_materialization=False,
        load_image_path_signatures=False,
    )
    projection = raw_config["data"].get("ssmax_single_response_projection")
    if not isinstance(projection, Mapping):
        raise TypeError("Direct checkpoint lacks SSMax single-response projection")
    projection_seed = projection.get("seed")
    if projection_seed != SSMAX_SINGLE_RESPONSE_PROJECTION_SEED:
        raise ValueError("Direct checkpoint has a non-canonical projection seed")
    loss_token_weighting = raw_config["data"].get("loss_token_weighting")
    datasets = {
        source: paired_runner._ModelInputDataset(
            SSMaxSingleResponseDataset(
                build_selected_perception_dataset(
                    provenance,
                    tokenizer,
                    token_ids,
                    source,
                    logical_split="validation",
                    validate_required_annotations=True,
                    verify_finevision_materialization=False,
                ),
                source_name=source,
                logical_split="validation",
                seed=projection_seed,
                loss_token_weighting=loss_token_weighting,
            ),
        )
        for source in SOURCES
    }
    examples = int(spec["evaluation"]["examples_per_source"])
    seed = int(spec["evaluation"]["pairing_seed"])
    for source in SOURCES:
        create_or_validate_pairing(
            datasets[source],
            path=Path(str(spec["pairing_paths"][source])),
            examples=examples,
            seed=seed,
            content_ids=provenance.selection(source, "validation").row_image_content_sha256,
        )
    manifest = build_manifest(
        spec,
        created_at=args.created_at or datetime.now(timezone.utc).isoformat(),
        hash_workers=args.hash_workers,
    )
    write_json_once(output, manifest)


if __name__ == "__main__":
    main()
