"""Build exact-geometry SSMax joint pairings aligned to the fixed 16-rank evidence world."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from olmo_core.data.multimodal.ssmax_single_response import SSMaxSingleResponseDataset
from olmo_core.data.multimodal.vision_alignment_joint_provenance import (
    build_selected_joint_dataset,
    load_joint_visual_projection_manifest,
)
from olmo_core.data.multimodal.vision_alignment_joint_sources import (
    JOINT_VISUAL_SOURCE_NAMES,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
    VISION_ALIGNMENT_TOKENIZER_ID,
    VISION_ALIGNMENT_TOKENIZER_REVISION,
    load_pinned_vision_alignment_tokenizer,
)
from olmo_core.eval.matched_wrong_image import (
    build_matched_wrong_image_pairing,
    matched_wrong_image_pairing_sha256,
    serialize_matched_wrong_image_pairing,
)
from olmo_core.eval.vision_alignment_ssmax_joint import (
    ELIGIBLE_VISUAL_ROWS,
    VISUAL_EXAMPLES_PER_SOURCE,
)

DEFAULT_PROJECTION = (
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/artifacts/"
    "joint-visual-projection-v1/vision-alignment-joint-visual-projection.json"
)
DEFAULT_PROJECTION_SHA256 = "11c1df56d7fbc270a9eff999193476c0c578c6964017d217a320b3d39305a730"
DEFAULT_CACHE = "/weka/oe-training-default/rustin/hf-cache/hub"


class _ModelInputProjection:
    """Remove source metadata before the generic pairing descriptor is computed."""

    fields = frozenset(
        {
            "input_ids",
            "labels",
            "loss_masks",
            "position_ids",
            "token_type_ids",
            "images",
            "pooled_patches_idx",
        }
    )

    def __init__(self, dataset: object, *, source: str):
        self.dataset = dataset
        self.source = source

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def get(self, index: int, epoch: int = 0) -> dict[str, object]:
        getter = getattr(self.dataset, "get", None)
        row = getter(index, epoch) if callable(getter) else self.dataset[index]  # type: ignore[index]
        if not isinstance(row, Mapping):
            raise TypeError(f"{self.source} validation row {index} is not an object")
        missing = self.fields - set(row)
        unknown = set(row) - self.fields - {"metadata"}
        if missing or unknown:
            raise ValueError(
                f"{self.source} validation row {index} fields differ: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return {field: row[field] for field in sorted(self.fields)}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-projection", default=DEFAULT_PROJECTION)
    parser.add_argument("--expected-joint-projection-sha256", default=DEFAULT_PROJECTION_SHA256)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--examples-per-source", type=int, default=VISUAL_EXAMPLES_PER_SOURCE)
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--pairing-seed", type=int, default=6198)
    parser.add_argument("--projection-seed", type=int, default=95818)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if (
        args.world_size != 16
        or args.examples_per_source != VISUAL_EXAMPLES_PER_SOURCE
        or args.examples_per_source % args.world_size
        or args.pairing_seed != 6198
        or args.projection_seed != 95818
    ):
        raise ValueError(
            "SSMax joint evidence requires 496 rows over 16 ranks, pairing seed 6198, "
            "and projection seed 95818"
        )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to replace pairing artifacts in {output_dir}")
    tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
        identifier=VISION_ALIGNMENT_TOKENIZER_ID,
        revision=VISION_ALIGNMENT_TOKENIZER_REVISION,
        expected_fingerprint=VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
        cache_dir=args.hf_cache_dir,
    )
    projection = load_joint_visual_projection_manifest(
        args.joint_projection,
        expected_token_ids=token_ids,
        expected_sha256=args.expected_joint_projection_sha256,
        verify_finevision_materialization=False,
        load_image_path_signatures=False,
    )
    payloads = {}
    for source in JOINT_VISUAL_SOURCE_NAMES:
        dataset = build_selected_joint_dataset(
            projection,
            tokenizer,
            token_ids,
            source,
            logical_split="validation",
            validate_required_annotations=True,
        )
        dataset = SSMaxSingleResponseDataset(
            dataset,
            source_name=source,
            logical_split="validation",
            seed=args.projection_seed,
            loss_token_weighting="root_subsegments_root_tokens",
        )
        content_ids = projection.selection(source, "validation").row_image_content_sha256
        payloads[source] = build_matched_wrong_image_pairing(
            _ModelInputProjection(dataset, source=source),
            recipient_count=args.examples_per_source,
            seed=args.pairing_seed,
            content_ids=content_ids,
            epoch=0,
        )
        if payloads[source]["coverage"]["eligible_count"] != ELIGIBLE_VISUAL_ROWS[source]:
            raise ValueError(f"{source} exact-geometry eligibility differs from the live pin")
        print(
            json.dumps(
                {
                    "source": source,
                    "eligible_rows": payloads[source]["coverage"]["eligible_count"],
                    "selected_rows": args.examples_per_source,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    pairings = {}
    for source, payload in payloads.items():
        path = output_dir / f"{source}.json"
        raw = serialize_matched_wrong_image_pairing(payload)
        with path.open("xb") as file_handle:
            file_handle.write(raw)
        pairings[source] = {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "canonical_sha256": matched_wrong_image_pairing_sha256(payload),
        }
    manifest = {
        "format": "vision_alignment_ssmax_joint_pairings",
        "version": 1,
        "projection": {
            "path": str(projection.path),
            "sha256": projection.raw_sha256,
            "content_sha256": projection.content_sha256,
        },
        "examples_per_source": args.examples_per_source,
        "eligible_rows_per_source": dict(ELIGIBLE_VISUAL_ROWS),
        "world_size": args.world_size,
        "pairing_seed": args.pairing_seed,
        "projection_seed": args.projection_seed,
        "pairings": pairings,
    }
    manifest["content_sha256"] = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with (output_dir / "manifest.json").open("x") as file_handle:
        json.dump(manifest, file_handle, indent=2, sort_keys=True, allow_nan=False)
        file_handle.write("\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
