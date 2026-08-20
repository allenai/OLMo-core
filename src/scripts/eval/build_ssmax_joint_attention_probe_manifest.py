"""Freeze the fixed SSMax attention probe over the pinned joint PixMo-caption projection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from olmo_core.data.multimodal.ssmax_single_response import SSMaxSingleResponseDataset
from olmo_core.data.multimodal.vision_alignment_joint_provenance import (
    build_selected_joint_dataset,
    load_joint_visual_projection_manifest,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
    VISION_ALIGNMENT_TOKENIZER_ID,
    VISION_ALIGNMENT_TOKENIZER_REVISION,
    load_pinned_vision_alignment_tokenizer,
)
from olmo_core.eval.ssmax_attention_diagnostics import (
    ProbeSequence,
    SSMaxProbeManifest,
    build_probe_manifest,
    serialize_probe_manifest,
)

DEFAULT_PROJECTION = (
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/artifacts/"
    "joint-visual-projection-v1/vision-alignment-joint-visual-projection.json"
)
DEFAULT_PROJECTION_SHA256 = "11c1df56d7fbc270a9eff999193476c0c578c6964017d217a320b3d39305a730"
DEFAULT_CACHE = "/weka/oe-training-default/rustin/hf-cache/hub"
PROJECTION_SEED = 95818


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-projection", default=DEFAULT_PROJECTION)
    parser.add_argument("--expected-joint-projection-sha256", default=DEFAULT_PROJECTION_SHA256)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hf-cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--max-queries-per-category-per-row", type=int, default=8)
    return parser.parse_args(argv)


def _selected_indices(content_ids: Sequence[str], *, rows: int, seed: int) -> list[int]:
    if rows <= 0 or rows > len(content_ids) or seed < 0:
        raise ValueError("row count must fit the projection and seed must be non-negative")
    ranked = []
    for index, content_id in enumerate(content_ids):
        priority = hashlib.sha256(
            f"ssmax-joint-probe-row-v1\0{seed}\0pixmo_caption\0{index}\0{content_id}".encode()
        ).digest()
        ranked.append((priority, index))
    ranked.sort()
    return sorted(index for _, index in ranked[:rows])


def _example(dataset: Any, index: int) -> Mapping[str, Any]:
    getter = getattr(dataset, "get", None)
    value = getter(index, 0) if callable(getter) else dataset[index]
    if not isinstance(value, Mapping):
        raise TypeError(f"joint probe row{index} is not an object")
    return value


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing joint probe {output}")
    projection_path = Path(args.joint_projection).expanduser().resolve()
    tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
        identifier=VISION_ALIGNMENT_TOKENIZER_ID,
        revision=VISION_ALIGNMENT_TOKENIZER_REVISION,
        expected_fingerprint=VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
        cache_dir=args.hf_cache_dir,
    )
    projection = load_joint_visual_projection_manifest(
        projection_path,
        expected_token_ids=token_ids,
        expected_sha256=args.expected_joint_projection_sha256,
        verify_finevision_materialization=False,
        load_image_path_signatures=False,
    )
    dataset = SSMaxSingleResponseDataset(
        build_selected_joint_dataset(
            projection,
            tokenizer,
            token_ids,
            "pixmo_caption",
            logical_split="validation",
            validate_required_annotations=True,
        ),
        source_name="pixmo_caption",
        logical_split="validation",
        seed=PROJECTION_SEED,
        loss_token_weighting="root_subsegments_root_tokens",
    )
    selection = projection.selection("pixmo_caption", "validation")
    content_ids = selection.row_image_content_sha256
    if len(content_ids) != len(dataset):
        raise RuntimeError("joint projection content IDs and runtime rows differ")
    indices = _selected_indices(content_ids, rows=args.rows, seed=args.seed)
    sequences = []
    for index in indices:
        row = _example(dataset, index)
        input_ids = torch.as_tensor(row["input_ids"])
        sequences.append(
            ProbeSequence(
                sample_id=f"pixmo_caption:{index}:{content_ids[index]}",
                dataset_index=index,
                input_ids=input_ids,
                token_type_ids=torch.as_tensor(row["token_type_ids"]),
                loss_masks=torch.as_tensor(row["loss_masks"]),
                valid_tokens=torch.ones_like(input_ids, dtype=torch.bool),
            )
        )
    manifest = build_probe_manifest(
        sequences,
        validation_manifest_path=projection_path,
        validation_manifest_sha256=projection.raw_sha256,
        seed=args.seed,
        max_queries_per_category_per_row=args.max_queries_per_category_per_row,
    )
    payload = manifest.as_dict()
    payload["population"] = {
        "source": "pixmo_caption",
        "split": "validation",
        "epoch": 0,
        "row_selection_algorithm": "sha256-priority-over-joint-content-id-v1",
        "row_selection_seed": args.seed,
        "selected_dataset_indices": indices,
        "selected_content_ids": [content_ids[index] for index in indices],
        "joint_projection": {
            "path": str(projection.path),
            "sha256": projection.raw_sha256,
            "content_sha256": projection.content_sha256,
            "source_spec_sha256": projection.source_spec_sha256,
            "runtime_dataset_fingerprint": selection.runtime_dataset_fingerprint,
            "selection_indices_sha256": selection.selection_indices_sha256,
            "examples": len(dataset),
        },
        "tokenizer": {
            "id": VISION_ALIGNMENT_TOKENIZER_ID,
            "revision": VISION_ALIGNMENT_TOKENIZER_REVISION,
            "fingerprint": VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
            "token_ids": token_ids.as_config_dict(),
        },
    }
    manifest = SSMaxProbeManifest.from_dict(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file_handle:
        file_handle.write(serialize_probe_manifest(manifest))
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": manifest.sha256,
                "rows": len(indices),
                "projection_sha256": projection.raw_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
