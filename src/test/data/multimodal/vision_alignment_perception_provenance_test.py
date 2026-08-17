"""Tests for perception provenance manifests and selected datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from olmo_core.data.multimodal import (
    vision_alignment_perception_provenance as provenance,
)
from olmo_core.data.multimodal.finevision import FINEVISION_ROOT
from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    PERCEPTION_SOURCE_NAMES,
    PerceptionSplitSelection,
    SelectedVisionAlignmentDataset,
    load_perception_provenance_manifest,
    selected_dataset_fingerprint,
)
from olmo_core.data.multimodal.vision_alignment_perception_sources import (
    VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION,
    VisionAlignmentPerceptionSourceSpec,
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_lines(root: Path, relative_path: str, values: Sequence[Any]) -> dict[str, Any]:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(f"{value}\n".encode() for value in values)
    path.write_bytes(raw)
    return {"path": relative_path, "sha256": _sha256_bytes(raw), "count": len(values)}


def _write_selection(root: Path, relative_path: str, indices: Sequence[int]) -> dict[str, Any]:
    reference = _write_lines(root, relative_path, indices)
    reference["indices_sha256"] = _canonical_sha256(list(indices))
    return reference


def _write_manifest(path: Path, value: Mapping[str, Any], *, update_content: bool = True) -> str:
    payload = deepcopy(dict(value))
    if update_content:
        payload.pop("content_sha256", None)
        payload["content_sha256"] = _canonical_sha256(payload)
    raw = _canonical_bytes(payload) + b"\n"
    path.write_bytes(raw)
    raw_sha256 = _sha256_bytes(raw)
    (path.parent / "COMPLETE").write_text(f"{raw_sha256}\n")
    return raw_sha256


@dataclass
class _ManifestArtifact:
    path: Path
    raw_sha256: str
    value: dict[str, Any]


def _source_spec(root: Path) -> VisionAlignmentPerceptionSourceSpec:
    return VisionAlignmentPerceptionSourceSpec(
        phase="perception",
        pixmo_cap_path=str(root / "pixmo-cap"),
        sequence_length=2560,
        max_crops=8,
        message_format="document",
        loss_token_weighting="root_subsegments_root_tokens",
        caption_prompt="Description:",
        transcript_prompt="Transcript:",
        require_transcript=True,
        finevision_root=FINEVISION_ROOT,
        finevision_visualweb_path=str(root / "finevision-visualweb"),
        finevision_geo170k_path=str(root / "finevision-geo170k"),
        finevision_visualweb_fingerprint="a" * 64,
        finevision_geo170k_fingerprint="b" * 64,
    )


def _build_manifest(tmp_path: Path, *, overlap: bool = False) -> _ManifestArtifact:
    root = tmp_path / "provenance"
    root.mkdir(parents=True)
    source_spec = _source_spec(root)
    source_spec_sha = source_spec.preprocessing_sha256

    upstream = root / "upstream" / provenance._FINEVISION_MATERIALIZATION_MANIFEST
    upstream.parent.mkdir()
    upstream_raw = b'{"synthetic":"finevision-materialization"}\n'
    upstream.write_bytes(upstream_raw)

    sources: dict[str, Any] = {}
    filtering: dict[str, Any] = {}
    unions: dict[str, set[str]] = {"train": set(), "validation": set()}
    first_train_hash: str | None = None
    for source_ordinal, source_name in enumerate(PERCEPTION_SOURCE_NAMES):
        splits: dict[str, Any] = {}
        for logical_split in ("train", "validation"):
            physical_split = (
                "train"
                if logical_split == "train" or source_name == "audited_alignment"
                else "validation"
            )
            if source_name == "audited_alignment":
                base_examples = 514
                indices = (512, 513) if logical_split == "train" else tuple(range(512))
            else:
                base_examples = 2 if logical_split == "train" else 512
                indices = tuple(range(base_examples))
            base_fingerprint = hashlib.sha256(
                f"{source_name}:{physical_split}:base".encode()
            ).hexdigest()[:16]
            row_hashes = [
                _sha256_bytes(f"{source_name}:{logical_split}:image:{index}".encode())
                for index in indices
            ]
            if logical_split == "train" and first_train_hash is None:
                first_train_hash = row_hashes[0]
            if overlap and logical_split == "validation" and source_ordinal == 0:
                assert first_train_hash is not None
                row_hashes[0] = first_train_hash

            prefix = f"sources/{source_name}/{logical_split}"
            selection = _write_selection(root, f"{prefix}/indices.txt", indices)
            unique_hashes = sorted(set(row_hashes))
            splits[logical_split] = {
                "physical_split": physical_split,
                "base_annotation_sha256": _sha256_bytes(
                    f"{source_name}:{physical_split}:annotations".encode()
                ),
                "base_dataset_fingerprint": base_fingerprint,
                "base_examples": base_examples,
                "selection": selection,
                "runtime_dataset_fingerprint": selected_dataset_fingerprint(
                    source_name=source_name,
                    logical_split=logical_split,
                    physical_split=physical_split,
                    base_fingerprint=base_fingerprint,
                    selection_indices_sha256=selection["indices_sha256"],
                    source_spec_sha256=source_spec_sha,
                ),
                "runtime_examples": len(indices),
                "row_image_content": _write_lines(root, f"{prefix}/row-images.txt", row_hashes),
                "unique_image_content": _write_lines(
                    root, f"{prefix}/unique-images.txt", unique_hashes
                ),
            }
            unions[logical_split].update(unique_hashes)

        components = {
            "audited_alignment": ["visualwebinstruct(filtered)", "geo170k(align)"],
            "ocr_document": list(source_spec.ocr_source_names),
        }.get(source_name, [source_name])
        sources[source_name] = {"components": components, **splits}
        candidate = splits["train"]["base_examples"]
        output = splits["train"]["runtime_examples"]
        filtering[source_name] = {
            "candidate_train_examples": candidate,
            "removed_train_examples": candidate - output,
            "output_train_examples": output,
        }

    signature_image = tmp_path / "signature-image.bin"
    signature_image.write_bytes(b"signature image")
    signature = signature_image.stat()
    signature_inventory = _write_lines(
        root,
        "inventories/image-path-signatures.jsonl",
        [
            _canonical_bytes(
                {
                    "path": str(signature_image.resolve()),
                    "size_bytes": signature.st_size,
                    "mtime_ns": signature.st_mtime_ns,
                    "ctime_ns": signature.st_ctime_ns,
                    "inode": signature.st_ino,
                    "device": signature.st_dev,
                    "sha256": _sha256_bytes(signature_image.read_bytes()),
                }
            ).decode()
        ],
    )
    union_refs = {
        f"{split}_unique_image_content": _write_lines(
            root, f"unions/{split}.txt", sorted(unions[split])
        )
        for split in ("train", "validation")
    }
    value: dict[str, Any] = {
        "format": provenance.PERCEPTION_PROVENANCE_FORMAT,
        "version": provenance.PERCEPTION_PROVENANCE_VERSION,
        "status": "verified",
        "phase": "perception",
        "created_at": "2026-08-12T00:00:00Z",
        "builder": {
            "name": "build_vision_alignment_perception_provenance",
            "version": 1,
            "script_sha256": "c" * 64,
        },
        "source_spec": source_spec.as_canonical_dict(),
        "source_spec_sha256": source_spec_sha,
        "source_registry_version": VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION,
        "source_registry_sha256": "d" * 64,
        "source_implementation_inventory": {"recorded": "opaque"},
        "finevision_materialization": {
            "path": f"upstream/{provenance._FINEVISION_MATERIALIZATION_MANIFEST}",
            "sha256": _sha256_bytes(upstream_raw),
            "content_sha256": "e" * 64,
            "visualweb_fingerprint": "a" * 64,
            "geo170k_fingerprint": "b" * 64,
        },
        "image_path_signatures": signature_inventory,
        "validation_selection": {
            "algorithm": "sha256-ranked-distinct-content-representatives-v1",
            "image_contents_per_source": 512,
        },
        "sources": sources,
        "unions": {**union_refs, "overlap_count": 0},
        "filtering": filtering,
    }
    path = root / provenance.PERCEPTION_PROVENANCE_MANIFEST
    raw_sha256 = _write_manifest(path, value)
    return _ManifestArtifact(path, raw_sha256, json.loads(path.read_text()))


def _load(artifact: _ManifestArtifact, *, expected_sha256: str | None = None):
    return load_perception_provenance_manifest(
        artifact.path,
        expected_sha256=artifact.raw_sha256 if expected_sha256 is None else expected_sha256,
        verify_finevision_materialization=False,
    )


def _rewrite(
    artifact: _ManifestArtifact,
    mutate: Callable[[dict[str, Any]], None],
    *,
    update_content: bool = True,
) -> str:
    value = deepcopy(artifact.value)
    mutate(value)
    return _write_manifest(artifact.path, value, update_content=update_content)


def test_loads_manifest_and_exact_selections(tmp_path: Path):
    artifact = _build_manifest(tmp_path)
    loaded = _load(artifact)

    assert len(loaded.selections) == 2 * len(PERCEPTION_SOURCE_NAMES)
    assert loaded.selection("audited_alignment", "train").indices == (512, 513)
    assert len(loaded.selection("pixmo_caption", "validation").indices) == 512
    loaded.validate_image_path_signatures()


@pytest.mark.parametrize("case", ["raw_pin", "content", "complete"])
def test_rejects_manifest_identity_tamper(tmp_path: Path, case: str):
    artifact = _build_manifest(tmp_path)
    if case == "raw_pin":
        expected_sha = "0" * 64
        match = "raw SHA mismatch"
    elif case == "content":
        expected_sha = _rewrite(
            artifact,
            lambda value: value.__setitem__("created_at", "2026-08-13T00:00:00Z"),
            update_content=False,
        )
        match = "content SHA-256 differs"
    else:
        (artifact.path.parent / "COMPLETE").write_text("0" * 64 + "\n")
        expected_sha = artifact.raw_sha256
        match = "COMPLETE"

    with pytest.raises(ValueError, match=match):
        _load(artifact, expected_sha256=expected_sha)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value.__setitem__("extra", True), "fields differ"),
        (
            lambda value: value["sources"]["ocr_document"].__setitem__("components", ["text_vqa"]),
            "components differ",
        ),
        (
            lambda value: value["sources"]["pixmo_caption"]["validation"].__setitem__(
                "physical_split", "train"
            ),
            "must use physical split",
        ),
        (
            lambda value: value["validation_selection"].__setitem__(
                "image_contents_per_source", 511
            ),
            "selection policy differs",
        ),
    ],
)
def test_rejects_schema_and_semantic_drift(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    match: str,
):
    artifact = _build_manifest(tmp_path)
    raw_sha256 = _rewrite(artifact, mutate)

    with pytest.raises(ValueError, match=match):
        _load(artifact, expected_sha256=raw_sha256)


@pytest.mark.parametrize("indices", [(1, 0), (0, 2)])
def test_rejects_invalid_row_selection(tmp_path: Path, indices: tuple[int, ...]):
    artifact = _build_manifest(tmp_path)
    value = deepcopy(artifact.value)
    split = value["sources"]["pixmo_caption"]["train"]
    selection_path = artifact.path.parent / split["selection"]["path"]
    raw = b"".join(f"{index}\n".encode() for index in indices)
    selection_path.write_bytes(raw)
    split["selection"].update(
        sha256=_sha256_bytes(raw),
        count=len(indices),
        indices_sha256=_canonical_sha256(list(indices)),
    )
    raw_sha256 = _write_manifest(artifact.path, value)

    with pytest.raises(ValueError, match="sorted unique, in-bounds"):
        _load(artifact, expected_sha256=raw_sha256)


@pytest.mark.parametrize("case", ["file_sha", "path_escape"])
def test_rejects_inventory_sha_and_path_tamper(tmp_path: Path, case: str):
    artifact = _build_manifest(tmp_path)
    reference = artifact.value["unions"]["train_unique_image_content"]
    if case == "file_sha":
        (artifact.path.parent / reference["path"]).write_text("0" * 64 + "\n")
        raw_sha256 = artifact.raw_sha256
        match = "SHA-256 mismatch"
    else:
        raw_sha256 = _rewrite(
            artifact,
            lambda value: value["unions"]["train_unique_image_content"].__setitem__(
                "path", "../outside.txt"
            ),
        )
        match = "escapes the provenance artifact root"

    with pytest.raises(ValueError, match=match):
        _load(artifact, expected_sha256=raw_sha256)


@pytest.mark.parametrize("case", ["mismatch", "overlap"])
def test_rejects_union_inventory_drift(tmp_path: Path, case: str):
    artifact = _build_manifest(tmp_path, overlap=case == "overlap")
    raw_sha256 = artifact.raw_sha256
    match = "unions overlap"
    if case == "mismatch":
        value = deepcopy(artifact.value)
        reference = value["unions"]["train_unique_image_content"]
        path = artifact.path.parent / reference["path"]
        rows = path.read_text().splitlines()[:-1]
        raw = b"".join(f"{row}\n".encode() for row in rows)
        path.write_bytes(raw)
        reference.update(sha256=_sha256_bytes(raw), count=len(rows))
        raw_sha256 = _write_manifest(artifact.path, value)
        match = "union inventories differ"

    with pytest.raises(ValueError, match=match):
        _load(artifact, expected_sha256=raw_sha256)


class _RawImageDataset:
    def __init__(self, images: Sequence[Path], fingerprint: str, annotation_sha256: str):
        self.images = tuple(images)
        self.content_fingerprint = fingerprint
        self.annotation_sha256 = annotation_sha256

    def __len__(self) -> int:
        return len(self.images)

    def get(self, index: int, epoch: int = 0) -> dict[str, int]:
        return {"index": index, "epoch": epoch}

    def raw_image_references(self, index: int):
        return (str(self.images[index]),)

    def annotation_content_sha256(self):
        return self.annotation_sha256


def test_selected_dataset_binds_rows_annotations_and_image_bytes(tmp_path: Path):
    images = [tmp_path / "0.img", tmp_path / "1.img"]
    images[0].write_bytes(b"zero")
    images[1].write_bytes(b"one")
    base_fingerprint = "0123456789abcdef"
    annotation_sha = "2" * 64
    indices_sha = _canonical_sha256([0, 1])
    selection = PerceptionSplitSelection(
        physical_split="train",
        base_annotation_sha256=annotation_sha,
        base_dataset_fingerprint=base_fingerprint,
        base_examples=2,
        indices=(0, 1),
        selection_indices_sha256=indices_sha,
        runtime_dataset_fingerprint=selected_dataset_fingerprint(
            source_name="pixmo_caption",
            logical_split="train",
            physical_split="train",
            base_fingerprint=base_fingerprint,
            selection_indices_sha256=indices_sha,
            source_spec_sha256="1" * 64,
        ),
        row_image_content_sha256=(_sha256_bytes(b"zero"), _sha256_bytes(b"one")),
        unique_image_content_sha256=tuple(sorted((_sha256_bytes(b"zero"), _sha256_bytes(b"one")))),
    )
    raw = _RawImageDataset(images, base_fingerprint, annotation_sha)
    selected = SelectedVisionAlignmentDataset(
        raw,
        source_name="pixmo_caption",
        logical_split="train",
        selection=selection,
    )

    assert selected.get(1, 7) == {"index": 1, "epoch": 7}
    assert selected.content_fingerprint == selection.runtime_dataset_fingerprint
    assert len(selected.validate_image_content()) == 64

    images[1].write_bytes(b"mutated")
    with pytest.raises(ValueError, match="image bytes differ"):
        selected.validate_image_content()
