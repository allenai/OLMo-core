"""Tests for joint visual projection provenance and selected datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from olmo_core.data.multimodal import vision_alignment_joint_provenance as provenance
from olmo_core.data.multimodal.finevision import FINEVISION_ROOT
from olmo_core.data.multimodal.vision_alignment_joint_provenance import (
    JOINT_VISUAL_PROJECTION_ALGORITHM,
    JOINT_VISUAL_PROJECTION_FORMAT,
    JOINT_VISUAL_PROJECTION_MANIFEST,
    JOINT_VISUAL_PROJECTION_VERSION,
    JointVisualSplitProjection,
    SelectedVisionAlignmentJointDataset,
    joint_selected_dataset_fingerprint,
    load_joint_visual_projection_manifest,
)
from olmo_core.data.multimodal.vision_alignment_joint_sources import (
    JOINT_TO_PERCEPTION_SOURCE,
    JOINT_VISUAL_SOURCE_NAMES,
    VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION,
    VisionAlignmentJointSourceSpec,
    build_vision_alignment_joint_dataset_config,
    vision_alignment_joint_adapter_projection_sha256,
)
from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    PERCEPTION_PROVENANCE_MANIFEST,
    PerceptionProvenanceManifest,
    PerceptionSplitSelection,
    image_reference_sha256,
)
from olmo_core.data.multimodal.vision_alignment_perception_sources import (
    VisionAlignmentPerceptionSourceSpec,
)
from olmo_core.nn.vision import Molmo2TokenIds

_TOKEN_IDS = Molmo2TokenIds(
    im_start_id=100278,
    im_end_id=100279,
    im_patch_id=100280,
    im_col_id=100281,
    low_res_im_start_id=100282,
    image_placeholder_id=100283,
    im_end_turn_id=100265,
)


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _spec() -> VisionAlignmentPerceptionSourceSpec:
    return VisionAlignmentPerceptionSourceSpec(
        phase="perception",
        pixmo_cap_path="/reviewed/pixmo-cap",
        sequence_length=2560,
        max_crops=8,
        message_format="document",
        loss_token_weighting="root_subsegments_root_tokens",
        caption_prompt="Description:",
        transcript_prompt="Transcript:",
        require_transcript=True,
        finevision_root=FINEVISION_ROOT,
        finevision_visualweb_path="/reviewed/finevision/visualweb",
        finevision_geo170k_path="/reviewed/finevision/geo170k",
        finevision_visualweb_fingerprint="a" * 64,
        finevision_geo170k_fingerprint="b" * 64,
    )


def _components(parent_source_name: str, spec: VisionAlignmentPerceptionSourceSpec) -> list[str]:
    if parent_source_name == "audited_alignment":
        return ["visualwebinstruct(filtered)", "geo170k(align)"]
    if parent_source_name == "ocr_document":
        return list(spec.ocr_source_names)
    return [parent_source_name]


def _parent_fixture(
    root: Path,
    *,
    overlap: bool = False,
) -> tuple[PerceptionProvenanceManifest, dict[str, Any]]:
    spec = _spec()
    selections: dict[tuple[str, str], PerceptionSplitSelection] = {}
    raw_sources: dict[str, Any] = {}
    for joint_name in JOINT_VISUAL_SOURCE_NAMES:
        parent_name = JOINT_TO_PERCEPTION_SOURCE[joint_name]
        raw_sources[parent_name] = {"components": _components(parent_name, spec)}
        for logical_split in ("train", "validation"):
            count = 3 if logical_split == "train" else 512
            indices = tuple(range(count))
            image_domain = "train" if overlap and logical_split == "validation" else logical_split
            row_hashes = tuple(
                _digest(f"{parent_name}/{image_domain}/image/{index}") for index in indices
            )
            selections[(parent_name, logical_split)] = PerceptionSplitSelection(
                physical_split=(
                    "train"
                    if logical_split == "train" or parent_name == "audited_alignment"
                    else "validation"
                ),
                base_annotation_sha256=_digest(f"{parent_name}/{logical_split}/parent-ann"),
                base_dataset_fingerprint=_digest(f"{parent_name}/{logical_split}/parent-base"),
                base_examples=count,
                indices=indices,
                selection_indices_sha256=_canonical_sha256(list(indices)),
                runtime_dataset_fingerprint=_digest(
                    f"{parent_name}/{logical_split}/parent-selected"
                ),
                row_image_content_sha256=row_hashes,
                unique_image_content_sha256=tuple(sorted(row_hashes)),
            )

    parent_root = {"source_registry_sha256": "c" * 64, "sources": raw_sources}
    parent_raw = json.dumps(parent_root, sort_keys=True, indent=2).encode()
    parent_path = root / PERCEPTION_PROVENANCE_MANIFEST
    parent_path.write_bytes(parent_raw)
    parent = PerceptionProvenanceManifest(
        path=parent_path,
        raw_sha256=hashlib.sha256(parent_raw).hexdigest(),
        content_sha256=_digest("parent-content"),
        source_spec=spec,
        source_spec_sha256=spec.preprocessing_sha256,
        finevision_materialization=None,
        image_path_signatures=(),
        selections=selections,
    )
    return parent, parent_root


def _artifact_root(
    parent: PerceptionProvenanceManifest,
    parent_root: dict[str, Any],
) -> dict[str, Any]:
    joint_spec = VisionAlignmentJointSourceSpec.from_perception(parent.source_spec)
    sources: dict[str, Any] = {}
    for source_name in JOINT_VISUAL_SOURCE_NAMES:
        parent_name = JOINT_TO_PERCEPTION_SOURCE[source_name]
        source: dict[str, Any] = {
            "parent_source_name": parent_name,
            "components": parent_root["sources"][parent_name]["components"],
        }
        for logical_split in ("train", "validation"):
            parent_selection = parent.selection(parent_name, logical_split)
            base_fingerprint = _digest(f"{source_name}/{logical_split}/joint-base")
            adapter_sha = vision_alignment_joint_adapter_projection_sha256(
                build_vision_alignment_joint_dataset_config(
                    joint_spec,
                    _TOKEN_IDS,
                    source_name,
                    split=parent_selection.physical_split,
                )
            )
            source[logical_split] = {
                "physical_split": parent_selection.physical_split,
                "base_examples": parent_selection.base_examples,
                "joint_base_dataset_fingerprint": base_fingerprint,
                "joint_base_annotation_sha256": (
                    _digest(f"{source_name}/{logical_split}/joint-ann")
                    if source_name in {"audited_alignment", "ocr_document"}
                    else parent_selection.base_annotation_sha256
                ),
                "adapter_projection_sha256": adapter_sha,
                "selection_indices_sha256": parent_selection.selection_indices_sha256,
                "runtime_examples": len(parent_selection.indices),
                "row_image_content_sha256": _canonical_sha256(
                    list(parent_selection.row_image_content_sha256)
                ),
                "unique_image_content_sha256": _canonical_sha256(
                    list(parent_selection.unique_image_content_sha256)
                ),
                "runtime_dataset_fingerprint": joint_selected_dataset_fingerprint(
                    source_name=source_name,
                    parent_source_name=parent_name,
                    logical_split=logical_split,
                    physical_split=parent_selection.physical_split,
                    joint_base_fingerprint=base_fingerprint,
                    selection_indices_sha256=parent_selection.selection_indices_sha256,
                    joint_source_spec_sha256=joint_spec.preprocessing_sha256,
                    parent_provenance_sha256=parent.raw_sha256,
                    parent_provenance_content_sha256=parent.content_sha256,
                ),
            }
        sources[source_name] = source

    train_union = sorted(
        {
            value
            for source_name in JOINT_VISUAL_SOURCE_NAMES
            for value in parent.selection(
                JOINT_TO_PERCEPTION_SOURCE[source_name], "train"
            ).unique_image_content_sha256
        }
    )
    validation_union = sorted(
        {
            value
            for source_name in JOINT_VISUAL_SOURCE_NAMES
            for value in parent.selection(
                JOINT_TO_PERCEPTION_SOURCE[source_name], "validation"
            ).unique_image_content_sha256
        }
    )
    return {
        "format": JOINT_VISUAL_PROJECTION_FORMAT,
        "version": JOINT_VISUAL_PROJECTION_VERSION,
        "status": "verified",
        "phase": "joint",
        "created_at": "2026-08-13T22:00:00Z",
        "builder": {
            "name": "build_vision_alignment_joint_projection",
            "version": 1,
            "script_sha256": "f" * 64,
        },
        "parent_perception_provenance": {
            "path": str(parent.path.resolve()),
            "sha256": parent.raw_sha256,
            "content_sha256": parent.content_sha256,
            "source_spec_sha256": parent.source_spec_sha256,
            "source_registry_sha256": parent_root["source_registry_sha256"],
        },
        "token_ids": asdict(_TOKEN_IDS),
        "source_name_projection": dict(JOINT_TO_PERCEPTION_SOURCE),
        "source_spec": joint_spec.as_canonical_dict(),
        "source_spec_sha256": joint_spec.preprocessing_sha256,
        "source_registry_version": VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION,
        "source_registry_sha256": "1" * 64,
        "source_implementation_inventory": {"recorded": "opaque"},
        "projection_policy": {
            "algorithm": JOINT_VISUAL_PROJECTION_ALGORITHM,
            "parent_sequence_length": 2560,
            "sequence_length": 8192,
            "allowed_adapter_config_delta": ["max_sequence_length"],
        },
        "sources": sources,
        "unions": {
            "train_unique_image_content_sha256": _canonical_sha256(train_union),
            "train_count": len(train_union),
            "validation_unique_image_content_sha256": _canonical_sha256(validation_union),
            "validation_count": len(validation_union),
            "overlap_count": 0,
        },
    }


def _write_artifact(path: Path, root: dict[str, Any], *, update_content: bool = True) -> str:
    if update_content:
        root.pop("content_sha256", None)
        root["content_sha256"] = _canonical_sha256(root)
    raw = json.dumps(root, sort_keys=True, indent=2).encode()
    path.write_bytes(raw)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    (path.parent / "COMPLETE").write_text(f"{raw_sha256}\n")
    return raw_sha256


def _make_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    overlap: bool = False,
) -> SimpleNamespace:
    parent, parent_root = _parent_fixture(tmp_path, overlap=overlap)
    root = _artifact_root(parent, parent_root)
    path = tmp_path / JOINT_VISUAL_PROJECTION_MANIFEST
    raw_sha256 = _write_artifact(path, root)

    def load_parent(path, *, expected_sha256=None, **_kwargs):
        assert Path(path) == parent.path
        assert expected_sha256 == parent.raw_sha256
        return parent

    monkeypatch.setattr(provenance, "load_perception_provenance_manifest", load_parent)
    return SimpleNamespace(path=path, raw_sha256=raw_sha256, root=root, parent=parent)


def _load(artifact: SimpleNamespace, *, expected_sha256: str | None = None):
    return load_joint_visual_projection_manifest(
        artifact.path,
        expected_token_ids=_TOKEN_IDS,
        expected_sha256=artifact.raw_sha256 if expected_sha256 is None else expected_sha256,
    )


def test_loads_exact_parent_rows_and_validation_population(tmp_path: Path, monkeypatch):
    artifact = _make_projection(tmp_path, monkeypatch)
    loaded = _load(artifact)

    count_selection = loaded.selection("count_numeric", "validation")
    parent_selection = artifact.parent.selection("scalar_count", "validation")
    assert count_selection.indices == parent_selection.indices


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda root: root.update(extra=True), "fields differ"),
        (
            lambda root: root["sources"]["count_numeric"]["train"].update(
                selection_indices_sha256="0" * 64
            ),
            "parent selection differs",
        ),
        (
            lambda root: root["sources"]["pixmo_caption"]["validation"].update(
                unique_image_content_sha256="0" * 64
            ),
            "image inventories differ",
        ),
        (
            lambda root: root["projection_policy"].update(sequence_length=4096),
            "policy differs",
        ),
    ],
)
def test_rejects_schema_selection_inventory_and_policy_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any]], None],
    match: str,
):
    artifact = _make_projection(tmp_path, monkeypatch)
    mutate(artifact.root)
    raw_sha256 = _write_artifact(artifact.path, artifact.root)

    with pytest.raises(ValueError, match=match):
        _load(artifact, expected_sha256=raw_sha256)


@pytest.mark.parametrize("case", ["raw_pin", "content", "parent_sha", "path"])
def test_rejects_raw_content_parent_sha_and_path_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
):
    artifact = _make_projection(tmp_path, monkeypatch)
    if case == "raw_pin":
        raw_sha256 = "0" * 64
        match = "raw SHA mismatch"
    elif case == "content":
        artifact.root["created_at"] = "2026-08-14T22:00:00Z"
        raw_sha256 = _write_artifact(artifact.path, artifact.root, update_content=False)
        match = "content SHA-256 differs"
    elif case == "parent_sha":
        artifact.root["parent_perception_provenance"]["sha256"] = "0" * 64
        raw_sha256 = _write_artifact(artifact.path, artifact.root)
        match = "raw SHA-256 differs"
    else:
        artifact.root["parent_perception_provenance"][
            "path"
        ] = f"../{PERCEPTION_PROVENANCE_MANIFEST}"
        raw_sha256 = _write_artifact(artifact.path, artifact.root)
        match = "path traversal"

    with pytest.raises(ValueError, match=match):
        _load(artifact, expected_sha256=raw_sha256)


def test_rejects_parent_train_validation_overlap(tmp_path: Path, monkeypatch):
    artifact = _make_projection(tmp_path, monkeypatch, overlap=True)

    with pytest.raises(ValueError, match="disjoint parent unions"):
        _load(artifact)


class _RawDataset:
    def __init__(self, selection: JointVisualSplitProjection):
        self.config = build_vision_alignment_joint_dataset_config(
            VisionAlignmentJointSourceSpec.from_perception(_spec()),
            _TOKEN_IDS,
            "pixmo_caption",
            split="train",
        )
        self.content_fingerprint = selection.joint_base_dataset_fingerprint
        self.annotation_sha256 = selection.joint_base_annotation_sha256
        self.images = [b"zero", b"one", b"two"]

    def __len__(self):
        return 3

    def get(self, index, epoch=0):
        return {"raw_index": index, "epoch": epoch}

    def raw_image_references(self, index):
        return (self.images[index],)

    def annotation_content_sha256(self):
        return self.annotation_sha256


def _wrapper_selection() -> JointVisualSplitProjection:
    config = build_vision_alignment_joint_dataset_config(
        VisionAlignmentJointSourceSpec.from_perception(_spec()),
        _TOKEN_IDS,
        "pixmo_caption",
        split="train",
    )
    image_hashes = (image_reference_sha256(b"zero"), image_reference_sha256(b"two"))
    return JointVisualSplitProjection(
        physical_split="train",
        base_examples=3,
        joint_base_dataset_fingerprint=_digest("joint-base"),
        joint_base_annotation_sha256=_digest("joint-annotation"),
        adapter_projection_sha256=vision_alignment_joint_adapter_projection_sha256(config),
        indices=(0, 2),
        selection_indices_sha256=_canonical_sha256([0, 2]),
        runtime_dataset_fingerprint=_digest("joint-selected"),
        row_image_content_sha256=image_hashes,
        unique_image_content_sha256=tuple(sorted(image_hashes)),
    )


def test_selected_wrapper_binds_rows_fingerprint_and_images():
    selection = _wrapper_selection()
    raw = _RawDataset(selection)
    selected = SelectedVisionAlignmentJointDataset(
        raw,
        source_name="pixmo_caption",
        logical_split="train",
        selection=selection,
        parent_annotation_sha256=selection.joint_base_annotation_sha256,
    )

    assert selected.get(1, epoch=7) == {"raw_index": 2, "epoch": 7}
    assert selected.content_fingerprint == selection.runtime_dataset_fingerprint
    assert len(selected.validate_image_content()) == 64
    raw.images[2] = b"changed"
    with pytest.raises(ValueError, match="image bytes differ"):
        selected.validate_image_content([1])
