"""Adversarial contracts for the joint visual projection runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from olmo_core.data.multimodal.finevision import FINEVISION_ROOT
from olmo_core.data.multimodal.vision_alignment_joint_provenance import (
    JOINT_VISUAL_PROJECTION_FORMAT,
    JOINT_VISUAL_PROJECTION_MANIFEST,
    JOINT_VISUAL_PROJECTION_VERSION,
    JointVisualSplitProjection,
    SelectedVisionAlignmentJointDataset,
    build_selected_joint_dataset,
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
    vision_alignment_joint_implementation_inventory,
    vision_alignment_joint_source_registry_sha256,
)
from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    PERCEPTION_PROVENANCE_MANIFEST,
    FineVisionMaterializationReference,
    PerceptionProvenanceManifest,
    PerceptionSplitSelection,
    image_reference_sha256,
)
from olmo_core.data.multimodal.vision_alignment_perception_sources import (
    VisionAlignmentPerceptionSourceSpec,
    vision_alignment_perception_source_registry_sha256,
)
from olmo_core.nn.vision import Molmo2TokenIds


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
    root: Path, *, overlap: bool = False
) -> tuple[PerceptionProvenanceManifest, bytes, dict[str, Any]]:
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
    parent_root = {
        "source_registry_sha256": vision_alignment_perception_source_registry_sha256(),
        "sources": raw_sources,
    }
    parent_raw = json.dumps(parent_root, sort_keys=True, indent=2).encode()
    parent_path = root / PERCEPTION_PROVENANCE_MANIFEST
    parent_path.write_bytes(parent_raw)
    parent_content_sha = _digest("parent-content")
    parent = PerceptionProvenanceManifest(
        path=parent_path,
        raw_sha256=hashlib.sha256(parent_raw).hexdigest(),
        content_sha256=parent_content_sha,
        source_spec=spec,
        source_spec_sha256=spec.preprocessing_sha256,
        finevision_materialization=FineVisionMaterializationReference(
            path=root / "finevision.json",
            sha256="c" * 64,
            content_sha256="d" * 64,
            visualweb_fingerprint="a" * 64,
            geo170k_fingerprint="b" * 64,
        ),
        image_path_signatures=(),
        selections=selections,
    )
    return parent, parent_raw, parent_root


def _artifact_root(
    parent: PerceptionProvenanceManifest,
    parent_root: dict[str, Any],
    builder_path: Path,
) -> dict[str, Any]:
    joint_spec = VisionAlignmentJointSourceSpec.from_perception(parent.source_spec)
    token_ids = Molmo2TokenIds()
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
                    token_ids,
                    source_name,
                    split=parent_selection.physical_split,
                )
            )
            source[logical_split] = {
                "physical_split": parent_selection.physical_split,
                "base_examples": parent_selection.base_examples,
                "joint_base_dataset_fingerprint": base_fingerprint,
                "joint_base_annotation_sha256": _digest(f"{source_name}/{logical_split}/joint-ann"),
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
            "script_sha256": hashlib.sha256(builder_path.read_bytes()).hexdigest(),
        },
        "parent_perception_provenance": {
            "path": str(parent.path.resolve()),
            "sha256": parent.raw_sha256,
            "content_sha256": parent.content_sha256,
            "source_spec_sha256": parent.source_spec_sha256,
            "source_registry_sha256": parent_root["source_registry_sha256"],
        },
        "source_name_projection": dict(JOINT_TO_PERCEPTION_SOURCE),
        "source_spec": joint_spec.as_canonical_dict(),
        "source_spec_sha256": joint_spec.preprocessing_sha256,
        "source_registry_version": VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION,
        "source_registry_sha256": vision_alignment_joint_source_registry_sha256(),
        "source_implementation_inventory": vision_alignment_joint_implementation_inventory(),
        "projection_policy": {
            "algorithm": "exact-parent-logical-row-selection-v1",
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
            "overlap_count": len(set(train_union).intersection(validation_union)),
        },
    }


def _write_artifact(path: Path, root: dict[str, Any], *, update_content: bool = True) -> str:
    if update_content:
        root.pop("content_sha256", None)
        root["content_sha256"] = _canonical_sha256(root)
    raw = json.dumps(root, sort_keys=True, indent=2).encode()
    path.write_bytes(raw)
    raw_sha = hashlib.sha256(raw).hexdigest()
    (path.parent / "COMPLETE").write_text(f"{raw_sha}\n")
    return raw_sha


@pytest.fixture
def projection_artifact(tmp_path: Path, monkeypatch):
    from olmo_core.data.multimodal import (
        vision_alignment_joint_provenance as provenance,
    )

    builder_path = tmp_path / "build_vision_alignment_joint_projection.py"
    builder_path.write_text("# reviewed builder\n")
    parent, _, parent_root = _parent_fixture(tmp_path)
    artifact_root = _artifact_root(parent, parent_root, builder_path)
    artifact_path = tmp_path / JOINT_VISUAL_PROJECTION_MANIFEST
    raw_sha = _write_artifact(artifact_path, artifact_root)

    def load_parent(path, *, expected_sha256=None, **_kwargs):
        assert Path(path) == parent.path
        assert expected_sha256 == parent.raw_sha256
        return parent

    monkeypatch.setattr(provenance, "_builder_script_path", lambda: builder_path)
    monkeypatch.setattr(provenance, "load_perception_provenance_manifest", load_parent)
    return artifact_path, raw_sha, artifact_root, parent, builder_path


def test_loads_exact_parent_rows_and_512_validation_population(projection_artifact):
    artifact_path, raw_sha, _, parent, _ = projection_artifact
    loaded = load_joint_visual_projection_manifest(artifact_path, expected_sha256=raw_sha)

    assert loaded.raw_sha256 == raw_sha
    assert loaded.parent_provenance is parent
    assert loaded.source_spec.sequence_length == 8192
    assert set(loaded.selections) == {
        (source_name, split)
        for source_name in JOINT_VISUAL_SOURCE_NAMES
        for split in ("train", "validation")
    }
    count_selection = loaded.selection("count_numeric", "validation")
    parent_selection = parent.selection("scalar_count", "validation")
    assert count_selection.indices == parent_selection.indices
    assert count_selection.selection_indices_sha256 == parent_selection.selection_indices_sha256
    assert len(count_selection.indices) == 512
    assert len(count_selection.unique_image_content_sha256) == 512


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda root: root.update(extra=True), "fields differ"),
        (
            lambda root: root["source_name_projection"].update(count_numeric="count_numeric"),
            "projection differs",
        ),
        (
            lambda root: root["sources"]["count_numeric"]["train"].update(
                selection_indices_sha256="0" * 64
            ),
            "parent selection differs",
        ),
        (
            lambda root: root["sources"]["count_numeric"]["train"].update(base_examples=4),
            "base example count differs",
        ),
        (
            lambda root: root["sources"]["count_numeric"]["train"].update(
                adapter_projection_sha256="0" * 64
            ),
            "adapter projection differs",
        ),
        (
            lambda root: root["sources"]["pixmo_caption"]["validation"].update(
                unique_image_content_sha256="0" * 64
            ),
            "image inventories differ",
        ),
        (
            lambda root: root.update(source_registry_sha256="0" * 64),
            "implementation identity differs",
        ),
        (
            lambda root: root["source_spec"].update(parent_perception_source_registry_version=True),
            "source specification differs",
        ),
        (
            lambda root: root["source_implementation_inventory"].update(version=True),
            "implementation identity differs",
        ),
    ],
)
def test_rejects_schema_mapping_selection_inventory_and_registry_mutation(
    projection_artifact, mutate, match
):
    artifact_path, _, root, _, _ = projection_artifact
    mutate(root)
    _write_artifact(artifact_path, root)
    with pytest.raises(ValueError, match=match):
        load_joint_visual_projection_manifest(artifact_path)


def test_rejects_raw_content_parent_pin_duplicate_key_and_path_traversal(projection_artifact):
    artifact_path, raw_sha, root, parent, _ = projection_artifact
    with pytest.raises(ValueError, match="raw SHA mismatch"):
        load_joint_visual_projection_manifest(artifact_path, expected_sha256="0" * 64)

    original_spec_sha = root["source_spec_sha256"]
    root["source_spec_sha256"] = "0" * 64
    _write_artifact(artifact_path, root, update_content=False)
    with pytest.raises(ValueError, match="content SHA-256 differs"):
        load_joint_visual_projection_manifest(artifact_path)

    root["source_spec_sha256"] = original_spec_sha
    root["parent_perception_provenance"]["sha256"] = "0" * 64
    _write_artifact(artifact_path, root)
    with pytest.raises(ValueError, match="raw SHA-256 differs"):
        load_joint_visual_projection_manifest(artifact_path)

    root["parent_perception_provenance"]["sha256"] = parent.raw_sha256
    root["parent_perception_provenance"]["path"] = f"../{PERCEPTION_PROVENANCE_MANIFEST}"
    _write_artifact(artifact_path, root)
    with pytest.raises(ValueError, match="path traversal"):
        load_joint_visual_projection_manifest(artifact_path)

    valid_raw = artifact_path.read_text().replace(
        f'"format": "{JOINT_VISUAL_PROJECTION_FORMAT}"',
        f'"format": "{JOINT_VISUAL_PROJECTION_FORMAT}", "format": "duplicate"',
        1,
    )
    artifact_path.write_text(valid_raw)
    duplicate_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    (artifact_path.parent / "COMPLETE").write_text(f"{duplicate_sha}\n")
    with pytest.raises(ValueError, match="repeats key"):
        load_joint_visual_projection_manifest(artifact_path)
    assert len(raw_sha) == 64


@pytest.mark.parametrize(
    "field",
    ["content_sha256", "source_spec_sha256", "source_registry_sha256"],
)
def test_rejects_each_parent_identity_reference(projection_artifact, field):
    artifact_path, _, root, _, _ = projection_artifact
    root["parent_perception_provenance"][field] = "0" * 64
    _write_artifact(artifact_path, root)
    with pytest.raises(ValueError, match="reference differs"):
        load_joint_visual_projection_manifest(artifact_path)


def test_rejects_builder_and_joint_registry_current_code_drift(projection_artifact, monkeypatch):
    from olmo_core.data.multimodal import (
        vision_alignment_joint_provenance as provenance,
    )

    artifact_path, _, root, _, builder_path = projection_artifact
    builder_path.write_text("# drifted builder\n")
    with pytest.raises(ValueError, match="builder identity or bytes differ"):
        load_joint_visual_projection_manifest(artifact_path)

    builder_path.write_text("# reviewed builder\n")
    monkeypatch.setattr(
        provenance, "vision_alignment_joint_source_registry_sha256", lambda: "0" * 64
    )
    with pytest.raises(ValueError, match="implementation identity differs"):
        load_joint_visual_projection_manifest(artifact_path)
    assert root["source_registry_sha256"] != "0" * 64


def test_rejects_union_overlap_even_when_artifact_reports_it(tmp_path: Path, monkeypatch):
    from olmo_core.data.multimodal import (
        vision_alignment_joint_provenance as provenance,
    )

    builder_path = tmp_path / "build_vision_alignment_joint_projection.py"
    builder_path.write_text("# reviewed builder\n")
    parent, _, parent_root = _parent_fixture(tmp_path, overlap=True)
    root = _artifact_root(parent, parent_root, builder_path)
    root["unions"]["overlap_count"] = 0
    artifact_path = tmp_path / JOINT_VISUAL_PROJECTION_MANIFEST
    _write_artifact(artifact_path, root)
    monkeypatch.setattr(provenance, "_builder_script_path", lambda: builder_path)
    monkeypatch.setattr(
        provenance,
        "load_perception_provenance_manifest",
        lambda *_args, **_kwargs: parent,
    )

    with pytest.raises(ValueError, match="disjoint parent unions"):
        load_joint_visual_projection_manifest(artifact_path)


class _RawDataset:
    def __init__(self, selection: JointVisualSplitProjection):
        self.config = build_vision_alignment_joint_dataset_config(
            VisionAlignmentJointSourceSpec.from_perception(_spec()),
            Molmo2TokenIds(),
            "pixmo_caption",
            split="train",
        )
        self.content_fingerprint = selection.joint_base_dataset_fingerprint
        self._annotation = selection.joint_base_annotation_sha256
        self.images = [b"zero", b"one", b"two"]
        self.annotation_validated = False

    def __len__(self):
        return 3

    def get(self, index, epoch=0):
        return {"raw_index": index, "epoch": epoch}

    def raw_image_references(self, index):
        return (self.images[index],)

    def annotation_content_sha256(self):
        return self._annotation

    def validate_required_annotations(self):
        self.annotation_validated = True


def _wrapper_selection() -> JointVisualSplitProjection:
    config = build_vision_alignment_joint_dataset_config(
        VisionAlignmentJointSourceSpec.from_perception(_spec()),
        Molmo2TokenIds(),
        "pixmo_caption",
        split="train",
    )
    return JointVisualSplitProjection(
        physical_split="train",
        base_examples=3,
        joint_base_dataset_fingerprint=_digest("joint-base"),
        joint_base_annotation_sha256=_digest("joint-annotation"),
        adapter_projection_sha256=vision_alignment_joint_adapter_projection_sha256(config),
        indices=(0, 2),
        selection_indices_sha256=_canonical_sha256([0, 2]),
        runtime_dataset_fingerprint=_digest("joint-selected"),
        row_image_content_sha256=(
            image_reference_sha256(b"zero"),
            image_reference_sha256(b"two"),
        ),
        unique_image_content_sha256=tuple(
            sorted((image_reference_sha256(b"zero"), image_reference_sha256(b"two")))
        ),
    )


def test_selected_wrapper_preserves_indices_fingerprint_and_image_checks():
    selection = _wrapper_selection()
    raw = _RawDataset(selection)
    selected = SelectedVisionAlignmentJointDataset(
        raw,
        source_name="pixmo_caption",
        logical_split="train",
        selection=selection,
    )

    assert len(selected) == 2
    assert selected.get(1, epoch=7) == {"raw_index": 2, "epoch": 7}
    assert selected.content_fingerprint == selection.runtime_dataset_fingerprint
    assert len(selected.validate_image_content()) == 64
    selected.validate_required_annotations()
    assert raw.annotation_validated is True
    for invalid_index in (-1, True, 2):
        with pytest.raises(IndexError, match="out of bounds"):
            selected.get(invalid_index)
        with pytest.raises(IndexError, match="out of bounds"):
            selected.raw_image_references(invalid_index)
    raw.images[2] = b"changed"
    with pytest.raises(ValueError, match="image bytes differ"):
        selected.validate_image_content([1])


@pytest.mark.parametrize("drift", ["base", "annotation", "adapter"])
def test_selected_wrapper_rejects_base_annotation_and_adapter_drift(drift):
    selection = _wrapper_selection()
    raw = _RawDataset(selection)
    if drift == "base":
        raw.content_fingerprint = "0" * 64
    elif drift == "annotation":
        raw._annotation = "0" * 64
    else:
        raw.config.max_sequence_length = 2560
    with pytest.raises(ValueError, match="dataset differs|adapter differs"):
        SelectedVisionAlignmentJointDataset(
            raw,
            source_name="pixmo_caption",
            logical_split="train",
            selection=selection,
        )


def test_build_selected_joint_dataset_and_unknown_names(projection_artifact, monkeypatch):
    from olmo_core.data.multimodal import (
        vision_alignment_joint_provenance as provenance,
    )

    artifact_path, _, _, _, _ = projection_artifact
    manifest = load_joint_visual_projection_manifest(artifact_path)
    selection = manifest.selection("pixmo_caption", "train")
    raw = _RawDataset(selection)
    monkeypatch.setattr(
        provenance,
        "build_vision_alignment_joint_dataset",
        lambda *_args, **_kwargs: raw,
    )
    selected = build_selected_joint_dataset(
        manifest,
        tokenizer=object(),
        token_ids=Molmo2TokenIds(),
        source_name="pixmo_caption",
        logical_split="train",
    )
    assert selected.indices == selection.indices
    with pytest.raises(ValueError, match="Unknown joint visual source"):
        manifest.selection("scalar_count", "train")
    with pytest.raises(ValueError, match="Unknown joint logical split"):
        manifest.selection("pixmo_caption", "test")


def test_validation_count_cannot_be_projected_below_512(projection_artifact):
    artifact_path, _, root, parent, _ = projection_artifact
    parent_selection = parent.selection("pixmo_caption", "validation")
    parent.selections[("pixmo_caption", "validation")] = replace(
        parent_selection,
        indices=parent_selection.indices[:-1],
        selection_indices_sha256=_canonical_sha256(list(parent_selection.indices[:-1])),
        row_image_content_sha256=parent_selection.row_image_content_sha256[:-1],
        unique_image_content_sha256=parent_selection.unique_image_content_sha256[:-1],
    )
    split = root["sources"]["pixmo_caption"]["validation"]
    split["selection_indices_sha256"] = parent.selection(
        "pixmo_caption", "validation"
    ).selection_indices_sha256
    split["runtime_examples"] = 511
    split["row_image_content_sha256"] = _canonical_sha256(
        list(parent.selection("pixmo_caption", "validation").row_image_content_sha256)
    )
    split["unique_image_content_sha256"] = _canonical_sha256(
        list(parent.selection("pixmo_caption", "validation").unique_image_content_sha256)
    )
    split["runtime_dataset_fingerprint"] = joint_selected_dataset_fingerprint(
        source_name="pixmo_caption",
        parent_source_name="pixmo_caption",
        logical_split="validation",
        physical_split="validation",
        joint_base_fingerprint=split["joint_base_dataset_fingerprint"],
        selection_indices_sha256=split["selection_indices_sha256"],
        joint_source_spec_sha256=root["source_spec_sha256"],
        parent_provenance_sha256=parent.raw_sha256,
        parent_provenance_content_sha256=parent.content_sha256,
    )
    _write_artifact(artifact_path, root)
    with pytest.raises(ValueError, match="exactly 512"):
        load_joint_visual_projection_manifest(artifact_path)
