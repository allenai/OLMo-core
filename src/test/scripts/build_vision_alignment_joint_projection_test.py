"""Adversarial end-to-end tests for the joint visual projection builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from copy import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from olmo_core.data.multimodal.finevision import FINEVISION_ROOT
from olmo_core.data.multimodal.vision_alignment_joint_provenance import (
    JOINT_VISUAL_PROJECTION_MANIFEST,
    load_joint_visual_projection_manifest,
)
from olmo_core.data.multimodal.vision_alignment_joint_sources import (
    JOINT_TO_PERCEPTION_SOURCE,
    JOINT_VISUAL_SOURCE_NAMES,
    VisionAlignmentJointSourceSpec,
    build_vision_alignment_joint_dataset_config,
)
from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    PERCEPTION_PROVENANCE_MANIFEST,
    FineVisionMaterializationReference,
    PerceptionProvenanceManifest,
    PerceptionSplitSelection,
)
from olmo_core.data.multimodal.vision_alignment_perception_sources import (
    VisionAlignmentPerceptionSourceSpec,
    vision_alignment_perception_source_registry_sha256,
)
from olmo_core.nn.vision import Molmo2TokenIds

_TEST_TOKEN_IDS = Molmo2TokenIds(
    im_start_id=100278,
    im_end_id=100279,
    im_patch_id=100280,
    im_col_id=100281,
    low_res_im_start_id=100282,
    image_placeholder_id=100283,
    im_end_turn_id=100265,
)


def _load_builder():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "data"
        / "build_vision_alignment_joint_projection.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_build_vision_alignment_joint_projection_test_module",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _digest(value: str | bytes) -> str:
    raw = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


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


class _RawDataset:
    def __init__(self, *, images: list[bytes], config: Any, identity: str):
        self.images = images
        self.config = config
        self.content_fingerprint = _digest(f"{identity}/joint-base")
        self.annotation_sha256 = _digest(f"{identity}/annotation")
        self.parent_annotation_sha256 = self.annotation_sha256
        self.joint_annotation_sha256 = self.annotation_sha256
        self.validation_calls = 0
        self.on_image_read: Callable[[int], None] | None = None

    def __len__(self):
        return len(self.images)

    def raw_image_references(self, index):
        if self.on_image_read is not None:
            self.on_image_read(index)
        return (self.images[index],)

    def validate_required_annotations(self):
        self.validation_calls += 1

    def annotation_content_sha256(self):
        return self.annotation_sha256


def _fixture(
    tmp_path: Path,
    *,
    overlap: bool = False,
) -> tuple[
    PerceptionProvenanceManifest,
    dict[tuple[str, str], _RawDataset],
    bytes,
]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    spec = _spec()
    joint_spec = VisionAlignmentJointSourceSpec.from_perception(spec)
    token_ids = _TEST_TOKEN_IDS
    datasets: dict[tuple[str, str], _RawDataset] = {}
    selections: dict[tuple[str, str], PerceptionSplitSelection] = {}
    raw_sources: dict[str, Any] = {}
    for source_name in JOINT_VISUAL_SOURCE_NAMES:
        parent_source = JOINT_TO_PERCEPTION_SOURCE[source_name]
        raw_sources[parent_source] = {"components": _components(parent_source, spec)}
        physical_splits = (
            ("train",)
            if source_name == "audited_alignment"
            else (
                "train",
                "validation",
            )
        )
        for physical_split in physical_splits:
            if source_name == "audited_alignment":
                images = [f"{source_name}/train/{index}".encode() for index in range(2)] + [
                    f"{source_name}/validation/{index}".encode() for index in range(512)
                ]
            else:
                count = 2 if physical_split == "train" else 512
                images = [
                    f"{source_name}/{physical_split}/{index}".encode() for index in range(count)
                ]
            config = build_vision_alignment_joint_dataset_config(
                joint_spec,
                token_ids,
                source_name,
                split=physical_split,
            )
            datasets[(source_name, physical_split)] = _RawDataset(
                images=images,
                config=config,
                identity=f"{source_name}/{physical_split}",
            )
        for logical_split in ("train", "validation"):
            physical_split = (
                "train"
                if logical_split == "train" or source_name == "audited_alignment"
                else "validation"
            )
            dataset = datasets[(source_name, physical_split)]
            indices = (
                tuple(range(2, 514))
                if source_name == "audited_alignment" and logical_split == "validation"
                else tuple(range(len(dataset)))
            )
            if source_name == "audited_alignment" and logical_split == "train":
                indices = (0, 1)
            rows = tuple(_digest(dataset.images[index]) for index in indices)
            if overlap and source_name == "audited_alignment" and logical_split == "validation":
                rows = (_digest(datasets[(source_name, "train")].images[0]), *rows[1:])
                dataset.images[indices[0]] = datasets[(source_name, "train")].images[0]
            selections[(parent_source, logical_split)] = PerceptionSplitSelection(
                physical_split=physical_split,
                base_annotation_sha256=(
                    _digest(f"{source_name}/{physical_split}/parent-annotation")
                    if source_name in {"audited_alignment", "ocr_document"}
                    else dataset.annotation_sha256
                ),
                base_dataset_fingerprint=_digest(
                    f"{parent_source}/{physical_split}/perception-base"
                ),
                base_examples=len(dataset),
                indices=indices,
                selection_indices_sha256=_canonical_sha256(list(indices)),
                runtime_dataset_fingerprint=_digest(
                    f"{parent_source}/{logical_split}/perception-selected"
                ),
                row_image_content_sha256=rows,
                unique_image_content_sha256=tuple(sorted(set(rows))),
            )
    parent_root = {
        "source_registry_sha256": vision_alignment_perception_source_registry_sha256(),
        "sources": raw_sources,
    }
    parent_raw = (json.dumps(parent_root, sort_keys=True, indent=2) + "\n").encode()
    parent_path = tmp_path / PERCEPTION_PROVENANCE_MANIFEST
    parent_path.write_bytes(parent_raw)
    raw_sha = hashlib.sha256(parent_raw).hexdigest()
    (tmp_path / "COMPLETE").write_text(f"{raw_sha}\n")
    parent = PerceptionProvenanceManifest(
        path=parent_path,
        raw_sha256=raw_sha,
        content_sha256=_digest("parent-content"),
        source_spec=spec,
        source_spec_sha256=spec.preprocessing_sha256,
        finevision_materialization=FineVisionMaterializationReference(
            path=tmp_path / "finevision.json",
            sha256="c" * 64,
            content_sha256="d" * 64,
            visualweb_fingerprint="a" * 64,
            geo170k_fingerprint="b" * 64,
        ),
        image_path_signatures=(),
        selections=selections,
    )
    return parent, datasets, parent_raw


def _patch_parent_and_datasets(module, monkeypatch, parent, datasets):
    from olmo_core.data.multimodal import (
        vision_alignment_joint_provenance as provenance,
    )

    def load_parent(path, *, expected_sha256=None, **_kwargs):
        assert Path(path).resolve() == parent.path
        assert expected_sha256 == parent.raw_sha256
        return parent

    monkeypatch.setattr(module, "load_perception_provenance_manifest", load_parent)
    monkeypatch.setattr(provenance, "load_perception_provenance_manifest", load_parent)
    monkeypatch.setattr(
        module,
        "build_vision_alignment_joint_dataset",
        lambda _spec, _tokenizer, _token_ids, source_name, *, split, **_kwargs: datasets[
            (source_name, split)
        ],
    )
    parent_datasets = {}
    for (source_name, split), joint_dataset in datasets.items():
        parent_source_name = JOINT_TO_PERCEPTION_SOURCE[source_name]
        parent_dataset = copy(joint_dataset)
        parent_dataset.config = replace(joint_dataset.config, max_sequence_length=2560)
        parent_selection = parent.selection(parent_source_name, "train")
        if parent_selection.physical_split != split:
            parent_selection = parent.selection(parent_source_name, "validation")
        parent_dataset.content_fingerprint = parent_selection.base_dataset_fingerprint
        parent_dataset.annotation_sha256 = parent_selection.base_annotation_sha256
        parent_dataset.parent_annotation_sha256 = parent_selection.base_annotation_sha256
        parent_dataset.joint_annotation_sha256 = joint_dataset.annotation_sha256
        joint_dataset.parent_annotation_sha256 = parent_selection.base_annotation_sha256
        joint_dataset.joint_annotation_sha256 = joint_dataset.annotation_sha256
        parent_datasets[(parent_source_name, split)] = parent_dataset
    monkeypatch.setattr(
        module,
        "build_vision_alignment_perception_dataset",
        lambda _spec, _tokenizer, _token_ids, source_name, *, split, **_kwargs: (
            parent_datasets[(source_name, split)]
        ),
    )
    monkeypatch.setattr(
        module,
        "vision_alignment_joint_annotation_replay_sha256",
        lambda dataset, source_name, *, sequence_length: (
            dataset.parent_annotation_sha256
            if sequence_length == module.PARENT_SEQUENCE_LENGTH
            else dataset.joint_annotation_sha256
        ),
    )
    return parent_datasets


def _run(module, tmp_path, monkeypatch, *, overlap=False):
    parent, datasets, parent_raw = _fixture(tmp_path, overlap=overlap)
    _patch_parent_and_datasets(module, monkeypatch, parent, datasets)
    output = tmp_path / "joint-projection"
    manifest = module.build_vision_alignment_joint_projection(
        parent_perception_provenance=parent.path,
        expected_parent_perception_sha256=parent.raw_sha256,
        output_dir=output,
        tokenizer=object(),
        token_ids=_TEST_TOKEN_IDS,
        created_at="2026-08-13T23:00:00Z",
    )
    return manifest, output, parent, datasets, parent_raw


def test_builder_replays_exact_rows_and_runtime_loader_accepts(tmp_path, monkeypatch):
    module = _load_builder()
    manifest, output, parent, datasets, _ = _run(module, tmp_path, monkeypatch)
    manifest_path = output / JOINT_VISUAL_PROJECTION_MANIFEST
    raw_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    loaded = load_joint_visual_projection_manifest(
        manifest_path,
        expected_token_ids=_TEST_TOKEN_IDS,
        expected_sha256=raw_sha,
    )

    assert loaded.parent_provenance is parent
    assert loaded.token_ids == _TEST_TOKEN_IDS
    with pytest.raises(ValueError, match="token IDs differ"):
        load_joint_visual_projection_manifest(
            manifest_path,
            expected_token_ids=Molmo2TokenIds(),
            expected_sha256=raw_sha,
        )
    assert (
        loaded.selection("count_numeric", "validation").indices
        == parent.selection("scalar_count", "validation").indices
    )
    assert manifest["created_at"] == "2026-08-13T23:00:00Z"
    assert (
        manifest["builder"]["script_sha256"]
        == hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    )
    assert manifest["projection_policy"]["allowed_adapter_config_delta"] == ["max_sequence_length"]
    assert (output / "COMPLETE").read_text() == f"{raw_sha}\n"
    assert not tuple(tmp_path.glob(".joint-projection.building-*"))
    assert all(dataset.validation_calls == 2 for dataset in datasets.values())


def test_builder_allows_only_sequence_bound_to_change_full_annotation_identity(
    tmp_path, monkeypatch
):
    module = _load_builder()
    parent, datasets, _ = _fixture(tmp_path)
    parent_selection = parent.selection("audited_alignment", "train")
    parent.selections[("audited_alignment", "train")] = replace(
        parent_selection,
        base_annotation_sha256=_digest("audited-alignment/parent-2560-annotation"),
    )
    validation_selection = parent.selection("audited_alignment", "validation")
    parent.selections[("audited_alignment", "validation")] = replace(
        validation_selection,
        base_annotation_sha256=_digest("audited-alignment/parent-2560-annotation"),
    )
    _patch_parent_and_datasets(module, monkeypatch, parent, datasets)

    manifest = module.build_vision_alignment_joint_projection(
        parent_perception_provenance=parent.path,
        expected_parent_perception_sha256=parent.raw_sha256,
        output_dir=tmp_path / "joint-projection-sequence-bound",
        tokenizer=object(),
        token_ids=_TEST_TOKEN_IDS,
        created_at="2026-08-13T23:00:00Z",
    )

    assert (
        manifest["sources"]["audited_alignment"]["train"]["joint_base_annotation_sha256"]
        == datasets[("audited_alignment", "train")].annotation_sha256
    )


def test_builder_rejects_a_single_cross_length_annotation_corner_drift(tmp_path, monkeypatch):
    module = _load_builder()
    parent, datasets, _ = _fixture(tmp_path)
    _patch_parent_and_datasets(module, monkeypatch, parent, datasets)
    joint_dataset = datasets[("audited_alignment", "train")]
    original_replay = module.vision_alignment_joint_annotation_replay_sha256

    def drifted_replay(dataset, source_name, *, sequence_length):
        if (
            dataset is joint_dataset
            and source_name == "audited_alignment"
            and sequence_length == module.PARENT_SEQUENCE_LENGTH
        ):
            return "0" * 64
        return original_replay(dataset, source_name, sequence_length=sequence_length)

    monkeypatch.setattr(
        module,
        "vision_alignment_joint_annotation_replay_sha256",
        drifted_replay,
    )
    with pytest.raises(ValueError, match="beyond the sequence bound"):
        module.build_vision_alignment_joint_projection(
            parent_perception_provenance=parent.path,
            expected_parent_perception_sha256=parent.raw_sha256,
            output_dir=tmp_path / "joint-projection-cross-corner",
            tokenizer=object(),
            token_ids=_TEST_TOKEN_IDS,
        )


def test_builder_is_write_once(tmp_path, monkeypatch):
    module = _load_builder()
    _, output, parent, _, _ = _run(module, tmp_path, monkeypatch)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.build_vision_alignment_joint_projection(
            parent_perception_provenance=parent.path,
            expected_parent_perception_sha256=parent.raw_sha256,
            output_dir=output,
            tokenizer=object(),
            token_ids=_TEST_TOKEN_IDS,
        )


def test_builder_rejects_parent_raw_drift(tmp_path, monkeypatch):
    module = _load_builder()
    parent, datasets, _ = _fixture(tmp_path)
    original_build = module.build_vision_alignment_joint_dataset
    _patch_parent_and_datasets(module, monkeypatch, parent, datasets)
    calls = 0

    def drift_parent(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            parent.path.write_bytes(parent.path.read_bytes() + b" ")
        return datasets[(args[3], kwargs["split"])]

    assert callable(original_build)
    monkeypatch.setattr(module, "build_vision_alignment_joint_dataset", drift_parent)
    with pytest.raises(ValueError, match="raw SHA-256 differs"):
        module.build_vision_alignment_joint_projection(
            parent_perception_provenance=parent.path,
            expected_parent_perception_sha256=parent.raw_sha256,
            output_dir=tmp_path / "joint-projection",
            tokenizer=object(),
            token_ids=_TEST_TOKEN_IDS,
        )


def test_builder_rejects_adapter_and_base_identity_drift(tmp_path, monkeypatch):
    module = _load_builder()
    parent, datasets, _ = _fixture(tmp_path)
    _patch_parent_and_datasets(module, monkeypatch, parent, datasets)
    dataset = datasets[("audited_alignment", "train")]
    dataset.config = replace(dataset.config, max_sequence_length=2560)
    with pytest.raises(ValueError, match="wrong config"):
        module.build_vision_alignment_joint_projection(
            parent_perception_provenance=parent.path,
            expected_parent_perception_sha256=parent.raw_sha256,
            output_dir=tmp_path / "joint-projection-adapter",
            tokenizer=object(),
            token_ids=_TEST_TOKEN_IDS,
        )

    parent, datasets, _ = _fixture(tmp_path / "base")
    _patch_parent_and_datasets(module, monkeypatch, parent, datasets)
    datasets[("audited_alignment", "train")].images.pop()
    with pytest.raises(ValueError, match="base identity differs"):
        module.build_vision_alignment_joint_projection(
            parent_perception_provenance=parent.path,
            expected_parent_perception_sha256=parent.raw_sha256,
            output_dir=tmp_path / "joint-projection-base",
            tokenizer=object(),
            token_ids=_TEST_TOKEN_IDS,
        )


def test_builder_rejects_image_bytes_and_snapshot_fingerprint_drift(tmp_path, monkeypatch):
    module = _load_builder()
    parent, datasets, _ = _fixture(tmp_path)
    _patch_parent_and_datasets(module, monkeypatch, parent, datasets)
    datasets[("audited_alignment", "train")].images[0] = b"different-image"
    with pytest.raises(ValueError, match="image bytes differ from parent"):
        module.build_vision_alignment_joint_projection(
            parent_perception_provenance=parent.path,
            expected_parent_perception_sha256=parent.raw_sha256,
            output_dir=tmp_path / "joint-projection-image",
            tokenizer=object(),
            token_ids=_TEST_TOKEN_IDS,
        )

    parent, datasets, _ = _fixture(tmp_path / "fingerprint")
    _patch_parent_and_datasets(module, monkeypatch, parent, datasets)
    dataset = datasets[("audited_alignment", "train")]
    changed = False

    def mutate_fingerprint(_index):
        nonlocal changed
        if not changed:
            dataset.content_fingerprint = "0" * 64
            changed = True

    dataset.on_image_read = mutate_fingerprint
    with pytest.raises(ValueError, match="identity changed during projection"):
        module.build_vision_alignment_joint_projection(
            parent_perception_provenance=parent.path,
            expected_parent_perception_sha256=parent.raw_sha256,
            output_dir=tmp_path / "joint-projection-fingerprint",
            tokenizer=object(),
            token_ids=_TEST_TOKEN_IDS,
        )


def test_builder_rejects_parent_union_overlap(tmp_path, monkeypatch):
    module = _load_builder()
    parent, datasets, _ = _fixture(tmp_path, overlap=True)
    _patch_parent_and_datasets(module, monkeypatch, parent, datasets)
    with pytest.raises(ValueError, match="unions are not non-empty and disjoint"):
        module.build_vision_alignment_joint_projection(
            parent_perception_provenance=parent.path,
            expected_parent_perception_sha256=parent.raw_sha256,
            output_dir=tmp_path / "joint-projection",
            tokenizer=object(),
            token_ids=_TEST_TOKEN_IDS,
        )


def test_builder_rejects_duplicate_parent_fields_and_invalid_types(tmp_path):
    module = _load_builder()
    path = tmp_path / PERCEPTION_PROVENANCE_MANIFEST
    raw = (
        b'{"source_registry_sha256":"'
        + b"a" * 64
        + b'","source_registry_sha256":"'
        + b"b" * 64
        + b'"}'
    )
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="repeats key"):
        module._read_parent_root(path, expected_sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        module.build_vision_alignment_joint_projection(
            parent_perception_provenance=path,
            expected_parent_perception_sha256=True,
            output_dir=tmp_path / "output",
            tokenizer=object(),
            token_ids=object(),
        )
    with pytest.raises(ValueError, match="include a timezone"):
        module._resolve_created_at("2026-08-13T23:00:00")


def test_atomic_staging_is_removed_when_runtime_validation_fails(tmp_path, monkeypatch):
    module = _load_builder()
    parent, datasets, _ = _fixture(tmp_path)
    _patch_parent_and_datasets(module, monkeypatch, parent, datasets)
    monkeypatch.setattr(
        module,
        "load_joint_visual_projection_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("synthetic runtime validation failure")
        ),
    )
    output = tmp_path / "joint-projection"
    with pytest.raises(ValueError, match="synthetic runtime validation failure"):
        module.build_vision_alignment_joint_projection(
            parent_perception_provenance=parent.path,
            expected_parent_perception_sha256=parent.raw_sha256,
            output_dir=output,
            tokenizer=object(),
            token_ids=_TEST_TOKEN_IDS,
        )
    assert not output.exists()
    assert not tuple(tmp_path.glob(".joint-projection.building-*"))


def test_builder_self_pin_is_checked_again_before_staging(tmp_path, monkeypatch):
    module = _load_builder()
    parent, datasets, _ = _fixture(tmp_path)
    _patch_parent_and_datasets(module, monkeypatch, parent, datasets)
    original_sha256_file = module._sha256_file
    builder_path = Path(module.__file__).resolve()
    builder_calls = 0

    def drifted_sha(path):
        nonlocal builder_calls
        if Path(path).resolve() == builder_path:
            builder_calls += 1
            if builder_calls > 1:
                return "0" * 64
        return original_sha256_file(path)

    monkeypatch.setattr(module, "_sha256_file", drifted_sha)
    with pytest.raises(ValueError, match="builder changed during artifact construction"):
        module.build_vision_alignment_joint_projection(
            parent_perception_provenance=parent.path,
            expected_parent_perception_sha256=parent.raw_sha256,
            output_dir=tmp_path / "joint-projection",
            tokenizer=object(),
            token_ids=_TEST_TOKEN_IDS,
        )
    assert builder_calls == 2
    assert not tuple(tmp_path.glob(".joint-projection.building-*"))
