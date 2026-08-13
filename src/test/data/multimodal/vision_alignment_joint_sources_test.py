"""Contracts for the phase-specific Vision Alignment joint visual registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from olmo_core.data.multimodal.pixmo_points import PixMoCountDatasetConfig
from olmo_core.data.multimodal.vision_alignment_joint_sources import (
    JOINT_TO_PERCEPTION_SOURCE,
    JOINT_VISUAL_SOURCE_NAMES,
    VISION_ALIGNMENT_JOINT_ADAPTER_PROJECTION_ALGORITHM,
    VISION_ALIGNMENT_JOINT_PARENT_SOURCE_MODULE_SHA256,
    VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION,
    VisionAlignmentJointSourceSpec,
    build_vision_alignment_joint_dataset_config,
    vision_alignment_joint_adapter_projection_sha256,
    vision_alignment_joint_implementation_inventory,
    vision_alignment_joint_source_registry_sha256,
)
from olmo_core.data.multimodal.vision_alignment_perception_sources import (
    VisionAlignmentPerceptionSourceSpec,
    build_vision_alignment_perception_dataset_config,
)
from olmo_core.nn.vision import Molmo2TokenIds


def _perception_spec() -> VisionAlignmentPerceptionSourceSpec:
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
        finevision_visualweb_path="/reviewed/finevision/visualweb",
        finevision_geo170k_path="/reviewed/finevision/geo170k",
        finevision_visualweb_fingerprint="a" * 64,
        finevision_geo170k_fingerprint="b" * 64,
    )


def _without_sequence_length(config):
    values = asdict(config)
    assert "max_sequence_length" in values
    del values["max_sequence_length"]
    return values


def test_joint_public_names_and_perception_mapping_are_exact():
    assert JOINT_VISUAL_SOURCE_NAMES == (
        "audited_alignment",
        "cosyn_point",
        "count_numeric",
        "ocr_document",
        "pixmo_caption",
        "pixmo_points_basic",
        "pixmo_points_high_frequency",
        "pixmo_transcript",
    )
    assert dict(JOINT_TO_PERCEPTION_SOURCE) == {
        "audited_alignment": "audited_alignment",
        "cosyn_point": "cosyn_point",
        "count_numeric": "scalar_count",
        "ocr_document": "ocr_document",
        "pixmo_caption": "pixmo_caption",
        "pixmo_points_basic": "pixmo_points_basic",
        "pixmo_points_high_frequency": "pixmo_points_high_frequency",
        "pixmo_transcript": "pixmo_transcript",
    }


def test_joint_spec_is_mechanically_derived_from_production_perception_spec():
    parent = _perception_spec()
    joint = VisionAlignmentJointSourceSpec.from_perception(parent)
    descriptor = joint.as_canonical_dict()
    expected = parent.as_canonical_dict()
    parent_registry_version = expected.pop("source_registry_version")
    expected["phase"] = "joint"
    expected["sequence_length"] = 8192

    assert descriptor == {
        "source_registry_version": VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION,
        "parent_perception_source_registry_version": parent_registry_version,
        "parent_perception_preprocessing_sha256": parent.preprocessing_sha256,
        **expected,
    }
    assert len(joint.preprocessing_sha256) == 64


@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize("source_name", JOINT_VISUAL_SOURCE_NAMES)
def test_every_joint_adapter_differs_from_perception_only_by_sequence_length(source_name, split):
    parent = _perception_spec()
    joint_spec = VisionAlignmentJointSourceSpec.from_perception(parent)
    token_ids = Molmo2TokenIds()
    joint_config = build_vision_alignment_joint_dataset_config(
        joint_spec, token_ids, source_name, split=split
    )
    parent_config = build_vision_alignment_perception_dataset_config(
        parent,
        token_ids,
        JOINT_TO_PERCEPTION_SOURCE[source_name],
        split=split,
    )

    assert type(joint_config) is type(parent_config)
    assert joint_config.max_sequence_length == 8192
    assert parent_config.max_sequence_length == 2560
    assert _without_sequence_length(joint_config) == _without_sequence_length(parent_config)
    assert vision_alignment_joint_adapter_projection_sha256(
        joint_config
    ) == vision_alignment_joint_adapter_projection_sha256(parent_config)


def test_count_numeric_is_the_scalar_count_adapter_at_joint_length():
    config = build_vision_alignment_joint_dataset_config(
        VisionAlignmentJointSourceSpec.from_perception(_perception_spec()),
        Molmo2TokenIds(),
        "count_numeric",
        split="validation",
    )

    assert isinstance(config, PixMoCountDatasetConfig)
    assert config.mode == "scalar_count"
    assert config.counting == "both"
    assert config.require_split is True
    assert config.split == "validation"
    assert config.max_sequence_length == 8192


@pytest.mark.parametrize("source_name", ["scalar_count", "native_text_replay", "unknown"])
def test_joint_registry_rejects_non_public_visual_names(source_name):
    with pytest.raises(KeyError, match=source_name):
        build_vision_alignment_joint_dataset_config(
            VisionAlignmentJointSourceSpec.from_perception(_perception_spec()),
            Molmo2TokenIds(),
            source_name,
        )


@pytest.mark.parametrize(
    "joint_spec",
    [
        VisionAlignmentJointSourceSpec(_perception_spec(), phase="perception"),
        VisionAlignmentJointSourceSpec(_perception_spec(), sequence_length=2560),
        VisionAlignmentJointSourceSpec(replace(_perception_spec(), caption_prompt="Caption:")),
    ],
)
def test_joint_spec_rejects_phase_length_and_parent_config_drift(joint_spec):
    with pytest.raises(ValueError):
        joint_spec.validate_production_contract()


def test_joint_build_fails_closed_if_adapter_projection_drifts(monkeypatch):
    from olmo_core.data.multimodal import (
        vision_alignment_joint_sources as joint_sources,
    )

    original = joint_sources.build_vision_alignment_perception_dataset_config

    def drifted_parent_config(*args, **kwargs):
        config = original(*args, **kwargs)
        config.seed = 1
        return config

    monkeypatch.setattr(
        joint_sources,
        "build_vision_alignment_perception_dataset_config",
        drifted_parent_config,
    )
    with pytest.raises(ValueError, match="differs from its pinned perception projection"):
        joint_sources.build_vision_alignment_joint_dataset_config(
            VisionAlignmentJointSourceSpec.from_perception(_perception_spec()),
            Molmo2TokenIds(),
            "pixmo_caption",
        )


def test_adapter_projection_hashes_class_and_every_field_except_sequence_length():
    config = build_vision_alignment_perception_dataset_config(
        _perception_spec(), Molmo2TokenIds(), "pixmo_caption"
    )
    values = asdict(config)
    del values["max_sequence_length"]
    descriptor = {
        "algorithm": VISION_ALIGNMENT_JOINT_ADAPTER_PROJECTION_ALGORITHM,
        "config_class": f"{type(config).__module__}.{type(config).__qualname__}",
        "config": values,
    }
    expected = hashlib.sha256(
        json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    assert vision_alignment_joint_adapter_projection_sha256(config) == expected
    config.seed = 1
    assert vision_alignment_joint_adapter_projection_sha256(config) != expected
    with pytest.raises(ValueError, match="dataclass"):
        vision_alignment_joint_adapter_projection_sha256(object())


def test_joint_registry_inventory_pins_parent_and_transitive_adapters():
    inventory = vision_alignment_joint_implementation_inventory()

    assert inventory["version"] == 1
    assert inventory["source_names"] == list(JOINT_VISUAL_SOURCE_NAMES)
    assert inventory["source_mapping"] == dict(JOINT_TO_PERCEPTION_SOURCE)
    assert (
        inventory["files"]["vision_alignment_perception_sources.py"]
        == VISION_ALIGNMENT_JOINT_PARENT_SOURCE_MODULE_SHA256
    )
    assert set(inventory["files"]) >= {
        "finevision.py",
        "pixmo_cap.py",
        "pixmo_points.py",
        "vision_alignment_joint_sources.py",
        "vision_alignment_perception.py",
        "vision_alignment_perception_sources.py",
    }
    assert all(len(value) == 64 for value in inventory["files"].values())
    assert len(vision_alignment_joint_source_registry_sha256()) == 64


def test_joint_registry_rejects_parent_source_module_drift(monkeypatch):
    from olmo_core.data.multimodal import (
        vision_alignment_joint_sources as joint_sources,
    )

    monkeypatch.setattr(joint_sources, "_sha256_file", lambda _path: "0" * 64)
    with pytest.raises(ValueError, match="pinned perception source module"):
        VisionAlignmentJointSourceSpec.from_perception(_perception_spec())
