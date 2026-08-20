from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).parents[2]
        / "scripts"
        / "data"
        / "build_ssmax_single_response_calibration.py"
    )
    spec = importlib.util.spec_from_file_location("ssmax_single_response_calibration_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_immutable_publisher_never_replaces_existing_target(tmp_path: Path) -> None:
    module = _load_module()
    target = tmp_path / "calibration.json"
    module._write_json_no_replace(target, {"value": 1})
    original = target.read_bytes()

    with pytest.raises(FileExistsError):
        module._write_json_no_replace(target, {"value": 2})

    assert target.read_bytes() == original
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_immutable_publisher_loses_creation_race_without_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    target = tmp_path / "calibration.json"
    stale = tmp_path / f".{target.name}.stale.tmp"
    stale.write_bytes(b"stale\n")
    competitor = b"competitor\n"
    original_link = module.os.link

    def racing_link(source: Path, destination: Path) -> None:
        Path(destination).write_bytes(competitor)
        original_link(source, destination)

    monkeypatch.setattr(module.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        module._write_json_no_replace(target, {"value": 1})

    assert target.read_bytes() == competitor
    assert stale.read_bytes() == b"stale\n"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == [stale]


def test_visual_weighting_resolver_uses_joint_parent_perception_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    parent = module.VisionAlignmentPerceptionSourceSpec(
        phase="perception",
        pixmo_cap_path="/tmp/pixmo-cap",
        sequence_length=2560,
        max_crops=8,
        message_format="document",
        loss_token_weighting="root_subsegments_root_tokens",
        caption_prompt="Description:",
        transcript_prompt="Transcript:",
        require_transcript=True,
        finevision_visualweb_path="/tmp/visualweb",
        finevision_geo170k_path="/tmp/geo170k",
        finevision_visualweb_fingerprint="a" * 64,
        finevision_geo170k_fingerprint="b" * 64,
    )
    joint = module.VisionAlignmentJointSourceSpec(perception_spec=parent)
    monkeypatch.setattr(
        module.VisionAlignmentPerceptionSourceSpec,
        "validate_production_contract",
        lambda _self: None,
    )
    monkeypatch.setattr(
        module.VisionAlignmentJointSourceSpec,
        "validate_production_contract",
        lambda _self: None,
    )

    assert (
        module._visual_loss_token_weighting(parent, phase="perception")
        == "root_subsegments_root_tokens"
    )
    assert module._visual_loss_token_weighting(joint, phase="joint") == parent.loss_token_weighting

    drifted_parent = replace(parent, loss_token_weighting="none")
    with pytest.raises(ValueError, match="production parent"):
        module._visual_loss_token_weighting(drifted_parent, phase="perception")
    with pytest.raises(ValueError, match="exact joint source spec"):
        module._visual_loss_token_weighting(parent, phase="joint")
