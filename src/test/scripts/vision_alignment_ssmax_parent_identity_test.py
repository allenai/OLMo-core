from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.distributed as dist

from olmo_core.eval import vision_alignment_ssmax_bridge as bridge


def _load_recipe():
    path = Path(__file__).parents[2] / "scripts" / "train" / "Vision-Alignment.py"
    spec = importlib.util.spec_from_file_location("vision_alignment_parent_identity_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("variant_name", ["ssmax_head_qknorm", "ssmax_no_qknorm"])
def test_ssmax_parent_full_checkpoint_identity_is_required_before_load(
    monkeypatch: pytest.MonkeyPatch, variant_name: str
) -> None:
    recipe = _load_recipe()
    variant = recipe.VisionAlignmentModelVariant(variant_name)
    artifacts = recipe.ArtifactConfig.for_model_variant(variant)
    identity = {
        "path": str(Path(artifacts.base_checkpoint).resolve()),
        "global_step": 65_799,
        "config_sha256": artifacts.base_config_sha256,
        "marker_sha256": artifacts.base_checkpoint_marker_sha256,
        "dcp_metadata_sha256": artifacts.base_checkpoint_metadata_sha256,
        "state_file_count": artifacts.base_checkpoint_state_file_count,
        "state_file_inventory_sha256": artifacts.base_checkpoint_state_file_inventory_sha256,
        "trainer_state_count": artifacts.base_checkpoint_trainer_state_count,
        "trainer_state_inventory_sha256": (
            artifacts.base_checkpoint_trainer_state_inventory_sha256
        ),
        "identity_sha256": artifacts.base_checkpoint_identity_sha256,
    }
    observed: dict[str, object] = {}

    def checkpoint_identity(path: Path, *, workers: int):
        observed.update(path=path, workers=workers)
        return dict(identity)

    monkeypatch.setattr(bridge, "checkpoint_identity", checkpoint_identity)
    monkeypatch.setattr(recipe, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "broadcast_object_list", lambda value, src: None)
    config = SimpleNamespace(artifacts=artifacts, checkpoint_load_threads=7)

    assert recipe._verify_ssmax_parent_checkpoint_bytes(config) == identity
    assert observed == {"path": Path(artifacts.base_checkpoint), "workers": 7}

    identity["state_file_inventory_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="state_file_inventory_sha256"):
        recipe._verify_ssmax_parent_checkpoint_bytes(config)


def test_ssmax_parent_full_checkpoint_identity_requires_every_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = _load_recipe()
    artifacts = recipe.ArtifactConfig.for_model_variant(
        recipe.VisionAlignmentModelVariant.ssmax_head_qknorm
    )
    artifacts.base_checkpoint_identity_sha256 = None
    monkeypatch.setattr(recipe, "get_rank", lambda: 0)

    with pytest.raises(RuntimeError, match="byte pins are incomplete"):
        recipe._verify_ssmax_parent_checkpoint_bytes(
            SimpleNamespace(artifacts=artifacts, checkpoint_load_threads=8)
        )
