"""Submission-profile invariants for the s002 Molmo2 stage-1 launcher."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from olmo_core.distributed.parallel import DataParallelType
from olmo_core.launch.beaker import BeakerEnvVar
from olmo_core.launch.beaker_presets import get_preset


def _load_stage1_module():
    path = Path(__file__).parents[2] / "scripts" / "train" / "Molmo2-Stage1.py"
    spec = importlib.util.spec_from_file_location("molmo2_stage1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_s002_stage1_uses_olmo_ddp_ep8_and_freezes_only_vision():
    stage1 = _load_stage1_module()
    config = stage1._build_train_module_config(
        sequence_length=64,
        rank_microbatch_size=64,
        ep_degree=8,
        compile_model=False,
    )

    assert config.dp_config.name == DataParallelType.ddp
    assert config.ep_config.degree == 8
    assert config.freeze_params == ["vision.*"]
    assert config.response_logits_only
    assert not config.compile_model

    overrides = {tuple(group.params): group.opts for group in config.optim.group_overrides}
    assert overrides[("*connector.*",)]["lr"] == stage1.CONNECTOR_LR
    assert config.optim.lr == stage1.LLM_LR


def test_stage1_runtime_preserves_pinned_dataset_stack_and_quiets_dynamo_logs():
    stage1 = _load_stage1_module()
    launch = SimpleNamespace(
        beaker_image=None,
        env_vars=[BeakerEnvVar(name="EXPLICIT_SETTING", value="kept")],
        post_setup=None,
    )

    stage1._configure_launch_runtime(launch)

    env = {item.name: item.value for item in launch.env_vars}
    preset = get_preset("olmo-ddp")
    assert launch.beaker_image == preset.beaker_image
    assert launch.post_setup == preset.post_setup
    assert "pip install" not in (launch.post_setup or "")
    assert env["EXPLICIT_SETTING"] == "kept"
    assert env["TORCH_LOGS"] == "-dynamo"


@pytest.mark.parametrize(
    "profile_name,expected_dataset",
    [
        ("stage1_ep8_2node_synthetic_1step.yaml", "--dataset.dataset_path=synthetic"),
        ("stage1_ep8_2node_real_1step.yaml", None),
    ],
)
def test_beaker_gate_profiles_use_approved_holmes_target(profile_name, expected_dataset):
    stage1 = _load_stage1_module()
    profile_path = Path(__file__).parents[3] / "configs" / "vision_moe" / profile_name

    profile, overrides = stage1._load_beaker_test_config([f"--beaker-test-config={profile_path}"])
    assert profile is not None
    assert profile["launch"] == {
        "num_nodes": 2,
        "num_gpus": 8,
        "workspace": "ai2/molmofication",
        "cluster": "ai2/holmes",
        "budget": "ai2/oe-other",
        "priority": "urgent",
        "min_runtime": "1h",
    }
    assert "--trainer.max_duration.value=1" in overrides
    assert expected_dataset is None or expected_dataset in overrides

    config = SimpleNamespace(
        launch=SimpleNamespace(
            num_nodes=1,
            num_gpus=8,
            workspace="must-be-cleared",
            clusters=["must-be-cleared"],
            budget="must-be-replaced",
            priority="normal",
            min_runtime=None,
            description=None,
        )
    )
    stage1._apply_beaker_test_config(config, profile)
    assert config.launch.num_nodes == 2
    assert config.launch.num_gpus == 8
    assert config.launch.workspace == "ai2/molmofication"
    assert config.launch.clusters == ["ai2/holmes"]
    assert config.launch.budget == "ai2/oe-other"
    assert config.launch.priority == "urgent"
    assert config.launch.min_runtime == "1h"


def test_beaker_gate_refuses_an_unset_submission_target():
    stage1 = _load_stage1_module()
    config = SimpleNamespace(launch=SimpleNamespace(workspace=None, clusters=[]))
    with pytest.raises(RuntimeError, match="workspace and cluster are unset"):
        stage1.launch(config)


def test_beaker_gate_cli_overrides_take_precedence():
    stage1 = _load_stage1_module()
    profile_path = (
        Path(__file__).parents[3]
        / "configs"
        / "vision_moe"
        / "stage1_ep8_2node_synthetic_1step.yaml"
    )
    cli_override = "--trainer.max_duration.value=2"

    _, overrides = stage1._load_beaker_test_config(
        [f"--beaker-test-config={profile_path}", cli_override]
    )

    assert overrides[-1] == cli_override
