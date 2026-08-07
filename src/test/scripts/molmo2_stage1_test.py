"""Submission-profile invariants for the s002 Molmo2 stage-1 launcher."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.nn as nn

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


def test_s002_stage1_uses_olmo_ddp_ep8_and_trains_all_components():
    stage1 = _load_stage1_module()
    config = stage1._build_train_module_config(
        sequence_length=64,
        rank_microbatch_size=64,
        ep_degree=8,
        compile_model=False,
    )

    assert config.dp_config.name == DataParallelType.ddp
    assert config.ep_config.degree == 8
    assert config.freeze_params is None
    assert config.vision_activation_checkpointing
    assert config.connector_activation_checkpointing
    assert config.response_logits_only
    assert not config.compile_model

    overrides = {tuple(group.params): group.opts for group in config.optim.group_overrides}
    assert overrides[("*connector.*",)]["lr"] == stage1.CONNECTOR_LR
    assert overrides[("*vision.*",)]["lr"] == stage1.VISION_LR
    assert config.optim.lr == stage1.LLM_LR
    assert config.optim.sigma_factor == 12
    assert config.dp_config.reduce_grads_in_fp32
    assert config.dp_config.accumulate_grads_in_fp32


def test_s002_stage1_optimizer_groups_cover_lm_connector_and_vision():
    stage1 = _load_stage1_module()
    config = stage1._build_train_module_config(
        sequence_length=64,
        rank_microbatch_size=64,
        ep_degree=1,
        compile_model=False,
    )
    model = nn.ModuleDict(
        {
            "lm": nn.Linear(4, 4),
            "connector": nn.Linear(4, 4),
            "vision": nn.Linear(4, 4),
        }
    )

    groups = config.optim.build_groups([model])
    grouped_names = {name: group for group in groups for name in group["named_params"]}

    assert set(grouped_names) == dict(model.named_parameters()).keys()
    assert all("lr" not in grouped_names[name] for name in grouped_names if name.startswith("lm."))
    assert all(
        grouped_names[name]["lr"] == stage1.CONNECTOR_LR
        and grouped_names[name]["scheduler_name"] == "connector"
        for name in grouped_names
        if name.startswith("connector.")
    )
    assert all(
        grouped_names[name]["lr"] == stage1.VISION_LR
        and grouped_names[name]["scheduler_name"] == "vision"
        for name in grouped_names
        if name.startswith("vision.")
    )


def test_s002_stage1_matches_released_molmo2_scale_defaults():
    stage1 = _load_stage1_module()

    assert stage1.SEQUENCE_LENGTH == 2536
    assert stage1.GLOBAL_BATCH_INSTANCES == 128
    assert stage1.RANK_MICROBATCH_INSTANCES == 4
    assert stage1.MAX_STEPS == 31_000
    assert stage1.PACK_BUFFER_SIZE == 48
    assert stage1.PACK_MAX_CROPS == 16
    assert stage1.DATA_PREFETCH_WORKERS == 4
    assert (
        stage1.ExperimentConfig.__dataclass_fields__["data_prefetch_workers"].default
        == stage1.DATA_PREFETCH_WORKERS
    )
    assert stage1.LOSS_TOKEN_WEIGHTING == "none"
    assert stage1.EVAL_INTERVAL == 1000
    assert stage1.EVAL_EXAMPLES == 64
    assert stage1.EVAL_RANK_BATCH_INSTANCES == 1
    assert stage1.BEAKER_CLUSTER == "ai2/holmes"
    assert stage1.BEAKER_WORKSPACE == "ai2/molmofication"
    assert stage1.BEAKER_BUDGET == "ai2/oe-other"


def test_s002_stage1_preserves_native_router_objectives():
    stage1 = _load_stage1_module()
    default_router = SimpleNamespace(lb_loss_weight=0.015, z_loss_weight=0.0001)
    override_router = SimpleNamespace(lb_loss_weight=0.015, z_loss_weight=0.0001)
    lm_config = SimpleNamespace(
        block=SimpleNamespace(routed_experts_router=default_router),
        block_overrides={
            0: SimpleNamespace(routed_experts_router=None),
            1: SimpleNamespace(routed_experts_router=override_router),
        },
    )

    configured = stage1._configure_router_load_balancing(lm_config, stage1.ROUTER_LB_LOSS_WEIGHT)

    assert configured == 2
    assert default_router.lb_loss_weight == 0.015
    assert override_router.lb_loss_weight == 0.015
    assert default_router.z_loss_weight == 0.0001
    assert override_router.z_loss_weight == 0.0001


def test_s002_stage1_router_load_balancing_is_overridable():
    stage1 = _load_stage1_module()
    router = SimpleNamespace(lb_loss_weight=None, z_loss_weight=0.0001)
    lm_config = SimpleNamespace(
        block=SimpleNamespace(routed_experts_router=router),
        block_overrides=None,
    )

    stage1._configure_router_load_balancing(lm_config, 0.015)

    assert router.lb_loss_weight == 0.015
    assert router.z_loss_weight == 0.0001


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
    assert env["TORCHINDUCTOR_COMPILE_THREADS"] == "8"
    assert env["TORCH_LOGS"] == "-dynamo"


@pytest.mark.parametrize(
    "profile_name,expected_dataset,expected_min_runtime",
    [
        (
            "stage1_ep8_2node_synthetic_1step.yaml",
            "--dataset.dataset_path=synthetic",
            "1h",
        ),
        ("stage1_ep8_2node_real_1step.yaml", None, None),
    ],
)
def test_beaker_gate_profiles_use_approved_holmes_target(
    profile_name, expected_dataset, expected_min_runtime
):
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
        "min_runtime": expected_min_runtime,
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
    assert config.launch.min_runtime == expected_min_runtime


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


def test_stage1_pilot_is_an_exact_prefix_of_the_production_schedule():
    stage1 = _load_stage1_module()
    profile_path = (
        Path(__file__).parents[3]
        / "configs"
        / "vision_moe"
        / "stage1_ep8_2node_real_500step_pilot.yaml"
    )

    profile, overrides = stage1._load_beaker_test_config([f"--beaker-test-config={profile_path}"])

    assert profile is not None
    assert profile["launch"]["workspace"] == "ai2/molmofication"
    assert profile["launch"]["cluster"] == "ai2/holmes"
    assert profile["launch"]["budget"] == "ai2/oe-other"
    assert profile["launch"]["priority"] == "urgent"
    assert profile["launch"]["min_runtime"] is None
    assert "--trainer.max_duration.value=500" in overrides


def test_stage1_b300_continuation_restores_step4000_and_runs_to8000():
    stage1 = _load_stage1_module()
    profile_path = (
        Path(__file__).parents[3]
        / "configs"
        / "vision_moe"
        / "stage1_ep8_2node_real_resume_to8000_b300.yaml"
    )

    profile, overrides = stage1._load_beaker_test_config([f"--beaker-test-config={profile_path}"])

    assert profile is not None
    assert profile["launch"] == {
        "num_nodes": 2,
        "num_gpus": 8,
        "workspace": "ai2/molmofication",
        "cluster": "ai2/holmes",
        "budget": "ai2/oe-other",
        "priority": "urgent",
        "min_runtime": "8h",
    }
    assert (
        "--trainer.load_path=/weka/oe-training-default/rustin/experiments/vision-moe/"
        "checkpoints/s002-stage1-padding-safe-real-resume-to4000-20260806/step4000" in overrides
    )
    assert "--trainer.load_optim_state=true" in overrides
    assert "--trainer.load_trainer_state=true" in overrides
    assert "--trainer.max_duration.value=8000" in overrides
    assert "--model.lm.recompute_each_block=false" in overrides
    assert "--trainer.callbacks.wandb.run_id=d7jwkm8w" in overrides
    assert "--train_module.scheduler.schedulers.connector.t_max=31000" in overrides
    assert "--train_module.scheduler.schedulers.vision.t_max=31000" in overrides
    assert "--train_module.scheduler.default.t_max=31000" in overrides
    assert not any("router_lb_loss_weight" in override for override in overrides)

    assert stage1.ROUTER_LB_LOSS_WEIGHT == 0.015


def test_stage1_resume_gate_restores_full_state_into_a_new_run():
    stage1 = _load_stage1_module()
    profile_path = (
        Path(__file__).parents[3]
        / "configs"
        / "vision_moe"
        / "stage1_ep8_2node_real_resume_2step.yaml"
    )

    profile, overrides = stage1._load_beaker_test_config([f"--beaker-test-config={profile_path}"])

    assert profile is not None
    assert profile["launch"] == {
        "num_nodes": 2,
        "num_gpus": 8,
        "workspace": "ai2/molmofication",
        "cluster": "ai2/holmes",
        "budget": "ai2/oe-other",
        "priority": "urgent",
        "min_runtime": None,
    }
    assert any(override.startswith("--trainer.load_path=") for override in overrides)
    assert "--trainer.load_strategy=always" in overrides
    assert "--trainer.load_optim_state=true" in overrides
    assert "--trainer.load_trainer_state=true" in overrides
    assert "--trainer.max_duration.value=2" in overrides
    assert "--trainer.callbacks.checkpointer.save_interval=2" in overrides
