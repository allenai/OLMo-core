"""Submission-profile invariants for the s002 Molmo2 stage-1 launcher."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
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
        image_token_ids=[120, 121],
    )

    assert config.dp_config.name == DataParallelType.ddp
    assert config.ep_config.degree == 8
    assert config.freeze_params == ["lm.lm_head.w_out.weight"]
    assert config.train_embedding_rows == [120, 121]
    assert config.vision_activation_checkpointing
    assert config.connector_activation_checkpointing
    assert config.response_logits_only
    assert not config.compile_model

    overrides = {tuple(group.params): group.opts for group in config.optim.group_overrides}
    assert overrides[("*connector.*",)]["lr"] == stage1.CONNECTOR_LR
    assert overrides[("*lm.embeddings.weight",)]["lr"] == stage1.CONNECTOR_LR
    assert overrides[("*vision.*",)]["lr"] == stage1.VISION_LR
    assert config.optim.lr == stage1.LLM_LR
    assert config.optim.sigma_factor == 12
    assert config.optim.clip_grad_norm_by_scheduler_group
    assert config.dp_config.reduce_grads_in_fp32
    assert config.dp_config.accumulate_grads_in_fp32


def test_s002_stage1_optimizer_groups_cover_lm_connector_and_vision():
    stage1 = _load_stage1_module()
    config = stage1._build_train_module_config(
        sequence_length=64,
        rank_microbatch_size=64,
        ep_degree=1,
        compile_model=False,
        image_token_ids=[120, 121],
    )
    model = nn.ModuleDict(
        {
            "lm": nn.ModuleDict(
                {
                    "embeddings": nn.Embedding(128, 4),
                    "body": nn.Linear(4, 4),
                }
            ),
            "connector": nn.Linear(4, 4),
            "vision": nn.Linear(4, 4),
        }
    )

    groups = config.optim.build_groups([model])
    grouped_names = {name: group for group in groups for name in group["named_params"]}

    assert set(grouped_names) == dict(model.named_parameters()).keys()
    assert grouped_names["lm.embeddings.weight"]["lr"] == stage1.CONNECTOR_LR
    assert grouped_names["lm.embeddings.weight"]["scheduler_name"] == "connector"
    assert all(
        "lr" not in grouped_names[name]
        for name in grouped_names
        if name.startswith("lm.") and name != "lm.embeddings.weight"
    )
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


def test_s002_stage1_can_override_only_routed_expert_adam_epsilon():
    stage1 = _load_stage1_module()
    train_module = stage1._build_train_module_config(
        sequence_length=64,
        rank_microbatch_size=64,
        ep_degree=1,
        compile_model=False,
        image_token_ids=[120, 121],
    )
    config = SimpleNamespace(routed_expert_eps=1e-8, train_module=train_module)

    stage1._configure_routed_expert_epsilon(config)

    override = train_module.optim.group_overrides[-1]
    assert override.params == ["*lm.blocks.*.routed_experts.*"]
    assert override.opts == {"eps": 1e-8}


def test_s002_stage1_matches_released_molmo2_scale_defaults():
    stage1 = _load_stage1_module()

    assert stage1.SEQUENCE_LENGTH == 2560
    assert stage1.GLOBAL_BATCH_INSTANCES == 128
    assert stage1.RANK_MICROBATCH_INSTANCES == 4
    assert stage1.MAX_STEPS == 32_000
    assert stage1.ExperimentConfig.__dataclass_fields__["data_seed"].default == 95818
    assert stage1.ExperimentConfig.__dataclass_fields__["init_seed"].default == 6198
    assert stage1.PACK_BUFFER_SIZE == 48
    assert stage1.PACK_MAX_CROPS == 9
    assert stage1.DATA_PREFETCH_WORKERS == 8
    assert (
        stage1.ExperimentConfig.__dataclass_fields__["data_prefetch_workers"].default
        == stage1.DATA_PREFETCH_WORKERS
    )
    assert stage1.LOSS_TOKEN_WEIGHTING == "none"
    assert stage1.EVAL_INTERVAL == 1000
    assert stage1.EVAL_EXAMPLES == 2048
    assert stage1.EVAL_RANK_BATCH_INSTANCES == 4
    assert stage1.EVAL_SEED == 6198
    assert stage1.FAST_VISION_EVAL_INTERVAL == 2000
    assert stage1.FAST_VISION_EVAL_EXAMPLES == 256
    assert stage1.FAST_LANGUAGE_EVAL_INTERVAL == 4000
    assert stage1.FAST_LANGUAGE_EVAL_TASKS == (
        "arc_challenge_test_mc_5shot_fast",
        "basic_skills_arithmetic_rc_5shot",
        "copycolors_10way_fast",
        "hellaswag_bpb_5shot",
    )
    assert stage1.BEAKER_CLUSTER == "ai2/holmes"
    assert stage1.BEAKER_WORKSPACE == "ai2/molmofication"
    assert stage1.BEAKER_BUDGET == "ai2/oe-other"
    assert stage1.MOLMO2_CONFIG_REVISION == "042abfa7a38879a376cec03d949eff0aefaa0600"
    assert stage1.VISION_REVISION == "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
    assert stage1.VISION_FINGERPRINT == (
        "9d9257ea672527b2e37cae7f61734afdf9280d3e77680f2c2d13d4da60aba6bf"
    )


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


def test_stage1_console_includes_multimodal_diagnostics():
    stage1 = _load_stage1_module()
    logger = stage1._build_console_logger()

    assert "data/*" in logger.metrics
    assert "multimodal/*" in logger.metrics
    assert "optim/* grad norm" in logger.metrics
    assert "optim/* clip coefficient" in logger.metrics


def test_stage1_validation_matches_released_loader_seed_without_reseeding_augmentation(
    monkeypatch,
):
    stage1 = _load_stage1_module()
    captured = {}

    @dataclass
    class DatasetConfig:
        split: str = "train"
        seed: int = 0

        def build(self, tokenizer):
            captured["dataset_config"] = self
            return self

    class Loader:
        def __init__(self, dataset, collator, **kwargs):
            captured["loader_dataset"] = dataset
            captured["loader_kwargs"] = kwargs

    class Evaluator:
        def __init__(self, **kwargs):
            captured["evaluator_kwargs"] = kwargs

    class Callback:
        def __init__(self, **kwargs):
            captured["callback_kwargs"] = kwargs

    trainer = SimpleNamespace(
        work_dir=Path("/tmp/stage1-validation-test"),
        device="cpu",
        dp_process_group=None,
        add_callback=lambda name, callback: captured.update(callback_name=name, callback=callback),
    )
    config = SimpleNamespace(
        dataset=DatasetConfig(),
        eval_interval=stage1.EVAL_INTERVAL,
        eval_examples=stage1.EVAL_EXAMPLES,
        eval_rank_batch_instances=stage1.EVAL_RANK_BATCH_INSTANCES,
        eval_seed=stage1.EVAL_SEED,
    )
    collator = SimpleNamespace(pad_sequence_length=stage1.SEQUENCE_LENGTH)
    monkeypatch.setattr(stage1, "MultimodalDataLoader", Loader)
    monkeypatch.setattr(stage1, "MultimodalLMEvaluator", Evaluator)
    monkeypatch.setattr(stage1, "EvaluatorCallback", Callback)

    stage1._add_validation_callback(
        trainer,
        tokenizer=object(),
        config=config,
        collator=collator,
        dp_world_size=2,
        dp_rank=0,
    )

    assert captured["dataset_config"].split == "validation"
    assert captured["dataset_config"].seed == 0
    assert captured["loader_kwargs"]["seed"] == 6198
    assert captured["loader_kwargs"]["shuffle"] is False
    assert captured["loader_kwargs"]["global_batch_size"] == 8 * stage1.SEQUENCE_LENGTH
    assert captured["callback_name"] == "pixmo_cap_validation"
    assert captured["callback_kwargs"]["eval_duration"] == stage1.Duration.steps(256)


def test_stage1_fast_vision_validation_uses_held_out_task_splits(monkeypatch):
    stage1 = _load_stage1_module()
    captured = {"loaders": [], "evaluators": []}

    base_dataset = stage1.PixMoCapDatasetConfig(
        dataset_path="synthetic", message_format="olmo3_chat"
    )
    monkeypatch.setattr(stage1.PixMoCapDatasetConfig, "build", lambda self, tokenizer: self)
    monkeypatch.setattr(stage1.PixMoCountDatasetConfig, "build", lambda self, tokenizer: self)
    monkeypatch.setattr(stage1.PixMoPointsDatasetConfig, "build", lambda self, tokenizer: self)

    class Loader:
        def __init__(self, dataset, collator, **kwargs):
            self.dataset = dataset
            captured["loaders"].append((dataset, kwargs))

    class Evaluator:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]
            captured["evaluators"].append(kwargs)

    class Callback:
        def __init__(self, **kwargs):
            captured["callback_kwargs"] = kwargs

    trainer = SimpleNamespace(
        work_dir=Path("/tmp/stage1-fast-vision-test"),
        device="cpu",
        dp_process_group=None,
        add_callback=lambda name, callback: captured.update(callback_name=name, callback=callback),
    )
    config = SimpleNamespace(
        dataset=base_dataset,
        fast_vision_eval_interval=stage1.FAST_VISION_EVAL_INTERVAL,
        fast_vision_eval_examples=stage1.FAST_VISION_EVAL_EXAMPLES,
        eval_rank_batch_instances=stage1.EVAL_RANK_BATCH_INSTANCES,
        eval_seed=stage1.EVAL_SEED,
    )
    collator = SimpleNamespace(pad_sequence_length=stage1.SEQUENCE_LENGTH)
    monkeypatch.setattr(stage1, "MultimodalDataLoader", Loader)
    monkeypatch.setattr(stage1, "MultimodalLMEvaluator", Evaluator)
    monkeypatch.setattr(stage1, "EvaluatorCallback", Callback)

    stage1._add_fast_vision_validation_callback(
        trainer,
        tokenizer=object(),
        config=config,
        collator=collator,
        dp_world_size=2,
        dp_rank=0,
    )

    datasets = [dataset for dataset, _ in captured["loaders"]]
    assert [evaluator["name"] for evaluator in captured["evaluators"]] == [
        "pixmo-cap-caption-validation",
        "pixmo-count-validation",
        "pixmo-points-validation",
    ]
    assert all(dataset.split == "validation" for dataset in datasets)
    assert datasets[0].mode == "caption"
    assert datasets[1].counting is True
    assert datasets[2].kind == "basic"
    assert all(dataset.message_format == "olmo3_chat" for dataset in datasets)
    assert all(kwargs["shuffle"] is False for _, kwargs in captured["loaders"])
    assert captured["callback_name"] == "fast_vision_validation"
    assert captured["callback_kwargs"]["eval_interval"] == 2000
    assert captured["callback_kwargs"]["eval_duration"] == stage1.Duration.steps(32)


def test_stage1_mixture_propagates_message_format_to_every_source(monkeypatch):
    stage1 = _load_stage1_module()
    created = []

    class DatasetConfig:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def build(self, tokenizer):
            return [object()]

    for name in (
        "PixMoPointsDatasetConfig",
        "PixMoCountDatasetConfig",
        "CoSynPointDatasetConfig",
        "Tulu4DatasetConfig",
    ):
        monkeypatch.setattr(stage1, name, DatasetConfig)

    dataset = SimpleNamespace(
        build=lambda tokenizer: [object()],
        max_sequence_length=stage1.SEQUENCE_LENGTH,
        loss_token_weighting="none",
        token_ids=object(),
        message_format="olmo3_chat",
    )
    config = SimpleNamespace(dataset=dataset, pointing_rate=0.3, nlp_rate=0.1)

    _, weights, _ = stage1._build_mixture_sources(object(), config)

    assert sum(weights) == pytest.approx(1.0)
    assert len(created) == 5
    assert all(kwargs["message_format"] == "olmo3_chat" for kwargs in created)
    assert created[-1]["max_first_msg_len"] == 2304
    assert created[-1]["style_length_conditioning"] is True


def test_stage1_fast_language_validation_uses_compact_dolma2_sentinels(monkeypatch):
    stage1 = _load_stage1_module()
    captured = {}
    callback = object()

    class CallbackConfig:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def build(self, trainer):
            captured["built_with"] = trainer
            return callback

    trainer = SimpleNamespace(
        dp_process_group=None,
        add_callback=lambda name, value: captured.update(callback_name=name, callback=value),
    )
    config = SimpleNamespace(
        fast_language_eval_interval=4000,
        fast_language_eval_rank_batch_instances=1,
        fast_language_eval_batches=30,
    )
    monkeypatch.setattr(stage1, "DownstreamEvaluatorCallbackConfig", CallbackConfig)

    stage1._add_fast_language_validation_callback(trainer, config)

    assert captured["built_with"] is trainer
    assert captured["kwargs"]["tasks"] == list(stage1.FAST_LANGUAGE_EVAL_TASKS)
    assert captured["kwargs"]["tokenizer"] == stage1.TokenizerConfig.dolma2()
    assert captured["kwargs"]["eval_interval"] == 4000
    assert captured["kwargs"]["eval_duration"] == stage1.Duration.steps(30)
    assert captured["kwargs"]["rank_batch_size_instances"] == 1
    assert captured["kwargs"]["log_interval"] == 7
    assert captured["kwargs"]["lazy"] is True
    assert captured["callback_name"] == "fast_language_validation"
    assert captured["callback"] is callback


def test_stage1_fast_language_validation_rejects_partial_choice_documents():
    stage1 = _load_stage1_module()
    trainer = SimpleNamespace(dp_process_group=None)
    config = SimpleNamespace(
        fast_language_eval_interval=4000,
        fast_language_eval_rank_batch_instances=1,
        fast_language_eval_batches=32,
    )

    with pytest.raises(ValueError, match="complete 10-choice Basic Skills documents"):
        stage1._add_fast_language_validation_callback(trainer, config)


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
    config = SimpleNamespace(launch=SimpleNamespace(workspace=None, clusters=[], hostnames=None))
    with pytest.raises(RuntimeError, match="workspace and placement constraints are unset"):
        stage1.launch(config)


def test_beaker_gate_accepts_hostname_only_placement():
    stage1 = _load_stage1_module()
    launched = False

    def launch(*, follow):
        nonlocal launched
        assert follow
        launched = True

    config = SimpleNamespace(
        launch=SimpleNamespace(
            workspace="ai2/molmofication", clusters=[], hostnames=["healthy-host"], launch=launch
        )
    )
    stage1.launch(config)
    assert launched


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
    assert "--train_module.scheduler.schedulers.connector.t_max=32000" in overrides
    assert "--train_module.scheduler.schedulers.vision.t_max=32000" in overrides
    assert "--train_module.scheduler.default.t_max=32000" in overrides


@pytest.mark.parametrize(
    "profile_name,num_nodes,rank_microbatch_size",
    [
        ("stage1_ep8_1node_real_200step_micro8.yaml", 1, 20_480),
        ("stage1_ep8_1node_real_200step_micro16_recompute.yaml", 1, 40_960),
        ("stage1_ep8_2node_real_200step_micro8.yaml", 2, 20_480),
    ],
)
def test_stage1_rerisk_topology_gates_hold_every_non_topology_control_fixed(
    profile_name, num_nodes, rank_microbatch_size
):
    stage1 = _load_stage1_module()
    profile_path = Path(__file__).parents[3] / "configs" / "vision_moe" / profile_name

    profile, overrides = stage1._load_beaker_test_config([f"--beaker-test-config={profile_path}"])

    assert profile is not None
    assert profile["launch"] == {
        "num_nodes": num_nodes,
        "num_gpus": 8,
        "workspace": "ai2/molmofication",
        "cluster": "ai2/holmes",
        "budget": "ai2/oe-other",
        "priority": "urgent",
        "min_runtime": "8h",
    }
    assert "--dataset.message_format=document" in overrides
    assert "--pack_max_crops=9" in overrides
    assert f"--train_module.rank_microbatch_size={rank_microbatch_size}" in overrides
    assert "--train_module.diagnostics_interval=10" in overrides
    assert "--trainer.load_strategy=never" in overrides
    assert "--trainer.max_duration.value=200" in overrides
    assert "--train_module.scheduler.schedulers.connector.t_max=32000" in overrides
    assert "--train_module.scheduler.schedulers.vision.t_max=32000" in overrides
    assert "--train_module.scheduler.default.t_max=32000" in overrides
    assert "--eval_interval=200" in overrides
    assert "--fast_vision_eval_interval=200" in overrides
    assert "--fast_language_eval_interval=200" in overrides
    expected_recompute = "true" if num_nodes == 1 else "false"
    assert f"--model.lm.recompute_each_block={expected_recompute}" in overrides
    assert not any(override.startswith("--base_checkpoint=") for override in overrides)
    assert not any(override.startswith("--routed_expert_eps=") for override in overrides)


def test_stage1_clean_b300_run_uses_the_full_32k_horizon():
    stage1 = _load_stage1_module()
    profile_path = (
        Path(__file__).parents[3] / "configs" / "vision_moe" / "stage1_ep8_2node_real_32k_b300.yaml"
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
    assert "--trainer.max_duration.value=32000" in overrides
    assert "--trainer.max_duration.unit=steps" in overrides
    assert "--train_module.scheduler.schedulers.connector.t_max=32000" in overrides
    assert "--train_module.scheduler.schedulers.vision.t_max=32000" in overrides
    assert "--train_module.scheduler.default.t_max=32000" in overrides
    assert "--model.lm.recompute_each_block=false" in overrides
    assert not any(override.startswith("--trainer.load_path=") for override in overrides)
    assert not any("wandb.run_id" in override for override in overrides)


def test_stage1_selected_micro8_continuation_restores_full_state_with_bounded_evals():
    stage1 = _load_stage1_module()
    profile_path = (
        Path(__file__).parents[3]
        / "configs"
        / "vision_moe"
        / "stage1_ep8_2node_real_resume_to32000_micro8.yaml"
    )

    profile, overrides = stage1._load_beaker_test_config([f"--beaker-test-config={profile_path}"])

    run_name = "s002-stage1-rerisk-200-micro8-2node-20260809-2db297f"
    save_folder = "/weka/oe-training-default/rustin/experiments/vision-moe/checkpoints/" + run_name
    assert profile is not None
    assert profile["launch"] == {
        "num_nodes": 2,
        "num_gpus": 8,
        "workspace": "ai2/molmofication",
        "hostnames": [
            "holmes-cs-aus-520.reviz.ai2.in",
            "holmes-cs-aus-511.reviz.ai2.in",
            "holmes-cs-aus-517.reviz.ai2.in",
        ],
        "budget": "ai2/oe-other",
        "priority": "urgent",
        "min_runtime": "8h",
    }
    launch_config = SimpleNamespace(
        num_nodes=1,
        num_gpus=1,
        workspace=None,
        clusters=["must-be-cleared"],
        hostnames=None,
        budget=None,
        priority="normal",
        min_runtime=None,
        description=None,
    )
    stage1._apply_beaker_test_config(SimpleNamespace(launch=launch_config), profile)
    assert launch_config.clusters == []
    assert launch_config.hostnames == profile["launch"]["hostnames"]
    assert f"--trainer.save_folder={save_folder}" in overrides
    assert f"--trainer.load_path={save_folder}/step4000" in overrides
    assert "--trainer.load_strategy=always" in overrides
    assert "--trainer.load_optim_state=true" in overrides
    assert "--trainer.load_trainer_state=true" in overrides
    assert "--dataset.message_format=document" in overrides
    assert "--pack_max_crops=9" in overrides
    assert "--train_module.rank_microbatch_size=20480" in overrides
    assert "--trainer.max_duration.value=32000" in overrides
    assert "--train_module.scheduler.schedulers.connector.t_max=32000" in overrides
    assert "--train_module.scheduler.schedulers.vision.t_max=32000" in overrides
    assert "--train_module.scheduler.default.t_max=32000" in overrides
    assert "--eval_interval=2000" in overrides
    assert "--eval_examples=512" in overrides
    assert "--fast_vision_eval_interval=2000" in overrides
    assert "--fast_vision_eval_examples=256" in overrides
    assert "--fast_language_eval_interval=4000" in overrides
    assert "--fast_language_eval_rank_batch_instances=1" in overrides
    assert "--fast_language_eval_batches=30" in overrides
    assert f"--trainer.callbacks.wandb.name={run_name}" in overrides
    assert "--trainer.callbacks.wandb.run_id=sdgbbjmz" in overrides
    assert "--trainer.callbacks.checkpointer.save_interval=2000" in overrides
    assert "--trainer.callbacks.checkpointer.ephemeral_save_interval=500" in overrides
    assert "--trainer.callbacks.checkpointer.max_checkpoints=2" in overrides


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
    assert "--data_prefetch_workers=8" in overrides
    assert "--trainer.callbacks.wandb.run_id=d7jwkm8w" in overrides
    assert "--train_module.scheduler.schedulers.connector.t_max=32000" in overrides
    assert "--train_module.scheduler.schedulers.vision.t_max=32000" in overrides
    assert "--train_module.scheduler.default.t_max=32000" in overrides
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
