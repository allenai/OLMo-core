"""Configuration invariants for the s002 Molmo2 stage-2 launcher."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from olmo_core.distributed.parallel import DataParallelType
from olmo_core.launch.beaker import BeakerEnvVar, BeakerLaunchConfig
from olmo_core.launch.beaker_presets import get_preset


def _load_stage2_module():
    path = Path(__file__).parents[2] / "scripts" / "train" / "Molmo2-Stage2.py"
    spec = importlib.util.spec_from_file_location("molmo2_stage2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_s002_stage2_uses_olmo_ddp_ep8_and_trains_all_components():
    stage2 = _load_stage2_module()
    config = stage2._build_train_module_config(
        sequence_length=64,
        rank_microbatch_size=64,
        ep_degree=8,
        compile_model=False,
    )

    assert config.dp_config.name == DataParallelType.ddp
    assert config.dp_config.reduce_grads_in_fp32
    assert config.dp_config.accumulate_grads_in_fp32
    assert config.ep_config.degree == 8
    assert config.freeze_params is None
    assert config.vision_activation_checkpointing
    assert config.connector_activation_checkpointing
    assert config.response_logits_only
    assert config.diagnostics_interval == stage2.DIAGNOSTICS_INTERVAL
    assert not config.compile_model

    overrides = {tuple(group.params): group.opts for group in config.optim.group_overrides}
    assert overrides[("*connector.*",)]["lr"] == stage2.CONNECTOR_LR
    assert overrides[("*vision.*",)]["lr"] == stage2.VISION_LR
    assert config.optim.lr == stage2.LLM_LR
    assert config.optim.sigma_factor == 12
    assert config.optim.clip_grad_norm_by_scheduler_group
    assert config.scheduler.schedulers["connector"].t_max == stage2.MAX_STEPS
    assert config.scheduler.schedulers["vision"].t_max == stage2.MAX_STEPS
    assert config.scheduler.default.t_max == stage2.MAX_STEPS


def test_s002_stage2_production_defaults():
    stage2 = _load_stage2_module()

    assert stage2.NUM_NODES == 2
    assert stage2.EP_DEGREE == 8
    assert stage2.STAGE2_MOE_CAPACITY_FACTOR == 2.0
    assert stage2.GLOBAL_BATCH_INSTANCES == 128
    assert stage2.RANK_MICROBATCH_INSTANCES == 1
    assert stage2.MAX_STEPS == 30_000
    assert stage2.PACK_MAX_CROPS == 45
    assert stage2.PACK_BUFFER_SIZE == 48
    assert stage2.PACK_IMAGE_WEIGHT == 30.0
    assert stage2.MMFINEREASON_RATE == 0.0
    assert stage2.FAST_VISION_EVAL_INTERVAL == 2000
    assert stage2.FAST_VISION_EVAL_EXAMPLES == 32
    assert stage2.FAST_VISION_EVAL_RANK_BATCH_INSTANCES == 1
    assert stage2.MAX_CONSECUTIVE_DATA_ERRORS == 10
    assert stage2.MAX_TOTAL_DATA_ERRORS == 1000
    assert stage2.BEAKER_CLUSTER == "ai2/holmes"
    assert stage2.BEAKER_WORKSPACE == "ai2/molmofication"
    assert stage2.TOKENIZER_ID.endswith(
        "s002-olmo3moe-instruct-sft-resume-to1000-fused-20260727-hf"
    )
    assert stage2.DEFAULT_LOAD_PATH.endswith(
        "s002-stage1-corrected-clean-32k-b300-20260807/step32000"
    )


def test_stage2_data_error_monitor_records_global_cumulative_count():
    stage2 = _load_stage2_module()
    data_loader = object.__new__(stage2.MixtureDataLoader)
    data_loader._total_data_errors = 3
    recorded = []
    trainer = SimpleNamespace(
        data_loader=data_loader,
        record_metric=lambda name, value, reduce_type: recorded.append((name, value, reduce_type)),
    )
    callback = stage2._DataErrorMonitorCallback()
    callback.trainer = trainer

    callback.post_train_batch()

    assert recorded == [("data/errors total", 3, stage2.ReduceType.sum)]


def test_stage2_runtime_preserves_pinned_dataset_stack_and_quiets_dynamo_logs():
    stage2 = _load_stage2_module()
    launch = BeakerLaunchConfig(
        name="stage2-runtime-test",
        cmd=["true"],
        env_vars=[BeakerEnvVar(name="EXPLICIT_SETTING", value="kept")],
    )

    stage2._configure_launch_runtime(launch)

    env = {item.name: item.value for item in launch.env_vars}
    preset = get_preset("olmo-ddp")
    assert launch.beaker_image == preset.beaker_image
    assert launch.post_setup == preset.post_setup
    assert launch.priority == "urgent"
    assert launch.min_runtime == "8h"
    assert "pip install" not in (launch.post_setup or "")
    assert env["EXPLICIT_SETTING"] == "kept"
    assert env["TORCHINDUCTOR_COMPILE_THREADS"] == "8"
    assert env["TORCH_LOGS"] == "-dynamo"

    restored = BeakerLaunchConfig.from_dict(launch.as_config_dict())
    assert restored.priority == "urgent"
    assert restored.min_runtime == "8h"


def test_stage2_batch_geometry_rejects_partial_sequence_override():
    stage2 = _load_stage2_module()
    config = SimpleNamespace(
        train_module=SimpleNamespace(max_sequence_length=8192, rank_microbatch_size=8192),
        collator=SimpleNamespace(pad_sequence_length=16384),
        global_batch_size=128 * 8192,
        global_batch_instances=128,
        rank_microbatch_instances=1,
    )

    with pytest.raises(ValueError, match="must update both"):
        stage2._validate_batch_geometry(config)


def test_stage2_batch_geometry_validates_runtime_accumulation():
    stage2 = _load_stage2_module()
    config = SimpleNamespace(
        train_module=SimpleNamespace(max_sequence_length=8192, rank_microbatch_size=8192),
        collator=SimpleNamespace(pad_sequence_length=8192),
        global_batch_size=128 * 8192,
        global_batch_instances=128,
        rank_microbatch_instances=1,
    )

    stage2._validate_batch_geometry(config, dp_world_size=16)
    config.train_module.rank_microbatch_size = 3 * 8192
    with pytest.raises(ValueError, match="rank microbatch"):
        stage2._validate_batch_geometry(config, dp_world_size=16)


@pytest.mark.parametrize(
    ("global_batch_size", "rank_microbatch_size", "match"),
    [
        (128 * 8192, 2 * 8192, "rank microbatch"),
        (256 * 8192, 8192, "global batch"),
    ],
)
def test_stage2_batch_geometry_rejects_silent_sequence_count_changes(
    global_batch_size, rank_microbatch_size, match
):
    stage2 = _load_stage2_module()
    config = SimpleNamespace(
        train_module=SimpleNamespace(
            max_sequence_length=8192,
            rank_microbatch_size=rank_microbatch_size,
        ),
        collator=SimpleNamespace(pad_sequence_length=8192),
        global_batch_size=global_batch_size,
        global_batch_instances=128,
        rank_microbatch_instances=1,
    )

    with pytest.raises(ValueError, match=match):
        stage2._validate_batch_geometry(config)


def test_stage2_batch_geometry_allows_explicit_small_smoke():
    stage2 = _load_stage2_module()
    config = SimpleNamespace(
        train_module=SimpleNamespace(max_sequence_length=8192, rank_microbatch_size=8192),
        collator=SimpleNamespace(pad_sequence_length=8192),
        global_batch_size=8 * 8192,
        global_batch_instances=8,
        rank_microbatch_instances=1,
    )

    stage2._validate_batch_geometry(config, dp_world_size=1)


def test_stage2_rejects_fixed_artifact_overrides():
    stage2 = _load_stage2_module()
    config = SimpleNamespace(
        base_checkpoint=stage2.BASE_CHECKPOINT,
        vision_model_id=stage2.VISION_MODEL_ID,
        vision_revision=stage2.VISION_REVISION,
        tokenizer_id="different-tokenizer",
        hf_cache_dir=stage2.HF_CACHE_DIR,
    )

    with pytest.raises(ValueError, match="tokenizer_id is fixed"):
        stage2._validate_fixed_artifacts(config)


def test_stage2_required_run_name_guard_accepts_exact_run_identity():
    stage2 = _load_stage2_module()
    run_name = "s002-stage2-v9-pilot-bounded-errors-5a81c40c"
    config = SimpleNamespace(
        required_run_name=run_name,
        trainer=SimpleNamespace(
            save_folder=f"{stage2.EXPERIMENT_ROOT}/checkpoints/{run_name}",
            callbacks={"wandb": SimpleNamespace(name=run_name)},
        ),
    )

    stage2._validate_required_run_name(config, run_name)


@pytest.mark.parametrize(
    ("positional_name", "save_folder_name", "wandb_name", "match"),
    [
        ("wrong-run", None, None, "requires run name"),
        (None, "wrong-folder", None, "save folder must match"),
        (None, None, "wrong-wandb", "W&B name must match"),
    ],
)
def test_stage2_required_run_name_guard_rejects_mismatched_identity(
    positional_name, save_folder_name, wandb_name, match
):
    stage2 = _load_stage2_module()
    required = "s002-stage2-v9-pilot-bounded-errors-5a81c40c"
    config = SimpleNamespace(
        required_run_name=required,
        trainer=SimpleNamespace(
            save_folder=(f"{stage2.EXPERIMENT_ROOT}/checkpoints/{save_folder_name or required}"),
            callbacks={"wandb": SimpleNamespace(name=wandb_name or required)},
        ),
    )

    with pytest.raises(ValueError, match=match):
        stage2._validate_required_run_name(config, positional_name or required)


def test_stage2_fast_vision_validation_uses_olmo3_held_out_tasks(monkeypatch):
    stage2 = _load_stage2_module()
    captured: Any = {"loaders": [], "evaluators": []}

    for dataset_config in (
        stage2.PixMoCapDatasetConfig,
        stage2.PixMoCountDatasetConfig,
        stage2.PixMoPointsDatasetConfig,
    ):
        monkeypatch.setattr(dataset_config, "build", lambda self, tokenizer: self)

    class Loader:
        def __init__(self, dataset, collator, **kwargs):
            self.dataset = dataset
            self.collator = collator
            captured["loaders"].append((dataset, kwargs))

    class Evaluator:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]
            captured["evaluators"].append(kwargs)

    class Callback:
        def __init__(self, **kwargs):
            captured["callback_kwargs"] = kwargs

    trainer = SimpleNamespace(
        work_dir=Path("/tmp/stage2-fast-vision-test"),
        device="cpu",
        dp_process_group=None,
        add_callback=lambda name, callback: captured.update(callback_name=name, callback=callback),
    )
    token_ids = SimpleNamespace(im_patch_id=7, image_token_ids=(7, 8, 9))
    config = SimpleNamespace(
        token_ids=token_ids,
        message_format="olmo3_chat",
        train_module=SimpleNamespace(max_sequence_length=8192),
        fast_vision_eval_interval=50,
        fast_vision_eval_examples=32,
        fast_vision_eval_seed=stage2.FAST_VISION_EVAL_SEED,
    )
    collator = SimpleNamespace(pad_sequence_length=8192)
    monkeypatch.setattr(stage2, "MultimodalDataLoader", Loader)
    monkeypatch.setattr(stage2, "MultimodalLMEvaluator", Evaluator)
    monkeypatch.setattr(stage2, "EvaluatorCallback", Callback)

    stage2._add_fast_vision_validation_callback(
        trainer,
        tokenizer=object(),
        config=config,
        collator=collator,
        dp_world_size=2,
        dp_rank=0,
    )

    wrapped_datasets = [dataset for dataset, _ in captured["loaders"]]
    datasets = [dataset.dataset for dataset in wrapped_datasets]
    assert [evaluator["name"] for evaluator in captured["evaluators"]] == [
        "pixmo-cap-caption-validation",
        "pixmo-count-validation",
        "pixmo-points-validation",
    ]
    assert all(dataset.split == "validation" for dataset in datasets)
    assert datasets[0].mode == "caption"
    assert datasets[0].max_sequence_length == 8192
    assert datasets[1].counting is True
    assert datasets[2].kind == "basic"
    assert datasets[2].both_mode == "duplicate"
    assert all(dataset.message_format == "olmo3_chat" for dataset in datasets)
    assert all(dataset.loss_token_weighting == "none" for dataset in datasets)
    assert all(dataset.max_sequence_length == 8192 for dataset in wrapped_datasets)
    assert all(kwargs["shuffle"] is False for _, kwargs in captured["loaders"])
    assert all(kwargs["global_batch_size"] == 2 * 8192 for _, kwargs in captured["loaders"])
    assert all(evaluator["deterministic"] is True for evaluator in captured["evaluators"])
    assert captured["callback_name"] == "fast_vision_validation"
    assert captured["callback_kwargs"]["eval_interval"] == 50
    assert captured["callback_kwargs"]["eval_duration"] == stage2.Duration.steps(16)
    assert captured["callback_kwargs"]["eval_on_startup"] is False
    assert captured["callback_kwargs"]["eval_on_finish"] is False


def test_stage2_fast_vision_validation_can_be_disabled():
    stage2 = _load_stage2_module()
    trainer = SimpleNamespace(
        add_callback=lambda *_args, **_kwargs: pytest.fail("callback should be disabled")
    )

    stage2._add_fast_vision_validation_callback(
        trainer,
        tokenizer=object(),
        config=SimpleNamespace(fast_vision_eval_interval=None),
        collator=object(),
        dp_world_size=2,
        dp_rank=0,
    )


def test_stage2_fast_vision_validation_rejects_non_divisible_example_count():
    stage2 = _load_stage2_module()
    config = SimpleNamespace(
        token_ids=object(),
        message_format="olmo3_chat",
        train_module=SimpleNamespace(max_sequence_length=8192),
        fast_vision_eval_interval=2000,
        fast_vision_eval_examples=31,
        fast_vision_eval_seed=stage2.FAST_VISION_EVAL_SEED,
    )

    with pytest.raises(ValueError, match="must be divisible"):
        stage2._add_fast_vision_validation_callback(
            trainer=object(),
            tokenizer=object(),
            config=config,
            collator=SimpleNamespace(pad_sequence_length=8192),
            dp_world_size=2,
            dp_rank=0,
        )


@pytest.mark.parametrize(
    ("profile_name", "max_step"),
    [
        ("stage2_ep8_2node_image_only_v9_to50.yaml", 50),
        ("stage2_ep8_2node_image_only_v9_resume_to200.yaml", 200),
    ],
)
def test_stage2_pilot_profiles_are_submission_safe_dry_run_inputs(profile_name, max_step):
    stage2 = _load_stage2_module()
    profile_path = Path(__file__).parents[3] / "configs" / "vision_moe" / profile_name

    profile, overrides = stage2._load_beaker_test_config([f"--beaker-test-config={profile_path}"])

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
    assert "--mixture=image-only-v9" in overrides
    assert "--message_format=olmo3_chat" in overrides
    assert "--collator.pad_sequence_length=16384" in overrides
    assert "--train_module.max_sequence_length=16384" in overrides
    assert "--global_batch_size=2097152" in overrides
    assert "--global_batch_instances=128" in overrides
    assert "--train_module.rank_microbatch_size=16384" in overrides
    assert "--rank_microbatch_instances=1" in overrides
    assert "--train_module.ep_config.degree=8" in overrides
    assert "--train_module.optim.sigma_factor=12" in overrides
    assert "--train_module.diagnostics_interval=10" in overrides
    assert "--pack_max_crops=45" in overrides
    assert "--max_consecutive_data_errors=10" in overrides
    assert "--max_total_data_errors=1000" in overrides
    assert f"--trainer.max_duration.value={max_step}" in overrides
    assert "--trainer.max_duration.unit=steps" in overrides
    assert "--train_module.scheduler.schedulers.connector.t_max=30000" in overrides
    assert "--train_module.scheduler.schedulers.vision.t_max=30000" in overrides
    assert "--train_module.scheduler.default.t_max=30000" in overrides
    assert "--fast_vision_eval_interval=50" in overrides
    assert not any("fast_language" in override for override in overrides)
    assert "--trainer.callbacks.wandb.auto_resume=true" in overrides
    assert "--trainer.callbacks.checkpointer.pre_train_checkpoint=false" in overrides
    assert "--trainer.callbacks.checkpointer.save_interval=null" in overrides
    assert "--trainer.callbacks.checkpointer.ephemeral_save_interval=25" in overrides
    assert "--trainer.callbacks.checkpointer.fixed_steps=[50,200]" in overrides
    assert "--trainer.callbacks.checkpointer.max_checkpoints=2" in overrides
    assert f"--trainer.load_path={stage2.DEFAULT_LOAD_PATH}" in overrides
    assert "--trainer.load_strategy=always" in overrides
    assert "--trainer.load_optim_state=false" in overrides
    assert "--trainer.load_trainer_state=false" in overrides
    assert stage2.STAGE2_MOE_CAPACITY_FACTOR == 2.0

    config = SimpleNamespace(
        launch=SimpleNamespace(
            num_nodes=1,
            num_gpus=1,
            workspace=None,
            clusters=[],
            budget=None,
            priority="normal",
            min_runtime=None,
            description=None,
        )
    )
    stage2._apply_beaker_test_config(config, profile)
    assert config.launch.num_nodes == 2
    assert config.launch.num_gpus == 8
    assert config.launch.workspace == "ai2/molmofication"
    assert config.launch.clusters == ["ai2/holmes"]
    assert config.launch.budget == "ai2/oe-other"
    assert config.launch.priority == "urgent"
    assert config.launch.min_runtime == "8h"


def test_stage2_resume_to400_profile_is_guarded_full_state_continuation():
    stage2 = _load_stage2_module()
    run_name = "s002-stage2-v9-pilot-bounded-errors-5a81c40c"
    profile_path = (
        Path(__file__).parents[3]
        / "configs"
        / "vision_moe"
        / "stage2_ep8_2node_image_only_v9_resume_to400.yaml"
    )

    profile, overrides = stage2._load_beaker_test_config([f"--beaker-test-config={profile_path}"])

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
    expected_overrides = {
        f"--required_run_name={run_name}",
        "--mixture=image-only-v9",
        "--message_format=olmo3_chat",
        "--mmfinereason_rate=0.0",
        "--finevision_rate=0.0",
        "--collator.pad_sequence_length=16384",
        "--train_module.max_sequence_length=16384",
        "--global_batch_size=2097152",
        "--global_batch_instances=128",
        "--train_module.rank_microbatch_size=16384",
        "--rank_microbatch_instances=1",
        "--train_module.ep_config.degree=8",
        "--train_module.optim.sigma_factor=12",
        "--train_module.diagnostics_interval=10",
        "--pack_sequences=true",
        "--pack_max_crops=45",
        "--pack_buffer_size=48",
        "--pack_image_weight=30.0",
        "--max_consecutive_data_errors=10",
        "--max_total_data_errors=1000",
        f"--trainer.load_path={stage2.DEFAULT_LOAD_PATH}",
        "--trainer.load_strategy=always",
        "--trainer.load_optim_state=false",
        "--trainer.load_trainer_state=false",
        "--trainer.max_duration.value=400",
        "--trainer.max_duration.unit=steps",
        "--train_module.scheduler.schedulers.connector.t_max=30000",
        "--train_module.scheduler.schedulers.vision.t_max=30000",
        "--train_module.scheduler.default.t_max=30000",
        "--model.lm.recompute_each_block=true",
        "--fast_vision_eval_interval=200",
        "--fast_vision_eval_examples=32",
        f"--trainer.callbacks.wandb.name={run_name}",
        "--trainer.callbacks.wandb.group=s002-stage2-image-only-v9-pilot",
        "--trainer.callbacks.wandb.auto_resume=true",
        "--trainer.callbacks.checkpointer.pre_train_checkpoint=false",
        "--trainer.callbacks.checkpointer.save_interval=null",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=100",
        "--trainer.callbacks.checkpointer.remove=all_non_permanent",
        "--trainer.callbacks.checkpointer.fixed_steps=[50,200,400]",
        "--trainer.callbacks.checkpointer.max_checkpoints=3",
    }
    assert expected_overrides <= set(overrides)
    assert not any("fast_language" in override for override in overrides)


def test_stage2_pilot_dry_run_cli_overrides_take_precedence():
    stage2 = _load_stage2_module()
    profile_path = (
        Path(__file__).parents[3]
        / "configs"
        / "vision_moe"
        / "stage2_ep8_2node_image_only_v9_to50.yaml"
    )
    cli_override = "--trainer.max_duration.value=1"

    _, overrides = stage2._load_beaker_test_config(
        [f"--beaker-test-config={profile_path}", cli_override]
    )

    assert overrides[-1] == cli_override
    assert all(not override.startswith("--beaker-test-config=") for override in overrides)
