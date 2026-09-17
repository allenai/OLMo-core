"""Tests for native OLMoDDP construction, optimizer state and checkpoint behavior."""

from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest
import torch

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType, PipelineScheduleType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.optim import OLMoDDPOptimizerConfig
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizer
from olmo_core.testing import requires_multi_gpu, run_distributed_test
from olmo_core.train.train_module import OLMoDDPTrainModule, OLMoDDPTrainModuleConfig
from olmo_core.train.train_module.transformer import (
    MoEV2TransformerTrainModuleConfig,
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
    TransformerPipelineParallelConfig,
)
from olmo_core.train.train_module.transformer import (
    ddp_train_module as train_module_impl,
)


class _ValidationPassed(Exception):
    pass


class _RecordingTrainModuleConfig(OLMoDDPTrainModuleConfig):
    def _build_train_module(self, **kwargs):
        return kwargs


@pytest.mark.parametrize("via_config", [False, True])
@pytest.mark.parametrize("optimizer_limit", [None, 0.75])
@pytest.mark.parametrize("module_limit", [None, 0.0, 0.25, 2.0])
def test_native_gradient_clipping_override_preserves_optimizer_config(
    monkeypatch, via_config, optimizer_limit, module_limit
):
    class RecordingOptimizer:
        def __init__(self, param_groups, **kwargs):
            self.param_groups = param_groups
            self.max_grad_norm = kwargs["max_grad_norm"]
            self.clip_grad_norm_by_scheduler_group = kwargs["clip_grad_norm_by_scheduler_group"]

    monkeypatch.setattr(
        OLMoDDPOptimizerConfig, "optimizer", classmethod(lambda cls: RecordingOptimizer)
    )
    monkeypatch.setattr(train_module_impl, "is_distributed", lambda: True)
    monkeypatch.setattr(
        OLMoDDPTrainModule,
        "_build_world_mesh",
        lambda self, **kwargs: self.world_mesh.update({"dense": {}}),
    )
    monkeypatch.setattr(
        OLMoDDPTrainModule, "parallelize_and_init_model", lambda self, model, **kwargs: [model]
    )
    model = torch.nn.Linear(2, 2)
    model._olmo_ddp_compatible = True
    model.has_grad_accum_fp32_buffer = False
    model.num_flops_per_token = lambda _: 0
    optim = OLMoDDPOptimizerConfig(clip_grad_norm_by_scheduler_group=True)
    if optimizer_limit is not None:
        optim.max_grad_norm = optimizer_limit
    before = optim.as_dict()
    kwargs = dict(
        optim=optim,
        max_grad_norm=module_limit,
        rank_microbatch_size=16,
        max_sequence_length=8,
        dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp),
    )
    if via_config:
        module = OLMoDDPTrainModuleConfig(**kwargs).build(model, device=torch.device("cpu"))
    else:
        module = OLMoDDPTrainModule(model=model, device=torch.device("cpu"), **kwargs)

    assert module.optim.max_grad_norm == (
        optim.max_grad_norm if module_limit is None else module_limit
    )
    assert module.optim.clip_grad_norm_by_scheduler_group is True
    assert optim.as_dict() == before


@pytest.mark.parametrize("determinism_check", [None, "default", "none"])
def test_native_activation_checkpointing_forwards_determinism_check(determinism_check):
    module = object.__new__(OLMoDDPTrainModule)
    module.world_mesh = {"dense": {"dp": None}}
    module.dense_dp_cp_group = None
    module.max_sequence_length = 8
    module.rank_microbatch_size = 16
    module.init_model_weights = Mock()
    module._cast_to_fwd_bwd_precision = Mock()
    model = SimpleNamespace(
        _olmo_ddp_compatible=True,
        apply_activation_checkpointing=Mock(),
        refresh_rowwise_fp8_cache=Mock(),
    )
    config = (
        TransformerActivationCheckpointingConfig(determinism_check=determinism_check)
        if determinism_check is not None
        else None
    )

    assert module.parallelize_and_init_model(
        model,
        dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp),
        ac_config=config,
        eval_only=True,
    ) == [model]

    if config is None:
        model.apply_activation_checkpointing.assert_not_called()
    else:
        model.apply_activation_checkpointing.assert_called_once_with(
            config.mode,
            block_interval=config.block_interval,
            modules=config.modules,
            activation_memory_budget=config.activation_memory_budget,
            determinism_check=determinism_check,
        )


def _eval_only_parallelism_kwargs(parallelism):
    kwargs = {"dp_config": TransformerDataParallelConfig(name=DataParallelType.ddp)}
    if parallelism in ("pp", "pp_cp"):
        kwargs["pp_config"] = TransformerPipelineParallelConfig(
            degree=2,
            schedule=PipelineScheduleType.custom_interleaved_1F1B,
            use_custom_stage_implementation=True,
        )
    if parallelism in ("cp", "pp_cp"):
        kwargs["cp_config"] = TransformerContextParallelConfig.zig_zag(degree=2)
    if parallelism == "ep":
        kwargs["ep_config"] = TransformerExpertParallelConfig(degree=2)
    return kwargs


@pytest.mark.parametrize(
    "config_type",
    [OLMoDDPTrainModuleConfig, MoEV2TransformerTrainModuleConfig, _RecordingTrainModuleConfig],
    ids=["native", "legacy", "subclass"],
)
@pytest.mark.parametrize("eval_only", [False, True])
@pytest.mark.parametrize("parallelism", ["dp", "ep", "pp", "cp", "pp_cp"])
def test_native_eval_only_config_validates_before_building_module(
    monkeypatch, config_type, eval_only, parallelism
):
    build_module = Mock(side_effect=lambda **kwargs: kwargs)
    monkeypatch.setattr(train_module_impl, "OLMoDDPTrainModule", build_module)
    kwargs = _eval_only_parallelism_kwargs(parallelism)
    config = config_type(
        rank_microbatch_size=16,
        max_sequence_length=8,
        optim=OLMoDDPOptimizerConfig(),
        **kwargs,
    )
    model = object()

    if eval_only and parallelism in ("pp", "cp", "pp_cp"):
        with pytest.raises(
            OLMoConfigurationError, match="eval_only=True.*pipeline or context parallelism"
        ):
            config.build(model, eval_only=eval_only)
        build_module.assert_not_called()
    else:
        result = config.build(model, eval_only=eval_only)
        assert result["model"] is model
        assert result["eval_only"] is eval_only
        for key, value in kwargs.items():
            assert result[key] is value
        assert build_module.call_count == (0 if config_type is _RecordingTrainModuleConfig else 1)


@pytest.mark.parametrize("eval_only", [False, True])
@pytest.mark.parametrize("parallelism", ["dp", "ep", "pp", "cp", "pp_cp"])
def test_native_eval_only_constructor_validates_before_model_and_device_initialization(
    monkeypatch, eval_only, parallelism
):
    model = SimpleNamespace(_olmo_ddp_compatible=True)
    check_model = Mock(wraps=train_module_impl._is_olmo_ddp_compatible)
    initialize_device = Mock(side_effect=_ValidationPassed)
    monkeypatch.setattr(train_module_impl, "_is_olmo_ddp_compatible", check_model)
    monkeypatch.setattr(train_module_impl, "get_default_device", initialize_device)
    kwargs = _eval_only_parallelism_kwargs(parallelism)

    invalid = eval_only and parallelism in ("pp", "cp", "pp_cp")
    expected_error = OLMoConfigurationError if invalid else _ValidationPassed
    match = "eval_only=True.*pipeline or context parallelism" if invalid else None
    with pytest.raises(expected_error, match=match):
        OLMoDDPTrainModule(
            model=model,
            optim=OLMoDDPOptimizerConfig(),
            rank_microbatch_size=16,
            max_sequence_length=8,
            eval_only=eval_only,
            **kwargs,
        )
    assert check_model.call_count == (0 if invalid else 1)
    assert initialize_device.call_count == (0 if invalid else 1)


@pytest.mark.parametrize("tbo", [False, True])
@pytest.mark.parametrize("microbatch_instances", [0, 1, 2, 3, 4])
def test_tbo_config_validates_instances_before_device_initialization(
    monkeypatch, tbo, microbatch_instances
):
    model = SimpleNamespace(_olmo_ddp_compatible=True, tbo=tbo)
    initialize_device = Mock(side_effect=_ValidationPassed)
    monkeypatch.setattr(train_module_impl, "get_default_device", initialize_device)

    invalid = tbo and (microbatch_instances <= 0 or microbatch_instances % 2 != 0)
    expected_error = OLMoConfigurationError if invalid else _ValidationPassed
    with pytest.raises(expected_error):
        OLMoDDPTrainModule(
            model=model,
            optim=OLMoDDPOptimizerConfig(lr=1e-3),
            rank_microbatch_size=microbatch_instances * 8,
            max_sequence_length=8,
        )
    assert initialize_device.call_count == (0 if invalid else 1)


@pytest.mark.parametrize("tbo", [False, True])
@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize(
    "batch_size,sequence_length,microbatch_tokens,num_pipeline_microbatches,valid",
    [
        (0, 8, 16, None, False),
        (1, 8, 16, None, False),
        (2, 8, 16, None, True),
        (3, 8, 16, None, False),
        (4, 8, 16, None, True),
        (2, 8, 24, None, True),
        (4, 8, 24, None, False),
        (6, 8, 24, None, False),
        (5, 8, 32, None, False),
        (6, 8, 32, None, True),
        (4, 10, 24, None, True),
        (8, 4, 24, None, True),
        (7, 4, 24, None, False),
        (2, 32, 16, None, False),
        (2, 8, 16, 2, False),
        (4, 8, 16, 2, True),
        (6, 8, 16, 2, False),
        (6, 8, 16, 4, False),
        (8, 8, 16, 4, True),
        (10, 8, 16, 4, False),
    ],
)
def test_tbo_train_batch_validates_all_chunks_before_processing(
    tbo, dry_run, batch_size, sequence_length, microbatch_tokens, num_pipeline_microbatches, valid
):
    module = object.__new__(OLMoDDPTrainModule)
    module.optim = object()
    module.rank_microbatch_size = microbatch_tokens
    module._pp_config = object() if num_pipeline_microbatches is not None else None
    module._train_pp_schedule = SimpleNamespace(num_microbatches=num_pipeline_microbatches)
    model = SimpleNamespace(tbo=tbo, train=Mock(side_effect=_ValidationPassed))
    module.model_parts = [model]
    batch = {"input_ids": torch.zeros(batch_size, sequence_length, dtype=torch.long)}

    invalid = tbo and not valid
    expected_error = OLMoConfigurationError if invalid else _ValidationPassed
    with pytest.raises(expected_error):
        module.train_batch(batch, dry_run=dry_run)
    assert model.train.call_count == (0 if invalid else 1)
    assert set(batch) == {"input_ids"}


@pytest.mark.parametrize("tbo", [False, True])
@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("batch_size", [0, 1, 2, 3, 4])
def test_tbo_eval_batch_validates_instances_before_processing(tbo, pipeline, batch_size):
    module = object.__new__(OLMoDDPTrainModule)
    module._cp_config = module._tp_config = None
    module._pp_config = object() if pipeline else None
    module.model_parts = [SimpleNamespace(tbo=tbo)]
    module._prepare_batch = Mock(side_effect=_ValidationPassed)
    batch = {"input_ids": torch.zeros(batch_size, 8, dtype=torch.long)}

    invalid = tbo and (pipeline or batch_size <= 0 or batch_size % 2 != 0)
    expected_error = OLMoConfigurationError if invalid else _ValidationPassed
    with pytest.raises(expected_error):
        module.eval_batch(batch)
    assert module._prepare_batch.call_count == (0 if invalid else 1)


class _MetricTrainerStub:
    def __init__(self):
        self.global_step = 1
        self.metrics = {}

    def record_metric(self, name, value, *, namespace=None, **kwargs):
        del kwargs
        self.metrics[f"{namespace}/{name}" if namespace else name] = value


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("explicit_labels", [False, True])
@pytest.mark.parametrize("document_lengths", [False, True])
def test_prepare_batch_preserves_label_precedence_and_pipeline_kwargs(
    pipeline, explicit_labels, document_lengths
):
    module = object.__new__(OLMoDDPTrainModule)
    module._pp_config = object() if pipeline else None
    input_ids = torch.tensor([[1, 2, 3]])
    batch_labels = torch.tensor([[2, 3, -100]])
    labels = torch.tensor([[4, 5, -100]]) if explicit_labels else None
    batch = {"input_ids": input_ids, "labels": batch_labels, "loss_masks": torch.ones(1, 3)}
    if document_lengths:
        batch.update(doc_lens=torch.tensor([[3]]), max_doc_lens=[3])

    actual_input, actual_labels, kwargs = module._prepare_batch(batch, labels)

    assert actual_input is input_ids
    assert actual_labels is (labels if explicit_labels else batch_labels)
    assert "input_ids" not in batch
    assert ("labels" in batch) is explicit_labels
    if pipeline:
        assert set(kwargs) == ({"doc_lens", "max_doc_lens"} if document_lengths else set())
    else:
        assert kwargs is batch
        assert "loss_masks" in kwargs


def test_nonfinite_gradient_diagnostics_use_trainer_step_without_dump_flags(monkeypatch, caplog):
    monkeypatch.setenv("OLMO_DDP_DEBUG_NONFINITE_GRAD", "1")
    monkeypatch.delenv("OLMO_DEBUG_DUMP_DIR", raising=False)
    module = object.__new__(OLMoDDPTrainModule)
    module._trainer = _MetricTrainerStub()
    module._trainer.global_step = 41
    module.scheduler = None
    module.model_parts = []
    optim = SimpleNamespace(
        latest_grad_norm=None,
        _iter_local_grads=lambda: iter(()),
        _local_total_norm=lambda _: torch.tensor(0.0),
    )
    optim.step = lambda: OLMoDDPOptimizer._maybe_debug_nan_inf_grad_norm(
        optim, torch.tensor(float("nan")), [], [], [], []
    )
    module.optim = optim

    module.optim_step()

    assert optim._debug_global_step == 41
    assert "rank 0 step 41" in caplog.text


def test_moe_v2_train_module_config_roundtrips():
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=1024,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3, foreach_chunk_size=50_000_000),
    )
    restored = OLMoDDPTrainModuleConfig.from_dict(config.as_dict())
    assert restored == config
    assert restored.optim.lr == 1e-3
    assert restored.optim.foreach_chunk_size == 50_000_000


def test_moe_v2_train_module_config_roundtrips_with_parallelism():
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=1024,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp, reduce_grads_in_fp32=False
        ),
        pp_config=TransformerPipelineParallelConfig(degree=2),
    )
    restored = OLMoDDPTrainModuleConfig.from_dict(config.as_dict())
    assert restored == config
    assert restored.dp_config is not None and restored.dp_config.reduce_grads_in_fp32 is False
    assert restored.pp_config is not None and restored.pp_config.degree == 2


def _tiny_model_config(
    *,
    d_model: int = 64,
    n_layers: int = 2,
    dtype: DType = DType.float32,
    router_bias_gamma: Optional[float] = None,
) -> OLMoDDPModelConfig:
    layer_norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=dtype)
    return OLMoDDPModelConfig(
        init_seed=0,
        d_model=d_model,
        recompute_each_block=False,
        vocab_size=128,
        n_layers=n_layers,
        name=TransformerType.moe_fused_v2,
        block=OLMoDDPTransformerBlockConfig(
            name=TransformerBlockType.moe_fused_v2,
            attention=AttentionConfig(
                name=AttentionType.default,
                n_heads=4,
                bias=False,
                use_flash=False,
                dtype=dtype,
            ),
            routed_experts=RoutedExpertsConfig(
                d_model=d_model, hidden_size=128, num_experts=4, bias=False, dtype=dtype
            ),
            routed_experts_router=MoERouterConfigV2(
                d_model=d_model,
                num_experts=4,
                top_k=2,
                dtype=dtype,
                bias_gamma=router_bias_gamma,
            ),
            shared_experts=None,
            layer_norm=layer_norm,
        ),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
    )


def test_frozen_checkpoint_skips_optimizer_owned_fp8_anchors():
    class _WeightStore:
        optimizer_enabled = True

        def __init__(self, anchor_param):
            self.anchor_param = anchor_param

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(4), requires_grad=False)
            self.store = _WeightStore(self.anchor)

        def named_fp8_weight_stores(self):
            return [("anchor", self.store)]

    train_module = object.__new__(OLMoDDPTrainModule)
    model = _Model()
    object.__setattr__(train_module, "model_parts", [model])

    assert train_module._optimizer_owned_anchor_param_ids() == {id(model.anchor)}
    assert train_module._frozen_model_param_state_dict() == {}
    assert (
        train_module._frozen_checkpoint_model_param_state_dict_for_load(
            {"frozen_model.anchor", "anchor.main"}
        )
        == {}
    )


def _run_construct_no_ep():
    model = _tiny_model_config().build(init_device="cpu")
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=512,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp),
    )
    # eval_only=True skips the optimizer build (its fp32-master-param setup is exercised on GPU);
    # this covers the world-mesh build + data-parallel wrapping with no expert parallelism.
    train_module = config.build(model, device=torch.device("cpu"), eval_only=True)

    assert len(train_module.model_parts) == 1  # no pipeline parallelism
    assert train_module.dp_world_size == 2
    assert train_module.world_mesh["dense"] is not None
    assert train_module.moe_mesh is None  # no expert parallelism


def test_moe_v2_train_module_construction_no_ep():
    run_distributed_test(
        _run_construct_no_ep,
        world_size=2,
        backend="gloo",
        start_method="spawn",
    )


def _run_construct_ep():
    # bf16 params → the fused optimizer maintains fp32 master params (its realistic config); a pure
    # fp32 model instead takes the optimizer's "expect fp32 param" branch.
    model = _tiny_model_config(dtype=DType.bfloat16).build(init_device="cuda")
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=512,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp),
        ep_config=TransformerExpertParallelConfig(degree=2),
    )
    # Full build (eval_only=False): wires expert parallelism through the train module (moe mesh +
    # apply_ep sharding the experts across the two ranks + DP wrapping) and builds the optimizer.
    train_module = config.build(model, device=torch.device("cuda"), eval_only=False)

    assert len(train_module.model_parts) == 1  # no pipeline parallelism
    assert train_module.moe_mesh is not None
    assert train_module.ep_mp_group is not None
    assert train_module.optim is not None
    assert train_module.num_flops_per_token(seq_len=512) > 0


@requires_multi_gpu
def test_moe_v2_train_module_construction_ep():
    run_distributed_test(
        _run_construct_ep,
        world_size=2,
        backend="nccl",
        start_method="spawn",
    )


def test_moe_v2_train_module_config_reset_optimizer_states_roundtrips():
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=1024,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        reset_optimizer_states_on_resume=True,
    )
    restored = OLMoDDPTrainModuleConfig.from_dict(config.as_dict())
    assert restored == config
    assert restored.reset_optimizer_states_on_resume is True
    # The resume flag is distinct from the generic on-load flag, which stays at its default.
    assert restored.reset_optimizer_states_on_load is False


def _run_rejects_per_microbatch_allreduce():
    model = _tiny_model_config().build(init_device="cpu")
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=512,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp, only_allreduce_last_microbatch=False
        ),
    )
    # MultiGroupDistributedDataParallel reduces each bucket once per accumulation window, so
    # per-micro-batch all-reduce is unsupported and must be rejected up front.
    with pytest.raises(OLMoConfigurationError, match="only_allreduce_last_microbatch"):
        config.build(model, device=torch.device("cpu"), eval_only=True)


def test_moe_v2_train_module_rejects_per_microbatch_allreduce():
    run_distributed_test(
        _run_rejects_per_microbatch_allreduce,
        world_size=2,
        backend="gloo",
        start_method="spawn",
    )


_MOMENT_SUFFIXES = (".exp_avg", ".exp_avg_sq")


def _build_ddp_train_module_for_checkpoint(
    *, router_bias_gamma: Optional[float] = None, ep_degree: Optional[int] = None
):
    model = _tiny_model_config(dtype=DType.bfloat16, router_bias_gamma=router_bias_gamma).build(
        init_device="cuda"
    )
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=512,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp),
        ep_config=(
            TransformerExpertParallelConfig(degree=ep_degree) if ep_degree is not None else None
        ),
    )
    return config.build(model, device=torch.device("cuda"), eval_only=False)


def _run_resume_resets_optimizer_moments(save_dir):
    # Save a checkpoint carrying non-zero optimizer moments, then verify that the resume flag
    # (threaded through as reset_optimizer_states_on_load) actually controls whether those moments
    # are restored or discarded on load.
    tm = _build_ddp_train_module_for_checkpoint()
    assert tm.optim is not None
    for key, state in tm.optim.states.items():
        if key.endswith(_MOMENT_SUFFIXES):
            state.to_local().fill_(0.5)
    tm.save_state_dict_direct(save_dir)

    # Reset on load: only the main params are restored, so freshly zero-initialized moments stay zero.
    tm_reset = _build_ddp_train_module_for_checkpoint()
    assert tm_reset.optim is not None
    tm_reset.load_state_dict_direct(save_dir, reset_optimizer_states_on_load=True)
    for key, state in tm_reset.optim.states.items():
        if key.endswith(_MOMENT_SUFFIXES):
            assert torch.count_nonzero(state.to_local()) == 0, key

    # No reset: the saved (non-zero) moments are restored.
    tm_restore = _build_ddp_train_module_for_checkpoint()
    assert tm_restore.optim is not None
    tm_restore.load_state_dict_direct(save_dir, reset_optimizer_states_on_load=False)
    restored_any_moment = any(
        key.endswith(_MOMENT_SUFFIXES) and torch.count_nonzero(state.to_local()) > 0
        for key, state in tm_restore.optim.states.items()
    )
    assert restored_any_moment


@requires_multi_gpu
def test_moe_v2_train_module_resume_resets_optimizer_moments(tmp_path):
    run_distributed_test(
        _run_resume_resets_optimizer_moments,
        world_size=2,
        backend="nccl",
        start_method="spawn",
        func_args=(str(tmp_path / "checkpoint"),),
    )


def _score_bias_buffers(train_module):
    return {
        name: buf
        for model_part in train_module.model_parts
        for name, buf in model_part.named_buffers()
        if name.endswith("score_bias") and buf is not None
    }


def _run_direct_checkpoint_restores_buffers(save_dir):
    # Persistent buffers (the router's aux-loss-free score_bias) are model state updated outside
    # the optimizer; the direct checkpoint must round-trip them.
    tm = _build_ddp_train_module_for_checkpoint(router_bias_gamma=1e-3)
    mutated = _score_bias_buffers(tm)
    assert mutated, "expected at least one score_bias buffer with bias_gamma set"
    saved = {}
    for name, buf in mutated.items():
        buf.copy_(torch.arange(buf.numel(), device=buf.device, dtype=buf.dtype) + 1.0)
        saved[name] = buf.detach().clone()
    tm.save_state_dict_direct(save_dir)

    tm_restored = _build_ddp_train_module_for_checkpoint(router_bias_gamma=1e-3)
    restored = _score_bias_buffers(tm_restored)
    # Freshly built buffers are zero-initialized, so they must differ before the load.
    for name, buf in restored.items():
        assert torch.count_nonzero(buf) == 0, name
    tm_restored.load_state_dict_direct(save_dir, reset_optimizer_states_on_load=False)
    for name, expected in saved.items():
        torch.testing.assert_close(restored[name], expected)


@requires_multi_gpu
def test_moe_v2_train_module_direct_checkpoint_restores_buffers(tmp_path):
    run_distributed_test(
        _run_direct_checkpoint_restores_buffers,
        world_size=2,
        backend="nccl",
        start_method="spawn",
        func_args=(str(tmp_path / "checkpoint"),),
    )
