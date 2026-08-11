"""Tests for :class:`OLMoDDPTrainModule` config and construction."""

from typing import Optional

import pytest
import torch
import torch.distributed as dist

import olmo_core.train.train_module.transformer.multimodal_train_module as multimodal_train_module
from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig, AttentionType
from olmo_core.nn.ddp import OLMoDDPModel
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe.loss import MoELoadBalancingLossGranularity
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.nn.vision import (
    MultimodalLMConfig,
    MultimodalOLMoDDPModel,
    VisionConnectorConfig,
    VisionEncoderConfig,
    VisionEncoderType,
)
from olmo_core.optim import OLMoDDPOptimizerConfig, OptimGroupOverride
from olmo_core.testing import requires_multi_gpu, run_distributed_test
from olmo_core.train import ReduceType
from olmo_core.train.train_module import (
    MultimodalOLMoDDPTrainModule,
    MultimodalOLMoDDPTrainModuleConfig,
    OLMoDDPTrainModuleConfig,
)
from olmo_core.train.train_module.transformer import (
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
    TransformerPipelineParallelConfig,
)
from olmo_core.train.train_module.transformer.multimodal_train_module import (
    _matched_component_grad_norm_patterns,
    _retain_embedding_gradient_rows,
)


class _MetricTrainerStub:
    def __init__(self):
        self.global_step = 1
        self.metrics = {}

    def record_metric(self, name, value, *, namespace=None, **kwargs):
        del kwargs
        self.metrics[f"{namespace}/{name}" if namespace else name] = value


def test_retain_embedding_gradient_rows_masks_every_other_row():
    grad = torch.arange(20, dtype=torch.float32).reshape(5, 4)

    masked = _retain_embedding_gradient_rows(grad, (1, 3))

    torch.testing.assert_close(masked[[1, 3]], grad[[1, 3]], rtol=0, atol=0)
    torch.testing.assert_close(masked[[0, 2, 4]], torch.zeros(3, 4), rtol=0, atol=0)


def test_component_grad_norm_patterns_only_keep_trainable_components():
    patterns = {
        "vision": ("vision.*", "*vision.*"),
        "LM output head": ("lm.lm_head.w_out.*", "*lm.lm_head.w_out.*"),
        "LM shared experts": ("lm.blocks.*.shared_experts.*",),
    }

    assert _matched_component_grad_norm_patterns(
        patterns,
        {"module.vision.patch_embedding.weight"},
    ) == {"vision": patterns["vision"]}
    assert _matched_component_grad_norm_patterns(
        patterns,
        {
            "module.vision.patch_embedding.weight",
            "module.lm.lm_head.w_out.weight",
        },
    ) == {
        "vision": patterns["vision"],
        "LM output head": patterns["LM output head"],
    }


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


def _tiny_multimodal_model_config(*, dtype: DType = DType.float32) -> MultimodalLMConfig:
    lm = _tiny_model_config(dtype=dtype)
    vision = VisionEncoderConfig(
        name=VisionEncoderType.siglip,
        use_cls_token=False,
        patch_embedding_bias=True,
        use_pre_ln=False,
        image_default_input_size=(28, 28),
        image_patch_size=14,
        image_emb_dim=16,
        image_num_heads=2,
        image_num_key_value_heads=2,
        image_num_layers=1,
        image_head_dim=8,
        image_mlp_dim=32,
        image_num_pos=4,
        dtype=dtype,
    )
    return MultimodalLMConfig(
        lm=lm,
        vision=vision,
        connector=VisionConnectorConfig.from_vision_encoder(
            vision, output_dim=lm.d_model, mlp_hidden_size=32
        ),
        image_patch_token_id=120,
    )


def test_multimodal_olmo_ddp_model_materializes_all_components():
    model = _tiny_multimodal_model_config().build(init_device="meta")
    assert isinstance(model, MultimodalOLMoDDPModel)

    model.init_weights(
        max_seq_len=16,
        max_local_microbatch_size=16,
        device=torch.device("cpu"),
        world_mesh={},
    )
    assert all(not param.is_meta for param in model.parameters())

    for param in model.vision.parameters():
        param.requires_grad_(False)
    model.train()
    assert model.lm.training
    assert model.connector.training
    assert not model.vision.training


def test_multimodal_olmo_ddp_prewarms_forward_only_rowwise_scratch(monkeypatch):
    model = _tiny_multimodal_model_config().build(init_device="meta")
    assert isinstance(model, MultimodalOLMoDDPModel)
    captured = {}

    def prewarm(_self, *args, **kwargs):
        captured.update(args=args, kwargs=kwargs)

    monkeypatch.setattr(OLMoDDPModel, "prewarm_ep_no_sync_symm_buffers", prewarm)

    model.prewarm_ep_no_sync_symm_buffers(
        max_local_microbatch_size=16,
        pad_to_block_count=2,
        rowwise_lifetime_lease_slots=1,
    )

    assert captured["args"] == ()
    assert captured["kwargs"] == {
        "max_local_microbatch_size": 16,
        "pad_to_block_count": 2,
        "rowwise_lifetime_lease_slots": 1,
        "prewarm_rowwise_scratch_buffers": True,
    }


def test_multimodal_olmo_ddp_routes_compact_flex_masks(monkeypatch):
    flex_config = _tiny_multimodal_model_config()
    flex_config.lm.block.sequence_mixer.backend = AttentionBackendName.flex

    torch.manual_seed(0)
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", lambda: True)
        model = flex_config.build(init_device="meta")
    model.init_weights(
        max_seq_len=8,
        max_local_microbatch_size=8,
        device=torch.device("cpu"),
        world_mesh={},
    )
    model.eval()
    assert model._use_compact_flex_masks

    class BlockSpy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kwargs = None

        def forward(self, x, **kwargs):
            self.kwargs = kwargs
            return x

    block_spies = []
    for key in model.lm.blocks:
        spy = BlockSpy()
        model.lm.blocks[key] = spy
        block_spies.append(spy)

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    token_type_ids = torch.tensor([[0, 1, 1, 0, 0, 0, 0, 0]])
    subsegment_ids = torch.tensor([[10000, 10000, 0, 0, 1, 1, 1, 1]])
    position_ids = torch.arange(8).unsqueeze(0)
    example_ids = torch.zeros_like(input_ids)

    with torch.no_grad():
        output = model(
            input_ids,
            token_type_ids=token_type_ids,
            subsegment_ids=subsegment_ids,
            position_ids=position_ids,
            example_ids=example_ids,
        )
    assert output.shape == (1, 8, 128)
    for spy in block_spies:
        assert spy.kwargs is not None
        assert spy.kwargs["flex_attn_block_mask"] is not None
        assert "or_mask" not in spy.kwargs
        assert "and_mask" not in spy.kwargs


def test_multimodal_olmo_ddp_config_and_native_checkpoint_aliases():
    config = MultimodalOLMoDDPTrainModuleConfig(
        rank_microbatch_size=16,
        max_sequence_length=16,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        freeze_params=["vision.*"],
        vision_activation_checkpointing=True,
        connector_activation_checkpointing=True,
        response_logits_only=True,
        train_embedding_rows=[120, 121],
    )
    restored = MultimodalOLMoDDPTrainModuleConfig.from_dict(config.as_dict())
    assert restored == config
    assert restored.vision_activation_checkpointing
    assert restored.connector_activation_checkpointing
    assert restored.response_logits_only
    assert restored.train_embedding_rows == [120, 121]

    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    checkpoint_keys = {
        "module.embeddings.weight.main",
        "module.blocks.0.attention.w_qkv.weight.main",
    }
    assert (
        train_module._resolve_optimizer_checkpoint_key(
            "module.lm.embeddings.weight.main", checkpoint_keys
        )
        == "module.embeddings.weight.main"
    )
    assert (
        train_module._resolve_optimizer_checkpoint_key(
            "module.lm.blocks.0.attention.w_qkv.weight.main", checkpoint_keys
        )
        == "module.blocks.0.attention.w_qkv.weight.main"
    )
    assert train_module._allow_missing_optimizer_checkpoint_key(
        "module.connector.projector.w1.weight.main"
    )
    assert train_module._allow_missing_optimizer_checkpoint_key(
        "module.vision.patch_embedding.weight.main"
    )
    assert not train_module._allow_missing_optimizer_checkpoint_key(
        "module.lm.embeddings.weight.main"
    )


def test_multimodal_checkpoint_frozen_params_can_become_trainable():
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    model = torch.nn.Module()
    model.vision = torch.nn.Linear(2, 2, bias=False)
    object.__setattr__(train_module, "model_parts", [model])

    checkpoint_keys = {"lm.weight.main", "frozen_model.vision.weight"}
    frozen_params = train_module._frozen_checkpoint_param_state_dict_for_load(checkpoint_keys)
    assert model.vision.weight.requires_grad
    assert set(frozen_params) == {"frozen_model.vision.weight"}
    assert frozen_params["frozen_model.vision.weight"] is model.vision.weight

    model.vision.weight.requires_grad_(False)
    native_checkpoint_keys = {"lm.weight.main", "module.vision.weight.main"}
    native_frozen_params = train_module._frozen_checkpoint_param_state_dict_for_load(
        native_checkpoint_keys
    )
    assert set(native_frozen_params) == {"module.vision.weight.main"}
    assert native_frozen_params["module.vision.weight.main"].shape == (4,)
    assert (
        native_frozen_params["module.vision.weight.main"].data_ptr()
        == model.vision.weight.data_ptr()
    )
    model.vision.weight.requires_grad_(True)

    checkpoint_state, checkpoint_to_current = train_module._optimizer_state_dict_for_load(
        {
            "lm.weight.main": torch.zeros(1),
            "vision.weight.main": torch.zeros(4),
        },
        checkpoint_keys,
        allow_component_missing=True,
        allowed_missing_keys={"vision.weight.main"},
    )
    assert set(checkpoint_state) == {"lm.weight.main"}
    assert checkpoint_to_current == {"lm.weight.main": "lm.weight.main"}


def test_multimodal_loss_divisor_uses_float_weights():
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    train_module.label_ignore_index = -100
    labels = torch.tensor([[1, 2, -100, 3]])
    loss_masks = torch.tensor([[0.25, 1.0, 0.0, 0.5]])
    divisor = train_module._batch_loss_divisor(
        {"loss_masks": loss_masks},
        labels,
        None,
        account_for_masked_instances=True,
    )
    torch.testing.assert_close(divisor, torch.tensor(1.75))


def test_multimodal_router_loss_divisor_uses_all_valid_tokens_not_response_weights():
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    train_module.device = torch.device("cpu")
    batch = {
        "input_ids": torch.zeros((2, 4), dtype=torch.long),
        "loss_masks": torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.5, 1.0, 0.0]]),
        "router_token_mask": torch.tensor([[True, True, False, False], [True, True, True, False]]),
    }

    kwargs = train_module._batch_auxiliary_loss_kwargs(batch)

    torch.testing.assert_close(kwargs["router_loss_div_factor"], torch.tensor(5))
    assert kwargs["router_loss_div_factor"] != batch["loss_masks"].sum()


def test_multimodal_data_metrics_report_packing_and_token_density():
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    recorded = {}

    def record_metric(name, value, reduce_type=None, namespace=None, **kwargs):
        del kwargs
        recorded[f"{namespace}/{name}"] = (value, reduce_type)

    train_module.record_metric = record_metric
    train_module._record_data_metrics(
        {
            "router_token_mask": torch.tensor(
                [[True, True, True, False], [True, True, False, False]]
            ),
            "loss_masks": torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            "token_type_ids": torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]]),
            "example_ids": torch.tensor([[0, 0, 1, -1], [0, 0, -1, -1]]),
        }
    )

    torch.testing.assert_close(recorded["data/packing fill"][0], torch.tensor(5 / 8))
    torch.testing.assert_close(recorded["data/response token density"][0], torch.tensor(3 / 8))
    torch.testing.assert_close(recorded["data/image token density"][0], torch.tensor(2 / 8))
    torch.testing.assert_close(recorded["data/examples per sequence"][0], torch.tensor(1.5))
    assert all(reduction == ReduceType.mean for _, reduction in recorded.values())


def test_multimodal_source_metrics_report_realized_loss_mass():
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    train_module.source_loss_mass_targets = {"caption": 0.75, "native": 0.25}
    recorded = {}

    def record_metric(name, value, reduce_type=None, namespace=None, **kwargs):
        del kwargs
        recorded[f"{namespace}/{name}"] = (value, reduce_type)

    train_module.record_metric = record_metric
    train_module._record_data_metrics(
        {
            "router_token_mask": torch.tensor([[True, True, True, True, True, False]]),
            "loss_masks": torch.tensor([[1.0, 1.0, 1.0, 0.5, 0.5, 0.0]]),
            "labels": torch.tensor([[10, 11, 12, 13, -100, -100]]),
            "token_type_ids": torch.zeros((1, 6), dtype=torch.long),
            "example_ids": torch.tensor([[0, 0, 0, 1, 1, -1]]),
            "pack_source_names": [["caption", "native"]],
        }
    )

    torch.testing.assert_close(recorded["data/source/caption/examples"][0], torch.tensor(1.0))
    torch.testing.assert_close(recorded["data/source/caption/tokens"][0], torch.tensor(3.0))
    torch.testing.assert_close(recorded["data/source/caption/loss_weight"][0], torch.tensor(3.0))
    torch.testing.assert_close(
        recorded["data/source/native/active_loss_weight"][0], torch.tensor(0.5)
    )
    torch.testing.assert_close(recorded["data/source/native/positive_tokens"][0], torch.tensor(1.0))
    torch.testing.assert_close(
        recorded["data/source/caption/loss_mass_share"][0], torch.tensor(0.75)
    )
    torch.testing.assert_close(
        recorded["data/source/native/loss_mass_target_abs_error"][0], torch.tensor(0.0)
    )
    assert recorded["data/source/caption/examples"][1] == ReduceType.mean
    assert recorded["data/source/caption/loss_mass_share"][1] == ReduceType.mean


def test_multimodal_source_loss_mass_share_uses_global_sums(monkeypatch):
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    train_module.source_loss_mass_targets = {"caption": 0.5, "native": 0.5}
    train_module.device = torch.device("cpu")
    train_module.dp_group = object()
    recorded = {}

    def record_metric(name, value, reduce_type=None, namespace=None, **kwargs):
        del kwargs
        recorded[f"{namespace}/{name}"] = (value, reduce_type)

    def all_reduce(value, group=None):
        assert group is train_module.dp_process_group
        # The local rank contributes caption=3/native=1 loss weight. Model a second
        # rank with caption=1/native=7; the correct global share is 4 / 12, whereas
        # averaging the rank-local shares would incorrectly produce 0.4375.
        remote = torch.zeros_like(value)
        remote[3] = 1.0
        remote[8] = 7.0
        value += remote

    train_module.record_metric = record_metric
    monkeypatch.setattr(multimodal_train_module, "is_distributed", lambda: True)
    monkeypatch.setattr(multimodal_train_module.dist, "all_reduce", all_reduce)
    train_module._record_source_data_metrics(
        {
            "loss_masks": torch.tensor([[1.0, 1.0, 1.0, 0.5, 0.5]]),
            "labels": torch.tensor([[10, 11, 12, 13, 14]]),
            "example_ids": torch.tensor([[0, 0, 0, 1, 1]]),
            "pack_source_names": [["caption", "native"]],
        },
        torch.ones((1, 5), dtype=torch.bool),
    )

    torch.testing.assert_close(
        recorded["data/source/caption/loss_mass_share"][0], torch.tensor(1 / 3)
    )
    torch.testing.assert_close(
        recorded["data/source/native/loss_mass_share"][0], torch.tensor(2 / 3)
    )


def test_multimodal_router_loss_divisor_requires_explicit_token_mask():
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    train_module.device = torch.device("cpu")
    with pytest.raises(OLMoConfigurationError, match="require router_token_mask"):
        train_module._batch_auxiliary_loss_kwargs(
            {"input_ids": torch.zeros((1, 4), dtype=torch.long)}
        )


def _run_multimodal_router_loss_divisor_distributed():
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    train_module.device = torch.device("cpu")
    train_module.dp_group = dist.group.WORLD
    rank = dist.get_rank()
    token_mask = torch.zeros((1, 4), dtype=torch.bool)
    token_mask[:, : 1 + 2 * rank] = True

    kwargs = train_module._batch_auxiliary_loss_kwargs(
        {
            "input_ids": torch.zeros_like(token_mask, dtype=torch.long),
            "router_token_mask": token_mask,
        }
    )

    # Rank-local valid counts are 1 and 3; OLMo DDP uses their global average.
    torch.testing.assert_close(kwargs["router_loss_div_factor"], torch.tensor(2.0))


def test_multimodal_router_loss_divisor_uses_global_dp_average():
    run_distributed_test(
        _run_multimodal_router_loss_divisor_distributed,
        world_size=2,
        backend="gloo",
        start_method="spawn",
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


def _run_multimodal_ep_step_impl(
    *, freeze_vision: bool, padded_router_compile: bool = False, fp32_accum: bool = False
):
    # The production Stage 1 model uses BF16 parameters backed by FP32 optimizer masters. Keep
    # the unfrozen test on that path so a post-optimizer vision load cannot silently regress.
    model_config = _tiny_multimodal_model_config(
        dtype=DType.float32 if freeze_vision else DType.bfloat16
    )
    if padded_router_compile:
        model_config.lm.recompute_each_block = True
        model_config.lm.block.ep = ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_nvshmem,
            capacity_factor=8.0,
            major_align=1,
        )
        router = model_config.lm.block.routed_experts_router
        assert router is not None
        router.lb_loss_weight = 0.015
        router.lb_loss_granularity = MoELoadBalancingLossGranularity.instance
        router.z_loss_weight = 0.0001
    model = model_config.build(init_device="meta")
    config = MultimodalOLMoDDPTrainModuleConfig(
        rank_microbatch_size=8,
        max_sequence_length=8,
        optim=OLMoDDPOptimizerConfig(
            lr=1e-3,
            weight_decay=0.0,
            group_overrides=[
                OptimGroupOverride(
                    params=["*connector.*", "*lm.embeddings.weight"],
                    opts={"scheduler_name": "connector"},
                ),
                OptimGroupOverride(params=["*vision.*"], opts={"scheduler_name": "vision"}),
            ],
            foreach_chunk_size=32,
            max_grad_norm=1.0,
            clip_grad_norm_by_scheduler_group=True,
            check_nan_inf_grad=True,
        ),
        freeze_params=(
            ["vision.*", "lm.lm_head.w_out.weight"]
            if freeze_vision
            else ["lm.lm_head.w_out.weight"]
        ),
        vision_activation_checkpointing=not freeze_vision,
        connector_activation_checkpointing=not freeze_vision,
        response_logits_only=True,
        diagnostics_interval=1,
        train_embedding_rows=[120, 121],
        compile_model=padded_router_compile,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            only_allreduce_last_microbatch=True,
            accumulate_grads_in_fp32=fp32_accum,
            reduce_grads_in_fp32=fp32_accum,
        ),
        ep_config=TransformerExpertParallelConfig(degree=2),
    )
    train_module = config.build(model, device=torch.device("cuda"))
    multimodal = train_module.multimodal_model

    train_module.reset_image_token_rows([120, 121], seed=19, reset_output_rows=False)
    optim = train_module._require_optimizer()
    assert optim.foreach_chunk_size == 32
    optim._check_model_param_main_param_the_same()
    lm_head_norm_name = next(
        name
        for group in optim.param_groups
        for name, param in group["named_params"].items()
        if param is multimodal.lm.lm_head.norm.weight
    )
    lm_head_norm_main_before = optim.states[f"{lm_head_norm_name}.main"].to_local().clone()

    if not freeze_vision:
        external_vision_state = {
            name: (
                tensor.detach().clone() + 0.125
                if tensor.is_floating_point()
                else tensor.detach().clone()
            )
            for name, tensor in multimodal.vision.state_dict().items()
        }
        train_module.load_vision_state_dict(external_vision_state)
        train_module.assert_vision_optimizer_state_synced()

        # An optimizer step starts by copying the masters into the model. Exercise that operation
        # directly and prove it preserves the externally loaded tower exactly.
        optim._copy_main_params_to_model_params()
        for name, tensor in multimodal.vision.state_dict().items():
            torch.testing.assert_close(tensor, external_vision_state[name], rtol=0, atol=0)
    vision_params = list(multimodal.vision.parameters())
    connector_params = list(multimodal.connector.parameters())
    routed_params = [
        param for name, param in multimodal.lm.named_parameters() if "routed_experts.w_" in name
    ]
    connector_before = [param.detach().clone() for param in connector_params]
    vision_before = [param.detach().clone() for param in vision_params]
    routed_before = [param.detach().clone() for param in routed_params]
    embedding_before = multimodal.lm.embeddings.weight.detach().clone()
    lm_head_before = multimodal.lm.lm_head.w_out.weight.detach().clone()

    rank = dist.get_rank()
    input_ids = torch.tensor([[120, 2 + rank, 4, 5, 6, 7, 8, 9]], device="cuda", dtype=torch.long)
    labels = torch.tensor([[2 + rank, 4, 5, 6, 7, 8, 9, 10]], device="cuda", dtype=torch.long)
    router_token_mask = torch.ones_like(input_ids, dtype=torch.bool)
    loss_masks = torch.tensor([[0.0, 0.25, 1.0, 0.5, 1.0, 0.75, 1.0, 0.5]], device="cuda")
    if padded_router_compile:
        router_token_mask[:, -3:] = False
        input_ids[:, -3:] = 0
        labels[:, -3:] = -100
        loss_masks[:, -3:] = 0
    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "router_token_mask": router_token_mask,
        "loss_masks": loss_masks,
        "token_type_ids": torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]], device="cuda", dtype=torch.long),
        "images": torch.randn(1, 1, 4, 14 * 14 * 3, device="cuda"),
        "pooled_patches_idx": torch.tensor([[[0, 1, 2, 3]]], device="cuda", dtype=torch.long),
    }

    train_module.zero_grads()
    second_batch = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }
    if not freeze_vision:
        multimodal.set_input_diagnostics(True)
    train_module.train_batch(batch, dry_run=True)
    if fp32_accum:
        # Exercise the production FP32 accumulation buffer across two backwards before clipping
        # and stepping. The embedding-row hook must mask every contributing microbatch.
        train_module.train_batch(second_batch, dry_run=True)

    def has_nonzero_grad(param):
        grad = getattr(param, "_main_grad_fp32", None) if fp32_accum else param.grad
        return grad is not None and torch.count_nonzero(grad) > 0

    if freeze_vision:
        assert all(param.grad is None for param in vision_params)
        assert not multimodal.vision.training
    else:
        assert any(has_nonzero_grad(param) for param in vision_params)
        assert multimodal.vision.training
        diagnostics = multimodal.pop_input_diagnostics(
            reduce_across_process_group=True,
            process_group=train_module.dp_process_group,
        )
        assert set(diagnostics) == {
            "text embedding RMS",
            "connector output RMS",
            "spliced image embedding RMS",
        }
        assert all(torch.isfinite(value) and value > 0 for value in diagnostics.values())
    assert any(has_nonzero_grad(param) for param in connector_params)
    assert any(has_nonzero_grad(param) for param in routed_params)
    trainer = _MetricTrainerStub()
    train_module._trainer = trainer  # type: ignore[assignment]
    optim.latest_loss = torch.zeros((), device="cuda")
    train_module.optim_step()
    expected_clip_groups = {optim.DEFAULT_CLIP_GROUP_NAME, "connector"}
    if not freeze_vision:
        expected_clip_groups.add("vision")
    assert set(optim.latest_clip_group_grad_norms) == expected_clip_groups
    assert set(optim.latest_clip_group_coefficients) == expected_clip_groups
    assert all(torch.isfinite(value) for value in optim.latest_clip_group_coefficients.values())
    expected_components = {
        "connector",
        "input embeddings",
        "LM attention",
        "LM routed experts",
        "LM routers",
        "LM normalization",
    }
    if not freeze_vision:
        expected_components.add("vision")
    assert set(optim.latest_component_grad_norms) == expected_components
    assert "optim/LM output head grad norm" not in trainer.metrics
    assert all(
        f"optim/{component} grad norm" in trainer.metrics for component in expected_components
    )
    assert all(
        torch.isfinite(norm) and norm > 0 for norm in optim.latest_component_grad_norms.values()
    )
    assert optim._component_grad_norm_patterns is None

    assert any(
        not torch.equal(param, before) for param, before in zip(connector_params, connector_before)
    )
    assert any(
        not torch.equal(param, before) for param, before in zip(routed_params, routed_before)
    )
    ordinary_embedding_rows = torch.ones(multimodal.lm.vocab_size, dtype=torch.bool, device="cuda")
    ordinary_embedding_rows[[120, 121]] = False
    torch.testing.assert_close(
        multimodal.lm.embeddings.weight[ordinary_embedding_rows],
        embedding_before[ordinary_embedding_rows],
        rtol=0,
        atol=0,
    )
    assert not torch.equal(multimodal.lm.embeddings.weight[120], embedding_before[120])
    torch.testing.assert_close(multimodal.lm.lm_head.w_out.weight, lm_head_before, rtol=0, atol=0)
    assert not torch.equal(
        optim.states[f"{lm_head_norm_name}.main"].to_local(), lm_head_norm_main_before
    )
    if freeze_vision:
        for param, before in zip(vision_params, vision_before):
            torch.testing.assert_close(param, before, rtol=0, atol=0)
    else:
        assert any(
            not torch.equal(param, before) for param, before in zip(vision_params, vision_before)
        )


def _run_multimodal_ep_step():
    _run_multimodal_ep_step_impl(freeze_vision=True)


def _run_multimodal_ep_unfrozen_vision_step():
    _run_multimodal_ep_step_impl(freeze_vision=False, fp32_accum=True)


def _run_multimodal_ep_padded_compile_step():
    _run_multimodal_ep_step_impl(freeze_vision=True, padded_router_compile=True)


@requires_multi_gpu
def test_multimodal_olmo_ddp_ep_step():
    run_distributed_test(
        _run_multimodal_ep_step,
        world_size=2,
        backend="nccl",
        start_method="spawn",
    )


@requires_multi_gpu
def test_multimodal_olmo_ddp_ep_unfrozen_vision_step():
    run_distributed_test(
        _run_multimodal_ep_unfrozen_vision_step,
        world_size=2,
        backend="nccl",
        start_method="spawn",
    )


@requires_multi_gpu
def test_multimodal_olmo_ddp_ep_padding_compile_and_checkpoint_step():
    run_distributed_test(
        _run_multimodal_ep_padded_compile_step,
        world_size=2,
        backend="nccl",
        start_method="spawn",
    )


def _build_multimodal_ddp_train_module_for_checkpoint(
    *, freeze_vision: bool = True, freeze_lm_head: bool = False
):
    model = _tiny_multimodal_model_config(dtype=DType.bfloat16).build(init_device="meta")
    freeze_params = []
    if freeze_vision:
        freeze_params.append("vision.*")
    if freeze_lm_head:
        freeze_params.append("lm.lm_head.w_out.weight")
    config = MultimodalOLMoDDPTrainModuleConfig(
        rank_microbatch_size=8,
        max_sequence_length=8,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        freeze_params=freeze_params or None,
        dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp),
        ep_config=TransformerExpertParallelConfig(degree=2),
    )
    return config.build(model, device=torch.device("cuda"))


def _run_native_checkpoint_into_multimodal(native_dir, hybrid_dir):
    native = _build_ddp_train_module_for_checkpoint(ep_degree=2)
    native_model = getattr(native.model_parts[0], "module", native.model_parts[0])
    expected_lm = {name: param.detach().clone() for name, param in native_model.named_parameters()}
    native.save_state_dict_direct(native_dir)

    unfrozen_hybrid = _build_multimodal_ddp_train_module_for_checkpoint(freeze_vision=False)
    unfrozen_model = unfrozen_hybrid.multimodal_model
    unfrozen_vision_before = {
        name: param.detach().clone() for name, param in unfrozen_model.vision.named_parameters()
    }
    unfrozen_connector_before = {
        name: param.detach().clone() for name, param in unfrozen_model.connector.named_parameters()
    }
    unfrozen_hybrid.load_state_dict_direct(native_dir, load_optim_state=False)
    for name, param in unfrozen_model.lm.named_parameters():
        torch.testing.assert_close(param, expected_lm[name], rtol=0, atol=0)
    for name, param in unfrozen_model.vision.named_parameters():
        torch.testing.assert_close(param, unfrozen_vision_before[name], rtol=0, atol=0)
    for name, param in unfrozen_model.connector.named_parameters():
        torch.testing.assert_close(param, unfrozen_connector_before[name], rtol=0, atol=0)
    unfrozen_hybrid._require_optimizer()._check_model_param_main_param_the_same()

    hybrid = _build_multimodal_ddp_train_module_for_checkpoint(freeze_lm_head=True)
    multimodal = hybrid.multimodal_model
    connector_before = {
        name: param.detach().clone() for name, param in multimodal.connector.named_parameters()
    }
    with torch.no_grad():
        for param in multimodal.lm.parameters():
            param.zero_()

    hybrid.load_state_dict_direct(native_dir, load_optim_state=False)
    for name, param in multimodal.lm.named_parameters():
        torch.testing.assert_close(param, expected_lm[name], rtol=0, atol=0)
    for name, param in multimodal.connector.named_parameters():
        torch.testing.assert_close(param, connector_before[name], rtol=0, atol=0)

    optim = hybrid._require_optimizer()
    embedding_name = next(
        name
        for group in optim.param_groups
        for name, param in group["named_params"].items()
        if param is multimodal.lm.embeddings.weight
    )
    embedding_main = optim.states[f"{embedding_name}.main"]
    # Model-only loading can retain precision in the FP32 optimizer master that is not
    # representable in the BF16 model. Resetting image rows must preserve that precision for
    # every ordinary token row.
    embedding_main.to_local().add_(1e-4)
    optim._copy_main_params_to_model_params()
    main_before_reset = (
        embedding_main.full_tensor().reshape_as(multimodal.lm.embeddings.weight).clone()
    )
    lm_head_before_reset = multimodal.lm.lm_head.w_out.weight.detach().clone()

    hybrid.reset_image_token_rows([120, 121], seed=19, reset_output_rows=False)
    main_after_reset = embedding_main.full_tensor().reshape_as(multimodal.lm.embeddings.weight)
    ordinary_rows = torch.ones(128, dtype=torch.bool, device="cuda")
    ordinary_rows[[120, 121]] = False
    torch.testing.assert_close(
        main_after_reset[ordinary_rows],
        main_before_reset[ordinary_rows],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        main_after_reset[[120, 121]],
        multimodal.lm.embeddings.weight[[120, 121]].float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        multimodal.lm.lm_head.w_out.weight,
        lm_head_before_reset,
        rtol=0,
        atol=0,
    )

    # Restore exact BF16/FP32 equality for the generic optimizer invariant used below.
    optim._copy_model_params_to_main_params({embedding_name})
    optim._check_model_param_main_param_the_same()

    with torch.no_grad():
        for param in multimodal.vision.parameters():
            param.fill_(0.125)
    saved_vision = {
        name: param.detach().clone() for name, param in multimodal.vision.named_parameters()
    }
    saved_connector = {
        name: param.detach().clone() for name, param in multimodal.connector.named_parameters()
    }
    saved_lm = {name: param.detach().clone() for name, param in multimodal.lm.named_parameters()}
    hybrid.save_state_dict_direct(hybrid_dir)

    with torch.no_grad():
        for param in multimodal.parameters():
            param.fill_(-0.75)
    hybrid.load_state_dict_direct(hybrid_dir)

    for name, param in multimodal.vision.named_parameters():
        torch.testing.assert_close(param, saved_vision[name], rtol=0, atol=0)
    for name, param in multimodal.connector.named_parameters():
        torch.testing.assert_close(param, saved_connector[name], rtol=0, atol=0)
    for name, param in multimodal.lm.named_parameters():
        torch.testing.assert_close(param, saved_lm[name], rtol=0, atol=0)
    optim._check_model_param_main_param_the_same()

    # A Stage 1 checkpoint stores its frozen vision tower outside the optimizer. Stage 2
    # unfreezes that tower, so a model-only load must restore those weights and seed their new
    # FP32 optimizer masters while retaining the trainable LM and connector weights.
    stage2 = _build_multimodal_ddp_train_module_for_checkpoint(freeze_vision=False)
    stage2_model = stage2.multimodal_model
    with torch.no_grad():
        for param in stage2_model.parameters():
            param.fill_(-0.5)
    stage2.load_state_dict_direct(hybrid_dir, load_optim_state=False)

    for name, param in stage2_model.vision.named_parameters():
        torch.testing.assert_close(param, saved_vision[name], rtol=0, atol=0)
    for name, param in stage2_model.connector.named_parameters():
        torch.testing.assert_close(param, saved_connector[name], rtol=0, atol=0)
    for name, param in stage2_model.lm.named_parameters():
        torch.testing.assert_close(param, saved_lm[name], rtol=0, atol=0)
    stage2._require_optimizer()._check_model_param_main_param_the_same()


@requires_multi_gpu
def test_native_checkpoint_loads_into_multimodal_and_roundtrips(tmp_path):
    run_distributed_test(
        _run_native_checkpoint_into_multimodal,
        world_size=2,
        backend="nccl",
        start_method="spawn",
        func_args=(str(tmp_path / "native"), str(tmp_path / "hybrid")),
    )


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
