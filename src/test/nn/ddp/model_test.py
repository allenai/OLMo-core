"""Tests for ``OLMoDDPModel`` construction and forward paths."""

import pytest
import torch

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.ddp import model as ddp_model_module
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMOutputWithLoss
from olmo_core.nn.moe.v2 import ep_no_sync_tbo_rowwise as tbo_module
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)


def _build_model_config(*, d_model: int = 64, n_layers: int = 2) -> OLMoDDPModelConfig:
    dtype = DType.float32
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
                name=AttentionType.default, n_heads=4, bias=False, use_flash=False, dtype=dtype
            ),
            routed_experts=RoutedExpertsConfig(
                d_model=d_model, hidden_size=128, num_experts=4, bias=False, dtype=dtype
            ),
            routed_experts_router=MoERouterConfigV2(
                d_model=d_model, num_experts=4, top_k=2, dtype=dtype
            ),
            shared_experts=None,
            layer_norm=layer_norm,
        ),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
    )


def test_moe_v2_model_builds():
    model = _build_model_config(n_layers=2).build(init_device="cpu")

    assert len(model.blocks) == 2
    assert any(p.numel() > 0 for p in model.parameters())
    assert model.num_flops_per_token(seq_len=512) > 0


def test_model_rejects_composable_ddp():
    model = _build_model_config(n_layers=1).build(init_device="cpu")
    with pytest.raises(OLMoConfigurationError, match="use apply_dp"):
        model.apply_ddp()


def test_model_rejects_extra_embedding_rows():
    config = _build_model_config()
    config.n_extra_vocab = 4

    with pytest.raises(OLMoConfigurationError, match="n_extra_vocab"):
        config.build(init_device="meta")


def test_fp32_gradient_accumulation_releases_parameter_gradient():
    param = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    param._main_grad_fp32 = None
    ddp_model_module._fp32_post_grad_acc_hook(param)
    assert param._main_grad_fp32 is None

    param.grad = torch.full_like(param, 0.5)
    ddp_model_module._fp32_post_grad_acc_hook(param)
    assert param.grad is None
    first_buffer = param._main_grad_fp32
    torch.testing.assert_close(first_buffer, torch.full((4,), 0.5, dtype=torch.float32))

    param.grad = torch.full_like(param, 0.25)
    ddp_model_module._fp32_post_grad_acc_hook(param)
    assert param.grad is None
    assert param._main_grad_fp32 is first_buffer
    torch.testing.assert_close(first_buffer, torch.full((4,), 0.75, dtype=torch.float32))


def test_deepep_rejects_chunk_recompute():
    config = _build_model_config(n_layers=2)
    config.recompute_all_blocks_by_chunk = True
    assert isinstance(config.block, OLMoDDPTransformerBlockConfig)
    config.block.ep = ExpertParallelConfig(path=ExpertParallelPath.deepep_v2)
    with pytest.raises(OLMoConfigurationError, match="recompute_all_blocks_by_chunk"):
        config.build(init_device="cpu")


def test_rowwise_prewarm_can_include_forward_only_scratch_buffers(monkeypatch):
    config = _build_model_config(n_layers=1)
    assert isinstance(config.block, OLMoDDPTransformerBlockConfig)
    config.block.ep = ExpertParallelConfig(path=ExpertParallelPath.rowwise_nvshmem)
    model = config.build(init_device="cpu")
    block = next(model.routed_blocks())
    block.ep_pg = object()  # type: ignore[assignment]

    buffer_calls = []
    lease_calls = []
    monkeypatch.setattr(ddp_model_module, "compute_ep_no_sync_rank_capacity", lambda *_: 8)
    monkeypatch.setattr(
        ddp_model_module,
        "get_ep_no_sync_buffers",
        lambda _block, **kwargs: buffer_calls.append(kwargs),
    )
    monkeypatch.setattr(
        ddp_model_module,
        "prewarm_ep_no_sync_rowwise_lifetime_leases",
        lambda _block, **kwargs: lease_calls.append(kwargs),
    )
    monkeypatch.setattr(
        ddp_model_module,
        "use_ep_no_sync_rowwise_symm_dispatch_in",
        lambda _block: False,
    )
    monkeypatch.setattr(
        ddp_model_module,
        "use_ep_no_sync_rowwise_symm_combine_out",
        lambda _block: False,
    )
    monkeypatch.setattr(
        ddp_model_module,
        "use_ep_no_sync_rowwise_symm_combine_gather",
        lambda _block: False,
    )

    model.prewarm_ep_no_sync_symm_buffers(
        max_local_microbatch_size=8,
        pad_to_block_count=1,
    )
    model.prewarm_ep_no_sync_symm_buffers(
        max_local_microbatch_size=8,
        pad_to_block_count=1,
        prewarm_rowwise_scratch_buffers=True,
    )

    assert [call["need_dispatch_out"] for call in buffer_calls] == [False, True]
    assert all(call["need_dispatch_out"] for call in lease_calls)
    assert block._ep_no_sync_force_scratch_lifetime_buffers is False


@pytest.mark.parametrize("with_labels", [False, True])
@pytest.mark.parametrize("selection", ["all", "response", "positions", "last"])
def test_tbo_splits_lm_head_inputs(monkeypatch, with_labels, selection):
    config = _build_model_config(n_layers=1)
    config.two_batch_overlap = True
    config.block.ep = ExpertParallelConfig(path=ExpertParallelPath.rowwise_nvshmem, shared_slots=2)
    model = config.build(init_device="cpu")
    model.ep_enabled = True
    block = next(model.routed_blocks())

    # Keep the TBO forward and LM-head paths; replace only expert communication/computation.
    def combined_forward(x0, x1_ctx, x1_is_fresh, **kwargs):
        assert x1_is_fresh
        return x0, tbo_module._NoSyncRowwiseTboPendingContext(
            block=block, lane_id=1, a_state=None, global_x_rank_major=x1_ctx["x1"]
        )

    monkeypatch.setattr(block, "combined_forward_rowwise_nvshmem_tbo", combined_forward)
    monkeypatch.setattr(tbo_module, "ep_no_sync_rowwise_tbo_stage_c_launch", lambda _, ctx: ctx)
    monkeypatch.setattr(
        tbo_module, "ep_no_sync_rowwise_tbo_stage_tail", lambda _, ctx: ctx.global_x_rank_major
    )

    input_ids = torch.arange(16).view(4, 4)
    labels = input_ids.clone() if with_labels else None
    weights = torch.arange(1, 17, dtype=torch.float32).view(4, 4) / 16
    kwargs = dict(
        loss_reduction="sum",
        loss_div_factor=torch.tensor(7.0),
        z_loss_multiplier=1e-4,
        return_logits=True,
    )
    if with_labels:
        labels[0, 0] = -100
        kwargs["loss_weights"] = weights
    if selection == "response":
        kwargs["response_logits_only"] = True
        kwargs["response_mask"] = input_ids % 3 == 0
    elif selection == "positions":
        kwargs["logits_to_keep"] = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    elif selection == "last":
        kwargs["logits_to_keep"] = 2

    actual = model(input_ids, labels=labels, **kwargs)
    expected = model.lm_head(model.forward_embed(input_ids), labels=labels, **kwargs)
    if with_labels:
        assert isinstance(actual, LMOutputWithLoss)
        assert isinstance(expected, LMOutputWithLoss)
        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(actual_value, expected_value)
        parameters = (model.embeddings.weight, model.lm_head.w_out.weight)
        actual_grads = torch.autograd.grad(actual.loss, parameters)
        expected_grads = torch.autograd.grad(expected.loss, parameters)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)
    else:
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("batch_size", [0, 1, 3])
def test_tbo_rejects_invalid_instance_count_before_embedding(monkeypatch, batch_size):
    config = _build_model_config(n_layers=1)
    config.two_batch_overlap = True
    model = config.build(init_device="cpu")
    model.ep_enabled = True

    def embed(_):
        pytest.fail("Invalid TBO microbatches must be rejected before embedding")

    monkeypatch.setattr(model.embeddings, "forward", embed)
    with pytest.raises(OLMoConfigurationError, match="even number of instances"):
        model(torch.zeros(batch_size, 4, dtype=torch.long))
