import types

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.config import DType
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.moe import MoERouterGatingFunction
from olmo_core.nn.moe.v2.block import MoEFusedV2TransformerBlock
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.testing import requires_multi_gpu, run_distributed_test


def _build_ep_mesh() -> DeviceMesh:
    world_size = dist.get_world_size()
    mesh = torch.arange(world_size, dtype=torch.int).view(1, world_size)
    return DeviceMesh(
        device_type="cuda",
        mesh=mesh,
        mesh_dim_names=("ep_dp", "ep_mp"),
    )


def _build_tbo_model(*, two_batch_overlap: bool, ep_no_sync: bool = False):
    from olmo_core.nn.attention import AttentionConfig, AttentionType
    from olmo_core.nn.lm_head import LMHeadConfig
    from olmo_core.nn.moe.v2.block import MoEFusedV2TransformerBlockConfig
    from olmo_core.nn.transformer import (
        MoEFusedV2TransformerConfig,
        TransformerBlockType,
        TransformerType,
    )

    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=DType.float32,
    )
    config = MoEFusedV2TransformerConfig(
        name=TransformerType.moe_fused_v2,
        d_model=512,
        vocab_size=128,
        n_layers=2,
        two_batch_overlap=two_batch_overlap,
        recompute_each_block=False,
        recompute_all_blocks_by_chunk=False,
        lm_head=LMHeadConfig(bias=False, dtype=DType.float32),
        block=MoEFusedV2TransformerBlockConfig(
            name=TransformerBlockType.moe_fused_v2,
            attention=AttentionConfig(
                name=AttentionType.default,
                n_heads=2,
                n_kv_heads=2,
                bias=False,
                use_flash=False,
                dtype=DType.float32,
            ),
            attention_norm=layer_norm,
            routed_experts_router=MoERouterConfigV2(
                d_model=512,
                num_experts=8,
                top_k=2,
                gating_function=MoERouterGatingFunction.softmax,
                uniform_expert_assignment=False,
                lb_loss_weight=None,
                z_loss_weight=None,
                dtype=DType.float32,
            ),
            shared_experts_router=None,
            shared_experts=None,
            routed_experts=RoutedExpertsConfig(
                d_model=512,
                hidden_size=1024,
                num_experts=8,
                bias=False,
                dtype=DType.float32,
            ),
            feed_forward_norm=layer_norm,
            ep_no_sync=ep_no_sync,
            ep_no_sync_capacity_factor=8.0,
            ep_no_sync_major_align=1,
            ep_no_sync_shared_slots=2 if two_batch_overlap and ep_no_sync else 1,
        ),
    )
    return config.build(init_device="cuda")


def _init_model_params(model):
    torch.manual_seed(2718)
    with torch.no_grad():
        for param in model.parameters():
            if param.is_floating_point():
                param.normal_(mean=0.0, std=0.02)


def _install_deterministic_topk_router(block: MoEFusedV2TransformerBlock):
    def _deterministic_router_forward(self, router, local_x, scores_only, loss_div_factor):
        del loss_div_factor
        B, S, _ = local_x.shape
        if scores_only:
            return (
                torch.ones(
                    B,
                    S,
                    router.num_experts,
                    device=local_x.device,
                    dtype=local_x.dtype,
                ),
                None,
                None,
                None,
            )

        top_k = router.top_k
        num_experts = router.num_experts
        token_ids = torch.arange(B * S, device=local_x.device, dtype=torch.long).unsqueeze(1)
        route_offsets = torch.arange(top_k, device=local_x.device, dtype=torch.long).unsqueeze(0)
        expert_indices = (token_ids + route_offsets + dist.get_rank() * 3) % num_experts
        expert_indices = expert_indices.view(B, S, top_k)

        weights = torch.arange(1, top_k + 1, device=local_x.device, dtype=local_x.dtype)
        weights = weights / weights.sum().clamp_min(1e-6)
        expert_weights = weights.view(1, 1, top_k).expand(B, S, top_k).contiguous()

        batch_size_per_expert = torch.bincount(
            expert_indices.reshape(-1),
            minlength=num_experts,
        ).to(dtype=torch.long)
        return expert_weights, expert_indices, batch_size_per_expert, None

    block.router_forward = types.MethodType(_deterministic_router_forward, block)


def _install_model_routers(model) -> None:
    for block in model.blocks.values():
        assert isinstance(block, MoEFusedV2TransformerBlock)
        _install_deterministic_topk_router(block)


def _apply_synced_ep(model) -> None:
    ep_mesh = _build_ep_mesh()
    model.apply_ep(
        dp_mesh=ep_mesh["ep_dp"],
        ep_mesh=ep_mesh,
        ep_mp_group=ep_mesh["ep_mp"].get_group(),
    )


def _assert_matching_grads(ref_model, tbo_model) -> None:
    ref_params = dict(ref_model.named_parameters())
    tbo_params = dict(tbo_model.named_parameters())
    for name, ref_param in ref_params.items():
        tbo_param = tbo_params[name]
        if ref_param.grad is None:
            assert tbo_param.grad is None, name
            continue
        assert tbo_param.grad is not None, name
        torch.testing.assert_close(tbo_param.grad, ref_param.grad, atol=1e-3, rtol=1e-3)


def _run_synced_ep_tbo_matches_standard_forward():
    ref_model = _build_tbo_model(two_batch_overlap=False)
    tbo_model = _build_tbo_model(two_batch_overlap=True)
    _apply_synced_ep(ref_model)
    _apply_synced_ep(tbo_model)
    _init_model_params(ref_model)
    tbo_model.load_state_dict(ref_model.state_dict())
    _install_model_routers(ref_model)
    _install_model_routers(tbo_model)
    ref_model.train()
    tbo_model.train()

    input_ids = torch.randint(0, ref_model.vocab_size, (4, 8), device="cuda", dtype=torch.long)
    labels = torch.randint(0, ref_model.vocab_size, (4, 8), device="cuda", dtype=torch.long)

    logits_ref = ref_model(input_ids, return_logits=True)
    logits_tbo = tbo_model(input_ids, return_logits=True)
    torch.testing.assert_close(logits_tbo, logits_ref, atol=5e-4, rtol=5e-4)

    out_ref = ref_model(
        input_ids,
        labels=labels,
        loss_reduction="sum",
        return_logits=True,
    )
    out_tbo = tbo_model(
        input_ids,
        labels=labels,
        loss_reduction="sum",
        return_logits=True,
    )
    assert out_ref.logits is not None
    assert out_tbo.logits is not None
    torch.testing.assert_close(out_tbo.logits, out_ref.logits, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(out_tbo.loss, out_ref.loss, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(out_tbo.ce_loss, out_ref.ce_loss, atol=5e-4, rtol=5e-4)

    out_ref.loss.backward()
    out_tbo.loss.backward()
    _assert_matching_grads(ref_model, tbo_model)


def _run_no_sync_ep_tbo_matches_standard_forward():
    ref_model = _build_tbo_model(two_batch_overlap=False, ep_no_sync=True)
    tbo_model = _build_tbo_model(two_batch_overlap=True, ep_no_sync=True)
    _apply_synced_ep(ref_model)
    _apply_synced_ep(tbo_model)
    _init_model_params(ref_model)
    tbo_model.load_state_dict(ref_model.state_dict())
    _install_model_routers(ref_model)
    _install_model_routers(tbo_model)
    ref_model.train()
    tbo_model.train()

    input_ids = torch.randint(0, ref_model.vocab_size, (4, 8), device="cuda", dtype=torch.long)
    labels = torch.randint(0, ref_model.vocab_size, (4, 8), device="cuda", dtype=torch.long)

    out_ref = ref_model(
        input_ids,
        labels=labels,
        loss_reduction="sum",
        return_logits=False,
    )
    out_tbo = tbo_model(
        input_ids,
        labels=labels,
        loss_reduction="sum",
        return_logits=False,
    )
    assert out_ref.logits is None
    assert out_tbo.logits is None
    torch.testing.assert_close(out_tbo.loss, out_ref.loss, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(out_tbo.ce_loss, out_ref.ce_loss, atol=5e-4, rtol=5e-4)

    out_ref.loss.backward()
    out_tbo.loss.backward()
    _assert_matching_grads(ref_model, tbo_model)


@requires_multi_gpu
def test_synced_ep_tbo_matches_standard_forward():
    run_distributed_test(
        _run_synced_ep_tbo_matches_standard_forward,
        backend="nccl",
        start_method="spawn",
    )


@requires_multi_gpu
def test_no_sync_ep_tbo_matches_standard_forward():
    run_distributed_test(
        _run_no_sync_ep_tbo_matches_standard_forward,
        backend="nccl",
        start_method="spawn",
    )
