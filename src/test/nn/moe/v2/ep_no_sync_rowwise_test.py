import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.moe import MoERouterGatingFunction
from olmo_core.nn.moe.v2.block import MoEFusedV2TransformerBlock
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.testing import (
    requires_multi_gpu,
    requires_symm_mem_vdev2d,
    run_distributed_test,
)


def _build_ep_mesh() -> DeviceMesh:
    world_size = dist.get_world_size()
    mesh = torch.arange(world_size, dtype=torch.int).view(1, world_size)
    return DeviceMesh(device_type="cuda", mesh=mesh, mesh_dim_names=("ep_dp", "ep_mp"))


def _build_block(*, ep_no_sync: bool, ep_no_sync_capacity_factor: float = 8.0):
    layer_norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=DType.float32)
    return MoEFusedV2TransformerBlock(
        d_model=512,
        block_idx=0,
        n_layers=1,
        sequence_mixer=AttentionConfig(
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
            d_model=512, hidden_size=1024, num_experts=8, bias=False, dtype=DType.float32
        ),
        feed_forward_norm=layer_norm,
        ep_no_sync=ep_no_sync,
        ep_no_sync_capacity_factor=ep_no_sync_capacity_factor,
        ep_no_sync_major_align=1,
        init_device="cuda",
    )


def _init_block_params(block: MoEFusedV2TransformerBlock):
    torch.manual_seed(1234)
    with torch.no_grad():
        for p in block.parameters():
            if p.is_floating_point():
                p.normal_(mean=0.0, std=0.02)


def _install_deterministic_topk_router(block: MoEFusedV2TransformerBlock):
    def _make(router):
        def _forward(local_x, scores_only, loss_div_factor=None):
            del loss_div_factor
            B, S, _ = local_x.shape
            if scores_only:
                return (
                    torch.ones(
                        B, S, router.num_experts, device=local_x.device, dtype=local_x.dtype
                    ),
                    None,
                    None,
                    None,
                )
            top_k, num_experts = router.top_k, router.num_experts
            token_ids = torch.arange(B * S, device=local_x.device, dtype=torch.long).unsqueeze(1)
            route_offsets = torch.arange(top_k, device=local_x.device, dtype=torch.long).unsqueeze(
                0
            )
            expert_indices = ((token_ids + route_offsets + dist.get_rank() * 3) % num_experts).view(
                B, S, top_k
            )
            weights = torch.arange(1, top_k + 1, device=local_x.device, dtype=local_x.dtype)
            weights = weights / weights.sum().clamp_min(1e-6)
            expert_weights = weights.view(1, 1, top_k).expand(B, S, top_k).contiguous()
            batch_size_per_expert = torch.bincount(
                expert_indices.reshape(-1), minlength=num_experts
            ).to(dtype=torch.long)
            return expert_weights, expert_indices, batch_size_per_expert, None

        return _forward

    assert block.routed_experts_router is not None
    block.routed_experts_router.forward = _make(block.routed_experts_router)


def _copy_no_ep_weights_to_ep_shard(no_ep_block, ep_block):
    no_ep_params = dict(no_ep_block.named_parameters())
    local_experts = ep_block.routed_experts.num_local_experts
    start = ep_block.routed_experts.ep_rank * local_experts
    end = start + local_experts
    with torch.no_grad():
        for name, param in ep_block.named_parameters():
            if name == "routed_experts.w_up_gate":
                param.copy_(no_ep_block.routed_experts.w_up_gate[start:end])
            elif name == "routed_experts.w_down":
                param.copy_(no_ep_block.routed_experts.w_down[start:end])
            else:
                param.copy_(no_ep_params[name])


def _assert_expert_grad_matches_no_ep_global_sum(no_ep_block, ep_block, *, atol, rtol):
    local_experts = ep_block.routed_experts.num_local_experts
    start = ep_block.routed_experts.ep_rank * local_experts
    end = start + local_experts
    for name in ("w_up_gate", "w_down"):
        no_ep_grad = getattr(no_ep_block.routed_experts, name).grad
        ep_grad = getattr(ep_block.routed_experts, name).grad
        assert no_ep_grad is not None and ep_grad is not None
        no_ep_grad_global = no_ep_grad.detach().clone()
        dist.all_reduce(no_ep_grad_global, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(ep_grad, no_ep_grad_global[start:end], atol=atol, rtol=rtol)


def _run_rowwise_ep_dropless_matches_no_ep():
    ep_mesh = _build_ep_mesh()
    no_ep_block = _build_block(ep_no_sync=False)
    ep_block = _build_block(ep_no_sync=True)
    # Select the rowwise path before apply_ep: apply_ep configures the rowwise symmetric-memory
    # buffers and rejects the non-rowwise (VDev) path under the default OLMo-owned symm-mem backend.
    ep_block.ep_no_sync_use_rowwise_all_to_all = True
    ep_block.ep_no_sync_rowwise_nblocks = 128
    ep_block.apply_ep(ep_mesh)

    _init_block_params(no_ep_block)
    _copy_no_ep_weights_to_ep_shard(no_ep_block, ep_block)
    _install_deterministic_topk_router(no_ep_block)
    _install_deterministic_topk_router(ep_block)
    no_ep_block.train()
    ep_block.train()

    x = torch.randn(
        1, 8, no_ep_block.d_model, device="cuda", dtype=torch.float32, requires_grad=True
    )
    x_ep = x.detach().clone().requires_grad_(True)

    y_no_ep = no_ep_block(x)
    y_ep = ep_block(x_ep)
    torch.testing.assert_close(y_ep, y_no_ep, atol=5e-4, rtol=5e-4)
    # capacity_factor=8.0 with this routing keeps every token.
    assert ep_block._ep_no_sync_rowwise_drop_tokens_sum is not None
    assert ep_block._ep_no_sync_rowwise_drop_tokens_sum.item() == 0

    (y_no_ep.square().mean() + 0.1 * y_no_ep.sum()).backward()
    (y_ep.square().mean() + 0.1 * y_ep.sum()).backward()
    torch.testing.assert_close(x_ep.grad, x.grad, atol=5e-4, rtol=5e-4)
    _assert_expert_grad_matches_no_ep_global_sum(no_ep_block, ep_block, atol=1e-3, rtol=1e-3)


@requires_multi_gpu
@requires_symm_mem_vdev2d
def test_v2_rowwise_ep_dropless_matches_no_ep():
    run_distributed_test(
        _run_rowwise_ep_dropless_matches_no_ep,
        backend="nccl",
        start_method="spawn",
    )
