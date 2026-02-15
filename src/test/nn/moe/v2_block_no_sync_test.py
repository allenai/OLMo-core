import types

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
from olmo_core.testing import requires_multi_gpu, run_distributed_test


def _build_ep_mesh() -> DeviceMesh:
    world_size = dist.get_world_size()
    mesh = torch.arange(world_size, dtype=torch.int).view(1, world_size)
    return DeviceMesh(
        device_type="cuda",
        mesh=mesh,
        mesh_dim_names=("ep_dp", "ep_mp"),
    )


def _build_block(
    *,
    ep_no_sync: bool,
    ep_no_sync_capacity_factor: float = 2.0,
    d_model: int = 16,
    hidden_size: int = 32,
    num_experts: int = 4,
    top_k: int = 1,
    uniform_expert_assignment: bool = True,
) -> MoEFusedV2TransformerBlock:
    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=DType.float32,
    )
    return MoEFusedV2TransformerBlock(
        d_model=d_model,
        block_idx=0,
        n_layers=1,
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
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            gating_function=MoERouterGatingFunction.softmax,
            uniform_expert_assignment=uniform_expert_assignment,
            lb_loss_weight=None,
            z_loss_weight=None,
            dtype=DType.float32,
        ),
        shared_experts_router=None,
        shared_experts=None,
        routed_experts=RoutedExpertsConfig(
            d_model=d_model,
            hidden_size=hidden_size,
            num_experts=num_experts,
            bias=False,
            dtype=DType.float32,
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


def _install_forced_router(block: MoEFusedV2TransformerBlock):
    def _forced_router_forward(self, router, local_x, scores_only, loss_div_factor):
        del loss_div_factor
        B, S, _ = local_x.shape
        if scores_only:
            return torch.ones(
                B,
                S,
                router.num_experts,
                device=local_x.device,
                dtype=local_x.dtype,
            ), None, None, None

        expert_weights = torch.ones(
            B,
            S,
            router.top_k,
            device=local_x.device,
            dtype=local_x.dtype,
        )
        expert_indices = torch.zeros(
            B,
            S,
            router.top_k,
            device=local_x.device,
            dtype=torch.long,
        )
        batch_size_per_expert = torch.zeros(
            router.num_experts,
            device=local_x.device,
            dtype=torch.long,
        )
        batch_size_per_expert[0] = B * S * router.top_k
        return expert_weights, expert_indices, batch_size_per_expert, None

    block.router_forward = types.MethodType(_forced_router_forward, block)


def _run_ep_no_sync_matches_synced():
    ep_mesh = _build_ep_mesh()

    block_ep = _build_block(ep_no_sync=False, uniform_expert_assignment=True)
    block_no_sync = _build_block(ep_no_sync=True, uniform_expert_assignment=True)
    block_ep.apply_ep(ep_mesh)
    block_no_sync.apply_ep(ep_mesh)

    _init_block_params(block_ep)
    block_no_sync.load_state_dict(block_ep.state_dict())
    block_ep.train()
    block_no_sync.train()

    x = torch.randn(2, 4, block_ep.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    y_ep = block_ep(x)
    y_no_sync = block_no_sync(x_ref)
    torch.testing.assert_close(y_no_sync, y_ep, atol=2e-4, rtol=2e-4)

    y_ep.sum().backward()
    y_no_sync.sum().backward()
    torch.testing.assert_close(x_ref.grad, x.grad, atol=3e-4, rtol=3e-4)

    ep_params = dict(block_ep.named_parameters())
    no_sync_params = dict(block_no_sync.named_parameters())
    for name, p_ep in ep_params.items():
        p_no_sync = no_sync_params[name]
        if p_ep.grad is None or p_no_sync.grad is None:
            continue
        torch.testing.assert_close(p_no_sync.grad, p_ep.grad, atol=8e-4, rtol=8e-4)


def _run_ep_no_sync_drop_behavior():
    ep_mesh = _build_ep_mesh()

    block_no_sync = _build_block(
        ep_no_sync=True,
        ep_no_sync_capacity_factor=0.25,
        uniform_expert_assignment=False,
    )
    block_no_sync.apply_ep(ep_mesh)
    _init_block_params(block_no_sync)
    _install_forced_router(block_no_sync)
    block_no_sync.train()

    x = torch.randn(2, 4, block_no_sync.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
    y = block_no_sync(x)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None

    dbg = block_no_sync._ep_no_sync_last_debug
    assert dbg["num_dropped"].item() > 0
    assert dbg["received_tokens_after_drop"].item() <= dbg["rank_capacity"].item()
    assert dbg["allowed_splits"].sum().item() == dbg["local_kept_tokens"].item()
    assert dbg["combined_tokens"].item() == dbg["local_kept_tokens"].item()
    assert dbg["zero_rows_after_local_unpermute"].item() >= dbg["num_dropped"].item()


def _run_ep_no_sync_quota_invariants():
    ep_mesh = _build_ep_mesh()

    block_no_sync = _build_block(
        ep_no_sync=True,
        ep_no_sync_capacity_factor=0.5,
        uniform_expert_assignment=False,
    )
    block_no_sync.apply_ep(ep_mesh)
    _init_block_params(block_no_sync)
    _install_forced_router(block_no_sync)
    block_no_sync.train()

    x = torch.randn(2, 4, block_no_sync.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
    y = block_no_sync(x)
    y.sum().backward()

    dbg = block_no_sync._ep_no_sync_last_debug
    assert dbg["allowed_splits"].sum().item() == dbg["local_kept_tokens"].item()
    assert dbg["received_tokens_after_drop"].item() <= dbg["rank_capacity"].item()
    assert dbg["combined_tokens"].item() == dbg["local_kept_tokens"].item()


def _run_ep_no_sync_hard_fail_setup():
    import olmo_core.nn.moe.v2.block as block_module

    ep_mesh = _build_ep_mesh()
    old_symm = block_module._symm_mem
    block_module._symm_mem = None
    try:
        block = _build_block(ep_no_sync=True)
        try:
            block.apply_ep(ep_mesh)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Expected RuntimeError when symmetric memory is unavailable")
    finally:
        block_module._symm_mem = old_symm


@requires_multi_gpu
def test_v2_ep_no_sync_matches_synced():
    run_distributed_test(_run_ep_no_sync_matches_synced, backend="nccl", start_method="spawn")


@requires_multi_gpu
def test_v2_ep_no_sync_drop_behavior():
    run_distributed_test(_run_ep_no_sync_drop_behavior, backend="nccl", start_method="spawn")


@requires_multi_gpu
def test_v2_ep_no_sync_quota_invariants():
    run_distributed_test(_run_ep_no_sync_quota_invariants, backend="nccl", start_method="spawn")


@requires_multi_gpu
def test_v2_ep_no_sync_hard_fail_setup():
    run_distributed_test(_run_ep_no_sync_hard_fail_setup, backend="nccl", start_method="spawn")
