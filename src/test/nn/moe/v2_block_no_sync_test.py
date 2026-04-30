import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.moe import MoERouterGatingFunction
from olmo_core.nn.moe.v2.block import MoEFusedV2TransformerBlock
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.testing import (
    requires_gpu,
    requires_grouped_gemm,
    requires_multi_gpu,
    run_distributed_test,
)


def test_v2_extracted_forward_module_names_importable():
    from olmo_core.nn.moe.v2 import (
        activation_debug,
        checkpointing,
        ep_no_sync_1d,
        ep_no_sync_buffers,
        ep_no_sync_rowwise,
        ep_no_sync_tbo_1d,
        ep_sync_1d,
        ep_sync_tbo,
        no_ep,
        tbo_state,
    )

    assert hasattr(activation_debug, "maybe_dump_ep_no_sync_saved_activations")
    assert hasattr(ep_sync_1d, "combined_forward_ep_1d")
    assert hasattr(ep_sync_tbo, "combined_forward_ep_tbo")
    assert hasattr(no_ep, "combined_forward_no_ep")
    assert hasattr(checkpointing, "checkpoint_recompute_context_fn")
    assert hasattr(ep_no_sync_buffers, "get_ep_no_sync_buffers")
    assert hasattr(ep_no_sync_buffers, "_NoSyncSymmBuffers")
    assert hasattr(ep_no_sync_1d, "combined_forward_ep_no_sync_1d")
    assert hasattr(ep_no_sync_rowwise, "combined_forward_ep_no_sync_rowwise")
    assert hasattr(ep_no_sync_tbo_1d, "combined_forward_ep_no_sync_tbo")
    assert hasattr(tbo_state, "SyncedTboPendingContext")


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
    d_model: int = 512,
    hidden_size: int = 1024,
    num_experts: int = 4,
    top_k: int = 1,
    uniform_expert_assignment: bool = True,
    init_device: str = "cuda",
    ep_no_sync_use_2d_all_to_all: bool = False,
) -> MoEFusedV2TransformerBlock:
    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=DType.float32,
    )
    from olmo_core.nn.attention import AttentionConfig, AttentionType

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
        ep_no_sync_use_2d_all_to_all=ep_no_sync_use_2d_all_to_all,
        ep_no_sync_capacity_factor=ep_no_sync_capacity_factor,
        ep_no_sync_major_align=1,
        init_device=init_device,
    )


def _init_block_params(block: MoEFusedV2TransformerBlock):
    torch.manual_seed(1234)
    with torch.no_grad():
        for p in block.parameters():
            if p.is_floating_point():
                p.normal_(mean=0.0, std=0.02)


def _install_forced_router(block: MoEFusedV2TransformerBlock):
    def _make_forced_forward(router):
        def _forced_forward(local_x, scores_only, loss_div_factor=None):
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

        return _forced_forward

    assert block.routed_experts_router is not None
    block.routed_experts_router.forward = _make_forced_forward(block.routed_experts_router)
    if block.shared_experts_router is not None:
        block.shared_experts_router.forward = _make_forced_forward(block.shared_experts_router)


def _install_deterministic_topk_router(block: MoEFusedV2TransformerBlock):
    def _make_deterministic_forward(router):
        def _deterministic_forward(local_x, scores_only, loss_div_factor=None):
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
            route_offsets = torch.arange(top_k, device=local_x.device, dtype=torch.long).unsqueeze(
                0
            )
            expert_indices = (token_ids + route_offsets + dist.get_rank() * 3) % num_experts
            expert_indices = expert_indices.view(B, S, top_k)

            weights = torch.arange(1, top_k + 1, device=local_x.device, dtype=local_x.dtype)
            weights = weights / weights.sum().clamp_min(1e-6)
            expert_weights = weights.view(1, 1, top_k).expand(B, S, top_k).contiguous()

            batch_size_per_expert = torch.bincount(
                expert_indices.reshape(-1), minlength=num_experts
            ).to(dtype=torch.long)
            return expert_weights, expert_indices, batch_size_per_expert, None

        return _deterministic_forward

    assert block.routed_experts_router is not None
    block.routed_experts_router.forward = _make_deterministic_forward(block.routed_experts_router)
    if block.shared_experts_router is not None:
        block.shared_experts_router.forward = _make_deterministic_forward(
            block.shared_experts_router
        )


@requires_gpu
@requires_grouped_gemm
def test_v2_no_ep_forward_backward_smoke():
    block = _build_block(ep_no_sync=False, init_device="cuda")
    _init_block_params(block)
    _install_forced_router(block)
    block.train()

    x = torch.randn(1, 8, block.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
    y = block(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for p in block.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


@requires_gpu
@requires_grouped_gemm
def test_v2_no_ep_apply_compile_forward_smoke():
    block = _build_block(
        ep_no_sync=False,
        d_model=128,
        hidden_size=256,
        num_experts=4,
        top_k=1,
        init_device="cuda",
    )
    _init_block_params(block)
    block.to(dtype=torch.bfloat16)
    _install_forced_router(block)
    block.train()
    block.apply_compile()

    x = torch.randn(1, 4, block.d_model, device="cuda", dtype=torch.bfloat16)
    y = block(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


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

    x = torch.randn(1, 8, block_ep.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
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

    x = torch.randn(
        1, 8, block_no_sync.d_model, device="cuda", dtype=torch.float32, requires_grad=True
    )
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

    x = torch.randn(
        1, 8, block_no_sync.d_model, device="cuda", dtype=torch.float32, requires_grad=True
    )
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


def _run_ep_no_sync_rowwise_matches_synced():
    ep_mesh = _build_ep_mesh()

    block_ep = _build_block(
        ep_no_sync=False,
        d_model=512,
        hidden_size=1024,
        num_experts=8,
        top_k=2,
        uniform_expert_assignment=False,
    )
    block_rowwise = _build_block(
        ep_no_sync=True,
        d_model=512,
        hidden_size=1024,
        num_experts=8,
        top_k=2,
        uniform_expert_assignment=False,
    )
    block_ep.apply_ep(ep_mesh)
    block_rowwise.apply_ep(ep_mesh)

    _init_block_params(block_ep)
    block_rowwise.load_state_dict(block_ep.state_dict())
    _install_deterministic_topk_router(block_ep)
    _install_deterministic_topk_router(block_rowwise)

    block_rowwise.ep_no_sync_use_rowwise_all_to_all = True
    block_rowwise.ep_no_sync_rowwise_nblocks = 128

    block_ep.train()
    block_rowwise.train()

    x = torch.randn(1, 8, block_ep.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
    x_rowwise = x.detach().clone().requires_grad_(True)

    y_ep = block_ep(x)
    y_rowwise = block_rowwise(x_rowwise)
    torch.testing.assert_close(y_rowwise, y_ep, atol=5e-4, rtol=5e-4)

    loss_ep = y_ep.square().mean() + (0.1 * y_ep.sum())
    loss_rowwise = y_rowwise.square().mean() + (0.1 * y_rowwise.sum())
    loss_ep.backward()
    loss_rowwise.backward()

    torch.testing.assert_close(x_rowwise.grad, x.grad, atol=5e-4, rtol=5e-4)

    ep_params = dict(block_ep.named_parameters())
    rowwise_params = dict(block_rowwise.named_parameters())
    for name, p_ep in ep_params.items():
        p_rowwise = rowwise_params[name]
        if p_ep.grad is None or p_rowwise.grad is None:
            continue
        torch.testing.assert_close(p_rowwise.grad, p_ep.grad, atol=1e-3, rtol=1e-3)


def _run_ep_no_sync_rowwise_drop_matches_independent_rowwise_block():
    ep_mesh = _build_ep_mesh()

    block_a = _build_block(
        ep_no_sync=True,
        ep_no_sync_capacity_factor=0.5,
        d_model=512,
        hidden_size=1024,
        num_experts=8,
        top_k=4,
        uniform_expert_assignment=False,
    )
    block_b = _build_block(
        ep_no_sync=True,
        ep_no_sync_capacity_factor=0.5,
        d_model=512,
        hidden_size=1024,
        num_experts=8,
        top_k=4,
        uniform_expert_assignment=False,
    )
    block_a.apply_ep(ep_mesh)
    block_b.apply_ep(ep_mesh)

    _init_block_params(block_a)
    block_b.load_state_dict(block_a.state_dict())
    _install_deterministic_topk_router(block_a)
    _install_deterministic_topk_router(block_b)

    block_a.ep_no_sync_use_rowwise_all_to_all = True
    block_a.ep_no_sync_rowwise_nblocks = 128

    block_b.ep_no_sync_use_rowwise_all_to_all = True
    block_b.ep_no_sync_rowwise_nblocks = 128

    block_a.train()
    block_b.train()

    x = torch.randn(2, 64, block_a.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
    x_b = x.detach().clone().requires_grad_(True)

    y_a = block_a(x)
    y_b = block_b(x_b)
    assert torch.isfinite(y_a).all()
    assert torch.isfinite(y_b).all()
    torch.testing.assert_close(y_b, y_a, atol=8e-4, rtol=8e-4)

    loss_a = y_a.square().mean() + (0.1 * y_a.sum())
    loss_b = y_b.square().mean() + (0.1 * y_b.sum())
    loss_a.backward()
    loss_b.backward()

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(x_b.grad).all()
    torch.testing.assert_close(x_b.grad, x.grad, atol=2e-3, rtol=2e-3)

    params_a = dict(block_a.named_parameters())
    params_b = dict(block_b.named_parameters())
    for name, p_a in params_a.items():
        p_b = params_b[name]
        if p_a.grad is None or p_b.grad is None:
            continue
        assert torch.isfinite(p_a.grad).all()
        assert torch.isfinite(p_b.grad).all()
        torch.testing.assert_close(p_b.grad, p_a.grad, atol=3e-3, rtol=3e-3)


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


@requires_multi_gpu
def test_v2_ep_no_sync_rowwise_matches_synced():
    run_distributed_test(
        _run_ep_no_sync_rowwise_matches_synced, backend="nccl", start_method="spawn"
    )


def test_v2_ep_no_sync_2d_all_to_all_rejected():
    with pytest.raises(OLMoConfigurationError, match="2D all_to_all path was removed"):
        _build_block(
            ep_no_sync=True,
            init_device="cpu",
            ep_no_sync_use_2d_all_to_all=True,
        )


@requires_multi_gpu
def test_v2_ep_no_sync_rowwise_drop_matches_independent_rowwise_block():
    run_distributed_test(
        _run_ep_no_sync_rowwise_drop_matches_independent_rowwise_block,
        backend="nccl",
        start_method="spawn",
    )
