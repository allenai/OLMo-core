import os
import struct

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.moe import MoERouterGatingFunction
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock
from olmo_core.nn.moe.v2.ep_config import (
    ExpertParallelConfig,
    ExpertParallelPath,
    ExpertParallelSchedule,
)
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.testing import requires_gpu, requires_grouped_gemm, requires_multi_gpu, run_distributed_test


def test_v2_extracted_forward_module_names_importable():
    from olmo_core.nn.moe.v2 import (
        activation_debug,
        checkpointing,
        ep_no_sync_1d,
        ep_no_sync_buffers,
        ep_no_sync_rowwise,
        ep_no_sync_rowwise_wave,
        ep_sync_1d,
        no_ep,
    )

    assert hasattr(activation_debug, "maybe_dump_ep_no_sync_saved_activations")
    assert hasattr(ep_sync_1d, "combined_forward_ep_1d")
    assert hasattr(no_ep, "combined_forward_no_ep")
    assert hasattr(checkpointing, "checkpoint_recompute_context_fn")
    assert hasattr(ep_no_sync_buffers, "get_ep_no_sync_buffers")
    assert hasattr(ep_no_sync_buffers, "_NoSyncSymmBuffers")
    assert hasattr(ep_no_sync_1d, "combined_forward_ep_no_sync_1d")
    assert hasattr(ep_no_sync_rowwise, "combined_forward_ep_no_sync_rowwise")
    assert hasattr(ep_no_sync_rowwise_wave, "combined_forward_ep_no_sync_rowwise_wave")


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
    ep: ExpertParallelConfig | None = None,
    ep_no_sync_capacity_factor: float = 2.0,
    d_model: int = 512,
    hidden_size: int = 1024,
    num_experts: int = 4,
    top_k: int = 1,
    num_shared_experts: int = 0,
    shared_hidden_size: int = 512,
    uniform_expert_assignment: bool = True,
    init_device: str = "cuda",
    checkpoint_combined_ep_tbo: bool = False,
    rowwise_fp8=None,
    ep_no_sync_use_2d_all_to_all: bool = False,
    ep_no_sync_use_rowwise_all_to_all: bool = False,
    ep_no_sync_rowwise_backend: str = "nvshmem",
) -> OLMoDDPTransformerBlock:
    if ep is None:
        rowwise_backend = ep_no_sync_rowwise_backend.lower()
        if rowwise_backend != "nvshmem":
            raise OLMoConfigurationError(
                "ep_no_sync_rowwise_backend must be 'nvshmem'"
            )
        if ep_no_sync_use_2d_all_to_all:
            path = ExpertParallelPath.no_sync_2d_removed
        elif not ep_no_sync:
            path = ExpertParallelPath.sync_1d
        elif ep_no_sync_use_rowwise_all_to_all:
            path = ExpertParallelPath.rowwise_nvshmem
        else:
            path = ExpertParallelPath.no_sync_1d
        ep = ExpertParallelConfig(
            path=path,
            capacity_factor=ep_no_sync_capacity_factor,
            rowwise_nblocks=256,
            checkpoint_tbo=checkpoint_combined_ep_tbo,
        )

    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=DType.float32,
    )
    from olmo_core.nn.attention import AttentionConfig, AttentionType

    return OLMoDDPTransformerBlock(
        d_model=d_model,
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
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            gating_function=MoERouterGatingFunction.softmax,
            uniform_expert_assignment=uniform_expert_assignment,
            lb_loss_weight=None,
            z_loss_weight=None,
            dtype=DType.float32,
        ),
        routed_experts=RoutedExpertsConfig(
            d_model=d_model,
            hidden_size=hidden_size,
            num_experts=num_experts,
            bias=False,
            dtype=DType.float32,
        ),
        shared_experts=(
            SharedExpertsConfig(
                d_model=d_model,
                hidden_size=shared_hidden_size,
                num_experts=num_shared_experts,
                bias=False,
                dtype=DType.float32,
            )
            if num_shared_experts > 0
            else None
        ),
        shared_experts_router=None,
        feed_forward_norm=layer_norm,
        ep=ep,
        rowwise_fp8=rowwise_fp8,
        init_device=init_device,
    )


def test_v2_ep_config_selects_rowwise_wave_path():
    block = _build_block(
        ep_no_sync=False,
        ep=ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_wave,
            rowwise_wave_num_waves=4,
            rowwise_wave_mode="EXPERT",
        ),
        init_device="cpu",
    )
    assert block.ep.path == ExpertParallelPath.rowwise_wave
    assert block.ep.no_sync is True
    assert block.ep.is_rowwise is True
    assert block.ep.uses_rowwise_buffers is True
    assert block.ep.rowwise_transport == "nvshmem"
    assert block.ep.rowwise_wave_num_waves == 4
    assert block.ep.rowwise_wave_mode == "expert"


def test_v2_ep_config_tbo_only_allows_rowwise_nvshmem():
    ExpertParallelConfig(
        path=ExpertParallelPath.rowwise_nvshmem,
        schedule=ExpertParallelSchedule.tbo,
    ).validate()

    for path in (
        ExpertParallelPath.sync_1d,
        ExpertParallelPath.no_sync_1d,
        ExpertParallelPath.rowwise_wave,
    ):
        with pytest.raises(OLMoConfigurationError, match="only supported"):
            ExpertParallelConfig(
                path=path,
                schedule=ExpertParallelSchedule.tbo,
            ).validate()


def test_v2_rowwise_nvshmem_tbo_forward_method_is_available():
    block = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
        init_device="cpu",
    )
    assert block.ep.path == ExpertParallelPath.rowwise_nvshmem
    assert callable(block.combined_forward_rowwise_nvshmem_tbo)


def test_v2_rowwise_wave_num_waves_requires_rowwise_wave_path():
    with pytest.raises(OLMoConfigurationError, match="rowwise_wave_num_waves"):
        ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_nvshmem,
            rowwise_wave_num_waves=2,
        ).validate()


def test_v2_rowwise_wave_rejects_invalid_mode_and_num_waves():
    with pytest.raises(OLMoConfigurationError, match="rowwise_wave_num_waves"):
        ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_wave,
            rowwise_wave_num_waves=0,
        ).validate()

    with pytest.raises(OLMoConfigurationError, match="rowwise_wave_mode"):
        ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_wave,
            rowwise_wave_mode="token",
        ).validate()


def test_v2_rowwise_backend_rejects_unknown_backend():
    with pytest.raises(OLMoConfigurationError, match="ep_no_sync_rowwise_backend"):
        _build_block(
            ep_no_sync=True,
            ep_no_sync_use_rowwise_all_to_all=True,
            ep_no_sync_rowwise_backend="unknown",
            init_device="cpu",
        )


def test_v2_rowwise_wave_forward_method_is_available():
    block = _build_block(
        ep_no_sync=False,
        ep=ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_wave,
            rowwise_wave_num_waves=2,
        ),
        init_device="cpu",
    )
    assert callable(block.combined_forward_ep_no_sync_rowwise_wave)


def _reference_rowwise_dispatch_bf16(
    source_input: torch.Tensor,
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    *,
    ep_world_size: int,
    rank_capacity: int,
) -> torch.Tensor:
    output = torch.zeros(
        (ep_world_size, rank_capacity, source_input.shape[1]),
        dtype=source_input.dtype,
        device=source_input.device,
    )
    for token_idx in range(dst_ranks.shape[0]):
        for topk_idx in range(dst_ranks.shape[1]):
            rank = int(dst_ranks[token_idx, topk_idx].item())
            row = int(dst_rows[token_idx, topk_idx].item())
            if rank >= 0 and row >= 0:
                output[rank, row] = source_input[token_idx]
    return output


def _reference_rowwise_combine_bf16(
    expert_out_by_rank: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    *,
    probs: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros(
        (src_ranks.shape[0], expert_out_by_rank.shape[2]),
        dtype=torch.float32,
        device=expert_out_by_rank.device,
    )
    for token_idx in range(src_ranks.shape[0]):
        for topk_idx in range(src_ranks.shape[1]):
            rank = int(src_ranks[token_idx, topk_idx].item())
            row = int(src_rows[token_idx, topk_idx].item())
            if rank >= 0 and row >= 0:
                output[token_idx] += (
                    expert_out_by_rank[rank, row].float()
                    * probs[token_idx, topk_idx]
                )
    return output.to(dtype=torch.bfloat16)


def _init_block_params(block: OLMoDDPTransformerBlock):
    torch.manual_seed(1234)
    with torch.no_grad():
        for p in block.parameters():
            if p.is_floating_point():
                p.normal_(mean=0.0, std=0.02)


def _install_forced_router(block: OLMoDDPTransformerBlock):
    def _make_forced_forward(router):
        def _forced_forward(local_x, scores_only, loss_div_factor=None):
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

        return _forced_forward

    assert block.routed_experts_router is not None
    block.routed_experts_router.forward = _make_forced_forward(block.routed_experts_router)
    if block.shared_experts_router is not None:
        block.shared_experts_router.forward = _make_forced_forward(block.shared_experts_router)


def _install_deterministic_topk_router(block: OLMoDDPTransformerBlock):
    def _make_deterministic_forward(router):
        def _deterministic_forward(local_x, scores_only, loss_div_factor=None):
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
                expert_indices.reshape(-1), minlength=num_experts
            ).to(dtype=torch.long)
            return expert_weights, expert_indices, batch_size_per_expert, None

        return _deterministic_forward

    assert block.routed_experts_router is not None
    block.routed_experts_router.forward = _make_deterministic_forward(block.routed_experts_router)
    if block.shared_experts_router is not None:
        block.shared_experts_router.forward = _make_deterministic_forward(block.shared_experts_router)


def _install_local_deterministic_topk_router(block: OLMoDDPTransformerBlock):
    """Deterministic router for single-process tests that do not initialize dist."""

    def _make_deterministic_forward(router):
        def _deterministic_forward(local_x, scores_only, loss_div_factor=None):
            del scores_only, loss_div_factor
            B, S, _ = local_x.shape
            tokens = B * S
            token_ids = torch.arange(tokens, device=local_x.device).unsqueeze(1)
            route_offsets = torch.arange(router.top_k, device=local_x.device).unsqueeze(0)
            expert_indices = (token_ids * 3 + route_offsets * 5) % router.num_experts
            logits = torch.linspace(
                0.25,
                0.75,
                steps=router.top_k,
                device=local_x.device,
                dtype=torch.float32,
            ).view(1, router.top_k)
            expert_weights = torch.softmax(logits.expand(tokens, router.top_k), dim=-1).to(
                dtype=local_x.dtype
            )
            batch_size_per_expert = torch.bincount(
                expert_indices.reshape(-1),
                minlength=router.num_experts,
            ).to(dtype=torch.int32)
            return expert_weights, expert_indices.to(dtype=torch.long), batch_size_per_expert, None

        return _deterministic_forward

    assert block.routed_experts_router is not None
    block.routed_experts_router.forward = _make_deterministic_forward(block.routed_experts_router)
    if block.shared_experts_router is not None:
        block.shared_experts_router.forward = _make_deterministic_forward(block.shared_experts_router)


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
def test_v2_no_ep_repeated_forward_backward_is_stable():
    block = _build_block(
        ep_no_sync=False,
        d_model=256,
        hidden_size=512,
        num_experts=8,
        top_k=4,
        num_shared_experts=1,
        shared_hidden_size=256,
        uniform_expert_assignment=False,
        init_device="cuda",
    )
    _init_block_params(block)
    _install_local_deterministic_topk_router(block)
    block.train()

    torch.manual_seed(1234)
    x0 = torch.randn(2, 256, block.d_model, device="cuda", dtype=torch.float32)

    def run_once():
        block.zero_grad(set_to_none=True)
        x = x0.detach().clone().requires_grad_(True)
        y = block(x)
        loss = y.float().square().mean() + 0.03125 * y.float().sum()
        loss.backward()
        torch.cuda.synchronize()
        grads = {
            name: p.grad.detach().clone()
            for name, p in block.named_parameters()
            if p.grad is not None
        }
        assert x.grad is not None
        return y.detach().clone(), x.grad.detach().clone(), grads

    ref_y, ref_x_grad, ref_grads = run_once()
    for _ in range(3):
        y, x_grad, grads = run_once()
        torch.testing.assert_close(y, ref_y, atol=0.0, rtol=0.0)
        torch.testing.assert_close(x_grad, ref_x_grad, atol=2e-8, rtol=0.0)
        for name, ref_grad in ref_grads.items():
            torch.testing.assert_close(grads[name], ref_grad, atol=1e-6, rtol=0.0)


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

    x = torch.randn(1, 8, block_no_sync.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
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

    x = torch.randn(1, 8, block_no_sync.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
    y = block_no_sync(x)
    y.sum().backward()

    dbg = block_no_sync._ep_no_sync_last_debug
    assert dbg["allowed_splits"].sum().item() == dbg["local_kept_tokens"].item()
    assert dbg["received_tokens_after_drop"].item() <= dbg["rank_capacity"].item()
    assert dbg["combined_tokens"].item() == dbg["local_kept_tokens"].item()


def _run_ep_no_sync_hard_fail_setup():
    import olmo_core.nn.ddp.block as block_module

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
        ep_no_sync_use_rowwise_all_to_all=True,
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

    block_rowwise.ep.rowwise_nblocks = 128
    block_rowwise.ep.validate()

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


def _run_ep_no_sync_rowwise_wave_matches_rowwise():
    ep_mesh = _build_ep_mesh()

    block_rowwise = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
        d_model=128,
        hidden_size=256,
        num_experts=8,
        top_k=4,
        uniform_expert_assignment=False,
    )
    block_wave = _build_block(
        ep_no_sync=False,
        ep=ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_wave,
            capacity_factor=2.0,
            rowwise_nblocks=128,
            rowwise_wave_num_waves=2,
        ),
        d_model=128,
        hidden_size=256,
        num_experts=8,
        top_k=4,
        uniform_expert_assignment=False,
    )
    block_rowwise.apply_ep(ep_mesh)
    block_wave.apply_ep(ep_mesh)

    _init_block_params(block_rowwise)
    block_wave.load_state_dict(block_rowwise.state_dict())
    _install_deterministic_topk_router(block_rowwise)
    _install_deterministic_topk_router(block_wave)

    block_rowwise.ep.rowwise_nblocks = 128
    block_wave.ep.rowwise_nblocks = 128
    block_rowwise.ep.validate()
    block_wave.ep.validate()
    block_rowwise.train()
    block_wave.train()

    x = torch.randn(1, 16, block_rowwise.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
    x_wave = x.detach().clone().requires_grad_(True)

    with pytest.warns(RuntimeWarning, match="rowwise_wave"):
        y_wave = block_wave(x_wave)
    y_rowwise = block_rowwise(x)
    assert y_wave.shape == y_rowwise.shape
    assert torch.isfinite(y_wave).all()
    torch.testing.assert_close(y_wave, y_rowwise, atol=1e-3, rtol=1e-3)

    loss_rowwise = y_rowwise.square().mean() + (0.1 * y_rowwise.sum())
    loss_wave = y_wave.square().mean() + (0.1 * y_wave.sum())
    loss_rowwise.backward()
    loss_wave.backward()

    assert x_wave.grad is not None
    assert x.grad is not None
    torch.testing.assert_close(x_wave.grad, x.grad, atol=2e-3, rtol=2e-3)

    rowwise_params = dict(block_rowwise.named_parameters())
    wave_params = dict(block_wave.named_parameters())
    for name, p_rowwise in rowwise_params.items():
        p_wave = wave_params[name]
        if p_rowwise.grad is None or p_wave.grad is None:
            continue
        torch.testing.assert_close(
            p_wave.grad,
            p_rowwise.grad,
            atol=3e-3,
            rtol=3e-3,
            msg=f"rowwise_wave gradient mismatch for {name}",
        )

    block_rowwise_bf16 = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
        d_model=128,
        hidden_size=256,
        num_experts=8,
        top_k=4,
        uniform_expert_assignment=False,
    )
    block_wave_bf16 = _build_block(
        ep_no_sync=False,
        ep=ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_wave,
            capacity_factor=2.0,
            rowwise_nblocks=32,
            rowwise_wave_num_waves=2,
        ),
        d_model=128,
        hidden_size=256,
        num_experts=8,
        top_k=4,
        uniform_expert_assignment=False,
    )
    block_rowwise_bf16.apply_ep(ep_mesh)
    block_wave_bf16.apply_ep(ep_mesh)

    _init_block_params(block_rowwise_bf16)
    block_wave_bf16.load_state_dict(block_rowwise_bf16.state_dict())
    _install_deterministic_topk_router(block_rowwise_bf16)
    _install_deterministic_topk_router(block_wave_bf16)

    block_rowwise_bf16.to(dtype=torch.bfloat16)
    block_wave_bf16.to(dtype=torch.bfloat16)
    block_rowwise_bf16.ep.rowwise_nblocks = 32
    block_wave_bf16.ep.rowwise_nblocks = 32
    block_rowwise_bf16.ep.validate()
    block_wave_bf16.ep.validate()
    block_rowwise_bf16.train()
    block_wave_bf16.train()

    x_bf16 = (0.2 * torch.randn(1, 16, block_rowwise_bf16.d_model, device="cuda")).to(
        dtype=torch.bfloat16
    )
    x_bf16.requires_grad_(True)
    x_wave_bf16 = x_bf16.detach().clone().requires_grad_(True)

    y_rowwise_bf16 = block_rowwise_bf16(x_bf16)
    y_wave_bf16 = block_wave_bf16(x_wave_bf16)
    assert y_wave_bf16.shape == y_rowwise_bf16.shape
    assert torch.isfinite(y_wave_bf16).all()
    torch.testing.assert_close(y_wave_bf16, y_rowwise_bf16, atol=2e-2, rtol=2e-2)

    loss_rowwise_bf16 = y_rowwise_bf16.square().mean() + (0.1 * y_rowwise_bf16.sum())
    loss_wave_bf16 = y_wave_bf16.square().mean() + (0.1 * y_wave_bf16.sum())
    loss_rowwise_bf16.backward()
    loss_wave_bf16.backward()

    assert x_wave_bf16.grad is not None
    assert x_bf16.grad is not None
    torch.testing.assert_close(x_wave_bf16.grad, x_bf16.grad, atol=2e-2, rtol=2e-2)

    rowwise_bf16_params = dict(block_rowwise_bf16.named_parameters())
    wave_bf16_params = dict(block_wave_bf16.named_parameters())
    for name, p_rowwise in rowwise_bf16_params.items():
        p_wave = wave_bf16_params[name]
        if p_rowwise.grad is None or p_wave.grad is None:
            continue
        torch.testing.assert_close(
            p_wave.grad,
            p_rowwise.grad,
            atol=3e-2,
            rtol=3e-2,
            msg=f"bf16 rowwise_wave gradient mismatch for {name}",
        )

    block_rowwise_eval = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
        d_model=128,
        hidden_size=256,
        num_experts=8,
        top_k=4,
        uniform_expert_assignment=False,
    )
    block_wave_eval = _build_block(
        ep_no_sync=False,
        ep=ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_wave,
            capacity_factor=2.0,
            rowwise_nblocks=32,
            rowwise_wave_num_waves=4,
        ),
        d_model=128,
        hidden_size=256,
        num_experts=8,
        top_k=4,
        uniform_expert_assignment=False,
    )
    block_rowwise_eval.apply_ep(ep_mesh)
    block_wave_eval.apply_ep(ep_mesh)

    _init_block_params(block_rowwise_eval)
    block_wave_eval.load_state_dict(block_rowwise_eval.state_dict())
    _install_deterministic_topk_router(block_rowwise_eval)
    _install_deterministic_topk_router(block_wave_eval)

    block_rowwise_eval.to(dtype=torch.bfloat16)
    block_wave_eval.to(dtype=torch.bfloat16)
    block_rowwise_eval.ep.rowwise_nblocks = 32
    block_wave_eval.ep.rowwise_nblocks = 32
    block_rowwise_eval.ep.validate()
    block_wave_eval.ep.validate()
    block_rowwise_eval.eval()
    block_wave_eval.eval()

    x_eval = (0.2 * torch.randn(1, 16, block_rowwise_eval.d_model, device="cuda")).to(
        dtype=torch.bfloat16
    )
    with torch.no_grad():
        y_rowwise_eval = block_rowwise_eval(x_eval)
        y_wave_eval = block_wave_eval(x_eval.detach().clone())
    assert y_wave_eval.shape == y_rowwise_eval.shape
    assert torch.isfinite(y_wave_eval).all()
    torch.testing.assert_close(y_wave_eval, y_rowwise_eval, atol=2e-2, rtol=2e-2)


def _run_ep_no_sync_rowwise_drop_matches_independent_rowwise_block():
    ep_mesh = _build_ep_mesh()

    block_a = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
        ep_no_sync_capacity_factor=0.5,
        d_model=512,
        hidden_size=1024,
        num_experts=8,
        top_k=4,
        uniform_expert_assignment=False,
    )
    block_b = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
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

    block_a.ep.rowwise_nblocks = 128
    block_a.ep.validate()

    block_b.ep.rowwise_nblocks = 128
    block_b.ep.validate()

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


def _poison_rowwise_capacity_tails(block: OLMoDDPTransformerBlock, *, value: float) -> int:
    recv_splits = getattr(block, "_debug_rowwise_recv_splits_by_src_local", None)
    if recv_splits is None:
        raise RuntimeError("rowwise debug tensors were not captured")
    valid_rows = int(recv_splits.sum().item())
    poisoned_rows = 0

    def poison_tensor(tensor: torch.Tensor | None) -> None:
        nonlocal poisoned_rows
        if tensor is None or tensor.ndim != 2 or not tensor.is_floating_point():
            return
        tail_rows = int(tensor.shape[0]) - valid_rows
        if tail_rows <= 0:
            return
        tensor.narrow(0, valid_rows, tail_rows).fill_(value)
        poisoned_rows += tail_rows

    with torch.no_grad():
        pools = getattr(block, "_ep_no_sync_symm_lease_pools", {})
        dispatch_pool = pools.get("dispatch_out")
        if dispatch_pool is not None:
            for slot in dispatch_pool._slots:
                poison_tensor(slot.get("dispatch_out"))

        for buffers in getattr(block, "_ep_no_sync_static_buffer_cache", {}).values():
            poison_tensor(getattr(buffers, "combine_in", None))

    return poisoned_rows


def _run_ep_no_sync_rowwise_capacity_tail_poison_does_not_change_backward():
    old_debug = os.environ.get("OLMO_MOE_ROWWISE_DEBUG_TENSORS")
    os.environ["OLMO_MOE_ROWWISE_DEBUG_TENSORS"] = "1"
    try:
        ep_mesh = _build_ep_mesh()

        block_a = _build_block(
            ep_no_sync=True,
            ep_no_sync_use_rowwise_all_to_all=True,
            ep_no_sync_capacity_factor=8.0,
            d_model=512,
            hidden_size=1024,
            num_experts=8,
            top_k=4,
            uniform_expert_assignment=False,
        )
        block_b = _build_block(
            ep_no_sync=True,
            ep_no_sync_use_rowwise_all_to_all=True,
            ep_no_sync_capacity_factor=8.0,
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

        block_a.ep.rowwise_nblocks = 128
        block_b.ep.rowwise_nblocks = 128
        block_a.ep.validate()
        block_b.ep.validate()
        block_a.train()
        block_b.train()

        x = torch.randn(2, 64, block_a.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
        x_b = x.detach().clone().requires_grad_(True)

        y_a = block_a(x)
        y_b = block_b(x_b)
        torch.testing.assert_close(y_b, y_a, atol=8e-4, rtol=8e-4)

        poisoned_rows = _poison_rowwise_capacity_tails(block_b, value=2048.0)
        assert poisoned_rows > 0

        loss_a = y_a.square().mean() + (0.1 * y_a.sum())
        loss_b = y_b.square().mean() + (0.1 * y_b.sum())
        loss_a.backward()
        loss_b.backward()

        assert x.grad is not None
        assert x_b.grad is not None
        torch.testing.assert_close(x_b.grad, x.grad, atol=2e-3, rtol=2e-3)

        params_a = dict(block_a.named_parameters())
        params_b = dict(block_b.named_parameters())
        for name, p_a in params_a.items():
            p_b = params_b[name]
            if p_a.grad is None or p_b.grad is None:
                continue
            torch.testing.assert_close(
                p_b.grad,
                p_a.grad,
                atol=3e-3,
                rtol=3e-3,
                msg=f"capacity tail poison changed gradient for {name}",
            )
    finally:
        if old_debug is None:
            os.environ.pop("OLMO_MOE_ROWWISE_DEBUG_TENSORS", None)
        else:
            os.environ["OLMO_MOE_ROWWISE_DEBUG_TENSORS"] = old_debug


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
    run_distributed_test(_run_ep_no_sync_rowwise_matches_synced, backend="nccl", start_method="spawn")


@requires_multi_gpu
@requires_grouped_gemm
def test_v2_ep_no_sync_rowwise_wave_matches_rowwise():
    run_distributed_test(
        _run_ep_no_sync_rowwise_wave_matches_rowwise,
        backend="nccl",
        start_method="spawn",
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


@requires_multi_gpu
def test_v2_ep_no_sync_rowwise_capacity_tail_poison_does_not_change_backward():
    run_distributed_test(
        _run_ep_no_sync_rowwise_capacity_tail_poison_does_not_change_backward,
        backend="nccl",
        start_method="spawn",
    )
