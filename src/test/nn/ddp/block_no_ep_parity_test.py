from typing import Any

import torch
import torch.distributed as dist

from olmo_core.distributed.utils import unhide_from_torch
from olmo_core.testing import (
    requires_multi_gpu,
    requires_symm_mem_vdev2d,
    run_distributed_test,
)

from .block_no_sync_test import (
    _build_block,
    _build_ep_mesh,
    _init_block_params,
    _install_deterministic_topk_router,
)


def _copy_no_ep_weights_to_ep_shard(no_ep_block, ep_block):
    no_ep_params = dict(no_ep_block.named_parameters())
    no_ep_buffers = dict(no_ep_block.named_buffers())
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

        for name, buffer in ep_block.named_buffers():
            if name in no_ep_buffers and buffer.shape == no_ep_buffers[name].shape:
                buffer.copy_(no_ep_buffers[name])


def _assert_expert_grad_matches_no_ep_global_sum(
    no_ep_block, ep_block, *, atol: float, rtol: float
):
    local_experts = ep_block.routed_experts.num_local_experts
    start = ep_block.routed_experts.ep_rank * local_experts
    end = start + local_experts

    for name in ("w_up_gate", "w_down"):
        no_ep_grad = getattr(no_ep_block.routed_experts, name).grad
        ep_grad = getattr(ep_block.routed_experts, name).grad
        assert no_ep_grad is not None
        assert ep_grad is not None

        no_ep_grad_global = no_ep_grad.detach().clone()
        dist.all_reduce(no_ep_grad_global, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(
            ep_grad,
            no_ep_grad_global[start:end],
            atol=atol,
            rtol=rtol,
        )


def _run_dropless_path_matches_no_ep(
    *,
    rowwise: bool,
    shared: bool = False,
):
    ep_mesh = _build_ep_mesh()
    shared_kwargs: dict[str, Any] = (
        {"num_shared_experts": 1, "shared_hidden_size": 512} if shared else {}
    )

    no_ep_block = _build_block(
        ep_no_sync=False,
        d_model=512,
        hidden_size=1024,
        num_experts=8,
        top_k=2,
        uniform_expert_assignment=False,
        **shared_kwargs,
    )
    ep_block = _build_block(
        ep_no_sync=rowwise,
        ep_no_sync_use_rowwise_all_to_all=rowwise,
        ep_no_sync_capacity_factor=8.0,
        d_model=512,
        hidden_size=1024,
        num_experts=8,
        top_k=2,
        uniform_expert_assignment=False,
        **shared_kwargs,
    )
    ep_block.apply_ep(ep_mesh)

    _init_block_params(no_ep_block)
    _copy_no_ep_weights_to_ep_shard(no_ep_block, ep_block)
    _install_deterministic_topk_router(no_ep_block)
    _install_deterministic_topk_router(ep_block)

    if rowwise:
        ep_block.ep.rowwise_get_nblocks = 128
        ep_block.ep.rowwise_put_nblocks = 128
        ep_block.ep.rowwise_weighted_put_nblocks = 128
        ep_block.ep.validate()
        # The rowwise weighted combine backward is bf16/fp16-only, so run both blocks in bf16
        # (the path's production dtype). Cast after the identical param copy so the two stay in
        # sync. The synced path (rowwise=False) keeps fp32 and its tight tolerances.
        no_ep_block.to(dtype=torch.bfloat16)
        ep_block.to(dtype=torch.bfloat16)

    no_ep_block.train()
    ep_block.train()

    # bf16 rounding in the rowwise comm needs looser tolerances than the exact fp32 synced path.
    out_tol = 2e-2 if rowwise else 5e-4
    grad_tol = 2e-2 if rowwise else 1e-3

    x = torch.randn(
        1,
        8,
        no_ep_block.d_model,
        device="cuda",
        dtype=torch.bfloat16 if rowwise else torch.float32,
        requires_grad=True,
    )
    x_ep = x.detach().clone().requires_grad_(True)

    y_no_ep = no_ep_block(x)
    y_ep = ep_block(x_ep)
    torch.testing.assert_close(y_ep, y_no_ep, atol=out_tol, rtol=out_tol)
    if rowwise:
        drop_tokens_sum = ep_block._ep_no_sync_rowwise_drop_tokens_sum
        assert drop_tokens_sum is not None
        # The metric is stored wrapped in a `_HiddenTensor` (kept off the autograd graph);
        # unwrap before reading its value.
        assert unhide_from_torch(drop_tokens_sum).item() == 0

    loss_no_ep = y_no_ep.square().mean() + (0.1 * y_no_ep.sum())
    loss_ep = y_ep.square().mean() + (0.1 * y_ep.sum())
    loss_no_ep.backward()
    loss_ep.backward()

    torch.testing.assert_close(x_ep.grad, x.grad, atol=grad_tol, rtol=grad_tol)
    _assert_expert_grad_matches_no_ep_global_sum(
        no_ep_block,
        ep_block,
        atol=grad_tol,
        rtol=grad_tol,
    )


@requires_multi_gpu
def test_v2_synced_ep_dropless_matches_no_ep():
    run_distributed_test(
        _run_dropless_path_matches_no_ep,
        func_kwargs={"rowwise": False},
        backend="nccl",
        start_method="spawn",
    )


@requires_multi_gpu
@requires_symm_mem_vdev2d
def test_v2_rowwise_ep_dropless_matches_no_ep():
    run_distributed_test(
        _run_dropless_path_matches_no_ep,
        func_kwargs={"rowwise": True},
        backend="nccl",
        start_method="spawn",
    )
