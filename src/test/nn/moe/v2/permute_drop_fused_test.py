import pytest
import torch

from olmo_core.nn.moe.utils import (
    moe_permute_1d_fused_drop_no_compile,
    moe_permute_no_compile,
)
from olmo_core.testing import requires_gpu, requires_te


def _build_keep_reorder(
    *,
    num_rows: int,
    keep_fraction: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    keep_count = int(round(keep_fraction * num_rows))
    keep_count = max(0, min(keep_count, num_rows))
    keep_mask = torch.zeros(num_rows, device=device, dtype=torch.bool)
    if keep_count > 0:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        keep_rows = torch.randperm(num_rows, generator=g, device=device)[:keep_count]
        keep_mask[keep_rows] = True

    token_ids = torch.arange(num_rows, device=device, dtype=torch.long)
    keep_i64 = keep_mask.to(dtype=torch.long)
    drop_i64 = (~keep_mask).to(dtype=torch.long)
    keep_rank = torch.cumsum(keep_i64, dim=0) - 1
    drop_rank = torch.cumsum(drop_i64, dim=0) - 1
    num_kept = keep_i64.sum(dtype=torch.long)
    packed_pos = torch.where(keep_mask, keep_rank, num_kept + drop_rank)

    reorder_indices = torch.empty_like(token_ids)
    reorder_indices.scatter_(0, packed_pos, token_ids)
    inverse_reorder_indices = torch.empty_like(reorder_indices)
    inverse_reorder_indices.scatter_(0, reorder_indices, token_ids)
    return reorder_indices, inverse_reorder_indices


@requires_gpu
@requires_te
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k", [1, 2, 4])
@pytest.mark.parametrize("keep_fraction", [1.0, 0.7, 0.3])
def test_permute_drop_1d_fused_matches_legacy(
    dtype: torch.dtype,
    top_k: int,
    keep_fraction: float,
):
    torch.manual_seed(13)
    device = torch.device("cuda")
    num_tokens = 128
    d_model = 256
    num_experts = 16
    num_out_tokens = num_tokens * top_k

    x = torch.randn(num_tokens, d_model, device=device, dtype=dtype)
    routing_map = torch.randint(
        low=0,
        high=num_experts,
        size=(num_tokens, top_k),
        device=device,
        dtype=torch.int32,
    )

    x_legacy = x.detach().clone().requires_grad_(True)
    permuted_legacy, row_id_map_legacy = moe_permute_no_compile(
        inp=x_legacy,
        routing_map=routing_map,
        num_out_tokens=num_out_tokens,
        map_type="index",
    )
    reorder_indices, inverse_reorder_indices = _build_keep_reorder(
        num_rows=permuted_legacy.shape[0],
        keep_fraction=keep_fraction,
        seed=19,
        device=device,
    )
    dropped_legacy = permuted_legacy.index_select(0, reorder_indices)

    x_fused = x.detach().clone().requires_grad_(True)
    dropped_fused, row_id_map_fused = moe_permute_1d_fused_drop_no_compile(
        inp=x_fused,
        routing_map=routing_map,
        num_out_tokens=num_out_tokens,
        reorder_indices=reorder_indices,
        inverse_reorder_indices=inverse_reorder_indices,
        map_type="index",
    )

    torch.testing.assert_close(dropped_fused, dropped_legacy, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(row_id_map_fused, row_id_map_legacy, atol=0, rtol=0)

    grad_out = torch.randn_like(dropped_legacy)
    (dropped_legacy * grad_out).sum().backward()
    (dropped_fused * grad_out).sum().backward()

    assert x_legacy.grad is not None
    assert x_fused.grad is not None
    torch.testing.assert_close(x_fused.grad, x_legacy.grad, atol=5e-3, rtol=5e-3)


@requires_gpu
@requires_te
def test_permute_drop_1d_fused_out_buffer():
    torch.manual_seed(17)
    device = torch.device("cuda")
    num_tokens = 64
    top_k = 2
    d_model = 128
    num_experts = 8
    num_out_tokens = num_tokens * top_k

    x = torch.randn(num_tokens, d_model, device=device, dtype=torch.float16, requires_grad=True)
    routing_map = torch.randint(
        low=0,
        high=num_experts,
        size=(num_tokens, top_k),
        device=device,
        dtype=torch.int32,
    )
    permuted, _ = moe_permute_no_compile(
        inp=x.detach(),
        routing_map=routing_map,
        num_out_tokens=num_out_tokens,
        map_type="index",
    )
    reorder_indices, inverse_reorder_indices = _build_keep_reorder(
        num_rows=permuted.shape[0],
        keep_fraction=0.6,
        seed=23,
        device=device,
    )
    out_buffer = torch.empty((num_out_tokens, d_model), device=device, dtype=x.dtype)

    dropped, _ = moe_permute_1d_fused_drop_no_compile(
        inp=x,
        routing_map=routing_map,
        num_out_tokens=num_out_tokens,
        reorder_indices=reorder_indices,
        inverse_reorder_indices=inverse_reorder_indices,
        out=out_buffer,
        map_type="index",
    )
    assert dropped.untyped_storage().data_ptr() == out_buffer.untyped_storage().data_ptr()

    (dropped * dropped).sum().backward()
    assert x.grad is not None
