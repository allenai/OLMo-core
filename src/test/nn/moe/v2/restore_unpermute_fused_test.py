import pytest
import torch

from olmo_core.nn.moe.utils import (
    moe_permute_no_compile,
    moe_unpermute_1d_fused_drop_no_compile,
    moe_unpermute_no_compile,
)
from olmo_core.testing import requires_gpu, requires_te


def _build_keep_reorder(
    *,
    num_rows: int,
    keep_fraction: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    packed_keep_mask = keep_mask.index_select(0, reorder_indices)
    return reorder_indices, inverse_reorder_indices, packed_keep_mask


def _legacy_restore_drop_unpermute(
    *,
    combine_out: torch.Tensor,
    row_id_map: torch.Tensor,
    local_inverse_reorder_indices: torch.Tensor,
    packed_keep_mask: torch.Tensor,
    merging_probs: torch.Tensor,
    restore_shape: torch.Size,
) -> torch.Tensor:
    restored = combine_out.index_select(0, local_inverse_reorder_indices)
    restored_keep_mask = packed_keep_mask.index_select(0, local_inverse_reorder_indices)
    restored = torch.where(
        restored_keep_mask.unsqueeze(-1),
        restored,
        torch.zeros_like(restored),
    )
    return moe_unpermute_no_compile(
        inp=restored,
        row_id_map=row_id_map,
        merging_probs=merging_probs,
        restore_shape=restore_shape,
        map_type="index",
    )


def _build_block(backend: str = "te_fused"):
    from olmo_core.config import DType
    from olmo_core.nn.attention import AttentionConfig, AttentionType
    from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
    from olmo_core.nn.moe import MoERouterGatingFunction
    from olmo_core.nn.moe.v2.block import MoEFusedV2TransformerBlock
    from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
    from olmo_core.nn.moe.v2.router import MoERouterConfigV2

    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=DType.float32,
    )
    return MoEFusedV2TransformerBlock(
        d_model=16,
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
            d_model=16,
            num_experts=4,
            top_k=2,
            gating_function=MoERouterGatingFunction.softmax,
            uniform_expert_assignment=True,
            lb_loss_weight=None,
            z_loss_weight=None,
            dtype=DType.float32,
        ),
        shared_experts_router=None,
        shared_experts=None,
        routed_experts=RoutedExpertsConfig(
            d_model=16,
            hidden_size=32,
            num_experts=4,
            bias=False,
            dtype=DType.float32,
        ),
        feed_forward_norm=layer_norm,
        ep_no_sync=True,
        ep_no_sync_restore_unpermute_backend=backend,
        init_device="cuda",
    )


@requires_gpu
@requires_te
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k", [1, 2, 4])
@pytest.mark.parametrize("keep_fraction", [1.0, 0.7, 0.3])
def test_restore_unpermute_1d_te_fused_matches_legacy(
    dtype: torch.dtype,
    top_k: int,
    keep_fraction: float,
):
    torch.manual_seed(7)
    device = torch.device("cuda")
    num_tokens = 128
    d_model = 256
    num_experts = 16

    x = torch.randn(num_tokens, d_model, device=device, dtype=dtype)
    routing_map = torch.randint(
        low=0,
        high=num_experts,
        size=(num_tokens, top_k),
        device=device,
        dtype=torch.int32,
    )
    permuted, row_id_map = moe_permute_no_compile(
        inp=x,
        routing_map=routing_map,
        num_out_tokens=num_tokens * top_k,
        map_type="index",
    )
    restore_shape = x.shape

    reorder_indices, local_inverse_reorder_indices, packed_keep_mask = _build_keep_reorder(
        num_rows=permuted.shape[0],
        keep_fraction=keep_fraction,
        seed=11,
        device=device,
    )
    combine_out = permuted.index_select(0, reorder_indices)
    merging_probs = torch.rand(num_tokens, top_k, device=device, dtype=torch.float32)

    combine_out_legacy = combine_out.detach().clone().requires_grad_(True)
    combine_out_fused = combine_out.detach().clone().requires_grad_(True)
    probs_legacy = merging_probs.detach().clone().requires_grad_(True)
    probs_fused = merging_probs.detach().clone().requires_grad_(True)

    out_legacy = _legacy_restore_drop_unpermute(
        combine_out=combine_out_legacy,
        row_id_map=row_id_map,
        local_inverse_reorder_indices=local_inverse_reorder_indices,
        packed_keep_mask=packed_keep_mask,
        merging_probs=probs_legacy,
        restore_shape=restore_shape,
    )
    out_fused = moe_unpermute_1d_fused_drop_no_compile(
        inp=combine_out_fused,
        row_id_map=row_id_map,
        local_inverse_reorder_indices=local_inverse_reorder_indices,
        packed_keep_mask=packed_keep_mask,
        merging_probs=probs_fused,
        map_type="index",
    )

    torch.testing.assert_close(out_fused, out_legacy, atol=5e-3, rtol=5e-3)

    grad_out = torch.randn_like(out_legacy)
    (out_legacy * grad_out).sum().backward()
    (out_fused * grad_out).sum().backward()

    assert combine_out_legacy.grad is not None
    assert combine_out_fused.grad is not None
    assert probs_legacy.grad is not None
    assert probs_fused.grad is not None
    torch.testing.assert_close(combine_out_fused.grad, combine_out_legacy.grad, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(probs_fused.grad, probs_legacy.grad, atol=5e-3, rtol=5e-3)


@requires_gpu
@requires_te
def test_restore_unpermute_backend_selector_behavior():
    try:
        block = _build_block()
    except ImportError as exc:
        pytest.skip(f"Skipping selector behavior test due import error: {exc}")
    assert block.ep_no_sync_restore_unpermute_backend == "te_fused"

    torch.manual_seed(23)
    device = torch.device("cuda")
    num_tokens = 64
    top_k = 2
    d_model = 256
    num_experts = 8

    x = torch.randn(num_tokens, d_model, device=device, dtype=torch.float16)
    routing_map = torch.randint(
        low=0,
        high=num_experts,
        size=(num_tokens, top_k),
        device=device,
        dtype=torch.int32,
    )
    permuted, row_id_map = moe_permute_no_compile(
        inp=x,
        routing_map=routing_map,
        num_out_tokens=num_tokens * top_k,
        map_type="index",
    )
    reorder_indices, local_inverse_reorder_indices, packed_keep_mask = _build_keep_reorder(
        num_rows=permuted.shape[0],
        keep_fraction=0.7,
        seed=5,
        device=device,
    )
    combine_out = permuted.index_select(0, reorder_indices)
    probs = torch.rand(num_tokens, top_k, device=device, dtype=torch.float32)
    num_kept = packed_keep_mask.to(dtype=torch.long).sum(dtype=torch.long)

    block.ep_no_sync_restore_unpermute_backend = "te_legacy"
    out_legacy = block._restore_drop_unpermute_1d(
        combine_out=combine_out,
        local_inverse_reorder_indices=local_inverse_reorder_indices,
        packed_keep_mask=packed_keep_mask,
        num_kept=num_kept,
        reversed_local_x_permutation_mapping=row_id_map,
        local_x_global_routed_expert_weights=probs,
        hidden_shape_before_permute=torch.Size([num_tokens, d_model]),
    )
    block.ep_no_sync_restore_unpermute_backend = "te_fused"
    out_fused = block._restore_drop_unpermute_1d(
        combine_out=combine_out,
        local_inverse_reorder_indices=local_inverse_reorder_indices,
        packed_keep_mask=packed_keep_mask,
        num_kept=num_kept,
        reversed_local_x_permutation_mapping=row_id_map,
        local_x_global_routed_expert_weights=probs,
        hidden_shape_before_permute=torch.Size([num_tokens, d_model]),
    )
    torch.testing.assert_close(out_fused, out_legacy, atol=5e-3, rtol=5e-3)

    block.ep_no_sync_restore_unpermute_backend = "cuda"
    with pytest.raises(RuntimeError, match="not implemented yet"):
        _ = block._restore_drop_unpermute_1d(
            combine_out=combine_out,
            local_inverse_reorder_indices=local_inverse_reorder_indices,
            packed_keep_mask=packed_keep_mask,
            num_kept=num_kept,
            reversed_local_x_permutation_mapping=row_id_map,
            local_x_global_routed_expert_weights=probs,
            hidden_shape_before_permute=torch.Size([num_tokens, d_model]),
        )
