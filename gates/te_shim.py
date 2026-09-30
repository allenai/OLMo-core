"""Torch stand-ins for the TransformerEngine permutation ops used by the OLMoDDP no-EP path.

Test-harness only; never repo code. TransformerEngine is not installable on the dev box, and the
no-EP MoE path calls ``moe_permute`` / ``moe_unpermute`` through ``olmo_core.nn.moe.utils``. The
shims reproduce the documented semantics (group token copies by expert, merge back with the
routing weights, fp32 accumulation) with autograd support. Both trees under comparison receive
the same shim, so tree-vs-tree comparisons stay bitwise; absolute numbers are validated against
the repo's naive reference test (``validate()`` below).
"""

from __future__ import annotations

import torch


def moe_permute(inp, routing_map, num_out_tokens=-1, map_type="index", **_):
    assert map_type == "index"
    tokens, top_k = routing_map.shape
    flat_experts = routing_map.reshape(-1).long()  # flat index f = token * top_k + slot
    order = torch.argsort(flat_experts, stable=True)  # permuted row r comes from flat index order[r]
    if num_out_tokens is not None and num_out_tokens >= 0:
        order = order[:num_out_tokens]
    permuted = inp.index_select(0, order // top_k)
    return permuted, order


def moe_unpermute(inp, row_id_map, restore_shape=None, map_type="index", merging_probs=None, **_):
    assert map_type == "index"
    order = row_id_map.long()
    tokens = restore_shape[0]
    top_k = merging_probs.shape[1] if merging_probs is not None else order.numel() // tokens
    # Deterministic: gather each token's top_k rows in slot order and reduce in fixed order.
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)  # flat index -> permuted row
    rows = inp.float()[inverse.view(tokens, top_k)]  # [tokens, top_k, hidden]
    if merging_probs is not None:
        rows = rows * merging_probs.float().unsqueeze(-1)
    out = rows[:, 0]
    for k in range(1, top_k):
        out = out + rows[:, k]
    return out.to(inp.dtype)


def moe_sort_chunks_by_index(inp, split_sizes, sorted_idx, **_):
    chunks = list(torch.split(inp, split_sizes.tolist() if torch.is_tensor(split_sizes) else list(split_sizes), dim=0))
    return torch.cat([chunks[i] for i in sorted_idx.tolist()], dim=0)


def install():
    """Point olmo-core's permutation entry points at the shims (only if TE is absent)."""
    import olmo_core.nn.moe.utils as u

    if u.moe_permute is None:
        u.moe_permute = moe_permute
        u.moe_unpermute = moe_unpermute
        u.moe_sort_chunks_by_index = moe_sort_chunks_by_index
        return True
    return False


def validate():
    """Re-run the repo's naive-reference check for the no-EP routed core with the shims."""
    from olmo_core.config import DType
    from olmo_core.nn.moe.utils import moe_permute_no_compile, moe_unpermute_no_compile
    from olmo_core.nn.moe.v2.routed_experts import RoutedExperts

    torch.manual_seed(17)
    device = torch.device("cuda")
    module = RoutedExperts(d_model=512, hidden_size=1024, num_experts=4, bias=False, dtype=DType.float32, init_device="cuda")
    module.eval()
    with torch.no_grad():
        module.w_up_gate.normal_(mean=0.0, std=0.02)
        module.w_down.normal_(mean=0.0, std=0.02)
    x = torch.randn(9, module.d_model, device=device, dtype=torch.float32, requires_grad=True)
    idx = torch.tensor([[0, 1], [1, 3], [2, 0], [3, 2], [0, 3], [2, 1], [1, 0], [3, 0], [2, 3]], device=device)
    w = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.8, 0.2], [0.4, 0.6], [0.9, 0.1], [0.25, 0.75], [0.55, 0.45], [0.35, 0.65]], device=device)
    counts = torch.bincount(idx.reshape(-1), minlength=4)
    permuted, reverse = moe_permute_no_compile(inp=x, routing_map=idx.int(), num_out_tokens=idx.numel(), map_type="index")
    from test.nn.moe.routed_experts_v2_test import _batch_sizes_for_runtime

    out_p = module(permuted, _batch_sizes_for_runtime(counts.tolist(), device))
    actual = moe_unpermute_no_compile(inp=out_p, row_id_map=reverse, restore_shape=x.shape, map_type="index", merging_probs=w)
    from test.nn.moe.routed_experts_v2_test import _naive_routed_mlp_reference  # repo test helper

    expected = _naive_routed_mlp_reference(x, idx, w, module.w_up_gate, module.w_down)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    actual.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    return float((actual - expected).abs().max())
