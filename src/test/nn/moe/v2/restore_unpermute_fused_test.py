import torch

from olmo_core.nn.moe.utils import moe_unpermute_1d_fused_drop_no_compile
from olmo_core.testing import requires_gpu, requires_te


def _reference_unpermute(
    *,
    input_fwd: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
) -> torch.Tensor:
    """Differentiable pure-torch unpermute-and-combine matching the TE row-id layout."""
    num_tokens, top_k = probs.shape
    out = input_fwd.new_zeros((num_tokens, input_fwd.shape[-1]))
    for token_idx in range(num_tokens):
        for topk_idx in range(top_k):
            row_idx = int(row_id_map[topk_idx * num_tokens + token_idx].item())
            if row_idx >= 0:
                out[token_idx] = out[token_idx] + input_fwd[row_idx] * probs[
                    token_idx, topk_idx
                ].to(dtype=input_fwd.dtype)
    return out


@requires_gpu
@requires_te
def test_restore_unpermute_1d_packed_backward_buffer_does_not_corrupt_prob_grads():
    torch.manual_seed(31)
    device = torch.device("cuda")
    num_tokens = 8
    top_k = 2
    d_model = 512
    num_rows = num_tokens * top_k

    row_id_map = torch.arange(num_rows, device=device, dtype=torch.int32)
    local_inverse_reorder_indices = torch.arange(num_rows, device=device, dtype=torch.long)
    packed_keep_mask = torch.ones(num_rows, device=device, dtype=torch.bool)
    probs = torch.rand(num_tokens, top_k, device=device, dtype=torch.float32)
    grad_out = torch.randn(num_tokens, d_model, device=device, dtype=torch.float32)

    inp_ref = torch.randn(num_rows, d_model, device=device, dtype=torch.float16, requires_grad=True)
    probs_ref = probs.detach().clone().requires_grad_(True)
    out_ref = moe_unpermute_1d_fused_drop_no_compile(
        inp=inp_ref,
        row_id_map=row_id_map,
        local_inverse_reorder_indices=local_inverse_reorder_indices,
        packed_keep_mask=packed_keep_mask,
        merging_probs=probs_ref,
        num_kept=torch.tensor(num_rows, device=device),
        row_id_map_is_packed=True,
        backward_grad_input_buffer=None,
        map_type="index",
    )
    (out_ref * grad_out.to(out_ref.dtype)).sum().backward()

    inp_alias = inp_ref.detach().clone().requires_grad_(True)
    probs_alias = probs.detach().clone().requires_grad_(True)
    out_alias = moe_unpermute_1d_fused_drop_no_compile(
        inp=inp_alias,
        row_id_map=row_id_map,
        local_inverse_reorder_indices=local_inverse_reorder_indices,
        packed_keep_mask=packed_keep_mask,
        merging_probs=probs_alias,
        num_kept=torch.tensor(num_rows, device=device),
        row_id_map_is_packed=True,
        backward_grad_input_buffer=inp_alias.detach(),
        map_type="index",
    )
    (out_alias * grad_out.to(out_alias.dtype)).sum().backward()

    assert inp_ref.grad is not None
    assert inp_alias.grad is not None
    assert probs_ref.grad is not None
    assert probs_alias.grad is not None
    torch.testing.assert_close(inp_alias.grad, inp_ref.grad, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(probs_alias.grad, probs_ref.grad, atol=5e-3, rtol=5e-3)


@requires_gpu
@requires_te
def test_restore_unpermute_1d_prob_grads_with_frozen_experts():
    """
    Router-probability gradients must still flow when the expert outputs (input 0)
    are frozen but ``merging_probs`` (input 2) requires grad. Regression test for the
    unpermute backward gating prob-gradient computation on ``needs_input_grad[0]``.
    """
    torch.manual_seed(37)
    device = torch.device("cuda")
    num_tokens = 8
    top_k = 2
    d_model = 512
    num_rows = num_tokens * top_k

    row_id_map = torch.arange(num_rows, device=device, dtype=torch.int32)
    local_inverse_reorder_indices = torch.arange(num_rows, device=device, dtype=torch.long)
    packed_keep_mask = torch.ones(num_rows, device=device, dtype=torch.bool)
    grad_out = torch.randn(num_tokens, d_model, device=device, dtype=torch.float32)

    # Expert outputs are frozen; only the router probabilities are trainable.
    inp = torch.randn(num_rows, d_model, device=device, dtype=torch.float32)
    probs = torch.rand(num_tokens, top_k, device=device, dtype=torch.float32, requires_grad=True)

    out = moe_unpermute_1d_fused_drop_no_compile(
        inp=inp,
        row_id_map=row_id_map,
        local_inverse_reorder_indices=local_inverse_reorder_indices,
        packed_keep_mask=packed_keep_mask,
        merging_probs=probs,
        num_kept=torch.tensor(num_rows, device=device),
        row_id_map_is_packed=True,
        map_type="index",
    )
    (out * grad_out).sum().backward()

    probs_ref = probs.detach().clone().requires_grad_(True)
    out_ref = _reference_unpermute(input_fwd=inp, row_id_map=row_id_map, probs=probs_ref)
    (out_ref * grad_out).sum().backward()

    assert probs.grad is not None
    torch.testing.assert_close(probs.grad, probs_ref.grad, atol=5e-3, rtol=5e-3)
