import torch

from olmo_core.kernels.moe_unpermute_bwd import moe_unpermute_bwd
from olmo_core.testing import requires_gpu


def _reference_unpermute(
    *,
    input_fwd: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
) -> torch.Tensor:
    num_tokens, top_k = probs.shape
    out = input_fwd.new_zeros((num_tokens, input_fwd.shape[-1]))
    for token_idx in range(num_tokens):
        for topk_idx in range(top_k):
            row_idx = int(row_id_map[topk_idx * num_tokens + token_idx].item())
            if row_idx >= 0:
                out[token_idx] = out[token_idx] + input_fwd[row_idx] * probs[token_idx, topk_idx].to(
                    dtype=input_fwd.dtype
                )
    return out


@requires_gpu
def test_moe_unpermute_bwd_matches_autograd_reference_with_out_buffer():
    torch.manual_seed(43)
    device = torch.device("cuda")
    num_tokens = 8
    top_k = 2
    d_model = 512
    num_rows = num_tokens * top_k

    route_rows = torch.arange(num_rows, device=device, dtype=torch.int32).reshape(top_k, num_tokens).T
    route_rows[3, 1] = -1
    dropped_row = 11
    row_id_map = route_rows.T.contiguous().reshape(-1)
    keep_mask = torch.ones(num_rows, device=device, dtype=torch.bool)
    keep_mask[dropped_row] = False

    input_fwd = torch.randn(num_rows, d_model, device=device, dtype=torch.float32)
    probs = torch.rand(num_tokens, top_k, device=device, dtype=torch.float32)
    grad_output = torch.randn(num_tokens, d_model, device=device, dtype=torch.float32)

    input_ref = input_fwd.detach().clone().requires_grad_(True)
    probs_ref = probs.detach().clone().requires_grad_(True)
    out_ref = _reference_unpermute(
        input_fwd=input_ref,
        row_id_map=row_id_map,
        probs=probs_ref,
    )
    grad_input_ref, grad_probs_ref = torch.autograd.grad(out_ref, (input_ref, probs_ref), grad_outputs=grad_output)

    out_buffer = torch.empty_like(input_fwd)
    grad_input, grad_probs = moe_unpermute_bwd(
        grad_output=grad_output,
        input_fwd=input_fwd,
        row_id_map=row_id_map,
        probs=probs,
        keep_mask=keep_mask,
        out=out_buffer,
    )

    assert grad_input.untyped_storage().data_ptr() == out_buffer.untyped_storage().data_ptr()
    torch.testing.assert_close(grad_input, grad_input_ref)
    torch.testing.assert_close(grad_probs, grad_probs_ref)
