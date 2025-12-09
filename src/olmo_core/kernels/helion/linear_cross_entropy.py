from typing import Literal, Tuple

import helion
import helion.language as hl
import torch
from helion._testing import DEVICE, run_example


def lce_baseline_fn(_input, weight, labels, ignore_index=-100, reduction="sum"):
    logits = torch.matmul(_input, weight).float()
    lse = logits.logsumexp(dim=-1)

    # Compute per-sample loss (no reduction for baseline comparison)
    loss = torch.nn.functional.cross_entropy(
        logits, labels, ignore_index=ignore_index, reduction=reduction
    )
    # Apply the same reduction to lse as to the loss
    mask = labels != ignore_index
    if reduction == "sum":
        lse = (lse * mask).sum()
    return loss, lse


@helion.kernel(
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    # autotune_baseline_fn=lce_baseline_fn,
    # autotune_accuracy_check=True,
    autotune_effort="none",
    autotune_ignore_errors=False,
)
def olmo_linear_cross_entropy_fwd(
    _input: torch.Tensor,  # [N, D] input to final layer
    weight: torch.Tensor,  # [D, V] weight matrix of final layer
    labels: torch.Tensor,  # [N] target labels
    ignore_index: int = -100,
    reduction: Literal["sum", "none"] = "sum",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the fused linear cross entropy loss with gradients.

    This kernel fuses the final linear layer computation (input @ weight) with the
    cross entropy loss computation, avoiding materialization of the full [N, V] logits tensor.
    The function computes logits on-the-fly and calculates the negative log likelihood
    of the true labels, along with gradients for backward pass.

    Args:
        _input: Input tensor of shape [N, D] where N is batch size and D is hidden dimension
        weight: Weight matrix of shape [D, V] where V is vocabulary size
        labels: Target labels tensor of shape [N] containing class indices
        ignore_index: Target value that is ignored and does not contribute to loss
        reduction: Specifies the reduction to apply to the output ("sum", or "none")

    Returns:
        A tuple of (loss, lse, grad_input, grad_weight) where:
        - loss: The cross entropy loss, tensor of shape [N]
        - lse: The log-sum-exp values (can be used to compute z_loss externally), shape [N]
        - grad_input: Gradient w.r.t. input, shape [N, D]
        - grad_weight: Gradient w.r.t. weight, shape [D, V]
    """
    n, d = _input.shape
    d_check, v = weight.shape
    assert d == d_check, f"Input dimension mismatch: {d} != {d_check}"

    # Accumulate losses in fp32 for numerical stability, even if inputs are bfloat16
    losses = torch.zeros([n], dtype=torch.float32, device=_input.device)
    lse = torch.zeros([n], dtype=torch.float32, device=_input.device)
    grad_input = torch.zeros([n, d], dtype=torch.float32, device=_input.device)
    grad_weight = torch.zeros([d, v], dtype=torch.float32, device=_input.device)

    # --- Device Code (compiles to triton) ---
    # Outer tile loop is mapped to the launch grid, this is what gets parallelized
    for tile_n in hl.tile(n):  # Tile over the batch dimension
        labels_tile = labels[tile_n]  # [t_n]
        valid_mask = labels_tile != ignore_index  # [t_n]

        # 1) compute logits for each element in the batch tile, tracking max and sum_exp for stable log-sum-exp
        max_logits = hl.zeros([tile_n], dtype=torch.float32)  # [t_n] - running max
        sum_exp = hl.zeros([tile_n], dtype=torch.float32)  # [t_n] - running sum of exp
        target_logits_tile = hl.zeros([tile_n], dtype=torch.float32)  # [t_n] - logits at target

        # 1) Combine matmul and online log-sum-exp computation in one loop
        for tile_v in hl.tile(v):  # inner tile loops are sequential
            # 1a) compute logits for this v tile
            logits_acc = hl.zeros([tile_n, tile_v], dtype=torch.float32)
            for tile_k in hl.tile(d):
                logits_acc = torch.addmm(logits_acc, _input[tile_n, tile_k], weight[tile_k, tile_v])

            # 1b) Update running max logit and shifted sum_exp
            new_max = torch.maximum(max_logits, logits_acc.amax(dim=-1))  # [t_n]
            sum_exp = sum_exp * torch.exp(max_logits - new_max) + torch.exp(
                logits_acc - new_max.unsqueeze(1)
            ).sum(dim=-1)
            max_logits = new_max

            # Gather logits at target indices that fall within this tile
            v_start = tile_v.begin  # Starting vocab index of this tile
            v_end = tile_v.end  # Ending vocab index of this tile (exclusive)
            in_tile_mask = (labels_tile >= v_start) & (labels_tile < v_end)  # [t_n]
            local_idx = labels_tile - v_start  # [t_n]

            # Gather values and apply mask to only accumulate for labels actually in this tile
            # gathered_logits = torch.gather(
            #     logits_acc, dim=1, index=local_idx.clamp(0, logits_acc.shape[1] - 1).to(torch.int64).unsqueeze(-1)
            # ).squeeze(-1)
            gathered_logits = hl.inline_triton(
                "tl.sum(tl.gather({0}, {1}.to(tl.int32)[:, None], axis=1), axis=1)",
                args=(logits_acc, local_idx),
                output_like=local_idx.to(torch.float32),
            )

            target_logits_tile = target_logits_tile + (
                gathered_logits * in_tile_mask.to(torch.float32)
            )

        # finalize lse for this tile
        lse_tile = max_logits + torch.log(sum_exp)

        # 4) Compute cross entropy loss: log_sum_exp - logit_at_target
        losses_tile = lse_tile - target_logits_tile  # [t_n]

        # Handle ignore_index: set loss to 0 for ignored labels
        losses[tile_n] = losses_tile * valid_mask.to(torch.float32)  # [t_n]
        lse[tile_n] = lse_tile * valid_mask.to(torch.float32)  # [t_n]

        # # 5) Compute grad_input = d_logits @ weight.T
        # # 6) Accumulate grad_weight += _input.T @ d_logits
        # for tile_d in hl.tile(d):
        #     gi_acc = hl.zeros([tile_n, tile_d], dtype=torch.float32)
        #     for tile_v in hl.tile(v):
        #         # Recompute d_logits for this tile_v using stored logits_tile
        #         softmax_v = torch.exp(
        #             logits_tile[:, tile_v] - max_logits.unsqueeze(-1)
        #         ) / sum_exp.unsqueeze(-1)

        #         # d_logits = softmax - 1 at target index, softmax elsewhere
        #         # Use scatter to subtract 1 at the target positions
        #         d_logits_v = softmax_v.clone()
        #         # Scatter -1 at the target label positions within this tile
        #         hl.inline_triton(
        #             """
        #             # Subtract 1 from positions where labels match the current tile indices
        #             tl.scatter({0}, {1}, tl.full({1}.shape, -1.0, dtype=tl.float32), axis=1)
        #             """,
        #             args=(d_logits_v, labels_tile.unsqueeze(1)),
        #         )

        #         d_logits_v = d_logits_v.to(torch.bfloat16) * valid_mask.unsqueeze(-1)

        #         gi_acc = torch.addmm(gi_acc, d_logits_v, weight[tile_d, tile_v].T)
        #         grad_weight[tile_d, tile_v] = torch.addmm(
        #             grad_weight[tile_d, tile_v],
        #             _input[tile_n, tile_d].T,
        #             d_logits_v,
        #         )

        #     grad_input[tile_n, tile_d] = gi_acc

    if reduction == "sum":
        losses = losses.sum()  # [N] -> scalar
        lse = lse.sum()  # [N] -> scalar

    return losses, lse, grad_input, grad_weight


@torch.compile
def olmo_linear_cross_entropy_bwd(grad_loss, grad_input, grad_weight):
    # Note: be wary of in-place operations in torch
    torch.mul(grad_loss, grad_input, out=grad_input)
    torch.mul(grad_loss, grad_weight, out=grad_weight)
    return grad_input, grad_weight


class OlmoFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        _input,
        weight,
        target,
        ignore_index=-100,
        reduction="sum",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loss, lse, grad_input, grad_weight = olmo_linear_cross_entropy_fwd(
            _input, weight, target, ignore_index, reduction
        )
        # Save gradients for backward pass
        ctx.save_for_backward(grad_input, grad_weight)
        ctx.reduction = reduction
        ctx.num_valid = (target != ignore_index).sum().float()
        return loss, lse

    @staticmethod
    def backward(ctx, grad_loss, grad_lse):
        del grad_lse
        grad_input, grad_weight = ctx.saved_tensors
        grad_input, grad_weight = olmo_linear_cross_entropy_bwd(grad_loss, grad_input, grad_weight)

        return (
            grad_input,  # input
            grad_weight,  # weight
            None,  # target
            None,  # ignore_index
            None,  # reduction
        )


class TorchLMHeadCE(torch.nn.Module):
    def __init__(
        self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100, reduction: str = "sum"
    ):
        super().__init__()
        self.lm_head = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, x, target):
        logits = self.lm_head(x).to(torch.float32)
        return self.ce_loss(logits, target)


class OlmoLMHeadCE(torch.nn.Module):
    def __init__(
        self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100, reduction: str = "sum"
    ):
        super().__init__()
        self.lm_head = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)

    def forward(self, x, target):
        return OlmoFusedLinearCrossEntropyFunction.apply(
            x,
            self.lm_head.weight,
            target,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


def main() -> None:
    """
    Main entry point that runs the fused linear cross entropy kernel verification.
    """
    batch_size, seq_len, vocab_size = 1, 8192, 100352
    hidden_size = 2560

    # batch_size, seq_len, vocab_size = 1, 8192, 8192 * 2
    # hidden_size = 2560

    n = batch_size * seq_len

    # Create inputs for fused version
    _input = torch.randn(n, hidden_size, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(hidden_size, vocab_size, device=DEVICE, dtype=torch.bfloat16)
    labels = torch.randint(0, vocab_size, (n,), device=DEVICE, dtype=torch.long)

    def helion_fn(_input, weight, labels):
        loss, lse = OlmoFusedLinearCrossEntropyFunction.apply(_input, weight, labels, -100, "sum")
        return loss

    @torch.compile
    def torch_fn(_input, weight, labels):
        loss, lse = lce_baseline_fn(_input, weight, labels, ignore_index=-100, reduction="sum")
        return loss

    # Baseline: compute logits first, then cross entropy
    torch_fn(_input, weight, labels)

    run_example(
        helion_fn,
        torch_fn,
        (_input, weight, labels),
        kernel_name="helion",
        baseline_name="torch",
        rtol=1e-4,
        atol=1e-4,
        bwd=False,
    )


if __name__ == "__main__":
    main()
