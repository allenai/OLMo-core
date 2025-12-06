from typing import Literal, Tuple

import helion
import helion.language as hl
import torch
from helion._testing import DEVICE, run_example


def lce_baseline_fn(_input, weight, labels, ignore_index=-100, reduction="mean"):
    logits = torch.matmul(_input, weight).float()
    lse = logits.logsumexp(dim=-1)

    # Compute per-sample loss (no reduction for baseline comparison)
    loss = torch.nn.functional.cross_entropy(
        logits, labels, ignore_index=ignore_index, reduction="none"
    )
    # Apply the same reduction to lse as to the loss
    mask = labels != ignore_index
    if reduction == "mean":
        lse = (lse * mask).sum() / mask.sum()
    elif reduction == "sum":
        lse = (lse * mask).sum()
    return loss, lse


@helion.kernel(
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    autotune_baseline_fn=lce_baseline_fn,
    autotune_accuracy_check=True,
    autotune_effort="quick",
    autotune_ignore_errors=True,
)
def olmo_linear_cross_entropy_fwd(
    _input: torch.Tensor,  # [N, D] input to final layer
    weight: torch.Tensor,  # [D, V] weight matrix of final layer
    labels: torch.Tensor,  # [N] target labels
    ignore_index: int = -100,
    reduction: Literal["mean", "sum", "none"] = "mean",
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
        reduction: Specifies the reduction to apply to the output ("mean", "sum", or "none")

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

    block_size_n = hl.register_block_size(n)
    logits_tile = torch.zeros(
        [block_size_n, v], dtype=torch.float32, device=_input.device
    )  # [t_n, v]

    # --- Device Code (compiles to triton) ---
    for tile_n in hl.tile(n, block_size=block_size_n):  # Tile over the batch dimension
        labels_tile = labels[tile_n]  # [t_n]

        # 1) compute logits for each element in the batch tile
        for tile_v in hl.tile(v):  # TODO: does it make more sense to transpose weight here?
            acc = hl.zeros([tile_n, tile_v], dtype=torch.float32)
            for tile_k in hl.tile(d):
                acc = torch.addmm(acc, _input[tile_n, tile_k], weight[tile_k, tile_v])
            logits_tile[tile_n, tile_v] = acc  #  not allowed ...

        # 2) Gather logits at target using one-hot mask
        one_hot_mask = labels_tile.unsqueeze(-1) == hl.arange(helion.next_power_of_2(v)).unsqueeze(
            0
        )  # [t_n, v]
        target_logits_tile = (logits_tile[tile_n, :] * one_hot_mask).sum(
            dim=-1
        )  # [t_n, v] -> [t_n]

        # 3) Compute stable log-sum-exp
        max_logits = logits_tile[tile_n, :].amax(dim=-1)  # [t_n]
        shifted_logits = logits_tile[tile_n, :] - max_logits.unsqueeze(-1)  # [t_n, v]
        exp_shifted = torch.exp(shifted_logits)  # [t_n, v]
        sum_exp = exp_shifted.sum(dim=-1)  # [t_n]
        lse[tile_n] = max_logits + torch.log(sum_exp)  # [t_n]

        # 4) Compute cross entropy loss: log_sum_exp - logit_at_target
        # This is equivalent to -log(softmax[target]) = -log(exp(logit[target]) / sum(exp(logits)))
        # = log_sum_exp - logit_at_target
        losses_tile = lse[tile_n] - target_logits_tile  # [t_n]

        # Handle ignore_index: set loss to 0 for ignored labels
        valid_mask = labels_tile != ignore_index  # [t_n]
        losses[tile_n] = losses_tile * valid_mask.to(torch.float32)  # [t_n]

        # 5) Compute grad_input = d_logits @ weight.T = [t_n, v] @ [v, d] = [t_n, d]
        # d_logits = softmax - one_hot(labels), computed per tile_v below
        # 6) Accumulate grad_weight += _input.T @ d_logits = [d, t_n] @ [t_n, v] = [d, v]
        for tile_d in hl.tile(d):
            gi_acc = hl.zeros([tile_n, tile_d], dtype=torch.float32)
            for tile_v in hl.tile(v):
                # Recompute d_logits for this tile_v
                softmax_v = torch.exp(
                    logits_tile[tile_n, tile_v] - max_logits.unsqueeze(-1)
                ) / sum_exp.unsqueeze(-1)
                one_hot_v = labels_tile.unsqueeze(-1) == hl.arange(tile_v.block_size).unsqueeze(0)
                d_logits_v = (softmax_v - one_hot_v.to(torch.float32)).to(
                    torch.bfloat16
                ) * valid_mask.unsqueeze(-1)
                gi_acc = torch.addmm(gi_acc, d_logits_v, weight[tile_d, tile_v].T)
                grad_weight[tile_d, tile_v] = torch.addmm(
                    grad_weight[tile_d, tile_v],
                    _input[tile_n, tile_d].T,
                    d_logits_v,
                )

    return losses, lse, grad_input, grad_weight

    # mask = (labels != ignore_index).to(torch.float32)
    # if reduction == "mean":
    #     num_valid = mask.sum()
    #     if num_valid > 0:
    #         reduced_losses = (losses * mask).sum() / num_valid
    #         reduced_lse = (lse * mask).sum() / num_valid
    #     else:
    #         reduced_losses = torch.tensor(0.0, dtype=torch.float32, device=losses.device)
    #         reduced_lse = torch.tensor(0.0, dtype=torch.float32, device=lse.device)
    #     return reduced_losses, reduced_lse
    # elif reduction == "sum":
    #     return (losses * mask).sum(), (lse * mask).sum()
    # else:  # reduction == "none"


@helion.kernel(
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def olmo_linear_cross_entropy_bwd(
    _input,
    weight,
    target,
    lse,
    ignore_index: int = -100,
    reduction: Literal["mean", "sum", "none"] = "mean",
):
    raise NotImplementedError("Not implemented")


class OlmoFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        _input,
        weight,
        target,
        ignore_index=-100,
        reduction="mean",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loss, lse, grad_input, grad_weight = olmo_linear_cross_entropy_fwd(
            _input,
            weight,
            target,
            ignore_index,
            reduction,
        )
        # Save gradients for backward pass
        ctx.save_for_backward(grad_input, grad_weight)
        ctx.reduction = reduction
        ctx.num_valid = (target != ignore_index).sum().float()
        return loss, lse

    @staticmethod
    def backward(ctx, grad_loss, grad_lse):
        x, target, lse = ctx.saved_tensors
        grad_input, grad_weight = ctx.saved_tensors
        # Scale gradients by upstream gradient
        # For mean reduction, we need to divide by number of valid samples
        if ctx.reduction == "mean" and ctx.num_valid > 0:
            scale = grad_loss / ctx.num_valid
        elif ctx.reduction == "sum":
            scale = grad_loss
        else:
            scale = grad_loss.unsqueeze(-1)  # [N, 1] for broadcasting
        return grad_input * scale, grad_weight * scale, None, None, None


class TorchLMHeadCE(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.lm_head = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, x, target):
        logits = self.lm_head(x).to(torch.float32)
        return self.ce_loss(logits, target)


class OlmoLMHeadCE(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.lm_head = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)

    def forward(self, x, target):
        return OlmoFusedLinearCrossEntropyFunction.apply(
            x,
            self.lm_head.weight,
            target,
            ignore_index=self.ignore_index,
            reduction=self.reductionm,
        )


def main() -> None:
    """
    Main entry point that runs the fused linear cross entropy kernel verification.
    """
    batch_size, seq_len, vocab_size = 1, 8192, 100352
    hidden_size = 2560
    n = batch_size * seq_len

    # Create inputs for fused version
    _input = torch.randn(n, hidden_size, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(hidden_size, vocab_size, device=DEVICE, dtype=torch.bfloat16)
    labels = torch.randint(0, vocab_size, (n,), device=DEVICE, dtype=torch.long)

    def helion_fn(_input, weight, labels):
        loss, lse = OlmoFusedLinearCrossEntropyFunction.apply(_input, weight, labels, -100, "mean")
        return loss.sum()  # Return scalar for run_example comparison

    @torch.compile
    def torch_fn(_input, weight, labels):
        loss, lse = lce_baseline_fn(_input, weight, labels, ignore_index=-100, reduction="mean")
        return loss.sum()  # Return scalar for run_example comparison

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
    )


if __name__ == "__main__":
    main()
