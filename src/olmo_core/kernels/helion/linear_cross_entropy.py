from typing import Literal, Tuple

import helion
import helion.language as hl
import torch
from helion._testing import DEVICE, run_example


def lce_baseline_fn(_input, weight, labels, ignore_index=-100, reduction="mean"):
    logits = torch.matmul(_input, weight).float()
    lse = logits.logsumexp(dim=-1)
    loss = torch.nn.functional.cross_entropy(
        logits, labels, ignore_index=ignore_index, reduction=reduction
    )
    # Apply the same reduction to lse as to the loss
    mask = labels != ignore_index
    if reduction == "mean":
        lse = (lse * mask).sum() / mask.sum()
    elif reduction == "sum":
        lse = (lse * mask).sum()
    return loss, lse


@helion.kernel(
    config=helion.Config(
        block_sizes=[1, 32, 64, 32, 64],
        indexing=["block_ptr", "pointer", "pointer", "pointer", "pointer", "pointer", "pointer"],
        load_eviction_policies=["", "first", "first", "last", ""],
        num_stages=6,
        num_warps=1,
        pid_type="flat",
        range_flattens=[None, False, False, True, None],
        range_multi_buffers=[None, None, False, False, False],
        range_num_stages=[0, 0, 4, 4, 1],
        range_unroll_factors=[0, 2, 1, 4, 1],
        range_warp_specializes=[],
    ),
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the fused linear cross entropy loss.

    This kernel fuses the final linear layer computation (input @ weight) with the
    cross entropy loss computation, avoiding materialization of the full [N, V] logits tensor.
    The function computes logits on-the-fly and calculates the negative log likelihood
    of the true labels.

    Args:
        _input: Input tensor of shape [N, D] where N is batch size and D is hidden dimension
        weight: Weight matrix of shape [D, V] where V is vocabulary size
        labels: Target labels tensor of shape [N] containing class indices
        ignore_index: Target value that is ignored and does not contribute to loss
        reduction: Specifies the reduction to apply to the output ("mean", "sum", or "none")

    Returns:
        A tuple of (loss, lse) where:
        - loss: The cross entropy loss, scalar if reduction is "mean" or "sum",
          tensor of shape [N] if "none"
        - lse: The log-sum-exp values (can be used to compute z_loss externally),
          same shape as loss
    """
    n, d = _input.shape
    d_check, v = weight.shape
    assert d == d_check, f"Input dimension mismatch: {d} != {d_check}"

    # Accumulate losses in fp32 for numerical stability, even if inputs are bfloat16
    losses = torch.zeros([n], dtype=torch.float32, device=_input.device)
    lse = torch.zeros([n], dtype=torch.float32, device=_input.device)

    # --- Device Code (compiles to triton) ---
    for tile_n in hl.tile(n):  # Tile over the batch dimension
        labels_tile = labels[tile_n]  # [tile_size_n]

        # init accumulators
        max_logits_acc = hl.full([tile_n], value=-float("inf"), dtype=torch.float32)
        sum_exp_acc = hl.zeros([tile_n], dtype=torch.float32)
        logits_at_target_acc = hl.zeros([tile_n], dtype=torch.float32)

        # First pass: find max logit for each element in the batch tile
        # this runs a tiled matmul and computes the logits for each element in the batch tile
        # max_logits_acc is used for numerical stability in the second pass
        for tile_v in hl.tile(v):  # TODO: does it make more sense to transpose weight here?
            logits_acc = hl.zeros([tile_n, tile_v], dtype=torch.float32)  # [bs, vocab_size]
            for tile_k in hl.tile(d):
                logits_acc = torch.addmm(logits_acc, _input[tile_n, tile_k], weight[tile_k, tile_v])

            # Update running max
            max_logits_acc = torch.maximum(max_logits_acc, logits_acc.amax(dim=-1))  # [bs]

        # Second pass: compute sum_exp and extract target logits
        for tile_v in hl.tile(v):
            # Recompute logits for this batch-vocab tile (TODO: dont double the matmul here?)
            logits_acc = hl.zeros([tile_n, tile_v], dtype=torch.float32)
            for tile_k in hl.tile(d):
                logits_acc = torch.addmm(logits_acc, _input[tile_n, tile_k], weight[tile_k, tile_v])

            # Gather logits at target
            mask = labels_tile.unsqueeze(-1) == tile_v.index.unsqueeze(0)  # [bs, tile_size_v]
            logits_at_target_acc += (logits_acc * mask).sum(dim=-1)

            # Compute exp(logit - max) for numerical stability
            exp_shifted = torch.exp(logits_acc - max_logits_acc.unsqueeze(-1))  # [bs, vocab_size]
            sum_exp_acc += exp_shifted.sum(dim=-1)  # accumulate sum

        # Compute log-sum-exp and write to output ONCE
        log_sum_exp = max_logits_acc + torch.log(sum_exp_acc)  # [tile_size_n]
        lse[tile_n] = log_sum_exp

        # Compute cross entropy loss: log_sum_exp - logit_at_target
        # This is equivalent to -log(softmax[target]) = -log(exp(logit[target]) / sum(exp(logits)))
        # = log_sum_exp - logit_at_target
        tile_losses = log_sum_exp - logits_at_target_acc  # [tile_size_n]

        # Handle ignore_index: set loss to 0 for ignored labels
        mask = (labels_tile != ignore_index).to(torch.float32)
        tile_losses = tile_losses * mask

        # Write to output ONCE at the end
        losses[tile_n] = tile_losses

    mask = (labels != ignore_index).to(torch.float32)

    if reduction == "mean":
        num_valid = mask.sum()
        if num_valid > 0:
            reduced_losses = (losses * mask).sum() / num_valid
            reduced_lse = (lse * mask).sum() / num_valid
        else:
            reduced_losses = torch.tensor(0.0, dtype=torch.float32, device=losses.device)
            reduced_lse = torch.tensor(0.0, dtype=torch.float32, device=lse.device)
        return reduced_losses, reduced_lse
    elif reduction == "sum":
        return (losses * mask).sum(), (lse * mask).sum()
    else:  # reduction == "none"
        return losses, lse


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
        bwd_impl="chunk",
    ):
        assert bwd_impl in ["chunk", "cce"]
        loss, lse = olmo_linear_cross_entropy_fwd(
            _input,
            weight,
            target,
            ignore_index,
            reduction,
        )
        ctx.save_for_backward(_input, lse, weight, target)
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.bwd_impl = bwd_impl
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        _input, lse, weight, target = ctx.saved_tensors
        grad_input, grad_weight = olmo_linear_cross_entropy_bwd(
            _input,
            weight,
            target,
            lse,
            ctx.ignore_index,
            ctx.reduction,
        )
        return grad_input * grad_output, grad_weight * grad_output, None, None, None, None, None


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
    batch_size, seq_len, vocab_size = 32, 64, 16384
    hidden_size = 128
    n = batch_size * seq_len

    # Create inputs for fused version
    _input = torch.randn(n, hidden_size, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(hidden_size, vocab_size, device=DEVICE, dtype=torch.bfloat16)
    labels = torch.randint(0, vocab_size, (n,), device=DEVICE, dtype=torch.long)

    # Baseline: compute logits first, then cross entropy
    lce_baseline_fn(_input, weight, labels)

    def helion_fn(_input, weight, labels):
        loss, _ = olmo_linear_cross_entropy_fwd(
            _input, weight, labels, ignore_index=-100, reduction="mean"
        )
        return loss

    def torch_fn(_input, weight, labels):
        loss, _ = lce_baseline_fn(_input, weight, labels, ignore_index=-100, reduction="mean")
        return loss

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
