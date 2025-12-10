import argparse
from typing import Tuple

import helion
import helion.language as hl
import torch
from helion._testing import DEVICE, run_example
from helion.autotuner import LFBOPatternSearch


@helion.kernel(
    config=helion.Config(
        block_sizes=[64, 512, 64],
        indexing=[
            "tensor_descriptor",
            "tensor_descriptor",
            "pointer",
            "tensor_descriptor",
            "pointer",
        ],
        load_eviction_policies=["first", "first", "first"],
        num_stages=1,
        num_warps=8,
        pid_type="flat",
        range_flattens=[None, None, False],
        range_multi_buffers=[None, None, None],
        range_num_stages=[0, 3, 3],
        range_unroll_factors=[0, 0, 0],
        range_warp_specializes=[],
    ),
    autotuner_fn=LFBOPatternSearch,
    static_shapes=True,
    autotune_ignore_errors=False,
    autotune_compile_timeout=20,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def olmo_linear_cross_entropy_fwd_kernel(
    inputs: torch.Tensor,  # [N, D] input to final layer
    weight: torch.Tensor,  # [D, V] weight matrix of final layer
    labels: torch.Tensor,  # [N] target labels
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the fused linear cross entropy loss with partial pre-computation of gradients.

    Args:
        inputs: Input tensor of shape [N, D] where N is batch size and D is hidden dimension
        weight: Weight matrix of shape [D, V] where V is vocabulary size
        labels: Target labels tensor of shape [N] containing class indices
        ignore_index: Target value that is ignored and does not contribute to loss
        compute_input_gradients: Whether to pre-compute the gradient with respect to the input

    Returns:
        A tuple of (loss, lse, grad_input) where:
        - loss: The cross entropy loss, tensor of shape [N]
        - lse: The log-sum-exp values (can be used to compute z_loss externally), shape [N]
        - grad_input: Gradient w.r.t. input, shape [N, D]
    """
    n, d = inputs.shape
    d_check, v = weight.shape
    assert d == d_check, f"Input dimension mismatch: {d} != {d_check}"

    # Accumulate losses in fp32 for numerical stability, even if inputs are bfloat16
    losses = torch.zeros([n], dtype=torch.float32, device=inputs.device)
    lse = torch.zeros([n], dtype=torch.float32, device=inputs.device)

    # --- Device Code (compiles to triton) ---
    # Outer tile loop is mapped to the launch grid, this is what gets parallelized
    for tile_n in hl.tile(n):  # Tile over the batch dimension
        labels_tile = labels[tile_n]  # [t_n]
        valid_mask = (labels_tile != ignore_index).to(torch.float32)  # [t_n]

        # Initialize running max, sum_exp, target_logits
        max_logits = hl.zeros([tile_n], dtype=torch.float32)  # [t_n] - running max
        sum_exp = hl.zeros([tile_n], dtype=torch.float32)  # [t_n] - running sum of exp
        target_logits = hl.zeros([tile_n], dtype=torch.float32)  # logits at target

        # 1) Combine matmul and online log-sum-exp computation in one loop
        for tile_v in hl.tile(v):  # inner tile loops are sequential
            # 1a) compute logits for this v tile
            logits = hl.zeros([tile_n, tile_v], dtype=torch.float32)
            for tile_k in hl.tile(d):
                weight_tile_k = hl.load(weight, index=[tile_k, tile_v])
                logits = torch.addmm(logits, inputs[tile_n, tile_k], weight_tile_k)

            # 1b) Update running max logit and sum_exp (online lse)
            new_max = torch.maximum(max_logits, logits.amax(dim=-1))  # [t_n]
            scale = torch.exp(max_logits - new_max)  # [t_n]
            probs = torch.exp(logits - new_max.unsqueeze(1))  # [t_n, v]
            sum_exp = sum_exp * scale + probs.sum(dim=-1)
            max_logits = new_max

            # 1c) Gather logits at target indices that fall within this tile
            local_idx = labels_tile - tile_v.begin  # [t_n]
            # Gather values and apply mask to only accumulate for labels actually in this tile
            gathered_target_logits = hl.inline_triton(
                "tl.sum(tl.gather({0}, {1}.to(tl.int32)[:, None], axis=1), axis=1)",
                args=(logits, local_idx),
                output_like=local_idx.to(torch.float32),
            )
            in_tile_mask = (labels_tile >= tile_v.begin) & (labels_tile < tile_v.end)  # [t_n]
            target_logits = target_logits + gathered_target_logits * in_tile_mask.to(torch.float32)

        # finalize lse for this tile
        lse_tile = max_logits + torch.log(sum_exp)

        # 4) Compute cross entropy loss: log_sum_exp - logit_at_target
        losses_tile = lse_tile - target_logits  # [t_n]

        # Handle ignore_index: set loss to 0 for ignored labels
        losses[tile_n] = losses_tile * valid_mask  # [t_n]
        lse[tile_n] = lse_tile * valid_mask  # [t_n]

    return losses, lse


@helion.kernel(
    config=helion.Config(
        block_sizes=[128, 256, 128, 64],
        indexing=[
            "pointer",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "pointer",
            "tensor_descriptor",
            "tensor_descriptor",
        ],
        l2_groupings=[8],
        load_eviction_policies=["first", "last", "last", "last", "last"],
        loop_orders=[[0, 1]],
        num_stages=1,
        num_warps=8,
        pid_type="flat",
        range_flattens=[None, True, None],
        range_multi_buffers=[None, True, None],
        range_num_stages=[0, 2, 2],
        range_unroll_factors=[0, 4, 0],
        range_warp_specializes=[],
    ),
    static_shapes=True,
    autotuner_fn=LFBOPatternSearch,
    # autotuner_effort="quick",
    autotune_ignore_errors=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def olmo_linear_cross_entropy_bwd_recompute_kernel(
    _input: torch.Tensor,  # [N, D] input to final layer
    weight: torch.Tensor,  # [D, V] weight matrix of final layer
    labels: torch.Tensor,  # [N] target labels
    lse: torch.Tensor,  # [N] log-sum-exp from forward pass
    grad_loss: torch.Tensor,  # [N] gradient of the loss
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the backward pass for the fused linear cross entropy loss.
    Recomputes the logits to compute the gradient with respect to the weight and input.
    """
    n, d = _input.shape
    d_check, v = weight.shape
    assert d == d_check, f"Input dimension mismatch: {d} != {d_check}"

    grad_weight = torch.zeros([d, v], dtype=torch.float32, device=_input.device)
    grad_input = torch.zeros([n, d], dtype=torch.float32, device=_input.device)

    # Parallelize over batch (N) and vocab (V) to minimize logit recomputation
    # This visits each (tile_n, tile_v) once, computes d_logits, and scatters updates to grad_weight
    for tile_n, tile_v in hl.tile([n, v]):
        labels_tile = labels[tile_n]
        lse_tile = lse[tile_n]
        grad_loss_tile = grad_loss[tile_n]

        # 1. Recompute logits for this tile of N and tile of V
        # We need full inner dimension D to compute logits
        logits = hl.zeros([tile_n, tile_v], dtype=torch.float32)
        for tile_k in hl.tile(d):
            weight_kv = hl.load(weight, index=[tile_k, tile_v], eviction_policy="evict_last")
            logits = torch.addmm(logits, _input[tile_n, tile_k], weight_kv)

        # 2. Compute softmax = exp(logits - lse)
        softmax = torch.exp(logits - lse_tile.unsqueeze(1))

        # 3. Compute d_logits = (softmax - target) * grad_loss
        local_idx = labels_tile - tile_v.begin
        in_tile_mask = (labels_tile >= tile_v.begin) & (labels_tile < tile_v.end)

        # Create one-hot mask for target labels within this tile
        # We subtract 1.0 from softmax probabilities at the target indices
        cols = hl.arange(tile_v.block_size)
        mask = (cols[None, :] == local_idx[:, None]) & in_tile_mask[:, None]
        d_logits = torch.where(mask, softmax - 1.0, softmax)

        # Apply valid mask and scaling according to grad_loss
        valid_mask = labels_tile != ignore_index
        scale = valid_mask * grad_loss_tile
        d_logits = d_logits * scale.unsqueeze(1)  # [t_n, v]
        d_logits = d_logits.to(_input.dtype)

        # 4. Accumulate grad_weight and grad_input using atomics
        for tile_k in hl.tile(d):
            # Compute partial update for grad_weight[tile_k, tile_v]
            update_gw = torch.matmul(_input[tile_n, tile_k].T, d_logits)
            hl.atomic_add(grad_weight, [tile_k, tile_v], update_gw)

            # Compute partial update for grad_input[tile_n, tile_k]
            update_gi = torch.matmul(d_logits, weight[tile_k, tile_v].T)
            hl.atomic_add(grad_input, [tile_n, tile_k], update_gi)

    return grad_input, grad_weight


@torch.compile
def olmo_linear_cross_entropy_bwd_basic(grad_loss, grad_input, grad_weight):
    # Note: be wary of in-place operations in torch
    torch.mul(grad_loss, grad_input, out=grad_input)
    torch.mul(grad_loss, grad_weight, out=grad_weight)
    return grad_input, grad_weight


class OlmoFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        weight,
        target,
        ignore_index=-100,
        reduction="sum",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loss, lse = olmo_linear_cross_entropy_fwd_kernel(
            inputs, weight, target, ignore_index=ignore_index
        )
        ctx.save_for_backward(inputs, weight, target, lse)
        if reduction == "sum":
            loss = loss.sum()
            lse = lse.sum()
        # Save inputs/outputs/grads for backward pass
        ctx.reduction = reduction
        return loss, lse

    @staticmethod
    def backward(ctx, grad_loss, grad_lse):
        del grad_lse  # not used for loss
        inputs, weight, target, lse = ctx.saved_tensors

        # Handle reduction for grad_loss
        if ctx.reduction == "sum":
            grad_loss = grad_loss.expand_as(target)

        grad_input, grad_weight = olmo_linear_cross_entropy_bwd_recompute_kernel(
            inputs, weight, target, lse, grad_loss
        )

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
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, x, target):
        loss, lse = OlmoFusedLinearCrossEntropyFunction.apply(
            x,
            self.lm_head.weight.T,
            target,
            self.ignore_index,
            self.reduction,
        )
        return loss


@torch.compile
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


def main() -> None:
    """
    Main entry point that runs the fused linear cross entropy kernel verification.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["fwd", "full"], default="full")
    args = parser.parse_args()

    batch_size, seq_len, vocab_size = 1, 8192, 100352
    hidden_size = 2560

    # batch_size, seq_len, vocab_size = 1, 8192, 8192 * 2
    # hidden_size = 2560

    n = batch_size * seq_len

    # Create inputs for fused version
    _input = torch.randn(n, hidden_size, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(
        hidden_size, vocab_size, device=DEVICE, dtype=torch.bfloat16, requires_grad=True
    )
    labels = torch.randint(0, vocab_size, (n,), device=DEVICE, dtype=torch.long)

    def helion_fn(_input, weight, labels):
        loss, lse = OlmoFusedLinearCrossEntropyFunction.apply(_input, weight, labels, -100, "sum")
        return loss

    def torch_fn(_input, weight, labels):
        loss, lse = lce_baseline_fn(_input, weight, labels, ignore_index=-100, reduction="sum")
        return loss

    # warmup / compile
    _input.grad = None
    weight.grad = None
    torch_fn(_input, weight, labels)
    _input.grad = None
    weight.grad = None
    helion_fn(_input, weight, labels)
    _input.grad = None
    weight.grad = None

    run_example(
        helion_fn,
        torch_fn,
        (_input, weight, labels),
        kernel_name="helion",
        baseline_name="torch",
        # rtol=1e-4,
        # atol=1e-4,
        bwd=args.mode == "full",
    )


if __name__ == "__main__":
    main()
