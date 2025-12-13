import argparse
from typing import Literal, Tuple

import helion
import helion.language as hl
import torch
from helion._testing import DEVICE, run_example
from helion.autotuner import LFBOPatternSearch

from olmo_core.kernels.helion.aot.aot_autotune import KernelKey


def linear_cross_entropy_fwd_key(inputs, weight, labels, ignore_index, reduction):
    bt, d = inputs.shape
    v = weight.shape[1]
    return KernelKey(
        numeric_key=bt * d,
        hash_key=(bt, d, v),
        exact_key=(weight.dtype, inputs.dtype, labels.dtype, reduction),
    )


def lce_fwd_primary_inputs(bt: list[int], d: list[int], v: list[int]):
    xprod = [
        (torch.randn(_bt, _d), torch.randn(_d, _v), torch.randint(0, _v, (_bt,)), -100, "sum")
        for _bt in bt
        for _d in d
        for _v in v
    ]
    for inputs, weight, labels, ignore_index, reduction in xprod:
        yield (
            inputs.to(device="cuda"),
            weight.to(device="cuda"),
            labels.to(device="cuda"),
            ignore_index,
            reduction,
        )


# @aot_autotune(
#     config_name="linear_cross_entropy_fwd",
#     kernel_key=linear_cross_entropy_fwd_key,
#     primary_inputs=lce_fwd_primary_inputs(
#         bt=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
#         d=[2560],
#         v=[100352],
#     ),
# )
@helion.kernel(
    # config=helion.Config(  # H100
    #     block_sizes=[64, 512, 64],
    #     indexing=["pointer", "tensor_descriptor", "pointer", "tensor_descriptor"],
    #     load_eviction_policies=["last", "last", "last"],
    #     num_stages=3,
    #     num_warps=8,
    #     pid_type="persistent_interleaved",
    #     range_flattens=[None, False, False],
    #     range_multi_buffers=[False, None, None],
    #     range_num_stages=[4, 3, 0],
    #     range_unroll_factors=[3, 1, 1],
    #     range_warp_specializes=[],
    # ),
    config=helion.Config(  # 4090
        block_sizes=[64, 512, 32],
        indexing=["pointer", "pointer", "pointer", "pointer"],
        load_eviction_policies=["first", "first", "first"],
        num_stages=1,
        num_warps=8,
        pid_type="flat",
        range_flattens=[None, None, None],
        range_multi_buffers=[None, False, True],
        range_num_stages=[0, 3, 3],
        range_unroll_factors=[0, 0, 0],
        range_warp_specializes=[],
    ),
    autotuner_fn=LFBOPatternSearch,
    static_shapes=False,  # allow dynamic shapes for the kernel, we specialize on specific dimensions
    autotune_ignore_errors=False,
    autotune_compile_timeout=30,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def olmo_linear_cross_entropy_fwd_kernel(
    inputs: torch.Tensor,  # [N, D] input to final layer
    weight: torch.Tensor,  # [D, V] weight matrix of final layer
    labels: torch.Tensor,  # [N] target labels
    ignore_index: hl.constexpr = -100,
    reduction: hl.constexpr = "sum",  # pyright: ignore[reportArgumentType]
) -> Tuple[torch.Tensor | float, torch.Tensor | float, torch.Tensor]:
    """
    Computes the fused linear cross entropy loss with partial pre-computation of gradients.

    Args:
        inputs: Input tensor of shape [N, D] where N is batch size and D is hidden dimension
        weight: Weight matrix of shape [D, V] where V is vocabulary size
        labels: Target labels tensor of shape [N] containing class indices
        ignore_index: Label value to ignore in loss computation (default: -100)
        reduction: Reduction mode, either "sum" or "none" (default: "sum")

    Returns:
        A tuple of (loss, z_squared, lse) where:
        - loss: The cross entropy loss, scalar if reduction="sum", shape [N] if reduction="none"
        - z_squared: The squared log-sum-exp values, scalar if reduction="sum", shape [N] if reduction="none"
        - lse: The log-sum-exp values (used for the backward pass), shape [N]
    """
    assert inputs.ndim == 2, f"Inputs must be 2D, got {inputs.ndim}D"
    assert inputs.shape[1] == weight.shape[0], (
        f"Input dimension mismatch: {inputs.shape[1]} != {weight.shape[0]}"
    )
    assert labels.ndim == 1, f"Labels must be 1D, got {labels.ndim}D"
    assert labels.shape[0] == inputs.shape[0], (
        f"Batch dimension mismatch: {labels.shape[0]} != {inputs.shape[0]}"
    )
    n = inputs.shape[0]
    d = hl.specialize(inputs.shape[1])
    v = hl.specialize(weight.shape[1])

    # Always need per-token lse for backward pass
    lse = torch.zeros([n], dtype=torch.float32, device=inputs.device)

    # Initialize outputs based on reduction mode - use same variable names, just different shapes
    if reduction == "sum":
        # Scalar accumulators for sum reduction
        losses = torch.zeros([], dtype=torch.float32, device=inputs.device)
        z_squared = torch.zeros([], dtype=torch.float32, device=inputs.device)
    else:  # reduction == "none"
        # Per-token outputs
        losses = torch.zeros([n], dtype=torch.float32, device=inputs.device)
        z_squared = torch.zeros([n], dtype=torch.float32, device=inputs.device)

    # --- Device Code (compiles to triton) ---
    # Outer tile loop is mapped to the launch grid, this is what gets parallelized
    for tile_n in hl.tile(n):  # Tile over the batch dimension
        labels_tile = labels[tile_n]  # [t_n]

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
                # compute logits in bf16, accumulate in fp32 for numerical stability
                logits = torch.addmm(logits, inputs[tile_n, tile_k], weight_tile_k)

            # 1b) Update running max logit and sum_exp (online lse)
            new_max = torch.maximum(max_logits, logits.amax(dim=-1))  # [t_n]
            scale = torch.exp(max_logits - new_max)  # [t_n]
            sum_exp = sum_exp * scale + torch.exp(logits - new_max.unsqueeze(1)).sum(dim=-1)
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
        z_squared_tile = lse_tile.pow(2)  # [t_n]

        # Always save lse for backward pass
        lse[tile_n] = lse_tile  # [t_n]

        # Apply masking and reduction
        valid_mask = (labels_tile != ignore_index).to(torch.float32)  # [t_n]

        if reduction == "sum":
            # Fused masking and sum reduction with atomic adds for parallel safety
            masked_ce = (losses_tile * valid_mask).sum()
            masked_z_sq = (z_squared_tile * valid_mask).sum()
            hl.atomic_add(losses, [], masked_ce)
            hl.atomic_add(z_squared, [], masked_z_sq)
        else:  # reduction == "none"
            # Store per-token values with masking applied
            losses[tile_n] = losses_tile * valid_mask
            z_squared[tile_n] = z_squared_tile * valid_mask

    return losses, z_squared, lse


@helion.kernel(
    static_shapes=False,
    autotuner_fn=LFBOPatternSearch,
    autotune_ignore_errors=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def olmo_linear_cross_entropy_bwd_recompute_kernel(
    _input: torch.Tensor,  # [N, D] input to final layer
    weight: torch.Tensor,  # [D, V] weight matrix of final layer
    labels: torch.Tensor,  # [N] target labels
    lse: torch.Tensor,  # [N] log-sum-exp from forward pass
    grad_ce_loss_scalar: torch.Tensor,  # [] scalar gradient of CE loss (for sum reduction)
    grad_z_loss_scalar: torch.Tensor,  # [] scalar gradient of z-loss (for sum reduction)
    z_loss_multiplier: float,  # z-loss multiplier
    ignore_index: hl.constexpr = -100,
    reduction: hl.constexpr = "sum",  # pyright: ignore[reportArgumentType]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the backward pass for the fused linear cross entropy loss.
    Recomputes the logits to compute the gradient with respect to the weight and input.

    The per-token gradient expansion is fused into the kernel:
    - For reduction="sum": grad_ce_per_token[i] = mask[i] * grad_ce_loss_scalar
    - For reduction="none": would need per-token grad arrays (not yet implemented)

    Args:
        grad_ce_loss_scalar: 0-dimensional tensor (scalar) - no host-device sync required
        grad_z_loss_scalar: 0-dimensional tensor (scalar) - no host-device sync required
    """
    n = _input.shape[0]
    d = hl.specialize(_input.shape[1])
    v = hl.specialize(weight.shape[1])
    assert d == weight.shape[0], f"Input dimension mismatch: {d} != {weight.shape[0]}"

    grad_weight = torch.zeros([d, v], dtype=torch.float32, device=_input.device)
    grad_input = torch.zeros([n, d], dtype=torch.float32, device=_input.device)

    # 1D tiling over batch dimension only (matches forward pass pattern)
    # Each batch tile processes ALL vocab tiles, allowing local grad_input accumulation
    for tile_n in hl.tile(n):
        labels_tile = labels[tile_n]
        lse_tile = lse[tile_n]
        valid_mask = (labels_tile != ignore_index).to(torch.float32)

        if reduction == "sum":
            # Load scalar gradients once per batch tile
            grad_ce_scalar = grad_ce_loss_scalar[()]
            grad_z_scalar = grad_z_loss_scalar[()]
            grad_ce_per_token = valid_mask * grad_ce_scalar
            # 5. Add contribution from the z-loss.
            # For each token i, if z_loss = z_loss_multiplier * lse_i^2 (with appropriate
            # masking/reduction handled in the Python wrapper), then:
            #   dL/dlogits_ij (from z-loss) = softmax_ij * dL/dlse_i
            # Chain rule: dL/dlse_i = dL/dz_squared_i * dz_squared_i/dlse_i
            #                        = grad_z_loss_scalar * z_loss_multiplier * mask_i * 2 * lse_i
            grad_z_per_token = valid_mask * grad_z_scalar * z_loss_multiplier * 2.0 * lse_tile
        else:
            raise NotImplementedError(
                f"Backward pass for reduction='{reduction}' not yet implemented"
            )

        # Process all vocab tiles for this batch tile
        for tile_v in hl.tile(v):
            # 1. Recompute logits for this (tile_n, tile_v)
            logits = hl.zeros([tile_n, tile_v], dtype=torch.float32)
            for tile_k in hl.tile(d):
                weight_kv = hl.load(weight, index=[tile_k, tile_v], eviction_policy="evict_last")
                logits = torch.addmm(logits, _input[tile_n, tile_k], weight_kv)

            # 2. Compute softmax = exp(logits - lse)
            # Because we stored the lse we dont need to do the online lse computation this time.
            softmax = torch.exp(logits - lse_tile.unsqueeze(1))

            # 3. Compute d_logits = (softmax - one_hot_target)
            local_vocab_idx = labels_tile - tile_v.begin
            is_target_in_tile = (labels_tile >= tile_v.begin) & (labels_tile < tile_v.end)
            cols = hl.arange(tile_v.block_size)
            eq = cols[None, :] == local_vocab_idx[:, None]
            mask = eq & is_target_in_tile[:, None]
            d_logits = softmax - mask.to(softmax.dtype)

            # 4. Apply CE gradient scaling
            d_logits = d_logits * grad_ce_per_token.unsqueeze(1)

            # 5. Add z-loss contribution
            d_logits = d_logits + softmax * grad_z_per_token.unsqueeze(1)
            d_logits = d_logits.to(_input.dtype)
            # basically the whole point is to avoid materializing the d_logits tensor

            # 6. Fused gradient accumulation (single D loop for better weight reuse)
            for tile_k in hl.tile(d):
                # grad_input: local accumulation (this tile_n owns these rows)
                weight_kv = hl.load(weight, index=[tile_k, tile_v], eviction_policy="evict_first")
                grad_input[tile_n, tile_k] = torch.addmm(
                    grad_input[tile_n, tile_k], d_logits, weight_kv.T
                )

            for tile_k in hl.tile(d):  # second loop to tune independently
                # grad_weight: atomic add (multiple tile_n contribute to same location)
                input_tile = _input[tile_n, tile_k]
                update_gw = hl.dot(input_tile.T, d_logits)
                hl.atomic_add(grad_weight, [tile_k, tile_v], update_gw)

    return grad_input, grad_weight


@helion.kernel(
    # autotune_effort="none",
    static_shapes=False,
    autotuner_fn=LFBOPatternSearch,
    autotune_ignore_errors=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def olmo_linear_cross_entropy_bwd_recompute_2dgrid_kernel(
    _input: torch.Tensor,  # [N, D] input to final layer
    weight: torch.Tensor,  # [D, V] weight matrix of final layer
    labels: torch.Tensor,  # [N] target labels
    lse: torch.Tensor,  # [N] log-sum-exp from forward pass
    grad_ce_loss_scalar: torch.Tensor,  # [] scalar gradient of CE loss (for sum reduction)
    grad_z_loss_scalar: torch.Tensor,  # [] scalar gradient of z-loss (for sum reduction)
    z_loss_multiplier: float,  # z-loss multiplier
    ignore_index: hl.constexpr = -100,
    reduction: hl.constexpr = "sum",  # pyright: ignore[reportArgumentType]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the backward pass for the fused linear cross entropy loss.
    Recomputes the logits to compute the gradient with respect to the weight and input.

    The per-token gradient expansion is fused into the kernel:
    - For reduction="sum": grad_ce_per_token[i] = mask[i] * grad_ce_loss_scalar
    - For reduction="none": would need per-token grad arrays (not yet implemented)

    Args:
        grad_ce_loss_scalar: 0-dimensional tensor (scalar) - no host-device sync required
        grad_z_loss_scalar: 0-dimensional tensor (scalar) - no host-device sync required
    """
    n = _input.shape[0]
    d = hl.specialize(_input.shape[1])
    v = hl.specialize(weight.shape[1])
    assert d == weight.shape[0], f"Input dimension mismatch: {d} != {weight.shape[0]}"

    grad_weight = torch.zeros([d, v], dtype=torch.float32, device=_input.device)
    grad_input = torch.zeros([n, d], dtype=torch.float32, device=_input.device)

    # This visits each (tile_n, tile_v) once, computes d_logits, and scatters updates to grad_weight
    for tile_n, tile_v in hl.tile([n, v]):
        # 1. Recompute logits for this tile of N and tile of V
        # We need full inner dimension D to compute logits
        logits = hl.zeros([tile_n, tile_v], dtype=torch.float32)
        for tile_k in hl.tile(d):
            weight_kv = hl.load(weight, index=[tile_k, tile_v], eviction_policy="evict_last")
            logits = torch.addmm(logits, _input[tile_n, tile_k], weight_kv)  # sadly we recompute

        # 2. Compute softmax = exp(logits - lse)
        # we already know lse for this batch tile so we dont need to see the whole vocab dimension
        lse_tile = lse[tile_n]  # [t_n]
        softmax = torch.exp(logits - lse_tile.unsqueeze(1))  # [t_n, t_v]
        d_logits = softmax

        # 3. Compute d_logits = (softmax - target)
        # We subtract 1.0 from softmax probabilities at the target indices.
        # Implement this without subscript assignment or boolean indexing that
        # produces data-dependent shapes by using pure elementwise ops.
        labels_tile = labels[tile_n]  # [t_n]
        local_vocab_idx = labels_tile - tile_v.begin  # [t_n]
        is_target_in_tile = (labels_tile >= tile_v.begin) & (labels_tile < tile_v.end)

        # Broadcast compare over the vocab dimension within this tile:
        # eq[i, j] = (j == local_vocab_idx[i])
        cols = hl.arange(tile_v.block_size)  # [t_v]
        eq = cols[None, :] == local_vocab_idx[:, None]  # [t_n, t_v] (boolean)
        mask = eq & is_target_in_tile[:, None]  # [t_n, t_v]
        d_logits = d_logits - mask.to(d_logits.dtype)

        # Apply valid mask and scaling according to grad_ce_loss
        # Fused gradient expansion: expand scalar gradient to per-token based on mask
        valid_mask = (labels_tile != ignore_index).to(torch.float32)  # [t_n]

        if reduction == "sum":
            # Load scalar gradients (0-dimensional tensors must be loaded inside tile loop)
            grad_ce_scalar = grad_ce_loss_scalar[()]
            grad_z_scalar = grad_z_loss_scalar[()]

            # For sum reduction: grad_ce_per_token[i] = mask[i] * grad_ce_loss_scalar
            grad_ce_per_token = valid_mask * grad_ce_scalar
        else:
            raise NotImplementedError(
                f"Backward pass for reduction='{reduction}' not yet implemented"
            )

        scale = grad_ce_per_token  # [t_n]
        d_logits = d_logits.mul(scale.unsqueeze(1))  # [t_n, v]

        # 5. Add contribution from the z-loss.
        # For each token i, if z_loss = z_loss_multiplier * lse_i^2 (with appropriate
        # masking/reduction handled in the Python wrapper), then:
        #   dL/dlogits_ij (from z-loss) = softmax_ij * dL/dlse_i
        # Chain rule: dL/dlse_i = dL/dz_squared_i * dz_squared_i/dlse_i
        #                        = grad_z_loss_scalar * z_loss_multiplier * mask_i * 2 * lse_i
        if reduction == "sum":
            grad_z_per_token = valid_mask * grad_z_scalar * z_loss_multiplier * 2.0 * lse_tile
        else:
            raise NotImplementedError(
                f"Backward pass for reduction='{reduction}' not yet implemented"
            )

        d_logits = d_logits + softmax * grad_z_per_token.unsqueeze(1)

        d_logits = d_logits.to(_input.dtype)

        # 4. Accumulate grad_weight and grad_input using atomics
        for tile_k in hl.tile(d):  # tile separately so they can be tuned independently
            # Compute partial update for grad_input[tile_n, tile_k]
            weight_kv = hl.load(weight, index=[tile_k, tile_v], eviction_policy="evict_first")
            update_gi = hl.dot(d_logits, weight_kv.T)
            hl.atomic_add(grad_input, [tile_n, tile_k], update_gi)

        for tile_k in hl.tile(d):
            # Compute partial update for grad_weight[tile_k, tile_v]
            input_tile = _input[tile_n, tile_k]
            update_gw = hl.dot(input_tile.T, d_logits)
            hl.atomic_add(grad_weight, [tile_k, tile_v], update_gw)

    return grad_input, grad_weight


class OlmoFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        weight,
        target,
        ignore_index: int = -100,
        reduction: Literal["sum", "none"] = "sum",
        z_loss_multiplier: float = 1e-4,
        bwd_impl: Literal["1d", "2d"] = "1d",
    ) -> Tuple[torch.Tensor | float, torch.Tensor | float]:
        # Masking and reduction now happens inside the kernel
        ce_loss, z_squared, lse = olmo_linear_cross_entropy_fwd_kernel(
            inputs, weight, target, ignore_index, reduction
        )

        # Save tensors/metadata needed for the backward pass.
        # NOTE: we keep the per-token `lse` so we can easily compute softmax in backward pass.
        ctx.save_for_backward(inputs, weight, target, lse)
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.z_loss_multiplier = z_loss_multiplier
        ctx.bwd_impl = bwd_impl

        z_loss = z_loss_multiplier * z_squared  # move inside kernel?
        return ce_loss, z_loss

    @staticmethod
    def backward(ctx, grad_ce_loss: torch.Tensor, grad_z_loss: torch.Tensor | None):
        """
        Backward for the fused linear + CE(+optional z-loss) op.

        The gradient expansion from scalar to per-token is now fused into the kernel.
        """
        (
            inputs,  # [N, D] input to final layer
            weight,  # [D, V] weight matrix of final layer
            target,  # [N] target labels
            lse,  # [N] log-sum-exp from forward pass
        ) = ctx.saved_tensors

        # Only support sum reduction for now
        if ctx.reduction != "sum":
            raise NotImplementedError(
                f"Backward pass for reduction='{ctx.reduction}' not yet implemented"
            )

        # Handle None gradients (when z_loss doesn't require grad)
        if grad_z_loss is None:
            grad_z_loss = torch.zeros([], dtype=lse.dtype, device=lse.device)

        # Select backward implementation:
        # - "1d": 1D tiling over N (default, no redundant logits recomputation)
        # - "2d": 2D tiling over (N, V) (original implementation with atomics on both grads)
        if ctx.bwd_impl == "2d":
            bwd_kernel = olmo_linear_cross_entropy_bwd_recompute_2dgrid_kernel
        else:
            bwd_kernel = olmo_linear_cross_entropy_bwd_recompute_kernel

        grad_input, grad_weight = bwd_kernel(
            inputs,
            weight,
            target,
            lse,
            grad_ce_loss,
            grad_z_loss,
            ctx.z_loss_multiplier,
            ctx.ignore_index,
            ctx.reduction,
        )

        return (
            grad_input,  # input
            grad_weight,  # weight
            None,  # target
            None,  # ignore_index
            None,  # reduction
            None,  # z_loss_multiplier
            None,  # bwd_impl
        )


# @torch.compile
def lce_baseline_fn(
    _input, weight, labels, ignore_index=-100, reduction="sum", z_loss_multiplier=1e-4
):
    logits = torch.matmul(_input, weight).float()
    z_squared = logits.logsumexp(dim=-1).pow(2)

    # Compute per-sample loss (no reduction for baseline comparison)
    # Compute per-sample loss without reduction
    ce_loss = torch.nn.functional.cross_entropy(
        logits, labels, ignore_index=ignore_index, reduction="none"
    )
    # Apply mask and reduction manually
    mask = labels != ignore_index
    if reduction == "sum":
        ce_loss = ce_loss.sum()
        z_squared = (z_squared * mask).sum()
    else:  # reduction == "none"
        ce_loss = ce_loss
        z_squared = z_squared * mask
    z_loss = z_loss_multiplier * z_squared
    return ce_loss, z_loss


class TorchLMHeadCE(torch.nn.Module):
    def __init__(
        self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100, reduction: str = "sum"
    ):
        super().__init__()
        self.lm_head = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, x, target):
        logits = self.lm_head(x)
        return self.ce_loss(logits.float(), target)


class OlmoLMHeadCE(torch.nn.Module):
    def __init__(
        self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100, reduction: str = "sum"
    ):
        super().__init__()
        self.lm_head = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, x, target):
        ce_loss, lse = OlmoFusedLinearCrossEntropyFunction.apply(  # pyright: ignore[reportGeneralTypeIssues]
            x,
            self.lm_head.weight.T,
            target,
            self.ignore_index,
            self.reduction,
        )
        return ce_loss


def tune_and_test() -> None:
    """
    Run fused linear cross entropy kernel tuning and verification.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["fwd", "full"], default="full")
    args = parser.parse_args()

    batch_size, seq_len, vocab_size = 1, 8192, 100352
    hidden_size = 2560

    # batch_size, seq_len, vocab_size = 1, 4096, 8192
    # hidden_size = 2560

    n = batch_size * seq_len

    # Create inputs for fused version
    _input = torch.randn(n, hidden_size, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(
        hidden_size, vocab_size, device=DEVICE, dtype=torch.bfloat16, requires_grad=True
    )
    labels = torch.randint(0, vocab_size, (n,), device=DEVICE, dtype=torch.long)

    # @torch.compile
    def helion_fn(_input, weight, labels):
        ce_loss, z_loss = OlmoFusedLinearCrossEntropyFunction.apply(
            _input, weight, labels, -100, "sum"
        )  # pyright: ignore[reportGeneralTypeIssues]
        return ce_loss + z_loss

    @torch.compile
    def torch_fn(_input, weight, labels):
        ce_loss, z_loss = lce_baseline_fn(
            _input, weight, labels, ignore_index=-100, reduction="sum"
        )
        return ce_loss + z_loss

    # warmup / compile
    torch_fn(_input, weight, labels)
    _input.grad = None
    weight.grad = None
    helion_fn(_input, weight, labels)
    _input.grad = None
    weight.grad = None

    rtol, atol = 2e-2, 1e-3  # tigher than recommended bfloat16 tolerances
    run_example(
        helion_fn,
        torch_fn,
        (_input, weight, labels),
        kernel_name="helion",
        baseline_name="torch",
        rtol=rtol,
        atol=atol,
        bwd=args.mode == "full",
    )


if __name__ == "__main__":
    tune_and_test()
