import itertools
from functools import partial
from typing import Literal, Tuple

import helion
import helion.language as hl
import torch
from helion.autotuner import LFBOPatternSearch
from triton.testing import do_bench  # pyright: ignore[reportMissingImports]

from olmo_core.data import TokenizerConfig
from olmo_core.kernels.helion.aot.aot_autotune import KernelKey, helion_aot_autotune

# --- AOT compilation helpers


# Shapes we tune/benchmark against (product of these lists).
LCE_TUNE_PRIMARY_BT: list[int] = [8192, 16384, 32768]
LCE_TUNE_PRIMARY_D: list[int] = [2560, 4096, 5120, 8192]
LCE_TUNE_V: list[int] = [TokenizerConfig.dolma2().padded_vocab_size()]

LCE_TUNE_SECONDARY_BT: list[int] = list(range(2048, 65536, 2048))
LCE_TUNE_SECONDARY_D: list[int] = list(range(256, 16385, 256))


def lce_fwd_key(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    reduction: str,
) -> KernelKey:
    """Key used to lookup the best config for the LCE forward kernel."""
    del ignore_index
    bt, d = inputs.shape
    v = weight.shape[1]
    return KernelKey(
        numeric_key=bt * d,  # used for routing if hash_key doesn't match
        hash_key=(bt, d, v),  # will be used if it matches
        exact_key=(weight.dtype, inputs.dtype, labels.dtype, reduction),  # must match
    )


def lce_fwd_inputs(bt: list[int], d: list[int], v: list[int]):
    with torch.device("cuda"):
        for bt_size, d_size, v_size in itertools.product(bt, d, v):
            inputs = torch.randn(bt_size, d_size, dtype=torch.bfloat16)
            weight = torch.randn(d_size, v_size, dtype=torch.bfloat16)
            labels = torch.randint(0, v_size, (bt_size,), dtype=torch.long)
            yield (inputs, weight, labels, -100, "sum")


def lce_bwd_key(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    grad_ce_loss_scalar: torch.Tensor,
    grad_z_loss_scalar: torch.Tensor,
    z_loss_multiplier: float,
    ignore_index: int,
    reduction: str,
):
    del z_loss_multiplier, ignore_index
    # Backward kernel is specialized on (N, D, V) and several constexprs (e.g. reduction).
    bt, d = inputs.shape
    v = weight.shape[1]
    return KernelKey(
        # Backward cost scales with N*D*V (recomputing logits and accumulating grads).
        numeric_key=bt * d * v,
        hash_key=(
            bt,
            d,
            v,
            lse.dtype,
            grad_ce_loss_scalar.dtype,
            grad_z_loss_scalar.dtype,
            reduction,
        ),
        exact_key=(
            weight.dtype,
            inputs.dtype,
            labels.dtype,
            lse.dtype,
            grad_ce_loss_scalar.dtype,
            grad_z_loss_scalar.dtype,
            reduction,
        ),
    )


def lce_bwd_inputs(bt: list[int], d: list[int], v: list[int]):
    with torch.device("cuda"):
        for bt_size, d_size, v_size in itertools.product(bt, d, v):
            inputs = torch.randn(bt_size, d_size, dtype=torch.bfloat16)
            weight = torch.randn(d_size, v_size, dtype=torch.bfloat16)
            labels = torch.randint(0, v_size, (bt_size,), dtype=torch.long)

            # Backward kernels consume per-token `lse` from forward. For autotuning we only need
            # correct shape/dtype; choose a large value to avoid overflow in exp(logits - lse).
            lse = torch.full((bt_size,), 1.0e4, dtype=torch.float32)

            # For reduction="sum", backward expands these scalar grads to per-token gradients.
            grad_ce_loss_scalar = torch.ones([], dtype=torch.float32)
            grad_z_loss_scalar = torch.ones([], dtype=torch.float32)
            z_loss_multiplier = 1e-4

            yield (
                inputs,
                weight,
                labels,
                lse,
                grad_ce_loss_scalar,
                grad_z_loss_scalar,
                z_loss_multiplier,
                -100,
                "sum",
            )


# --- Helion kernels


@helion_aot_autotune(
    config_name="linear_cross_entropy_fwd",
    kernel_key=lce_fwd_key,
    primary_inputs=partial(
        lce_fwd_inputs, bt=LCE_TUNE_PRIMARY_BT, d=LCE_TUNE_PRIMARY_D, v=LCE_TUNE_V
    ),
    secondary_inputs=partial(
        lce_fwd_inputs, bt=LCE_TUNE_SECONDARY_BT, d=LCE_TUNE_SECONDARY_D, v=LCE_TUNE_V
    ),
)
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
    # config=helion.Config(  # 4090
    #     block_sizes=[64, 512, 32],
    #     indexing=["pointer", "pointer", "pointer", "pointer"],
    #     load_eviction_policies=["first", "first", "first"],
    #     num_stages=1,
    #     num_warps=8,
    #     pid_type="flat",
    #     range_flattens=[None, None, None],
    #     range_multi_buffers=[None, False, True],
    #     range_num_stages=[0, 3, 3],
    #     range_unroll_factors=[0, 0, 0],
    #     range_warp_specializes=[],
    # ),
    autotuner_fn=LFBOPatternSearch,
    static_shapes=False,  # allow dynamic shapes for the kernel, we specialize on specific dimensions
    autotune_ignore_errors=False,
    print_output_code=False,
    autotune_compile_timeout=30,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def olmo_linear_cross_entropy_fwd_kernel(
    inputs: torch.Tensor,  # [N, D] input to final layer
    weight: torch.Tensor,  # [D, V] weight matrix of final layer
    labels: torch.Tensor,  # [N] target labels
    ignore_index: hl.constexpr = -100,  # pyright: ignore[reportArgumentType]
    reduction: hl.constexpr = "sum",  # pyright: ignore[reportArgumentType]
) -> Tuple[torch.Tensor | float, torch.Tensor | float, torch.Tensor]:
    """
    Fused linear + cross-entropy forward.

    This avoids materializing the full logits matrix by streaming over the vocab dimension
    and maintaining an online log-sum-exp (LSE) state. We also emit per-token `lse` for the
    backward kernels.

    Args:
        inputs: Input tensor of shape [N, D] where N is batch size and D is hidden dimension
        weight: Weight matrix of shape [D, V] where V is vocabulary size
        labels: Target labels tensor of shape [N] containing class indices
        ignore_index: Label value to ignore in loss computation (default: -100)
        reduction: Reduction mode, either "sum" or "none" (default: "sum")

    Returns:
        (ce_loss, z_squared, lse):
        - ce_loss: Scalar if reduction="sum", else shape [N].
        - z_squared: Scalar if reduction="sum", else shape [N].
        - lse: Per-token log-sum-exp values, shape [N].
    """
    assert inputs.ndim == 2, f"Inputs must be 2D, got {inputs.ndim}D"
    assert inputs.shape[1] == weight.shape[0], (
        f"Input dimension mismatch: {inputs.shape[1]} != {weight.shape[0]}"
    )
    assert labels.ndim == 1, f"Labels must be 1D, got {labels.ndim}D"
    assert labels.shape[0] == inputs.shape[0], (
        f"Batch dimension mismatch: {labels.shape[0]} != {inputs.shape[0]}"
    )
    bt = inputs.shape[0]
    d = hl.specialize(inputs.shape[1])
    v = hl.specialize(weight.shape[1])

    # Always need per-token lse for backward pass
    lse = torch.zeros([bt], dtype=torch.float32, device=inputs.device)

    # Initialize outputs based on reduction mode - use same variable names, just different shapes
    if reduction == "sum":
        # Scalar accumulators for sum reduction
        losses = torch.zeros([], dtype=torch.float32, device=inputs.device)
        z_squared = torch.zeros([], dtype=torch.float32, device=inputs.device)
    else:  # reduction == "none"
        # Per-token outputs
        losses = torch.zeros([bt], dtype=torch.float32, device=inputs.device)
        z_squared = torch.zeros([bt], dtype=torch.float32, device=inputs.device)

    # --- Device Code (compiles to triton) ---
    # Outer tile loop is mapped to the launch grid, this is what gets parallelized
    for tile_bt in hl.tile(bt):  # Tile over the token (batch) dimension
        labels_bt = labels[tile_bt]  # [t_bt]

        # Online log-sum-exp state.
        max_logits = hl.full([tile_bt], float("-inf"), dtype=torch.float32)  # [t_bt]
        sum_exp = hl.zeros([tile_bt], dtype=torch.float32)  # [t_bt] running sum of exp
        target_logits = hl.zeros([tile_bt], dtype=torch.float32)  # [t_bt] logit at target

        # Stream over vocab tiles: compute logits, update LSE state, and capture target logits.
        for tile_v in hl.tile(v):  # inner tile loops are sequential
            # 1) Compute logits for this vocab tile.
            logits = hl.zeros([tile_bt, tile_v], dtype=torch.float32)
            for tile_d in hl.tile(d):
                weight_tile = hl.load(weight, index=[tile_d, tile_v])
                # compute logits in bf16, accumulate in fp32 for numerical stability
                logits = torch.addmm(logits, inputs[tile_bt, tile_d], weight_tile)

            # 2) Update online log-sum-exp.
            new_max = torch.maximum(max_logits, logits.amax(dim=-1))  # [t_n]
            scale = torch.exp(max_logits - new_max)  # [t_n]
            sum_exp = sum_exp * scale + torch.exp(logits - new_max.unsqueeze(1)).sum(dim=-1)
            max_logits = new_max

            # 3) Extract logits at the target indices for labels that fall within this tile.
            # IMPORTANT: `labels_tile` may contain `ignore_index` (e.g. -100). Never use
            # out-of-range indices with `tl.gather`; mask them out before and after gathering.
            is_target_in_tile = (labels_bt >= tile_v.begin) & (labels_bt < tile_v.end)  # [t_n]
            local_vocab_idx = labels_bt - tile_v.begin  # [t_n]
            safe_local_vocab_idx = torch.where(is_target_in_tile, local_vocab_idx, 0)
            gathered_target_logits = hl.inline_triton(
                "tl.sum(tl.gather({0}, {1}.to(tl.int32)[:, None], axis=1), axis=1)",
                args=(logits, safe_local_vocab_idx),
                output_like=safe_local_vocab_idx.to(torch.float32),
            )
            target_logits = target_logits + gathered_target_logits * is_target_in_tile.to(  # pyright: ignore[reportAttributeAccessIssue]
                torch.float32
            )

        # Finalize per-token LSE for this batch tile.
        lse_tile = max_logits + torch.log(sum_exp)

        # Cross-entropy: LSE - logit_at_target (masked below).
        losses_tile = lse_tile - target_logits  # [t_n]
        z_squared_tile = lse_tile.pow(2)  # [t_n]

        # Always save LSE for backward pass.
        lse[tile_bt] = lse_tile  # [t_n]

        # Apply masking and reduction.
        is_valid = (labels_bt != ignore_index).to(torch.float32)  # pyright: ignore[reportAttributeAccessIssue]  # [t_n]

        if reduction == "sum":
            # Fused masking and sum reduction with atomics for parallel safety.
            masked_ce = (losses_tile * is_valid).sum()
            masked_z_sq = (z_squared_tile * is_valid).sum()
            hl.atomic_add(losses, [], masked_ce)
            hl.atomic_add(z_squared, [], masked_z_sq)
        else:  # reduction == "none"
            # Store per-token values with masking applied
            losses[tile_bt] = losses_tile * is_valid
            z_squared[tile_bt] = z_squared_tile * is_valid

    return losses, z_squared, lse


@helion_aot_autotune(
    config_name="linear_cross_entropy_bwd_1d",
    kernel_key=lce_bwd_key,
    primary_inputs=partial(
        lce_bwd_inputs, bt=LCE_TUNE_PRIMARY_BT, d=LCE_TUNE_PRIMARY_D, v=LCE_TUNE_V
    ),
    secondary_inputs=partial(
        lce_bwd_inputs, bt=LCE_TUNE_SECONDARY_BT, d=LCE_TUNE_SECONDARY_D, v=LCE_TUNE_V
    ),
)
@helion.kernel(
    # config=helion.Config(
    #     block_sizes=[64, 512, 64, 128, 32],
    #     indexing=[
    #         "pointer",
    #         "tensor_descriptor",
    #         "pointer",
    #         "pointer",
    #         "tensor_descriptor",
    #         "tensor_descriptor",
    #         "tensor_descriptor",
    #         "pointer",
    #         "tensor_descriptor",
    #         "pointer",
    #     ],
    #     load_eviction_policies=["first", "last", "", "", "", "", "first"],
    #     num_stages=5,
    #     num_warps=8,
    #     pid_type="flat",
    #     range_flattens=[None, None, False, False, True],
    #     range_multi_buffers=[None, True, None, True, None],
    #     range_num_stages=[0, 1, 3, 3, 4],
    #     range_unroll_factors=[0, 1, 1, 4, 1],
    #     range_warp_specializes=[],
    # ),
    static_shapes=False,
    autotuner_fn=LFBOPatternSearch,
    autotune_ignore_errors=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def olmo_linear_cross_entropy_bwd_recompute_kernel(
    inputs: torch.Tensor,  # [N, D] input to final layer
    weight: torch.Tensor,  # [D, V] weight matrix of final layer
    labels: torch.Tensor,  # [N] target labels
    lse: torch.Tensor,  # [N] log-sum-exp from forward pass
    grad_ce_loss_scalar: torch.Tensor,  # [] scalar gradient of CE loss (for sum reduction)
    grad_z_loss_scalar: torch.Tensor,  # [] scalar gradient of z-loss (for sum reduction)
    z_loss_multiplier: float,  # z-loss multiplier
    ignore_index: hl.constexpr = -100,  # pyright: ignore[reportArgumentType]
    reduction: hl.constexpr = "sum",  # pyright: ignore[reportArgumentType]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the backward pass for the fused linear cross entropy loss.
    Recomputes the logits to compute the gradient with respect to the weight and input.

    The per-token gradient expansion is fused into the kernel:
    - For reduction="sum": grad_ce_per_token[i] = mask[i] * grad_ce_loss_scalar
    - For reduction="none": would need per-token grad arrays (not yet implemented)

    Args:
        grad_ce_loss_scalar: 0-dimensional tensor (scalar)
        grad_z_loss_scalar: 0-dimensional tensor (scalar)
    """
    bt = inputs.shape[0]
    d = hl.specialize(inputs.shape[1])
    v = hl.specialize(weight.shape[1])
    assert d == weight.shape[0], f"Input dimension mismatch: {d} != {weight.shape[0]}"

    # Accumulate gradients in fp32 for stability, then cast to the parameter/input dtype
    # at the end to satisfy autograd dtype requirements.
    grad_weight = torch.zeros([d, v], dtype=torch.float32, device=inputs.device)
    grad_input = torch.zeros([bt, d], dtype=torch.float32, device=inputs.device)

    # 1D tiling over batch dimension only (matches forward pass pattern)
    # Each batch tile processes ALL vocab tiles, allowing local grad_input accumulation
    for tile_bt in hl.tile(bt):
        labels_bt = labels[tile_bt]
        lse_bt = lse[tile_bt]
        is_valid = (labels_bt != ignore_index).to(  # pyright: ignore[reportAttributeAccessIssue]
            torch.float32
        )

        if reduction == "sum":
            # Load scalar gradients once per batch tile
            grad_ce_scalar = grad_ce_loss_scalar[()]
            grad_z_scalar = grad_z_loss_scalar[()]
            grad_ce_per_token = is_valid * grad_ce_scalar
            # z-loss contribution: dL/dlogits_ij += softmax_ij * dL/dlse_i
            # where dL/dlse_i = grad_z_loss_scalar * z_loss_multiplier * mask_i * 2 * lse_i.
            grad_z_per_token = is_valid * grad_z_scalar * z_loss_multiplier * 2.0 * lse_bt
        else:
            raise NotImplementedError(
                f"Backward pass for reduction='{reduction}' not yet implemented"
            )

        # Process all vocab tiles for this batch tile
        for tile_v in hl.tile(v):
            # 1. Recompute logits for this (tile_n, tile_v)
            logits = hl.zeros([tile_bt, tile_v], dtype=torch.float32)
            for tile_d in hl.tile(d):
                weight_tile = hl.load(weight, index=[tile_d, tile_v], eviction_policy="evict_last")
                logits = torch.addmm(logits, inputs[tile_bt, tile_d], weight_tile)

            # 2. Compute softmax = exp(logits - lse)
            # Because we stored the LSE, we don't need the online LSE computation here.
            softmax = torch.exp(logits - lse_bt.unsqueeze(1))

            # 3. Compute d_logits = (softmax - one_hot_target)
            local_vocab_idx = labels_bt - tile_v.begin
            is_target_in_tile = (labels_bt >= tile_v.begin) & (labels_bt < tile_v.end)
            cols = hl.arange(tile_v.block_size)
            is_target = (cols[None, :] == local_vocab_idx[:, None]) & is_target_in_tile[:, None]
            grad_logits = softmax - is_target.to(softmax.dtype)

            # 4. Apply CE gradient scaling
            grad_logits = grad_logits * grad_ce_per_token.unsqueeze(1)

            # 5. Add z-loss contribution
            grad_logits = grad_logits + softmax * grad_z_per_token.unsqueeze(1)
            grad_logits = grad_logits.to(inputs.dtype)

            # 6. Fused gradient accumulation (single D loop for better weight reuse)
            for tile_d in hl.tile(d):
                # grad_input: local accumulation (this tile_n owns these rows)
                weight_tile = hl.load(weight, index=[tile_d, tile_v], eviction_policy="evict_first")
                grad_input[tile_bt, tile_d] = torch.addmm(
                    grad_input[tile_bt, tile_d], grad_logits, weight_tile.T
                )

            for tile_d in hl.tile(d):  # second loop to tune independently
                # grad_weight: atomic add (multiple tile_n contribute to same location)
                input_tile = inputs[tile_bt, tile_d]
                update_gw = hl.dot(input_tile.T, grad_logits)
                hl.atomic_add(grad_weight, [tile_d, tile_v], update_gw)

    return grad_input.to(inputs.dtype), grad_weight.to(weight.dtype)


@helion_aot_autotune(
    config_name="linear_cross_entropy_bwd_2d",
    kernel_key=lce_bwd_key,
    primary_inputs=partial(
        lce_bwd_inputs, bt=LCE_TUNE_PRIMARY_BT, d=LCE_TUNE_PRIMARY_D, v=LCE_TUNE_V
    ),
    secondary_inputs=partial(
        lce_bwd_inputs, bt=LCE_TUNE_SECONDARY_BT, d=LCE_TUNE_SECONDARY_D, v=LCE_TUNE_V
    ),
)
@helion.kernel(
    # autotune_effort="none",
    static_shapes=False,
    autotuner_fn=LFBOPatternSearch,
    autotune_ignore_errors=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def olmo_linear_cross_entropy_bwd_recompute_2dgrid_kernel(
    inputs: torch.Tensor,  # [N, D] input to final layer
    weight: torch.Tensor,  # [D, V] weight matrix of final layer
    labels: torch.Tensor,  # [N] target labels
    lse: torch.Tensor,  # [N] log-sum-exp from forward pass
    grad_ce_loss_scalar: torch.Tensor,  # [] scalar gradient of CE loss (for sum reduction)
    grad_z_loss_scalar: torch.Tensor,  # [] scalar gradient of z-loss (for sum reduction)
    z_loss_multiplier: float,  # z-loss multiplier
    ignore_index: hl.constexpr = -100,  # pyright: ignore[reportArgumentType]
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
    bt = inputs.shape[0]
    d = hl.specialize(inputs.shape[1])
    v = hl.specialize(weight.shape[1])
    assert d == weight.shape[0], f"Input dimension mismatch: {d} != {weight.shape[0]}"

    # Accumulate gradients in fp32 for stability, then cast to the parameter/input dtype
    # at the end to satisfy autograd dtype requirements.
    grad_weight = torch.zeros([d, v], dtype=torch.float32, device=inputs.device)
    grad_input = torch.zeros([bt, d], dtype=torch.float32, device=inputs.device)

    # This visits each (tile_n, tile_v) once, computes d_logits, and scatters updates to grad_weight
    for tile_bt, tile_v in hl.tile([bt, v]):
        # 1. Recompute logits for this tile of N and tile of V
        # We need full inner dimension D to compute logits
        logits = hl.zeros([tile_bt, tile_v], dtype=torch.float32)
        for tile_d in hl.tile(d):
            weight_tile = hl.load(weight, index=[tile_d, tile_v], eviction_policy="evict_last")
            logits = torch.addmm(logits, inputs[tile_bt, tile_d], weight_tile)

        # 2. Compute softmax = exp(logits - lse)
        # We already know the LSE for this batch tile, so we don't need the full vocab dimension.
        lse_bt = lse[tile_bt]  # [t_n]
        softmax = torch.exp(logits - lse_bt.unsqueeze(1))  # [t_n, t_v]
        grad_logits = softmax

        # 3. Compute d_logits = (softmax - target)
        # We subtract 1.0 from softmax probabilities at the target indices.
        # Implement this without subscript assignment or boolean indexing that
        # produces data-dependent shapes by using pure elementwise ops.
        labels_bt = labels[tile_bt]  # [t_n]
        local_vocab_idx = labels_bt - tile_v.begin  # [t_n]
        is_target_in_tile = (labels_bt >= tile_v.begin) & (labels_bt < tile_v.end)

        # Broadcast compare over the vocab dimension within this tile:
        # eq[i, j] = (j == local_vocab_idx[i])
        cols = hl.arange(tile_v.block_size)  # [t_v]
        is_target = (cols[None, :] == local_vocab_idx[:, None]) & is_target_in_tile[:, None]
        grad_logits = grad_logits - is_target.to(grad_logits.dtype)

        # Apply valid mask and scaling according to grad_ce_loss
        # Fused gradient expansion: expand scalar gradient to per-token based on mask
        is_valid = (labels_bt != ignore_index).to(  # pyright: ignore[reportAttributeAccessIssue]
            torch.float32
        )  # [t_n]

        if reduction == "sum":
            # Load scalar gradients (0-dimensional tensors must be loaded inside tile loop)
            grad_ce_scalar = grad_ce_loss_scalar[()]
            grad_z_scalar = grad_z_loss_scalar[()]

            # For sum reduction: grad_ce_per_token[i] = mask[i] * grad_ce_loss_scalar
            grad_ce_per_token = is_valid * grad_ce_scalar
        else:
            raise NotImplementedError(
                f"Backward pass for reduction='{reduction}' not yet implemented"
            )

        scale = grad_ce_per_token  # [t_n]
        grad_logits = grad_logits.mul(scale.unsqueeze(1))  # [t_n, v]

        # 5. Add contribution from the z-loss.
        # For each token i, if z_loss = z_loss_multiplier * lse_i^2 (with appropriate
        # masking/reduction handled in the Python wrapper), then:
        #   dL/dlogits_ij (from z-loss) = softmax_ij * dL/dlse_i
        # Chain rule: dL/dlse_i = dL/dz_squared_i * dz_squared_i/dlse_i
        #                        = grad_z_loss_scalar * z_loss_multiplier * mask_i * 2 * lse_i
        if reduction == "sum":
            grad_z_per_token = is_valid * grad_z_scalar * z_loss_multiplier * 2.0 * lse_bt
        else:
            raise NotImplementedError(
                f"Backward pass for reduction='{reduction}' not yet implemented"
            )

        grad_logits = grad_logits + softmax * grad_z_per_token.unsqueeze(1)

        grad_logits = grad_logits.to(inputs.dtype)

        # 4. Accumulate grad_weight and grad_input using atomics
        for tile_d in hl.tile(d):  # tile separately so they can be tuned independently
            # Compute partial update for grad_input[tile_n, tile_k]
            weight_tile = hl.load(weight, index=[tile_d, tile_v], eviction_policy="evict_first")
            update_gi = hl.dot(grad_logits, weight_tile.T)
            hl.atomic_add(grad_input, [tile_bt, tile_d], update_gi)

        for tile_d in hl.tile(d):
            # Compute partial update for grad_weight[tile_k, tile_v]
            input_tile = inputs[tile_bt, tile_d]
            update_gw = hl.dot(input_tile.T, grad_logits)
            hl.atomic_add(grad_weight, [tile_d, tile_v], update_gw)

    return grad_input.to(inputs.dtype), grad_weight.to(weight.dtype)


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
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx, grad_ce_loss: torch.Tensor, grad_z_loss: torch.Tensor | None
    ):
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
            x, self.lm_head.weight.T, target, self.ignore_index, self.reduction
        )
        return ce_loss


# TODO: make sure I like the perf at our specific dims
# TODO: then tune kernel across all dims
# TODO: then test kernel against baseline implementations in actual training


def tune_aot() -> None:
    """
    Trigger Helion AOT tuning for the LCE kernels.
    """
    dtype: torch.dtype = torch.bfloat16
    ignore_index: int = -100
    z_loss_multiplier: float = 1e-4
    reduction: str = "sum"

    # Trigger AOT routing/tuning by running one representative shape through each kernel.
    # Any call into a @helion_aot_autotune-wrapped kernel triggers the AOT logic.
    bt_grid = sorted({*LCE_TUNE_PRIMARY_BT, *LCE_TUNE_SECONDARY_BT})
    d_grid = sorted({*LCE_TUNE_PRIMARY_D, *LCE_TUNE_SECONDARY_D})
    v_grid = list(LCE_TUNE_V)
    bt0, d0, v0 = bt_grid[0], d_grid[0], v_grid[0]

    device = torch.device("cuda")
    with torch.device(device):
        x0 = torch.randn(bt0, d0, device=device, dtype=dtype)
        w0 = torch.randn(d0, v0, device=device, dtype=dtype)
        y0 = torch.randint(0, v0, (bt0,), device=device, dtype=torch.long)

        print("AOT tuning")

        # Forward (triggers fwd AOT)
        print("- tuning forward...")
        _, _, lse0 = olmo_linear_cross_entropy_fwd_kernel(x0, w0, y0, ignore_index, reduction)
        torch.cuda.synchronize()

        # Backward kernels (trigger bwd AOT)
        grad_ce0 = torch.ones([], device=device, dtype=torch.float32)
        grad_z0 = torch.ones([], device=device, dtype=torch.float32)

        print("- tuning backward (1d grid)...")
        olmo_linear_cross_entropy_bwd_recompute_kernel(
            x0, w0, y0, lse0, grad_ce0, grad_z0, z_loss_multiplier, ignore_index, reduction
        )
        torch.cuda.synchronize()

        print("- tuning backward (2d grid)...")
        olmo_linear_cross_entropy_bwd_recompute_2dgrid_kernel(
            x0, w0, y0, lse0, grad_ce0, grad_z0, z_loss_multiplier, ignore_index, reduction
        )
        torch.cuda.synchronize()

        print("AOT tuning complete")


def benchmark() -> None:
    """
    Benchmark forward and full (forward+backward) across the configured size grid.
    """
    from olmo_core.nn.lm_head import LMHead, LMLossImplementation

    dtype: torch.dtype = torch.bfloat16
    z_loss_multiplier: float = 1e-4
    loss_reduction: str = "sum"

    # Interpret `bt` as N (flattened batch tokens), so we use batch_size=1 and seq_len=bt.
    bt_grid = sorted({*LCE_TUNE_PRIMARY_BT, *LCE_TUNE_SECONDARY_BT})
    d_grid = sorted({*LCE_TUNE_PRIMARY_D, *LCE_TUNE_SECONDARY_D})
    v_grid = list(LCE_TUNE_V)

    device = torch.device("cuda")
    rep = 500

    def build_head(d_model: int, vocab_size: int, impl: LMLossImplementation) -> LMHead:
        return LMHead(
            d_model=d_model,
            vocab_size=vocab_size,
            dtype=dtype,
            bias=False,
            init_device=device.type,
            loss_implementation=impl,
        )

    print("\nBenchmarking forward and full (forward+backward) across sizes...")
    print(f"V={v_grid} dtype={dtype} reduction={loss_reduction} z={z_loss_multiplier} rep={rep}")
    print("-" * 100)
    print(f"{'BT':>8} {'D':>6} {'V':>7} {'kind':>5} | {'helion':>10} {'default':>11} {'liger':>9}")
    print("-" * 100)

    for v in v_grid:
        for d in d_grid:
            # Create LMHead instances with different loss implementations (per (d, v)).
            helion_head = build_head(
                d_model=d, vocab_size=v, impl=LMLossImplementation.helion_fused_linear
            )
            default_head = build_head(d_model=d, vocab_size=v, impl=LMLossImplementation.default)
            liger_head = build_head(
                d_model=d, vocab_size=v, impl=LMLossImplementation.liger_fused_linear
            )

            # Copy weights to ensure all heads use the same weights for fair comparison.
            with torch.no_grad():
                default_head.w_out.weight.copy_(helion_head.w_out.weight)
                liger_head.w_out.weight.copy_(helion_head.w_out.weight)

            def loss_fn(head: LMHead, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                out = head(
                    x,
                    labels=y,
                    loss_reduction=loss_reduction,
                    z_loss_multiplier=z_loss_multiplier,
                )
                return out.loss

            impls: dict[str, tuple[LMHead, torch.Tensor]] = {
                "helion": (helion_head, helion_head.w_out.weight),
                "default": (default_head, default_head.w_out.weight),
                "liger": (liger_head, liger_head.w_out.weight),
            }

            for bt in bt_grid:
                x = torch.randn(1, bt, d, device=device, dtype=dtype, requires_grad=True)
                y = torch.randint(0, v, (1, bt), device=device, dtype=torch.long)

                results: dict[str, tuple[float, float]] = {}
                for name, (head, w) in impls.items():
                    fwd_ms = float(do_bench(lambda _h=head: loss_fn(_h, x, y), rep=rep))

                    def full_pass(_h=head):
                        loss = loss_fn(_h, x, y)
                        loss.backward()

                    full_ms = float(do_bench(full_pass, grad_to_none=[x, w], rep=rep))
                    results[name] = (fwd_ms, full_ms)

                h_fwd, h_full = results["helion"]
                d_fwd, d_full = results["default"]
                l_fwd, l_full = results["liger"]
                print(
                    f"{bt:8d} {d:6d} {v:7d} {'fwd':>5} | {h_fwd:10.3f} {d_fwd:11.3f} {l_fwd:9.3f}"
                )
                print(
                    f"{bt:8d} {d:6d} {v:7d} {'full':>5} | {h_full:10.3f} {d_full:11.3f} {l_full:9.3f}"
                )

            del helion_head, default_head, liger_head


if __name__ == "__main__":
    tune_aot()
    benchmark()
