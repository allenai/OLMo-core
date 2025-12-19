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

# Shapes we tune against (product of these lists)
LCE_TUNE_PRIMARY_BT: list[int] = [8192, 16384, 32768]
LCE_TUNE_PRIMARY_D: list[int] = [2560, 4096, 5120, 8192]
LCE_TUNE_V: list[int] = [TokenizerConfig.dolma2().padded_vocab_size()]

# Shapes we benchmark against and map to tuned (primary) configs
LCE_TUNE_SECONDARY_BT: list[int] = list(range(4096, 65536, 4096))
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
    n_valid: torch.Tensor | None,
    grad_ce_loss_scalar: torch.Tensor,
    grad_z_loss_scalar: torch.Tensor,
    z_loss_multiplier: float,
    ignore_index: int,
    reduction: str,
):
    del n_valid, z_loss_multiplier, ignore_index
    bt, d = inputs.shape
    v = weight.shape[1]
    return KernelKey(
        numeric_key=bt * d,
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

            # For reduction="sum", this is unused, but the backward kernel signature includes it.
            n_valid = torch.ones([], dtype=torch.float32)

            # For reduction="sum", backward expands these scalar grads to per-token gradients.
            grad_ce_loss_scalar = torch.ones([], dtype=torch.float32)
            grad_z_loss_scalar = torch.ones([], dtype=torch.float32)
            z_loss_multiplier = 1e-4

            yield (
                inputs,
                weight,
                labels,
                lse,
                n_valid,
                grad_ce_loss_scalar,
                grad_z_loss_scalar,
                z_loss_multiplier,
                -100,
                "sum",
            )


# --- Helion kernel definitions


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
    print_repro=False,
    autotune_compile_timeout=30,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def olmo_linear_cross_entropy_fwd_kernel(
    inputs: torch.Tensor,  # [B, D]
    weight: torch.Tensor,  # [D, V]
    labels: torch.Tensor,  # [B]
    ignore_index: hl.constexpr,
    reduction: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Fused linear + cross-entropy forward pass kernel.

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
        - ce_loss: Cross-entropy loss. Scalar if reduction="sum", else shape [N].
        - z_squared: Squared logsumexp term used to compute z-loss. Scalar if reduction="sum", else shape [N].
        - lse: Per-token log-sum-exp values, shape [N].
        - n_valid: Number of valid tokens, scalar if reduction="mean", else None.
    """
    assert inputs.ndim == 2, f"Inputs must be 2D, got {inputs.ndim}D"
    assert inputs.shape[1] == weight.shape[0], (
        f"Input dimension mismatch: {inputs.shape[1]} != {weight.shape[0]}"
    )
    assert labels.ndim == 1, f"Labels must be 1D, got {labels.ndim}D"
    assert labels.shape[0] == inputs.shape[0], (
        f"Batch dimension mismatch: {labels.shape[0]} != {inputs.shape[0]}"
    )
    b = inputs.shape[0]
    d = hl.specialize(inputs.shape[1])
    v = hl.specialize(weight.shape[1])

    lse = torch.zeros([b], dtype=torch.float32, device=inputs.device)
    ce_loss = torch.zeros([], dtype=torch.float32, device=inputs.device)
    z_squared = torch.zeros([], dtype=torch.float32, device=inputs.device)
    if reduction == "mean":
        n_valid = torch.zeros([], dtype=torch.float32, device=inputs.device)
    else:
        n_valid = None

    for tile_b in hl.tile(b):
        labels_b = labels[tile_b]
        max_logits_b = hl.full([tile_b], float("-inf"), dtype=torch.float32)
        sum_exp_b = hl.zeros([tile_b], dtype=torch.float32)
        target_logits_b = hl.zeros([tile_b], dtype=torch.float32)
        for tile_v in hl.tile(v):
            logits_bv = hl.zeros([tile_b, tile_v], dtype=torch.float32)
            for tile_d in hl.tile(d):
                weight_dv = hl.load(weight, index=[tile_d, tile_v])
                logits_bv = torch.addmm(logits_bv, inputs[tile_b, tile_d], weight_dv)

            new_max = torch.maximum(max_logits_b, logits_bv.amax(dim=-1))
            scale = torch.exp(max_logits_b - new_max)
            sum_exp_b = sum_exp_b * scale + (
                torch.exp(logits_bv - new_max.unsqueeze(1)).sum(dim=-1)
            )
            max_logits_b = new_max

            # labels_bt may contain `ignore_index` values, so we need to filter them out
            is_target_in_tile = (labels_b >= tile_v.begin) & (labels_b < tile_v.end)
            local_vocab_idx = labels_b - tile_v.begin
            safe_local_vocab_idx = torch.where(is_target_in_tile, local_vocab_idx, 0)
            gathered_target_logits = (
                # torch.gather(logits_bv, dim=1, index=safe_local_vocab_idx.unsqueeze(1)).squeeze(1)
                hl.inline_triton(  # TODO(tylerr): replace with torch.gather once supported
                    "tl.sum(tl.gather({0}, {1}.to(tl.int32)[:, None], axis=1), axis=1)",
                    args=(logits_bv, safe_local_vocab_idx),
                    output_like=safe_local_vocab_idx.to(torch.float32),
                )
            )
            target_logits_b = target_logits_b + gathered_target_logits * is_target_in_tile.to(
                torch.float32
            )

        lse_b = max_logits_b + torch.log(sum_exp_b)
        ce_losses_b = lse_b - target_logits_b
        z_squared_b = lse_b.pow(2)
        lse[tile_b] = lse_b

        is_valid = (labels_b != ignore_index).to(torch.float32)
        if reduction == "sum":
            masked_ce = (ce_losses_b * is_valid).sum()
            masked_z_sq = (z_squared_b * is_valid).sum()
            hl.atomic_add(ce_loss, [], masked_ce)
            hl.atomic_add(z_squared, [], masked_z_sq)
        elif reduction == "mean":
            assert n_valid is not None, "n_valid must not be None for mean reduction"
            masked_ce = (ce_losses_b * is_valid).sum()
            masked_z_sq = (z_squared_b * is_valid).sum()
            hl.atomic_add(ce_loss, [], masked_ce)
            hl.atomic_add(z_squared, [], masked_z_sq)
            hl.atomic_add(n_valid, [], is_valid.sum())
        else:
            raise NotImplementedError(f"Forward pass for reduction='{reduction}' not supported")

    return ce_loss, z_squared, lse, n_valid


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
    static_shapes=False,
    autotuner_fn=LFBOPatternSearch,
    autotune_ignore_errors=False,
    print_repro=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def olmo_linear_cross_entropy_bwd_recompute_kernel(
    inputs: torch.Tensor,  # [B, D]
    weight: torch.Tensor,  # [D, V]
    labels: torch.Tensor,  # [B]
    lse: torch.Tensor,  # [B]
    n_valid: torch.Tensor | None,
    grad_ce_loss_scalar: torch.Tensor,
    grad_z_loss_scalar: torch.Tensor,
    z_loss_multiplier: float,
    ignore_index: hl.constexpr,
    reduction: hl.constexpr,
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
    b = inputs.shape[0]
    d = hl.specialize(inputs.shape[1])
    v = hl.specialize(weight.shape[1])

    if reduction == "mean":
        assert n_valid is not None, "n_valid must be provided for mean reduction"

    grad_weight = torch.zeros([d, v], dtype=torch.float32, device=inputs.device)
    grad_input = torch.zeros([b, d], dtype=torch.float32, device=inputs.device)
    for tile_b in hl.tile(b):
        labels_b = labels[tile_b]
        lse_b = lse[tile_b].to(torch.float32)
        is_valid = (labels_b != ignore_index).to(torch.float32)

        if reduction == "sum":
            grad_ce_scalar = grad_ce_loss_scalar[()].to(torch.float32)
            grad_z_scalar = grad_z_loss_scalar[()].to(torch.float32)
        elif reduction == "mean":
            assert n_valid is not None, "n_valid must be provided for mean reduction"
            n_valid_scalar = n_valid[()].to(torch.float32)
            grad_ce_scalar = grad_ce_loss_scalar[()].to(torch.float32) / n_valid_scalar
            grad_z_scalar = grad_z_loss_scalar[()].to(torch.float32) / n_valid_scalar
        else:
            raise NotImplementedError(f"Backward pass for reduction='{reduction}' not supported")

        grad_ce_per_token = is_valid * grad_ce_scalar
        grad_z_per_token = is_valid * grad_z_scalar * z_loss_multiplier * 2.0 * lse_b

        for tile_v in hl.tile(v):
            logits = hl.zeros([tile_b, tile_v], dtype=torch.float32)
            for tile_d in hl.tile(d):
                weight_dv = hl.load(weight, index=[tile_d, tile_v], eviction_policy="evict_last")
                logits = torch.addmm(logits, inputs[tile_b, tile_d], weight_dv)

            softmax = torch.exp(logits - lse_b.unsqueeze(1))
            local_vocab_idx = labels_b - tile_v.begin
            is_target_in_tile = (labels_b >= tile_v.begin) & (labels_b < tile_v.end)
            cols = hl.arange(tile_v.block_size)
            is_target = (cols[None, :] == local_vocab_idx[:, None]) & is_target_in_tile[:, None]
            grad_logits = softmax - is_target.to(softmax.dtype)

            grad_logits = grad_logits * grad_ce_per_token.unsqueeze(1)
            grad_logits = grad_logits + softmax * grad_z_per_token.unsqueeze(1)
            grad_logits = grad_logits.to(inputs.dtype)

            for tile_d in hl.tile(d):
                weight_dv = hl.load(weight, index=[tile_d, tile_v])
                grad_input[tile_b, tile_d] = torch.addmm(
                    grad_input[tile_b, tile_d], grad_logits, weight_dv.T
                )
            for tile_d in hl.tile(d):
                update_gw = hl.dot(inputs[tile_b, tile_d].T, grad_logits)
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
    print_repro=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def olmo_linear_cross_entropy_bwd_recompute_2dgrid_kernel(
    inputs: torch.Tensor,  # [N, D] input to final layer
    weight: torch.Tensor,  # [D, V] weight matrix of final layer
    labels: torch.Tensor,  # [N] target labels
    lse: torch.Tensor,  # [N] log-sum-exp from forward pass
    n_valid: torch.Tensor | None,
    grad_ce_loss_scalar: torch.Tensor,  # [] scalar gradient of CE loss
    grad_z_loss_scalar: torch.Tensor,  # [] scalar gradient of z-loss
    z_loss_multiplier: float,  # z-loss multiplier
    ignore_index: hl.constexpr,
    reduction: hl.constexpr,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the backward pass for the fused linear cross entropy loss.
    Recomputes the logits to compute the gradient with respect to the weight and input.

    Args:
        grad_ce_loss_scalar: 0-dimensional tensor (scalar)
        grad_z_loss_scalar: 0-dimensional tensor (scalar)
    """
    b = inputs.shape[0]
    d = hl.specialize(inputs.shape[1])
    v = hl.specialize(weight.shape[1])

    if reduction == "mean":
        assert n_valid is not None, "n_valid must be provided for mean reduction"

    # Accumulate gradients in fp32 for stability, then cast to the parameter/input dtype
    # at the end to satisfy autograd dtype requirements.
    grad_weight = torch.zeros([d, v], dtype=torch.float32, device=inputs.device)
    grad_input = torch.zeros([b, d], dtype=torch.float32, device=inputs.device)

    # Each program instance processes one (tile_b, tile_v) pair.
    for tile_b, tile_v in hl.tile([b, v]):
        labels_b = labels[tile_b]
        lse_b = lse[tile_b].to(torch.float32)
        is_valid = (labels_b != ignore_index).to(  # pyright: ignore[reportAttributeAccessIssue]
            torch.float32
        )

        if reduction == "sum":
            grad_ce_scalar = grad_ce_loss_scalar[()].to(torch.float32)
            grad_z_scalar = grad_z_loss_scalar[()].to(torch.float32)
        elif reduction == "mean":
            n_valid_scalar = n_valid[()].to(torch.float32)
            grad_ce_scalar = grad_ce_loss_scalar[()].to(torch.float32) / n_valid_scalar
            grad_z_scalar = grad_z_loss_scalar[()].to(torch.float32) / n_valid_scalar
        else:
            raise NotImplementedError(f"Backward pass for reduction='{reduction}' not supported")

        grad_ce_per_token = is_valid * grad_ce_scalar
        grad_z_per_token = is_valid * grad_z_scalar * z_loss_multiplier * 2.0 * lse_b

        logits = hl.zeros([tile_b, tile_v], dtype=torch.float32)
        for tile_d in hl.tile(d):
            weight_dv = hl.load(weight, index=[tile_d, tile_v])
            logits = torch.addmm(logits, inputs[tile_b, tile_d], weight_dv)

        softmax = torch.exp(logits - lse_b.unsqueeze(1))
        local_vocab_idx = labels_b - tile_v.begin
        is_target_in_tile = (labels_b >= tile_v.begin) & (labels_b < tile_v.end)
        cols = hl.arange(tile_v.block_size)
        is_target = (cols[None, :] == local_vocab_idx[:, None]) & is_target_in_tile[:, None]
        grad_logits = softmax - is_target.to(softmax.dtype)

        grad_logits = grad_logits * grad_ce_per_token.unsqueeze(1)
        grad_logits = grad_logits + softmax * grad_z_per_token.unsqueeze(1)
        grad_logits = grad_logits.to(inputs.dtype)

        for tile_d in hl.tile(d):
            weight_dv = hl.load(weight, index=[tile_d, tile_v])
            update_gi = hl.dot(grad_logits, weight_dv.T)
            hl.atomic_add(grad_input, [tile_b, tile_d], update_gi)

        for tile_d in hl.tile(d):
            input_tile = hl.load(inputs, index=[tile_b, tile_d], eviction_policy="evict_first")
            update_gw = hl.dot(input_tile.T, grad_logits)
            hl.atomic_add(grad_weight, [tile_d, tile_v], update_gw)

    return grad_input.to(inputs.dtype), grad_weight.to(weight.dtype)


# --- Autograd Function


class OlmoFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int = -100,
        reduction: Literal["sum", "mean"] = "mean",
        compute_z_loss: bool = False,
        z_loss_multiplier: float = 0.0,
    ) -> Tuple[torch.Tensor | float, torch.Tensor | float | None]:
        # Masking and reduction now happens inside the kernel
        ce_loss, z_squared, lse, n_valid = olmo_linear_cross_entropy_fwd_kernel(
            inputs, weight, target, ignore_index, reduction
        )
        if reduction == "mean":
            assert n_valid is not None, "n_valid must not be None for mean reduction"
            ce_loss = ce_loss / n_valid
            if compute_z_loss:
                z_squared = z_squared / n_valid

        if compute_z_loss:
            z_loss = z_loss_multiplier * z_squared
        else:
            z_loss = None

        # Save tensors/metadata needed for the backward pass.
        # NOTE: we keep the per-token `lse` so we can easily compute softmax in backward pass.
        ctx.save_for_backward(inputs, weight, target, lse, n_valid)
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.z_loss_multiplier = z_loss_multiplier
        return ce_loss, z_loss

    @staticmethod
    def backward(
        ctx: object, grad_ce_loss: torch.Tensor, grad_z_loss: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None, None, None]:
        assert grad_ce_loss is not None
        inputs, weight, target, lse, n_valid = ctx.saved_tensors
        if grad_z_loss is None:
            grad_z_loss = torch.zeros([], dtype=lse.dtype, device=lse.device)

        grad_input, grad_weight = olmo_linear_cross_entropy_bwd_recompute_kernel(
            inputs,
            weight,
            target,
            lse,
            n_valid,
            grad_ce_loss,
            grad_z_loss,
            ctx.z_loss_multiplier,
            ctx.ignore_index,
            ctx.reduction,
        )

        # TODO: switch
        # grad_input, grad_weight = olmo_linear_cross_entropy_bwd_recompute_2dgrid_kernel(
        #     inputs,
        #     weight,
        #     target,
        #     lse,
        #     n_valid,
        #     grad_ce_loss,
        #     grad_z_loss,
        #     ctx.z_loss_multiplier,  # type: ignore[attr-defined]
        #     ctx.ignore_index,  # type: ignore[attr-defined]
        #     ctx.reduction,  # type: ignore[attr-defined]
        # )

        return (
            grad_input,  # input
            grad_weight,  # weight
            None,  # target
            None,  # ignore_index
            None,  # reduction
            None,  # compute_z_loss
            None,  # z_loss_multiplier
        )


# --- helper fns

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
        _, _, lse0, _ = olmo_linear_cross_entropy_fwd_kernel(x0, w0, y0, ignore_index, reduction)
        torch.cuda.synchronize()

        # Backward kernels (trigger bwd AOT)
        grad_ce0 = torch.ones([], device=device, dtype=torch.float32)
        grad_z0 = torch.ones([], device=device, dtype=torch.float32)
        n_valid0 = torch.ones([], device=device, dtype=torch.float32)

        print("- tuning backward (1d grid)...")
        olmo_linear_cross_entropy_bwd_recompute_kernel(
            x0,
            w0,
            y0,
            lse0,
            n_valid0,
            grad_ce0,
            grad_z0,
            z_loss_multiplier,
            ignore_index,
            reduction,
        )
        torch.cuda.synchronize()

        print("- tuning backward (2d grid)...")
        olmo_linear_cross_entropy_bwd_recompute_2dgrid_kernel(
            x0,
            w0,
            y0,
            lse0,
            n_valid0,
            grad_ce0,
            grad_z0,
            z_loss_multiplier,
            ignore_index,
            reduction,
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
