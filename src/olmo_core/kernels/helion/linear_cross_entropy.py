import argparse
from typing import Literal, Tuple

import helion
import helion.language as hl
import torch
from helion._testing import DEVICE, run_example
from helion.autotuner import LFBOPatternSearch


@helion.kernel(
    # config=helion.Config(
    #     block_sizes=[64, 512, 64],
    #     indexing=[
    #         "tensor_descriptor",
    #         "tensor_descriptor",
    #         "pointer",
    #         "tensor_descriptor",
    #         "pointer",
    #     ],
    #     load_eviction_policies=["first", "first", "first"],
    #     num_stages=1,
    #     num_warps=8,
    #     pid_type="flat",
    #     range_flattens=[None, None, False],
    #     range_multi_buffers=[None, None, None],
    #     range_num_stages=[0, 3, 3],
    #     range_unroll_factors=[0, 0, 0],
    #     range_warp_specializes=[],
    # ),
    autotune_effort="none",
    autotuner_fn=LFBOPatternSearch,
    static_shapes=False,  # allow dynamic shapes for the kernel, we specialize on specific dimensions
    autotune_ignore_errors=False,
    autotune_compile_timeout=20,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def olmo_linear_cross_entropy_fwd_kernel(
    inputs: torch.Tensor,  # [N, D] input to final layer
    weight: torch.Tensor,  # [D, V] weight matrix of final layer
    labels: torch.Tensor,  # [N] target labels
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the fused linear cross entropy loss with partial pre-computation of gradients.

    Args:
        inputs: Input tensor of shape [N, D] where N is batch size and D is hidden dimension
        weight: Weight matrix of shape [D, V] where V is vocabulary size
        labels: Target labels tensor of shape [N] containing class indices

    Returns:
        A tuple of (loss, lse) where:
        - loss: The cross entropy loss, tensor of shape [N]
        - lse: The log-sum-exp values (can be used to compute z_loss externally), shape [N]
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

    # Accumulate losses in fp32 for numerical stability, even if inputs are bfloat16
    losses = torch.zeros([n], dtype=torch.float32, device=inputs.device)
    lse = torch.zeros([n], dtype=torch.float32, device=inputs.device)

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

        losses[tile_n] = losses_tile  # [t_n]
        lse[tile_n] = lse_tile  # [t_n]

    return losses, lse


@helion.kernel(
    # config=helion.Config(
    #     block_sizes=[128, 256, 128, 64],
    #     indexing=[
    #         "pointer",
    #         "pointer",
    #         "tensor_descriptor",
    #         "pointer",
    #         "pointer",
    #         "tensor_descriptor",
    #         "tensor_descriptor",
    #     ],
    #     l2_groupings=[8],
    #     load_eviction_policies=["first", "last", "last", "last", "last"],
    #     loop_orders=[[0, 1]],
    #     num_stages=1,
    #     num_warps=8,
    #     pid_type="flat",
    #     range_flattens=[None, True, None],
    #     range_multi_buffers=[None, True, None],
    #     range_num_stages=[0, 2, 2],
    #     range_unroll_factors=[0, 4, 0],
    #     range_warp_specializes=[],
    # ),
    static_shapes=True,
    autotuner_fn=LFBOPatternSearch,
    autotune_effort="none",
    autotune_ignore_errors=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def olmo_linear_cross_entropy_bwd_recompute_kernel(
    _input: torch.Tensor,  # [N, D] input to final layer
    weight: torch.Tensor,  # [D, V] weight matrix of final layer
    labels: torch.Tensor,  # [N] target labels
    lse: torch.Tensor,  # [N] log-sum-exp from forward pass
    grad_ce_loss: torch.Tensor,  # [N] gradient of the cross entropy loss
    grad_z_loss: torch.Tensor,  # [N] gradient of the z-loss
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

    # This visits each (tile_n, tile_v) once, computes d_logits, and scatters updates to grad_weight
    for tile_n, tile_v in hl.tile([n, v]):
        # 1. Recompute logits for this tile of N and tile of V
        # We need full inner dimension D to compute logits
        logits = hl.zeros([tile_n, tile_v], dtype=torch.float32)
        for tile_k in hl.tile(d):
            weight_kv = hl.load(  # eviction_policy="evict_last", needed later
                weight, index=[tile_k, tile_v]
            )
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
        valid_mask = labels_tile != ignore_index
        scale = valid_mask * grad_ce_loss[tile_n]
        d_logits = d_logits.mul(scale.unsqueeze(1))  # [t_n, v]

        # 5. Add contribution from the z-loss.
        # For each token i, if z_loss = z_loss_multiplier * lse_i^2 (with appropriate
        # masking/reduction handled in the Python wrapper), then:
        #   dL/dlogits_ij (from z-loss) = softmax_ij * dL/dlse_i
        # Note: grad_z_loss already contains the full gradient w.r.t. lse (including the 2*lse factor)
        grad_z_tile = grad_z_loss[tile_n]  # [t_n], gradient w.r.t. lse_i
        d_logits = d_logits + softmax * grad_z_tile.unsqueeze(1)

        d_logits = d_logits.to(_input.dtype)

        # 4. Accumulate grad_weight and grad_input using atomics
        for tile_k in hl.tile(d):
            # Compute partial update for grad_weight[tile_k, tile_v]
            input_tile = _input[tile_n, tile_k]
            update_gw = hl.dot(input_tile.T, d_logits)
            hl.atomic_add(grad_weight, [tile_k, tile_v], update_gw)

        for tile_k in hl.tile(d):  # tile separately so they can be tuned independently
            # Compute partial update for grad_input[tile_n, tile_k]
            weight_kv = hl.load(weight, index=[tile_k, tile_v], eviction_policy="evict_first")
            update_gi = hl.dot(d_logits, weight_kv.T)
            hl.atomic_add(grad_input, [tile_n, tile_k], update_gi)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ce_loss, lse = olmo_linear_cross_entropy_fwd_kernel(inputs, weight, target)

        # Save tensors/metadata needed for the backward pass.
        # NOTE: we keep the per-token `lse` so we can compute softmax in backward pass.

        ctx.save_for_backward(inputs, weight, target, lse)
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.z_loss_multiplier = z_loss_multiplier

        # Apply masking + reduction on the host; the gradient wrt the per-token CE
        # values will be reconstructed in `backward()`.
        mask = target != ignore_index
        z_squared = lse.pow(2)  # Compute z_squared for z-loss
        if reduction == "sum":
            ce_loss = (ce_loss * mask).sum()
            z_squared = (z_squared * mask).sum()

        z_loss = z_loss_multiplier * z_squared
        return ce_loss, z_loss

    @staticmethod
    def backward(ctx, grad_ce_loss, grad_z_loss):
        """
        Backward for the fused linear + CE(+optional z-loss) op.

        The Helion kernel expects *per-token* gradients for the CE loss and z-loss,
        so here we expand the reduced upstream gradients back to per-token form.
        """
        (
            inputs,  # [N, D] input to final layer
            weight,  # [D, V] weight matrix of final layer
            target,  # [N] target labels
            lse,  # [N] log-sum-exp from forward pass
        ) = ctx.saved_tensors
        # ctx has ignore_index, reduction, z_loss_multiplier

        mask = target != ctx.ignore_index

        # --- CE loss grads (per-token) ---
        # TODO: foreach ops?
        if ctx.reduction == "none":
            # Already per-token.
            grad_ce_per_token = grad_ce_loss
        elif ctx.reduction == "sum":
            grad_ce_per_token = mask.to(lse.dtype) * grad_ce_loss
        else:
            raise ValueError(f"Unsupported reduction: {ctx.reduction}")

        # --- z-loss grads (per-token, w.r.t. z_squared) ---
        # We model the z-loss exactly as in the unfused implementation:
        #   z_loss = z_loss_multiplier * reduce(mask * lse^2)
        # Chain rule: dL/dlse_i = dL/dz_squared_i * dz_squared_i/dlse_i
        #                        = dL/dz_squared_i * 2 * lse_i
        # where dL/dz_squared_i = grad_z_loss * z_loss_multiplier * mask_i
        if grad_z_loss is None:
            grad_z_per_token = torch.zeros_like(lse)
        else:
            grad_z_loss = grad_z_loss * ctx.z_loss_multiplier
            if ctx.reduction == "none":
                grad_z_per_token = grad_z_loss * 2.0 * lse
            elif ctx.reduction == "sum":
                grad_z_per_token = mask.to(lse.dtype) * grad_z_loss * 2.0 * lse
            else:
                raise ValueError(f"Unsupported reduction: {ctx.reduction}")

        grad_input, grad_weight = olmo_linear_cross_entropy_bwd_recompute_kernel(
            inputs, weight, target, lse, grad_ce_per_token, grad_z_per_token
        )

        return (
            grad_input,  # input
            grad_weight,  # weight
            None,  # target
            None,  # ignore_index
            None,  # reduction
            None,  # z_loss_multiplier
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


def main() -> None:
    """
    Main entry point that runs the fused linear cross entropy kernel verification.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["fwd", "full"], default="full")
    args = parser.parse_args()

    # batch_size, seq_len, vocab_size = 1, 8192, 100352
    # hidden_size = 2560

    batch_size, seq_len, vocab_size = 1, 4096, 8192
    hidden_size = 2560

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
    _input.grad = None
    weight.grad = None
    torch_fn(_input, weight, labels)
    _input.grad = None
    weight.grad = None
    helion_fn(_input, weight, labels)
    _input.grad = None
    weight.grad = None

    try:
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
    except AssertionError as e:
        print(f"\nCaught AssertionError in run_example: {e}")
        if args.mode == "full":
            print("Running manual gradient comparison for debugging...")

            # 1. Compute Reference Gradients
            _input.grad = None
            weight.grad = None
            loss_ref = torch_fn(_input, weight, labels)
            loss_ref.backward()
            grad_input_ref = _input.grad.clone() if _input.grad is not None else None
            grad_weight_ref = weight.grad.clone() if weight.grad is not None else None

            # 2. Compute Kernel Gradients
            _input.grad = None
            weight.grad = None
            loss_act = helion_fn(_input, weight, labels)
            loss_act.backward()
            grad_input_act = _input.grad.clone() if _input.grad is not None else None
            grad_weight_act = weight.grad.clone() if weight.grad is not None else None

            # Print data types of gradients
            print("\n--- Gradient Data Types ---")
            print(
                f"grad_input_ref dtype: {grad_input_ref.dtype if grad_input_ref is not None else None}"
            )
            print(
                f"grad_input_act dtype: {grad_input_act.dtype if grad_input_act is not None else None}"
            )
            print(
                f"grad_weight_ref dtype: {grad_weight_ref.dtype if grad_weight_ref is not None else None}"
            )
            print(
                f"grad_weight_act dtype: {grad_weight_act.dtype if grad_weight_act is not None else None}"
            )

            def report_diff(name, ref, act):
                if ref is None or act is None:
                    print(
                        f"{name}: One of the gradients is None. Ref: {ref is not None}, Act: {act is not None}"
                    )
                    return

                diff = (ref - act).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                median_diff = diff.median().item()
                ref_abs_max = ref.abs().max().item()
                ref_abs_mean = ref.abs().mean().item()
                ref_abs_median = ref.abs().median().item()
                act_abs_max = act.abs().max().item()
                act_abs_mean = act.abs().mean().item()
                act_abs_median = act.abs().median().item()

                print(f"\n--- {name} Gradient Comparison ---")
                print(f"Max Diff: {max_diff:.2e}")
                print(f"Mean Diff: {mean_diff:.2e}")
                print(f"Median Diff: {median_diff:.2e}")
                print(
                    f"Ref Max Abs: {ref_abs_max:.2e}, Mean Abs: {ref_abs_mean:.2e}, Median Abs: {ref_abs_median:.2e}"
                )
                print(
                    f"Act Max Abs: {act_abs_max:.2e}, Mean Abs: {act_abs_mean:.2e}, Median Abs: {act_abs_median:.2e}"
                )

                if max_diff > 0:
                    # Find indices of max diff
                    flat_indices = torch.topk(diff.flatten(), 5).indices
                    print("Top 5 discrepancies:")
                    for idx in flat_indices:
                        # Convert flat index to coords
                        coords = []
                        rem = idx.item()
                        for dim in reversed(ref.shape):
                            coords.append(rem % dim)
                            rem //= dim
                        coords = tuple(reversed(coords))

                        r_val = ref[coords].item()
                        a_val = act[coords].item()
                        d_val = diff[coords].item()
                        annotation = ""
                        # For input gradients, coords[0] corresponds to flattened (batch, seq) index
                        if name.lower() == "input" and len(coords) == 2:
                            token_idx = coords[0]
                            label_id = labels[token_idx].item()
                            batch_idx = token_idx // seq_len
                            seq_idx = token_idx % seq_len
                            annotation = (
                                f" | token_idx={token_idx}, batch={batch_idx}, seq={seq_idx}, "
                                f"label_id={label_id}"
                            )
                        # For weight gradients, coords[1] is the vocab / label dimension
                        elif name.lower() == "weight" and len(coords) == 2:
                            vocab_idx = coords[1]
                            # Find where this vocab index actually appears as a target label
                            matching = (labels == vocab_idx).nonzero(as_tuple=True)[0]
                            if matching.numel() > 0:
                                # Show up to first 3 occurrences
                                shown = matching[:3].tolist()
                                extra = matching.numel() - len(shown)
                                annotation = (
                                    f" | vocab_idx={vocab_idx} (label dim), "
                                    f"as target at token_idx(s)={shown}"
                                )
                                if extra > 0:
                                    annotation += f" (+{extra} more)"
                            else:
                                annotation = (
                                    f" | vocab_idx={vocab_idx} (label dim), "
                                    "never a target label in this batch"
                                )

                        print(
                            f"  Idx {coords}: Ref={r_val:.4e}, Act={a_val:.4e}, "
                            f"Diff={d_val:.4e}{annotation}"
                        )

            if grad_input_ref is not None and grad_input_act is not None:
                report_diff("Input", grad_input_ref, grad_input_act)

            if grad_weight_ref is not None and grad_weight_act is not None:
                report_diff("Weight", grad_weight_ref, grad_weight_act)

        raise e


if __name__ == "__main__":
    main()
