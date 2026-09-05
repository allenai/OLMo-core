"""Experimental BF16-rounded weight GEMM into an owned FP32 DDP bucket.

Restricted to BF16, EP1, no output buffers, and exclusive one-use-per-forward
expert parameters. Never fabricates a dummy gradient to trigger autograd hooks.
The explicit DDP completion callback runs after the real accumulation is enqueued.
"""

import importlib.metadata
from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(None)
def _compile(device_capacity, majors):
    import cutlass
    import cutlass.cute as cute
    from quack.gemm_default_epi import GemmDefaultEpiMixin, GemmDefaultSm100
    from quack.gemm_tvm_ffi_utils import (
        compile_gemm_kernel,
        make_fake_gemm_tensors,
        make_fake_scheduler_args,
        make_fake_varlen_args,
    )

    assert importlib.metadata.version("quack-kernels") == "0.5.0"
    assert device_capacity[0] == 10, device_capacity

    class RoundedGradientGemm(GemmDefaultSm100):
        @cute.jit
        def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
            tRS_rD.store(tRS_rD.load().to(cutlass.BFloat16).to(cutlass.Float32))
            return GemmDefaultEpiMixin.epi_visit_subtile(
                self, params, epi_loop_tensors, tRS_rD, tRS_rC
            )

    a, b, d, c, _, _, k, groups = make_fake_gemm_tensors(
        cutlass.BFloat16,
        cutlass.BFloat16,
        cutlass.Float32,
        cutlass.Float32,
        *majors,
        varlen_k=True,
    )
    # A dedicated compiler/cache: never patch QuACK's global class or cache.
    return compile_gemm_kernel(
        RoundedGradientGemm,
        cutlass.BFloat16,
        (256, 256),
        (2, 1, 1),
        False,
        True,
        False,
        True,
        device_capacity,
        a,
        b,
        d,
        c,
        RoundedGradientGemm.EpilogueArguments(),
        make_fake_scheduler_args(False, False, groups),
        make_fake_varlen_args(False, True, False, k),
    )


def rounded_wgrad_add(a, b, output, cumulative):
    """output += BF16(a @ b.T), with per-expert variable reduction lengths."""
    from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
    from quack.gemm_default_epi import GemmDefaultEpiMixin
    from quack.gemm_tvm_ffi_utils import (
        get_majors,
        make_scheduler_args,
        make_varlen_args,
        perm3d,
    )

    if (
        a.dtype != torch.bfloat16
        or b.dtype != torch.bfloat16
        or output.dtype != torch.float32
        or not output.is_contiguous()
        or a.stride(0) != 1
        or b.stride(0) != 1
    ):
        raise ValueError("Unsupported rounded weight-gradient shape/dtype/layout")
    capacity = get_device_capacity(a.device)
    tensors = perm3d(a, b, output, output, varlen_k=True)
    compiled = _compile(capacity, get_majors(*tensors))
    compiled(
        *tensors,
        GemmDefaultEpiMixin.EpilogueArguments(add_to_output=None, rounding_mode=None),
        make_scheduler_args(get_max_active_clusters(2, device_capacity=capacity), 8, None),
        make_varlen_args(None, cumulative, None),
        None,
        None,
        None,
    )


class _RoundedWeightGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, cumulative, transpose):
        ctx.save_for_backward(x, weight, cumulative)
        ctx.transpose = transpose
        return F.grouped_mm(x, weight.transpose(1, 2) if transpose else weight, offs=cumulative[1:])

    @staticmethod
    def backward(ctx, grad):
        x, weight, cumulative = ctx.saved_tensors
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Rounded weight-gradient accumulation does not support higher derivatives"
            )
        grad = grad.contiguous()
        destination, done = weight._olmo_profile_begin_external_grad(weight)
        if ctx.transpose:
            rounded_wgrad_add(grad.T, x.T, destination, cumulative)
        else:
            rounded_wgrad_add(x.T, grad.T, destination, cumulative)
        done()
        dx = None
        if ctx.needs_input_grad[0]:
            rhs = weight if ctx.transpose else weight.transpose(1, 2)
            dx = F.grouped_mm(grad, rhs, offs=cumulative[1:])
        return dx, None, None, None


@torch.compiler.disable
def rounded_weight_gmm(x, weight, counts, transpose):
    """Graph-break prototype: qualify ordering and measure overhead before promotion."""
    if not hasattr(weight, "_olmo_profile_begin_external_grad"):
        raise RuntimeError("Rounded weight GEMM requires explicit FP32 DDP bucket ownership")
    if not weight.is_leaf or not weight.requires_grad:
        raise RuntimeError("Rounded weight GEMM requires an exclusive trainable leaf parameter")
    cumulative = torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))
    return _RoundedWeightGemm.apply(x, weight, cumulative, transpose)
