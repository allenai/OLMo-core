# Adapted from
# https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/fused_norm_gate.py
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li (MIT license).
#
# This is a specialized port of fla's gated RMS norm for the configuration actually used by
# :class:`~olmo_core.nn.attention.recurrent.GatedDeltaNet2`:
#
# - RMS norm only (no layer-norm/mean path)
# - optional elementwise weight, no bias
# - no residual/prenorm handling
# - sigmoid or swish/silu output gate
# - the tiled ``D <= 512`` kernel path only (GDN2 uses ``D = head_v_dim = 128``)
#
# Numerics (fp32 accumulation, full-row reductions) are kept identical to fla so outputs are
# bitwise-comparable to the original implementation.

import functools
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import triton  # type: ignore
import triton.language as tl  # type: ignore

MAX_SUPPORTED_DIM = 512
"""Only the tiled (``D <= 512``) kernel path is vendored; use fla's implementation beyond this."""


@functools.lru_cache(maxsize=None)
def _get_multiprocessor_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@triton.heuristics({"HAS_WEIGHT": lambda args: args["w"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for num_warps in [4, 8, 16]
    ],
    key=["D", "NB", "HAS_WEIGHT"],
)
@triton.jit
def rms_norm_gated_fwd_kernel(
    x,  # pointer to the input
    g,  # pointer to the gate
    y,  # pointer to the output
    w,  # pointer to the weights
    rstd,  # pointer to the 1/std
    eps,  # epsilon to avoid division by zero
    T,  # number of rows in x
    D: tl.constexpr,  # number of columns in x
    BT: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_t = o_t < T
    m_x = m_t[:, None] & m_d[None, :]

    p_x = x + o_t[:, None] * D + o_d[None, :]
    b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)

    b_xbar = tl.where(m_d[None, :], b_x, 0.0)
    b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)

    p_rstd = rstd + o_t
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), mask=m_t)

    b_x_hat = b_x * b_rstd[:, None]
    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
        b_y = b_x_hat * b_w[None, :]
    else:
        b_y = b_x_hat

    # swish/sigmoid output gate
    p_g = g + o_t[:, None] * D + o_d[None, :]
    b_g = tl.load(p_g, mask=m_x, other=0.0).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == "sigmoid":
        b_y = b_y * tl.sigmoid(b_g)

    p_y = y + o_t[:, None] * D + o_d[None, :]
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_x)


@triton.heuristics({"HAS_WEIGHT": lambda args: args["w"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for num_warps in [4, 8, 16]
    ],
    key=["D", "NB", "HAS_WEIGHT"],
)
@triton.jit
def rms_norm_gated_bwd_kernel(
    x,  # pointer to the input
    g,  # pointer to the gate
    w,  # pointer to the weights
    dy,  # pointer to the output gradient
    dx,  # pointer to the input gradient
    dg,  # pointer to the gate gradient
    dw,  # pointer to the partial sum of weights gradient
    rstd,  # pointer to the 1/std
    T,
    BS,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    i_s = tl.program_id(0)
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
        b_dw = tl.zeros((BT, BD), dtype=tl.float32)

    # the caller guarantees NS = min(SM, T), so every program has at least one token.
    # the last program's range may slightly exceed T (since BS = ceil(T/NS));
    # accesses are bounded by the true tensor shape (T, D), so the partial
    # tail tile is handled by zero-padding loads and skipping stores.
    # the m_t mask below further ensures dw only accumulates valid rows (< T).
    for i_t in range(i_s * BS, i_s * BS + BS, BT):
        o_t = (i_t + tl.arange(0, BT)).to(tl.int64)
        m_t = o_t < T
        m_x = m_t[:, None] & m_d[None, :]
        p_x = x + o_t[:, None] * D + o_d[None, :]
        p_g = g + o_t[:, None] * D + o_d[None, :]
        p_dy = dy + o_t[:, None] * D + o_d[None, :]
        p_dx = dx + o_t[:, None] * D + o_d[None, :]
        p_dg = dg + o_t[:, None] * D + o_d[None, :]
        # [BT, BD]
        b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)
        b_g = tl.load(p_g, mask=m_x, other=0.0).to(tl.float32)
        b_dy = tl.load(p_dy, mask=m_x, other=0.0).to(tl.float32)

        p_rstd = rstd + o_t
        b_rstd = tl.load(p_rstd, mask=m_t, other=0.0)
        # recompute the normalized output from the saved input and rstd
        b_xhat = b_x * b_rstd[:, None]
        b_xhat = tl.where(m_d[None, :], b_xhat, 0.0)
        b_y = b_xhat * b_w[None, :] if HAS_WEIGHT else b_xhat

        b_sigmoid_g = tl.sigmoid(b_g)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            b_dg = b_dy * b_y * (b_sigmoid_g + b_g * b_sigmoid_g * (1 - b_sigmoid_g))
            b_dy = b_dy * b_g * b_sigmoid_g
        elif ACTIVATION == "sigmoid":
            b_dg = b_dy * b_y * b_sigmoid_g * (1 - b_sigmoid_g)
            b_dy = b_dy * b_sigmoid_g
        b_wdy = b_dy

        if HAS_WEIGHT:
            # when BT > BS, a tile may span into the next program's range;
            # mask to this program's upper bound to avoid double-counting dw.
            m_t = (i_t + tl.arange(0, BT)) < min(i_s * BS + BS, T)
            b_wdy = b_dy * b_w
            b_dw += tl.where(m_t[:, None], b_dy * b_xhat, 0.0)

        b_c1 = tl.sum(b_xhat * b_wdy, axis=1) / D
        b_dx = (b_wdy - b_xhat * b_c1[:, None]) * b_rstd[:, None]

        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=m_x)
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_x)

    if HAS_WEIGHT:
        tl.store(dw + i_s * D + o_d, tl.sum(b_dw, axis=0), mask=m_d)


def rms_norm_gated_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: Optional[torch.Tensor],
    activation: str,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the fused gated RMS norm forward kernel over a 2D ``(T, D)`` input.

    :returns: The output ``y`` and the per-row fp32 ``rstd`` (needed by the backward pass).
    """
    T, D = x.shape
    if D > MAX_SUPPORTED_DIM:
        raise ValueError(
            f"Only D <= {MAX_SUPPORTED_DIM} is supported (got {D}). "
            "Use fla.modules.FusedRMSNormGated for larger feature dims."
        )
    if weight is not None:
        assert weight.shape == (D,)
    y = torch.empty_like(x)
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)
    BD = triton.next_power_of_2(D)
    # NOTE: 'NB' only exists to be an autotune key: it buckets T at 64Ki-row granularity so a
    # varying number of rows doesn't cause excessive recompilation/re-autotuning.
    NB = triton.cdiv(T, 2048 * 32)

    def grid(meta):
        return (triton.cdiv(T, meta["BT"]),)

    rms_norm_gated_fwd_kernel[grid](
        x=x,
        g=g,
        y=y,
        w=weight,
        rstd=rstd,
        eps=eps,
        T=T,
        D=D,
        BD=BD,
        NB=NB,
        ACTIVATION=activation,
    )
    return y, rstd


def rms_norm_gated_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    g: torch.Tensor,
    weight: Optional[torch.Tensor],
    rstd: torch.Tensor,
    activation: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Run the fused gated RMS norm backward kernel over 2D ``(T, D)`` inputs.

    :returns: Gradients ``(dx, dg, dw)``; ``dw`` is ``None`` when ``weight`` is ``None``.
    """
    T, D = x.shape
    assert dy.shape == (T, D)
    dx = torch.empty_like(x)
    dg = torch.empty_like(g)

    BD = triton.next_power_of_2(D)
    # cap program count to T so no program is completely idle.
    # without this, high-SM GPUs (e.g. B200, 160 SMs) with small T would
    # launch idle programs whose tile offsets exceed the tensor shape.
    NS = min(_get_multiprocessor_count(x.device.index), T)
    BS = math.ceil(T / NS)

    dw = (
        torch.empty((NS, D), dtype=torch.float, device=weight.device)
        if weight is not None
        else None
    )
    # See the forward pass note on 'NB'.
    NB = triton.cdiv(T, 2048 * 32)

    rms_norm_gated_bwd_kernel[(NS,)](
        x=x,
        g=g,
        w=weight,
        dy=dy,
        dx=dx,
        dg=dg,
        dw=dw,
        rstd=rstd,
        T=T,
        D=D,
        BS=BS,
        BD=BD,
        NB=NB,
        ACTIVATION=activation,
    )
    if weight is not None:
        assert dw is not None
        return dx, dg, dw.sum(0).to(weight.dtype)
    return dx, dg, None


class RMSNormGatedFunction(torch.autograd.Function):
    """
    Autograd wrapper around the fused gated RMS norm kernels.

    The backward pass recomputes the normalized output from the saved input and ``rstd``
    rather than saving any activations beyond ``(x, g, weight, rstd)``.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: Optional[torch.Tensor],
        activation: str,
        eps: float,
    ) -> torch.Tensor:
        shape_og = x.shape
        x = x.reshape(-1, x.shape[-1])
        g = g.reshape(-1, g.shape[-1])
        if not x.is_contiguous():
            x = x.contiguous()
        if not g.is_contiguous():
            g = g.contiguous()
        if weight is not None and not weight.is_contiguous():
            weight = weight.contiguous()
        y, rstd = rms_norm_gated_fwd(x=x, g=g, weight=weight, activation=activation, eps=eps)
        ctx.save_for_backward(x, g, weight, rstd)
        ctx.activation = activation
        ctx.shape_og = shape_og
        return y.reshape(shape_og)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):  # type: ignore[override]
        x, g, weight, rstd = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1])
        if not dy.is_contiguous():
            dy = dy.contiguous()
        assert dy.shape == x.shape
        dx, dg, dw = rms_norm_gated_bwd(
            dy=dy, x=x, g=g, weight=weight, rstd=rstd, activation=ctx.activation
        )
        return dx.reshape(ctx.shape_og), dg.reshape(ctx.shape_og), dw, None, None


def rms_norm_gated(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: Optional[torch.Tensor],
    activation: str = "swish",
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused gated RMS norm: ``y = rms_norm(x) * weight * act(g)`` where ``act`` is
    ``sigmoid(g)`` or ``g * sigmoid(g)`` (swish/silu).

    :param x: Input of shape ``(..., D)`` with ``D <= 512``.
    :param g: Gate of the same shape as ``x``.
    :param weight: Optional elementwise scale of shape ``(D,)``.
    :param activation: ``"sigmoid"``, ``"swish"``, or ``"silu"``.
    :param eps: Epsilon added to the mean square before the reciprocal square root.
    """
    return RMSNormGatedFunction.apply(x, g, weight, activation, eps)


class FusedRMSNormGated(nn.Module):
    """
    Gated RMS norm module, a drop-in replacement for ``fla.modules.FusedRMSNormGated`` in the
    (no bias, no residual) configuration used by GDN layers.
    """

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        activation: str = "swish",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.activation = activation

        if self.activation not in ["swish", "silu", "sigmoid"]:
            raise ValueError(f"Unsupported activation: {self.activation}")

        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += f", activation={self.activation}"
        s += ")"
        return s

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return rms_norm_gated(x, g, self.weight, activation=self.activation, eps=self.eps)
