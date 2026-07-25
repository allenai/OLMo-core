#!/usr/bin/env python3
"""Numerically validate FLA 0.4.1 KDA against its recurrent Torch reference."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from fla.ops.kda import chunk_kda
from fla.ops.kda.gate import naive_kda_gate
from fla.ops.kda.naive import naive_recurrent_kda


def _clone_grad(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(True)


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    difference = (actual.float() - expected.float()).abs()
    print(f"{name}: max_abs={difference.max().item():.6g} mean_abs={difference.mean().item():.6g}")
    torch.testing.assert_close(actual.float(), expected.float(), rtol=3e-2, atol=3e-2)


def validate_reference_forward_backward() -> None:
    torch.manual_seed(2026)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, length, heads, dim = 1, 64, 2, 32

    inputs = [
        torch.randn(batch, length, heads, dim, device=device, dtype=dtype),
        torch.randn(batch, length, heads, dim, device=device, dtype=dtype),
        torch.randn(batch, length, heads, dim, device=device, dtype=dtype),
        torch.randn(batch, length, heads, dim, device=device, dtype=dtype),
        torch.randn(batch, length, heads, device=device, dtype=dtype),
        torch.log(torch.empty(heads, device=device, dtype=torch.float32).uniform_(1, 16)),
        torch.zeros(heads * dim, device=device, dtype=torch.float32),
    ]
    chunk_inputs = [_clone_grad(tensor) for tensor in inputs]
    ref_inputs = [_clone_grad(tensor) for tensor in inputs]
    q, k, v, raw_gate, raw_beta, A_log, dt_bias = chunk_inputs
    q_ref, k_ref, v_ref, raw_gate_ref, raw_beta_ref, A_log_ref, dt_bias_ref = ref_inputs

    actual, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=raw_gate,
        beta=raw_beta.float().sigmoid(),
        A_log=A_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
    )
    gate_ref = naive_kda_gate(raw_gate_ref, A_log_ref, dt_bias_ref)
    expected, _ = naive_recurrent_kda(
        q=F.normalize(q_ref.float(), p=2, dim=-1).to(dtype),
        k=F.normalize(k_ref.float(), p=2, dim=-1).to(dtype),
        v=v_ref,
        g=gate_ref,
        beta=raw_beta_ref.float().sigmoid(),
    )
    _assert_close("reference output", actual, expected)

    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    for name, actual_input, expected_input in zip(
        ("dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt_bias"),
        chunk_inputs,
        ref_inputs,
        strict=True,
    ):
        assert actual_input.grad is not None and expected_input.grad is not None
        _assert_close(name, actual_input.grad, expected_input.grad)


def validate_packed_documents() -> None:
    torch.manual_seed(2027)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    length, heads, dim = 128, 2, 32
    q = torch.randn(1, length, heads, dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.randn(1, length, heads, device=device, dtype=dtype).float().sigmoid()
    A_log = torch.log(torch.empty(heads, device=device, dtype=torch.float32).uniform_(1, 16))
    dt_bias = torch.zeros(heads * dim, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 64, 128], device=device, dtype=torch.int32)

    packed, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    pieces = []
    for start, end in ((0, 64), (64, 128)):
        piece, _ = chunk_kda(
            q=q[:, start:end],
            k=k[:, start:end],
            v=v[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
        )
        pieces.append(piece)
    _assert_close("packed-document output", packed, torch.cat(pieces, dim=1))


if __name__ == "__main__":
    validate_reference_forward_backward()
    validate_packed_documents()
    print("KDA reference validation passed")
