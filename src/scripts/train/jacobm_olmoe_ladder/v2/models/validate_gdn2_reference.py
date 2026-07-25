#!/usr/bin/env python3
"""Validate the production GDN2 kernel against FLA's Torch recurrence."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from fla.ops.gdn2 import chunk_gdn2, naive_recurrent_gdn2


@dataclass(frozen=True)
class Case:
    expand_v: int
    allow_neg_eigval: bool

    @property
    def value_dim(self) -> int:
        return 128 * self.expand_v

    @property
    def name(self) -> str:
        eigval = "neg" if self.allow_neg_eigval else "noneg"
        return f"ev{self.expand_v}-{eigval}"


CASES = tuple(
    Case(expand_v=expand_v, allow_neg_eigval=allow_neg_eigval)
    for expand_v in (1, 2)
    for allow_neg_eigval in (False, True)
)

# These match or tighten the tolerances in FLA's own GDN2 tests. Differences
# are reported before the assertion so the job output remains useful.
TOLERANCES = {
    "output": 5e-3,
    "final_state": 5e-3,
    "dq": 1e-2,
    "dk": 1e-2,
    "dv": 1e-2,
    "draw_decay": 2e-2,
    "draw_erase": 2e-2,
    "draw_write": 2e-2,
    "dA_log": 2e-2,
    "ddt_bias": 2e-2,
    "dh0": 1e-2,
}


def _clone_grad(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(True)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    difference = (actual_float - expected_float).abs()
    denominator = expected_float.abs().clamp_min(1e-3)
    return (
        difference.max().item(),
        difference.mean().item(),
        (difference / denominator).max().item(),
    )


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(f"{name} contains a nonfinite value")
    max_abs, mean_abs, max_rel = _metrics(actual, expected)
    tolerance = TOLERANCES[name]
    print(
        f"  {name:>12}: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
        f"max_rel(|ref|>=1e-3)={max_rel:.6g} tol={tolerance:.3g}"
    )
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=tolerance,
        atol=tolerance,
    )


def _make_inputs(
    case: Case,
    *,
    length: int,
    seed: int,
) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, heads, key_dim = 1, 8, 128
    value_dim = case.value_dim
    return [
        torch.randn(batch, length, heads, key_dim, device=device, dtype=dtype),
        torch.randn(batch, length, heads, key_dim, device=device, dtype=dtype),
        torch.randn(batch, length, heads, value_dim, device=device, dtype=dtype) * 0.5,
        torch.randn(batch, length, heads, key_dim, device=device, dtype=dtype),
        torch.randn(batch, length, heads, key_dim, device=device, dtype=dtype),
        torch.randn(batch, length, heads, value_dim, device=device, dtype=dtype),
        torch.log(torch.empty(heads, device=device, dtype=torch.float32).uniform_(1, 16)),
        torch.empty(heads * key_dim, device=device, dtype=torch.float32).uniform_(-4.0, 0.0),
        torch.randn(batch, heads, key_dim, value_dim, device=device, dtype=torch.float32) * 0.01,
    ]


def _activate(
    inputs: list[torch.Tensor],
    case: Case,
) -> tuple[torch.Tensor, ...]:
    q, k, v, raw_decay, raw_erase, raw_write, A_log, dt_bias, h0 = inputs
    heads, key_dim = q.shape[-2:]
    decay = F.softplus(raw_decay.float() + dt_bias.view(1, 1, heads, key_dim))
    decay = -A_log.exp().view(1, 1, heads, 1) * decay
    erase = raw_erase.sigmoid()
    if case.allow_neg_eigval:
        erase = erase * 2.0
    write = raw_write.sigmoid()
    return q, k, v, decay, erase, write, h0


def validate_forward_backward(case: Case, *, disable_recompute: bool) -> None:
    base_inputs = _make_inputs(
        case,
        length=64,
        seed=20260725 + 100 * case.expand_v + int(case.allow_neg_eigval),
    )
    actual_inputs = [_clone_grad(tensor) for tensor in base_inputs]
    reference_inputs = [_clone_grad(tensor) for tensor in base_inputs]
    q, k, v, decay, erase, write, h0 = _activate(actual_inputs, case)
    q_ref, k_ref, v_ref, decay_ref, erase_ref, write_ref, h0_ref = _activate(
        reference_inputs, case
    )

    actual, actual_state = chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=decay,
        b=erase,
        w=write,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        disable_recompute=disable_recompute,
    )
    expected, expected_state = naive_recurrent_gdn2(
        q=F.normalize(q_ref.float(), p=2, dim=-1).to(q_ref.dtype),
        k=F.normalize(k_ref.float(), p=2, dim=-1).to(k_ref.dtype),
        v=v_ref,
        g=decay_ref,
        b=erase_ref,
        w=write_ref,
        initial_state=h0_ref,
        output_final_state=True,
    )
    assert actual_state is not None and expected_state is not None

    mode = "retain" if disable_recompute else "recompute"
    print(f"{case.name} mode={mode} K=128 V={case.value_dim}")
    _assert_close("output", actual, expected)
    _assert_close("final_state", actual_state, expected_state)

    output_weight = torch.randn_like(actual)
    state_weight = torch.randn_like(actual_state)
    ((actual * output_weight).sum() + (actual_state * state_weight).sum()).backward()
    ((expected * output_weight).sum() + (expected_state * state_weight).sum()).backward()
    for name, actual_input, expected_input in zip(
        (
            "dq",
            "dk",
            "dv",
            "draw_decay",
            "draw_erase",
            "draw_write",
            "dA_log",
            "ddt_bias",
            "dh0",
        ),
        actual_inputs,
        reference_inputs,
        strict=True,
    ):
        assert actual_input.grad is not None and expected_input.grad is not None
        _assert_close(name, actual_input.grad, expected_input.grad)


def validate_packed_documents(case: Case) -> None:
    base_inputs = _make_inputs(
        case,
        length=128,
        seed=20261725 + 100 * case.expand_v + int(case.allow_neg_eigval),
    )
    # Packed training does not carry initial states in our OLMo integration.
    base_inputs[-1].zero_()
    actual_inputs = [_clone_grad(tensor) for tensor in base_inputs[:-1]]
    reference_inputs = [_clone_grad(tensor) for tensor in base_inputs[:-1]]
    actual_with_h0 = actual_inputs + [_clone_grad(base_inputs[-1])]
    q, k, v, decay, erase, write, _ = _activate(actual_with_h0, case)

    cu_seqlens = torch.tensor([0, 64, 128], device=q.device, dtype=torch.int32)
    actual, _ = chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=decay,
        b=erase,
        w=write,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )

    ref_with_h0 = reference_inputs + [_clone_grad(base_inputs[-1])]
    q_ref, k_ref, v_ref, decay_ref, erase_ref, write_ref, _ = _activate(ref_with_h0, case)
    pieces = []
    for start, end in ((0, 64), (64, 128)):
        piece, _ = naive_recurrent_gdn2(
            q=F.normalize(q_ref[:, start:end].float(), p=2, dim=-1).to(q_ref.dtype),
            k=F.normalize(k_ref[:, start:end].float(), p=2, dim=-1).to(k_ref.dtype),
            v=v_ref[:, start:end],
            g=decay_ref[:, start:end],
            b=erase_ref[:, start:end],
            w=write_ref[:, start:end],
        )
        pieces.append(piece)
    expected = torch.cat(pieces, dim=1)

    print(f"{case.name} packed-documents K=128 V={case.value_dim}")
    _assert_close("output", actual, expected)
    output_weight = torch.randn_like(actual)
    (actual * output_weight).sum().backward()
    (expected * output_weight).sum().backward()
    for name, actual_input, expected_input in zip(
        ("dq", "dk", "dv", "draw_decay", "draw_erase", "draw_write", "dA_log", "ddt_bias"),
        actual_inputs,
        reference_inputs,
        strict=True,
    ):
        assert actual_input.grad is not None and expected_input.grad is not None
        _assert_close(name, actual_input.grad, expected_input.grad)


if __name__ == "__main__":
    print(f"device={torch.cuda.get_device_name()} torch={torch.__version__}")
    for candidate in CASES:
        validate_forward_backward(candidate, disable_recompute=False)
        validate_forward_backward(candidate, disable_recompute=True)
        validate_packed_documents(candidate)
    print("GDN2 reference validation passed for all four production settings")
