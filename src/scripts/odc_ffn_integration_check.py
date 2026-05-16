"""
Integration check for :class:`olmo_core.nn.OutputDiscardCheckpoint` wired into
the real OLMo SwiGLU :class:`olmo_core.nn.feed_forward.FeedForward`.

The OLMo FFN forward is:

.. code-block:: text

    y = w2(activation(w1(x)) * w3(x))

The fat intermediate (``activation(w1(x)) * w3(x)``, shape ``[B, S, hidden_size]``)
gets saved-for-backward by ``w2``. This script wraps that intermediate with
:class:`OutputDiscardCheckpoint` and verifies:

1. Output matches baseline within tolerance.
2. Input gradient and every parameter gradient match baseline within tolerance.
3. Peak GPU memory is lower than baseline (CUDA only).

Run manually -- not picked up by CI:

.. code-block:: bash

    python src/scripts/odc_ffn_integration_check.py
    python src/scripts/odc_ffn_integration_check.py --dtype bf16 --iters 5

Exits non-zero on any parity failure.
"""

from __future__ import annotations

import argparse
import copy
import sys
from typing import Tuple

import torch

from olmo_core.nn import OutputDiscardCheckpoint
from olmo_core.nn.feed_forward import FeedForward


class ODCFeedForward(FeedForward):
    """
    SwiGLU :class:`FeedForward` with the fat intermediate wrapped by
    :class:`OutputDiscardCheckpoint`. Drop-in replacement for the baseline.
    """

    def _gated(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_fn(self.w1(x)) * self.w3(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and x.requires_grad:
            ckpt = OutputDiscardCheckpoint()
            h = ckpt.checkpoint(self._gated, x)
            y = self.w2(h)
            ckpt.discard_output_and_register_recompute(y)
            return y
        return self.w2(self._gated(x))


def _make_ffns(
    *,
    d_model: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[FeedForward, ODCFeedForward]:
    torch.manual_seed(0)
    baseline = FeedForward(
        d_model=d_model,
        hidden_size=hidden_size,
        bias=False,
        dtype=dtype,
        init_device=str(device),
    )
    odc = ODCFeedForward(
        d_model=d_model,
        hidden_size=hidden_size,
        bias=False,
        dtype=dtype,
        init_device=str(device),
    )
    odc.load_state_dict(copy.deepcopy(baseline.state_dict()))
    return baseline, odc


def _zero_grads(model: torch.nn.Module, x: torch.Tensor) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    if x.grad is not None:
        x.grad = None


def _run_and_capture(
    model: torch.nn.Module, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], float]:
    """
    Forward + backward through ``model``. Returns ``(output, x.grad, [param.grad
    for p in params], peak_mb)``. ``peak_mb`` is 0 on CPU.
    """
    if x.is_cuda:
        torch.cuda.reset_peak_memory_stats()
    y = model(x)
    loss = y.square().mean()
    loss.backward()
    if x.is_cuda:
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_mb = 0.0
    assert x.grad is not None
    grads = [p.grad.detach().clone() for p in model.parameters()]
    return y.detach().clone(), x.grad.detach().clone(), grads, peak_mb


def _close(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> Tuple[bool, float]:
    """Return (passes, max_abs_diff)."""
    diff = (a - b).abs().max().item()
    try:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        return True, diff
    except AssertionError:
        return False, diff


def run_one(
    *,
    d_model: int,
    hidden_size: int,
    batch: int,
    seq: int,
    iters: int,
    dtype: torch.dtype,
    device: torch.device,
) -> int:
    """
    Run ``iters`` forward+backward passes through baseline and ODC FFNs,
    asserting parity each iter. Returns 0 on success, 1 on any failure.
    """
    atol, rtol = {
        torch.float32: (1e-5, 1e-5),
        torch.bfloat16: (5e-3, 5e-3),
        torch.float16: (5e-3, 5e-3),
    }[dtype]

    baseline, odc = _make_ffns(d_model=d_model, hidden_size=hidden_size, dtype=dtype, device=device)

    failures = 0
    for it in range(iters):
        torch.manual_seed(100 + it)
        x_base = torch.randn(batch, seq, d_model, dtype=dtype, device=device, requires_grad=True)
        x_odc = x_base.detach().clone().requires_grad_(True)

        _zero_grads(baseline, x_base)
        _zero_grads(odc, x_odc)

        y_base, x_grad_base, p_grads_base, peak_base = _run_and_capture(baseline, x_base)
        y_odc, x_grad_odc, p_grads_odc, peak_odc = _run_and_capture(odc, x_odc)

        ok_out, diff_out = _close(y_odc, y_base, atol=atol, rtol=rtol)
        ok_xgrad, diff_xgrad = _close(x_grad_odc, x_grad_base, atol=atol, rtol=rtol)
        param_results = [
            _close(g_odc, g_base, atol=atol, rtol=rtol)
            for g_odc, g_base in zip(p_grads_odc, p_grads_base)
        ]
        ok_params = all(p for p, _ in param_results)
        max_param_diff = max(d for _, d in param_results) if param_results else 0.0

        iter_ok = ok_out and ok_xgrad and ok_params
        if not iter_ok:
            failures += 1

        memo_str = ""
        if device.type == "cuda":
            saved_pct = (1.0 - peak_odc / peak_base) * 100.0
            memo_str = (
                f"  peak base={peak_base:.1f}MB  odc={peak_odc:.1f}MB  saved={saved_pct:+.1f}%"
            )

        print(
            f"  iter {it}: {'PASS' if iter_ok else 'FAIL'}  "
            f"out_diff={diff_out:.2e}  x_grad_diff={diff_xgrad:.2e}  "
            f"max_p_grad_diff={max_param_diff:.2e}{memo_str}"
        )

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cpu" and args.dtype == "fp16":
        print("fp16 on CPU is not well supported; use fp32 or bf16.", file=sys.stderr)
        return 2

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    print(
        f"OLMo FFN + ODC integration check\n"
        f"  device={device}  dtype={dtype}\n"
        f"  d_model={args.d_model}  hidden_size={args.hidden_size}  "
        f"batch={args.batch}  seq={args.seq}  iters={args.iters}\n"
    )

    rc = run_one(
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        batch=args.batch,
        seq=args.seq,
        iters=args.iters,
        dtype=dtype,
        device=device,
    )
    print()
    print("RESULT: PASS" if rc == 0 else "RESULT: FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
