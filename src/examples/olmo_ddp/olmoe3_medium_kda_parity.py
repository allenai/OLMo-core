"""Single-GPU FLA/CuTe diagnostic at exact small and medium microbatch head shapes.

Only the CTA eligibility threshold changes. Both paths receive the same tensors,
negative-eigenvalue beta parameterization, fused QK norm, and fused gate inputs.
No optimizer updates, trainer, EP collectives, or model checkpoints are involved.
"""

import json
import os
from pathlib import Path

import torch


def error(left, right):
    """Report full-tensor forward/input-gradient differences, not sampled errors."""
    # Float32 reductions over tens of millions of elements can report cosine
    # above one even for A/A. Accumulate the diagnostic in FP64.
    x, y = left.double().flatten(), right.double().flatten()
    norm = x.norm().clamp_min(1e-30)
    return {
        "relative_l2": float((x - y).norm() / norm),
        "cosine": float(torch.dot(x, y) / (norm * y.norm().clamp_min(1e-30))),
        "max_abs": float((x - y).abs().max()),
        "finite": bool(torch.isfinite(y).all()),
    }


def main():
    """Run A/B/A for both current production microbatch shapes with real gate activation."""
    from kernel_fun._common import support
    from kernel_fun.kda import chunk_kda, is_supported

    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    root = Path(os.environ.get("RESULTS_DIR", "/results"))
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for batch, heads, label in [(2, 16, "medium-mb2"), (4, 8, "small-mb4")]:
        for strength in (0.1, 0.8):
            torch.manual_seed(12536)
            shape = (batch, 8192, heads, 128)
            q = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn_like(q, requires_grad=True)
            v = torch.randn(
                batch, 8192, heads, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            g = (torch.randn_like(q) * strength).requires_grad_()
            beta = (2 * torch.randn(batch, 8192, heads, device="cuda").sigmoid()).requires_grad_()
            a = torch.empty(heads, device="cuda").uniform_(1, 16).log().requires_grad_()
            dt = torch.zeros(heads * 128, device="cuda", requires_grad=True)
            dy = torch.randn_like(v)
            inputs = (q, k, v, g, beta, a, dt)
            kwargs = dict(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=a,
                dt_bias=dt,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
            )

            def execute(floor):
                support.MIN_CTAS = floor
                supported, reason = is_supported(
                    q, v, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True
                )
                assert supported == (floor == 128), (floor, supported, reason)
                y, _ = chunk_kda(**kwargs)
                gradients = torch.autograd.grad(y, inputs, dy)
                torch.cuda.synchronize()
                return (y.detach(), *(t.detach() for t in gradients))

            reference = execute(256)
            for floor, arm in [(128, "cute-vs-fla"), (256, "fla-aa")]:
                actual = execute(floor)
                row = {
                    "model_shape": label,
                    "raw_gate_std": strength,
                    "comparison": arm,
                    "errors": {
                        name: error(x, y)
                        for name, x, y in zip(
                            ("output", "dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt_bias"),
                            reference,
                            actual,
                        )
                    },
                }
                results.append(row)
                print("MEDIUM_KDA_PARITY", json.dumps(row), flush=True)
                (root / "medium-kda-parity.json").write_text(json.dumps(results, indent=2))
                del actual
            del reference, inputs, kwargs, q, k, v, g, beta, a, dt, dy
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
