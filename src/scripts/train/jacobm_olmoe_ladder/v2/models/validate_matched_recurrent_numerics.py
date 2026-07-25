#!/usr/bin/env python3
"""Run an apples-to-apples KDA/GDN2 forward and backward numerical audit."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

TOLERANCES = {
    "output": 5e-3,
    "final_state": 5e-3,
    "q": 1e-2,
    "k": 1e-2,
    "v": 1e-2,
    "raw_decay": 2e-2,
    "raw_gate": 2e-2,
    "raw_erase": 2e-2,
    "raw_write": 2e-2,
    "A_log": 2e-2,
    "dt_bias": 2e-2,
    "h0": 1e-2,
}
GRADIENT_NAMES = {
    "q",
    "k",
    "v",
    "raw_decay",
    "raw_gate",
    "raw_erase",
    "raw_write",
    "A_log",
    "dt_bias",
    "h0",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixer", choices=("kda", "gdn2"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("KDA_JSON", "GDN2_JSON"))
    parser.add_argument("--markdown", type=Path)
    return parser


def _clone_leaves(base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in base.items()
        if name not in {"output_weight", "state_weight"}
    }


def _make_base(
    *,
    mixer: str,
    length: int,
    value_dim: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, heads, key_dim = 1, 2, 128
    base = {
        "q": torch.randn(batch, length, heads, key_dim, device=device, dtype=dtype),
        "k": torch.randn(batch, length, heads, key_dim, device=device, dtype=dtype),
        "v": torch.randn(batch, length, heads, value_dim, device=device, dtype=dtype) * 0.5,
        "raw_decay": torch.randn(
            batch, length, heads, key_dim, device=device, dtype=dtype
        ),
        "A_log": torch.log(
            torch.empty(heads, device=device, dtype=torch.float32).uniform_(1, 16)
        ),
        "dt_bias": torch.empty(
            heads * key_dim, device=device, dtype=torch.float32
        ).uniform_(-4.0, 0.0),
        "h0": torch.randn(
            batch, heads, key_dim, value_dim, device=device, dtype=torch.float32
        )
        * 0.01,
        "output_weight": torch.randn(
            batch, length, heads, value_dim, device=device, dtype=dtype
        ),
        "state_weight": torch.randn(
            batch, heads, key_dim, value_dim, device=device, dtype=torch.float32
        ),
    }
    # Draw mixer-specific gates after all common inputs and upstream gradients,
    # so the common tensors are identical across the two Python processes.
    if mixer == "kda":
        base["raw_gate"] = torch.randn(
            batch, length, heads, device=device, dtype=dtype
        )
    else:
        base["raw_erase"] = torch.randn(
            batch, length, heads, key_dim, device=device, dtype=dtype
        )
        base["raw_write"] = torch.randn(
            batch, length, heads, value_dim, device=device, dtype=dtype
        )
    return base


def _gdn2_decay(leaves: dict[str, torch.Tensor]) -> torch.Tensor:
    _, _, heads, key_dim = leaves["raw_decay"].shape
    decay = F.softplus(
        leaves["raw_decay"].float()
        + leaves["dt_bias"].view(1, 1, heads, key_dim)
    )
    return -leaves["A_log"].exp().view(1, 1, heads, 1) * decay


def _run_actual(
    *,
    mixer: str,
    leaves: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
    loss_mode: str,
    allow_neg_eigval: bool,
    disable_recompute: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mixer == "kda":
        from fla.ops.kda import chunk_kda

        beta = leaves["raw_gate"].float().sigmoid()
        output, state = chunk_kda(
            q=leaves["q"],
            k=leaves["k"],
            v=leaves["v"],
            g=leaves["raw_decay"],
            beta=beta,
            A_log=leaves["A_log"],
            dt_bias=leaves["dt_bias"],
            initial_state=leaves["h0"],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
        )
    else:
        from fla.ops.gdn2 import chunk_gdn2

        erase = leaves["raw_erase"].sigmoid()
        if allow_neg_eigval:
            erase = erase * 2.0
        output, state = chunk_gdn2(
            q=leaves["q"],
            k=leaves["k"],
            v=leaves["v"],
            g=_gdn2_decay(leaves),
            b=erase,
            w=leaves["raw_write"].sigmoid(),
            initial_state=leaves["h0"],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            disable_recompute=disable_recompute,
        )
    assert state is not None
    _backward(output, state, base, loss_mode)
    return output, state


def _run_reference(
    *,
    mixer: str,
    leaves: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
    loss_mode: str,
    allow_neg_eigval: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = F.normalize(leaves["q"].float(), p=2, dim=-1).to(leaves["q"].dtype)
    k = F.normalize(leaves["k"].float(), p=2, dim=-1).to(leaves["k"].dtype)
    if mixer == "kda":
        from fla.ops.kda.gate import naive_kda_gate
        from fla.ops.kda.naive import naive_recurrent_kda

        decay = naive_kda_gate(
            leaves["raw_decay"], leaves["A_log"], leaves["dt_bias"]
        )
        output, state = naive_recurrent_kda(
            q=q,
            k=k,
            v=leaves["v"],
            g=decay,
            beta=leaves["raw_gate"].float().sigmoid(),
            initial_state=leaves["h0"],
            output_final_state=True,
        )
    else:
        from fla.ops.gdn2 import naive_recurrent_gdn2

        erase = leaves["raw_erase"].sigmoid()
        if allow_neg_eigval:
            erase = erase * 2.0
        output, state = naive_recurrent_gdn2(
            q=q,
            k=k,
            v=leaves["v"],
            g=_gdn2_decay(leaves),
            b=erase,
            w=leaves["raw_write"].sigmoid(),
            initial_state=leaves["h0"],
            output_final_state=True,
        )
    assert state is not None
    _backward(output, state, base, loss_mode)
    return output, state


def _backward(
    output: torch.Tensor,
    state: torch.Tensor,
    base: dict[str, torch.Tensor],
    loss_mode: str,
) -> None:
    output_term = (output * base["output_weight"]).sum()
    state_term = (state * base["state_weight"]).sum()
    if loss_mode == "output":
        loss = output_term + 0.0 * state_term
    elif loss_mode == "state":
        loss = 0.0 * output_term + state_term
    else:
        raise ValueError(f"unknown loss mode: {loss_mode}")
    loss.backward()


def _snapshot(
    output: torch.Tensor,
    state: torch.Tensor,
    leaves: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    snapshot = {
        "output": output.detach().float().cpu(),
        "final_state": state.detach().float().cpu(),
    }
    for name, leaf in leaves.items():
        if leaf.grad is None:
            snapshot[name] = torch.zeros_like(leaf, dtype=torch.float32, device="cpu")
        else:
            snapshot[name] = leaf.grad.detach().float().cpu()
    return snapshot


def _metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    tolerance: float,
) -> dict[str, float | int | bool]:
    actual = actual.float().flatten()
    expected = expected.float().flatten()
    difference = (actual - expected).abs()
    reference_abs = expected.abs()
    reference_norm = torch.linalg.vector_norm(expected)
    difference_norm = torch.linalg.vector_norm(difference)
    actual_norm = torch.linalg.vector_norm(actual)
    denominator = actual_norm * reference_norm
    cosine = (
        torch.dot(actual, expected) / denominator
        if denominator.item() > 0
        else torch.tensor(1.0 if difference.max().item() == 0 else 0.0)
    )
    allowed = tolerance + tolerance * reference_abs
    violations = difference > allowed
    quantiles = torch.quantile(
        difference,
        torch.tensor([0.99, 0.999], dtype=difference.dtype),
    )
    return {
        "count": difference.numel(),
        "max_abs": difference.max().item(),
        "mean_abs": difference.mean().item(),
        "p99_abs": quantiles[0].item(),
        "p999_abs": quantiles[1].item(),
        "reference_max_abs": reference_abs.max().item(),
        "reference_l2": reference_norm.item(),
        "relative_l2": (
            difference_norm / reference_norm.clamp_min(torch.finfo(torch.float32).tiny)
        ).item(),
        "cosine": cosine.item(),
        "max_abs_over_reference_max": (
            difference.max() / reference_abs.max().clamp_min(torch.finfo(torch.float32).tiny)
        ).item(),
        "violation_count": violations.sum().item(),
        "violation_fraction": violations.float().mean().item(),
        "tolerance": tolerance,
        "passed": not violations.any().item(),
    }


def _run_case(
    *,
    mixer: str,
    length: int,
    value_dim: int,
    loss_mode: str,
    allow_neg_eigval: bool,
    disable_recompute: bool,
) -> dict[str, Any]:
    seed = 20260725 + length * 10 + value_dim + (1 if loss_mode == "state" else 0)
    base = _make_base(mixer=mixer, length=length, value_dim=value_dim, seed=seed)
    actual_leaves = _clone_leaves(base)
    started = time.perf_counter()
    actual_output, actual_state = _run_actual(
        mixer=mixer,
        leaves=actual_leaves,
        base=base,
        loss_mode=loss_mode,
        allow_neg_eigval=allow_neg_eigval,
        disable_recompute=disable_recompute,
    )
    torch.cuda.synchronize()
    actual_snapshot = _snapshot(actual_output, actual_state, actual_leaves)
    del actual_output, actual_state, actual_leaves
    torch.cuda.empty_cache()

    reference_leaves = _clone_leaves(base)
    reference_output, reference_state = _run_reference(
        mixer=mixer,
        leaves=reference_leaves,
        base=base,
        loss_mode=loss_mode,
        allow_neg_eigval=allow_neg_eigval,
    )
    torch.cuda.synchronize()
    reference_snapshot = _snapshot(reference_output, reference_state, reference_leaves)
    elapsed = time.perf_counter() - started
    del reference_output, reference_state, reference_leaves, base
    torch.cuda.empty_cache()

    components = {
        name: _metrics(actual_snapshot[name], expected, tolerance=TOLERANCES[name])
        for name, expected in reference_snapshot.items()
    }
    passed = all(bool(component["passed"]) for component in components.values())
    gradient_components = {
        name: component for name, component in components.items() if name in GRADIENT_NAMES
    }
    worst_gradient_name, worst_gradient = max(
        gradient_components.items(), key=lambda item: float(item[1]["relative_l2"])
    )
    print(
        f"{mixer:>4} T={length:<3} V={value_dim:<3} loss={loss_mode:<6} "
        f"neg={int(allow_neg_eigval)} retain={int(disable_recompute)} "
        f"output_max={components['output']['max_abs']:.3g} "
        f"worst_rel_l2={worst_gradient_name}:{worst_gradient['relative_l2']:.3g} "
        f"pass={passed} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return {
        "mixer": mixer,
        "length": length,
        "heads": 2,
        "key_dim": 128,
        "value_dim": value_dim,
        "loss_mode": loss_mode,
        "allow_neg_eigval": allow_neg_eigval,
        "disable_recompute": disable_recompute,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "passed": passed,
        "worst_gradient": worst_gradient_name,
        "components": components,
    }


def _run_suite(mixer: str, output_path: Path) -> None:
    print(
        f"mixer={mixer} fla={__import__('fla').__version__} "
        f"device={torch.cuda.get_device_name()} torch={torch.__version__}",
        flush=True,
    )
    results = []
    for length in (64, 256):
        for value_dim in (128, 256):
            for loss_mode in ("output", "state"):
                if mixer == "kda":
                    results.append(
                        _run_case(
                            mixer=mixer,
                            length=length,
                            value_dim=value_dim,
                            loss_mode=loss_mode,
                            allow_neg_eigval=False,
                            disable_recompute=False,
                        )
                    )
                else:
                    for allow_neg_eigval in (False, True):
                        for disable_recompute in (False, True):
                            results.append(
                                _run_case(
                                    mixer=mixer,
                                    length=length,
                                    value_dim=value_dim,
                                    loss_mode=loss_mode,
                                    allow_neg_eigval=allow_neg_eigval,
                                    disable_recompute=disable_recompute,
                                )
                            )
    payload = {
        "mixer": mixer,
        "fla_version": __import__("fla").__version__,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "results": results,
        "passed": all(result["passed"] for result in results),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output_path}; passed={payload['passed']}", flush=True)


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    return f"{value:.3e}"


def _compare(kda_path: Path, gdn2_path: Path, markdown_path: Path | None) -> None:
    payloads = [json.loads(path.read_text()) for path in (kda_path, gdn2_path)]
    lines = [
        "# Matched KDA/GDN2 numerical audit",
        "",
        (
            "Each row uses H=2, K=128, identical common random tensors and upstream "
            "gradients, and the mixer's own sequential PyTorch reference."
        ),
        "",
        "| Mixer | T | V | Loss | Negative eigvals | Retain intermediates | Output max abs | State max abs | Worst gradient rel-L2 | Gradient | Allclose |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for payload in payloads:
        for result in payload["results"]:
            components = result["components"]
            worst_name = result["worst_gradient"]
            lines.append(
                "| {mixer} | {length} | {value_dim} | {loss_mode} | {neg} | {retain} | {output} | {state} | {relative} | {gradient} | {passed} |".format(
                    mixer=result["mixer"].upper(),
                    length=result["length"],
                    value_dim=result["value_dim"],
                    loss_mode=result["loss_mode"],
                    neg=int(result["allow_neg_eigval"]),
                    retain=int(result["disable_recompute"]),
                    output=_format_float(float(components["output"]["max_abs"])),
                    state=_format_float(float(components["final_state"]["max_abs"])),
                    relative=_format_float(float(components[worst_name]["relative_l2"])),
                    gradient=worst_name,
                    passed="yes" if result["passed"] else "NO",
                )
            )
    lines.extend(
        [
            "",
            f"KDA FLA version: `{payloads[0]['fla_version']}`.",
            f"GDN2 FLA version: `{payloads[1]['fla_version']}`.",
            "",
        ]
    )
    report = "\n".join(lines)
    print(report, flush=True)
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(report)


def main() -> None:
    args = _parser().parse_args()
    if args.compare is not None:
        _compare(args.compare[0], args.compare[1], args.markdown)
        return
    if args.mixer is None or args.output is None:
        raise SystemExit("--mixer and --output are required unless --compare is used")
    _run_suite(args.mixer, args.output)


if __name__ == "__main__":
    main()
