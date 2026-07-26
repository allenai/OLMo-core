#!/usr/bin/env python3
"""Compare an exact failing GDN2 activation with FLA's recurrent reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from olmo_core.nn.attention.gdn2 import GatedDeltaNet2


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach()
    finite = torch.isfinite(tensor)
    all_finite = bool(finite.all().item())
    finite_values = tensor[finite]
    first_bad_flat: int | None = None
    first_bad_token: int | None = None
    if not all_finite:
        first_bad_flat = int((~finite).reshape(-1).nonzero()[0].item())
        if tensor.ndim >= 2:
            per_token = finite.reshape(tensor.shape[0], tensor.shape[1], -1).all(dim=-1)
            first_bad_token = int((~per_token).nonzero()[0, 1].item())
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "all_finite": all_finite,
        "nan_count": int(torch.isnan(tensor).sum().item()),
        "posinf_count": int(torch.isposinf(tensor).sum().item()),
        "neginf_count": int(torch.isneginf(tensor).sum().item()),
        "finite_abs_max": (
            float(finite_values.abs().max().item()) if finite_values.numel() else None
        ),
        "finite_mean": (
            float(finite_values.float().mean().item()) if finite_values.numel() else None
        ),
        "first_bad_flat": first_bad_flat,
        "first_bad_token": first_bad_token,
    }


def difference_summary(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    if not finite.any():
        return {"finite_overlap": 0, "max_abs": None, "relative_l2": None, "cosine": None}
    actual_f = actual[finite].float()
    expected_f = expected[finite].float()
    delta = actual_f - expected_f
    actual_norm = torch.linalg.vector_norm(actual_f)
    expected_norm = torch.linalg.vector_norm(expected_f)
    return {
        "finite_overlap": int(finite.sum().item()),
        "max_abs": float(delta.abs().max().item()),
        "relative_l2": float(
            (torch.linalg.vector_norm(delta) / expected_norm.clamp_min(1e-12)).item()
        ),
        "cosine": float(
            (
                torch.dot(actual_f, expected_f) / (actual_norm * expected_norm).clamp_min(1e-12)
            ).item()
        ),
    }


def build_module(payload: dict[str, Any], device: torch.device) -> GatedDeltaNet2:
    config = payload["gdn2_config"]
    dtype_name = str(config["dtype"])
    if dtype_name == "torch.bfloat16":
        dtype = torch.bfloat16
    elif dtype_name == "torch.float32":
        dtype = torch.float32
    else:
        raise ValueError(f"unsupported captured GDN2 dtype {dtype_name}")
    module = GatedDeltaNet2(
        d_model=int(config["d_model"]),
        n_heads=int(config["n_heads"]),
        n_v_heads=int(config["n_v_heads"]),
        head_dim=int(config["head_dim"]),
        expand_v=float(config["expand_v"]),
        allow_neg_eigval=bool(config["allow_neg_eigval"]),
        conv_size=int(config["conv_size"]),
        disable_recompute=bool(config["disable_recompute"]),
        dtype=dtype,
        init_device="cpu",
    )
    module.load_state_dict(payload["module_state"], strict=True)
    return module.to(device).eval()


@torch.inference_mode()
def recurrent_inputs(
    module: GatedDeltaNet2,
    x: torch.Tensor,
    cu_doc_lens: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    batch_size, seq_len, _ = x.shape
    q = module.q_conv1d(x=module.w_q(x), cu_seqlens=cu_doc_lens)
    k = module.k_conv1d(x=module.w_k(x), cu_seqlens=cu_doc_lens)
    v = module.v_conv1d(x=module.w_v(x), cu_seqlens=cu_doc_lens)
    g = F.softplus(module.f_proj_2(module.f_proj_1(x)).float() + module.dt_bias)
    b = module.w_b(x).sigmoid()
    w = module.w_w(x).sigmoid()
    q = q.view(batch_size, seq_len, module.n_heads, module.head_k_dim)
    k = k.view(batch_size, seq_len, module.n_heads, module.head_k_dim)
    g = g.view(batch_size, seq_len, module.n_heads, module.head_k_dim)
    b = b.view(batch_size, seq_len, module.n_heads, module.head_k_dim)
    v = v.view(batch_size, seq_len, module.n_v_heads, module.head_v_dim)
    w = w.view(batch_size, seq_len, module.n_v_heads, module.head_v_dim)
    g = -module.A_log.float().exp().view(1, 1, module.n_heads, 1) * g
    if module.n_v_heads > module.n_heads:
        repeat_factor = module.n_v_heads // module.n_heads
        q = q.repeat_interleave(repeat_factor, dim=-2)
        k = k.repeat_interleave(repeat_factor, dim=-2)
        g = g.repeat_interleave(repeat_factor, dim=-2)
        b = b.repeat_interleave(repeat_factor, dim=-2)
    if module.allow_neg_eigval:
        b = b * 2.0
    return q, k, v, g, b, w


@torch.inference_mode()
def post_process(module: GatedDeltaNet2, x: torch.Tensor, recurrent: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape
    output_gate = module.g_proj_2(module.g_proj_1(x)).view(
        batch_size, seq_len, module.n_v_heads, module.head_v_dim
    )
    # FLA's token-by-token reference intentionally retains its recurrent output
    # in FP32, whereas the production chunk op returns the value-projection
    # dtype. Preserve FP32 for the raw/state comparisons below, but cast at the
    # same normalization/output-projection boundary used by the trained module.
    recurrent_for_projection = recurrent.to(output_gate.dtype)
    return module.w_out(
        module.o_norm(recurrent_for_projection, output_gate).view(batch_size, seq_len, -1)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the exact GDN2 comparison requires a CUDA device")
    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    if payload.get("module_type") != "GatedDeltaNet2":
        raise ValueError(
            f"capture is from {payload.get('module_type')}, not a GatedDeltaNet2 boundary"
        )
    if payload.get("phase") != "forward":
        raise ValueError(f"expected a forward capture, found {payload.get('phase')}")

    device = torch.device("cuda")
    module = build_module(payload, device)
    x = payload["module_input"].to(device)
    cu_doc_lens = payload.get("cu_doc_lens")
    if cu_doc_lens is not None:
        cu_doc_lens = cu_doc_lens.to(device)
    saved_output = payload["bad_output"].to(device)
    q, k, v, g, b, w = recurrent_inputs(module, x, cu_doc_lens)

    from fla.ops.gdn2 import chunk_gdn2
    from fla.ops.gdn2.naive import naive_recurrent_gdn2

    actual_raw, actual_state = chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        disable_recompute=module.disable_recompute,
        cu_seqlens=cu_doc_lens,
    )
    expected_raw, expected_state = naive_recurrent_gdn2(
        q=F.normalize(q.float(), p=2, dim=-1).to(q.dtype),
        k=F.normalize(k.float(), p=2, dim=-1).to(k.dtype),
        v=v,
        g=g,
        b=b,
        w=w,
        output_final_state=True,
    )
    assert actual_state is not None and expected_state is not None
    actual_output = post_process(module, x, actual_raw)
    expected_output = post_process(module, x, expected_raw)
    torch.cuda.synchronize()

    tensors = {
        "input": x,
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "b": b,
        "w": w,
        "captured_output": saved_output,
        "chunk_raw": actual_raw,
        "reference_raw": expected_raw,
        "chunk_state": actual_state,
        "reference_state": expected_state,
        "chunk_output": actual_output,
        "reference_output": expected_output,
    }
    result = {
        "capture": str(args.capture),
        "rank": payload["rank"],
        "step": payload["step"],
        "module_name": payload["module_name"],
        "gdn2_config": payload["gdn2_config"],
        "summaries": {name: tensor_summary(tensor) for name, tensor in tensors.items()},
        "differences": {
            "captured_vs_recomputed_chunk": difference_summary(saved_output, actual_output),
            "chunk_vs_reference_raw": difference_summary(actual_raw, expected_raw),
            "chunk_vs_reference_state": difference_summary(actual_state, expected_state),
            "chunk_vs_reference_output": difference_summary(actual_output, expected_output),
        },
    }
    output = args.output or args.capture.with_name(args.capture.stem + "_reference.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
