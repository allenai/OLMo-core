#!/usr/bin/env python3
"""Compare an exact failing GDN activation with a sequential FP32 reference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from olmo_core.nn.attention.gdn2 import GatedDeltaNet2
from olmo_core.nn.attention.recurrent import GatedDeltaNet


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


def token_repetition_summary(token_ids: torch.Tensor) -> dict[str, Any]:
    """Summarize repetition in the exact sequence that drove the bad activation."""
    tokens = token_ids.detach().cpu().reshape(-1).to(torch.int64)
    length = int(tokens.numel())
    values, counts = torch.unique(tokens, return_counts=True)
    order = torch.argsort(counts, descending=True)
    top_tokens = [
        {
            "token_id": int(values[idx].item()),
            "count": int(counts[idx].item()),
            "fraction": float(counts[idx].item() / max(length, 1)),
        }
        for idx in order[:10]
    ]
    probabilities = counts.double() / max(length, 1)
    entropy_bits = float(-(probabilities * probabilities.log2()).sum().item())

    max_lag = min(2_048, length - 1)
    lag_matches: list[tuple[int, int, float]] = []
    for lag in range(1, max_lag + 1):
        matches = int((tokens[lag:] == tokens[:-lag]).sum().item())
        lag_matches.append((lag, matches, matches / (length - lag)))
    strongest = sorted(lag_matches, key=lambda item: (item[2], item[1]), reverse=True)[:10]

    longest_run = 0
    if length:
        boundaries = torch.cat(
            [
                torch.tensor([True]),
                tokens[1:] != tokens[:-1],
                torch.tensor([True]),
            ]
        ).nonzero().flatten()
        longest_run = int((boundaries[1:] - boundaries[:-1]).max().item())

    by_lag = {lag: (matches, fraction) for lag, matches, fraction in lag_matches}
    selected_lags = {}
    for lag in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048):
        if lag in by_lag:
            matches, fraction = by_lag[lag]
            selected_lags[str(lag)] = {"matches": matches, "fraction": fraction}
    return {
        "length": length,
        "unique_tokens": int(values.numel()),
        "entropy_bits": entropy_bits,
        "perplexity_from_unigram_entropy": float(math.exp2(entropy_bits)),
        "longest_identical_run": longest_run,
        "top_tokens": top_tokens,
        "selected_lag_matches": selected_lags,
        "strongest_lag_matches": [
            {"lag": lag, "matches": matches, "fraction": fraction}
            for lag, matches, fraction in strongest
        ],
    }


def find_local_loss_capture(
    activation_capture: Path, payload: dict[str, Any], explicit: Path | None
) -> Path | None:
    if explicit is not None:
        return explicit
    pattern = f"step{int(payload['step']):06d}_mb*_local_loss.pt"
    candidates = sorted(activation_capture.parent.glob(pattern))
    return candidates[0] if len(candidates) == 1 else None


def _dtype(config: dict[str, Any]) -> torch.dtype:
    dtype_name = str(config["dtype"])
    if dtype_name == "torch.bfloat16":
        return torch.bfloat16
    elif dtype_name == "torch.float32":
        return torch.float32
    raise ValueError(f"unsupported captured GDN dtype {dtype_name}")


def build_module(
    payload: dict[str, Any], device: torch.device
) -> GatedDeltaNet | GatedDeltaNet2:
    module_type = payload["module_type"]
    if module_type == "GatedDeltaNet2":
        config = payload["gdn2_config"]
        module: GatedDeltaNet | GatedDeltaNet2 = GatedDeltaNet2(
            d_model=int(config["d_model"]),
            n_heads=int(config["n_heads"]),
            n_v_heads=int(config["n_v_heads"]),
            head_dim=int(config["head_dim"]),
            expand_v=float(config["expand_v"]),
            allow_neg_eigval=bool(config["allow_neg_eigval"]),
            conv_size=int(config["conv_size"]),
            disable_recompute=bool(config["disable_recompute"]),
            dtype=_dtype(config),
            init_device="cpu",
        )
    elif module_type == "GatedDeltaNet":
        config = payload["gdn1_config"]
        module = GatedDeltaNet(
            d_model=int(config["d_model"]),
            n_heads=int(config["n_heads"]),
            n_v_heads=int(config["n_v_heads"]),
            head_dim=int(config["head_dim"]),
            expand_v=float(config["expand_v"]),
            allow_neg_eigval=bool(config["allow_neg_eigval"]),
            conv_size=int(config["conv_size"]),
            dtype=_dtype(config),
            init_device="cpu",
        )
    else:
        raise ValueError(f"capture is from {module_type}, not a supported GDN boundary")
    module.load_state_dict(payload["module_state"], strict=True)
    return module.to(device).eval()


@torch.inference_mode()
def gdn2_recurrent_inputs(
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
def gdn2_post_process(
    module: GatedDeltaNet2, x: torch.Tensor, recurrent: torch.Tensor
) -> torch.Tensor:
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


@torch.inference_mode()
def gdn1_recurrent_inputs(
    module: GatedDeltaNet,
    x: torch.Tensor,
    cu_doc_lens: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    batch_size, seq_len, _ = x.shape
    q = module.q_conv1d(x=module.w_q(x), cu_seqlens=cu_doc_lens)
    k = module.k_conv1d(x=module.w_k(x), cu_seqlens=cu_doc_lens)
    v = module.v_conv1d(x=module.w_v(x), cu_seqlens=cu_doc_lens)
    beta = module.w_b(x).sigmoid()
    if module.allow_neg_eigval:
        beta = beta * 2.0
    g = -module.A_log.float().exp() * F.softplus(module.w_a(x).float() + module.dt_bias)
    q = q.view(batch_size, seq_len, module.n_heads, module.head_k_dim)
    k = k.view(batch_size, seq_len, module.n_heads, module.head_k_dim)
    v = v.view(batch_size, seq_len, module.n_v_heads, module.head_v_dim)
    if module.n_v_heads > module.n_heads:
        repeat_factor = module.n_v_heads // module.n_heads
        q = q.repeat_interleave(repeat_factor, dim=-2)
        k = k.repeat_interleave(repeat_factor, dim=-2)
    return q, k, v, g, beta


@torch.inference_mode()
def sequential_gdn1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.shape[0] != 1:
        raise ValueError("captured recurrent comparisons require one selected sequence")
    qf = q.float() / torch.sqrt(q.float().square().sum(dim=-1, keepdim=True) + 1e-6)
    kf = k.float() / torch.sqrt(k.float().square().sum(dim=-1, keepdim=True) + 1e-6)
    vf, gf, betaf = v.float(), g.float(), beta.float()
    _, seq_len, n_heads, key_dim = q.shape
    value_dim = v.shape[-1]
    state = torch.zeros(n_heads, key_dim, value_dim, device=q.device, dtype=torch.float32)
    outputs = []
    scale = key_dim**-0.5
    for token in range(seq_len):
        state = state * gf[0, token, :, None, None].exp()
        prediction = torch.einsum("hkv,hk->hv", state, kf[0, token])
        delta = betaf[0, token, :, None] * (vf[0, token] - prediction)
        state = state + torch.einsum("hk,hv->hkv", kf[0, token], delta)
        outputs.append(torch.einsum("hkv,hk->hv", state, qf[0, token] * scale))
    return torch.stack(outputs, dim=0).unsqueeze(0), state.unsqueeze(0)


@torch.inference_mode()
def gdn1_post_process(
    module: GatedDeltaNet, x: torch.Tensor, recurrent: torch.Tensor
) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape
    output_gate = module.w_g(x).view(
        batch_size, seq_len, module.n_v_heads, module.head_v_dim
    )
    return module.w_out(
        module.o_norm(recurrent.to(output_gate.dtype), output_gate).view(
            batch_size, seq_len, -1
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--local-loss-capture",
        type=Path,
        help="matching local-loss dump containing the exact token IDs (auto-detected when unique)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the exact GDN2 comparison requires a CUDA device")
    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    if payload.get("module_type") not in {"GatedDeltaNet", "GatedDeltaNet2"}:
        raise ValueError(f"capture is from unsupported boundary {payload.get('module_type')}")
    if payload.get("phase") != "forward":
        raise ValueError(f"expected a forward capture, found {payload.get('phase')}")

    device = torch.device("cuda")
    module = build_module(payload, device)
    x = payload["module_input"].to(device)
    cu_doc_lens = payload.get("cu_doc_lens")
    if cu_doc_lens is not None:
        cu_doc_lens = cu_doc_lens.to(device)
    saved_output = payload["bad_output"].to(device)
    if isinstance(module, GatedDeltaNet2):
        q, k, v, g, b, w = gdn2_recurrent_inputs(module, x, cu_doc_lens)
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
        actual_output = gdn2_post_process(module, x, actual_raw)
        expected_output = gdn2_post_process(module, x, expected_raw)
        recurrent_tensors = {"b": b, "w": w}
        recurrent_config = payload["gdn2_config"]
    else:
        if cu_doc_lens is not None:
            raise ValueError("GDN1 sequential capture analysis does not yet support packed resets")
        q, k, v, g, beta = gdn1_recurrent_inputs(module, x, cu_doc_lens)
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        actual_raw, actual_state = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        expected_raw, expected_state = sequential_gdn1(q, k, v, g, beta)
        actual_output = gdn1_post_process(module, x, actual_raw)
        expected_output = gdn1_post_process(module, x, expected_raw)
        recurrent_tensors = {"beta": beta}
        recurrent_config = payload["gdn1_config"]
    assert actual_state is not None and expected_state is not None
    torch.cuda.synchronize()

    tensors = {
        "input": x,
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        **recurrent_tensors,
        "captured_output": saved_output,
        "chunk_raw": actual_raw,
        "reference_raw": expected_raw,
        "chunk_state": actual_state,
        "reference_state": expected_state,
        "chunk_output": actual_output,
        "reference_output": expected_output,
    }
    local_loss_capture = find_local_loss_capture(
        args.capture, payload, args.local_loss_capture
    )
    exact_sequence = None
    token_stats = None
    if local_loss_capture is not None:
        local_loss = torch.load(local_loss_capture, map_location="cpu", weights_only=False)
        input_ids = local_loss.get("input_ids")
        bad_batch_idx = int(payload["bad_batch_idx"])
        if isinstance(input_ids, torch.Tensor) and bad_batch_idx < input_ids.shape[0]:
            exact_sequence = input_ids[bad_batch_idx]
            token_stats = token_repetition_summary(exact_sequence)
    result = {
        "capture": str(args.capture),
        "rank": payload["rank"],
        "step": payload["step"],
        "module_name": payload["module_name"],
        "module_type": payload["module_type"],
        "recurrent_config": recurrent_config,
        "dataset_metadata": (
            payload.get("batch", {}).get("metadata", [])[int(payload["bad_batch_idx"])]
            if isinstance(payload.get("batch"), dict)
            and int(payload["bad_batch_idx"])
            < len(payload.get("batch", {}).get("metadata", []))
            else None
        ),
        "local_loss_capture": (
            str(local_loss_capture) if local_loss_capture is not None else None
        ),
        "token_repetition": token_stats,
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
