"""Replay captured KDA inputs across kernel calling conventions, without a model load."""

import argparse
import itertools
import json

import torch
from fla.ops.kda import chunk_kda
from safetensors import safe_open

from hero_hf_download import SCRATCH, write_json


def stats(actual, expected):
    """Compare all captured values, including exact equality."""
    if actual.shape != expected.shape:
        return {"actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    error = (actual.float() - expected.float()).abs()
    return dict(
        max_abs=error.max().item(),
        mean_abs=error.mean().item(),
        mismatches=int((error > 0).sum()),
        total=error.numel(),
    )


def main():
    """Run a diagnostic-only replay; never write acceptance or cleanup receipts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=2)
    args = parser.parse_args()
    root = SCRATCH / "emo" / "step6000"
    debug = root / "layer-debug-random257-flash3"
    hf = torch.load(debug / "hf-layer-debug.pt", map_location="cpu", weights_only=True, mmap=True)[
        "captures"
    ]
    native = torch.load(
        debug / "vllm-layer-debug.pt", map_location="cpu", weights_only=True, mmap=True
    )
    config = json.loads((root / "hf.partial/config.json").read_text())
    prefix = f"model.layers.{args.layer}.self_attn"
    for name, reference in hf.items():
        if name.startswith(prefix) and name in native:
            actual = native[name]
            if actual.ndim == reference.ndim + 1 and actual.shape[0] == 1:
                actual = actual[0]
            print("KDA_CAPTURE_COMPARE", name, json.dumps(stats(actual, reference)), flush=True)
    h, k, v = (
        config["linear_num_key_heads"],
        config["linear_key_head_dim"],
        config["linear_value_head_dim"],
    )
    q = hf[prefix + ".q_conv1d/out0"].reshape(1, -1, h, k).cuda()
    keys = hf[prefix + ".k_conv1d/out0"].reshape(1, -1, h, k).cuda()
    values = hf[prefix + ".v_conv1d/out0"].reshape(1, -1, h, v).cuda()
    gate = hf[prefix + ".f_proj_2/out0"].reshape(1, -1, h, k).cuda()
    beta = hf[prefix + ".beta_proj/out0"].unsqueeze(0).cuda().float().sigmoid()
    if config["linear_allow_neg_eigval"]:
        beta = beta * 2.0
    expected = hf[prefix + ".o_norm/in0"].unsqueeze(0).cuda()
    native_expected = native[prefix + ".o_norm/in0"].cuda()
    if native_expected.ndim == 3:
        native_expected = native_expected.unsqueeze(0)
    with safe_open(root / "hf.partial/model.safetensors", framework="pt", device="cpu") as handle:
        alog = handle.get_tensor(prefix + ".A_log")
        bias = handle.get_tensor(prefix + ".dt_bias")
    rows = []
    with torch.inference_mode():
        for value_first, variable, zero_state, dtype_name in itertools.product(
            (False, True), (False, True), (False, True), ("bfloat16", "float32")
        ):
            dtype = getattr(torch, dtype_name)
            initial = None
            if zero_state:
                dims = (v, k) if value_first else (k, v)
                initial = torch.zeros((1, h, *dims), device="cuda", dtype=torch.float32)
            output, _ = chunk_kda(
                q=q,
                k=keys,
                v=values,
                g=gate,
                beta=beta,
                A_log=alog.to(device="cuda", dtype=dtype),
                dt_bias=bias.to(device="cuda", dtype=dtype),
                initial_state=initial,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                state_v_first=value_first,
                cu_seqlens=(
                    torch.tensor([0, q.shape[1]], device="cuda", dtype=torch.int32)
                    if variable
                    else None
                ),
            )
            row = dict(
                value_first=value_first,
                variable_length=variable,
                zero_state=zero_state,
                parameter_dtype=dtype_name,
                versus_hf=stats(output, expected),
                versus_native=stats(output, native_expected),
            )
            rows.append(row)
            print("KDA_REPLAY", json.dumps(row), flush=True)
    write_json(
        debug / f"kda-replay-layer{args.layer}.json",
        dict(layer=args.layer, cases=rows, diagnostic_only=True),
    )


if __name__ == "__main__":
    main()
