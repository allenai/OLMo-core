#!/usr/bin/env python3
"""Dump the learned per-layer landmark gate temperature from a training checkpoint.

:class:`~olmo_core.nn.attention.landmark_compressive.FastCompressiveLandmarkAttention` with
``gate_temperature=True`` holds one scalar parameter ``log_gate_temp`` per layer, and the gate
(landmark-selection) softmax is taken over

    gate_logits = (q @ k.T) * softmax_scale * exp(-log_gate_temp)

i.e. the gate logits are the ordinary attention logits divided by a temperature
``T = exp(log_gate_temp)``. ``T > 1`` flattens that layer's gate (more landmarks share the mass),
``T < 1`` sharpens it. ``log_gate_temp`` is initialized to 0 (``T = 1``, gate == attention softmax),
so the printed values are exactly how far SFT moved each layer off that init.

This reads the parameters straight out of the distributed checkpoint -- no model build, no forward
pass, CPU only.

Usage::

    python analysis/gate_scores/dump_gate_temp.py /weka/.../step8550
"""
import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys  # noqa: E402


def _model_dir(path: str) -> str:
    """Accept either a step dir or the inner ``model_and_optim`` dir."""
    if os.path.basename(path.rstrip("/")) == "model_and_optim":
        return path
    inner = os.path.join(path, "model_and_optim")
    return inner if os.path.isdir(inner) else path


def _layer_idx(key: str) -> int:
    m = re.search(r"blocks\.(\d+)\.", key)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="path to a step dir (or its model_and_optim subdir)")
    ap.add_argument(
        "--pattern",
        default="log_gate_temp",
        help="substring the checkpoint keys must contain (default: log_gate_temp)",
    )
    ap.add_argument(
        "--include-optim",
        action="store_true",
        help="also dump the matching optimizer-state entries (exp_avg etc.), off by default",
    )
    args = ap.parse_args()

    ckpt = _model_dir(args.checkpoint)
    print(f"checkpoint: {ckpt}", flush=True)

    # Config context (softmax_scale / block size / gate_temperature flag), if the run saved one.
    for cfg_name in ("config.json", os.path.join("..", "config.json")):
        cfg_path = os.path.join(ckpt, cfg_name)
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                blob = f.read()
            for field in ("gate_temperature", "mem_freq", "block_size", "softmax_scale", "top_k"):
                for m in re.finditer(rf'"{field}":\s*([^,}}\n]+)', blob):
                    print(f"config: {field} = {m.group(1).strip()}")
            break

    md = get_checkpoint_metadata(ckpt)
    keys = sorted(
        (
            k
            for k in md.state_dict_metadata
            if args.pattern in k and (args.include_optim or not k.startswith("optim"))
        ),
        key=lambda k: (_layer_idx(k), k),
    )
    if not keys:
        print(f"NO KEYS matching {args.pattern!r}. Sample of model keys:")
        for k in list(md.state_dict_metadata)[:40]:
            print("   ", k)
        sys.exit(1)

    print(f"\nfound {len(keys)} '{args.pattern}' tensors\n", flush=True)
    print(f"{'layer':>5}  {'log_gate_temp':>14}  {'T=exp(log)':>11}  {'1/T':>8}  key")
    print("-" * 92)

    rows = []
    for key, tensor in zip(keys, load_keys(ckpt, keys)):
        vals = tensor.float().flatten().tolist()
        for i, v in enumerate(vals):
            layer = _layer_idx(key)
            t = math.exp(v)
            label = f"{layer}" if len(vals) == 1 else f"{layer}[{i}]"
            print(f"{label:>5}  {v:>14.6f}  {t:>11.6f}  {1.0 / t:>8.4f}  {key}")
            rows.append({"layer": layer, "idx": i, "key": key, "log_gate_temp": v, "temp": t})

    temps = [r["temp"] for r in rows]
    print("-" * 92)
    print(
        f"summary: n={len(temps)}  min T={min(temps):.4f}  max T={max(temps):.4f}  "
        f"mean T={sum(temps) / len(temps):.4f}"
    )
    print("(T > 1 => flatter gate softmax than attention; T < 1 => sharper; init was T = 1)")
    print("\nJSON " + json.dumps(rows))


if __name__ == "__main__":
    main()
