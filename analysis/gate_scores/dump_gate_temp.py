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
pass, CPU only. It deliberately uses ``torch.distributed.checkpoint`` directly rather than
:mod:`olmo_core.distributed.checkpoint` so it runs in a bare image without the package installed
(the checkpoint is a local weka path, so no remote-filesystem reader is needed).

Usage::

    python analysis/gate_scores/dump_gate_temp.py /weka/.../step8550
"""
import argparse
import json
import math
import os
import re
import sys
from typing import Any, Dict, List

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict


def _load_keys(ckpt: str, keys: List[str]) -> Dict[str, Any]:
    """Load just ``keys`` (unsharded) out of a DCP checkpoint, single-process."""
    state_dict: Dict[str, Any] = {}
    _load_state_dict(
        state_dict,
        storage_reader=dcp.FileSystemReader(ckpt),
        planner=_EmptyStateDictLoadPlanner(keys=keys),
        no_dist=True,
    )
    return state_dict


def _get_key(state_dict: Dict[str, Any], key: str) -> Any:
    """Fetch a dotted key. The empty-planner load may return it flat or nested, so handle both."""
    if key in state_dict:
        return state_dict[key]
    root, rest = key.split(".", 1)
    return _get_key(state_dict[root], rest)


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
    for cfg_name in ("config.json", "../config.json", "../../config.json"):
        cfg_path = os.path.join(ckpt, cfg_name)
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                blob = f.read()
            for field in ("gate_temperature", "mem_freq", "block_size", "softmax_scale", "top_k"):
                for m in re.finditer(rf'"{field}":\s*([^,}}\n]+)', blob):
                    print(f"config: {field} = {m.group(1).strip()}")
            break

    md = dcp.FileSystemReader(ckpt).read_metadata()
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

    loaded = _load_keys(ckpt, keys)
    rows = []
    for key in keys:
        vals = _get_key(loaded, key).float().flatten().tolist()
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
