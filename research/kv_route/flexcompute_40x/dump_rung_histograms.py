"""Per-layer rung-usage histograms from a TRAINED nested-FFN routing checkpoint.

The holder already accumulates exactly this (`usage_by_layer`: hard argmax counts per rung, per
layer, never reset), so this loads a trained checkpoint, runs it over real eval prompts, and dumps
the counts. No re-derivation -- these are the router's actual decisions on real tokens.

    python research/kv_route/flexcompute_40x/dump_rung_histograms.py --ckpt <run-dir> --out rungs_v11.json
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

from olmo_core.nn.nested_ffn_moe import post_build_hook_from_config
from olmo_core.nn.transformer import TransformerConfig

VOCAB = 248320


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt", required=True, help="run dir containing config.json + model_and_optim"
    )
    ap.add_argument(
        "--data", default="/net/sneetches/data/prasann/ffnmoe_exp/contra_n100_qboth_train"
    )
    ap.add_argument("--seq-len", type=int, default=6144)
    ap.add_argument("--rows", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")

    cfg_path = os.path.join(args.ckpt, "config.json")
    cfg_json = json.load(open(cfg_path))
    print(f"# checkpoint {args.ckpt}")

    # Build the model the checkpoint actually is, not an assumed one. These ffnmoe runs are
    # Qwen3-4B (vocab 151936, 36 layers, hidden 9728) -- NOT Qwen3.5-4B (248320, 32, 9216).
    mj = cfg_json.get("model", {})
    vocab = int(mj.get("vocab_size", VOCAB))
    n_layers = int(mj.get("n_layers", 0))
    fac = TransformerConfig.qwen3_4B if n_layers == 36 else TransformerConfig.qwen3_5_4B
    print(f"# building {fac.__name__} vocab={vocab} n_layers={n_layers}")
    model_cfg = fac(vocab_size=vocab)
    model = model_cfg.build(init_device="cpu")
    # `post_build_hook_from_config` is a FACTORY: it takes the checkpoint PATH and returns a hook
    # that is then applied to the model. Routing must be enabled before the load, or the router
    # keys have nowhere to go.
    hook = post_build_hook_from_config(args.ckpt)
    if hook is None:
        raise SystemExit(f"no ffn_moe block in {cfg_path} — not a routed checkpoint?")
    hook(model)

    # The distributed checkpoint loader issues collectives (reduce_scatter) even for a single
    # process, so a group must exist. gloo on localhost, world size 1, is enough.
    import torch.distributed as dist

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29571")
        dist.init_process_group("gloo", rank=0, world_size=1)

    from olmo_core.distributed.checkpoint import load_model_and_optim_state

    load_model_and_optim_state(os.path.join(args.ckpt, "model_and_optim"), model)
    model = model.to(dev).to(torch.bfloat16).eval()

    # The holder is NOT a submodule: `install_nested_ffn_moe` hangs it off each FeedForward as a
    # plain attribute (`ff._nffn_holder`, nested_ffn_moe.py:932), so `model.modules()` never sees
    # it. Reach it through a routed FeedForward instead.
    holder = None
    for mod in model.modules():
        h = getattr(mod, "_nffn_holder", None)
        if h is not None and hasattr(h, "usage_by_layer"):
            holder = h
            break
    if holder is None:
        raise SystemExit("no nested-FFN holder found — routing did not install")
    print(f"# rungs (widths): {getattr(holder, 'widths', '?')}  costs: {holder.costs}")

    toks = sorted(p for p in glob.glob(os.path.join(args.data, "*.npy")) if "labels_mask" not in p)
    arr = np.memmap(toks[0], dtype=np.uint32, mode="r")
    n = args.rows * args.seq_len
    ids = torch.from_numpy(np.asarray(arr[:n], dtype=np.int64)).view(args.rows, args.seq_len)
    print(f"# {args.rows} rows x {args.seq_len} tokens from {os.path.basename(toks[0])}")

    with torch.no_grad():
        for i in range(args.rows):
            model(ids[i : i + 1].to(dev))
            if i % 8 == 0:
                print(f"  row {i+1}/{args.rows}", flush=True)

    by_layer = {int(k): list(map(int, v)) for k, v in holder.usage_by_layer.items()}
    out = {
        "ckpt": args.ckpt,
        "costs": list(map(float, holder.costs)),
        "widths": [int(w) for w in getattr(holder, "widths", [])],
        "usage_by_layer": by_layer,
        "usage_total": list(map(int, holder.usage_total)),
    }
    json.dump(out, open(args.out, "w"), indent=1)
    tot = sum(out["usage_total"])
    print(
        "\nrung shares overall: "
        + "  ".join(f"w={w}:{100*c/tot:.1f}%" for w, c in zip(out["widths"], out["usage_total"]))
    )
    print(f"# wrote {args.out}")


if __name__ == "__main__":
    main()
