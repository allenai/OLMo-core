"""Set ``scalable_softmax: true`` on a checkpoint whose weights carry ``ssmax_scale``.

The flag is what makes the patched plugin register the parameter. Without it the weights load as
UNEXPECTED and are ignored -- the checkpoint runs, scores, and is quietly missing a trained
component. Verifies the weights are actually present before touching the config, so this cannot
mark a non-SSMax model as having SSMax.

    python debug/hybridish_sft/enable_ssmax_in_config.py --ckpt <dir> [--ckpt <dir> ...]
"""
import argparse, json, os
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", action="append", required=True)
a = ap.parse_args()

for ck in a.ckpt:
    cfg_p = os.path.join(ck, "config.json")
    st = os.path.join(ck, "model.safetensors")
    with safe_open(st, framework="pt") as f:
        layers = sorted(
            int(k.split(".layers.")[1].split(".")[0]) for k in f.keys() if "ssmax_scale" in k
        )
    if not layers:
        print(f"[skip] {ck}: no ssmax_scale weights; leaving config alone")
        continue
    cfg = json.load(open(cfg_p))
    was = cfg.get("scalable_softmax")
    cfg["scalable_softmax"] = True
    json.dump(cfg, open(cfg_p, "w"), indent=2)
    print(f"[set ] {ck}: scalable_softmax {was!r} -> True  (ssmax_scale on layers {layers})")
