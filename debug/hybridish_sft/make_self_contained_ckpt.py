"""Make a checkpoint loadable by ANY HF consumer via trust_remote_code, with SSMax live.

The `mainline_ladder` plugin registers itself only when someone calls `transformers_plugin.register()`
-- there is no entry point -- so every consumer needs bespoke wiring, and a consumer that forgets it
does not fail loudly: the stock plugin drops `ssmax_scale` and scores a model missing a trained
component. Copying the (SSMax-patched) config+modeling modules into the checkpoint and declaring
`auto_map` removes the wiring entirely: olmo-eval, a bare AutoModel, anything.

Sets `scalable_softmax` from the WEIGHTS, never from a flag, so this cannot mislabel a model.

    python debug/hybridish_sft/make_self_contained_ckpt.py --ckpt <dir> --plugin <patched plugin>
"""
import argparse, json, os, shutil
from safetensors import safe_open

AUTO_MAP = {
    "AutoConfig": "configuration_mainline_ladder.MainlineLadderConfig",
    "AutoModel": "modeling_mainline_ladder.MainlineLadderModel",
    "AutoModelForCausalLM": "modeling_mainline_ladder.MainlineLadderForCausalLM",
}
FILES = ("configuration_mainline_ladder.py", "modeling_mainline_ladder.py")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", action="append", required=True)
ap.add_argument("--plugin", required=True, help="patched plugin package root")
a = ap.parse_args()

pkg = os.path.join(a.plugin, "transformers_plugin")
for f in FILES:
    assert os.path.exists(os.path.join(pkg, f)), f"missing {f} in {pkg}"
src_mod = open(os.path.join(pkg, "modeling_mainline_ladder.py")).read()
assert "_apply_scalable_softmax" in src_mod, "plugin is NOT the SSMax-patched one -- refusing"

for ck in a.ckpt:
    for f in FILES:
        shutil.copy2(os.path.join(pkg, f), os.path.join(ck, f))
    with safe_open(os.path.join(ck, "model.safetensors"), framework="pt") as fh:
        layers = sorted(
            int(k.split(".layers.")[1].split(".")[0]) for k in fh.keys() if "ssmax_scale" in k
        )
    cfg_p = os.path.join(ck, "config.json")
    cfg = json.load(open(cfg_p))
    cfg["auto_map"] = AUTO_MAP
    cfg["scalable_softmax"] = bool(layers)   # from the weights, not from a flag
    json.dump(cfg, open(cfg_p, "w"), indent=2)
    print(f"[self-contained] {ck}")
    print(f"    auto_map set, modules copied, scalable_softmax={bool(layers)} (layers {layers})")
