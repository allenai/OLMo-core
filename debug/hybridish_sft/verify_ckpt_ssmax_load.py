"""End-to-end: a real checkpoint loads under the patched plugin with ssmax_scale CONSUMED.

Under the stock plugin these keys report UNEXPECTED and are dropped. The assertion here is that the
loaded module's parameters carry the checkpoint's ssmax values bit-for-bit -- which is the whole
point of the port, and is not something a clean-looking load report proves on its own.
"""
import argparse, sys, torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--plugin", required=True)
a = ap.parse_args()
sys.path.insert(0, a.plugin)

import transformers_plugin
for fn in ("register", "register_config"):
    try: getattr(transformers_plugin, fn)()
    except Exception: pass
from transformers import AutoModelForCausalLM, AutoConfig

cfg = AutoConfig.from_pretrained(a.ckpt)
print(f"[cfg] scalable_softmax={getattr(cfg,'scalable_softmax',None)}  layers={cfg.num_hidden_layers}")
assert getattr(cfg, "scalable_softmax", False), "config flag not set; run enable_ssmax_in_config.py"

model = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=torch.bfloat16)

want = {}
with safe_open(f"{a.ckpt}/model.safetensors", framework="pt") as f:
    for k in f.keys():
        if "ssmax_scale" in k:
            want[int(k.split(".layers.")[1].split(".")[0])] = f.get_tensor(k)
assert want, "checkpoint has no ssmax_scale"

for i, ref in sorted(want.items()):
    got = model.model.layers[i].self_attn.ssmax_scale
    assert got is not None, f"layer {i}: ssmax_scale is None -- the plugin dropped it"
    assert torch.equal(got.detach().cpu().to(ref.dtype), ref.cpu()), f"layer {i}: value mismatch"
print(f"[ok ] ssmax_scale LOADED and matches the checkpoint on layers {sorted(want)}")
print(f"[ok ] per-head scale sample (layer {min(want)}): "
      f"{want[min(want)].float().flatten()[:6].tolist()}")

n_ss = sum(1 for n, _ in model.named_parameters() if "ssmax" in n)
assert n_ss == len(want), f"{n_ss} ssmax params vs {len(want)} in checkpoint"
print(f"[ok ] {n_ss} ssmax parameters registered, {sum(p.numel() for p in model.parameters()):,} total params")
print("\nPASS -- SSMax is live under the patched plugin")
