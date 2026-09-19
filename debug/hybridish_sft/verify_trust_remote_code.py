"""Load the checkpoint with NO plugin importable, purely through trust_remote_code.

This is the property olmo-eval (and any other consumer) depends on. Deliberately does not touch
sys.path: if `transformers_plugin` is reachable the test proves nothing, so it asserts the import
fails first. Then it checks SSMax actually came along -- a load that "works" while silently dropping
ssmax_scale is the exact failure this whole path exists to prevent -- and runs a real generation so
the remote modeling code is exercised, not just constructed.
"""
import argparse, importlib, sys, torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
a = ap.parse_args()

try:
    importlib.import_module("transformers_plugin")
    sys.exit("FAIL: transformers_plugin is importable; this test cannot prove anything")
except ImportError:
    print("[env] transformers_plugin NOT importable -- good, remote code is the only route")

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from safetensors import safe_open

cfg = AutoConfig.from_pretrained(a.ckpt, trust_remote_code=True)
print(f"[cfg] {cfg.model_type}  layers={cfg.num_hidden_layers}  "
      f"scalable_softmax={getattr(cfg,'scalable_softmax',None)}")

model = AutoModelForCausalLM.from_pretrained(
    a.ckpt, trust_remote_code=True, dtype=torch.bfloat16
).eval().cuda()

want = {}
with safe_open(f"{a.ckpt}/model.safetensors", framework="pt") as f:
    for k in f.keys():
        if "ssmax_scale" in k:
            want[int(k.split(".layers.")[1].split(".")[0])] = f.get_tensor(k)
for i, ref in sorted(want.items()):
    got = model.model.layers[i].self_attn.ssmax_scale
    assert got is not None, f"layer {i}: SSMax dropped under remote code"
    assert torch.equal(got.detach().cpu().to(ref.dtype), ref.cpu()), f"layer {i} mismatch"
print(f"[ok ] ssmax_scale live on layers {sorted(want)} via remote code")

tok = AutoTokenizer.from_pretrained(a.ckpt)
ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=12, do_sample=False,
                         pad_token_id=cfg.eos_token_id if isinstance(cfg.eos_token_id,int) else 100257)
print(f"[gen] {tok.decode(out[0], skip_special_tokens=True)!r}")
print(f"[ok ] {sum(p.numel() for p in model.parameters()):,} params")
print("\nPASS -- loads and generates with no plugin installed")
