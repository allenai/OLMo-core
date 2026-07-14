"""Pinpoint the block-0 FFN corruption: compare the converter's path
(convert_qwen3_5_state_from_hf on RAW safetensors + SimpleNamespace cfg) vs the proven-correct path
(convert_state_from_hf on a from_pretrained model). Inspect block-0 FFN in the inputs and outputs.
"""
import glob, json, os, types
import torch, transformers
from safetensors.torch import load_file
from olmo_core.nn.hf.convert import convert_state_from_hf, convert_qwen3_5_state_from_hf

LOCAL = "/data/prasann/hf-cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"

# --- converter path inputs ---
raw = json.load(open(os.path.join(LOCAL, "config.json")))
cfg_obj = types.SimpleNamespace(**raw); cfg_obj.text_config = types.SimpleNamespace(**raw["text_config"])
hf_state = {}
for shard in sorted(glob.glob(os.path.join(LOCAL, "*.safetensors"))):
    hf_state.update(load_file(shard))
print(f"raw safetensors: {len(hf_state)} tensors", flush=True)
# what block-0 mlp keys exist in the raw state?
mlp0 = [k for k in hf_state if ".0." in k and ("mlp" in k or "feed" in k)]
print("raw block-0 mlp keys:", mlp0, flush=True)
mlp1 = [k for k in hf_state if ".1." in k and "mlp" in k]
print("raw block-1 mlp keys:", mlp1, flush=True)

conv_A = convert_qwen3_5_state_from_hf(cfg_obj, hf_state)
def n(d, k): return d[k].float().norm().item() if k in d else None
for L in (0, 1, 2):
    print(f"[A converter] blocks.{L}.feed_forward.w1 norm = {n(conv_A, f'blocks.{L}.feed_forward.w1.weight')}", flush=True)

# --- proven-correct path ---
hf = transformers.Qwen3_5ForCausalLM.from_pretrained(LOCAL, torch_dtype=torch.float32, attn_implementation="eager")
sd = hf.state_dict()
mlp0b = [k for k in sd if "layers.0." in k and "mlp" in k]
print("\nfrom_pretrained block-0 mlp keys:", mlp0b, flush=True)
conv_B = convert_state_from_hf(hf.config, sd, model_type="qwen3_5_text")
for L in (0, 1, 2):
    print(f"[B correct]   blocks.{L}.feed_forward.w1 norm = {n(conv_B, f'blocks.{L}.feed_forward.w1.weight')}", flush=True)

# direct compare the two converted block-0 FFNs
import torch as T
for L in (0, 1):
    kA = f"blocks.{L}.feed_forward.w1.weight"
    if kA in conv_A and kA in conv_B:
        a, b = conv_A[kA].float(), conv_B[kA].float()
        if a.shape == b.shape:
            cos = T.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
            print(f"  block{L} FFN w1: cos(A,B)={cos:+.4f}  |A|={a.norm():.3f} |B|={b.norm():.3f}", flush=True)
        else:
            print(f"  block{L} FFN w1 SHAPE A{tuple(a.shape)} vs B{tuple(b.shape)}", flush=True)
# compare RAW input values for block0 vs block1 mlp (are they actually present & sane?)
print("\nraw input norms:", flush=True)
for k in sorted(set([x for x in hf_state if 'layers.0.mlp' in x] + [x for x in hf_state if 'layers.1.mlp' in x])):
    print(f"  {k}: norm={hf_state[k].float().norm().item():.3f} shape={tuple(hf_state[k].shape)}", flush=True)
