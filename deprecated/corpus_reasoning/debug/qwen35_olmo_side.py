"""olmo side of the length-parity check: build olmo Qwen3.5, load converted weights, run forward
at several sequence lengths on a FIXED deterministic input, save logits for comparison vs HF."""
import glob, json, os, types
import numpy as np, torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from olmo_core.nn.hf.convert import convert_qwen3_5_state_from_hf
from corpus_reasoning.lib.olmo_models import resolve_olmo_model, build_transformer_config

dev = "cuda"
LENS = [8, 128, 1024, 4096, 8192]
spec = resolve_olmo_model("Qwen/Qwen3.5-0.8B-Base")
model = build_transformer_config(spec).build().to(dev).eval()
snap = snapshot_download("Qwen/Qwen3.5-0.8B-Base")
raw = json.load(open(os.path.join(snap, "config.json")))
cfg = types.SimpleNamespace(**raw); cfg.text_config = types.SimpleNamespace(**raw["text_config"])
hf_state = {}
for s in sorted(glob.glob(os.path.join(snap, "*.safetensors"))): hf_state.update(load_file(s))
model.load_state_dict(convert_qwen3_5_state_from_hf(cfg, hf_state), strict=False)

g = torch.Generator().manual_seed(1234)
full = torch.randint(0, 100000, (1, max(LENS)), generator=g)
out = {}
for L in LENS:
    x = full[:, :L].to(dev)
    with torch.no_grad():
        lg = model(x).float().cpu().numpy()
    out[str(L)] = lg[0, -1]            # last-token logits
    out["ids_%d" % L] = full[0, :L].numpy()
np.savez("/data/prasann/runtime/olmo_q35_logits.npz", **out)
print("olmo logits saved for lens", LENS)
