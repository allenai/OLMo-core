"""Diagnose why olmo-core Qwen3.5 (hybrid GDN) trains (grads flow, norm~13, no skipped steps)
but doesn't LEARN (CE plateaus ~0.83 even on trivial niah; dense Qwen3 -> 0.0001).
Loads the REAL converted weights (HF->olmo), then:
 (1) train-mode vs eval-mode forward (a GDN train-path bug shows here; eval-only parity misses it),
 (2) per-MODULE grad norms after backward (is the GDN/linear_attn getting gradient, or is the
     norm-13 all from the 6 softmax layers + MLPs?),
 (3) magnitude of each linear_attn block's OUTPUT vs the residual stream (is GDN contributing?).
Run NFS-free on /data with the amandab olmo-core src on PYTHONPATH.
"""
import glob, json, math, os, types
import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from olmo_core.nn.hf.convert import convert_qwen3_5_state_from_hf
from corpus_reasoning.lib.olmo_models import resolve_olmo_model, build_transformer_config

torch.manual_seed(0)
dev = "cuda"
spec = resolve_olmo_model("Qwen/Qwen3.5-0.8B-Base")
model = build_transformer_config(spec).build().to(dev)
print(f"builder={spec.builder} is_hybrid={getattr(spec,'is_hybrid',None)} "
      f"n_params={sum(p.numel() for p in model.parameters())/1e9:.3f}B")

# --- load REAL converted weights ---
snap = snapshot_download("Qwen/Qwen3.5-0.8B-Base")
raw = json.load(open(os.path.join(snap, "config.json")))
cfg_obj = types.SimpleNamespace(**raw); cfg_obj.text_config = types.SimpleNamespace(**raw["text_config"])
hf_state = {}
for s in sorted(glob.glob(os.path.join(snap, "*.safetensors"))): hf_state.update(load_file(s))
converted = convert_qwen3_5_state_from_hf(cfg_obj, hf_state)
missing, unexpected = model.load_state_dict(converted, strict=False)
print(f"[load] converted={len(converted)} missing={len(missing)} unexpected={len(unexpected)}")
if missing: print("  MISSING (left at random init!):", [m for m in missing][:20])

def kind(n):
    if any(s in n for s in ("linear_attn","in_proj_a","in_proj_b","A_log","dt_bias","conv1d","in_proj_qkv")): return "GDN"
    if any(s in n for s in ("self_attn",".attention.","q_norm","k_norm")): return "softmax_attn"
    if any(s in n for s in ("feed_forward","mlp","w1","w2","w3")): return "mlp"
    if any(s in n for s in ("embed","lm_head","w_out")): return "embed/head"
    return "other"

x = torch.randint(0, 1000, (1, 1024), device=dev)

# --- PRECISION TEST: training runs bf16-autocast (FSDP param_dtype=bf16); parity/my-fp32 don't.
# GDN's SSM scan is bf16-fragile. If bf16 logits diverge from fp32, that's the training bug.
model.eval()
with torch.no_grad():
    l_fp32 = model(x).float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l_bf16 = model(x).float()
md_p = (l_fp32 - l_bf16).abs().max().item(); rel_p = md_p/(l_fp32.abs().max().item()+1e-9)
print(f"\n[fp32-vs-bf16 forward] max|delta|={md_p:.3e} rel={rel_p:.3e} "
      f"finite_bf16={torch.isfinite(l_bf16).all().item()} "
      f"{'<-- bf16 GDN DIVERGES (precision bug in training!)' if rel_p>0.1 else 'ok-ish'}")

model.eval()
with torch.no_grad(): le = model(x).float()
model.train()
with torch.no_grad(): lt = model(x).float()
md = (le-lt).abs().max().item(); rel = md/(le.abs().max().item()+1e-9)
print(f"\n[train-vs-eval forward] finite_eval={torch.isfinite(le).all().item()} "
      f"max|delta|={md:.3e} rel={rel:.3e} {'<-- TRAIN FORWARD DIFFERS (GDN train bug)' if rel>0.05 else 'ok'}")

model.train()
out = model(x)
loss = torch.nn.functional.cross_entropy(out[:, :-1].reshape(-1, out.shape[-1]).float(), x[:, 1:].reshape(-1))
loss.backward()
agg = {}
for n, p in model.named_parameters():
    k = kind(n); a = agg.setdefault(k, [0.0,0,0])
    if p.grad is None: a[2]+=1; continue
    a[0]+=p.grad.float().norm().item()**2; a[1]+=1
print(f"\n[per-module grad L2] loss={loss.item():.3f} finite={torch.isfinite(loss).item()}")
for k,(sq,c,nog) in sorted(agg.items()):
    print(f"  {k:13s} L2={math.sqrt(sq):.3e} over {c} tensors" + (f"  ({nog} NO-grad!)" if nog else ""))
