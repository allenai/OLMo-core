"""Quick check: does the smoothing hook hit every layer of Qwen2.5-0.5B?"""

from __future__ import annotations

import os
import sys

import torch
 # sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
from corpus_reasoning.lib.attn_smoothing import install_smoothed_attention  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

model_id = "Qwen/Qwen2.5-0.5B"
print(f"Loading {model_id}...")
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"Top-level class: {type(model).__name__}")
print(f"Config: {model.config.__class__.__name__}")
print(f"num_hidden_layers: {model.config.num_hidden_layers}")
print(f"hidden_size: {model.config.hidden_size}, heads: {model.config.num_attention_heads}, "
      f"kv_heads: {model.config.num_key_value_heads}")

# Enumerate decoder layer sub-module classes
print("\n--- Each decoder layer's self_attn class ---")
attn_classes = {}
for i, layer in enumerate(model.model.layers):
    cls = type(layer.self_attn).__name__ if hasattr(layer, "self_attn") else "NO self_attn"
    attn_classes[cls] = attn_classes.get(cls, 0) + 1
    if i < 3 or i >= model.config.num_hidden_layers - 2:
        print(f"  layer {i}: {type(layer).__name__} -> self_attn={cls}")
print(f"Unique attn classes: {attn_classes}")

# Install hook
state = {"alpha": 0.0}
install_smoothed_attention(model, state)
print(f"\nmodel.config._attn_implementation after install = {model.config._attn_implementation!r}")

# Instrument and fire once
fire_log = []
orig = ALL_ATTENTION_FUNCTIONS["sdpa_smoothed"]
id_to_path = {id(m): name for name, m in model.named_modules()}
def log_fn(module, *args, **kwargs):
    fire_log.append(id_to_path.get(id(module), "?"))
    return orig(module, *args, **kwargs)
ALL_ATTENTION_FUNCTIONS["sdpa_smoothed"] = log_fn

enc = tok("The capital of France is", return_tensors="pt").to(device)
B_, T_ = enc.input_ids.shape
state["row_mask"] = torch.zeros(B_, T_, dtype=torch.bool, device=device)
state["pad_mask"] = torch.ones(B_, T_, dtype=torch.bool, device=device)
with torch.no_grad():
    _ = model(enc.input_ids)

print(f"\nTotal hook fires in one forward: {len(fire_log)}")
print(f"Expected (num_hidden_layers): {model.config.num_hidden_layers}")
if len(fire_log) == model.config.num_hidden_layers:
    print("✓ Hook reaches every layer — clean testbed")
else:
    print("✗ Mismatch — hook does NOT cover all layers")

print("\nFirst 3 / last 3 fires:")
for p in fire_log[:3] + ["..."] + fire_log[-3:]:
    print(f"  {p}")

# Quick intervention sanity: α=1 all rows should shift logits
state["alpha"] = 0.0
state["row_mask"] = torch.zeros(B_, T_, dtype=torch.bool, device=device)
with torch.no_grad():
    L0 = model(enc.input_ids).logits

state["alpha"] = 1.0
state["row_mask"] = torch.ones(B_, T_, dtype=torch.bool, device=device)
with torch.no_grad():
    L1 = model(enc.input_ids).logits

diff = (L1 - L0).abs()
rel = (diff.mean() / L0.abs().mean()).item()
print(f"\nα=1 all rows vs α=0: max logit diff = {diff.max().item():.2f}, "
      f"mean|Δ|/mean|L| = {rel:.3f}")
if rel > 0.3:
    print("✓ Intervention is dramatic — smoothing is reaching the active attention path")
elif rel > 0.1:
    print("~ Moderate intervention — may still be partial")
else:
    print("✗ Tiny intervention — hook not reaching where it should")
