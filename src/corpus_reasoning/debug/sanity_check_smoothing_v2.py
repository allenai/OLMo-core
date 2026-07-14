"""Identify which attention modules actually go through sdpa_smoothed.

Walks the model tree, finds every module that exposes an attention-implementation
hook, then during forward logs full module path + class for each fire.
"""

from __future__ import annotations

import os
import sys

import torch
 # sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
from corpus_reasoning.lib.attn_smoothing import install_smoothed_attention  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402


model_id = "Qwen/Qwen3.5-0.8B-Base"
print(f"Loading {model_id}...")
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Loaded. Top-level class: {type(model).__name__}")
print(f"Config: {model.config.__class__.__name__}, num_hidden_layers={getattr(model.config, 'num_hidden_layers', 'n/a')}")

# ---- Walk tree: find modules carrying attention-impl attrs BEFORE install ----
print("\n--- Tree walk: modules with attention-related attrs ---")
impl_attr_count = 0
class_counts = {}
for name, m in model.named_modules():
    has_impl = hasattr(m, "_attn_implementation")
    has_cfg_impl = hasattr(m, "config") and hasattr(m.config, "_attn_implementation")
    cls = type(m).__name__
    if has_impl or ("Attention" in cls and "Layer" not in cls and "Block" not in cls):
        class_counts[cls] = class_counts.get(cls, 0) + 1
        if impl_attr_count < 12:  # sample first few
            val = getattr(m, "_attn_implementation", None)
            print(f"  {name:70}  cls={cls:40}  _attn_impl={val}")
        impl_attr_count += 1

print(f"\nTotal modules with _attn_implementation OR name-match 'Attention': {impl_attr_count}")
print("Class name counts:")
for cls, n in sorted(class_counts.items(), key=lambda x: -x[1]):
    print(f"  {cls}: {n}")

# ---- Install hook and see what changed ----
print("\n--- Installing smoothed attention ---")
state = {"alpha": 0.0}
install_smoothed_attention(model, state)

flipped_names = []
for name, m in model.named_modules():
    if getattr(m, "_attn_implementation", None) == "sdpa_smoothed":
        flipped_names.append(name)
print(f"After install: {len(flipped_names)} modules with _attn_implementation='sdpa_smoothed'")
for n in flipped_names[:10]:
    print(f"  {n}")

# model.config itself
print(f"model.config._attn_implementation = {model.config._attn_implementation!r}")

# Sub-configs?
for attr in ["text_config", "language_config", "vision_config"]:
    sub = getattr(model.config, attr, None)
    if sub is not None:
        print(f"model.config.{attr}._attn_implementation = {getattr(sub, '_attn_implementation', 'MISSING')!r}")

# ---- Instrument hook to log caller ----
print("\n--- Instrumenting sdpa_smoothed to log call site ---")
fire_log = []
orig_fn = ALL_ATTENTION_FUNCTIONS["sdpa_smoothed"]

# Build id->path map
id_to_path = {id(m): name for name, m in model.named_modules()}

def logging_fn(module, *args, **kwargs):
    path = id_to_path.get(id(module), "?")
    fire_log.append((path, type(module).__name__))
    return orig_fn(module, *args, **kwargs)

ALL_ATTENTION_FUNCTIONS["sdpa_smoothed"] = logging_fn

# ---- Forward ----
print("\n--- Forward pass ---")
prompt = "The capital of France is"
enc = tok(prompt, return_tensors="pt").to(device)
B_, T_ = enc.input_ids.shape
state["alpha"] = 0.0
state["row_mask"] = torch.zeros(B_, T_, dtype=torch.bool, device=device)
state["pad_mask"] = torch.ones(B_, T_, dtype=torch.bool, device=device)
with torch.no_grad():
    _ = model(enc.input_ids)

print(f"\nTotal fires: {len(fire_log)}")
print("\nEach fire (module path → class):")
for i, (path, cls) in enumerate(fire_log):
    print(f"  {i:2}. {path:70}  {cls}")

print("\n--- Module-tree: first-level children of model ---")
for name, m in model.named_children():
    print(f"  {name}: {type(m).__name__}")

# Deeper: show the LM decoder structure if present
print("\n--- Looking for decoder layers ---")
for name, m in model.named_modules():
    cls = type(m).__name__
    if "DecoderLayer" in cls or "Block" in cls:
        print(f"  {name}: {cls}")
        if name.count(".") <= 3:
            continue  # keep going but only print near-top
        break
