"""Sanity-check the attention smoothing hook.

Three sections:
  1. Pure-tensor unit tests of blend_with_mean_v / _mean_v_cumulative.
  2. Hook installation: every attention module flipped, hook fires once per
     attention forward.
  3. Real model forward: α=0 logits match baseline; α=1 with all rows smoothed
     differs substantially from baseline; α=1 with no rows smoothed == baseline.
"""

from __future__ import annotations

import os
import sys
import time

import torch
 # sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
from corpus_reasoning.lib.attn_smoothing import (  # noqa: E402
    blend_with_mean_v,
    _mean_v_cumulative,
    install_smoothed_attention,
)


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Section 1: unit tests of the blend math
# ---------------------------------------------------------------------------
section("Section 1: blend math (pure tensor)")

torch.manual_seed(0)
B, H, T, D = 1, 2, 5, 4
v = torch.randn(B, H, T, D)

# (a) _mean_v_cumulative with no padding
pad = torch.ones(B, T, dtype=torch.bool)
mean_v = _mean_v_cumulative(v, pad)
expected = torch.stack([v[:, :, : i + 1, :].mean(dim=2) for i in range(T)], dim=2)
assert torch.allclose(mean_v, expected, atol=1e-5), "mean_V cumulative mismatch"
print(f"[OK] _mean_v_cumulative: max diff {(mean_v - expected).abs().max():.2e}")

# (b) respects padding mask
pad2 = torch.ones(B, T, dtype=torch.bool)
pad2[0, 3:] = False
mean_v2 = _mean_v_cumulative(v, pad2)
exp_t3 = v[:, :, :3, :].sum(dim=2) / 3
assert torch.allclose(mean_v2[:, :, 3, :], exp_t3, atol=1e-5)
assert torch.allclose(mean_v2[:, :, 4, :], exp_t3, atol=1e-5)
print("[OK] _mean_v_cumulative: masked positions don't contribute to average")

# (c) blend α=1.0 swaps smoothed rows, leaves others
attn_out = torch.randn(B, H, T, D)
row_mask = torch.zeros(B, T, dtype=torch.bool)
row_mask[0, 2:4] = True
blended = blend_with_mean_v(attn_out, v, row_mask, pad, alpha=1.0)
assert torch.allclose(blended[0, :, 2], mean_v[0, :, 2])
assert torch.allclose(blended[0, :, 3], mean_v[0, :, 3])
assert torch.equal(blended[0, :, 0], attn_out[0, :, 0])
assert torch.equal(blended[0, :, 1], attn_out[0, :, 1])
assert torch.equal(blended[0, :, 4], attn_out[0, :, 4])
print("[OK] blend α=1.0: smoothed rows == mean_V; others unchanged")

# (d) blend α=0.5 linear combination
blended05 = blend_with_mean_v(attn_out, v, row_mask, pad, alpha=0.5)
expected05 = 0.5 * attn_out[0, :, 2] + 0.5 * mean_v[0, :, 2]
assert torch.allclose(blended05[0, :, 2], expected05, atol=1e-5)
print("[OK] blend α=0.5: smoothed row == 0.5·attn + 0.5·mean_V")

# (e) α=0 bit-identity
blended0 = blend_with_mean_v(attn_out, v, row_mask, pad, alpha=0.0)
assert torch.equal(blended0, attn_out)
print("[OK] blend α=0: bit-identical to attn_out")


# ---------------------------------------------------------------------------
# Section 2: real model — hook installation and firing
# ---------------------------------------------------------------------------
section("Section 2: hook installation on Qwen3.5-0.8B-Base")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

model_id = "Qwen/Qwen3.5-0.8B-Base"
print(f"Loading {model_id} (this takes ~30–60s)...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Loaded in {time.time()-t0:.1f}s, device={device}")

state = {"alpha": 0.0}
install_smoothed_attention(model, state)

# Count attention modules flipped
flipped = [m for m in model.modules()
           if getattr(m, "_attn_implementation", None) == "sdpa_smoothed"]
print(f"Modules flipped to sdpa_smoothed: {len(flipped)}")
print(f"Config num_hidden_layers: {model.config.num_hidden_layers}")

# Instrument sdpa_smoothed to count calls per forward
fire_count = [0]
orig_fn = ALL_ATTENTION_FUNCTIONS["sdpa_smoothed"]
def counting_fn(*args, **kwargs):
    fire_count[0] += 1
    return orig_fn(*args, **kwargs)
ALL_ATTENTION_FUNCTIONS["sdpa_smoothed"] = counting_fn


# ---------------------------------------------------------------------------
# Section 3: real forward α=0 vs α=1
# ---------------------------------------------------------------------------
section("Section 3: forward pass — α=0 vs α=1")

prompt = "The capital of France is"
enc = tok(prompt, return_tensors="pt").to(device)
ids = enc.input_ids
B_, T_ = ids.shape
print(f"Input: {prompt!r}, shape {tuple(ids.shape)}")

# Baseline (α=0, no row mask)
state["alpha"] = 0.0
state["row_mask"] = torch.zeros(B_, T_, dtype=torch.bool, device=device)
state["pad_mask"] = torch.ones(B_, T_, dtype=torch.bool, device=device)
fire_count[0] = 0
with torch.no_grad():
    logits_a0 = model(ids).logits
fires_a0 = fire_count[0]
print(f"α=0 forward: {fires_a0} attention calls (expected {len(flipped)})")

# Top-5 predictions at last position
top5 = torch.topk(logits_a0[0, -1], 5)
print("α=0 top-5 next-token predictions:")
for tid, score in zip(top5.indices.tolist(), top5.values.tolist()):
    print(f"   {tok.decode([tid])!r:15}  logit={score:.2f}")

# α=1, all rows smoothed
state["alpha"] = 1.0
state["row_mask"] = torch.ones(B_, T_, dtype=torch.bool, device=device)
fire_count[0] = 0
with torch.no_grad():
    logits_a1_all = model(ids).logits
fires_a1_all = fire_count[0]
print(f"\nα=1 (all rows smoothed): {fires_a1_all} attention calls")

top5 = torch.topk(logits_a1_all[0, -1], 5)
print("α=1 (all) top-5 next-token predictions:")
for tid, score in zip(top5.indices.tolist(), top5.values.tolist()):
    print(f"   {tok.decode([tid])!r:15}  logit={score:.2f}")

# α=1 but empty row_mask → should equal α=0 (short-circuit)
state["alpha"] = 1.0
state["row_mask"] = torch.zeros(B_, T_, dtype=torch.bool, device=device)
with torch.no_grad():
    logits_a1_none = model(ids).logits
diff_none = (logits_a1_none - logits_a0).abs().max().item()
print(f"\nα=1 with empty row_mask vs α=0 baseline: max logit diff = {diff_none:.2e}")

# α=1 all rows vs baseline
diff_all = (logits_a1_all - logits_a0).abs().max().item()
rel_change = ((logits_a1_all - logits_a0).abs().mean() /
              logits_a0.abs().mean()).item()
print(f"α=1 all rows vs α=0 baseline: max logit diff = {diff_all:.2e}, "
      f"mean|Δ|/mean|logit| = {rel_change:.3f}")

# Partial row mask — only last row smoothed (mimics single answer token)
state["row_mask"] = torch.zeros(B_, T_, dtype=torch.bool, device=device)
state["row_mask"][0, -1] = True
with torch.no_grad():
    logits_a1_last = model(ids).logits
# Earlier positions should match baseline exactly (causal); last position should differ
diff_earlier = (logits_a1_last[0, :-1] - logits_a0[0, :-1]).abs().max().item()
diff_last = (logits_a1_last[0, -1] - logits_a0[0, -1]).abs().max().item()
print(f"\nα=1 last-row-only vs α=0 baseline:")
print(f"   positions 0..T-2 max diff: {diff_earlier:.2e}  (causal: should be ~0)")
print(f"   last position  max diff:   {diff_last:.2e}  (should be large)")

print("\n=== SUMMARY ===")
print(f"  layers flipped       : {len(flipped)}")
print(f"  fires per forward    : {fires_a0}")
print(f"  α=1 empty mask ≡ α=0 : {diff_none:.2e}")
print(f"  α=1 all rows Δ       : {diff_all:.2f}")
print(f"  α=1 last only Δ last : {diff_last:.2f}")
print(f"  α=1 last only Δ prior: {diff_earlier:.2e}  (causal isolation)")
