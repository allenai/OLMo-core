"""Validate the SSMax port: formula matches olmo-core, and KV-cache decoding matches no-cache.

The second test is the point. Our previous eval re-attached SSMax with a forward hook that read the
CURRENT forward's sequence length, so every cached decode step computed position 1 -- log(2) where
the truth is log(prompt_len+step). Prefill looked fine and generation was quietly wrong. A cached
decode must be numerically indistinguishable from re-running the whole prefix uncached; this asserts
that, and fails on the hook's behaviour.

CPU, seconds, no checkpoint needed.
"""
import sys, torch

PLUGIN = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/hybridish_sft/plugin_ssmax"
sys.path.insert(0, PLUGIN)

from transformers_plugin.configuration_mainline_ladder import MainlineLadderConfig
from transformers_plugin.modeling_mainline_ladder import MainlineLadderAttention

torch.manual_seed(0)
H, D, T = 4, 16, 9
cfg = MainlineLadderConfig(
    hidden_size=H * D, num_attention_heads=H, num_key_value_heads=H, head_dim=D,
    num_hidden_layers=5, intermediate_size=64, scalable_softmax=True,
)
attn = MainlineLadderAttention(cfg, layer_idx=4).eval()

# ---- 1. parameter exists and is per-head -------------------------------------------------------
assert attn.ssmax_scale is not None, "ssmax_scale not registered"
assert attn.ssmax_scale.shape == (H,), attn.ssmax_scale.shape
print(f"[1] ssmax_scale registered, shape {tuple(attn.ssmax_scale.shape)}  OK")

# ---- 2. formula matches olmo-core's _apply_scalable_softmax ------------------------------------
with torch.no_grad():
    attn.ssmax_scale.copy_(torch.linspace(0.5, 1.5, H))
q = torch.randn(1, H, T, D)
got = attn._apply_scalable_softmax(q, torch.arange(T), None)

# reference, transcribed from olmo-core (q as (B,T,H,D)) then transposed to compare
qr = q.transpose(1, 2)                                  # (1,T,H,D)
visible = torch.arange(1, T + 1).unsqueeze(0).expand(1, -1)
ref_scale = visible.log().to(qr.dtype).unsqueeze(-1)    # (1,T,1)
ref_scale = ref_scale * attn.ssmax_scale.view(1, 1, -1)  # (1,T,H)
ref = (qr * ref_scale.unsqueeze(-1)).transpose(1, 2)
assert torch.allclose(got, ref, atol=1e-6), (got - ref).abs().max()
print(f"[2] matches olmo-core reference, max diff {(got-ref).abs().max():.2e}  OK")
assert torch.equal(got[:, :, 0], torch.zeros_like(got[:, :, 0])), "position 0 must be log(1)=0"
print("[2b] position 0 scales by log(1)=0, as upstream  OK")

# ---- 3. cached decoding == uncached full forward ------------------------------------------------
# The hook bug: a decode step sees seq_len 1 and computes position 1 instead of its true position.
prompt, steps = 6, 3
q_full = torch.randn(1, H, prompt + steps, D)
full = attn._apply_scalable_softmax(q_full, torch.arange(prompt + steps), None)

pieces = [attn._apply_scalable_softmax(q_full[:, :, :prompt], torch.arange(prompt), None)]
for s in range(steps):
    pos = prompt + s
    pieces.append(attn._apply_scalable_softmax(
        q_full[:, :, pos:pos + 1], torch.tensor([pos]), None))
cached = torch.cat(pieces, dim=2)
assert torch.allclose(full, cached, atol=1e-6), (full - cached).abs().max()
print(f"[3] cached decode == uncached forward, max diff {(full-cached).abs().max():.2e}  OK")

# What the old hook did, for contrast: it read the CURRENT forward's length, so each cached decode
# step saw seq_len 1 -> local position 1, clamped to 2. Model that at a realistic prompt length.
REAL_PROMPT = 8000
hook_scale = torch.tensor(2.0).log()                       # clamp(min=2) -> log(2)
true_scale = torch.tensor(float(REAL_PROMPT + 1)).log()    # log(prompt_len + step)
print(f"[3b] at prompt_len={REAL_PROMPT} the old hook scaled decode q by log(2)={hook_scale:.3f} "
      f"instead of log({REAL_PROMPT+1})={true_scale:.3f} "
      f"-- {true_scale/hook_scale:.1f}x too small (the bug this port removes)")

# ---- 4. default off: no parameter, no behaviour change ------------------------------------------
cfg2 = MainlineLadderConfig(
    hidden_size=H * D, num_attention_heads=H, num_key_value_heads=H, head_dim=D,
    num_hidden_layers=5, intermediate_size=64,
)
a2 = MainlineLadderAttention(cfg2, layer_idx=4)
assert a2.ssmax_scale is None
assert not any("ssmax" in n for n, _ in a2.named_parameters())
print("[4] scalable_softmax defaults off; no ssmax parameter  OK")
print("\nALL PASS")
