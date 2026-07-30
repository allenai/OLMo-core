"""Parity of the COMPRESSIVE ragged (right-padded, cross-length) batched decode against the legacy
bs=1 per-row compressive decode.

``FastCompressiveLandmarkAttention._decode_ragged`` decodes a whole batch of rows that each sit at a
DIFFERENT absolute position / prompt length / top-k through the *compressive* grouped softmax (the
landmark token folds into its block's within-block softmax, plus the reserved non-selected-landmark
mass). The eager decode math runs on CPU, so we validate it here without the Triton kernel or a GPU:
for every row the ragged batched output must equal the legacy scalar ``_decode_one`` /
``_decode_one_eval`` run on that row alone (cache sliced to its own length).
"""

import pytest
import torch

from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.layer_norm import LayerNormConfig

MEM_FREQ = 15  # block_size = mem_freq + 1 = 16; landmarks at j % 16 == 15
BLOCK = MEM_FREQ + 1


def _build(nonselected_mass):
    attn = AttentionConfig(
        name=AttentionType.fast_compressive_landmark,
        n_heads=2,
        n_kv_heads=2,
        head_dim=24,
        bias=False,
        mem_freq=MEM_FREQ,
        nonselected_landmark_mass=nonselected_mass,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
    ).build(24, layer_idx=0, n_layers=1, init_device="cpu")
    attn.eval()
    return attn


@pytest.mark.parametrize("mode", ["extend_last_block", "generation_only"])
@pytest.mark.parametrize("top_k", [None, 2])
@pytest.mark.parametrize("ns_mass", [0.0, 0.1])
def test_compressive_ragged_decode_matches_scalar_per_row(mode, top_k, ns_mass):
    torch.manual_seed(0)
    attn = _build(ns_mass)
    H, D = attn.n_heads, attn.head_dim

    rows = [
        (48, 47),  # final prompt token, 3-block prompt (non-eval)
        (32, 40),  # generated token, shorter prompt (eval)
        (80, 95),  # generated token, longer prompt (eval)
        (64, 63),  # landmark-position prompt token, non-eval, drops self
        (48, 60),  # generated token, extends past prompt
    ]
    B = len(rows)
    plen = torch.tensor([r[0] for r in rows])
    qpos = torch.tensor([r[1] for r in rows])
    total = int(qpos.max().item()) + 1

    k = torch.randn(B, H, total, D)
    v = torch.randn(B, H, total, D)
    q = torch.randn(B, H, 1, D)

    top_k_t = None if top_k is None else torch.full((B,), top_k, dtype=torch.long)
    attn.set_landmark_ragged_decode(plen, mode=mode, top_k=top_k_t)
    attn.set_ragged_qpos(qpos)
    with torch.no_grad():
        out = attn._decode_ragged(q, k, v)

    attn.clear_ragged_decode()
    for b in range(B):
        attn._eval_prompt_len = int(plen[b].item())
        attn._eval_decode_mode = mode
        attn._eval_top_k = top_k
        qb = int(qpos[b].item())
        kb = k[b : b + 1, :, : qb + 1]
        vb = v[b : b + 1, :, : qb + 1]
        with torch.no_grad():
            ref = attn._decode_one(q[b : b + 1], kb, vb, qb)
        assert torch.allclose(out[b : b + 1], ref, atol=1e-5, rtol=1e-4), (
            f"row {b} (prompt_len={plen[b]}, qpos={qb}) mismatch: "
            f"max abs diff {(out[b:b+1]-ref).abs().max().item():.2e}"
        )
