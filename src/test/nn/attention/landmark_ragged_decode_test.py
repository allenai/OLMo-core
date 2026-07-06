"""Parity of the ragged (right-padded, cross-length) batched landmark decode against the legacy
bs=1 per-row decode.

``FastLandmarkAttention._decode_ragged`` decodes a whole batch of rows that each sit at a DIFFERENT
absolute position / prompt length / top-k (the right-padding fast-eval path). The eager decode math
runs on CPU, so we validate it here without the Triton kernel or a GPU: for every row the ragged
batched output must equal the legacy scalar ``_decode_one`` / ``_decode_one_eval`` run on that row
alone (with its cache sliced to its own length). This is the parity gate that lets the eval batch
variable-length prompts together.
"""

import pytest
import torch

from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.layer_norm import LayerNormConfig

MEM_FREQ = 15  # FastLandmarkAttention requires mem_freq >= 15 -> block_size 16 (landmarks at %16==15)
BLOCK = MEM_FREQ + 1


def _build():
    attn = AttentionConfig(
        name=AttentionType.fast_landmark,
        n_heads=2,
        n_kv_heads=2,
        head_dim=24,
        bias=False,
        mem_freq=MEM_FREQ,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
    ).build(24, layer_idx=0, n_layers=1, init_device="cpu")
    attn.eval()
    return attn


@pytest.mark.parametrize("mode", ["extend_last_block", "generation_only"])
@pytest.mark.parametrize("top_k", [None, 2])
def test_ragged_decode_matches_scalar_per_row(mode, top_k):
    torch.manual_seed(0)
    attn = _build()
    H, D = attn.n_heads, attn.head_dim

    # A batch of rows with different prompt lengths and query positions. Cover: prompt-position (the
    # re-decoded final prompt token, non-eval), generated tokens (eval / one-long-local-block), and a
    # landmark-position query (qpos % BLOCK == BLOCK-1, which must drop its own key in non-eval mode).
    #              prompt_len, qpos
    rows = [
        (48, 47),  # final prompt token of a 3-block prompt (non-eval)
        (32, 40),  # generated token, shorter prompt (eval)
        (80, 95),  # generated token, longer prompt (eval)
        (64, 63),  # landmark-position prompt token (63 % 16 == 15), non-eval, drops self
        (48, 60),  # generated token, extend past the prompt
    ]
    B = len(rows)
    plen = torch.tensor([r[0] for r in rows])
    qpos = torch.tensor([r[1] for r in rows])
    total = int(qpos.max().item()) + 1

    # Shared right-padded cache: each row's keys beyond its own qpos are "pad" (random) and must be
    # ignored by the per-row causal mask.
    k = torch.randn(B, H, total, D)
    v = torch.randn(B, H, total, D)
    q = torch.randn(B, H, 1, D)

    top_k_t = None if top_k is None else torch.full((B,), top_k, dtype=torch.long)
    attn.set_landmark_ragged_decode(plen, mode=mode, top_k=top_k_t)
    attn.set_ragged_qpos(qpos)
    with torch.no_grad():
        out = attn._decode_ragged(q, k, v)  # (B,H,1,D)

    # Scalar per-row reference.
    attn.clear_ragged_decode()
    for b in range(B):
        attn._eval_prompt_len = int(plen[b].item())
        attn._eval_decode_mode = mode
        attn._eval_top_k = top_k
        qb = int(qpos[b].item())
        kb = k[b : b + 1, :, : qb + 1]
        vb = v[b : b + 1, :, : qb + 1]
        with torch.no_grad():
            ref = attn._decode_one(q[b : b + 1], kb, vb, qb)  # (1,H,1,D)
        assert torch.allclose(out[b : b + 1], ref, atol=1e-5, rtol=1e-4), (
            f"row {b} (prompt_len={plen[b]}, qpos={qb}) mismatch: "
            f"max abs diff {(out[b:b+1]-ref).abs().max().item():.2e}"
        )
