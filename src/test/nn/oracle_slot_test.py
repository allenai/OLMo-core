import numpy as np
import torch

from olmo_core.nn.attention import Attention
from olmo_core.nn.oracle_slot import (
    OracleSlotCache,
    OracleSlotCacheWriter,
    derotate_keys,
    doc_hash64,
    fit_oracle_slots_layer,
    rotate_keys,
)
from olmo_core.nn.rope import RoPEConfig, RoPEType, RotaryEmbedding
from olmo_core.utils import seed_all


def test_rotate_matches_rope_and_derotate_inverts():
    seed_all(0)
    D, H, N = 16, 3, 12
    rope = RotaryEmbedding(head_size=D, theta=10_000)
    pos = torch.tensor([0, 5, 31, 2, 17, 8, 63, 1, 40, 22, 9, 58])
    pos_sin, pos_cos = rope._get_rotary_embedding(64, torch.device("cpu"))
    x = torch.randn(N, H, D)

    rotated = rotate_keys(x, pos_sin[pos], pos_cos[pos])
    # The module's own forward (position_ids path) must agree exactly.
    _, k_ref = rope(x[None], x[None], head_first=False, position_ids=pos[None])
    torch.testing.assert_close(rotated, k_ref[0], atol=1e-5, rtol=1e-5)

    back = derotate_keys(rotated, pos_sin[pos], pos_cos[pos])
    torch.testing.assert_close(back, x, atol=1e-5, rtol=1e-5)


def test_fit_oracle_slots_layer_beats_meanpool():
    seed_all(1)
    D, H_kv, H_q, M = 16, 2, 4, 96
    n_docs, doc_len = 8, 24
    Nt = n_docs * doc_len
    # Docs with one dominant high-norm key so the true log-mass is far from the mean-key logit.
    keys = torch.randn(Nt, H_kv, D) * 0.3
    keys[::doc_len] += torch.randn(n_docs, H_kv, D) * 2.5
    values = torch.randn(Nt, H_kv, D)
    doc_of = torch.arange(Nt) // doc_len
    q_stash = torch.randn(M, H_q, D) + 0.5  # non-zero mean, like RMSNorm'd real queries
    # Identity rotation buffers (relative-offset geometry is exercised in the rope test above).
    delta_sin = torch.zeros(M, D)
    delta_cos = torch.ones(M, D)

    k_star, v_star, bias, diags = fit_oracle_slots_layer(
        keys,
        values,
        doc_of,
        n_docs,
        q_stash,
        delta_sin,
        delta_cos,
        scale=D**-0.5,
        ridge=1e-3,
        doc_block=3,  # exercise the doc-block loop with a non-divisor block size
    )
    assert k_star.shape == (n_docs, H_kv, D)
    assert v_star.shape == (n_docs, H_kv, D)
    assert bias.shape == (n_docs,)
    assert torch.isfinite(bias).all()
    # The bias must be carrying the constant part of the log-mass (roughly log doc_len + typical
    # logit scale), not zero.
    assert bias.abs().mean() > 0.5
    assert diags["r2_oracle"] > diags["r2_meanpool"]
    assert diags["r2_oracle"] > 0.5

    # v* must lie inside the doc's value span: compare against a directly computed target
    # for one doc/head.
    scale = D**-0.5
    d, g = 3, 1
    group = H_q // H_kv
    sel = doc_of == d
    kd, vd = keys[sel][:, g], values[sel][:, g]
    q_all = q_stash[:, g * group : (g + 1) * group, :].reshape(-1, D)
    lg = (q_all @ kd.T) * scale
    w = torch.softmax(lg, dim=-1)
    v_ref = (w @ vd).mean(dim=0)
    torch.testing.assert_close(v_star[d, g], v_ref, atol=1e-4, rtol=1e-3)


def test_cache_writer_reader_roundtrip(tmp_path):
    L, H_kv, D = 4, 2, 8
    h1 = [doc_hash64([1, 2, 3]), doc_hash64([4, 5])]
    h2 = [doc_hash64([9, 9, 9])]
    s1, b1 = torch.randn(2, L, 2, H_kv, D), torch.randn(2, L)
    s2, b2 = torch.randn(1, L, 2, H_kv, D), torch.randn(1, L)
    w = OracleSlotCacheWriter(tmp_path, "rank0", L, H_kv, D)
    w.append(h1, s1, b1)
    w.close()
    w = OracleSlotCacheWriter(tmp_path, "rank1", L, H_kv, D)
    w.append(h2, s2, b2)
    w.close()

    cache = OracleSlotCache(tmp_path)
    assert cache.n_docs == 3
    idx = cache.lookup([h2[0], doc_hash64([7]), h1[0]])
    assert idx[1] == -1 and idx[0] >= 0 and idx[2] >= 0
    got, got_b = cache.gather(idx[[0, 2]])
    torch.testing.assert_close(got.float(), torch.stack([s2[0], s1[0]]).half().float())
    torch.testing.assert_close(got_b.float(), torch.stack([b2[0], b1[0]]).half().float())
    assert cache.hits == 2 and cache.misses == 1


def test_attention_soft_kv_override_self_consistency():
    """Deroting a layer's own post-RoPE K at some columns and passing it back through
    ``soft_kv_override`` must reproduce the baseline output exactly -- validating the full
    center-frame storage -> runtime re-rotation -> injection path."""
    seed_all(2)
    d_model, n_heads, n_kv_heads, T = 64, 4, 2, 24
    att = Attention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        rope=RoPEConfig(name=RoPEType.default, theta=10_000),
        init_device="cpu",
    )
    att.eval()
    x = torch.randn(2, T, d_model)
    pos_ids = torch.arange(T)[None].expand(2, T).contiguous()

    with torch.no_grad():
        base = att(x, position_ids=pos_ids)
        q, k, v = att._prepare_qkv(x, position_ids=pos_ids)

    rows = torch.tensor([0, 0, 1])
    cols = torch.tensor([3, 10, 7])
    pos = pos_ids[rows, cols]
    pos_sin, pos_cos = att.rope._get_rotary_embedding(T, torch.device("cpu"))
    k_cf = derotate_keys(k[rows, cols], pos_sin[pos], pos_cos[pos])

    with torch.no_grad():
        out = att(
            x,
            position_ids=pos_ids,
            soft_kv_override={
                "rows": rows,
                "cols": cols,
                "pos": pos,
                "k": k_cf,
                "v": v[rows, cols],
            },
        )
    torch.testing.assert_close(out, base, atol=1e-5, rtol=1e-5)


def test_attention_soft_kv_override_changes_output():
    seed_all(3)
    d_model, n_heads, n_kv_heads, T = 64, 4, 2, 24
    att = Attention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        rope=RoPEConfig(name=RoPEType.default, theta=10_000),
        init_device="cpu",
    )
    att.eval()
    x = torch.randn(1, T, d_model)
    pos_ids = torch.arange(T)[None].contiguous()
    hd = d_model // n_heads
    with torch.no_grad():
        base = att(x, position_ids=pos_ids)
        out = att(
            x,
            position_ids=pos_ids,
            soft_kv_override={
                "rows": torch.tensor([0]),
                "cols": torch.tensor([5]),
                "pos": torch.tensor([5]),
                "k": torch.randn(1, n_kv_heads, hd),
                "v": torch.randn(1, n_kv_heads, hd) * 3,
            },
        )
    # Queries at positions > 5 see the overridden KV -> outputs there must differ.
    assert not torch.allclose(out[0, 6:], base[0, 6:], atol=1e-4)
    # Positions before the overridden column are causally unaffected.
    torch.testing.assert_close(out[0, :5], base[0, :5], atol=1e-5, rtol=1e-5)


def test_attention_soft_kv_override_bias_path():
    """Slot bias of 0 through the attn_bias path must match the no-bias override; a large bias
    must boost the slot's influence downstream."""
    seed_all(4)
    d_model, n_heads, n_kv_heads, T = 64, 4, 2, 24
    att = Attention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        rope=RoPEConfig(name=RoPEType.default, theta=10_000),
        init_device="cpu",
    )
    att.eval()
    x = torch.randn(1, T, d_model)
    pos_ids = torch.arange(T)[None].contiguous()
    causal = torch.where(
        torch.ones(T, T, dtype=torch.bool).tril(),
        torch.zeros(()),
        torch.full((), torch.finfo(torch.float32).min),
    )[None, None]
    hd = d_model // n_heads
    ovr = {
        "rows": torch.tensor([0]),
        "cols": torch.tensor([5]),
        "pos": torch.tensor([5]),
        "k": torch.randn(1, n_kv_heads, hd),
        "v": torch.randn(1, n_kv_heads, hd),
    }
    with torch.no_grad():
        base = att(x, position_ids=pos_ids, attn_bias=causal, soft_kv_override=dict(ovr))
        zero = att(
            x,
            position_ids=pos_ids,
            attn_bias=causal,
            soft_kv_override={**ovr, "bias": torch.tensor([0.0])},
        )
        big = att(
            x,
            position_ids=pos_ids,
            attn_bias=causal,
            soft_kv_override={**ovr, "bias": torch.tensor([8.0])},
        )
    torch.testing.assert_close(zero, base, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(big[0, 6:], base[0, 6:], atol=1e-4)


def test_doc_hash64_stability():
    a = np.array([151648, 17, 42, 151649], dtype=np.uint32)
    assert doc_hash64(a) == doc_hash64([151648, 17, 42, 151649])
    assert doc_hash64(a) != doc_hash64(a[:-1])
