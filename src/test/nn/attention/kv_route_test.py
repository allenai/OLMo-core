import torch

from olmo_core.nn.attention import Attention, AttentionBackendName
from olmo_core.nn.attention.kv_route import (
    KVRouteHolder,
    _masked_attention,
    install_kv_route,
)
from olmo_core.nn.transformer import TransformerConfig


def _tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    cfg = TransformerConfig.llama_like(
        d_model=64, n_layers=3, n_heads=4, n_kv_heads=2, vocab_size=128
    )
    model = cfg.build(init_device="cpu")
    model.init_weights()
    return model


def test_keep_all_matches_base():
    model = _tiny_model()
    ids = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        ref = model(ids)
    model.enable_kv_route(target=0.5)
    assert model._kv_route["routed"] == [0, 1, 2]
    with torch.no_grad():
        out = model(ids)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_masked_attention_drops_keys():
    B, T, H, Hk, D = 1, 8, 4, 2, 8
    q, k, v = (torch.randn(B, T, h, D) for h in (H, Hk, Hk))
    keep = torch.ones(B, T, dtype=torch.bool)
    full = _masked_attention(q, k, v, keep, None, None, D**-0.5)
    # dropping key 2 must change queries >2 and leave queries <=2 untouched
    keep[0, 2] = False
    part = _masked_attention(q, k, v, keep, None, None, D**-0.5)
    torch.testing.assert_close(part[:, :3], full[:, :3])
    assert not torch.allclose(part[:, 3:], full[:, 3:])
    # dropping every key = each query attends only to itself
    keep[:] = False
    solo = _masked_attention(q, k, v, keep, None, None, D**-0.5)
    from olmo_core.nn.attention.backend import _repeat_kv

    torch.testing.assert_close(solo, _repeat_kv(v, H // Hk))


def test_doc_mask_respected():
    B, T, H, D = 1, 6, 2, 4
    q, k, v = (torch.randn(B, T, H, D) for _ in range(3))
    keep = torch.ones(B, T, dtype=torch.bool)
    doc = torch.tensor([0, 0, 0, 1, 1, 1])
    out = _masked_attention(q, k, v, keep, doc, None, D**-0.5)
    # second doc's first token sees only itself
    torch.testing.assert_close(out[0, 3], v[0, 3])


def test_budget_gradient_reaches_router_and_loss_terms():
    model = _tiny_model()
    model.enable_kv_route(target=0.25, budget_weight=1.0)
    holder: KVRouteHolder = model._kv_route["holder"]
    ids = torch.randint(0, 128, (1, 24))
    out = model(ids, labels=ids.clone())
    out.loss.backward()
    router = model.blocks["0"].attention._kvr_router
    assert router.w.bias.grad is not None and router.w.bias.grad.abs().sum() > 0
    # keep-all at init -> budget pushes probs DOWN (positive gradient on the keep logit)
    assert router.w.bias.grad.item() > 0
    m = holder.metrics()
    assert abs(m["kv_route/mean_keep"] - 1.0) < 1e-6 or holder.mean_keep(last_forward=False) == 1.0


def test_hard_drop_changes_output_and_compacts_cache():
    model = _tiny_model()
    model.enable_kv_route(target=0.5)
    attn = model.blocks["1"].attention
    # force layer 1 to drop the first half of the tokens
    attn._kvr_router.w.bias.data.fill_(-10.0)
    ids = torch.randint(0, 128, (2, 12))
    with torch.no_grad():
        out = model(ids)
    holder = model._kv_route["holder"]
    keep = holder.mean_keep(last_forward=False)
    assert 0.6 < keep < 0.7  # 2 of 3 layers keep everything, layer 1 keeps nothing
    # prefill with a cache: evicted rows shrink the cache via leftpad
    from olmo_core.nn.attention.kv_cache import KVCacheManager

    for blk in model.blocks.values():  # (the CPU torch backend refuses init_kv_cache_manager)
        a = blk.attention
        a.kv_cache_manager = KVCacheManager(
            2, 32, a.n_kv_heads, a.head_dim, torch.device("cpu"), torch.float32
        )
    with torch.no_grad():
        model(ids, logits_to_keep=1)
    kvm = attn.kv_cache_manager
    assert kvm.cache_leftpad.tolist() == [12, 12]  # everything evicted at this layer
    assert model.blocks["0"].attention.kv_cache_manager.cache_leftpad.tolist() == [0, 0]
    assert int(kvm.cache_seqlens) == 12
    del out
