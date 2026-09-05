"""
GPU smoke test for learned KV-cache routing (olmo_core.nn.attention.kv_route) on the FlexAttention
path: (1) keep-all == base (packed rows with cu_doc_lens, and padded rows), (2) backward reaches the
router, (3) flash-backend prefill with eviction + one decode step == a no-cache routed forward over
the same tokens, (4) timing of a 64k routed forward+backward vs dense.

    srun -p berkeleynlp --gres=gpu:1 -w horton /data/prasann/conda/envs/corpus-reasoning-olmo/bin/python \
        debug/flop_scaling/smoke_kv_route_gpu.py
"""

import sys
import time

import torch

sys.path.insert(0, "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/src")
from olmo_core.nn.attention import AttentionBackendName  # noqa: E402
from olmo_core.nn.attention.kv_cache import KVCacheManager  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402

dev = torch.device("cuda")


def build(backend="flash_2", seed=0):
    torch.manual_seed(seed)
    cfg = TransformerConfig.llama_like(d_model=256, n_layers=4, n_heads=8, n_kv_heads=2, vocab_size=512)
    cfg.apply(lambda c: setattr(c, "backend", AttentionBackendName(backend)) if hasattr(c, "backend") else None)
    m = cfg.build(init_device="cuda")
    m.init_weights()
    return m.to(torch.bfloat16)


def cu(lens):
    return torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=dev)


def main():
    m = build()
    torch.manual_seed(1)
    T = 4096
    ids = torch.randint(0, 512, (1, T), device=dev)
    lens = [1500, 1000, 1596]
    with torch.no_grad():
        ref = m(ids, doc_lens=torch.tensor([lens], device=dev), max_doc_lens=[max(lens)]).float()
    m.enable_kv_route(target=0.5)
    with torch.no_grad():
        out = m(ids, doc_lens=torch.tensor([lens], device=dev), max_doc_lens=[max(lens)]).float()
    d = (out - ref).abs().max().item()
    print(f"[1] keep-all vs base (packed, flex vs flash): max|diff|={d:.4f}  rel={d / ref.abs().max().item():.2e}")
    assert d / ref.abs().max().item() < 2e-2, "flex keep-all should match flash within bf16 noise"

    # [2] backward with a budget: router gets gradient, keep-all at init
    m.train()
    o = m(ids, labels=ids.clone(), doc_lens=torch.tensor([lens], device=dev), max_doc_lens=[max(lens)])
    o.loss.backward()
    g = m.blocks["0"].attention._kvr_router.w.bias.grad
    print(f"[2] loss={o.loss.item():.3f} router bias grad={g.item():.3e} mean_keep={m._kv_route['holder'].mean_keep(last_forward=False):.3f}")
    assert g is not None and g.item() > 0
    m.zero_grad(set_to_none=True)
    m.eval()

    # [3] eviction + KV cache: prefill (flex + compaction) then a flash decode step == no-cache routed forward
    for li in ("1", "2"):
        m.blocks[li].attention._kvr_router.w.weight.data.normal_(0, 0.5)  # random keep/drop pattern
        m.blocks[li].attention._kvr_router.w.bias.data.fill_(0.0)
    B, P = 2, 300
    prompt = torch.randint(0, 512, (B, P), device=dev)
    nxt = torch.randint(0, 512, (B, 1), device=dev)
    with torch.no_grad():
        full = m(torch.cat([prompt, nxt], 1))[:, -1].float()
        for blk in m.blocks.values():
            a = blk.attention
            a.kv_cache_manager = KVCacheManager(B, P + 8, a.n_kv_heads, a.head_dim, dev)
        m(prompt, logits_to_keep=1)
        lp = {li: blk.attention.kv_cache_manager.cache_leftpad.tolist() for li, blk in m.blocks.items()}
        dec = m(nxt, logits_to_keep=1)[:, -1].float()
    d = (dec - full).abs().max().item()
    print(f"[3] cache leftpad after eviction per layer: {lp}")
    print(f"[3] decode(cached, compacted) vs no-cache routed forward: max|diff|={d:.4f} rel={d / full.abs().max().item():.2e}")
    assert any(v > 0 for v in lp["1"]) and lp["0"] == [0] * B
    assert d / full.abs().max().item() < 3e-2
    for blk in m.blocks.values():
        blk.attention.kv_cache_manager = None

    # [4] timing at 64k packed, dense vs routed (keep 0.25 forced) -- forward+backward
    m2 = build()
    m2.train()
    T = 65536
    ids = torch.randint(0, 512, (1, T), device=dev)
    lens = [8192] * 8
    dl = torch.tensor([lens], device=dev)

    def step(model):
        o = model(ids, labels=ids.clone(), doc_lens=dl, max_doc_lens=[8192])
        o.loss.backward()
        model.zero_grad(set_to_none=True)

    def bench(model, n=3):
        step(model); torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            step(model)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n

    t_dense = bench(m2)
    m2.enable_kv_route(target=0.25)
    for blk in m2.blocks.values():
        blk.attention._kvr_router.w.weight.data.normal_(0, 1.0)
        blk.attention._kvr_router.w.bias.data.fill_(-1.1)  # ~25% keep
    t_route = bench(m2)
    kf = m2._kv_route['holder'].mean_keep(last_forward=False)
    for blk in m2.blocks.values():
        blk.attention._kvr_router.w.bias.data.fill_(10.0)
    t_all = bench(m2)
    print(f"[4] 64k fwd+bwd: dense(flash) {t_dense*1e3:.0f} ms | routed keep-all {t_all*1e3:.0f} ms | routed keep~{kf:.2f} {t_route*1e3:.0f} ms")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
