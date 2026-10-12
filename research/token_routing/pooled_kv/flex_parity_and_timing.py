"""GPU validation for PooledDocKVAttention's FlexAttention fast path.

1. PARITY: the block-sparse flex path must match the dense materialized-mask path (which the CPU
   unit tests already prove exactly equivalent to full attention over the perturbed corpus).
2. TIMING: rough fwd+bwd wall-clock of pooled (flex) vs plain causal SDPA vs the dense-mask
   fallback at a few sequence lengths, keep_prob=0.1.

Run on a GPU node with the node-local env, e.g.::

    srun -p berkeleynlp -w horton --gres=gpu:1 -t 30 \
      /data/prasann/conda/envs/corpus-reasoning-olmo/bin/python \
      research/token_routing/pooled_kv/flex_parity_and_timing.py
"""

import time

import torch

from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType

D_MODEL, N_HEADS, N_KV, HEAD_DIM = 1024, 16, 4, 64


def build(seed=0):
    torch.manual_seed(seed)
    cfg = AttentionConfig(
        name=AttentionType.pooled_doc_kv,
        n_heads=N_HEADS,
        n_kv_heads=N_KV,
        head_dim=HEAD_DIM,
        bias=False,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
        rope=RoPEConfig(name=RoPEType.default, theta=10_000),
        pooled_keep_prob=0.1,
    )
    return cfg.build(D_MODEL, layer_idx=0, n_layers=1).cuda()


def make_chunk_ids(T, doc_len=128, free_head=64, free_tail=256):
    n_docs = (T - free_head - free_tail) // doc_len
    ids = [-1] * free_head
    for d in range(n_docs):
        ids += [d] * doc_len
    ids += [-1] * (T - len(ids))
    return torch.tensor([ids], dtype=torch.int32, device="cuda"), n_docs


def bench(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    print(f"device: {torch.cuda.get_device_name()}", flush=True)
    # ---- parity at T=8192 (>= _FLEX_MIN_SEQ_LEN so the flex path engages), fp32 ----
    T = 8192
    attn = build()
    chunk_ids, n_docs = make_chunk_ids(T)
    keep = torch.rand(1, n_docs, device="cuda") < 0.1
    attn._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep)
    attn.eval()
    x = torch.randn(1, T, D_MODEL, device="cuda")
    with torch.no_grad():
        out_flex = attn(x, chunk_ids=chunk_ids)
        assert not attn._force_eager_mask, "flex path fell back -- parity test is vacuous"
        attn._force_eager_mask = True
        out_dense = attn(x, chunk_ids=chunk_ids)
        attn._force_eager_mask = False
    diff = (out_flex - out_dense).abs().max().item()
    print(f"[parity] T={T} n_docs={n_docs} kept={int(keep.sum())} max|flex-dense|={diff:.3e}")
    assert diff < 5e-4, f"flex/dense mismatch: {diff}"

    # ---- timing: pooled(flex) vs plain causal vs dense-mask fallback, fwd+bwd, bf16 ----
    for T in (8192, 16384, 32768):
        attn = build()
        attn = attn.to(torch.bfloat16)
        attn.train()
        chunk_ids, n_docs = make_chunk_ids(T)
        keep = torch.rand(1, n_docs, device="cuda") < 0.1
        attn._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep)
        x = torch.randn(1, T, D_MODEL, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        def fwd_bwd_pooled():
            attn(x, chunk_ids=chunk_ids).sum().backward()
            x.grad = None

        def fwd_bwd_causal():
            attn(x).sum().backward()
            x.grad = None

        t_pooled = bench(fwd_bwd_pooled)
        t_causal = bench(fwd_bwd_causal)
        line = (
            f"[timing] T={T} n_docs={n_docs} kept={int(keep.sum())}: "
            f"pooled(flex)={t_pooled*1e3:.1f}ms causal-SDPA={t_causal*1e3:.1f}ms "
            f"ratio={t_causal/t_pooled:.2f}x"
        )
        if T <= 16384:
            attn._force_eager_mask = True
            t_dense = bench(fwd_bwd_pooled)
            attn._force_eager_mask = False
            line += f" dense-mask={t_dense*1e3:.1f}ms"
        print(line, flush=True)

    print("OK", flush=True)


if __name__ == "__main__":
    main()
