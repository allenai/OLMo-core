"""Where does the pooled-KV flex time go? Isolate score_mod, block alignment, and keep_prob.

Arms at T=16384 (fwd+bwd, bf16, B=1):
  A. baseline pooled flex (len_bias on, docs straddle blocks, keep=0.1)
  B. no score_mod (len_bias off)
  C. block-aligned docs (free_head=128, doc_len=128 -> no straddle)
  D. keep_prob=0.02 (kept columns are the dominant visited mass)
  E. plain causal SDPA (cuDNN reference)
"""

import time

import torch

from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType

D_MODEL, N_HEADS, N_KV, HEAD_DIM = 1024, 16, 4, 64
T = 16384


def build(len_bias=True, flex_block_size=None):
    torch.manual_seed(0)
    cfg = AttentionConfig(
        name=AttentionType.pooled_doc_kv,
        n_heads=N_HEADS,
        n_kv_heads=N_KV,
        head_dim=HEAD_DIM,
        bias=False,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
        rope=RoPEConfig(name=RoPEType.default, theta=10_000),
        pooled_len_bias=len_bias,
        flex_block_size=flex_block_size,
    )
    return cfg.build(D_MODEL, layer_idx=0, n_layers=1).cuda().to(torch.bfloat16)


def make_chunk_ids(free_head, doc_len=128, free_tail=256):
    n_docs = (T - free_head - free_tail) // doc_len
    ids = [-1] * free_head
    for d in range(n_docs):
        ids += [d] * doc_len
    ids += [-1] * (T - len(ids))
    return torch.tensor([ids], dtype=torch.int32, device="cuda"), n_docs


def bench(fn, iters=5, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def run(tag, len_bias=True, free_head=64, keep_prob=0.1, flex_block_size=None):
    attn = build(len_bias, flex_block_size)
    attn.train()
    chunk_ids, n_docs = make_chunk_ids(free_head)
    torch.manual_seed(1)
    keep = torch.rand(1, n_docs, device="cuda") < keep_prob
    attn._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep)
    x = torch.randn(1, T, D_MODEL, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def f():
        attn(x, chunk_ids=chunk_ids).sum().backward()
        x.grad = None

    print(f"{tag}: {bench(f):.1f}ms (n_docs={n_docs} kept={int(keep.sum())})", flush=True)
    return attn, x


def main():
    print(f"device: {torch.cuda.get_device_name()} T={T}", flush=True)
    run("A baseline (bias, straddle, keep=0.1)")
    run("B no score_mod (len_bias off)     ", len_bias=False)
    run("C block-aligned docs              ", free_head=128)
    run("D keep_prob=0.02                  ", keep_prob=0.02)
    run("D' aligned + no bias + keep=0.02  ", len_bias=False, free_head=128, keep_prob=0.02)
    run("F straddle + block_size=64        ", flex_block_size=64)
    run("G straddle + block_size=32        ", flex_block_size=32)
    attn, x = run("A again (sanity)                  ")

    def causal():
        attn(x).sum().backward()
        x.grad = None

    print(f"E plain causal SDPA               : {bench(causal):.1f}ms", flush=True)


if __name__ == "__main__":
    main()
