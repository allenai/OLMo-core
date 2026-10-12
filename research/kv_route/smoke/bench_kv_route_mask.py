"""Micro-bench: where does the routed-attention time go at 64k? (block-mask creation vs flex kernel vs flash)."""
import time, torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from flash_attn import flash_attn_func
dev = torch.device("cuda"); T = 65536; B = 1; H = 32; Hk = 8; D = 128
q = torch.randn(B, H, T, D, device=dev, dtype=torch.bfloat16); k = torch.randn(B, Hk, T, D, device=dev, dtype=torch.bfloat16); v = torch.randn_like(k)
doc = torch.arange(T, device=dev) // 8192
def timeit(fn, n=3):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3
fa = timeit(lambda: flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True))
print(f"flash dense causal fwd: {fa:.1f} ms")
cflex = torch.compile(flex_attention)
for keep_frac in (1.0, 0.25, 0.05):
    keep = torch.rand(B, T, device=dev) < keep_frac
    def mm(b, h, qi, ki):
        return (ki <= qi) & (keep[b, ki] | (ki == qi)) & (doc[qi] == doc[ki])
    for comp in (False, True):
        t_mask = timeit(lambda: create_block_mask(mm, B, None, T, T, device=dev, BLOCK_SIZE=(128, 128), _compile=comp))
        bm = create_block_mask(mm, B, None, T, T, device=dev, BLOCK_SIZE=(128, 128), _compile=comp)
        t_attn = timeit(lambda: cflex(q, k, v, block_mask=bm, enable_gqa=True))
        print(f"keep={keep_frac:.2f} compile_mask={comp}: create_block_mask {t_mask:.1f} ms, flex fwd {t_attn:.1f} ms, sparsity {bm.sparsity():.1f}%")
# compacted-key alternative: gather kept keys, causal on original positions (block-sparse like causal)
for keep_frac in (0.25, 0.05):
    keep = torch.rand(T, device=dev) < keep_frac
    idx = keep.nonzero().squeeze(1); K = idx.numel(); Kp = (K + 127) // 128 * 128
    pos_k = torch.full((Kp,), T + 1, device=dev); pos_k[:K] = idx
    kk = torch.zeros(B, Hk, Kp, D, device=dev, dtype=k.dtype); kk[:, :, :K] = k[:, :, idx]; vv = torch.zeros_like(kk); vv[:, :, :K] = v[:, :, idx]
    def mm2(b, h, qi, kj):
        return (pos_k[kj] <= qi) & (doc[qi] == doc[pos_k[kj].clamp(max=T - 1)])
    t_mask = timeit(lambda: create_block_mask(mm2, B, None, T, Kp, device=dev, BLOCK_SIZE=(128, 128), _compile=True))
    bm = create_block_mask(mm2, B, None, T, Kp, device=dev, BLOCK_SIZE=(128, 128), _compile=True)
    t_attn = timeit(lambda: cflex(q, kk, vv, block_mask=bm, enable_gqa=True))
    print(f"COMPACTED keep={keep_frac:.2f}: K={K} mask {t_mask:.1f} ms, flex fwd {t_attn:.1f} ms, sparsity {bm.sparsity():.1f}%")
