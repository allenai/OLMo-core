"""Per-layer fwd+bwd at Qwen3.5-4B attention geometry (32q/8kv heads, D=128, 64k packed as 8x8k):
flash varlen (dense) vs kv_route._masked_attention (keep-all / 0.25 / 0.05), fresh keep tensor per call."""
import sys, time, torch
sys.path.insert(0, "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/src")
from olmo_core.nn.attention.kv_route import _masked_attention, _doc_ids
from flash_attn import flash_attn_varlen_func
dev = torch.device("cuda"); T = 65536; Hq, Hk, D = 32, 8, 128
q = torch.randn(1, T, Hq, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(1, T, Hk, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(1, T, Hk, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
cu = torch.arange(0, T + 1, 8192, device=dev, dtype=torch.int32); doc = _doc_ids(cu, T)
def timeit(fn, n=3):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3
def flash():
    o = flash_attn_varlen_func(q[0], k[0], v[0], cu, cu, 8192, 8192, causal=True); o.sum().backward()
print(f"flash varlen fwd+bwd: {timeit(flash):.1f} ms")
for kf in (1.0, 0.25, 0.05):
    def routed():
        keep = torch.rand(1, T, device=dev) < kf
        o = _masked_attention(q, k, v, keep, doc, None, D**-0.5); o.sum().backward()
    def routed_fwd():
        keep = torch.rand(1, T, device=dev) < kf
        with torch.no_grad(): _masked_attention(q, k, v, keep, doc, None, D**-0.5)
    print(f"routed keep={kf:.2f}: fwd {timeit(routed_fwd):.1f} ms, fwd+bwd {timeit(routed):.1f} ms")
