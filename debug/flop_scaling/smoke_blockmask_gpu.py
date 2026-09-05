"""Block-level vs dense-materialised BlockMask for kv_route._masked_attention: outputs must match
(bf16 noise), and report time/memory at Qwen3-4B geometry (32q/8kv/128, 64k as 8x8k)."""
import os, sys, time, torch
sys.path.insert(0, "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/src")
from olmo_core.nn.attention.kv_route import _masked_attention, _doc_ids
dev = torch.device("cuda"); Hq, Hk, D = 32, 8, 128
for T, n_docs in ((4096, 3), (65536, 8)):
    q = torch.randn(1, T, Hq, D, device=dev, dtype=torch.bfloat16); k = torch.randn(1, T, Hk, D, device=dev, dtype=torch.bfloat16); v = torch.randn_like(k)
    lens = [T // n_docs] * (n_docs - 1); lens.append(T - sum(lens))
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=dev, dtype=torch.int32); doc = _doc_ids(cu, T)
    for kf in (1.0, 0.3, 0.05):
        keep = torch.rand(1, T, device=dev) < kf
        outs, stats = {}, {}
        for mode in ("dense", "block"):
            os.environ["KV_ROUTE_BLOCKMASK"] = mode
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
            with torch.no_grad(): o = _masked_attention(q, k, v, keep, doc, None, D**-0.5)  # warm/compile
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.no_grad():
                for _ in range(3): o = _masked_attention(q, k, v, keep, doc, None, D**-0.5)
            torch.cuda.synchronize(); stats[mode] = ((time.perf_counter() - t) / 3 * 1e3, (torch.cuda.max_memory_allocated() - base) / 2**30); outs[mode] = o.float()
        d = (outs["dense"] - outs["block"]).abs().max().item() / outs["dense"].abs().max().item()
        print(f"T={T} keep={kf:.2f}: dense {stats['dense'][0]:.1f} ms / peak +{stats['dense'][1]:.2f} GB | block {stats['block'][0]:.1f} ms / +{stats['block'][1]:.2f} GB | rel diff {d:.2e}")
        assert d < 1e-2, "block-level mask changed the output"
print("BLOCKMASK OK")
