"""Single-GPU memory profile of the three routers on Qwen3-4B with FULL activation checkpointing:
peak memory of fwd+bwd on one packed row (T tokens, 8 docs) for dense / kv / kv+ffn(L12+) / kv+ffn(all) /
kv+ffn(all)+blockskip, plus the largest live allocations at the peak grouped by source file.
    python debug/flop_scaling/mem_profile_routers.py [T]"""
import sys, gc, collections, torch
sys.path.insert(0, "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/src")
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig, TransformerActivationCheckpointingMode
T = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
dev = torch.device("cuda")

def build():
    cfg = TransformerConfig.qwen3_4B(vocab_size=151936)
    cfg.apply(lambda c: setattr(c, "backend", AttentionBackendName("flash_2")) if hasattr(c, "backend") else None)
    m = cfg.build(init_device="cuda").to(torch.bfloat16); m.init_weights()
    return m

def run(name, enable):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = build(); enable(m)
    m.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)
    m.train()
    base = torch.cuda.memory_allocated()
    ids = torch.randint(0, 150000, (1, T), device=dev); lens = [T // 8] * 8
    torch.cuda.memory._record_memory_history(max_entries=200000)
    out = m(ids, labels=ids.clone(), doc_lens=torch.tensor([lens], device=dev), max_doc_lens=[T // 8])
    fwd_alloc = torch.cuda.memory_allocated() - base
    out.loss.backward()
    peak = torch.cuda.max_memory_allocated() - base
    snap = torch.cuda.memory._snapshot(); torch.cuda.memory._record_memory_history(enabled=None)
    # live blocks at snapshot time (after backward) are small; instead attribute the largest allocations in the trace
    by_src = collections.Counter()
    for seg in snap["segments"]:
        for b in seg["blocks"]:
            if b["state"] != "active_allocated": continue
            frames = b.get("frames", [])
            src = next((f["filename"].split("/")[-1] + ":" + str(f["line"]) for f in frames if "olmo_core" in f["filename"]), frames[0]["filename"].split("/")[-1] if frames else "?")
            by_src[src] += b["size"]
    print(f"{name:28s} after-fwd +{fwd_alloc/2**30:.2f} GB  peak +{peak/2**30:.2f} GB (params {base/2**30:.1f} GB) | live-after-bwd top: " +
          ", ".join(f"{k} {v/2**30:.2f}GB" for k, v in by_src.most_common(4)), flush=True)
    del m, out; gc.collect(); torch.cuda.empty_cache()

run("dense", lambda m: None)
run("kv", lambda m: m.enable_kv_route(target=0.5))
run("kv+ffn L12+", lambda m: (m.enable_kv_route(target=0.5), m.enable_nested_ffn_moe(start_layer=12, divisors=(1, 16, 64, 256, 1024, 9728), width_multiple=1, target_cost=0.1)))
run("kv+ffn all", lambda m: (m.enable_kv_route(target=0.5), m.enable_nested_ffn_moe(start_layer=0, divisors=(1, 16, 64, 256, 1024, 9728), width_multiple=1, target_cost=0.1)))
run("kv+ffn all+skip", lambda m: (m.enable_kv_route(target=0.5), m.enable_nested_ffn_moe(start_layer=0, divisors=(1, 16, 64, 256, 1024, 9728), width_multiple=1, target_cost=0.1), m.enable_block_skip(target=0.5)))
print("PROFILE DONE")
