"""GPU smoke for block skipping (olmo_core.nn.block_skip) composed with the KV router on the flex path:
(1) run-all == base (packed); (2) forced skips change outputs, skipped tokens leave the key set, budget grads
flow; (3) prefill+decode with cache == no-cache forward under skipping; (4) joint budget over 3 routers.
    srun -p berkeleynlp --qos=preemptive_high_sewonm --gres=gpu:1 -w horton \
      /data/prasann/conda/envs/corpus-reasoning-olmo/bin/python debug/flop_scaling/smoke_block_skip_gpu.py"""
import sys, torch
sys.path.insert(0, "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/src")
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.attention.kv_cache import KVCacheManager
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.joint_budget import install_joint_budget
dev = torch.device("cuda")

def build(seed=0):
    torch.manual_seed(seed)
    cfg = TransformerConfig.llama_like(d_model=256, n_layers=4, n_heads=8, n_kv_heads=2, vocab_size=512)
    cfg.apply(lambda c: setattr(c, "backend", AttentionBackendName("flash_2")) if hasattr(c, "backend") else None)
    m = cfg.build(init_device="cuda"); m.init_weights(); return m.to(torch.bfloat16)

m = build(); T = 4096; ids = torch.randint(0, 512, (1, T), device=dev); lens = [1500, 1000, 1596]
dl = dict(doc_lens=torch.tensor([lens], device=dev), max_doc_lens=[max(lens)])
with torch.no_grad(): ref = m(ids, **dl).float()
m.enable_block_skip(target=0.5); m.enable_kv_route(target=0.5)
with torch.no_grad(): out = m(ids, **dl).float()
d = (out - ref).abs().max().item() / ref.abs().max().item(); print(f"[1] run-all + keep-all vs base: rel {d:.2e}"); assert d < 2e-2

m.train(); o = m(ids, labels=ids.clone(), **dl); o.loss.backward()
g = m.blocks["0"]._bskip_router.w.bias.grad.item(); print(f"[2] loss {o.loss.item():.3f} skip-router grad {g:.2e} (>0), run {m._block_skip['holder'].mean_keep(last_forward=False):.3f}"); assert g > 0
m.zero_grad(set_to_none=True); m.eval()
for li in ("1", "2"):
    m.blocks[li]._bskip_router.w.weight.data.normal_(0, 0.5); m.blocks[li]._bskip_router.w.bias.data.fill_(0.0)
    m.blocks[li].attention._kvr_router.w.weight.data.normal_(0, 0.5); m.blocks[li].attention._kvr_router.w.bias.data.fill_(0.5)
with torch.no_grad(): out2 = m(ids, **dl).float()
print(f"[2] forced skips: run {m._block_skip['holder'].per_layer_keep(last_forward=False)} kv keep {m._kv_route['holder'].mean_keep(last_forward=False):.3f}; output rel change {(out2-ref).abs().max().item()/ref.abs().max().item():.3f}")

B, P = 2, 300
prompt = torch.randint(0, 512, (B, P), device=dev); nxt = torch.randint(0, 512, (B, 1), device=dev)
with torch.no_grad():
    full = m(torch.cat([prompt, nxt], 1))[:, -1].float()
    for blk in m.blocks.values():
        a = blk.attention; a.kv_cache_manager = KVCacheManager(B, P + 8, a.n_kv_heads, a.head_dim, dev)
    m(prompt, logits_to_keep=1)
    lp = {li: blk.attention.kv_cache_manager.cache_leftpad.tolist() for li, blk in m.blocks.items()}
    dec = m(nxt, logits_to_keep=1)[:, -1].float()
d = (dec - full).abs().max().item() / full.abs().max().item()
print(f"[3] leftpad after prefill (skipped+evicted): {lp}; decode vs no-cache rel {d:.2e}"); assert d < 3e-2 and any(v > 0 for v in lp["1"])
for blk in m.blocks.values(): blk.attention.kv_cache_manager = None

m2 = build(); m2.enable_block_skip(target=0.5); m2.enable_kv_route(target=0.5)
m2.enable_nested_ffn_moe(start_layer=2, divisors=(1, 4, 16), width_multiple=1, target_cost=0.1)
jb = install_joint_budget(m2, target=0.5, seq_len=T); m2.train()
o = m2(ids, labels=ids.clone(), **dl); o.loss.backward()
print(f"[4] joint cost {jb['last_cost']:.3f} target {jb['last_target']:.3f}; grads: skip {m2.blocks['0']._bskip_router.w.bias.grad.item():.2e} kv {m2.blocks['0'].attention._kvr_router.w.bias.grad.item():.2e} ffn {m2.blocks['2'].feed_forward._nffn_router.w.bias.grad.abs().sum().item():.2e}")
print("SMOKE OK")
