"""HF side: load HF Qwen3.5-0.8B-Base under the transformers-5.x shadow, run the SAME fixed inputs
(from olmo side's saved ids) at each length, save logits; then print olmo-vs-HF divergence vs length."""
import numpy as np, torch, transformers
print("transformers", transformers.__version__, "has Qwen3_5:", hasattr(transformers, "Qwen3_5ForCausalLM"))
dev = "cuda"
o = np.load("/data/prasann/runtime/olmo_q35_logits.npz")
LENS = [8, 128, 1024, 4096, 8192]
cfg = transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
tc.tie_word_embeddings = getattr(tc, "tie_word_embeddings", True)
hf = transformers.Qwen3_5ForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B-Base", torch_dtype=torch.float32, attn_implementation="eager").to(dev).eval()
print("\nlen   olmo-vs-HF last-token logit:  max|Δ|     rel       cos")
for L in LENS:
    ids = torch.tensor(o["ids_%d" % L][None, :]).to(dev)
    with torch.no_grad():
        hl = hf(input_ids=ids).logits.float().cpu().numpy()[0, -1]
    ol = o[str(L)]
    md = np.abs(hl - ol).max(); rel = md / (np.abs(hl).max() + 1e-9)
    cos = float(np.dot(hl, ol) / (np.linalg.norm(hl) * np.linalg.norm(ol) + 1e-9))
    flag = "  <-- DIVERGES" if rel > 0.05 else ""
    print(f"{L:5d}  max|Δ|={md:.3e}  rel={rel:.3e}  cos={cos:.5f}{flag}")
print("\nIf rel/cos worsen as length grows -> olmo GDN forward diverges from HF at length "
      "(the 8-token parity test misses it) = the training bug.")
