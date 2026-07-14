"""Correct length-parity: follow the OFFICIAL olmo parity test exactly (load HF once, convert ITS
state_dict into olmo so both sides share identical weights), but sweep sequence length.
If olmo==HF at 8 tokens (official test passes) but diverges as length grows -> olmo GDN long-seq bug.
Needs transformers-with-qwen3_5 (shadow) + amandab olmo_core + fla in ONE process."""
import numpy as np, torch, transformers
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.transformer import TransformerConfig

dev = "cuda"
LENS = [8, 128, 512, 2048, 8192]
print("transformers", transformers.__version__, "Qwen3_5:", hasattr(transformers, "Qwen3_5ForCausalLM"))

hf = transformers.Qwen3_5ForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B-Base", torch_dtype=torch.float32, attn_implementation="eager").to(dev).eval()
olmo = TransformerConfig.qwen3_5_0_8B(
    vocab_size=hf.config.vocab_size, attn_backend=AttentionBackendName.torch).build(init_device="cpu")
converted = convert_state_from_hf(hf.config, hf.state_dict(), model_type="qwen3_5_text")
missing, unexpected = olmo.load_state_dict(converted, strict=False)
print(f"olmo load: missing={len(missing)} unexpected={len(unexpected)}")
olmo = olmo.to(dev).eval()

g = torch.Generator().manual_seed(7)
full = torch.randint(0, 100000, (1, max(LENS)), generator=g).to(dev)
print("\nlen     max|Δ|       rel        cos       (olmo vs HF, last-token logits)")
for L in LENS:
    x = full[:, :L]
    with torch.no_grad():
        h = hf(input_ids=x).logits.float()[0, -1]
        o = olmo(x).float()[0, -1]
    md = (h-o).abs().max().item(); rel = md/(h.abs().max().item()+1e-9)
    cos = torch.nn.functional.cosine_similarity(h, o, dim=0).item()
    print(f"{L:5d}  {md:.3e}  {rel:.3e}  {cos:.5f}  {'<-- DIVERGES' if rel>0.05 else 'OK'}")
