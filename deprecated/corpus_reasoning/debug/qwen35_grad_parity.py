"""Forward matches HF exactly (cos=1.0 at all lengths) yet training plateaus -> check the BACKWARD.
Load HF + olmo with shared weights, same input+labels, CE loss, backward, compare GRADIENTS.
If olmo grads diverge from HF (esp. embeddings = the same tensor, and GDN params), olmo's GDN
backward is wrong (grads flow with healthy norm but point the wrong way) = the training bug."""
import torch, transformers
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.transformer import TransformerConfig

dev = "cuda"; L = 512
hf = transformers.Qwen3_5ForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B-Base", torch_dtype=torch.float32, attn_implementation="eager").to(dev).train()
olmo = TransformerConfig.qwen3_5_0_8B(
    vocab_size=hf.config.vocab_size, attn_backend=AttentionBackendName.torch).build(init_device="cpu")
olmo.load_state_dict(convert_state_from_hf(hf.config, hf.state_dict(), model_type="qwen3_5_text"), strict=False)
olmo = olmo.to(dev).train()

import os
SPARSE = os.environ.get("SPARSE_LOSS", "0") == "1"  # loss only on the LAST token (mimic completion-only)
g = torch.Generator().manual_seed(3)
x = torch.randint(0, 100000, (1, L), generator=g).to(dev)
y = x.clone()

def ce(logits):
    lg = logits[:, :-1].reshape(-1, logits.shape[-1]).float()
    tg = y[:, 1:].reshape(-1)
    if SPARSE:  # only the final position contributes loss -> grad must flow back through the whole GDN scan
        return torch.nn.functional.cross_entropy(lg[-1:], tg[-1:])
    return torch.nn.functional.cross_entropy(lg, tg)
print(f"SPARSE_LOSS={SPARSE} (loss on {'last token only' if SPARSE else 'all tokens'})")
hf.zero_grad(); olmo.zero_grad()
lh = ce(hf(input_ids=x).logits); lh.backward()
lo = ce(olmo(x)); lo.backward()
print(f"loss: HF={lh.item():.4f} olmo={lo.item():.4f}  (forward already proven equal)")

def find(model, *subs):
    for n, p in model.named_parameters():
        if all(s in n for s in subs) and p.grad is not None: return n, p
    return None, None

def cmp(tag, hn, hp, on, op):
    if hp is None or op is None: print(f"  {tag:16s} (missing: HF={hn} olmo={on})"); return
    a, b = hp.grad.flatten().float(), op.grad.flatten().float()
    if a.shape != b.shape: print(f"  {tag:16s} shape mismatch {tuple(hp.shape)} vs {tuple(op.shape)}"); return
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = (a-b).norm().item()/(a.norm().item()+1e-9)
    print(f"  {tag:16s} cos={cos:+.4f} rel={rel:.3f} |HF|={a.norm():.2e} |olmo|={b.norm():.2e}"
          + ("  <-- BACKWARD DIVERGES" if cos < 0.95 else ""))

print("grad cosine olmo-vs-HF (1.0 = identical backward):")
cmp("embed", *find(hf, "embed_tokens", "weight"), *find(olmo, "embeddings", "weight"))
cmp("lm_head", *find(hf, "lm_head", "weight"), *find(olmo, "lm_head", "w_out"))
# layer 0 is GDN (linear_attn) in Qwen3.5 (3:1 GDN:full); pick GDN-specific + a full-attn layer
cmp("GDN A_log", *find(hf, "layers.0", "A_log"), *find(olmo, "blocks.0", "A_log"))
cmp("GDN in_proj", *find(hf, "layers.0", "in_proj_qkv"), *find(olmo, "blocks.0", "in_proj_qkv"))
cmp("GDN conv1d", *find(hf, "layers.0", "conv1d", "weight"), *find(olmo, "blocks.0", "conv1d", "weight"))
cmp("mlp.0", *find(hf, "layers.0", "gate_proj"), *find(olmo, "blocks.0", "w1"))
