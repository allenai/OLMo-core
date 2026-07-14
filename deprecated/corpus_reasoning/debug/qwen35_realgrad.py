"""Decisive: compare olmo-vs-HF GRADIENTS on a REAL niah batch with completion-only loss (the
actual training objective). grad-parity matched on random+mean-CE; if the real structured batch +
masked loss makes olmo's GDN grad diverge from HF, that's the training bug."""
import glob, json, os, types
import numpy as np, torch, torch.nn.functional as F, transformers
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.transformer import TransformerConfig

dev="cuda"
PRE="data/.cache/olmo/niah_contradiction_train_n49_p1_retrieval_qboth_069b1b99ab117809"
meta=json.load(open(PRE+"_meta.json")); PAD=meta["pad_token_id"]
t=np.fromfile(PRE+"_tokens.npy",dtype=np.uint32); lm=np.fromfile(PRE+"_label_mask.npy",dtype=np.bool_)
# first example = up to first 2149 real tokens (use a 2304 window covering ex0)
L=2304
ids=torch.tensor(t[:L].astype(np.int64))[None].to(dev)
mask=torch.tensor(lm[:L])[None].to(dev)
print("loss tokens in window:", int(mask.sum()))

hf=transformers.Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B-Base",torch_dtype=torch.float32,attn_implementation="eager").to(dev).train()
olmo=TransformerConfig.qwen3_5_0_8B(vocab_size=hf.config.vocab_size,attn_backend=AttentionBackendName.torch).build().to(dev)
olmo.load_state_dict(convert_state_from_hf(hf.config,hf.state_dict(),model_type="qwen3_5_text"),strict=False)
olmo=olmo.train()

def mce(logits):
    lg=logits[:,:-1].reshape(-1,logits.shape[-1]).float(); tg=ids[:,1:].reshape(-1); mk=mask[:,1:].reshape(-1)
    return F.cross_entropy(lg[mk], tg[mk])
hf.zero_grad(); olmo.zero_grad()
lh=mce(hf(input_ids=ids).logits); lh.backward()
lo=mce(olmo(ids)); lo.backward()
print(f"masked CE: HF={lh.item():.4f} olmo={lo.item():.4f}")
def find(m,*s):
    for n,p in m.named_parameters():
        if all(k in n for k in s) and p.grad is not None: return p
def cmp(tag,hp,op):
    if hp is None or op is None: print(f"  {tag}: missing"); return
    a,b=hp.grad.flatten().float(),op.grad.flatten().float()
    if a.shape!=b.shape: print(f"  {tag}: shape {tuple(hp.shape)} vs {tuple(op.shape)}"); return
    print(f"  {tag}: cos={F.cosine_similarity(a,b,dim=0).item():+.4f} |HF|={a.norm():.2e} |olmo|={b.norm():.2e}")
print("GDN/mlp grad cos olmo-vs-HF on REAL batch:")
cmp("GDN A_log", find(hf,"layers.0","A_log"), find(olmo,"blocks.0","A_log"))
cmp("GDN conv1d", find(hf,"layers.0","conv1d","weight"), find(olmo,"blocks.0","conv1d","weight"))
cmp("mlp.0", find(hf,"layers.0","gate_proj"), find(olmo,"blocks.0","w1"))
cmp("softmax.6 q", find(hf,"layers.6","q_proj"), find(olmo,"blocks.6","w_q"))
