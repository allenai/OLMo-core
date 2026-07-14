"""Does olmo's FUSED internal loss (model(ids, labels=...), the path training uses) produce the
SAME gradient as plain logits->cross_entropy (the path my grad-parity validated == HF)? If not,
the fused lm_head loss is the bug. All olmo, no HF/export confound."""
import glob, json, types
import numpy as np, torch, torch.nn.functional as F
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.convert import convert_qwen3_5_state_from_hf
from olmo_core.nn.transformer import TransformerConfig
dev="cuda"
PRE="data/.cache/olmo/niah_contradiction_train_n49_p1_retrieval_qboth_069b1b99ab117809"
meta=json.load(open(PRE+"_meta.json"))
t=np.fromfile(PRE+"_tokens.npy",dtype=np.uint32); lm=np.fromfile(PRE+"_label_mask.npy",dtype=np.bool_)
L=2304
ids=torch.tensor(t[:L].astype(np.int64))[None].to(dev); mask=torch.tensor(lm[:L])[None].to(dev)
snap=snapshot_download("Qwen/Qwen3.5-0.8B-Base"); raw=json.load(open(snap+"/config.json"))
cfg=types.SimpleNamespace(**raw); cfg.text_config=types.SimpleNamespace(**raw["text_config"])
hf_state={}
for s in sorted(glob.glob(snap+"/*.safetensors")): hf_state.update(load_file(s))
m=TransformerConfig.qwen3_5_0_8B(vocab_size=raw["text_config"]["vocab_size"],attn_backend=AttentionBackendName.torch).build().to(dev).train()
m.load_state_dict(convert_qwen3_5_state_from_hf(cfg,hf_state),strict=False)
# PRE-SHIFTED labels (olmo lm_head does NOT shift internally): labels[i]=ids[i+1], -100 elsewhere
labels=torch.full_like(ids,-100)
labels[:,:-1]=ids[:,1:]
loss_pos=mask[:,1:]                     # mask aligned to next-token targets
labels[:,:-1][~loss_pos]=-100
nloss=int(loss_pos.sum())
m.zero_grad()
outF=m(ids, labels=labels, loss_reduction="sum", loss_div_factor=float(nloss))
lossF = outF.loss if hasattr(outF,"loss") else outF[1]
lossF.backward()
gF={n:p.grad.detach().clone() for n,p in m.named_parameters() if p.grad is not None}
# Path MANUAL (validated == HF): logits -> mean CE on shifted mask
m.zero_grad()
out=m(ids, return_logits=True)
logits=(out.logits if hasattr(out,"logits") else out).float()
lg=logits[:,:-1].reshape(-1,logits.shape[-1]); tg=ids[:,1:].reshape(-1); mk=mask[:,1:].reshape(-1)
lossM=F.cross_entropy(lg[mk], tg[mk])
lossM.backward()
gM={n:p.grad.detach().clone() for n,p in m.named_parameters() if p.grad is not None}
print(f"loss FUSED={lossF.item():.4f}  MANUAL={lossM.item():.4f}  (should match)")
def kind(n):
    if any(s in n for s in("A_log","dt_bias","conv1d","in_proj_a","in_proj_b")):return "GDN"
    if "feed_forward" in n or n.endswith((".w1.weight",".w2.weight",".w3.weight")):return "mlp"
    if any(s in n for s in("w_q","w_k","w_v","q_norm","k_norm")):return "softmax"
    if "w_out" in n or "embed" in n:return "head/embed"
    return "other"
from collections import defaultdict
agg=defaultdict(lambda:[0.0,0.0,0.0,0])
for n in gF:
    if n not in gM: continue
    a,b=gF[n].flatten().float(),gM[n].flatten().float()
    c=F.cosine_similarity(a,b,dim=0).item()
    k=kind(n); agg[k][0]+=min(agg[k][0] if agg[k][3] else 9,c) if False else 0
    agg[k][1]+=a.norm().item()**2; agg[k][2]+=b.norm().item()**2; agg[k][3]+=1
    agg[k][0]=agg[k][0]+c
print("per-group  meanCos(fused vs manual)   |fused|   |manual|")
for k,(cs,fs,ms,cnt) in sorted(agg.items()):
    print(f"  {k:12s} cos~{cs/cnt:+.4f}  |F|={fs**.5:.2e} |M|={ms**.5:.2e}  ({cnt})")
