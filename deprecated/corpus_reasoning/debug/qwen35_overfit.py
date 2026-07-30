"""DECISIVE model-vs-trainer split: overfit ONE real batch with a MANUAL train loop using olmo's
own forward/backward + plain torch AdamW. A correct model MUST drive a single fixed batch to ~0.
  - plateau ~0.6  -> bug is in olmo MODEL/backward (reproduces with no FSDP/grad-accum/trainer).
  - descends ~0   -> model is fine; bug is in olmo's TRAINER plumbing.
Loads the converted olmo-format base ckpt directly (no transformers / no HF needed).
Runs BOTH the manual-CE path (proven == HF at init) and olmo's FUSED-loss path (the path training uses).
"""
import json, sys
import numpy as np, torch, torch.nn.functional as F
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state

dev = "cuda"
CKPT = "/scratch/users/prasann/olmo_ckpts/converted/Qwen_Qwen3.5-0.8B-Base_248320/model_and_optim"
VOCAB = 248320
PRE = "data/.cache/olmo/niah_contradiction_train_n49_p1_retrieval_qboth_069b1b99ab117809"
meta = json.load(open(PRE + "_meta.json"))
t  = np.fromfile(PRE + "_tokens.npy",     dtype=np.uint32)
lm = np.fromfile(PRE + "_label_mask.npy", dtype=np.bool_)
L = 2304
ids  = torch.tensor(t[:L].astype(np.int64))[None].to(dev)
mask = torch.tensor(lm[:L])[None].to(dev)
nloss = int(mask[:, 1:].sum())
print(f"loss tokens={nloss}  seq_len={L}", flush=True)

def fresh_model():
    m = TransformerConfig.qwen3_5_0_8B(
        vocab_size=VOCAB, attn_backend=AttentionBackendName.torch).build().to(dev)
    load_model_and_optim_state(CKPT, m)
    return m.train()

def manual_ce(logits):
    lg = logits[:, :-1].reshape(-1, logits.shape[-1]).float()
    tg = ids[:, 1:].reshape(-1)
    mk = mask[:, 1:].reshape(-1)
    return F.cross_entropy(lg[mk], tg[mk])

labels = torch.full_like(ids, -100)       # pre-shifted (olmo lm_head expects pre-shifted labels)
labels[:, :-1] = ids[:, 1:]
lp = mask[:, 1:]
labels[:, :-1][~lp] = -100

def run(loss_mode, steps=200, lr=1e-5):
    torch.manual_seed(0)
    m = fresh_model()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    print(f"\n=== overfit one batch | loss_mode={loss_mode} lr={lr} ===", flush=True)
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        if loss_mode == "manual":
            loss = manual_ce(m(ids))
        else:  # fused — exactly the training path
            out = m(ids, labels=labels, loss_reduction="sum", loss_div_factor=nloss)
            loss = out.loss if hasattr(out, "loss") else out[1]
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9)  # measure, don't clip
        opt.step()
        if s < 10 or s % 20 == 0 or s == steps - 1:
            print(f"  step {s:3d}  loss={loss.item():.4f}  gradnorm={gn.item():.3f}", flush=True)
    return loss.item()

mode = sys.argv[1] if len(sys.argv) > 1 else "manual"
lr = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-5
final = run(mode, steps=200, lr=lr)
print(f"\nFINAL loss ({mode}, lr={lr}) = {final:.4f}  "
      f"({'MODEL OK -> bug in trainer' if final < 0.1 else 'PLATEAU -> bug in model/backward'})",
      flush=True)
