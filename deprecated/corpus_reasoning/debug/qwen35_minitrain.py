"""DECISIVE data-vs-trainer split. The model is PROVEN to overfit a single batch to ~0 (manual &
fused). Here we run OUR OWN manual training loop (plain AdamW, grad-accum, NO olmo trainer / FSDP /
SkipStep) over the REAL contradiction dataset the failing olmo run used. Replicates the trainer's
batch math: micro_batch=2, accum=4 (eff 8), fused loss summed / total-loss-tokens (= mean), lr 1e-5.
  - descends toward axolotl (~0.35)  -> olmo's TRAINER (FSDP/grad-accum/SkipStep) is the bug.
  - plateaus like olmo (~0.62)       -> the DATA (tokenization/label_mask) is the bug.
"""
import json, sys, math
import numpy as np, torch
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.data import NumpyPaddedFSLDatasetConfig, TokenizerConfig
from olmo_core.data.types import NumpyDatasetDType

dev = "cuda"
CKPT = "/scratch/users/prasann/olmo_ckpts/converted/Qwen_Qwen3.5-0.8B-Base_248320/model_and_optim"
VOCAB = 248320
PREFIX = sys.argv[1]   # cache prefix without _tokens.npy
SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
MICRO, ACCUM = 2, 4

meta = json.load(open(PREFIX + "_meta.json"))
tok = TokenizerConfig(vocab_size=VOCAB, eos_token_id=meta["eos_token_id"],
                      pad_token_id=meta["pad_token_id"], bos_token_id=None,
                      identifier="Qwen/Qwen3.5-0.8B-Base")
ds = NumpyPaddedFSLDatasetConfig(
    paths=[PREFIX + "_tokens.npy"], label_mask_paths=[PREFIX + "_label_mask.npy"],
    sequence_length=SEQ, tokenizer=tok, dtype=NumpyDatasetDType.uint32,
    work_dir="/tmp/minitrain_wd").build()
ds.prepare()
N = len(ds)
print(f"dataset instances={N}  seq_len={SEQ}  lr={LR} epochs={EPOCHS}", flush=True)

m = TransformerConfig.qwen3_5_0_8B(vocab_size=VOCAB, attn_backend=AttentionBackendName.torch).build().to(dev)
load_model_and_optim_state(CKPT, m)
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

def get(i):
    d = ds[i]
    ids = torch.as_tensor(np.asarray(d["input_ids"]), dtype=torch.long)
    lmask = torch.as_tensor(np.asarray(d["label_mask"]), dtype=torch.bool)
    labels = torch.full_like(ids, -100)
    labels[:-1] = ids[1:]
    lp = lmask[1:]
    labels[:-1][~lp] = -100
    return ids, labels

g = torch.Generator().manual_seed(0)
total_steps = (N // (MICRO * ACCUM)) * EPOCHS
warmup = max(1, int(0.1 * total_steps))
def lr_at(step):  # cosine w/ warmup, matching olmo
    if step < warmup: return LR * step / warmup
    p = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * LR * (1 + math.cos(math.pi * p))

step = 0
for ep in range(EPOCHS):
    order = torch.randperm(N, generator=g).tolist()
    for b in range(0, N - MICRO * ACCUM + 1, MICRO * ACCUM):
        idxs = order[b:b + MICRO * ACCUM]
        micros = [idxs[k:k + MICRO] for k in range(0, len(idxs), MICRO)]
        # total loss tokens across the effective batch (= olmo's loss_div_factor)
        batch = [get(i) for i in idxs]
        ntot = sum(int((lb != -100).sum()) for _, lb in batch)
        for pg in opt.param_groups: pg["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        run_loss = 0.0
        for mb in micros:
            ids = torch.stack([batch[idxs.index(i)][0] for i in mb]).to(dev)
            lbs = torch.stack([batch[idxs.index(i)][1] for i in mb]).to(dev)
            out = m(ids, labels=lbs, loss_reduction="sum", loss_div_factor=ntot)
            loss = out.loss if hasattr(out, "loss") else out[1]
            loss.backward()
            run_loss += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)  # match olmo max_grad_norm=1.0
        opt.step()
        if step < 12 or step % 25 == 0:
            print(f"  ep{ep} step {step:4d}/{total_steps}  loss={run_loss:.4f}  gn={gn.item():.2f}  lr={lr_at(step):.2e}", flush=True)
        step += 1
print(f"\nFINAL run_loss={run_loss:.4f}  (axolotl~0.35 => trainer-bug; ~0.62 plateau => data-bug)", flush=True)
