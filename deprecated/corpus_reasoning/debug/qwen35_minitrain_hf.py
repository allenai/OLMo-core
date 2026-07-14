"""HEAD-TO-HEAD with qwen35_minitrain.py: identical manual loop (plain AdamW, grad-accum, NO trainer)
on the SAME olmo-tokenized contradiction data, but using the HF/fla Qwen3.5 model instead of olmo's
GatedDeltaNet. olmo's loop plateaus ~0.61; axolotl (HF) reaches ~0.35.
  - HF loop descends ~0.35  -> olmo's GDN backward DRIFTS under training (the bug is the impl).
  - HF loop also plateaus    -> not the impl; the regime/lr is the story, axolotl 0.35 came from elsewhere.
"""
import json, sys, math
import numpy as np, torch, torch.nn.functional as F
import transformers
from olmo_core.data import NumpyPaddedFSLDatasetConfig, TokenizerConfig
from olmo_core.data.types import NumpyDatasetDType

dev = "cuda"
VOCAB = 248320
PREFIX = sys.argv[1]
SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 1
MICRO, ACCUM = 2, 4

meta = json.load(open(PREFIX + "_meta.json"))
EOS, PAD = meta["eos_token_id"], meta["pad_token_id"]
tok = TokenizerConfig(vocab_size=VOCAB, eos_token_id=EOS, pad_token_id=PAD,
                      bos_token_id=None, identifier="Qwen/Qwen3.5-0.8B-Base")
ds = NumpyPaddedFSLDatasetConfig(
    paths=[PREFIX + "_tokens.npy"], label_mask_paths=[PREFIX + "_label_mask.npy"],
    sequence_length=SEQ, tokenizer=tok, dtype=NumpyDatasetDType.uint32,
    work_dir="/tmp/minitrain_hf_wd").build()
ds.prepare()
N = len(ds)
print(f"[HF] dataset instances={N} seq={SEQ} lr={LR} epochs={EPOCHS}", flush=True)

m = transformers.Qwen3_5ForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B-Base", torch_dtype=torch.float32, attn_implementation="eager").to(dev).train()
opt = torch.optim.AdamW(m.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

def get(i):
    d = ds[i]
    ids = torch.as_tensor(np.asarray(d["input_ids"]), dtype=torch.long)
    lmask = torch.as_tensor(np.asarray(d["label_mask"]), dtype=torch.bool)
    return ids, lmask

def mce_sum(logits, ids, lmask):
    # sum CE over supervised next-token positions
    lg = logits[:, :-1].reshape(-1, logits.shape[-1]).float()
    tg = ids[:, 1:].reshape(-1)
    mk = lmask[:, 1:].reshape(-1)
    return F.cross_entropy(lg[mk], tg[mk], reduction="sum")

g = torch.Generator().manual_seed(0)
total_steps = (N // (MICRO * ACCUM)) * EPOCHS
warmup = max(1, int(0.1 * total_steps))
def lr_at(s):
    if s < warmup: return LR * s / warmup
    p = (s - warmup) / max(1, total_steps - warmup)
    return 0.5 * LR * (1 + math.cos(math.pi * p))

step = 0
for ep in range(EPOCHS):
    order = torch.randperm(N, generator=g).tolist()
    for b in range(0, N - MICRO * ACCUM + 1, MICRO * ACCUM):
        idxs = order[b:b + MICRO * ACCUM]
        batch = {i: get(i) for i in idxs}
        ntot = sum(int(batch[i][1][1:].sum()) for i in idxs)
        for pg in opt.param_groups: pg["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        run = 0.0
        for k in range(0, len(idxs), MICRO):
            mb = idxs[k:k + MICRO]
            ids = torch.stack([batch[i][0] for i in mb]).to(dev)
            lmask = torch.stack([batch[i][1] for i in mb]).to(dev)
            loss = mce_sum(m(input_ids=ids).logits, ids, lmask) / ntot
            loss.backward(); run += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step < 12 or step % 25 == 0:
            print(f"  [HF] ep{ep} step {step:4d}/{total_steps} loss={run:.4f} gn={gn.item():.2f} lr={lr_at(step):.2e}", flush=True)
        step += 1
print(f"\n[HF] FINAL run_loss={run:.4f} (~0.35 => olmo GDN impl is the bug; ~0.61 => not the impl)", flush=True)
