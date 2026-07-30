"""Inspect olmo's tokenized contradiction cache (the data my manual loop ALSO plateaus on, while
axolotl learns from the same jsonl). For the first few instances: real token count vs seq_len
(truncation?), the SUPERVISED (label_mask=True) span decoded, and whether the gold answer string is
recoverable. Plus: overfit ONE contradiction instance — if it does NOT reach ~0, the per-example
labels are broken; if it does, the plateau is an aggregate/data-content issue.
"""
import json, sys
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.data import NumpyPaddedFSLDatasetConfig, TokenizerConfig
from olmo_core.data.types import NumpyDatasetDType

dev = "cuda"
CKPT = "/scratch/users/prasann/olmo_ckpts/converted/Qwen_Qwen3.5-0.8B-Base_248320/model_and_optim"
VOCAB = 248320
PREFIX = sys.argv[1]
SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
tkz = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
meta = json.load(open(PREFIX + "_meta.json"))
PAD, EOS = meta["pad_token_id"], meta["eos_token_id"]
print(f"meta eos={EOS} pad={PAD}", flush=True)

tok = TokenizerConfig(vocab_size=VOCAB, eos_token_id=EOS, pad_token_id=PAD,
                      bos_token_id=None, identifier="Qwen/Qwen3.5-0.8B-Base")
ds = NumpyPaddedFSLDatasetConfig(
    paths=[PREFIX + "_tokens.npy"], label_mask_paths=[PREFIX + "_label_mask.npy"],
    sequence_length=SEQ, tokenizer=tok, dtype=NumpyDatasetDType.uint32,
    work_dir="/tmp/inspect_wd").build()
ds.prepare()
print(f"instances={len(ds)}", flush=True)

for i in range(3):
    d = ds[i]
    ids = np.asarray(d["input_ids"]); lm = np.asarray(d["label_mask"]).astype(bool)
    real = int((ids != PAD).sum())
    sup = ids[lm]
    print(f"\n--- instance {i}: real_tokens={real}/{SEQ}  truncated={real>=SEQ}  supervised={lm.sum()} ---", flush=True)
    print("  SUPERVISED span decoded: ", repr(tkz.decode(sup[:120])), flush=True)
    # last 200 non-pad tokens (tail = where prompt+answer live)
    nonpad = ids[ids != PAD]
    print("  TAIL(last 160 nonpad): ", repr(tkz.decode(nonpad[-160:])), flush=True)

# overfit ONE contradiction instance
print("\n=== overfit ONE contradiction instance (idx 0) ===", flush=True)
d = ds[0]
ids = torch.as_tensor(np.asarray(d["input_ids"]), dtype=torch.long)[None].to(dev)
lm = torch.as_tensor(np.asarray(d["label_mask"]), dtype=torch.bool)[None].to(dev)
labels = torch.full_like(ids, -100); labels[:, :-1] = ids[:, 1:]
lp = lm[:, 1:]; labels[:, :-1][~lp] = -100
nloss = int((labels != -100).sum())
print(f"  supervised loss tokens={nloss}", flush=True)
m = TransformerConfig.qwen3_5_0_8B(vocab_size=VOCAB, attn_backend=AttentionBackendName.torch).build().to(dev)
load_model_and_optim_state(CKPT, m); m.train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.0)
for s in range(60):
    opt.zero_grad(set_to_none=True)
    out = m(ids, labels=labels, loss_reduction="sum", loss_div_factor=nloss)
    loss = out.loss if hasattr(out, "loss") else out[1]
    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9); opt.step()
    if s < 8 or s % 10 == 0: print(f"  step {s:3d} loss={loss.item():.4f}", flush=True)
print(f"  FINAL single-contradiction loss={loss.item():.4f} "
      f"({'per-example labels OK' if loss.item()<0.1 else 'PER-EXAMPLE LABELS BROKEN'})", flush=True)
