"""Minimal distcp roundtrip: take a CORRECT in-memory-converted olmo model (init loss ~0.82), save
via save_model_and_optim_state to a FRESH dir, reload via load_model_and_optim_state into a fresh
model, recheck init loss + block-0 FFN norm.
  - reload loss ~6.8 / block0 FFN norm changes -> LIVE olmo-core distcp roundtrip bug.
  - reload loss ~0.82 / matches               -> the CACHED ckpt is stale; re-converting fixes it.
"""
import json, sys, os
import numpy as np, torch, torch.nn.functional as F
import transformers
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.distributed.checkpoint import save_model_and_optim_state, load_model_and_optim_state
from olmo_core.data import NumpyPaddedFSLDatasetConfig, TokenizerConfig
from olmo_core.data.types import NumpyDatasetDType

dev = "cuda"
VOCAB = 248320
PREFIX = sys.argv[1]
FRESH = "/data/prasann/tmp_roundtrip_ckpt/model_and_optim"
OLDCK = "/scratch/users/prasann/olmo_ckpts/converted/Qwen_Qwen3.5-0.8B-Base_248320/model_and_optim"
meta = json.load(open(PREFIX + "_meta.json"))
tok = TokenizerConfig(vocab_size=VOCAB, eos_token_id=meta["eos_token_id"], pad_token_id=meta["pad_token_id"],
                      bos_token_id=None, identifier="Qwen/Qwen3.5-0.8B-Base")
ds = NumpyPaddedFSLDatasetConfig(paths=[PREFIX + "_tokens.npy"], label_mask_paths=[PREFIX + "_label_mask.npy"],
    sequence_length=4096, tokenizer=tok, dtype=NumpyDatasetDType.uint32, work_dir="/tmp/rt_wd").build()
ds.prepare(); d = ds[0]
ids = torch.as_tensor(np.asarray(d["input_ids"]), dtype=torch.long)[None].to(dev)
lm = torch.as_tensor(np.asarray(d["label_mask"]), dtype=torch.bool)[None].to(dev)
def mce(lg):
    x = lg[:, :-1].reshape(-1, lg.shape[-1]).float(); tg = ids[:, 1:].reshape(-1); mk = lm[:, 1:].reshape(-1)
    return F.cross_entropy(x[mk], tg[mk]).item()
def ffn0(m):  # block-0 FFN w1 norm
    return dict(m.named_parameters())["blocks.0.feed_forward.w1.weight"].float().norm().item()

# correct in-memory model
hf = transformers.Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B-Base", torch_dtype=torch.float32, attn_implementation="eager")
sd = convert_state_from_hf(hf.config, hf.state_dict(), model_type="qwen3_5_text"); del hf
m0 = TransformerConfig.qwen3_5_0_8B(vocab_size=VOCAB, attn_backend=AttentionBackendName.torch).build().to(dev)
m0.load_state_dict(sd, strict=False); m0.eval()
with torch.no_grad(): l0 = mce(m0(ids))
print(f"[0] in-memory correct model: loss={l0:.4f}  ffn0_norm={ffn0(m0):.4f}", flush=True)

# save fresh, reload
os.makedirs(os.path.dirname(FRESH), exist_ok=True)
save_model_and_optim_state(FRESH, m0, save_overwrite=True)
m1 = TransformerConfig.qwen3_5_0_8B(vocab_size=VOCAB, attn_backend=AttentionBackendName.torch).build().to(dev)
print(f"    fresh-built (pre-load) ffn0_norm={ffn0(m1):.4f}", flush=True)
load_model_and_optim_state(FRESH, m1); m1.eval()
with torch.no_grad(): l1 = mce(m1(ids))
print(f"[1] FRESH save->reload model: loss={l1:.4f}  ffn0_norm={ffn0(m1):.4f}", flush=True)

# the OLD cached ckpt for reference
m2 = TransformerConfig.qwen3_5_0_8B(vocab_size=VOCAB, attn_backend=AttentionBackendName.torch).build().to(dev)
load_model_and_optim_state(OLDCK, m2); m2.eval()
with torch.no_grad(): l2 = mce(m2(ids))
print(f"[2] OLD cached ckpt:          loss={l2:.4f}  ffn0_norm={ffn0(m2):.4f}", flush=True)
print(f"\nVERDICT: {'LIVE distcp roundtrip BUG' if l1>3 else ('cached ckpt was STALE - reconvert fixes it' if l2>3 else 'all fine?!')}", flush=True)
