"""Quick: load an olmo converted ckpt and report init masked-CE on a contradiction batch.
Correct conversion -> ~0.82; corrupt (block-0 random init) -> ~6.8. No HF needed."""
import json, sys
import numpy as np, torch, torch.nn.functional as F
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.data import NumpyPaddedFSLDatasetConfig, TokenizerConfig
from olmo_core.data.types import NumpyDatasetDType

dev = "cuda"; VOCAB = 248320
CKPT, PREFIX = sys.argv[1], sys.argv[2]
meta = json.load(open(PREFIX + "_meta.json"))
tok = TokenizerConfig(vocab_size=VOCAB, eos_token_id=meta["eos_token_id"], pad_token_id=meta["pad_token_id"],
                      bos_token_id=None, identifier="Qwen/Qwen3.5-0.8B-Base")
ds = NumpyPaddedFSLDatasetConfig(paths=[PREFIX + "_tokens.npy"], label_mask_paths=[PREFIX + "_label_mask.npy"],
    sequence_length=4096, tokenizer=tok, dtype=NumpyDatasetDType.uint32, work_dir="/tmp/il_wd").build()
ds.prepare(); d = ds[0]
ids = torch.as_tensor(np.asarray(d["input_ids"]), dtype=torch.long)[None].to(dev)
lm = torch.as_tensor(np.asarray(d["label_mask"]), dtype=torch.bool)[None].to(dev)
m = TransformerConfig.qwen3_5_0_8B(vocab_size=VOCAB, attn_backend=AttentionBackendName.torch).build().to(dev)
load_model_and_optim_state(CKPT, m); m.eval()
ffn0 = dict(m.named_parameters())["blocks.0.feed_forward.w1.weight"].float().norm().item()
with torch.no_grad():
    lg = m(ids)[:, :-1].reshape(-1, VOCAB).float(); tg = ids[:, 1:].reshape(-1); mk = lm[:, 1:].reshape(-1)
    loss = F.cross_entropy(lg[mk], tg[mk]).item()
print(f"INIT-LOSS={loss:.4f}  block0_ffn_norm={ffn0:.4f}  -> {'OK (correct conversion)' if loss<3 else 'STILL CORRUPT'}", flush=True)
