"""Decisive padding/doc_lens test: run the olmo Qwen3.5 model on a REAL padded-FSL training batch
(exactly what the train module feeds it) and compare to (a) the same model WITHOUT doc_lens (= my
matching parity path) and (b) HF. If the training-path forward (with doc_lens/padding) diverges at
the answer positions, that's the root cause of the SFT failure."""
import glob, json, os, types
import numpy as np, torch, torch.nn.functional as F, transformers
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from olmo_core.data import NumpyPaddedFSLDatasetConfig, NumpyDatasetDType
from olmo_core.data import TokenizerConfig
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.convert import convert_qwen3_5_state_from_hf
from olmo_core.nn.transformer import TransformerConfig

dev = "cuda"
PRE = "data/.cache/olmo/niah_contradiction_train_n49_p1_retrieval_qboth_069b1b99ab117809"
meta = json.load(open(PRE + "_meta.json"))
SEQ = meta["seq_len"]; PAD = meta["pad_token_id"]; EOS = meta["eos_token_id"]
print(f"seq_len={SEQ} eos={EOS} pad={PAD}")

tok_cfg = TokenizerConfig(vocab_size=meta_vocab if (meta_vocab:=meta.get("vocab_size")) else 248320,
                          eos_token_id=EOS, pad_token_id=PAD, bos_token_id=None,
                          identifier="Qwen/Qwen3.5-0.8B-Base")
ds = NumpyPaddedFSLDatasetConfig(paths=[PRE + "_tokens.npy"], label_mask_paths=[PRE + "_label_mask.npy"],
                                 sequence_length=SEQ, tokenizer=tok_cfg,
                                 dtype=NumpyDatasetDType.uint32, work_dir="/tmp/ds_work").build()
ds.prepare()
item = {k: (v if not torch.is_tensor(v) else v) for k, v in ds[0].items()}
print("batch item keys:", list(item.keys()))
ids = torch.as_tensor(item["input_ids"]).long()
lm = torch.as_tensor(item["label_mask"]).bool() if "label_mask" in item else None
real_len = int((ids != PAD).sum())
print(f"real_len={real_len} of {SEQ} (padding={SEQ-real_len}); label-True={int(lm.sum()) if lm is not None else '?'}")
doc_lens = item.get("doc_lens"); max_doc = item.get("max_doc_lens")
print("doc_lens:", None if doc_lens is None else np.asarray(doc_lens).tolist()[:10], "max_doc:", max_doc)

# build + load model
snap = snapshot_download("Qwen/Qwen3.5-0.8B-Base"); raw = json.load(open(os.path.join(snap, "config.json")))
cfg = types.SimpleNamespace(**raw); cfg.text_config = types.SimpleNamespace(**raw["text_config"])
hf_state = {}
for s in sorted(glob.glob(os.path.join(snap, "*.safetensors"))): hf_state.update(load_file(s))
olmo = TransformerConfig.qwen3_5_0_8B(vocab_size=raw["text_config"]["vocab_size"],
                                      attn_backend=AttentionBackendName.torch).build().to(dev).eval()
olmo.load_state_dict(convert_qwen3_5_state_from_hf(cfg, hf_state), strict=False)

x = ids[None].to(dev)
kw = {}
if doc_lens is not None:
    kw["doc_lens"] = torch.as_tensor(doc_lens)[None].to(dev)
    kw["max_doc_lens"] = int(max_doc) if max_doc is not None else int(np.asarray(doc_lens).max())
with torch.no_grad():
    o_train = olmo(x, **kw).float()         # exactly what training feeds
    o_plain = olmo(x).float()               # no doc_lens (= matching parity path)
# compare the two olmo forwards at the answer (label) positions
pos = lm.to(dev) if lm is not None else (x[0] != PAD)
if pos.dim() == 1: pos = pos[None]
def at(t): return t[0][pos[0]]
d = (at(o_train) - at(o_plain)).abs()
cos = F.cosine_similarity(at(o_train).reshape(-1), at(o_plain).reshape(-1), dim=0).item()
print(f"\n[olmo train-path vs plain-path @answer positions] max|Δ|={d.max():.3e} cos={cos:.5f} "
      f"{'<-- doc_lens/padding CHANGES the GDN output (suspect!)' if cos<0.999 else 'identical (padding benign)'}")
