"""ROOT-CAUSE confirmation: the ON-DISK converted olmo base ckpt (what training loads) vs an
IN-MEMORY conversion of the HF state_dict (what all the parity tests used). If init loss differs
and tensors mismatch, the cached converted checkpoint is corrupt -> that's the training bug.
"""
import json, sys
import numpy as np, torch, torch.nn.functional as F
import transformers
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.data import NumpyPaddedFSLDatasetConfig, TokenizerConfig
from olmo_core.data.types import NumpyDatasetDType

dev = "cuda"
CKPT = "/scratch/users/prasann/olmo_ckpts/converted/Qwen_Qwen3.5-0.8B-Base_248320/model_and_optim"
VOCAB = 248320
PREFIX = sys.argv[1]
meta = json.load(open(PREFIX + "_meta.json"))
EOS, PAD = meta["eos_token_id"], meta["pad_token_id"]
tok = TokenizerConfig(vocab_size=VOCAB, eos_token_id=EOS, pad_token_id=PAD,
                      bos_token_id=None, identifier="Qwen/Qwen3.5-0.8B-Base")
ds = NumpyPaddedFSLDatasetConfig(paths=[PREFIX + "_tokens.npy"], label_mask_paths=[PREFIX + "_label_mask.npy"],
    sequence_length=4096, tokenizer=tok, dtype=NumpyDatasetDType.uint32, work_dir="/tmp/verify_wd").build()
ds.prepare()
d = ds[0]
ids = torch.as_tensor(np.asarray(d["input_ids"]), dtype=torch.long)[None].to(dev)
lm = torch.as_tensor(np.asarray(d["label_mask"]), dtype=torch.bool)[None].to(dev)
def mce(model_logits):
    lg = model_logits[:, :-1].reshape(-1, model_logits.shape[-1]).float()
    tg = ids[:, 1:].reshape(-1); mk = lm[:, 1:].reshape(-1)
    return F.cross_entropy(lg[mk], tg[mk]).item()

# (A) on-disk converted checkpoint
mA = TransformerConfig.qwen3_5_0_8B(vocab_size=VOCAB, attn_backend=AttentionBackendName.torch).build().to(dev)
load_model_and_optim_state(CKPT, mA); mA.eval()
with torch.no_grad(): lA = mce(mA(ids))
print(f"[A] on-disk converted ckpt   init masked-CE = {lA:.4f}", flush=True)

# (B) in-memory conversion of HF state_dict
hf = transformers.Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B-Base", torch_dtype=torch.float32, attn_implementation="eager")
sdB = convert_state_from_hf(hf.config, hf.state_dict(), model_type="qwen3_5_text")
mB = TransformerConfig.qwen3_5_0_8B(vocab_size=hf.config.vocab_size, attn_backend=AttentionBackendName.torch).build().to(dev)
missing, unexpected = mB.load_state_dict(sdB, strict=False)
mB.eval()
with torch.no_grad(): lB = mce(mB(ids))
print(f"[B] in-memory HF conversion   init masked-CE = {lB:.4f}", flush=True)
print(f"    HF vocab={hf.config.vocab_size}  olmo-build vocab(A)={VOCAB}  missing={len(missing)} unexpected={len(unexpected)}", flush=True)
if missing[:8]: print("    missing[:8]=", missing[:8], flush=True)

# tensor-by-tensor diff A vs B
print("\n=== per-tensor diff (on-disk A vs in-memory B), top mismatches ===", flush=True)
sdA = dict(mA.named_parameters()); sdBp = dict(mB.named_parameters())
rows = []
for n in sdA:
    if n in sdBp and sdA[n].shape == sdBp[n].shape:
        a, b = sdA[n].float(), sdBp[n].float()
        rel = (a - b).norm().item() / (b.norm().item() + 1e-9)
        rows.append((rel, n, a.norm().item(), b.norm().item()))
    elif n in sdBp:
        print(f"   SHAPE-MISMATCH {n}: A{tuple(sdA[n].shape)} vs B{tuple(sdBp[n].shape)}", flush=True)
rows.sort(reverse=True)
for rel, n, na, nb in rows[:15]:
    print(f"   rel_diff={rel:.4f}  {n}  |A|={na:.3e} |B|={nb:.3e}", flush=True)
print(f"\n   tensors compared={len(rows)}  max_rel={rows[0][0]:.4f}  median_rel={sorted(r[0] for r in rows)[len(rows)//2]:.4f}", flush=True)
