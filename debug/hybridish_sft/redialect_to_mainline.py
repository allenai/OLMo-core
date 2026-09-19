"""Re-dialect an olmo3_5_hybrid HF export into mainline_ladder form.

`convert_checkpoint_to_hf.py` writes the `olmo3_5_hybrid` dialect, but the transformers plugin (and
every released hybridish checkpoint) uses `mainline_ladder`. The weights are identical -- same 384
tensors, same shapes -- only the norm keys are spelled differently, and config.json declares a
model_type that transformers 5.14 does not know.

So rather than re-running a converter, this renames keys against a RELEASED checkpoint as the
reference and reuses its config.json. The mapping is derived from the two key sets and every key
must be accounted for; an unmapped or shape-mismatched key raises instead of being dropped.
"""
import argparse, json, os, re, shutil
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="olmo3_5_hybrid HF export (the SFT'd weights)")
ap.add_argument("--ref", required=True, help="released mainline_ladder ckpt (naming + config donor)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

src = load_file(os.path.join(a.src, "model.safetensors"))
ref = load_file(os.path.join(a.ref, "model.safetensors"))

def norm(k):  # canonicalise the differing spellings
    k = k.replace("model.embedding_norm.", "model.embed_norm.")
    k = re.sub(r"\.pre_attention_norm\.", ".input_layernorm.", k)
    k = re.sub(r"\.pre_feedforward_norm\.", ".ffn_layernorm.", k)
    k = re.sub(r"\.self_attn\.g_proj\.", ".self_attn.attn_gate.", k)
    return k

out, unmapped = {}, []
for k, v in src.items():
    nk = norm(k)
    if nk not in ref:
        unmapped.append((k, nk)); continue
    if tuple(ref[nk].shape) != tuple(v.shape):
        # Only the vocab dimension may legitimately differ: convert_checkpoint_to_hf trims the
        # embedding/lm_head to the tokenizer's true vocab (100,278) while the released checkpoints
        # keep the padded 100,352. The dropped rows are untrained padding no token maps to, so the
        # model is unchanged -- but config.json must then declare the real size, handled below.
        if nk not in ("lm_head.weight", "model.embed_tokens.weight") or v.shape[1:] != ref[nk].shape[1:]:
            raise SystemExit(f"shape mismatch {k} -> {nk}: {v.shape} vs {ref[nk].shape}")
        print(f"  vocab-dim differs on {nk}: keeping ours {tuple(v.shape)} (ref {tuple(ref[nk].shape)})")
    out[nk] = v

missing = sorted(set(ref) - set(out))
if unmapped or missing:
    print("UNMAPPED (src -> guessed):", unmapped[:8])
    print("MISSING in output       :", missing[:8])
    raise SystemExit(f"{len(unmapped)} unmapped, {len(missing)} missing -- refusing to write a partial checkpoint")

os.makedirs(a.out, exist_ok=True)
save_file(out, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
for f in ("config.json", "tokenizer.json", "tokenizer_config.json"):
    p = os.path.join(a.ref, f)
    if os.path.exists(p): shutil.copy(p, os.path.join(a.out, f))
cfg_path = os.path.join(a.out, "config.json")
cfg = json.load(open(cfg_path))
real_vocab = int(out["lm_head.weight"].shape[0]) if "lm_head.weight" in out else cfg.get("vocab_size")
if cfg.get("vocab_size") != real_vocab:
    print(f"  config vocab_size {cfg.get('vocab_size')} -> {real_vocab} (matching the exported tensors)")
    cfg["vocab_size"] = real_vocab
    json.dump(cfg, open(cfg_path, "w"), indent=2)
print(f"wrote {len(out)} tensors -> {a.out}")
print(f"  model_type={cfg.get('model_type')} arch={cfg.get('architectures')}")
print(f"  ssmax tensors: {[k for k in out if 'ssmax' in k]}")
