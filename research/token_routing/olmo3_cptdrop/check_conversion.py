"""
fp32 parity check: HF ``allenai/Olmo-3-1025-7B@<rev>`` vs its OLMo-core conversion.

The converter's built-in validation runs both models in the checkpoint's bf16 at atol 1e-4, which bf16
kernel differences alone exceed. This compares in fp32 (random ids, plus one real dev row) and in bf16
on a real row by mean CE, which is what the experiment consumes.

    python check_conversion.py --hf-dir /data/.../hf/stage2-step47684 \
        --olmo-dir /data/.../base/stage2-step47684 --dev /data/.../data/dev32k [--yarn-factor 8]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F


def build_olmo(olmo_dir, yarn_factor, dtype, device):
    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.rope import YaRNRoPEScalingConfig
    from olmo_core.nn.transformer import TransformerConfig

    cfg = TransformerConfig.olmo3_7B(vocab_size=100352, attn_backend=AttentionBackendName.torch)
    if yarn_factor > 1:
        cfg = cfg.with_rope_scaling(
            YaRNRoPEScalingConfig(factor=yarn_factor, beta_fast=32, beta_slow=1, old_context_len=8192)
        )
    m = cfg.build(init_device="cpu")
    load_model_and_optim_state(f"{olmo_dir}/model_and_optim", m)
    return m.to(device=device, dtype=dtype).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--olmo-dir", required=True)
    ap.add_argument("--dev", required=True, help="shard dir with token_ids_part_0000.npy + metadata.json")
    ap.add_argument("--yarn-factor", type=float, default=0.0)
    ap.add_argument("--real-len", type=int, default=2048)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM

    dev = "cuda"
    meta = json.load(open(f"{a.dev}/metadata.json"))
    row = np.fromfile(f"{a.dev}/token_ids_part_0000.npy", dtype=np.uint32)[: meta["max_example_len"]]
    real = torch.from_numpy(row[: a.real_len].astype(np.int64))[None].to(dev)
    rand = torch.randint(0, 100278, (1, 512), generator=torch.Generator().manual_seed(0)).to(dev)
    out = {}
    for dtype in (torch.float32, torch.bfloat16):
        hf = AutoModelForCausalLM.from_pretrained(a.hf_dir, dtype=dtype, attn_implementation="sdpa").to(dev).eval()
        olmo = build_olmo(a.olmo_dir, a.yarn_factor, dtype, dev)
        tag = str(dtype).split(".")[-1]
        with torch.no_grad():
            for name, ids in (("rand512", rand), (f"real{a.real_len}", real)):
                lh = hf(input_ids=ids).logits.float()
                lo = olmo(input_ids=ids).float()[..., : lh.shape[-1]]
                ce_h = F.cross_entropy(lh[0, :-1], ids[0, 1:]).item()
                ce_o = F.cross_entropy(lo[0, :-1], ids[0, 1:]).item()
                out[f"{tag}/{name}"] = {
                    "max_abs_logit_diff": (lh - lo).abs().max().item(),
                    "mean_abs_logit_diff": (lh - lo).abs().mean().item(),
                    "argmax_agree": (lh.argmax(-1) == lo.argmax(-1)).float().mean().item(),
                    "ce_hf": ce_h, "ce_olmo": ce_o, "ce_diff": ce_o - ce_h,
                }
                print(tag, name, json.dumps(out[f"{tag}/{name}"]), flush=True)
        del hf, olmo
        torch.cuda.empty_cache()
    ok = out["float32/rand512"]["max_abs_logit_diff"] < 5e-3 and abs(out[f"bfloat16/real{a.real_len}"]["ce_diff"]) < 5e-3
    print("PARITY", "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
