"""
Drop-robustness dev loss for the drop-CPT study (records/ffndrop-cpt-plan.md), one checkpoint, one GPU.

On the held-out marker-wrapped ``cpt_dev`` shard (the soft-detach CPT dev set, never trained on), the
teacher-forced body-token CE with every FFN of layers >= ``--start-layer`` skipped per token at a FIXED
rate r, for each r in ``--rates`` (r = 0 is the plain dense model). The drop pattern depends only on
(row, layer, position), so every checkpoint sees the SAME dropped tokens -- quote paired per-row deltas.

The question it answers: does drop-CPT flatten CE(r)? If the drop-CPT curve is no flatter than the
dense-CPT one, the routed SFT stage has nothing to build on.

    python eval_drop_devloss.py --ckpt <run>/model_and_optim --dev <shards>/cpt_dev --out x.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "softdetach"))
from eval_cpt_devloss import VOCAB, load_rows, per_token_loss  # noqa: E402

from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig


def log(m):
    print(f"[drop-devloss] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help=".../model_and_optim (distcp)")
    ap.add_argument("--dev", required=True, help="held-out marker-wrapped shard dir (cpt_dev)")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--rates", default="0,0.25,0.5,0.75")
    ap.add_argument("--start-layer", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--attn-backend", default="flash_2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rates = [float(r) for r in a.rates.split(",")]
    rows, masks, meta = load_rows(a.dev, a.rows)
    log(f"{len(rows)} dev rows x {len(rows[0])} tokens; rates {rates}")
    cfg = TransformerConfig.qwen3_5_4B(vocab_size=VOCAB, attn_backend=AttentionBackendName(a.attn_backend))
    cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear
    model = cfg.build(init_device="cpu")
    t0 = time.time()
    load_model_and_optim_state(a.ckpt, model)
    log(f"loaded {a.ckpt} in {time.time() - t0:.0f}s")
    model.enable_ffn_token_drop(max_rate=0.0, start_layer=a.start_layer, seed=a.seed)
    holder = model._ffn_token_drop["holder"]
    holder.active_in_eval = True
    model = model.cuda().to(torch.bfloat16).eval()
    for mod in model.modules():
        rope = getattr(mod, "rope", None)
        if rope is not None and hasattr(rope, "warmup_cache"):
            rope.warmup_cache(len(rows[0]) + 8, torch.device("cuda"))
    res = {f"ce_r{r:g}": [] for r in rates}
    frac = {f"ce_r{r:g}": [] for r in rates}
    t_start = time.time()
    for ri, (row, m) in enumerate(zip(rows, masks)):
        x = torch.tensor(row[None], device="cuda")
        tgt_mask = torch.tensor(m[None], device="cuda")
        lab = torch.full_like(x, -100)  # labels are PRE-SHIFTED in olmo-core
        lab[:, :-1] = torch.where(tgt_mask[:, 1:], x[:, 1:], torch.full_like(x[:, 1:], -100))
        for r in rates:
            holder.fixed_rate = r
            holder.calls = ri  # pattern = f(row, layer, position): identical across checkpoints
            holder.begin_forward(training=False)
            holder.pop_metrics()
            pt = per_token_loss(model, x, lab)
            res[f"ce_r{r:g}"].append(float(torch.nanmean(pt)))
            frac[f"ce_r{r:g}"].append(holder.pop_metrics().get("ffn_drop/frac", 0.0))
        if ri + 1 in (1, 2, 5, 10) or (ri + 1) % 10 == 0:
            el = time.time() - t_start
            log(f"row {ri + 1}/{len(rows)} " + " ".join(f"{k}={np.mean(v):.4f}" for k, v in res.items())
                + f"  ({el:.0f}s, ETA {el / (ri + 1) * (len(rows) - ri - 1):.0f}s)")
    base = np.array(res[f"ce_r{rates[0]:g}"])
    summ = {}
    for k, v in res.items():
        v = np.array(v)
        summ[k] = float(v.mean())
        summ[k + "_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None
        d = v - base
        summ[k + "_minus_r0"] = float(d.mean())
        summ[k + "_minus_r0_se"] = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None
        summ[k + "_realized_frac"] = float(np.mean(frac[k]))
    out = {"ckpt": a.ckpt, "dev": a.dev, "eval_size": len(rows), "rates": rates, "start_layer": a.start_layer,
           "summary": summ, "per_row": res,
           "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "argv": sys.argv}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    log("SUMMARY " + " ".join(f"{k}={summ[k]:.4f} (+{summ[k + '_minus_r0']:.4f})" for k in res))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
