"""Kept fraction and FLOP-meter actual/dense ratio of the olmo3_cptdrop drop arms (CPU, ~1 min).

The keep set comes from the real model code (``Transformer._compact_pooled_soft_tokens`` in
positional-block mode) on synthetic EOS + 32,767-token rows; the ratio is the FLOP meter's
per-example formula (``olmo_core.nn.token_exit.token_exit_example_flops``) on the OLMo 3 7B
config the trainer builds (``olmo3_7B_ctc_swa(yarn_factor=4)``, built on the meta device).

    PYTHONPATH=src:recipes/train/research_trainer python research/token_routing/olmo3_cptdrop/measure_keep_rules.py
"""

import argparse
import json

import torch

from olmo_core.config import DType
from olmo_core.nn.token_exit import token_exit_example_flops
from olmo_core.nn.transformer import TransformerConfig

EOS = 100257
T = 32768

IFL = {"keep_prob": 0.0, "keep_token_rule": "first_last", "keep_token_k": 0.25}
IRAND = {"keep_prob": 0.2}
ARMS = {
    "IRAND": (IRAND, 0),
    "IFL": (IFL, 0),
    "IFL_R4K (recent 4096 + I-fl)": ({**IFL, "keep_recent": 4096}, 0),
    "IFL_RAMP (first_last 0.1->0.5)": ({**IFL, "keep_ramp": (0.1, 0.5)}, 0),
    "IFL_X4 (I-fl, exit L=4)": (IFL, 4),
    "IFL_X8 (I-fl, exit L=8)": (IFL, 8),
    "IRAND_X4 (I-rand, exit L=4)": (IRAND, 4),
    "IRAND_X8 (I-rand, exit L=8)": (IRAND, 8),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from olmo3_configs import OLMO3_VOCAB_SIZE, olmo3_7B_ctc_swa  # recipes/train/research_trainer

    big = olmo3_7B_ctc_swa(vocab_size=OLMO3_VOCAB_SIZE, yarn_factor=4).build(init_device="meta")
    full_layers = [int(i) for i, b in big.blocks.items() if not b.attention.window_size]
    print(f"[measure] OLMo 3 7B: {len(big.blocks)} blocks, full attention at {full_layers}")

    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1, 50000, (a.rows, T), generator=g)
    ids[:, 0] = EOS
    lab = torch.full_like(ids, -100)
    lab[:, :-1] = ids[:, 1:]

    cache: dict = {}
    dense = a.rows * T * big.num_flops_per_token(T)
    out = {}
    for name, (rule, L) in ARMS.items():
        small = TransformerConfig.olmo2_30M(
            vocab_size=100352, n_layers=1, fused_ops=False, dtype=DType.float32
        ).build(init_device="cpu")
        small.enable_pooled_soft_tokens(
            -5, -6, EOS, placeholder_id=100300, drop_slots=True, positional_block_size=512, **rule
        )
        small.train()
        kept = []
        for r in range(0, a.rows, 8):
            cb = small._compact_pooled_soft_tokens(ids[r : r + 8], lab[r : r + 8], -100)[0]
            kept += [int(x) for x in cb.row_lens.tolist()]
        actual = sum(token_exit_example_flops(big, T, k, L, cache=cache) for k in kept)
        kt = torch.tensor(kept, dtype=torch.float64) / T
        out[name] = {
            "exit_layer": L,
            "kept_frac_mean": float(kt.mean()),
            "kept_frac_sd_rows": float(kt.std()) if len(kept) > 1 else 0.0,
            "meter_actual_over_dense": actual / dense,
        }
        print(
            f"[measure] {name:34s} L={L}  kept {kt.mean():.4f} (sd {out[name]['kept_frac_sd_rows']:.4f} "
            f"over {len(kept)} rows)  actual/dense {actual / dense:.4f}",
            flush=True,
        )
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
