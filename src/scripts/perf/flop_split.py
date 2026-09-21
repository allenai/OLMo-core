"""Decompose the crops=80 Stage-2 win in FLOPs, using the real Molmo2-4B config.

Run with::

    PYTHONPATH=$PWD/src python src/scripts/perf/flop_split.py

Builds the model on the meta device (shapes only, no weights, no GPU) and uses the
model's own `num_flops_per_token` / `image_encoder_flops` accounting.

Two things this establishes that are easy to get wrong by eye:

1. **The LM dominates FLOPs even at crops=80, roughly 6:1.** The ViT is ~14% with the
   encoder trained, ~5% if you (incorrectly, for Stage 2) charge it forward-only. Any
   reasoning that starts from "the ViT is about a third of the step" is using a measured
   *time* share as if it were a FLOP share, and the two differ a lot here.
2. **crops=80 lowers raw FLOP throughput by ~35% and still wins**, because at crops=25
   ~69% of the LM's FLOPs went to padding. The +87% in *useful* LM FLOPs/s is what shows
   up as the measured +85-87% useful TPS.

Step times are hardcoded from the 8xB300 sweep, so the PF/s columns are only valid for
that measurement; the per-pack FLOP columns are config-derived and general.
"""
from olmo_core.nn.vision.multimodal import MultimodalLMConfig

cfg = MultimodalLMConfig.molmo2_4B()
m = cfg.build(init_device="meta")
S, P, POOLED = 16384, 729, 144
lm_fpt = m.num_flops_per_token(S)
d, L = cfg.vision.image_emb_dim, cfg.vision.image_num_layers


def vit(nc):
    raw = nc * P
    return (
        raw * (2 * m._n_vision_params + 4 * L * P * d) * 3 + nc * POOLED * 6 * m._n_connector_params
    )


# (label, crops/pack, packs per rank, real tokens/pack, step seconds)
ARMS = [("crops=25", 24, 16, 5046, 9.62), ("crops=80", 79, 6, 16180, 6.17)]
print(
    "%-10s %8s %10s %10s %10s %10s"
    % ("arm", "PF/pack", "PF/s tot", "PF/s LM-useful", "tok occ", "rank packs")
)
res = {}
for name, nc, packs, real, sec in ARMS:
    lm_all = lm_fpt * S
    tot = (lm_all + vit(nc)) * packs
    useful = lm_fpt * real * packs
    res[name] = (tot / sec, useful / sec)
    print(
        "%-10s %8.4f %10.3f %14.3f %9.1f%% %10d"
        % (
            name,
            (lm_all + vit(nc)) / 1e15,
            tot / sec / 1e15,
            useful / sec / 1e15,
            100 * real / S,
            packs,
        )
    )
a, b = res["crops=25"], res["crops=80"]
print()
print(
    "total FLOP throughput   : %.3f -> %.3f PF/s  = %+.0f%%"
    % (a[0] / 1e15, b[0] / 1e15, 100 * (b[0] / a[0] - 1))
)
print(
    "USEFUL LM FLOP throughput: %.3f -> %.3f PF/s  = %+.0f%%"
    % (a[1] / 1e15, b[1] / 1e15, 100 * (b[1] / a[1] - 1))
)
print()
lm_all = lm_fpt * S
for nc in (24, 79):
    fro = (
        lm_all
        + nc * P * (2 * m._n_vision_params + 4 * L * P * d)
        + nc * POOLED * 6 * m._n_connector_params
    )
    tra = lm_all + vit(nc)
    print(
        "crops=%-3d MFU undercount from frozen-ViT accounting: %.1f%% relative "
        "(reported 41.35%% -> %.1f%%)" % (nc, 100 * (tra / fro - 1), 41.35 * tra / fro)
    )
