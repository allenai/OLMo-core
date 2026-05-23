"""SA fold-overlap MASK long-anchor + pack-fixed combo (DD-7 from
notes/long_anchor_design.md) on v02 — N=2 folds, anchor:adj ratio = 1:1,
fixed-length neighborhood pack with adj_rank_size=512 (4 neighbors per
slot at adj_size=2048).

Combines:
  - Long-anchor layout (one 4096-token contiguous anchor split into 2
    chunks of 2048 tokens, contiguously causal across chunks)
  - Pack-fixed per-slot adj region (each adj_k holds multiple SA-walked
    neighbors of anchor_chunk_k at fixed 512 tokens each, idealized
    algorithm within the slot)

Per-instance layout:

    [ anchor_chunk_1 | anchor_chunk_2 | adj_1 (4 nbrs * 512) | adj_2 ]
      ^---- A = 2 * 2048 = 4096 ----^   ^---- 2*2048 = 4096 ----^

Uses LA MASK v2 conventions (vis_limit = scope_end_excl, label_mask
conditional on per-neighbor L==0). Each slot's adj region runs the
idealized MASK algorithm with FRESH depth state (anchor_chunk_k as the
first "suffix" + each neighbor as a subsequent suffix). Anchor cont
depths shared by all neighbors stay open through the slot and close at
adj_pos_end. Under-fill from SA walk exhaustion gets anchor-continuation
pad (NOT EOS pad).

Pairs with the LA-N2-R1-v2 ladder and the standalone pack-fixed ladder
for direct comparison: this experiment isolates whether the LA layout
and the pack-fixed layout compose without regression.

Data: 50B tokens (no cycling at 190M / 2xC = 7.6B).
"""
import argparse
import logging

import olmo_core.io as io
from olmo_core.data import NumpyDatasetDType, TokenizerConfig
from olmo_core.data.composable import *
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import main
from olmo_core.internal.smoke import (
    Olmo3SmokeConfigurator,
    WSDSChinchillaSmoke,
    add_smoke_args,
)
from olmo_core.model_ladder import (
    ModelLadder,
    TransformerSize,
    WSDSChinchillaRunConfigurator,
)

log = logging.getLogger(__name__)

SA_FOLD_MASK_BASE = (
    "/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/"
    "dolma2-0625-v02/sa-fold-overlap-mask-LA-N2-R1-pf-R512-50B/"
    "allenai/dolma2-tokenizer"
)


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    instance_sources: list[InstanceSourceConfig] = [
        PreChunkedInstanceSourceConfig(
            token_paths=[f"{SA_FOLD_MASK_BASE}/*-tokens.npy"],
            pos_ids_paths=[f"{SA_FOLD_MASK_BASE}/*-pos_ids.npy"],
            vis_limit_paths=[f"{SA_FOLD_MASK_BASE}/*-vis_limit.npy"],
            label_mask_paths=[f"{SA_FOLD_MASK_BASE}/*-label_mask.npy"],
            sequence_length=args.sequence_length,
            token_dtype=NumpyDatasetDType.uint32,
        ),
    ]

    smoke_1gpu = getattr(args, "smoke_1gpu", False)
    ladder = ModelLadder(
        name=args.name,
        dir=str(io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=[s for s in TransformerSize if s.approx_num_params <= 1e9],
        max_devices=1 if smoke_1gpu else args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=Olmo3SmokeConfigurator(
            model_construction_kwargs={
                "sliding_window": None,
                "attn_backend": "torch",
                "vis_limit_eos_token_id": tokenizer.eos_token_id,
            },
            rank_microbatch_size=None
            if args.rank_mbz is None
            else args.rank_mbz * args.sequence_length,
            smoke_1gpu=smoke_1gpu,
        ),
        run_configurator=(
            WSDSChinchillaSmoke(chinchilla_multiple=args.chinchilla_multiple)
            if smoke_1gpu
            else WSDSChinchillaRunConfigurator(
                chinchilla_multiple=args.chinchilla_multiple
            )
        ),
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=instance_sources,
        data_loader=ComposableDataLoaderConfig(
            num_workers=8, instance_filter_config=None
        ),
    )
    return ladder


if __name__ == "__main__":
    main(configure_ladder, add_additional_args=add_smoke_args)
