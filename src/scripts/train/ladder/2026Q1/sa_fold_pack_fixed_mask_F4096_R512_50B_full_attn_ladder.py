"""SA fold-pack-fixed MASK F=4096 R=512 50B — neighborhood-packed MASK on fixed-len SA neighbors.

Step 4 of new_data_gen_plan.md. Per anchor (= fold_size flat-token chunk),
walk SA forward from R = ISA(anchor_start), pack multiple SA-neighbor
suffixes (each FIXED adj_rank_size=512 tokens, NOT truncated at DOC_SEP
— that's the source-mix-bias fix per task #72). Apply the idealized MASK
algorithm: dict-based depth tracking (no skipped-depth bug), per-token
pos_ids and vis_limit, label_mask conditional on L==0.

Per anchor budget = 2*F emit positions (anchor F + adj F compressed
divergent tails). K_target = F/R = 8 expected SA neighbors per
anchor (variable due to compression).

Pairs against sa-fold-overlap-mask-v02-F4096-50B (single-pair MASK at
same F) and sa-chunk-shuffle-F4096-mask-50B (non-overlap baseline) for
the increase-compression-or-confound analysis.

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

BASE = (
    "/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/"
    "dolma2-0625-v02/sa-fold-pack-fixed-mask-F4096-R512-50B/allenai/dolma2-tokenizer"
)


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    instance_sources: list[InstanceSourceConfig] = [
        PreChunkedInstanceSourceConfig(
            token_paths=[f"{BASE}/*-tokens.npy"],
            pos_ids_paths=[f"{BASE}/*-pos_ids.npy"],
            vis_limit_paths=[f"{BASE}/*-vis_limit.npy"],
            label_mask_paths=[f"{BASE}/*-label_mask.npy"],
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
