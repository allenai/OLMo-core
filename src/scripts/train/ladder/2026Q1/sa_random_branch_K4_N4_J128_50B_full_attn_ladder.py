"""SA random-branch K=4 N=4 J=128 50B — random-branch-in-anchor MASK.

Per notes/random_branch_anchor_design.md: a baseline-style anchor with
K=4 random k-positions, each carrying N=4 walked SA-neighbors as a
tree-mask chain with one J=128-token head per walked. Heads
loss-masked. Overlap shared with anchor (vis_limit=max_seq_len).

Built via build_sa_random_branch_anchor.py with --K 4 --N-per-kpos 4
--J 128 --baseline-anchor-mult 0.65. Measured comp_ratio=1.3354.

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
    "dolma2-0625-v02/sa-random-branch-K4-N4-J128-50B/allenai/dolma2-tokenizer"
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
