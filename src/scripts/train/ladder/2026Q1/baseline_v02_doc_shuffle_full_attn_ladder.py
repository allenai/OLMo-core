"""Baseline full-attention ladder on v02 data with per-doc shuffle.

Same content as `baseline_v02_full_attn_ladder` (v0.1 sources + v0.2 s2pdf
swap) but with global per-document shuffle inserted via
`SamplingDocumentSource` before the concat-and-chunk step. This breaks
the per-source topical clustering that occurs in the default baseline
(where adjacent docs in a chunk are from the same `cc_all_dressed/...`
crawl or the same arxiv shard, etc.).

Motivation: investigating the +0.03 nat gap that all SA-based suffix-
sampling methods show vs the default doc-sampling baseline. One
hypothesis is that the default baseline benefits from incidental
in-context pretraining within a training instance (related docs as
neighbors), while SA-sampling emits docs in random-SA-pointer order
which destroys that locality. If this baseline (random doc order)
matches or moves closer to the SA-sampling baselines, locality-based
in-context pretraining was a real component of the gap.
"""
import argparse
import logging

import olmo_core.io as io
from olmo_core.data import TokenizerConfig
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

DOLMA2_BASELINE_PATHS = [
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed-snazzy2-fixed/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/arxiv/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/finemath-3plus/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.2/allenai/dolma2-tokenizer/s2pdf/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/stack-edu/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/wikipedia/**/*.npy",
]


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    instance_sources: list[InstanceSourceConfig] = [
        ConcatAndChunkInstanceSourceConfig(
            sources=[
                # SamplingDocumentSource with seed → shuffles documents globally
                # across all baseline sources, then concat-and-chunk sees them
                # in random order. factor=1.0 means use all docs (no truncation,
                # no extra repetition beyond the one natural epoch).
                SamplingDocumentSourceConfig(
                    sources=[
                        NumpyDocumentSourceConfig(
                            source_paths=DOLMA2_BASELINE_PATHS,
                            tokenizer=tokenizer,
                            # Read doc boundaries from .csv.gz sidecars (all baseline
                            # paths have them) instead of mmap-scanning the .npy
                            # files for EOS — the scan is what made rank-0 startup
                            # take hours and force the 360-min distributed-init
                            # timeout. With sidecars, startup is minutes.
                            prefer_metadata_files=True,
                        ),
                    ],
                    factor=1.0,
                    seed=42,
                ),
            ],
            sequence_length=args.sequence_length,
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
            model_construction_kwargs={"sliding_window": None},
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
            num_workers=8, instance_filter_config=InstanceFilterConfig()
        ),
    )
    return ladder


if __name__ == "__main__":
    main(configure_ladder, add_additional_args=add_smoke_args)
