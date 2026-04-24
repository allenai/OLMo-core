"""Baseline full-attention ladder on v02 data with data-loader seed=42.

Identical to ``baseline_v02_full_attn_ladder.py`` except the
``ComposableDataLoaderConfig.seed`` is set to 42 (matching the seed used
by the old-pipeline ``NumpyDataLoaderConfig``). Paired with
``icl_overlap_v02_align_fix_seed42_full_attn_ladder.py``.

Motivation: old-pipeline uses data-loader seed 42; the default new-pipeline
uses seed 0. This isolates whether the integer seed alone (i.e., a
particular random trajectory) is responsible for any of the observed
old-pipeline-vs-new-pipeline TE gap on C4 CE.
"""
import argparse
import logging

import olmo_core.io as io
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import *
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import (
    ModelLadder,
    Olmo3ModelConfigurator,
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
                NumpyDocumentSourceConfig(
                    source_paths=DOLMA2_BASELINE_PATHS,
                    tokenizer=tokenizer,
                ),
            ],
            sequence_length=args.sequence_length,
        ),
    ]

    ladder = ModelLadder(
        name=args.name,
        dir=str(io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=[s for s in TransformerSize if s.approx_num_params <= 1e9],
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=Olmo3ModelConfigurator(
            model_construction_kwargs={"sliding_window": None},
            rank_microbatch_size=None
            if args.rank_mbz is None
            else args.rank_mbz * args.sequence_length,
        ),
        run_configurator=WSDSChinchillaRunConfigurator(
            chinchilla_multiple=args.chinchilla_multiple
        ),
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=instance_sources,
        data_loader=ComposableDataLoaderConfig(
            num_workers=8,
            instance_filter_config=InstanceFilterConfig(),
            seed=42,  # match old-pipeline NumpyDataLoaderConfig.seed
        ),
    )
    return ladder


if __name__ == "__main__":
    main(configure_ladder)
