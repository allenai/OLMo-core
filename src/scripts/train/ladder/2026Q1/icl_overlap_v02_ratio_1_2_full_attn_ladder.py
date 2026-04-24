"""ICL-overlap (compression-ratio ≥ 1.2 filter) full-attention ladder, v02.

Same as `icl_overlap_v02_full_attn_ladder.py` except the ICL treatment
uses the compression-ratio-filtered variant `icl-overlap-v02-8192-wo-replace-ratio-1_2`
(data generated with `--min-compression-ratio 1.2`, which biases the SA
sampler toward high-suffix-reuse regions). Compared against the plain
icl-overlap-v02 run at matched compute, this tests whether the ICL
advantage comes from the overlap structure itself or just from the
exposure to high-reuse regions of the corpus.

Baseline mix is the v02 clean-source mix (v0.2/s2pdf replaces
v0.1/s2pdf_redacted). See `data_gen/v02_rerun_plan.md` run #5.
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

ICL_OVERLAP_PATHS = [
    "/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/dolma2-0625-v02/icl-overlap-v02-8192-wo-replace-ratio-1_2/allenai/dolma2-tokenizer/*.npy",
]

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
        MixingInstanceSourceConfig(
            source_specs=[
                MixingInstanceSourceSpecConfig(
                    # See icl_overlap_v02_full_attn_ladder.py for the alignment
                    # reasoning; PerFileChunked prevents cross-file phase drift.
                    source=PerFileChunkedInstanceSourceConfig(
                        source_paths=ICL_OVERLAP_PATHS,
                        sequence_length=args.sequence_length,
                        tokenizer=tokenizer,
                    ),
                    ratio=0.5,
                    label="icl-overlap-ratio-1_2",
                ),
                MixingInstanceSourceSpecConfig(
                    source=ConcatAndChunkInstanceSourceConfig(
                        sources=[
                            NumpyDocumentSourceConfig(
                                source_paths=DOLMA2_BASELINE_PATHS,
                                tokenizer=tokenizer,
                            ),
                        ],
                        sequence_length=args.sequence_length,
                    ),
                    ratio=0.5,
                    label="baseline",
                ),
            ],
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
            num_workers=8, instance_filter_config=InstanceFilterConfig()
        ),
    )
    return ladder


if __name__ == "__main__":
    main(configure_ladder)
