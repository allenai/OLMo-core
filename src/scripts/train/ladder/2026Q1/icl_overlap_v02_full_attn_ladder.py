"""ICL overlap full-attention ladder on the v02 combined suffix array.

Differences from `icl_overlap_full_attn_ladder.py` (v01):

- ICL overlap treatment uses the v02 data-gen output
  `icl-overlap-v02-8192-wo-replace` (from the dolma2-0625-v02 combined SA
  which replaces v01's pad-masked `s2pdf_redacted` shards with
  document-level-denylisted `s2pdf` at v0.2).
- Baseline mix swaps the s2pdf subsource from v0.1/`s2pdf_redacted` to
  v0.2/`s2pdf`. The other 5 subsources stay at v0.1 because they're
  unchanged between v01 and v02.

See `data_gen/v02_rerun_plan.md` and `tex/instance_filter_analysis/` for the
motivation (removing the differential-cleanup confound identified in the
main paper's pipeline-discrepancy table).
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
    "/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/dolma2-0625-v02/icl-overlap-v02-8192-wo-replace/allenai/dolma2-tokenizer/*.npy",
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
                    # PerFileChunked, not ConcatAndChunk: the ICL build script
                    # emits a short tail at each worker's budget exhaustion
                    # (~125/192 v02 shards are non-seq_len-aligned). Cross-file
                    # flat-concat reads would bisect <o>...</o> pairs in
                    # ~99% of chunks after the first tail. See
                    # tex/instance_filter_analysis/ §sec:update-alignment.
                    source=PerFileChunkedInstanceSourceConfig(
                        source_paths=ICL_OVERLAP_PATHS,
                        sequence_length=args.sequence_length,
                    ),
                    ratio=0.5,
                    label="icl-overlap",
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
