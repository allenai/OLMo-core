"""ICL-overlap (ratio ≥ 1.2 with plain-emit fallback) full-attention
ladder, v02, with the canonical 50/50 baseline mix.

Same as ``icl_overlap_v02_ratio_1_2_full_attn_ladder`` except the ICL
treatment uses the plain-emit variant of the compression-ratio-filtered
data: low-compression segments are emitted plain at their original 8192
length instead of being discarded. This widens doc-coverage vs the
discard-only ratio-1.2 variant.

Trained with the standard ICL-overlap pipeline: 50% ICL-overlap data
mixed with 50% baseline-v02 data. The companion run
``icl-overlap-v02-dedup-perworker-ratio-1_2-plain-emit-50B-190M``
trains on 100% ICL-overlap (no baseline mix) as a direct-distribution
comparison; this one is the canonical evaluation against the existing
ratio-1.2 ladder.

Data sources:
- ICL: ``weka://...dedup-by-doc-perworker-ratio-1_2-plain-emit-50B/``
- Baseline: same 6 paths as ``baseline_v02_full_attn_ladder``
  (v0.1 sources + v0.2 s2pdf swap).
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

ICL_OVERLAP_PATHS = [
    "/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/dolma2-0625-v02/icl-overlap-v02-8192-wo-replace-dedup-by-doc-perworker-ratio-1_2-plain-emit-50B/allenai/dolma2-tokenizer/*.npy",
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
                    # PerFileChunked preserves alignment of the variable-length
                    # emissions with `<o>`/`<\o>` markers and plain-emitted
                    # fallback blocks; cross-file phase drift would otherwise
                    # bisect the overlap structure.
                    source=PerFileChunkedInstanceSourceConfig(
                        source_paths=ICL_OVERLAP_PATHS,
                        sequence_length=args.sequence_length,
                        tokenizer=tokenizer,
                    ),
                    ratio=0.5,
                    label="icl-overlap-ratio-1_2-plain-emit",
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
