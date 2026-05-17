"""ICL-overlap (compression-ratio ≥ 1.2 with plain-emit fallback)
full-attention ladder on the v02 50B per-worker dedup data.

Trains *directly* on the ICL-overlap SA-sampled data — no baseline mix,
no overlap-format treatment, the model just sees the token stream
including the `<o>`/`<\\o>` overlap markers as regular tokens.

Sister run to the existing 5B/50B dedup-perworker canaries
(`ucm-sim-overlap`, `ucm-sim-adj-infix`, `random-doc-sample`). The
plain-emit fallback is the key difference from the existing
ICL-overlap dedup variants: low-compression segments (ratio < 1.2)
are emitted plain at their original 8192 length instead of being
discarded, so the data covers a wider distribution of doc kinds
rather than over-selecting only high-reuse regions.

Question this addresses: does ICL-overlap data with plain-emit
fallback show the same +0.03 nat gap to baseline that the other
SA-based variants do, or does the wider distribution (closer to
natural doc-sampling) close part of the gap?

Data source (50B compressed-equivalent token budget, no cycling at
190M / 2xC = 7.6B):
``/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/dolma2-0625-v02/icl-overlap-v02-8192-wo-replace-dedup-by-doc-perworker-ratio-1_2-plain-emit-50B/allenai/dolma2-tokenizer/``.
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


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    instance_sources: list[InstanceSourceConfig] = [
        # PerFileChunked (not ConcatAndChunk): ICL-overlap data has variable-length
        # emissions with `<o>`/`<\o>` markers around LCP-compressed segments and
        # plain-emitted fallback blocks. Per-file chunking preserves alignment
        # so cross-file phase drift can't bisect the overlap structure.
        PerFileChunkedInstanceSourceConfig(
            source_paths=ICL_OVERLAP_PATHS,
            sequence_length=args.sequence_length,
            tokenizer=tokenizer,
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
