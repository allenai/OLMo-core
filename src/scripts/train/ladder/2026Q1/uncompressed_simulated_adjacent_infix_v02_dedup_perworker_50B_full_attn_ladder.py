"""Uncompressed-simulated-adjacent-infix full-attention ladder on the v02
50B per-worker dedup data.

Suffix-sampling baseline of the *pair* variety: each emitted sample is
a pair of full documents (the anchor + a partner from an adjacent SA
position), concatenated with a separator token, no overlap markers,
no LCP compression. Trains directly on the dedup data as if it were
normal pretraining (no special model-side handling).

The question this addresses: does pair-builder + per-doc dedup +
full-doc emission close the +0.03 nat gap to doc-sampling baseline
that ucm-sim *overlap* (iterating-builder) variants showed? Compares
against:
  - baseline-v02-full-attn-190M                          (doc-sampling)
  - ucm-sim-overlap-v02-dedup-perworker-5B-190M          (iterating-builder + per-worker dedup, prior canary)

Data source (50B actual tokens, no cycling at 190M / 2xC = 7.6B):
``/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/dolma2-0625-v02/uncompressed-simulated-adjacent-infix-v02-8192-wo-replace-dedup-by-doc-perworker-50B/allenai/dolma2-tokenizer/``.
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

UCM_SIM_ADJ_INFIX_PATHS = [
    "/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/dolma2-0625-v02/uncompressed-simulated-adjacent-infix-v02-8192-wo-replace-dedup-by-doc-perworker-50B/allenai/dolma2-tokenizer/*.npy",
]


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    instance_sources: list[InstanceSourceConfig] = [
        # ConcatAndChunk is fine here: the adj-infix-ucm-sim format is
        # independent pair samples (anchor + partner, each a full doc)
        # concatenated with a separator. No context-boundary invariant
        # between adjacent emitted chunks, so frankenstein chunks at
        # file boundaries don't bisect any structural relationship.
        ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceConfig(
                    source_paths=UCM_SIM_ADJ_INFIX_PATHS,
                    tokenizer=tokenizer,
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
