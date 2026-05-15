"""Uncompressed-simulated-overlap full-attention ladder on the v02 5B
per-worker dedup canary.

Companion to ``uncompressed_simulated_overlap_v02_dedup_shared_5B_full_attn_ladder``
— same setup, but trained on the per-worker (`--dedup-by-doc`) variant of
the canary instead of the shared-bloom (`--dedup-by-doc-shared`) variant.

Per-worker dedup leaks ~0.22% cross-worker duplicate emissions vs the
shared-bloom version (one doc reaches dup_count=8, the rest stay <=5),
but is ~47x faster to generate and avoids the asymptotic stragglers.
This pair of experiments lets us measure whether the leaked
cross-worker duplication translates into a measurable model-quality
gap — the comparison metric is in-loop evals at matched compute
against both the shared canary and the standard baseline-v02 ladders.

Data source (5.9B actual tokens / 5.0B compressed-equivalent budget):
``/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/dolma2-0625-v02/uncompressed-simulated-overlap-v02-8192-wo-replace-dedup-by-doc-perworker-5B/allenai/dolma2-tokenizer/``.

At 190M / 2xC (~7.6B tokens) the dataset cycles ~1.3x, intentional —
we're measuring distribution similarity, not testing overfitting.
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

UNCOMPRESSED_SIM_OVERLAP_PATHS = [
    "/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/dolma2-0625-v02/uncompressed-simulated-overlap-v02-8192-wo-replace-dedup-by-doc-perworker-5B/allenai/dolma2-tokenizer/*.npy",
]


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    instance_sources: list[InstanceSourceConfig] = [
        # ConcatAndChunk is fine here (not PerFileChunked): uncompressed-simulated
        # data is concatenated independent suffix samples with a separator
        # token between them — there is no context-boundary invariant between
        # adjacent emitted chunks, so frankenstein chunks at file boundaries
        # don't bisect any structural relationship. Effectively
        # suffix-sampled baseline data, analogous to how doc-sampled baseline
        # doesn't need per-file alignment either.
        ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceConfig(
                    source_paths=UNCOMPRESSED_SIM_OVERLAP_PATHS,
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
