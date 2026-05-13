"""100% uncompressed-simulated-adjacent-infix full-attention ladder on v01 data.

This is a *doc-pair-sampling baseline* — trains entirely on full-document
pairs drawn via SA-rank pointers (no left-truncation, no infix-format
restructuring). The data-gen sampler picks an SA rank, then **reads the
full document around that pointer plus the full document around rank+1**,
emitting them as ``<doc A><doc B>`` separated by EOS.

Compared head-to-head with:
  - The doc-sampling baseline at matched compute, this isolates the cost
    of *length-weighted doc sampling* (longer docs get more SA pointers,
    so they are over-sampled relative to natural doc-sampling).
  - An equivalent ICL uncompressed-sim run, this isolates the cost of
    *left-truncation* (ICL ucm-sim emits suffix-from-position; this
    variant emits the full doc around the position).

V01 because the existing ``uncompressed-simulated-adjacent-infix-8192``
data on S3 was generated against the v01 SA. Comparable v01 references
should be used (``baseline-full-attn-ladder`` exists; v01 ICL ucm-sim
does not exist as a trained run as of 2026-05-12). See
``notes/overlap_pretraining_improvements.md`` and the discussion of the
suffix-sampling axis decomposition.
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

UNCOMPRESSED_SIM_ADJACENT_INFIX_PATHS = [
    "/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/dolma2-0625-v01/uncompressed-simulated-adjacent-infix-8192-wo-replace/allenai/dolma2-tokenizer/*.npy",
]


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    instance_sources: list[InstanceSourceConfig] = [
        # ConcatAndChunk is fine here (matches the v02 ICL ucm-sim ladder):
        # uncompressed-simulated data is concatenated independent doc-pair
        # samples with EOS between them — no context-boundary invariant
        # between adjacent emitted chunks, so frankenstein chunks at file
        # boundaries don't bisect any structural relationship.
        ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceConfig(
                    source_paths=UNCOMPRESSED_SIM_ADJACENT_INFIX_PATHS,
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
