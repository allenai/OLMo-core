"""Contrived SA-baseline full-attention ladder on v02 50B per-worker
dedup random-doc-sample data, **clean re-run** of
``random_doc_sample_v02_dedup_perworker_50B_full_attn_ladder``.

Companion to the original contrived ladder. The original's data was
contaminated by an append-mode write bug in
``build_random_doc_sample.py``: the first crashed run left partial
shards behind, the fixed re-run appended on top, and the model trained
on a hybrid of ~239B uint32 tokens (vs the 50B reported). Final C4 CE
was 3.89 (+0.40 nat above baseline) — obviously broken signal.

This run uses the clean data produced after commit ``70c726d`` (which
fixes the append bug + adds a fail-fast on existing output) at output
path ``...random-doc-sample-v02-8192-dedup-by-doc-perworker-fixed-50B/``.
Same builder args, same seed, same dedup mode — only difference vs the
original is a fresh output dir + the per-shard "wb"-then-"ab" fix.

Hypothesis test (now uncontaminated): if this run still shows the +0.03
nat gap vs the standard doc-sampling baseline, the gap is from
something deeper than the SA-sampling distribution itself. If the gap
closes, it's the truncation/overlap structure in the existing SA-based
builders.

Data: 50B token target (no cycling at 190M / 2xC = 7.6B).
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

PATHS = [
    "/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/dolma2-0625-v02/random-doc-sample-v02-8192-dedup-by-doc-perworker-fixed-50B/allenai/dolma2-tokenizer/*.npy",
]


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    instance_sources: list[InstanceSourceConfig] = [
        ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceConfig(
                    source_paths=PATHS,
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
