"""SA fold-overlap F=2048 MASK (idealized-overlap-style, variant 1) ladder on v02.

Variant 1 of the fold-overlap data-gen plan (notes/new_data_gen_plan.md):
walk the flat tokenized stream in fold_size chunks, follow ISA → SA[r±k]
adjacency to find a non-dup neighbor, emit each pair as
``anchor[0:F] + adjacent[L:F]`` with idealized-overlap-style position-ID
sharing and visibility-limit attention masking — the duplicated LCP
region is only emitted once and the divergent adjacent tail attends back
through it via shared pos_ids. Pos IDs ``0..F-1`` (anchor) and ``L..F-1``
(adjacent); vis_limit per the spec in build_sa_fold_overlap.py.

DOC_SEP → EOS substitution inside chunks (same as the UCM sibling). No
explicit fork-separator token: the mask + posid encoding fully expresses
fork structure.

Pairs to its UCM sibling
``sa_fold_overlap_ucm_v02_50B_full_attn_ladder.py`` (same anchor/adjacent
pairs, vanilla causal presentation). Comparing the two isolates the
benefit of the mask + posid encoding vs the special-token-separator
encoding.

Data: 50B tokens (no cycling at 190M / 2xC = 7.6B).
"""
import argparse
import logging

import olmo_core.io as io
from olmo_core.data import NumpyDatasetDType, TokenizerConfig
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

SA_FOLD_MASK_BASE = (
    "/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/"
    "dolma2-0625-v02/sa-fold-overlap-mask-F2048-50B/allenai/dolma2-tokenizer"
)


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    instance_sources: list[InstanceSourceConfig] = [
        PreChunkedInstanceSourceConfig(
            token_paths=[f"{SA_FOLD_MASK_BASE}/*-tokens.npy"],
            pos_ids_paths=[f"{SA_FOLD_MASK_BASE}/*-pos_ids.npy"],
            vis_limit_paths=[f"{SA_FOLD_MASK_BASE}/*-vis_limit.npy"],
            label_mask_paths=[f"{SA_FOLD_MASK_BASE}/*-label_mask.npy"],
            sequence_length=args.sequence_length,
            # dolma2 tokens are stored as uint32 (vocab ~100K exceeds uint16 range).
            token_dtype=NumpyDatasetDType.uint32,
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
            model_construction_kwargs={
                "sliding_window": None,
                "attn_backend": "torch",
                "vis_limit_eos_token_id": tokenizer.eos_token_id,
            },
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
            num_workers=8, instance_filter_config=None
        ),
    )
    return ladder


if __name__ == "__main__":
    main(configure_ladder, add_additional_args=add_smoke_args)
