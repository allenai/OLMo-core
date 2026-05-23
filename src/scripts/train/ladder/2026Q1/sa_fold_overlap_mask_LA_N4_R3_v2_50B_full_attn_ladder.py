"""SA fold-overlap MASK long-anchor (step 2 of the new fold-overlap plan)
on v02 — N=4 folds, anchor:adj ratio = 3:1.

v2 (task #1 bug-fix): vis_limit uses scope_end_excl (no +1) and label_mask conditions on `L_k == 0 or is_fallback` for ALL k, matching packed-mask v2 convention. v1 (LA-N4-R3-50B Weka path) preserved for direct comparison.

Per-instance layout (see notes/long_anchor_design.md for the full design
spec):

    [ anchor_chunk_1 | ... | anchor_chunk_4 | adj_1 | ... | adj_4 ]
      ^---- A = 4 * 1536 = 6144 contiguous anchor ----^   ^- 4*512 -^

The anchor is one contiguous 6144-token span read from the flat
tokenized stream, split into 4 chunks of 1536 tokens. Each adj_k is the
SA-adjacent neighbor of anchor_chunk_k, emitted at adj_size=512 tokens
starting from the LCP-end of the paired chunk.

Anchor chunks attend contiguously to each other (so a doc up to 6144
tokens lives inside one anchor without an artificial fold cut); each
adj only attends to its paired anchor chunk's LCP region. Cross-adj
boundaries and within-pair L=0 starts are loss-masked
(label_mask) — analogous to the multi-pair fold-overlap MASK rule.

Pairs with the single-pair MASK ladders at the same effective fold
sizes for direct comparison of the long-anchor design (long-contiguous
anchor) vs. the single-pair design (each fold restarts the anchor
independently).

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
    "dolma2-0625-v02/sa-fold-overlap-mask-LA-N4-R3-v2-50B/allenai/dolma2-tokenizer"
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
