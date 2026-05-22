"""SA fold-overlap MASK packed-adj (no-pad variable-K) ladder on v02.

Step 1 of new_data_gen_plan, MASK encoding. Per anchor, pack multiple
SA-adjacent neighbors into the F-token adj budget. Variable K per anchor:
walk neighbors with truncate-last on the final accepted branch + fill
any leftover budget with real anchor-continuation tokens. Zero EOS pad.

Per pair (F+F=2F tokens):

    [ anchor[0:F] ] [ tail_1 ] [ tail_2 ] ... [ tail_K ] [ anchor_ext ]

Branches sorted by descending LCP. vis_limit and pos_ids encode the
tree-attention structure: each branch attends back through the shared
LCP region of anchor; branches don't see each other. Anchor extension
(real source-stream continuation) is an isolated chunk at the end with
vis_limit = total_len+1, pos_ids continuing from F. K=0 special case
collapses to a contiguous 2F baseline window.

Empirical stats: avg_K≈5.16, avg_ext≈71 (1.73%), k0_rate≈1%,
comp_ratio≈1.1463. Pairs to ICL sibling.

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
    "dolma2-0625-v02/sa-fold-overlap-mask-packed-50B/allenai/dolma2-tokenizer"
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
