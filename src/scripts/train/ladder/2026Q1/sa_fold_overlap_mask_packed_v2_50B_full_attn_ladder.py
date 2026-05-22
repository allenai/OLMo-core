"""SA fold-overlap MASK packed-adj v2 (bug-fixed) ladder on v02.

**v2 vs v1**: v1 (sa-fold-overlap-mask-packed-50B Weka path) had two
data-gen bugs identified in task #71 and fixed in commit ___:
  1. vis_limit was scope_end_excl + 1 throughout, allowing the last
     query of each segment to see segment keys when predicting across
     the boundary — bogus gradient. v2 uses scope_end_excl matching
     single-pair MASK convention.
  2. label_mask masked branch_k for k>=1 unconditionally; v2 only
     masks when L_sorted[k] == 0 (single-token context).
Combined magnitude ~0.12% of tokens. v1's run (training was canceled
at 1xC after underperforming single-pair MASK) is preserved for
comparison; v2 may or may not close the gap — the dominant suspect for
v1's underperformance is the source-mix bias from DOC_SEP-bounded
branch tails (task #72), which is unchanged in v2.

v1 packed semantics, encoding family, and worker-loop algorithm
otherwise unchanged. Variable K per anchor, anchor-continuation fill
for leftover budget, K=0 baseline window special case.

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
    "dolma2-0625-v02/sa-fold-overlap-mask-packed-v2-50B/allenai/dolma2-tokenizer"
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
