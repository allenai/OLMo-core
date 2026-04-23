"""Idealized-overlap full-attention ladder on v02 data.

Matches `idealized_overlap_full_attn_ladder.py` structure (100% idealized
overlap, no baseline mix) with the data path pointed at the v02 output
`idealized-overlap-v02-8192-wo-replace`.

**Important context on prior bugs** — results from any idealized-overlap
training run launched before commit f0479c5f ("fix bad idealized overlap
mask") should be treated as compromised. That commit fixes how the
tree-structured attention mask handles the first token of a new branch.
The v01 idealized run (`experiments/idealized-overlap/idealized-overlap-full-attn-190M.md`,
launched at commit 5275f4c) predates the fix AND also sits on the v0.1
pad-masked s2pdf data, so it carries both the attention-mask bug and the
instance-filter differential-cleanup confound (see
`tex/instance_filter_analysis/`). This v02 rerun is the first untainted
idealized training — it uses the fixed mask code and the v02 clean-source
baseline mix (s2pdf at v0.2 instead of v0.1/s2pdf_redacted).
"""
import argparse
import logging

import olmo_core.io as io
from olmo_core.data import NumpyDatasetDType, TokenizerConfig
from olmo_core.data.composable import *
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import (
    ModelLadder,
    Olmo3ModelConfigurator,
    TransformerSize,
    WSDSChinchillaRunConfigurator,
)

log = logging.getLogger(__name__)

IDEALIZED_OVERLAP_BASE = (
    "/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/"
    "dolma2-0625-v02/idealized-overlap-v02-8192-wo-replace/"
    "allenai/dolma2-tokenizer"
)


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    instance_sources: list[InstanceSourceConfig] = [
        PreChunkedInstanceSourceConfig(
            token_paths=[f"{IDEALIZED_OVERLAP_BASE}/*-tokens.npy"],
            pos_ids_paths=[f"{IDEALIZED_OVERLAP_BASE}/*-pos_ids.npy"],
            vis_limit_paths=[f"{IDEALIZED_OVERLAP_BASE}/*-vis_limit.npy"],
            sequence_length=args.sequence_length,
            # dolma2 tokens are stored as uint32 (vocab ~100K exceeds uint16 range).
            token_dtype=NumpyDatasetDType.uint32,
        ),
    ]

    ladder = ModelLadder(
        name=args.name,
        dir=str(io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=[s for s in TransformerSize if s.approx_num_params <= 1e9],
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=Olmo3ModelConfigurator(
            model_construction_kwargs={
                "sliding_window": None,
                "attn_backend": "torch",
                "vis_limit_eos_token_id": tokenizer.eos_token_id,
            },
            rank_microbatch_size=None
            if args.rank_mbz is None
            else args.rank_mbz * args.sequence_length,
        ),
        run_configurator=WSDSChinchillaRunConfigurator(
            chinchilla_multiple=args.chinchilla_multiple
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
    main(configure_ladder)
