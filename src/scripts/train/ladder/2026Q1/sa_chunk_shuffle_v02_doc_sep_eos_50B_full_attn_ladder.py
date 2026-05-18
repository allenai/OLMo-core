"""SA chunk-shuffle (DOC_SEP→EOS substituted) full-attention ladder, v02.

Trains directly on the SA tokenized data — but rather than reading the raw
SA tokenized files (which are byte-clustered per-source within each shard
and use DOC_SEP=0xFFFFFFFF as the doc-boundary marker), we pre-process a
50B-token sample on the data-gen box:

  - Enumerate all max_seq_len-sized chunks across all 25 SA shards
  - Uniformly sample ~0.84% of chunks (target 50B / 5.93T)
  - Substitute DOC_SEP → EOS (100257) at write time
  - Write standard uint32 LE .npy shards (cross-shard shuffled per worker)

The result is a globally-mixed, EOS-bounded 50B-token corpus that is the
SA-stream analog of the baseline doc-shuffle: same source-mix (preserved by
uniform chunk sampling from flat tokenized data), no SA-pointer-induced
source bias. Trained with vanilla ConcatAndChunkInstanceSource — no
custom data-loader filter required.

Diagnostic role: control variant for the new fold-overlap data-gen plan
(grounded chunks alone, before adding any SA-adjacent-suffix overlap
content). If this lands close to the standard baseline, the path of
"sample from SA flat tokens with DOC_SEP→EOS substitution" is viable.

Data: 50B tokens (no cycling at 190M / 2xC = 7.6B).
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
    "/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/dolma2-0625-v02-sa-shuffled/sa-chunk-shuffle-50B/allenai/dolma2-tokenizer/*.npy",
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
