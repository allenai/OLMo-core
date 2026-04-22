"""100% uncompressed-simulated-overlap full-attention ladder on v02 data.

This is a *suffix-sampling baseline* — trains entirely on suffix-array
sampled sequences instead of documents. Compared against the standard
baseline (trained on doc-sampled mixes), this isolates what training on
sampled suffixes does vs. training on sampled documents.

Unlike earlier uncompressed-simulated experiments, we do **not** adjust the
token budget to compensate for the format's expansion ratio. The chinchilla
multiple is applied as-is, and the dataset just cycles through its actual
tokens. The "uncompressed-simulated" name is a historical artifact (the
format doesn't wrap content in `<o>`/`</o>` markers and doesn't exploit
LCP-based compression — it just emits full sampled suffixes separated by
a sentinel token).
"""
import argparse
import logging

import olmo_core.io as io
from olmo_core.data import TokenizerConfig
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

UNCOMPRESSED_SIM_OVERLAP_PATHS = [
    "/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/dolma2-0625-v02/uncompressed-simulated-overlap-v02-8192-wo-replace/allenai/dolma2-tokenizer/*.npy",
]


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    instance_sources: list[InstanceSourceConfig] = [
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

    ladder = ModelLadder(
        name=args.name,
        dir=str(io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=[s for s in TransformerSize if s.approx_num_params <= 1e9],
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=Olmo3ModelConfigurator(
            model_construction_kwargs={"sliding_window": None},
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
            num_workers=8, instance_filter_config=InstanceFilterConfig()
        ),
    )
    return ladder


if __name__ == "__main__":
    main(configure_ladder)
