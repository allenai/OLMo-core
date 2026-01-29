import argparse
import logging

import olmo_core.io as io
from olmo_core.data import DataMix, TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    InstanceFilterConfig,
    NumpyDocumentSourceMixConfig,
)
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import (
    ModelLadder,
    Olmo3ModelConfigurator,
    TransformerSize,
    WSDSChinchillaRunConfigurator,
)
from olmo_core.train import callbacks

log = logging.getLogger(__name__)

# use my project for testing
DEFAULT_PROJECT = "bailey-testing"


class ModelMergingLadder(ModelLadder):
    """ModelLadder that includes ModelMergeCallback for weight averaging."""

    def _configure_trainer(self, size_spec: str, for_benchmarking: bool = False):
        config = super()._configure_trainer(size_spec, for_benchmarking)
        config.callbacks["model_merger"] = callbacks.ModelMergeCallback(
            merge_last_n_steps=100,
            enabled=not for_benchmarking,
        )
        return config


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    return ModelMergingLadder(
        name=args.name,
        project=args.project or DEFAULT_PROJECT,
        dir=str(io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=list(TransformerSize),
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=Olmo3ModelConfigurator(
            rank_microbatch_size=None
            if args.rank_mbz is None
            else args.rank_mbz * args.sequence_length,
        ),
        run_configurator=WSDSChinchillaRunConfigurator(
            chinchilla_multiple=args.chinchilla_multiple,
            lr_multiplier=args.lr_multiplier,
            stepped_schedule=args.stepped_schedule,
        ),
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=[
            ConcatAndChunkInstanceSourceConfig(
                sources=[
                    NumpyDocumentSourceMixConfig(
                        tokenizer=tokenizer, mix=DataMix.OLMo_mix_0925, mix_base_dir="gs://ai2-llm/"
                    )
                ],
                sequence_length=args.sequence_length,
            ),
        ],
        data_loader=ComposableDataLoaderConfig(
            num_workers=8, instance_filter_config=InstanceFilterConfig()
        ),
    )


if __name__ == "__main__":
    main(configure_ladder=configure_ladder)
