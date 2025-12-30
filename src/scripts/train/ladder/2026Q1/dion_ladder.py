import argparse
import logging
from dataclasses import dataclass

import olmo_core.io as io
from olmo_core.data import DataMix, TokenizerConfig
from olmo_core.data.composable import *
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import *
from olmo_core.model_ladder.utils import get_mix_base_dir
from olmo_core.optim.dion import DionConfig

log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DionWSDSChinchillaRunConfigurator(WSDSChinchillaRunConfigurator):
    def configure_target_batch_size(self, num_params: int) -> int:
        # One of the points of using Muon / Dion is that they can handle
        # over-large batch sizes without much degradation in data efficiency.
        # Although, they have similar critical batch sizes wrt AdamW:
        # https://arxiv.org/abs/2505.02222
        # For the purpose of comparing peak performance, we will use the same
        # batch size as AdamW. However, this does not demonstrate one of the primary
        # benefits of Muon / Dion: scaling over-large batch sizes, which allows
        # for additional use of data parallelism.
        return super().configure_target_batch_size(num_params)

    def configure_optimizer(self, num_params: int, batch_size: int) -> DionConfig:
        del num_params  # unused
        del batch_size  # unused

        # Dion optimal lr stays at 0.01 across model scales
        # https://arxiv.org/pdf/2504.05295
        return DionConfig(lr=0.01, weight_decay=0.1)


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    instance_sources: list[InstanceSourceConfig] = [
        ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceMixConfig(
                    tokenizer=tokenizer,
                    mix=DataMix.OLMo_mix_0925_official
                    if args.cluster == "lambda"
                    else DataMix.OLMo_mix_0925,
                    mix_base_dir=get_mix_base_dir(args.cluster),
                )
            ],
            sequence_length=args.sequence_length,
        ),
    ]
    ladder = ModelLadder(
        name=args.name,
        dir=str(io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=list(TransformerSize),
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=TransformerModelConfigurator(),
        run_configurator=DionWSDSChinchillaRunConfigurator(
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
