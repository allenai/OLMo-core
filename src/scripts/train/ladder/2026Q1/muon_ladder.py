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
from olmo_core.optim.muon import MuonConfig

log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MuonWSDSChinchillaRunConfigurator(WSDSChinchillaRunConfigurator):
    def configure_optimizer(self, num_params: int, batch_size: int) -> MuonConfig:
        del batch_size  # unused
        # Calculate LR according to https://api.semanticscholar.org/CorpusID:270764838
        # but divide by 2 for WSD schedule (seems to work emperically).
        lr = 0.0047 * (num_params / 108_000_000) ** (-1 / 3)
        lr /= 2.0
        return MuonConfig(lr=lr, weight_decay=0.1, adjust_lr="rms_norm")


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
        run_configurator=MuonWSDSChinchillaRunConfigurator(
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
