import argparse
import logging
import math
from dataclasses import dataclass

import olmo_core.io as io
from olmo_core.data import DataMix, TokenizerConfig
from olmo_core.data.composable import *
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import *
from olmo_core.model_ladder.utils import get_mix_base_dir
from olmo_core.optim import SkipStepMuonConfig

log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MuonWSDSChinchillaRunConfigurator(WSDSChinchillaRunConfigurator):
    def configure_target_batch_size(self, num_params: int) -> int:
        # Calculate global batch size according to https://api.semanticscholar.org/CorpusID:270764838
        # which assumes a sequence length of 2048.
        muon_multiplier = 2
        return round(muon_multiplier * 2048 * 160 * (num_params / 108_000_000) ** (2 / 3))

    def configure_optimizer(self, num_params: int, batch_size: int) -> SkipStepMuonConfig:
        del batch_size  # unused
        # Calculate LR according to https://api.semanticscholar.org/CorpusID:270764838
        # but divide by 2 for WSD schedule (seems to work emperically).
        lr = 0.0047 * (num_params / 108_000_000) ** (-1 / 3)
        lr /= 2.0
        return SkipStepMuonConfig(  # same lr for Muon (moonlight) and AdamW
            lr=lr,
            weight_decay=0.1,
            _adam_lr=lr,
            _adam_betas=(
                0.9,
                0.95,  # NOTE: paper above suggest using larger beta2 (~0.99) for small batch sizes.
            ),
            # Muon auto sets emb weight decay to 0.0, so we don't need to set it here.
        )

    def configure_chinchilla_periods(self, num_params: int) -> tuple[int, list[float]]:
        # NOTE! Chinchilla periods are probably different for Muon.
        # Warm up 1 token per parameter according to https://api.semanticscholar.org/CorpusID:270764838
        warmup = num_params

        # Generate Chinchilla (decay) periods as multiples of two, but at least the minimum.
        chinchilla_periods: list[float] = []
        max_pow = math.log(self.chinchilla_multiple, 2)
        assert max_pow.is_integer()  # checked in `__post_init__()` as well.
        for p in range(-1, int(max_pow) + 1):
            period = 2**p
            chinchilla_periods.append(period)

        return warmup, chinchilla_periods


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
