import argparse
import logging
from dataclasses import dataclass

import olmo_core.io as io
from olmo_core.data import DataMix, TokenizerConfig
from olmo_core.data.composable import *
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import *
from olmo_core.model_ladder.utils import get_mix_base_dir
from olmo_core.nn.transformer import TransformerBlockType, TransformerConfig

log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MoreNormsModelConfigurator(Olmo3ModelConfigurator):
    style: str = "peri"
    embedding_norm: bool = False

    def configure_model(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        tokenizer: TokenizerConfig,
        device_type: str,
    ) -> TransformerConfig:
        model = super().configure_model(
            size_spec=size_spec,
            sequence_length=sequence_length,
            tokenizer=tokenizer,
            device_type=device_type,
        )

        if self.style == "peri":
            # Peri-LN (https://arxiv.org/pdf/2502.02732) plus QK-norm (equivalent to Gemma3).
            model.block.name = TransformerBlockType.peri_norm
        else:
            raise OLMoConfigurationError(f"Unknown style: {self.style}")

        if self.embedding_norm:
            model.embedding_norm = model.block.layer_norm

        return model


def add_additional_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    del cmd
    parser.add_argument(
        "--embedding-norm",
        action="store_true",
        help="Whether to apply normalization to the embedding layer.",
    )
    parser.add_argument(
        "--norm-style",
        choices=["peri"],
        default="peri",
        help="Normalization style to use.",
    )


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
        project=args.project,
        dir=str(io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=list(TransformerSize),
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=MoreNormsModelConfigurator(
            style=args.norm_style,
            embedding_norm=args.embedding_norm,
            rank_microbatch_size=args.rank_mbz * args.sequence_length,
        ),
        run_configurator=WSDSChinchillaRunConfigurator(
            chinchilla_multiple=args.chinchilla_multiple,
            lr_multiplier=args.lr_multiplier,
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
    main(configure_ladder, add_additional_args=add_additional_args)
