import argparse
import logging
import math
from dataclasses import dataclass

from olmo_core.data import TokenizerConfig
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import Olmo3ModelConfigurator, TransformerModelConfigurator
from olmo_core.nn.transformer import TransformerConfig

log = logging.getLogger(__name__)


@dataclass(kw_only=True, eq=True)
class Olmo3ModelConfiguratorForInitStyle(Olmo3ModelConfigurator):
    init_style: str = "dirk"

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
        if self.init_style == "dirk":
            model.embedding_init_std = 1.0
            model.init_std = math.sqrt(1 / model.d_model)
        elif self.init_style != "olmo3":
            raise OLMoConfigurationError(f"Unknown init style: {self.init_style}")
        return model


def add_additional_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    del cmd
    parser.add_argument(
        "--init-style",
        choices=["olmo3", "dirk"],
        default="dirk",
        help="Initialization style to use.",
    )


def configure_model(args: argparse.Namespace) -> TransformerModelConfigurator:
    return Olmo3ModelConfiguratorForInitStyle(
        rank_microbatch_size=None
        if args.rank_mbz is None
        else args.rank_mbz * args.sequence_length,
        init_style=args.init_style,
    )


if __name__ == "__main__":
    main(configure_model=configure_model, add_additional_args=add_additional_args)
