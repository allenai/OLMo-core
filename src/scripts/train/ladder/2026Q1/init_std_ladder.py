import argparse
import logging
import math
from dataclasses import dataclass

from olmo_core.data import TokenizerConfig
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import Olmo3ModelConfigurator, TransformerModelConfigurator
from olmo_core.nn.transformer import TransformerConfig

log = logging.getLogger(__name__)


@dataclass(kw_only=True, eq=True)
class Olmo3ModelConfiguratorWithInitStd(Olmo3ModelConfigurator):
    scale_init_std_with_d_model: bool = True

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
        model.embedding_init_std = 1.0
        model.init_std = math.sqrt(1 / model.d_model)
        return model


def configure_model(args: argparse.Namespace) -> TransformerModelConfigurator:
    return Olmo3ModelConfiguratorWithInitStd(
        rank_microbatch_size=None
        if args.rank_mbz is None
        else args.rank_mbz * args.sequence_length,
        scale_init_std_with_d_model=True,
    )


if __name__ == "__main__":
    main(configure_model=configure_model)
