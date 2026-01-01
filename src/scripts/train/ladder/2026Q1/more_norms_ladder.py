import argparse
import logging
from dataclasses import dataclass

from olmo_core.data import TokenizerConfig
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import *
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


def configure_model(args: argparse.Namespace) -> ModelConfigurator:
    return MoreNormsModelConfigurator(
        style=args.norm_style,
        embedding_norm=args.embedding_norm,
        rank_microbatch_size=None
        if args.rank_mbz is None
        else args.rank_mbz * args.sequence_length,
    )


if __name__ == "__main__":
    main(configure_model=configure_model, add_additional_args=add_additional_args)
