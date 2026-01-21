import argparse
import logging

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import Olmo3ModelConfigurator, TransformerModelConfigurator
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.transformer import TransformerBlockType

log = logging.getLogger(__name__)

# This ladder has been run under the name "olmo3-peri-ln"
# https://wandb.ai/ai2-llm/olmo3-peri-ln


def add_additional_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    del cmd
    parser.add_argument(
        "--embedding-norm",
        action="store_true",
        help="Whether to apply normalization to the embedding layer.",
    )
    parser.add_argument(
        "--embedding-init-std",
        type=float,
        default=None,
        help="Standard deviation for embedding initialization.",
    )
    parser.add_argument(
        "--norm-style",
        choices=["peri", "olmo3"],
        default="olmo3",
        help="Normalization style to use.",
    )


def configure_model(args: argparse.Namespace) -> TransformerModelConfigurator:
    kwargs: dict = {}

    if args.norm_style == "peri":
        # Peri-LN (https://arxiv.org/pdf/2502.02732) plus QK-norm (equivalent to Gemma3).
        kwargs["block_name"] = TransformerBlockType.peri_norm
    elif args.norm_style != "olmo3":
        raise OLMoConfigurationError(f"Unknown norm style: {args.norm_style}")

    if args.embedding_norm:
        kwargs["embedding_norm"] = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False)

    if args.embedding_init_std is not None:
        kwargs["embedding_init_std"] = args.embedding_init_std

    return Olmo3ModelConfigurator(
        rank_microbatch_size=None
        if args.rank_mbz is None
        else args.rank_mbz * args.sequence_length,
        model_construction_kwargs=kwargs,
    )


if __name__ == "__main__":
    main(configure_model=configure_model, add_additional_args=add_additional_args)
