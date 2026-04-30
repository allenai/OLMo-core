import argparse
import logging

from olmo_core.internal.ladder import main
from olmo_core.model_ladder import Olmo3ModelConfigurator, TransformerModelConfigurator

log = logging.getLogger(__name__)


# This ladder has been run under the name "olmo3-gnope"
# https://wandb.ai/ai2-llm/olmo3-gnope


def configure_model(args: argparse.Namespace) -> TransformerModelConfigurator:
    return Olmo3ModelConfigurator(
        rank_microbatch_size=None
        if args.rank_mbz is None
        else args.rank_mbz * args.sequence_length,
        model_construction_kwargs=dict(
            no_global_rope=True,
        ),
    )


if __name__ == "__main__":
    main(configure_model=configure_model)
