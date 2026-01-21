import argparse
import logging

from olmo_core.internal.ladder import main
from olmo_core.model_ladder import *

log = logging.getLogger(__name__)


# A early version of this ladder has been run under the name "olmo3-baseline-ladder"
# https://wandb.ai/ai2-llm/olmo3-baseline-ladder


def configure_model(args: argparse.Namespace) -> ModelConfigurator:
    return Olmo3ModelConfigurator(
        rank_microbatch_size=None
        if args.rank_mbz is None
        else args.rank_mbz * args.sequence_length,
    )


if __name__ == "__main__":
    main(configure_model=configure_model)
