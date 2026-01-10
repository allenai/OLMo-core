import argparse
import logging

from olmo_core.internal.ladder import main
from olmo_core.model_ladder import *
from olmo_core.optim import WSDS, DecayShape, Scheduler

log = logging.getLogger(__name__)


class SqrtDecayRunConfigurator(WSDSChinchillaRunConfigurator):
    def configure_lr_scheduler(self, num_params: int, batch_size: int) -> Scheduler:
        scheduler: WSDS = super().configure_lr_scheduler(num_params, batch_size)
        scheduler.decay_shape = DecayShape.square_root
        return scheduler


def configure_run(args: argparse.Namespace) -> SqrtDecayRunConfigurator:
    return SqrtDecayRunConfigurator(
        chinchilla_multiple=args.chinchilla_multiple,
        lr_multiplier=args.lr_multiplier,
    )


def configure_model(args: argparse.Namespace) -> ModelConfigurator:
    return Olmo3ModelConfigurator(
        rank_microbatch_size=None
        if args.rank_mbz is None
        else args.rank_mbz * args.sequence_length,
    )


if __name__ == "__main__":
    main(configure_model=configure_model, configure_run=configure_run)
