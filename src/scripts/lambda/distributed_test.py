import logging

import torch

import olmo_core.distributed.utils as dist_utils
from olmo_core.train import prepare_training_environment

log = logging.getLogger(__name__)


def main():
    prepare_training_environment()
    log.info(f"Torch version: {torch.__version__}")
    log.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    log.info(f"World size: {dist_utils.get_world_size()}")
    log.info("Done!")


if __name__ == "__main__":
    main()
