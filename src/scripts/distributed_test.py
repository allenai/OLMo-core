import logging

import torch
import torch.distributed as dist

import olmo_core.distributed.utils as dist_utils
from olmo_core.train import prepare_training_environment, teardown_training_environment

log = logging.getLogger(__name__)


def main():
    prepare_training_environment()

    log.info(f"Torch version: {torch.__version__}")
    log.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    log.info(f"World size: {dist_utils.get_world_size()}")

    log.info("Checking barrier...")
    dist_utils.barrier()
    log.info("Done!")

    log.info("Checking all_reduce...")
    tensor = torch.tensor([1.0], device="cuda")
    dist.all_reduce(tensor)
    assert tensor.item() == float(dist_utils.get_world_size())
    log.info("Done!")

    log.info("Checking broadcast_object...")
    res = dist_utils.broadcast_object(dist_utils.get_rank() == 0)
    assert res is True
    log.info("Done!")

    teardown_training_environment()


if __name__ == "__main__":
    main()
