import json
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class GradientDumperCallback(Callback):
    enabled: bool = True
    start_step: int = 0
    end_step: Optional[int] = None
    step_interval: int = 1
    save_first_n: Optional[int] = None

    def pre_optim_step(self):
        if not self.enabled:
            return

        if self.step < self.start_step:
            return

        if self.end_step is not None and self.step > self.end_step:
            return

        if (self.step - self.start_step) % self.step_interval != 0:
            return

        # Validate save_first_n
        if self.save_first_n is not None and self.save_first_n <= 0:
            raise ValueError(f"save_first_n must be positive, got {self.save_first_n}")

        output_dir = self.trainer.work_dir / "grad_dumper"
        output_dir.mkdir(exist_ok=True, parents=True)

        step_dir = output_dir / f"step{self.step}"
        step_dir.mkdir(exist_ok=True, parents=True)

        assert hasattr(self.trainer.train_module, "model")
        for name, p in self.trainer.train_module.model.named_parameters():
            if p.grad is None:
                continue

            grad_tensor = p.grad.detach()

            # Extract metadata before converting to local tensor
            metadata = {}
            if hasattr(grad_tensor, "placements") and grad_tensor.placements:
                for placement in grad_tensor.placements:
                    if placement.is_shard():
                        metadata["shard_dim"] = str(placement.dim)
                        break

            if hasattr(grad_tensor, "_spec") and hasattr(grad_tensor._spec, "shape"):
                metadata["full_shape"] = json.dumps(list(grad_tensor._spec.shape))

            # Convert to local tensor for safetensors
            if hasattr(grad_tensor, "to_local"):
                grad_tensor = grad_tensor.to_local()

            grad_tensor = grad_tensor.to(device="cpu", copy=True)

            if self.save_first_n is None:
                # if we want to save all the gradients, we save per-rank shards for later reconstruction
                filename = f"rank{get_rank()}_{name}.safetensors"
                filepath = step_dir / filename
                save_file({"gradient": grad_tensor}, str(filepath), metadata=metadata)
                log.info(f"Saved gradient '{name}' to '{filepath}'")
            else:
                # if we want to save only the first N elements, we gather to rank 0 and save only the first N elements
                shard_dim = int(metadata.get("shard_dim", 0))

                if shard_dim < 0 or shard_dim >= grad_tensor.ndim:
                    log.warning(
                        f"Parameter '{name}': invalid shard_dim={shard_dim} for tensor with "
                        f"{grad_tensor.ndim} dimensions, using dim 0"
                    )
                    shard_dim = 0

                if is_distributed():
                    # Create list of tensors to receive gathered shards (only on rank 0)
                    if get_rank() == 0:
                        gathered = [torch.zeros_like(grad_tensor) for _ in range(get_world_size())]
                    else:
                        gathered = None

                    # Gather all CPU tensors from all ranks to rank 0 only
                    dist.gather(grad_tensor, gather_list=gathered, dst=0)

                    if get_rank() == 0:
                        assert gathered is not None
                        full_grad = torch.cat(gathered, dim=shard_dim)

                        # cap save_first_n to actual dimension size
                        dim_size = full_grad.shape[shard_dim]
                        actual_n = min(self.save_first_n, dim_size)
                        if actual_n < self.save_first_n:
                            log.warning(
                                f"Parameter '{name}': save_first_n={self.save_first_n} exceeds "
                                f"dimension size {dim_size}, capping to {actual_n}"
                            )

                        # Slice first N elements along shard dimension
                        sliced_grad = full_grad.narrow(shard_dim, 0, actual_n)
                        slice_filename = f"{name}_first{actual_n}.safetensors"
                        slice_filepath = step_dir / slice_filename
                        save_file({"gradient": sliced_grad}, str(slice_filepath), metadata=metadata)
                        log.info(f"Saved first {actual_n} of '{name}' to '{slice_filepath}'")
                else:
                    # Single GPU: just slice and save
                    dim_size = grad_tensor.shape[shard_dim]
                    actual_n = min(self.save_first_n, dim_size)
                    if actual_n < self.save_first_n:
                        log.warning(
                            f"Parameter '{name}': save_first_n={self.save_first_n} exceeds "
                            f"dimension size {dim_size}, capping to {actual_n}"
                        )

                    sliced_grad = grad_tensor.narrow(shard_dim, 0, actual_n)
                    slice_filename = f"{name}_first{actual_n}.safetensors"
                    slice_filepath = step_dir / slice_filename
                    save_file({"gradient": sliced_grad}, str(slice_filepath), metadata=metadata)
                    log.info(f"Saved first {actual_n} of '{name}' to '{slice_filepath}'")
        log.info(f"Saved gradients for step {self.step} on rank {get_rank()}")
