import json
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class GradientDumperCallback(Callback):
    enabled: bool = True
    start_step: int = 0
    end_step: Optional[int] = None
    step_interval: int = 1
    _metadata_saved: bool = False

    def _save_metadata(self, output_dir):
        """Save metadata about the distributed configuration (only on rank 0, only once)."""
        if get_rank() != 0:
            return

        metadata = {
            "world_size": get_world_size() if is_distributed() else 1,
            "parallel_type": None,
            "shard_degree": None,
            "num_replicas": None,
        }

        # Extract distributed config if available
        if (
            hasattr(self.trainer.train_module, "dp_config")
            and self.trainer.train_module.dp_config is not None
        ):
            dp_config = self.trainer.train_module.dp_config
            log.info(f"Extracting distributed config from dp_config: {dp_config}")
            # Get parallel type name
            if hasattr(dp_config, "name"):
                metadata["parallel_type"] = str(dp_config.name)  # type: ignore[assignment]

            # For HSDP, get shard degree and compute num_replicas
            if hasattr(dp_config, "shard_degree") and dp_config.shard_degree is not None:
                metadata["shard_degree"] = dp_config.shard_degree
                metadata["num_replicas"] = metadata["world_size"] // dp_config.shard_degree
                log.info(
                    f"Detected HSDP with shard_degree={dp_config.shard_degree}, num_replicas={metadata['num_replicas']}"
                )
            else:
                # For FSDP, shard_degree = world_size, num_replicas = 1
                metadata["shard_degree"] = metadata["world_size"]
                metadata["num_replicas"] = 1
                log.info("Detected FSDP configuration with shard_degree equal to world_size")
        else:
            # No distributed config - probably single GPU or DDP
            metadata["shard_degree"] = metadata["world_size"]
            metadata["num_replicas"] = 1
            log.info("No distributed config found; assuming single GPU or DDP")

        # Save to JSON file
        metadata_path = output_dir / "config.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        log.info(f"Saved gradient dumper metadata to {metadata_path}")
        log.info(f"  Parallel type: {metadata['parallel_type']}")
        log.info(f"  World size: {metadata['world_size']}")
        log.info(f"  Shard degree: {metadata['shard_degree']}")
        log.info(f"  Num replicas: {metadata['num_replicas']}")

    def pre_optim_step(self):
        if not self.enabled:
            return

        if self.step < self.start_step:
            return

        if self.end_step is not None and self.step > self.end_step:
            return

        if (self.step - self.start_step) % self.step_interval != 0:
            return

        output_dir = self.trainer.work_dir / "grad_dumper"
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save metadata on first successful dump
        if not self._metadata_saved:
            self._save_metadata(output_dir)
            self._metadata_saved = True
            if is_distributed():
                dist.barrier()

        assert hasattr(self.trainer.train_module, "model")
        for name, p in self.trainer.train_module.model.named_parameters():
            if p.grad is None:
                continue

            if not hasattr(p.grad, "_local_tensor"):
                log.warning(f"Gradient for '{name}' is not a DTensor - may not be properly sharded")

            filename = f"rank{get_rank()}_step{self.step}_{name}.pt"
            filepath = output_dir / filename
            log.info(f"Saving gradient of '{name}' to '{filepath}' on rank {get_rank()}")
            torch.save(p.grad.cpu(), filepath)
        log.info(f"Saved gradients for step {self.step} on rank {get_rank()}")
