from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn

from olmo_core.train.train_module.transformer.pipeline.pipeline_schedule import (
    CustomScheduleInterleaved1F1B,
)
from olmo_core.train.train_module.transformer.pipeline.pipeline_stage import CustomPipelineStage
from olmo_core.nn.lm_head import LMOutputWithLoss


class ToyStage(nn.Module):
    d_model = 8

    def __init__(self, stage_index: int, *, is_last: bool) -> None:
        super().__init__()
        self.stage_index = stage_index
        self.is_last = is_last
        self.bias = nn.Parameter(torch.full((self.d_model,), 0.01 * (stage_index + 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor | LMOutputWithLoss:
        if x.dtype == torch.long:
            h = x.to(torch.float32).unsqueeze(-1).expand(-1, -1, self.d_model)
            h = h.to(torch.bfloat16)
        else:
            h = x
        h = h + self.bias.to(dtype=h.dtype)
        if not self.is_last:
            return h

        loss = h.to(torch.float32).mean()
        return LMOutputWithLoss(
            logits=None,
            loss=loss,
            ce_loss=loss.detach(),
            z_loss=None,
        )


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"pipeline_rma_schedule_smoke.py expects exactly 2 ranks, got {world_size}")

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    p2p_backend = os.getenv("OLMO_RMA_SCHEDULE_SMOKE_BACKEND", "nccl_rma")

    num_stages = 4
    local_stage_ids = (rank, rank + world_size)
    stages: list[CustomPipelineStage] = []
    for stage_index in local_stage_ids:
        module = ToyStage(stage_index, is_last=stage_index == num_stages - 1).to(device)
        stages.append(
            CustomPipelineStage(
                module,
                stage_index,
                num_stages,
                device,
                group=dist.group.WORLD,
                p2p_backend=p2p_backend,
            )
        )

    schedule = CustomScheduleInterleaved1F1B(stages, n_microbatches=2)
    schedule.prepare_step(global_batch_size=4, seqlen=4)
    print(f"[rank {rank}] schedule prepared", flush=True)

    if rank == 0:
        input_ids = torch.arange(16, device=device, dtype=torch.long).view(4, 4)
        print(f"[rank {rank}] starting schedule step", flush=True)
        outputs = schedule.step(input_ids)
    else:
        print(f"[rank {rank}] starting schedule step", flush=True)
        outputs = schedule.step()
    print(f"[rank {rank}] schedule step complete", flush=True)

    for stage in stages:
        for param in stage.submod.parameters():
            if param.grad is None:
                raise AssertionError(f"rank {rank} stage {stage.stage_index} has no grad")

    dist.barrier()
    if rank == 0:
        print(f"Pipeline {p2p_backend} schedule smoke test passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
