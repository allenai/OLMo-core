"""
Multi-GPU execution tests for the custom pipeline-parallel stage/schedule.

``test_custom_schedule`` runs the interleaved-1F1B schedule end-to-end over a toy 4-stage model on
2 GPUs using standard NCCL point-to-point (``p2p_backend="nccl"``) — no special kernel required, so
it runs on any multi-GPU box. The ``*_rma`` tests additionally exercise the NCCL-RMA transport and
are guarded by :func:`requires_nccl_rma` (an RMA-capable NCCL is not present in CI).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.testing import (
    requires_multi_gpu,
    requires_nccl_rma,
    run_distributed_test,
)
from olmo_core.train.train_module.transformer.pipeline.p2p_transport import (
    NCCLRMAPipelineP2PTransport,
)
from olmo_core.train.train_module.transformer.pipeline.pipeline_schedule import (
    CustomScheduleInterleaved1F1B,
)
from olmo_core.train.train_module.transformer.pipeline.pipeline_stage import (
    CustomPipelineStage,
)


class ToyStage(nn.Module):
    """A minimal, single-parameter stage module for exercising the pipeline machinery."""

    d_model = 8

    def __init__(self, stage_index: int, *, is_last: bool) -> None:
        super().__init__()
        self.stage_index = stage_index
        self.is_last = is_last
        self.bias = nn.Parameter(torch.full((self.d_model,), 0.01 * (stage_index + 1)))

    def forward(self, x: torch.Tensor):
        if x.dtype == torch.long:
            h = x.to(torch.float32).unsqueeze(-1).expand(-1, -1, self.d_model).to(torch.bfloat16)
        else:
            h = x
        h = h + self.bias.to(dtype=h.dtype)
        if not self.is_last:
            return h
        loss = h.to(torch.float32).mean()
        return LMOutputWithLoss(logits=None, loss=loss, ce_loss=loss.detach(), z_loss=None)


def _run_custom_schedule(p2p_backend: str):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    num_stages = 4
    stages = []
    for stage_index in (rank, rank + world_size):
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

    if rank == 0:
        input_ids = torch.arange(16, device=device, dtype=torch.long).view(4, 4)
        schedule.step(input_ids)
    else:
        schedule.step()

    for stage in stages:
        for name, param in stage.submod.named_parameters():
            assert (
                param.grad is not None
            ), f"rank {rank} stage {stage.stage_index} param {name} has no grad"


def _run_rma_transport():
    rank = dist.get_rank()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    transport = NCCLRMAPipelineP2PTransport(group=dist.group.WORLD, device=device, num_stages=2)
    transport.prepare_step(num_microbatches=1, payload_shape=(4, 8), payload_dtype=torch.float32)
    peer = 1 - rank

    # rank 0 -> rank 1 (forward), then rank 1 -> rank 0 (backward); each receiver checks the payload.
    if rank == 0:
        src = torch.arange(32, device=device, dtype=torch.float32).view(4, 8) + 101
        transport.make_send_op(("F", 0, 1, 0), peer=peer, tensor=src).start().wait()
    else:
        op = transport.make_recv_op(("F", 0, 1, 0), peer=peer)
        op.start().wait()
        expected = torch.arange(32, device=device, dtype=torch.float32).view(4, 8) + 101
        assert torch.equal(op.recv_slot, expected)
    dist.barrier()

    if rank == 1:
        src = torch.arange(32, device=device, dtype=torch.float32).view(4, 8) + 202
        transport.make_send_op(("B", 1, 0, 0), peer=peer, tensor=src).start().wait()
    else:
        op = transport.make_recv_op(("B", 1, 0, 0), peer=peer)
        op.start().wait()
        expected = torch.arange(32, device=device, dtype=torch.float32).view(4, 8) + 202
        assert torch.equal(op.recv_slot, expected)
    dist.barrier()

    transport.close()


@requires_multi_gpu
def test_custom_schedule_nccl():
    run_distributed_test(
        _run_custom_schedule,
        world_size=2,
        backend="nccl",
        start_method="spawn",
        func_args=("nccl",),
    )


@requires_nccl_rma
def test_custom_schedule_nccl_rma():
    run_distributed_test(
        _run_custom_schedule,
        world_size=2,
        backend="nccl",
        start_method="spawn",
        func_args=("nccl_rma",),
    )


@requires_nccl_rma
def test_rma_transport():
    run_distributed_test(
        _run_rma_transport,
        world_size=2,
        backend="nccl",
        start_method="spawn",
    )
