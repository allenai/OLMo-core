from __future__ import annotations

import os

import torch
import torch.distributed as dist

from olmo_core.train.train_module.transformer.pipeline.p2p_transport import (
    NCCLRMAPipelineP2PTransport,
)


def _check_equal(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(actual, expected):
        max_abs = (actual - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch: max_abs={max_abs}")


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"pipeline_rma_transport_smoke.py expects exactly 2 ranks, got {world_size}")

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    transport = NCCLRMAPipelineP2PTransport(
        group=dist.group.WORLD,
        device=device,
        num_stages=2,
    )
    transport.prepare_step(
        num_microbatches=4,
        payload_shape=(4, 8),
        payload_dtype=torch.float32,
        slot_depth=2,
    )

    peer = 1 - rank

    if rank == 0:
        src = torch.arange(32, device=device, dtype=torch.float32).view(4, 8) + 101
        handle = transport.make_send_op(("F", 0, 1, 0), peer=peer, tensor=src).start()
        handle.wait()
    else:
        op = transport.make_recv_op(("F", 0, 1, 0), peer=peer)
        handle = op.start()
        handle.wait()
        expected = torch.arange(32, device=device, dtype=torch.float32).view(4, 8) + 101
        _check_equal("fwd", op.output_tensor, expected)
        first_output = op.output_tensor
    dist.barrier()

    if rank == 0:
        src = torch.arange(32, device=device, dtype=torch.float32).view(4, 8) + 303
        handle = transport.make_send_op(("F", 0, 1, 2), peer=peer, tensor=src).start()
        handle.wait()
    else:
        op = transport.make_recv_op(("F", 0, 1, 2), peer=peer)
        handle = op.start()
        handle.wait()
        expected = torch.arange(32, device=device, dtype=torch.float32).view(4, 8) + 303
        _check_equal("fwd reused slot", op.output_tensor, expected)
        _check_equal(
            "fwd copied output after slot reuse",
            first_output,
            torch.arange(32, device=device, dtype=torch.float32).view(4, 8) + 101,
        )
    dist.barrier()

    if rank == 1:
        src = torch.arange(32, device=device, dtype=torch.float32).view(4, 8) + 202
        handle = transport.make_send_op(("B", 1, 0, 0), peer=peer, tensor=src).start()
        handle.wait()
    else:
        op = transport.make_recv_op(("B", 1, 0, 0), peer=peer)
        handle = op.start()
        handle.wait()
        expected = torch.arange(32, device=device, dtype=torch.float32).view(4, 8) + 202
        _check_equal("bwd", op.output_tensor, expected)
    dist.barrier()

    transport.close()
    dist.barrier()
    if rank == 0:
        print("Pipeline NCCL RMA transport smoke test passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
