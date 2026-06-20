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


def _payload(device: torch.device, offset: int) -> torch.Tensor:
    return torch.arange(32, device=device, dtype=torch.float32).view(4, 8) + offset


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"pipeline_rma_ack_transport_smoke.py expects exactly 2 ranks, got {world_size}")

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    transport = NCCLRMAPipelineP2PTransport(
        group=dist.group.WORLD,
        device=device,
        num_stages=2,
        use_ack=True,
    )
    transport.prepare_step(
        num_microbatches=3,
        payload_shape=(4, 8),
        payload_dtype=torch.float32,
        slot_depth=1,
    )
    if transport.num_slots != 2:
        raise AssertionError(f"expected one F and one B slot, got {transport.num_slots}")

    peer = 1 - rank

    if rank == 0:
        transport.make_send_op(("F", 0, 1, 0), peer=peer, tensor=_payload(device, 101)).start().wait()
        transport.make_send_op(("F", 0, 1, 1), peer=peer, tensor=_payload(device, 303)).start().wait()
    else:
        op0 = transport.make_recv_op(("F", 0, 1, 0), peer=peer)
        op0.start().wait()
        _check_equal("fwd mb0", op0.output_tensor, _payload(device, 101))
        first_output = op0.output_tensor

        op1 = transport.make_recv_op(("F", 0, 1, 1), peer=peer)
        op1.start().wait()
        _check_equal("fwd mb1", op1.output_tensor, _payload(device, 303))
        _check_equal("fwd copied output after one-lane reuse", first_output, _payload(device, 101))
    dist.barrier()

    if rank == 1:
        transport.make_send_op(("B", 1, 0, 0), peer=peer, tensor=_payload(device, 202)).start().wait()
        transport.make_send_op(("B", 1, 0, 1), peer=peer, tensor=_payload(device, 404)).start().wait()
    else:
        op0 = transport.make_recv_op(("B", 1, 0, 0), peer=peer)
        op0.start().wait()
        _check_equal("bwd mb0", op0.output_tensor, _payload(device, 202))
        first_output = op0.output_tensor

        op1 = transport.make_recv_op(("B", 1, 0, 1), peer=peer)
        op1.start().wait()
        _check_equal("bwd mb1", op1.output_tensor, _payload(device, 404))
        _check_equal("bwd copied output after one-lane reuse", first_output, _payload(device, 202))
    dist.barrier()

    transport.close()
    dist.barrier()
    if rank == 0:
        print("Pipeline NCCL RMA ack transport smoke test passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
