"""Isolated optimizer-gather prototype; NOT connected to a training configuration.

Compare the actual optimizer's packed gather against direct gathers for large
parameters plus bounded packed gathers for smaller parameters. This deliberately
keeps synchronous collectives and FP32-master -> BF16-model conversion unchanged.
Two-rank intra-node timings cannot establish a 64-rank training improvement.
"""

import json
import os
import statistics
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from olmo_core.optim.moe_optimizer import (
    OLMoDDPOptimizer,
    _FlatModelParamSyncEntry,
    _FlatModelParamSyncGroup,
)

MIB = 1024**2


def gather_buckets(entries, element_size, large_bytes, small_bucket_bytes):
    """Keep large entries individual; coalesce small entries in deterministic order."""
    if min(element_size, large_bytes, small_bucket_bytes) <= 0:
        raise ValueError("Gather thresholds and element size must be positive")
    large, small = [], []
    for entry in entries:
        (large if entry.numel * element_size >= large_bytes else small).append(entry)
    buckets = [[entry] for entry in large]
    bucket, byte_count = [], 0
    for entry in small:
        entry_bytes = entry.numel * element_size
        if bucket and byte_count + entry_bytes > small_bucket_bytes:
            buckets.append(bucket)
            bucket, byte_count = [], 0
        bucket.append(entry)
        byte_count += entry_bytes
    if bucket:
        buckets.append(bucket)
    return buckets


@torch.no_grad()
def direct_large_gather(owner, large_bytes=256 * MIB, small_bucket_bytes=64 * MIB):
    """Prototype only: write single-parameter gathers directly into the model view."""
    for group in owner._flat_model_sync_groups.values():
        for entry in group.replicated_entries:
            entry.flat_slice.copy_(owner.states[entry.state_key].to_local().reshape(-1))
        if not group.sharded_entries:
            continue
        if group.world_size == 1:
            for entry in group.sharded_entries:
                entry.flat_slice.copy_(owner.states[entry.state_key].to_local().reshape(-1))
            continue
        assert group.process_group is not None
        for bucket in gather_buckets(
            group.sharded_entries, group.dtype.itemsize, large_bytes, small_bucket_bytes
        ):
            if len(bucket) == 1:
                entry = bucket[0]
                local = owner.states[entry.state_key].to_local().reshape(-1).to(group.dtype)
                assert entry.flat_slice.is_contiguous()
                assert entry.numel == group.world_size * local.numel()
                # One parameter is already rank-major within its contiguous final view.
                # With multiple parameters, this would be the WRONG output layout.
                dist.all_gather_into_tensor(entry.flat_slice, local, group=group.process_group)
            else:
                local_numel = sum(entry.local_numel for entry in bucket)
                packed = torch.empty(local_numel, device=owner.device, dtype=group.dtype)
                gathered = torch.empty(
                    group.world_size * local_numel, device=owner.device, dtype=group.dtype
                )
                offset = 0
                for entry in bucket:
                    packed[offset : offset + entry.local_numel].copy_(
                        owner.states[entry.state_key].to_local().reshape(-1)
                    )
                    offset += entry.local_numel
                dist.all_gather_into_tensor(gathered, packed, group=group.process_group)
                matrix = gathered.view(group.world_size, local_numel)
                offset = 0
                for entry in bucket:
                    entry.sharded_target.copy_(matrix[:, offset : offset + entry.local_numel])
                    offset += entry.local_numel
                # No cached global buffer remains live between optimizer updates.
                del packed, gathered, matrix


class LocalMain:
    """Expose just the local-master interface used by the real gather method."""

    def __init__(self, tensor):
        self.tensor = tensor

    def to_local(self):
        """Return the independently owned FP32 master storage."""
        return self.tensor


def make_owner(specs, device, dtype, varying_elements=False):
    """Build actual optimizer sync-entry/group objects without a model or Adam state.

    Each spec contains (tag, process group, [(full element count, sharded), ...]).
    Source masters do not alias model storage. Test-only synthetic inputs must not
    be confused with a complete real-model parameter inventory or optimizer test.
    """
    owner = SimpleNamespace(device=device, states={}, _flat_model_sync_groups=OrderedDict())
    for tag, pg, entries_spec in specs:
        world_size = 1 if pg is None else dist.get_world_size(pg)
        rank = 0 if pg is None else dist.get_rank(pg)
        flat = torch.empty(sum(n for n, _ in entries_spec), device=device, dtype=dtype)
        entries, replicated = [], []
        offset, local_offset = 0, 0
        for index, (numel, sharded) in enumerate(entries_spec):
            assert not sharded or numel % world_size == 0
            local_numel = numel // world_size if sharded else numel
            state_key = f"{tag}.{index}.main"
            master = torch.full(
                (local_numel,),
                index % 23 + (rank * 0.125 if sharded else 0) + 0.00035,
                device=device,
                dtype=torch.float32,
            )
            if varying_elements:
                master.add_(torch.arange(local_numel, device=device).remainder(31) / 64)
            owner.states[state_key] = LocalMain(master)
            view = flat[offset : offset + numel]
            entry = _FlatModelParamSyncEntry(
                state_key=state_key,
                param=torch.nn.Parameter(view, requires_grad=False),
                flat_slice=view,
                sharded_target=view.view(world_size, local_numel) if sharded else None,
                numel=numel,
                is_sharded=sharded,
                local_numel=local_numel if sharded else 0,
                local_offset=local_offset,
            )
            (entries if sharded else replicated).append(entry)
            if sharded:
                local_offset += local_numel
            offset += numel
        owner._flat_model_sync_groups[tag] = _FlatModelParamSyncGroup(
            tag=tag,
            dtype=dtype,
            flat_buffer=flat,
            sharded_entries=entries,
            replicated_entries=replicated,
            total_sharded_local_numel=local_offset,
            process_group=pg,
            world_size=world_size,
        )
    return owner


def verify_constant_inputs(owner):
    """Check every output element and source master outside the measured window."""
    for group in owner._flat_model_sync_groups.values():
        group_rank = 0 if group.process_group is None else dist.get_rank(group.process_group)
        for entry in group.sharded_entries + group.replicated_entries:
            index = int(entry.state_key.split(".")[-2])
            for rank in range(group.world_size if entry.is_sharded else 1):
                value = index % 23 + (rank * 0.125 if entry.is_sharded else 0) + 0.00035
                expected = torch.tensor(value, device=owner.device, dtype=group.dtype)
                output = entry.sharded_target[rank] if entry.is_sharded else entry.flat_slice
                assert bool((output == expected).all().item()), entry.state_key
            master = owner.states[entry.state_key].to_local()
            value = index % 23 + (group_rank * 0.125 if entry.is_sharded else 0) + 0.00035
            assert bool((master == value).all().item()), entry.state_key


def main():
    """Time the actual packed path / direct-large prototype / actual packed path."""
    torch.set_num_threads(1)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    report = {
        "source_commit": os.environ.get("GIT_REF"),
        "gpu": torch.cuda.get_device_name(),
        "world_size": dist.get_world_size(),
        "torch": torch.__version__,
        "nccl": torch.cuda.nccl.version(),
        "cases": [],
        "caveat": "Synthetic weight layout, two-rank NVLink, not a 64-GPU training speedup.",
    }
    try:
        # 15 pairs with the real expert tensor byte sizes, plus synthetic small tensors.
        # This is deliberately not labeled an exact full-model inventory.
        full_bytes = [n * MIB for _ in range(15) for n in (4, 512, 1024, 8)]
        specs = [("dp", dist.group.WORLD, [(n // 2, True) for n in full_bytes] + [(128, False)])]
        owner = make_owner(specs, device, torch.bfloat16)
        group = owner._flat_model_sync_groups["dp"]
        plan = gather_buckets(group.sharded_entries, 2, 256 * MIB, 64 * MIB)
        report.update(
            full_sharded_output_gib=sum(full_bytes) / 1024**3,
            local_fp32_master_gib=sum(x.tensor.nbytes for x in owner.states.values()) / 1024**3,
            prototype_collectives=len(plan),
            prototype_direct_collectives=sum(len(x) == 1 for x in plan),
            prototype_max_packed_output_mib=max(
                (
                    sum(entry.numel * 2 for entry in bucket) / MIB
                    for bucket in plan
                    if len(bucket) > 1
                ),
                default=0,
            ),
        )
        for label, operation in (
            ("packed-before", OLMoDDPOptimizer._copy_main_params_to_flat_model_buffers),
            ("direct-large", direct_large_gather),
            ("packed-after", OLMoDDPOptimizer._copy_main_params_to_flat_model_buffers),
        ):
            cuda_ms, wall_ms = [], []
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            base_allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            for iteration in range(25):
                dist.barrier()
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                wall_start = time.perf_counter()
                start.record()
                operation(owner)
                end.record()
                end.synchronize()
                wall_elapsed = (time.perf_counter() - wall_start) * 1000
                elapsed = torch.tensor([start.elapsed_time(end), wall_elapsed], device=device)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                if iteration >= 5:
                    values = elapsed.tolist()
                    cuda_ms.append(values[0])
                    wall_ms.append(values[1])
            extra_peak = torch.tensor(
                torch.cuda.max_memory_allocated() - base_allocated, device=device
            )
            dist.all_reduce(extra_peak, op=dist.ReduceOp.MAX)
            verify_constant_inputs(owner)
            entry = {
                "arm": label,
                "median_cuda_ms": statistics.median(cuda_ms),
                "mean_cuda_ms": statistics.mean(cuda_ms),
                "median_wall_ms": statistics.median(wall_ms),
                "all_cuda_ms": cuda_ms,
                "all_wall_ms": wall_ms,
                "additional_peak_allocated_gib": extra_peak.item() / 1024**3,
                "correct": True,
            }
            report["cases"].append(entry)
            if dist.get_rank() == 0:
                print("MODEL_GATHER_BENCH", json.dumps(entry), flush=True)
                Path("/results/model-gather-bench.json").write_text(json.dumps(report, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
