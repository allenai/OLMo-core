# OLMo-Owned Symmetric Memory Backend

## Problem

PyTorch's NVSHMEM-backed `torch.distributed._symmetric_memory.empty()` allocates through the
backend's bootstrap group, which is hardwired to c10d group `"0"` in the stock backend. The EP
group is only applied later at `rendezvous()`. In EP-only runs this is usually tolerable because
all ranks allocate in the same order. In EP+PP runs, lazy allocation can happen while only one PP
stage is executing a MoE block. Ranks in another PP stage may be waiting in pipeline receive, so
the allocating ranks can block before the EP group is ever used.

This is not fixed by changing only the Python call site. NVSHMEM symmetric allocation itself is
collective over the initialized NVSHMEM PE universe. The correct control point is therefore the
NVSHMEM initialization universe and the allocation plan, not just `rendezvous(group=...)`.

## Desired Feature

The feature OLMo needs is essentially one of these equivalent forms:

```text
subgroup-scoped SymmMem allocation
initialize NVSHMEM with a chosen c10d subgroup
make the symmetric allocation universe explicit instead of always using group "0"
```

In stock PyTorch, the NVSHMEM symmetric-memory allocator does not provide that control. The Python
`symm_mem.empty()` API has no `group` argument, but the limitation is deeper than Python. The
underlying NVSHMEM allocator rejects a group name for allocation and initializes NVSHMEM from c10d
group `"0"`. The EP or subgroup argument only appears later at `rendezvous()`.

So this is not a neat Python-level workaround. A proper fix requires backend control: either patch
PyTorch's NVSHMEM symmetric-memory backend, or provide an OLMo-owned backend that initializes
NVSHMEM with the subgroup that actually participates in symmetric-memory communication.

## PyTorch Assumption That Breaks Uneven PP

The stock PyTorch backend effectively assumes that the ranks in c10d group `"0"` can participate in
the same symmetric-memory allocation sequence. That assumption is reasonable for simple EP-only
programs, but it is too strong for pipeline parallelism.

Pipeline stages are allowed to have different layer counts. For example, an 8-rank job may be laid
out as:

```text
global ranks 0-3: PP stage 0
global ranks 4-7: PP stage 1

EP groups inside each stage:
  stage 0: [0, 2], [1, 3]
  stage 1: [4, 6], [5, 7]
```

If stage 0 owns one more MoE layer than stage 1, or if the 1F1B schedule reaches stage 0's first
MoE allocation before stage 1 reaches any MoE allocation, the runtime can look like:

```text
ranks 0-3: enter OLMo MoE block
ranks 0-3: call symm_mem.empty()
ranks 0-3: PyTorch NVSHMEM backend bootstraps/allocates through group "0"

ranks 4-7: wait in PP recv
ranks 4-7: do not call symm_mem.empty()
```

The user-visible log stops before `rendezvous()`:

```text
shared_symm_alloc_begin ... group=15 shape=(40960, 3584)
```

and never reaches the line after `symm_mem.empty()`. The EP group name printed here is misleading:
the tensor is intended for that EP group, but PyTorch has not reached the EP-group rendezvous yet.
It is still inside NVSHMEM allocation/bootstrap.

The important consequence is:

```text
Uneven PP layers are valid model structure.
Uneven PP allocation order is valid pipeline behavior.
PyTorch's stock NVSHMEM symmetric allocator treats allocation as if group "0" can still move
together, which is the assumption that fails.
```

We should not solve this by padding model layers or forcing all PP stages to execute fake MoE work.
The allocation universe should match the communication universe, or allocations should be planned
outside the PP schedule.

## Related Pytorch Issues

One is directly in the same area: [#170479](https://github.com/pytorch/pytorch/issues/170479), where the reporter says they are implementing EP using torch.distributed._symmetric_memory with PyTorch 2.9 and NVSHMEM, enabling SymmMem for WORLD and a device-mesh group; they also note that without enabling WORLD, symm_mem.empty() fails with get_group_info: no group info associated with the group name 0, and they ask what the expected way is to enable SymmMem for EP.  

There is also [#173514](https://github.com/pytorch/pytorch/issues/173514), “Improve documentation of assumptions for symmetric memory,” which asks PyTorch to clarify the assumptions around empty, rendezvous, same-order calls, same sizes, and memory-pool behavior. That issue is not exactly “two NVSHMEM worlds,” but it is about the same underdocumented symmetry constraints that are biting you.

## Goal

Own the symmetric-memory semantics used by OLMo's EP no-sync path:

- Initialize NVSHMEM over the ranks that actually use the symmetric buffers.
- Allocate symmetric tensors through OLMo code instead of PyTorch's stock `symm_mem.empty()`.
- Keep tensors as regular `torch.Tensor` values so the rest of the model can use normal PyTorch
  operations.
- Remove `c10d::symmetric_memory::rendezvous()` from OLMo's rowwise CUDA kernels.
- Keep the legacy PyTorch symmetric-memory path available for comparison and rollback.

## Initial Scope

The active training configs use:

```python
USE_NO_SYNC_EP = True
USE_ROWWISE_A2A = True
USE_FP8 = False
```

For this path, the hot kernels are already OLMo-owned in
`olmo_core.kernels._symm_mem_vdev2d_ext_gpu`. The first implementation replaces the allocation and
rowwise rendezvous dependency for:

- `rowwise_dispatch_put`
- `rowwise_combine_get`
- `rowwise_gather_get`

The legacy 1D `torch.ops.symm_mem.all_to_all_vdev` path remains on the PyTorch backend until it is
ported or removed. If `OLMO_USE_OWN_SYMM_MEM=1`, non-rowwise EP no-sync is rejected early because
the legacy `torch.ops.symm_mem.all_to_all_vdev` op cannot consume OLMo-owned symmetric tensors.

## Backend Shape

Python API:

```python
from olmo_core.kernels import olmo_symm_mem

tensor = olmo_symm_mem.empty(shape, dtype=dtype, device=device, group=ep_group)
olmo_symm_mem.rendezvous(tensor, group=ep_group)
```

`rendezvous(..., barrier=True)` registers the group metadata and, by default, runs a process-group
barrier after allocation. The barrier keeps peers in the same EP group from entering the next
symmetric allocation while another peer is still finishing the previous one. Callers that can prove
the allocation sequence is already externally synchronized may pass `barrier=False`.

C++/CUDA extension responsibilities:

- `olmo_symm_get_unique_id()`
- `olmo_symm_init(unique_ids, rank, world_size, device_idx)`
- `olmo_symm_empty(shape, dtype, device)`
- `olmo_symm_register_group(group_name, rank_to_pe)`
- `olmo_symm_has_group(group_name)`

The first version initializes NVSHMEM over the EP process group passed to `empty()`. That is even
smaller than a PP-stage-local world and matches the current rowwise communication pattern: ranks in
different EP groups do not communicate through symmetric memory. If a future topology needs several
EP groups inside one process to share a single NVSHMEM world, the same backend can be extended to
bootstrap on the PP-stage group and register multiple EP subgroup mappings.

The rowwise CUDA entry points require that the OLMo group is registered. They hard-fail instead of
falling back to `c10d::symmetric_memory::rendezvous()` so the EP no-sync path cannot silently
re-enter PyTorch's stock SymmMem allocation assumptions.

EP no-sync also prewarms OLMo-owned symmetric buffers by default
(`OLMO_OWN_SYMM_PREWARM=1`). Prewarming allocates the per-block/shared symmetric buffers before the
pipeline dry run, so lazy allocation does not happen inside an interleaved PP schedule. This should
stay enabled for PP jobs; disabling it is mainly useful for debugging allocation behavior.

### Example: Current EP-Local Bootstrap

For an EP group `[0, 2]` inside PP stage 0:

```text
global rank 0: EP rank 0, NVSHMEM PE 0
global rank 2: EP rank 1, NVSHMEM PE 1
```

Only ranks 0 and 2 call `nvshmemx_init_attr()` for this backend instance. A symmetric allocation
for this EP group is therefore collective over two PEs, not over the whole 8-rank job. Route tensors
continue to use EP-local rank IDs:

```text
dst_ranks = 0 means EP peer 0, not global rank 0
dst_ranks = 1 means EP peer 1, not global rank 1
```

The backend registers the mapping:

```text
EP rank -> NVSHMEM PE
0       -> 0
1       -> 1
```

and kernels translate route peers through that mapping.

### Example: Future PP-Stage Bootstrap

If one process ever needs multiple EP groups inside a PP stage to share one NVSHMEM world, the
bootstrap group can instead be the PP stage:

```text
PP stage 0 NVSHMEM world: global ranks [0, 1, 2, 3]

EP group A: [0, 2]
  EP rank 0 -> NVSHMEM PE 0
  EP rank 1 -> NVSHMEM PE 2

EP group B: [1, 3]
  EP rank 0 -> NVSHMEM PE 1
  EP rank 1 -> NVSHMEM PE 3
```

In that model, `nvshmem_malloc()` is collective over the PP stage, but never over unrelated PP
stages. This still avoids the rank 0-3 versus rank 4-7 deadlock, while allowing several EP groups
inside one PP-stage NVSHMEM universe.

## Kernel Semantics

For OLMo-owned rowwise kernels, peer ranks in route tensors are EP-group ranks. The custom backend
registers a `group_name -> rank_to_pe` mapping. Kernels translate route peers through that mapping
instead of asking PyTorch's `SymmetricMemory` handle for group metadata.

With EP-group-local NVSHMEM initialization, the mapping is normally identity:

```text
EP rank 0 -> NVSHMEM PE 0
EP rank 1 -> NVSHMEM PE 1
...
```

The kernels use `NVSHMEM_TEAM_WORLD` for NVSHMEM barriers and collective launches because, under
this backend, NVSHMEM world is already the EP group.

More generally:

```text
PyTorch/c10d world:
  all ranks in the training job
  used for PP, DP, checkpoint coordination, etc.

OLMo NVSHMEM world:
  ranks that participate in this symmetric-memory allocation universe
  initially EP-local
  possibly PP-stage-local later

EP group:
  logical communication group for MoE routes
  represented in kernels by route rank IDs and rank_to_pe mapping
```

These three concepts should remain separate. The PyTorch backend conflates the allocation world
with c10d group `"0"`; this backend makes the allocation world explicit.

## Rollback

Set:

```bash
export OLMO_USE_OWN_SYMM_MEM=0
```

to use the previous PyTorch symmetric-memory allocation path for non-PP experiments. This rollback
is intentionally rejected for EP no-sync plus pipeline parallelism, because that combination is the
case where PyTorch's group-`"0"` allocation assumption can deadlock. Use the OLMo-owned backend for
EP+PP.

## Prewarm Policy

Prewarm is not required to avoid the EP+PP hang when the OLMo-owned backend is enabled. The
original prewarm/padding workaround existed because PyTorch's NVSHMEM allocator effectively needed
unrelated PP stages to keep the same group-`"0"` allocation sequence. Once allocation is scoped to
the EP-local NVSHMEM world, the first allocation can happen lazily inside the PP schedule without
waiting for ranks in other PP stages.

Default behavior:

```text
OLMO_USE_OWN_SYMM_MEM=1:
  lazy allocation
  no dummy padding across PP stages

OLMO_USE_OWN_SYMM_MEM=0:
  legacy PyTorch symm_mem path
  rejected when EP no-sync and PP are both enabled
  no rank-global padding workaround
```

Lazy allocation is preferable for the OLMo backend because it avoids allocating buffers for blocks
or slots that may never be used, removes the need to align layer counts across PP stages, and keeps
the memory policy tied to actual communication. If first-iteration latency needs to be moved out of
the hot path for benchmarking, set:

```bash
export OLMO_OWN_SYMM_PREWARM=1
```

That prewarms only the local model part's actual EP no-sync blocks. It intentionally does not pad
to the maximum block count across unrelated PP stages.

The legacy PyTorch backend no longer attempts the old rank-global prewarm/padding workaround. Padding
all PP stages to the same number of symmetric-memory allocation sites hides the real backend
constraint, wastes memory for uneven model partitions, and is brittle under schedule changes. When
EP no-sync and PP are used together, the correct answer is subgroup-scoped allocation through the
OLMo-owned backend.

## Follow-Ups

- Add an explicit allocation/free lifecycle. The first version intentionally keeps NVSHMEM
  allocations alive for process lifetime to avoid accidental non-collective `nvshmem_free()` from
  tensor garbage collection.
- Port or delete legacy `torch.ops.symm_mem.all_to_all_vdev` usage.
- Add a small distributed smoke test that runs two independent EP groups in the same global job.
- Decide whether the long-term bootstrap universe should be EP-local or PP-stage-local for larger
  topologies.
