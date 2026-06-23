# OLMo BF16 TMA/IBGDA EP

This directory is the CUDA-side contract for the OLMo-owned rowwise
TMA/IBGDA transport backend.

It is intentionally separate from:

- the current rowwise NVSHMEM extension;
- the wave / MegaMoE persistent-kernel path;
- external DeepEP or Megatron-LM HybridEP runtime code.

The current contract is still fail-closed, but it now includes the host/CUDA
layout pieces needed by the future transport:

- `RouteRecord` is fixed at 32 bytes;
- route maps use OLMo rowwise `[num_tokens, top_k]` rank/row semantics;
- dropped routes are represented by missing `ROUTE_FLAG_VALID`;
- workspace layout is owned by the new backend;
- each rank has a registered peer window with route records, per-rank counts,
  rank offsets, overflow flags, BF16 payload rows, send doorbells, and receive
  completion counters;
- peer base addresses are represented as a device table of `uint64_t` base
  pointers, ready to be replaced by registered RDMA/IBGDA handles.

CUDA kernels should consume `PeerWindowLayout`/`PeerWindowView`; they should not
fall back to the current NVSHMEM rowwise extension.
