# Rowwise EP No-Sync TBO Notes

This file tracks the first rowwise no-sync EP implementation for two-batch
overlap (TBO). The code lives in `ep_no_sync_tbo_rowwise.py` and is separate
from the existing non-TBO rowwise path and the existing 1D no-sync TBO path.

## Schedule

The implementation splits a rowwise no-sync MoE block into these stages:

- `A`: attention, MoE input prep, routed/shared routers, capacity/drop
  metadata, rowwise route maps, symmetric buffers, and shared expert `forward1`.
- `D`: rowwise dispatch on a comm stream.
- `E`: routed expert grouped GEMM after dispatch completes.
- `C`: rowwise weighted combine on a comm stream.
- `tail`: wait for combine, run shared expert `forward2`, merge routed/shared
  output, apply residual/norm, and attach routed auxiliary loss.

The intended overlap is cross-batch:

- previous pending combine overlaps with current `x0` attention/router;
- `x0` dispatch overlaps with `x1` attention/router;
- `x0` combine overlaps with `x1` routed expert grouped GEMM.

## Current Scope

The initial implementation supports the regular rowwise path only. Rowwise FP8
fails closed for now because it needs separate per-lane symmetric q/scale
buffers before two lanes can be safely in flight.

The main performance knob is `ep_no_sync_rowwise_nblocks`. The rowwise
NVSHMEM kernels are collective launches with stream barriers, so CUDA stream
concurrency is possible, but useful SM overlap with attention/GEMM should be
validated with Nsight and an `nblocks` sweep.
