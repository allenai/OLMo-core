# kv_route/smoke

GPU smoke tests and benches for the KV-route and block-skip layers, moved from the old `debug/flop_scaling/`.
They put the repo's `src/` on `sys.path` from `__file__`.

- `smoke_kv_route_gpu.py`, `smoke_block_skip_gpu.py`, `smoke_blockmask_gpu.py` — end-to-end GPU correctness smokes.
- `bench_kv_route_layer.py`, `bench_kv_route_mask.py` — layer / mask-build speed.
- `mem_profile_routers.py` — memory profile of the routers.
- `beaker_pytest_launch.py` — runs the GPU pytest set on Beaker.

Context: `../README.md`, `records/kv-route-attention-allocation.md`.
