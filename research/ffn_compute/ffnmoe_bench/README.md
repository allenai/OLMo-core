# ffnmoe_bench

Speed benches and profilers for the nested-width routed FFN (the training family is `../ffnmoe_train/`).
Code only; outputs (`bench_ffn_routing.json`, `profile_train_step*.json`, kernel tables, logs) are written to
`research/ffn_compute/ffnmoe_bench/` by default and are not in git. The numbers they produced are quoted in
`../ffnmoe_train/README.md`.

- `bench_ffn_routing.py` — standalone layer: does the FLOP saving convert into wall-clock?
- `profile_routed_layer.py` — why the real routed `FeedForward` is slower than the standalone bench.
- `profile_train_step.py` — where a Qwen3-4B training step's time goes, and what FFN routing can save.
- `smoke_router_learns.py` — CPU smoke: does the router give up compute while CE holds?
