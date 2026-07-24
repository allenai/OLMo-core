# Geometry-matched NoPE gated-attention GDN2 intervention

Generated: `2026-07-24T06:18:18.127963+00:00`

Selection metric: final `250M`-token mean training CE. Only finished runs are eligible.
The optimal-LR summary includes only bracketed 275M sweeps with a valid quadratic fit.
The all-size figure uses observed-optimal points for 275M and fixed wide-LR transfer points for larger sizes; pending hybrid cells are labeled explicitly.
Fitted LR minima in the 275M U-plot are visual aids and are never used to select results.

## Completed results

| Model | Cx | Mode | Status | References (loss @ LR) | Intervention (loss @ LR) | Deltas vs references |
|---|---:|---|---|---|---:|---|
| 275m | Cx1 | LR sweep | complete | wide_integration: 2.741044 @ 0.0016; geometry_gdn_ev2_rope_gated: 2.691980 @ 0.0016 | 2.646730 (0.0016) | wide_integration: -0.094314; geometry_gdn_ev2_rope_gated: -0.045251 |
| 275m | Cx2 | LR sweep | provisional (3/4) | wide_integration: 2.608295 @ 0.0016; geometry_gdn_ev2_rope_gated: 2.573449 @ 0.0016 | 2.544136 (0.0008) | wide_integration: -0.064159; geometry_gdn_ev2_rope_gated: -0.029313 |
| 275m | Cx4 | LR sweep | provisional (0/4) | wide_integration: 2.506019 @ 0.0008; geometry_gdn_ev2_rope_gated: 2.470110 @ 0.0008 | — | wide_integration: —; geometry_gdn_ev2_rope_gated: — |
| 275m | Cx8 | LR sweep | provisional (0/4) | wide_integration: 2.419273 @ 0.0008; geometry_gdn_ev2_rope_gated: 2.386206 @ 0.0016 | — | wide_integration: —; geometry_gdn_ev2_rope_gated: — |

## Runs

| Model | Variant | Cx | LR | State | Tokens (B) | Final-window CE | Replay handling | W&B |
|---|---|---:|---:|---|---:|---:|---|---|
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 1 | 0.0004 | finished | 4.839 | 2.676958 | 0 reset(s); 1 duplicate token sample(s) removed | [rsxmn720](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rsxmn720) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 1 | 0.0008 | finished | 4.839 | 2.657052 | 0 reset(s); 1 duplicate token sample(s) removed | [pqrdvu63](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/pqrdvu63) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 1 | 0.0016 | finished | 4.839 | 2.646730 | 0 reset(s); 1 duplicate token sample(s) removed | [5uzr9dva](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5uzr9dva) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 1 | 0.0032 | finished | 4.839 | 2.661486 | 0 reset(s); 1 duplicate token sample(s) removed | [j2t5c2jb](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/j2t5c2jb) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, RoPE, gated attention) | 1 | 0.0004 | finished | 4.557 | 2.715839 | 0 reset(s); 1 duplicate token sample(s) removed | [kd3fyszi](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kd3fyszi) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, RoPE, gated attention) | 1 | 0.0008 | finished | 4.557 | 2.697808 | 0 reset(s); 1 duplicate token sample(s) removed | [ezdsfb9n](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ezdsfb9n) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, RoPE, gated attention) | 1 | 0.0016 | finished | 4.557 | 2.691980 | 0 reset(s); 1 duplicate token sample(s) removed | [eo5bm8gw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/eo5bm8gw) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, RoPE, gated attention) | 1 | 0.0032 | finished | 4.557 | 2.703272 | 0 reset(s); 1 duplicate token sample(s) removed | [l4tp6qmo](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/l4tp6qmo) |
| 275m | wide integration (SWA) | 1 | 0.0008 | finished | 4.063 | 2.749142 | 0 reset(s); 1 duplicate token sample(s) removed | [kfua3dcq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kfua3dcq) |
| 275m | wide integration (SWA) | 1 | 0.0016 | finished | 4.063 | 2.741044 | 0 reset(s); 1 duplicate token sample(s) removed | [h86x1nv3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h86x1nv3) |
| 275m | wide integration (SWA) | 1 | 0.0032 | finished | 4.063 | 2.749132 | 0 reset(s); 1 duplicate token sample(s) removed | [afxq80js](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/afxq80js) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 2 | 0.0004 | finished | 9.679 | 2.564391 | 0 reset(s); 1 duplicate token sample(s) removed | [7yh4rfi1](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7yh4rfi1) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 2 | 0.0008 | finished | 9.679 | 2.544136 | 0 reset(s); 1 duplicate token sample(s) removed | [2egeqyvo](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/2egeqyvo) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 2 | 0.0032 | finished | 9.679 | 2.548521 | 0 reset(s); 1 duplicate token sample(s) removed | [xwtxd1pv](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xwtxd1pv) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, RoPE, gated attention) | 2 | 0.0004 | finished | 9.115 | 2.597840 | 0 reset(s); 1 duplicate token sample(s) removed | [7gmi969q](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7gmi969q) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, RoPE, gated attention) | 2 | 0.0008 | finished | 9.115 | 2.579025 | 0 reset(s); 1 duplicate token sample(s) removed | [0ovig11c](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0ovig11c) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, RoPE, gated attention) | 2 | 0.0016 | finished | 9.115 | 2.573449 | 0 reset(s); 1 duplicate token sample(s) removed | [8mkt4xpz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8mkt4xpz) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, RoPE, gated attention) | 2 | 0.0032 | finished | 9.115 | 2.589305 | 0 reset(s); 1 duplicate token sample(s) removed | [66u6ekx2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/66u6ekx2) |
| 275m | wide integration (SWA) | 2 | 0.0008 | finished | 8.126 | 2.612348 | 0 reset(s); 1 duplicate token sample(s) removed | [o2bdr3gw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/o2bdr3gw) |
| 275m | wide integration (SWA) | 2 | 0.0016 | finished | 8.126 | 2.608295 | 0 reset(s); 1 duplicate token sample(s) removed | [6porpbo2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6porpbo2) |
| 275m | wide integration (SWA) | 2 | 0.0032 | finished | 8.126 | 2.614495 | 0 reset(s); 1 duplicate token sample(s) removed | [0f782vrw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0f782vrw) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, RoPE, gated attention) | 4 | 0.0004 | finished | 18.229 | 2.486140 | 0 reset(s); 1 duplicate token sample(s) removed | [o1p6n2v7](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/o1p6n2v7) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, RoPE, gated attention) | 4 | 0.0008 | finished | 18.229 | 2.470110 | 0 reset(s); 1 duplicate token sample(s) removed | [iqxc5n9x](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/iqxc5n9x) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, RoPE, gated attention) | 4 | 0.0016 | finished | 18.229 | 2.470261 | 0 reset(s); 1 duplicate token sample(s) removed | [n6suaxul](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/n6suaxul) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, RoPE, gated attention) | 4 | 0.0032 | finished | 18.229 | 2.490518 | 0 reset(s); 1 duplicate token sample(s) removed | [clfmsyx8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/clfmsyx8) |
| 275m | wide integration (SWA) | 4 | 0.0004 | finished | 16.251 | 2.520624 | 0 reset(s); 1 duplicate token sample(s) removed | [n1gjknwg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/n1gjknwg) |
| 275m | wide integration (SWA) | 4 | 0.0008 | finished | 16.251 | 2.506019 | 0 reset(s); 1 duplicate token sample(s) removed | [9n3xk8gs](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9n3xk8gs) |
| 275m | wide integration (SWA) | 4 | 0.0016 | finished | 16.251 | 2.508140 | 0 reset(s); 1 duplicate token sample(s) removed | [ttjquo05](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ttjquo05) |
| 275m | wide integration (SWA) | 4 | 0.0032 | finished | 16.251 | 2.522459 | 0 reset(s); 1 duplicate token sample(s) removed | [5u03fshf](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5u03fshf) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, RoPE, gated attention) | 8 | 0.0004 | finished | 36.459 | 2.400511 | 0 reset(s); 1 duplicate token sample(s) removed | [y1dh1cb5](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/y1dh1cb5) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, RoPE, gated attention) | 8 | 0.0008 | finished | 36.459 | 2.387352 | 0 reset(s); 1 duplicate token sample(s) removed | [65bsc0wk](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/65bsc0wk) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, RoPE, gated attention) | 8 | 0.0016 | finished | 36.459 | 2.386206 | 0 reset(s); 1 duplicate token sample(s) removed | [8rgf3myq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8rgf3myq) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, RoPE, gated attention) | 8 | 0.0032 | finished | 36.459 | 2.408689 | 0 reset(s); 1 duplicate token sample(s) removed | [klgge8er](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/klgge8er) |
| 275m | wide integration (SWA) | 8 | 0.0004 | finished | 32.502 | 2.435915 | 0 reset(s); 1 duplicate token sample(s) removed | [iv901lom](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/iv901lom) |
| 275m | wide integration (SWA) | 8 | 0.0008 | finished | 32.502 | 2.419273 | 0 reset(s); 1 duplicate token sample(s) removed | [qe052lo4](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qe052lo4) |
| 275m | wide integration (SWA) | 8 | 0.0016 | finished | 32.502 | 2.423303 | 0 reset(s); 1 duplicate token sample(s) removed | [qu2zaxr7](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qu2zaxr7) |
| 275m | wide integration (SWA) | 8 | 0.0032 | finished | 32.502 | 2.441430 | 0 reset(s); 1 duplicate token sample(s) removed | [235ye5lg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/235ye5lg) |
