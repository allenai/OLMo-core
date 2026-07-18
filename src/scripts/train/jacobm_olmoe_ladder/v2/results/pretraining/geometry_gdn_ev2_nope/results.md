# Geometry-matched NoPE active hybrid GDN intervention

Generated: `2026-07-18T02:46:33.286448+00:00`

Selection metric: final `250M`-token mean training CE. Only finished runs are eligible.
The optimal-LR summary includes only bracketed 275M sweeps with a valid quadratic fit.
The all-size figure uses observed-optimal points for 275M and fixed wide-LR transfer points for larger sizes; pending hybrid cells are labeled explicitly.
Fitted LR minima in the 275M U-plot are visual aids and are never used to select results.

## Completed results

| Model | Cx | Mode | Status | References (loss @ LR) | Intervention (loss @ LR) | Deltas vs references |
|---|---:|---|---|---|---:|---|
| 275m | Cx1 | LR sweep | complete | wide_integration: 2.741044 @ 0.0016; hybrid_gdn_ev1: 2.694642 @ 0.0016; geometry_gdn_ev2: 2.707881 @ 0.0016 | 2.712805 (0.0008) | wide_integration: -0.028239; hybrid_gdn_ev1: +0.018164; geometry_gdn_ev2: +0.004924 |
| 275m | Cx2 | LR sweep | provisional (0/4) | wide_integration: 2.608295 @ 0.0016; hybrid_gdn_ev1: 2.569988 @ 0.0016; geometry_gdn_ev2: 2.578963 @ 0.0016 | — | wide_integration: —; hybrid_gdn_ev1: —; geometry_gdn_ev2: — |
| 275m | Cx4 | LR sweep | provisional (0/4) | wide_integration: 2.506019 @ 0.0008; hybrid_gdn_ev1: 2.478165 @ 0.0016; geometry_gdn_ev2: 2.474631 @ 0.0016 | — | wide_integration: —; hybrid_gdn_ev1: —; geometry_gdn_ev2: — |
| 275m | Cx8 | LR sweep | provisional (0/4) | wide_integration: 2.419273 @ 0.0008; hybrid_gdn_ev1: 2.396380 @ 0.0016; geometry_gdn_ev2: 2.389857 @ 0.0008 | — | wide_integration: —; hybrid_gdn_ev1: —; geometry_gdn_ev2: — |

## Runs

| Model | Variant | Cx | LR | State | Tokens (B) | Final-window CE | Replay handling | W&B |
|---|---|---:|---:|---|---:|---:|---|---|
| 275m | geometry-matched hybrid (GDN, expand_v=2) | 1 | 0.0004 | finished | 4.531 | 2.726539 | — | [sa70hegz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sa70hegz) |
| 275m | geometry-matched hybrid (GDN, expand_v=2) | 1 | 0.0008 | finished | 4.531 | 2.709238 | — | [3ddxwqks](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3ddxwqks) |
| 275m | geometry-matched hybrid (GDN, expand_v=2) | 1 | 0.0016 | finished | 4.531 | 2.707881 | — | [8zx9zgnw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8zx9zgnw) |
| 275m | geometry-matched hybrid (GDN, expand_v=2) | 1 | 0.0032 | finished | 4.531 | 2.717324 | — | [terfkng8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/terfkng8) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 1 | 0.0004 | finished | 4.531 | 2.727856 | — | [52ph1l67](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/52ph1l67) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 1 | 0.0008 | finished | 4.531 | 2.712805 | — | [epdjswap](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/epdjswap) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 1 | 0.0016 | finished | 4.531 | 2.713057 | — | [8mnuuecq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8mnuuecq) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 1 | 0.0032 | finished | 4.531 | 2.757006 | — | [7gfls4r6](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7gfls4r6) |
| 275m | hybrid (GDN, expand_v=1) | 1 | 0.0004 | finished | 4.223 | 2.720016 | — | [fkm77yos](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/fkm77yos) |
| 275m | hybrid (GDN, expand_v=1) | 1 | 0.0008 | finished | 4.223 | 2.700254 | — | [yo22u93q](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/yo22u93q) |
| 275m | hybrid (GDN, expand_v=1) | 1 | 0.0016 | finished | 4.223 | 2.694642 | — | [moknw6oc](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/moknw6oc) |
| 275m | hybrid (GDN, expand_v=1) | 1 | 0.0032 | finished | 4.223 | 2.710482 | — | [mettf0d3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mettf0d3) |
| 275m | wide integration (SWA) | 1 | 0.0008 | finished | 4.063 | 2.749142 | — | [kfua3dcq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kfua3dcq) |
| 275m | wide integration (SWA) | 1 | 0.0016 | finished | 4.063 | 2.741044 | — | [h86x1nv3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h86x1nv3) |
| 275m | wide integration (SWA) | 1 | 0.0032 | finished | 4.063 | 2.749132 | — | [afxq80js](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/afxq80js) |
| 275m | geometry-matched hybrid (GDN, expand_v=2) | 2 | 0.0004 | finished | 9.062 | 2.601221 | — | [oaazdm2h](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/oaazdm2h) |
| 275m | geometry-matched hybrid (GDN, expand_v=2) | 2 | 0.0008 | finished | 9.062 | 2.582057 | — | [u4cinuz5](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/u4cinuz5) |
| 275m | geometry-matched hybrid (GDN, expand_v=2) | 2 | 0.0016 | finished | 9.062 | 2.578963 | — | [3oqkg24h](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3oqkg24h) |
| 275m | geometry-matched hybrid (GDN, expand_v=2) | 2 | 0.0032 | finished | 9.062 | 2.595069 | — | [pz6377bu](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/pz6377bu) |
| 275m | hybrid (GDN, expand_v=1) | 2 | 0.0004 | finished | 8.445 | 2.593894 | — | [s5qmhyb2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/s5qmhyb2) |
| 275m | hybrid (GDN, expand_v=1) | 2 | 0.0008 | finished | 8.445 | 2.577620 | — | [07qo96gy](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/07qo96gy) |
| 275m | hybrid (GDN, expand_v=1) | 2 | 0.0016 | finished | 8.445 | 2.569988 | — | [j12fk559](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/j12fk559) |
| 275m | hybrid (GDN, expand_v=1) | 2 | 0.0032 | finished | 8.445 | 2.585829 | — | [mem73c7g](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mem73c7g) |
| 275m | wide integration (SWA) | 2 | 0.0008 | finished | 8.126 | 2.612348 | — | [o2bdr3gw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/o2bdr3gw) |
| 275m | wide integration (SWA) | 2 | 0.0016 | finished | 8.126 | 2.608295 | — | [6porpbo2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6porpbo2) |
| 275m | wide integration (SWA) | 2 | 0.0032 | finished | 8.126 | 2.614495 | — | [0f782vrw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0f782vrw) |
| 275m | geometry-matched hybrid (GDN, expand_v=2) | 4 | 0.0004 | finished | 18.125 | 2.492137 | — | [7jzlrolc](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7jzlrolc) |
| 275m | geometry-matched hybrid (GDN, expand_v=2) | 4 | 0.0008 | finished | 18.125 | 2.474632 | — | [gwve4pn6](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/gwve4pn6) |
| 275m | geometry-matched hybrid (GDN, expand_v=2) | 4 | 0.0016 | finished | 18.125 | 2.474631 | — | [hwjvw532](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/hwjvw532) |
| 275m | geometry-matched hybrid (GDN, expand_v=2) | 4 | 0.0032 | finished | 18.125 | 2.496382 | — | [hmjkig0r](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/hmjkig0r) |
| 275m | hybrid (GDN, expand_v=1) | 4 | 0.0004 | finished | 16.890 | 2.496429 | — | [socvue3a](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/socvue3a) |
| 275m | hybrid (GDN, expand_v=1) | 4 | 0.0008 | finished | 16.890 | 2.479545 | — | [xvk92054](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xvk92054) |
| 275m | hybrid (GDN, expand_v=1) | 4 | 0.0016 | finished | 16.890 | 2.478165 | — | [uhw9wfed](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/uhw9wfed) |
| 275m | hybrid (GDN, expand_v=1) | 4 | 0.0032 | finished | 16.890 | 2.496406 | — | [sr1jgmao](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sr1jgmao) |
| 275m | wide integration (SWA) | 4 | 0.0004 | finished | 16.251 | 2.520624 | — | [n1gjknwg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/n1gjknwg) |
| 275m | wide integration (SWA) | 4 | 0.0008 | finished | 16.251 | 2.506019 | — | [9n3xk8gs](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9n3xk8gs) |
| 275m | wide integration (SWA) | 4 | 0.0016 | finished | 16.251 | 2.508140 | — | [ttjquo05](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ttjquo05) |
| 275m | wide integration (SWA) | 4 | 0.0032 | finished | 16.251 | 2.522459 | — | [5u03fshf](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5u03fshf) |
| 275m | geometry-matched hybrid (GDN, expand_v=2) | 8 | 0.0004 | finished | 36.250 | 2.404090 | 1 reset(s); 498 duplicate token sample(s) removed | [7mlzc5x4](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7mlzc5x4) / [9k8mo2q5](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9k8mo2q5) |
| 275m | geometry-matched hybrid (GDN, expand_v=2) | 8 | 0.0008 | finished | 36.250 | 2.389857 | 1 reset(s); 92 duplicate token sample(s) removed | [wo8raj1p](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/wo8raj1p) / [xdo7p86h](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xdo7p86h) |
| 275m | geometry-matched hybrid (GDN, expand_v=2) | 8 | 0.0016 | finished | 36.250 | 2.390902 | — | [0x3i869n](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0x3i869n) |
| 275m | geometry-matched hybrid (GDN, expand_v=2) | 8 | 0.0032 | finished | 36.250 | 2.412928 | — | [aholwcgr](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/aholwcgr) |
| 275m | hybrid (GDN, expand_v=1) | 8 | 0.0004 | finished | 33.780 | 2.412233 | — | [b0z3qfmi](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/b0z3qfmi) |
| 275m | hybrid (GDN, expand_v=1) | 8 | 0.0008 | finished | 33.780 | 2.397966 | — | [rkxojd03](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rkxojd03) |
| 275m | hybrid (GDN, expand_v=1) | 8 | 0.0016 | finished | 33.780 | 2.396380 | — | [66aja50m](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/66aja50m) |
| 275m | hybrid (GDN, expand_v=1) | 8 | 0.0032 | finished | 33.780 | 2.414943 | — | [ntoo8vlo](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ntoo8vlo) |
| 275m | wide integration (SWA) | 8 | 0.0004 | finished | 32.502 | 2.435915 | — | [iv901lom](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/iv901lom) |
| 275m | wide integration (SWA) | 8 | 0.0008 | finished | 32.502 | 2.419273 | — | [qe052lo4](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qe052lo4) |
| 275m | wide integration (SWA) | 8 | 0.0016 | finished | 32.502 | 2.423303 | — | [qu2zaxr7](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qu2zaxr7) |
| 275m | wide integration (SWA) | 8 | 0.0032 | finished | 32.502 | 2.441430 | — | [235ye5lg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/235ye5lg) |
