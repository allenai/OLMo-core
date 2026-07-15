# Active hybrid GDN intervention

Generated: `2026-07-15T20:57:58.231858+00:00`

Selection metric: final `250M`-token mean training CE. Only finished runs are eligible.
The optimal-LR summary includes only bracketed 275M sweeps with a valid quadratic fit.
The larger-size figure is a fixed-LR transfer comparison at the wide-optimal LR; it does not claim that LR is hybrid-optimal.
Fitted LR minima in the 275M U-plot are visual aids and are never used to select results.

## Completed results

| Model | Cx | Mode | Status | Wide reference (LR) | Hybrid result (LR) | Delta |
|---|---:|---|---|---:|---:|---:|
| 275m | Cx1 | LR sweep | complete | 2.741044 (0.0016) | 2.694642 (0.0016) | -0.046402 |
| 275m | Cx2 | LR sweep | complete | 2.608295 (0.0016) | 2.569988 (0.0016) | -0.038307 |
| 275m | Cx4 | LR sweep | complete | 2.506019 (0.0008) | 2.478165 (0.0016) | -0.027853 |
| 275m | Cx8 | LR sweep | provisional (3/4) | 2.419273 (0.0008) | 2.396380 (0.0016) | -0.022893 |
| 480m | Cx1 | fixed-LR transfer | finished | 2.543281 (0.0012) | 2.510874 (0.0012) | -0.032407 |
| 480m | Cx2 | fixed-LR transfer | finished | 2.423888 (0.0009) | 2.412790 (0.0009) | -0.011097 |
| 810m | Cx1 | fixed-LR transfer | finished | 2.373197 (0.0006) | 2.364345 (0.0006) | -0.008852 |
| 810m | Cx2 | fixed-LR transfer | pending | 2.268948 (0.00056) | — | — |
| 1p2b | Cx1 | fixed-LR transfer | pending | 2.273062 (0.0004) | — | — |
| 1p2b | Cx2 | fixed-LR transfer | pending | 2.178332 (0.0006) | — | — |

## Runs

| Model | Variant | Cx | LR | State | Tokens (B) | Final-window CE | W&B |
|---|---|---:|---:|---|---:|---:|---|
| 275m | hybrid (GDN, expand_v=1) | 1 | 0.0004 | finished | 4.223 | 2.720016 | [fkm77yos](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/fkm77yos) |
| 275m | hybrid (GDN, expand_v=1) | 1 | 0.0008 | finished | 4.223 | 2.700254 | [yo22u93q](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/yo22u93q) |
| 275m | hybrid (GDN, expand_v=1) | 1 | 0.0016 | finished | 4.223 | 2.694642 | [moknw6oc](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/moknw6oc) |
| 275m | hybrid (GDN, expand_v=1) | 1 | 0.0032 | finished | 4.223 | 2.710482 | [mettf0d3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mettf0d3) |
| 275m | wide integration (SWA) | 1 | 0.0008 | finished | 4.063 | 2.749142 | [kfua3dcq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kfua3dcq) |
| 275m | wide integration (SWA) | 1 | 0.0016 | finished | 4.063 | 2.741044 | [h86x1nv3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h86x1nv3) |
| 275m | wide integration (SWA) | 1 | 0.0032 | finished | 4.063 | 2.749132 | [afxq80js](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/afxq80js) |
| 275m | hybrid (GDN, expand_v=1) | 2 | 0.0004 | finished | 8.445 | 2.593894 | [s5qmhyb2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/s5qmhyb2) |
| 275m | hybrid (GDN, expand_v=1) | 2 | 0.0008 | finished | 8.445 | 2.577620 | [07qo96gy](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/07qo96gy) |
| 275m | hybrid (GDN, expand_v=1) | 2 | 0.0016 | finished | 8.445 | 2.569988 | [j12fk559](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/j12fk559) |
| 275m | hybrid (GDN, expand_v=1) | 2 | 0.0032 | finished | 8.445 | 2.585829 | [mem73c7g](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mem73c7g) |
| 275m | wide integration (SWA) | 2 | 0.0008 | finished | 8.126 | 2.612348 | [o2bdr3gw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/o2bdr3gw) |
| 275m | wide integration (SWA) | 2 | 0.0016 | finished | 8.126 | 2.608295 | [6porpbo2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6porpbo2) |
| 275m | wide integration (SWA) | 2 | 0.0032 | finished | 8.126 | 2.614495 | [0f782vrw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0f782vrw) |
| 275m | hybrid (GDN, expand_v=1) | 4 | 0.0004 | finished | 16.890 | 2.496429 | [socvue3a](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/socvue3a) |
| 275m | hybrid (GDN, expand_v=1) | 4 | 0.0008 | finished | 16.890 | 2.479545 | [xvk92054](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xvk92054) |
| 275m | hybrid (GDN, expand_v=1) | 4 | 0.0016 | finished | 16.890 | 2.478165 | [uhw9wfed](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/uhw9wfed) |
| 275m | hybrid (GDN, expand_v=1) | 4 | 0.0032 | finished | 16.890 | 2.496406 | [sr1jgmao](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sr1jgmao) |
| 275m | wide integration (SWA) | 4 | 0.0004 | finished | 16.251 | 2.520624 | [n1gjknwg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/n1gjknwg) |
| 275m | wide integration (SWA) | 4 | 0.0008 | finished | 16.251 | 2.506019 | [9n3xk8gs](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9n3xk8gs) |
| 275m | wide integration (SWA) | 4 | 0.0016 | finished | 16.251 | 2.508140 | [ttjquo05](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ttjquo05) |
| 275m | wide integration (SWA) | 4 | 0.0032 | finished | 16.251 | 2.522459 | [5u03fshf](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5u03fshf) |
| 275m | hybrid (GDN, expand_v=1) | 8 | 0.0004 | finished | 33.780 | 2.412233 | [b0z3qfmi](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/b0z3qfmi) |
| 275m | hybrid (GDN, expand_v=1) | 8 | 0.0008 | finished | 33.780 | 2.397966 | [rkxojd03](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rkxojd03) |
| 275m | hybrid (GDN, expand_v=1) | 8 | 0.0016 | finished | 33.780 | 2.396380 | [66aja50m](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/66aja50m) |
| 275m | wide integration (SWA) | 8 | 0.0004 | finished | 32.502 | 2.435915 | [iv901lom](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/iv901lom) |
| 275m | wide integration (SWA) | 8 | 0.0008 | finished | 32.502 | 2.419273 | [qe052lo4](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qe052lo4) |
| 275m | wide integration (SWA) | 8 | 0.0016 | finished | 32.502 | 2.423303 | [qu2zaxr7](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qu2zaxr7) |
| 275m | wide integration (SWA) | 8 | 0.0032 | finished | 32.502 | 2.441430 | [235ye5lg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/235ye5lg) |
| 480m | hybrid (GDN, expand_v=1) | 1 | 0.0012 | finished | 7.969 | 2.510874 | [wl8ebsd8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/wl8ebsd8) |
| 480m | wide integration (SWA) | 1 | 0.0012 | finished | 7.672 | 2.543281 | [z4wxvc6h](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/z4wxvc6h) |
| 480m | hybrid (GDN, expand_v=1) | 2 | 0.0009 | finished | 15.939 | 2.412790 | [4vzmrld1](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4vzmrld1) |
| 480m | wide integration (SWA) | 2 | 0.0009 | finished | 15.344 | 2.423888 | [ywj13bkw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ywj13bkw) |
| 810m | hybrid (GDN, expand_v=1) | 1 | 0.0006 | finished | 14.619 | 2.364345 | [h1rmcm2p](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h1rmcm2p) |
| 810m | wide integration (SWA) | 1 | 0.0006 | finished | 13.903 | 2.373197 | [w912irkq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/w912irkq) |
| 810m | wide integration (SWA) | 2 | 0.00056 | finished | 27.805 | 2.268948 | [jpbqhfvc](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jpbqhfvc) |
| 1p2b | wide integration (SWA) | 1 | 0.0004 | finished | 21.417 | 2.273062 | [hww8eksq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/hww8eksq) |
| 1p2b | wide integration (SWA) | 2 | 0.0006 | finished | 42.835 | 2.178332 | [jfwntmwm](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jfwntmwm) |
