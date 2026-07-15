# 275M active hybrid intervention

Generated: `2026-07-15T17:02:15.070915+00:00`

Selection metric: final `250M`-token mean training CE. Only finished runs are eligible.
The summary includes a Cx only when its observed best is bracketed and supports a valid quadratic fit.
Fitted LR minima in the U-plot are visual aids and are never used to select results.

## Observed best

| Cx | Status | Wide loss (LR) | Intervention loss (LR) | Delta |
|---:|---|---:|---:|---:|
| Cx1 | complete | 2.741044 (0.0016) | 2.694642 (0.0016) | -0.046402 |
| Cx2 | complete | 2.608295 (0.0016) | 2.569988 (0.0016) | -0.038307 |
| Cx4 | complete | 2.506019 (0.0008) | 2.478165 (0.0016) | -0.027853 |
| Cx8 | provisional (3/4) | 2.419273 (0.0008) | 2.396380 (0.0016) | -0.022893 |

## Runs

| Variant | Cx | LR | State | Tokens (B) | Final-window CE | W&B |
|---|---:|---:|---|---:|---:|---|
| hybrid (GDN, expand_v=1) | 1 | 0.0004 | finished | 4.223 | 2.720016 | [fkm77yos](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/fkm77yos) |
| hybrid (GDN, expand_v=1) | 1 | 0.0008 | finished | 4.223 | 2.700254 | [yo22u93q](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/yo22u93q) |
| hybrid (GDN, expand_v=1) | 1 | 0.0016 | finished | 4.223 | 2.694642 | [moknw6oc](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/moknw6oc) |
| hybrid (GDN, expand_v=1) | 1 | 0.0032 | finished | 4.223 | 2.710482 | [mettf0d3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mettf0d3) |
| wide integration (SWA) | 1 | 0.0008 | finished | 4.063 | 2.749142 | [kfua3dcq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kfua3dcq) |
| wide integration (SWA) | 1 | 0.0016 | finished | 4.063 | 2.741044 | [h86x1nv3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h86x1nv3) |
| wide integration (SWA) | 1 | 0.0032 | finished | 4.063 | 2.749132 | [afxq80js](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/afxq80js) |
| hybrid (GDN, expand_v=1) | 2 | 0.0004 | finished | 8.445 | 2.593894 | [s5qmhyb2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/s5qmhyb2) |
| hybrid (GDN, expand_v=1) | 2 | 0.0008 | finished | 8.445 | 2.577620 | [07qo96gy](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/07qo96gy) |
| hybrid (GDN, expand_v=1) | 2 | 0.0016 | finished | 8.445 | 2.569988 | [j12fk559](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/j12fk559) |
| hybrid (GDN, expand_v=1) | 2 | 0.0032 | finished | 8.445 | 2.585829 | [mem73c7g](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mem73c7g) |
| wide integration (SWA) | 2 | 0.0008 | finished | 8.126 | 2.612348 | [o2bdr3gw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/o2bdr3gw) |
| wide integration (SWA) | 2 | 0.0016 | finished | 8.126 | 2.608295 | [6porpbo2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6porpbo2) |
| wide integration (SWA) | 2 | 0.0032 | finished | 8.126 | 2.614495 | [0f782vrw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0f782vrw) |
| hybrid (GDN, expand_v=1) | 4 | 0.0004 | finished | 16.890 | 2.496429 | [socvue3a](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/socvue3a) |
| hybrid (GDN, expand_v=1) | 4 | 0.0008 | finished | 16.890 | 2.479545 | [xvk92054](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xvk92054) |
| hybrid (GDN, expand_v=1) | 4 | 0.0016 | finished | 16.890 | 2.478165 | [uhw9wfed](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/uhw9wfed) |
| hybrid (GDN, expand_v=1) | 4 | 0.0032 | finished | 16.890 | 2.496406 | [sr1jgmao](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sr1jgmao) |
| wide integration (SWA) | 4 | 0.0004 | finished | 16.251 | 2.520624 | [n1gjknwg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/n1gjknwg) |
| wide integration (SWA) | 4 | 0.0008 | finished | 16.251 | 2.506019 | [9n3xk8gs](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9n3xk8gs) |
| wide integration (SWA) | 4 | 0.0016 | finished | 16.251 | 2.508140 | [ttjquo05](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ttjquo05) |
| wide integration (SWA) | 4 | 0.0032 | finished | 16.251 | 2.522459 | [5u03fshf](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5u03fshf) |
| hybrid (GDN, expand_v=1) | 8 | 0.0004 | finished | 33.780 | 2.412233 | [b0z3qfmi](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/b0z3qfmi) |
| hybrid (GDN, expand_v=1) | 8 | 0.0008 | finished | 33.780 | 2.397966 | [rkxojd03](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rkxojd03) |
| hybrid (GDN, expand_v=1) | 8 | 0.0016 | finished | 33.780 | 2.396380 | [66aja50m](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/66aja50m) |
| wide integration (SWA) | 8 | 0.0004 | finished | 32.502 | 2.435915 | [iv901lom](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/iv901lom) |
| wide integration (SWA) | 8 | 0.0008 | finished | 32.502 | 2.419273 | [qe052lo4](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qe052lo4) |
| wide integration (SWA) | 8 | 0.0016 | finished | 32.502 | 2.423303 | [qu2zaxr7](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qu2zaxr7) |
| wide integration (SWA) | 8 | 0.0032 | finished | 32.502 | 2.441430 | [235ye5lg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/235ye5lg) |
