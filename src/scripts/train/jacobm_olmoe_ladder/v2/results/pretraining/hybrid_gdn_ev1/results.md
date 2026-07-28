# Active hybrid GDN intervention

Generated: `2026-07-28T16:25:38.789024+00:00`

Selection metric: final `250M`-token mean training CE. Only finished runs are eligible.
The optimal-LR summary includes only bracketed 275M sweeps with a valid quadratic fit.
The all-size figure uses observed-optimal points for 275M and fixed wide-LR transfer points for larger sizes; pending hybrid cells are labeled explicitly.
Fitted LR minima in the 275M U-plot are visual aids and are never used to select results.

## Completed results

| Model | Cx | Mode | Status | References (loss @ LR) | Intervention (loss @ LR) | Deltas vs references |
|---|---:|---|---|---|---:|---|
| 275m | Cx1 | LR sweep | complete | wide_integration: 2.741044 @ 0.0016 | 2.694642 (0.0016) | wide_integration: -0.046402 |
| 275m | Cx2 | LR sweep | complete | wide_integration: 2.608295 @ 0.0016 | 2.569988 (0.0016) | wide_integration: -0.038307 |
| 275m | Cx4 | LR sweep | complete | wide_integration: 2.506019 @ 0.0008 | 2.478165 (0.0016) | wide_integration: -0.027853 |
| 275m | Cx8 | LR sweep | complete | wide_integration: 2.419273 @ 0.0008 | 2.396380 (0.0016) | wide_integration: -0.022893 |
| 480m | Cx1 | fixed-LR transfer | finished | wide_integration: 2.543281 @ 0.0012 | 2.510874 (0.0012) | wide_integration: -0.032407 |
| 480m | Cx2 | fixed-LR transfer | finished | wide_integration: 2.423888 @ 0.0009 | 2.412790 (0.0009) | wide_integration: -0.011097 |
| 480m | Cx4 | fixed-LR transfer | finished | wide_integration: 2.329976 @ 0.0008 | 2.305996 (0.0008) | wide_integration: -0.023980 |
| 480m | Cx8 | fixed-LR transfer | finished | wide_integration: 2.251305 @ 0.0008 | 2.236205 (0.0008) | wide_integration: -0.015100 |
| 810m | Cx1 | fixed-LR transfer | finished | wide_integration: 2.373197 @ 0.0006 | 2.364345 (0.0006) | wide_integration: -0.008852 |
| 810m | Cx2 | fixed-LR transfer | finished | wide_integration: 2.268948 @ 0.00056 | 2.247185 (0.00056) | wide_integration: -0.021762 |
| 810m | Cx4 | fixed-LR transfer | finished | wide_integration: 2.192802 @ 0.0004 | 2.160440 (0.0004) | wide_integration: -0.032362 |
| 810m | Cx8 | fixed-LR transfer | finished | wide_integration: 2.104939 @ 0.0004 | 2.095585 (0.0004) | wide_integration: -0.009354 |
| 1p2b | Cx1 | fixed-LR transfer | finished | wide_integration: 2.273062 @ 0.0004 | 2.253953 (0.0004) | wide_integration: -0.019109 |
| 1p2b | Cx2 | fixed-LR transfer | finished | wide_integration: 2.178332 @ 0.0006 | 2.163788 (0.0006) | wide_integration: -0.014545 |
| 1p2b | Cx4 | fixed-LR transfer | finished | wide_integration: 2.094219 @ 0.0003 | 2.081180 (0.0003) | wide_integration: -0.013039 |
| 1p2b | Cx8 | fixed-LR transfer | finished | wide_integration: 2.022641 @ 0.0004 | 2.016369 (0.0004) | wide_integration: -0.006272 |

## Runs

| Model | Variant | Cx | LR | State | Tokens (B) | Final-window CE | Replay handling | W&B |
|---|---|---:|---:|---|---:|---:|---|---|
| 275m | hybrid (GDN, expand_v=1) | 1 | 0.0004 | finished | 4.223 | 2.720016 | 0 reset(s); 1 duplicate token sample(s) removed | [fkm77yos](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/fkm77yos) |
| 275m | hybrid (GDN, expand_v=1) | 1 | 0.0008 | finished | 4.223 | 2.700254 | 0 reset(s); 1 duplicate token sample(s) removed | [yo22u93q](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/yo22u93q) |
| 275m | hybrid (GDN, expand_v=1) | 1 | 0.0016 | finished | 4.223 | 2.694642 | 0 reset(s); 1 duplicate token sample(s) removed | [moknw6oc](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/moknw6oc) |
| 275m | hybrid (GDN, expand_v=1) | 1 | 0.0032 | finished | 4.223 | 2.710482 | 0 reset(s); 1 duplicate token sample(s) removed | [mettf0d3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mettf0d3) |
| 275m | wide integration (SWA) | 1 | 0.0008 | finished | 4.063 | 2.749142 | 0 reset(s); 1 duplicate token sample(s) removed | [kfua3dcq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kfua3dcq) |
| 275m | wide integration (SWA) | 1 | 0.0016 | finished | 4.063 | 2.741044 | 0 reset(s); 1 duplicate token sample(s) removed | [h86x1nv3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h86x1nv3) |
| 275m | wide integration (SWA) | 1 | 0.0032 | finished | 4.063 | 2.749132 | 0 reset(s); 1 duplicate token sample(s) removed | [afxq80js](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/afxq80js) |
| 275m | hybrid (GDN, expand_v=1) | 2 | 0.0004 | finished | 8.445 | 2.593894 | 0 reset(s); 1 duplicate token sample(s) removed | [s5qmhyb2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/s5qmhyb2) |
| 275m | hybrid (GDN, expand_v=1) | 2 | 0.0008 | finished | 8.445 | 2.577620 | 0 reset(s); 1 duplicate token sample(s) removed | [07qo96gy](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/07qo96gy) |
| 275m | hybrid (GDN, expand_v=1) | 2 | 0.0016 | finished | 8.445 | 2.569988 | 0 reset(s); 1 duplicate token sample(s) removed | [j12fk559](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/j12fk559) |
| 275m | hybrid (GDN, expand_v=1) | 2 | 0.0032 | finished | 8.445 | 2.585829 | 0 reset(s); 1 duplicate token sample(s) removed | [mem73c7g](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mem73c7g) |
| 275m | wide integration (SWA) | 2 | 0.0008 | finished | 8.126 | 2.612348 | 0 reset(s); 1 duplicate token sample(s) removed | [o2bdr3gw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/o2bdr3gw) |
| 275m | wide integration (SWA) | 2 | 0.0016 | finished | 8.126 | 2.608295 | 0 reset(s); 1 duplicate token sample(s) removed | [6porpbo2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6porpbo2) |
| 275m | wide integration (SWA) | 2 | 0.0032 | finished | 8.126 | 2.614495 | 0 reset(s); 1 duplicate token sample(s) removed | [0f782vrw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0f782vrw) |
| 275m | hybrid (GDN, expand_v=1) | 4 | 0.0004 | finished | 16.890 | 2.496429 | 0 reset(s); 1 duplicate token sample(s) removed | [socvue3a](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/socvue3a) |
| 275m | hybrid (GDN, expand_v=1) | 4 | 0.0008 | finished | 16.890 | 2.479545 | 0 reset(s); 1 duplicate token sample(s) removed | [xvk92054](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xvk92054) |
| 275m | hybrid (GDN, expand_v=1) | 4 | 0.0016 | finished | 16.890 | 2.478165 | 0 reset(s); 1 duplicate token sample(s) removed | [uhw9wfed](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/uhw9wfed) |
| 275m | hybrid (GDN, expand_v=1) | 4 | 0.0032 | finished | 16.890 | 2.496406 | 0 reset(s); 1 duplicate token sample(s) removed | [sr1jgmao](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sr1jgmao) |
| 275m | wide integration (SWA) | 4 | 0.0004 | finished | 16.251 | 2.520624 | 0 reset(s); 1 duplicate token sample(s) removed | [n1gjknwg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/n1gjknwg) |
| 275m | wide integration (SWA) | 4 | 0.0008 | finished | 16.251 | 2.506019 | 0 reset(s); 1 duplicate token sample(s) removed | [9n3xk8gs](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9n3xk8gs) |
| 275m | wide integration (SWA) | 4 | 0.0016 | finished | 16.251 | 2.508140 | 0 reset(s); 1 duplicate token sample(s) removed | [ttjquo05](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ttjquo05) |
| 275m | wide integration (SWA) | 4 | 0.0032 | finished | 16.251 | 2.522459 | 0 reset(s); 1 duplicate token sample(s) removed | [5u03fshf](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5u03fshf) |
| 275m | hybrid (GDN, expand_v=1) | 8 | 0.0004 | finished | 33.780 | 2.412233 | 0 reset(s); 1 duplicate token sample(s) removed | [b0z3qfmi](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/b0z3qfmi) |
| 275m | hybrid (GDN, expand_v=1) | 8 | 0.0008 | finished | 33.780 | 2.397966 | 0 reset(s); 1 duplicate token sample(s) removed | [rkxojd03](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rkxojd03) |
| 275m | hybrid (GDN, expand_v=1) | 8 | 0.0016 | finished | 33.780 | 2.396380 | 0 reset(s); 1 duplicate token sample(s) removed | [66aja50m](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/66aja50m) |
| 275m | hybrid (GDN, expand_v=1) | 8 | 0.0032 | finished | 33.780 | 2.414943 | 0 reset(s); 1 duplicate token sample(s) removed | [ntoo8vlo](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ntoo8vlo) |
| 275m | wide integration (SWA) | 8 | 0.0004 | finished | 32.502 | 2.435915 | 0 reset(s); 1 duplicate token sample(s) removed | [iv901lom](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/iv901lom) |
| 275m | wide integration (SWA) | 8 | 0.0008 | finished | 32.502 | 2.419273 | 0 reset(s); 1 duplicate token sample(s) removed | [qe052lo4](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qe052lo4) |
| 275m | wide integration (SWA) | 8 | 0.0016 | finished | 32.502 | 2.423303 | 0 reset(s); 1 duplicate token sample(s) removed | [qu2zaxr7](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qu2zaxr7) |
| 275m | wide integration (SWA) | 8 | 0.0032 | finished | 32.502 | 2.441430 | 0 reset(s); 1 duplicate token sample(s) removed | [235ye5lg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/235ye5lg) |
| 480m | hybrid (GDN, expand_v=1) | 1 | 0.0012 | finished | 7.969 | 2.510874 | 0 reset(s); 1 duplicate token sample(s) removed | [wl8ebsd8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/wl8ebsd8) |
| 480m | wide integration (SWA) | 1 | 0.0012 | finished | 7.672 | 2.543281 | 0 reset(s); 1 duplicate token sample(s) removed | [z4wxvc6h](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/z4wxvc6h) |
| 480m | hybrid (GDN, expand_v=1) | 2 | 0.0009 | finished | 15.939 | 2.412790 | 0 reset(s); 1 duplicate token sample(s) removed | [4vzmrld1](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4vzmrld1) |
| 480m | wide integration (SWA) | 2 | 0.0009 | finished | 15.344 | 2.423888 | 0 reset(s); 1 duplicate token sample(s) removed | [ywj13bkw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ywj13bkw) |
| 480m | hybrid (GDN, expand_v=1) | 4 | 0.0008 | finished | 31.878 | 2.305996 | 0 reset(s); 1 duplicate token sample(s) removed | [h06m5ls2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h06m5ls2) |
| 480m | wide integration (SWA) | 4 | 0.0008 | finished | 30.687 | 2.329976 | 0 reset(s); 1 duplicate token sample(s) removed | [rblv9hpr](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rblv9hpr) |
| 480m | hybrid (GDN, expand_v=1) | 8 | 0.0008 | finished | 63.755 | 2.236205 | 0 reset(s); 1 duplicate token sample(s) removed | [d34a9o4t](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/d34a9o4t) |
| 480m | wide integration (SWA) | 8 | 0.0008 | finished | 61.375 | 2.251305 | 0 reset(s); 1 duplicate token sample(s) removed | [vdcrgfy0](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vdcrgfy0) |
| 810m | hybrid (GDN, expand_v=1) | 1 | 0.0006 | finished | 14.619 | 2.364345 | 0 reset(s); 1 duplicate token sample(s) removed | [h1rmcm2p](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h1rmcm2p) |
| 810m | wide integration (SWA) | 1 | 0.0006 | finished | 13.903 | 2.373197 | 0 reset(s); 1 duplicate token sample(s) removed | [w912irkq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/w912irkq) |
| 810m | hybrid (GDN, expand_v=1) | 2 | 0.00056 | finished | 29.238 | 2.247185 | 0 reset(s); 1 duplicate token sample(s) removed | [1d5gxgjv](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1d5gxgjv) |
| 810m | wide integration (SWA) | 2 | 0.00056 | finished | 27.805 | 2.268948 | 0 reset(s); 1 duplicate token sample(s) removed | [jpbqhfvc](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jpbqhfvc) |
| 810m | hybrid (GDN, expand_v=1) | 4 | 0.0004 | finished | 58.476 | 2.160440 | 0 reset(s); 1 duplicate token sample(s) removed | [kye1c19u](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kye1c19u) |
| 810m | wide integration (SWA) | 4 | 0.0004 | finished | 55.610 | 2.192802 | 0 reset(s); 1 duplicate token sample(s) removed | [58ftjxmw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/58ftjxmw) |
| 810m | hybrid (GDN, expand_v=1) | 8 | 0.0004 | finished | 116.953 | 2.095585 | 0 reset(s); 1 duplicate token sample(s) removed | [s5gvyjiz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/s5gvyjiz) |
| 810m | wide integration (SWA) | 8 | 0.0004 | finished | 111.220 | 2.104939 | 0 reset(s); 1 duplicate token sample(s) removed | [kyti8h1y](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kyti8h1y) |
| 1p2b | hybrid (GDN, expand_v=1) | 1 | 0.0004 | finished | 22.691 | 2.253953 | 0 reset(s); 1 duplicate token sample(s) removed | [1d24xfx5](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1d24xfx5) |
| 1p2b | wide integration (SWA) | 1 | 0.0004 | finished | 21.417 | 2.273062 | 0 reset(s); 1 duplicate token sample(s) removed | [hww8eksq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/hww8eksq) |
| 1p2b | hybrid (GDN, expand_v=1) | 2 | 0.0006 | finished | 45.381 | 2.163788 | 0 reset(s); 1 duplicate token sample(s) removed | [4k1bh4k2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4k1bh4k2) |
| 1p2b | wide integration (SWA) | 2 | 0.0006 | finished | 42.835 | 2.178332 | 0 reset(s); 1 duplicate token sample(s) removed | [jfwntmwm](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jfwntmwm) |
| 1p2b | hybrid (GDN, expand_v=1) | 4 | 0.0003 | finished | 90.762 | 2.081180 | 0 reset(s); 1 duplicate token sample(s) removed | [vc3c6gj6](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vc3c6gj6) |
| 1p2b | wide integration (SWA) | 4 | 0.0003 | finished | 85.670 | 2.094219 | 0 reset(s); 1 duplicate token sample(s) removed | [u7ab1tpb](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/u7ab1tpb) |
| 1p2b | hybrid (GDN, expand_v=1) | 8 | 0.0004 | finished | 181.524 | 2.016369 | 0 reset(s); 1 duplicate token sample(s) removed | [7eemhu7g](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7eemhu7g) |
| 1p2b | wide integration (SWA) | 8 | 0.0004 | finished | 171.340 | 2.022641 | 0 reset(s); 1 duplicate token sample(s) removed | [bqjzmiqi](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/bqjzmiqi) |
