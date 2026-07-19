# Geometry-matched NoPE active hybrid GDN intervention

Generated: `2026-07-19T04:29:41.768268+00:00`

Selection metric: final `250M`-token mean training CE. Only finished runs are eligible.
The optimal-LR summary includes only bracketed 275M sweeps with a valid quadratic fit.
The all-size figure uses observed-optimal points for 275M and fixed wide-LR transfer points for larger sizes; pending hybrid cells are labeled explicitly.
Fitted LR minima in the 275M U-plot are visual aids and are never used to select results.

## Completed results

| Model | Cx | Mode | Status | References (loss @ LR) | Intervention (loss @ LR) | Deltas vs references |
|---|---:|---|---|---|---:|---|
| 275m | Cx1 | LR sweep | complete | wide_integration: 2.741044 @ 0.0016; hybrid_gdn_ev1: 2.694642 @ 0.0016; geometry_gdn_ev2: 2.707881 @ 0.0016 | 2.712805 (0.0008) | wide_integration: -0.028239; hybrid_gdn_ev1: +0.018164; geometry_gdn_ev2: +0.004924 |
| 275m | Cx2 | LR sweep | complete | wide_integration: 2.608295 @ 0.0016; hybrid_gdn_ev1: 2.569988 @ 0.0016; geometry_gdn_ev2: 2.578963 @ 0.0016 | 2.585179 (0.0016) | wide_integration: -0.023116; hybrid_gdn_ev1: +0.015191; geometry_gdn_ev2: +0.006217 |
| 275m | Cx4 | LR sweep | complete | wide_integration: 2.506019 @ 0.0008; hybrid_gdn_ev1: 2.478165 @ 0.0016; geometry_gdn_ev2: 2.474631 @ 0.0016 | 2.477784 (0.0008) | wide_integration: -0.028234; hybrid_gdn_ev1: -0.000381; geometry_gdn_ev2: +0.003153 |
| 275m | Cx8 | LR sweep | complete | wide_integration: 2.419273 @ 0.0008; hybrid_gdn_ev1: 2.396380 @ 0.0016; geometry_gdn_ev2: 2.389857 @ 0.0008 | 2.391953 (0.0008) | wide_integration: -0.027321; hybrid_gdn_ev1: -0.004427; geometry_gdn_ev2: +0.002096 |
| 480m | Cx1 | fixed-LR transfer | finished | wide_integration: 2.543281 @ 0.0012; hybrid_gdn_ev1: 2.510874 @ 0.0012; geometry_gdn_ev2: — | 2.526546 (0.0012) | wide_integration: -0.016735; hybrid_gdn_ev1: +0.015672; geometry_gdn_ev2: — |
| 480m | Cx2 | fixed-LR transfer | finished | wide_integration: 2.423888 @ 0.0009; hybrid_gdn_ev1: 2.412790 @ 0.0009; geometry_gdn_ev2: — | 2.419441 (0.0009) | wide_integration: -0.004447; hybrid_gdn_ev1: +0.006651; geometry_gdn_ev2: — |
| 480m | Cx4 | fixed-LR transfer | finished | wide_integration: 2.329976 @ 0.0008; hybrid_gdn_ev1: 2.305996 @ 0.0008; geometry_gdn_ev2: — | 2.323768 (0.0008) | wide_integration: -0.006208; hybrid_gdn_ev1: +0.017772; geometry_gdn_ev2: — |
| 480m | Cx8 | fixed-LR transfer | finished | wide_integration: 2.251305 @ 0.0008; hybrid_gdn_ev1: 2.236205 @ 0.0008; geometry_gdn_ev2: — | 2.239326 (0.0008) | wide_integration: -0.011979; hybrid_gdn_ev1: +0.003121; geometry_gdn_ev2: — |
| 810m | Cx1 | fixed-LR transfer | finished | wide_integration: 2.373197 @ 0.0006; hybrid_gdn_ev1: 2.364345 @ 0.0006; geometry_gdn_ev2: — | 2.377107 (0.0006) | wide_integration: +0.003910; hybrid_gdn_ev1: +0.012762; geometry_gdn_ev2: — |
| 810m | Cx2 | fixed-LR transfer | finished | wide_integration: 2.268948 @ 0.00056; hybrid_gdn_ev1: 2.247185 @ 0.00056; geometry_gdn_ev2: — | 2.278701 (0.00056) | wide_integration: +0.009754; hybrid_gdn_ev1: +0.031516; geometry_gdn_ev2: — |
| 810m | Cx4 | fixed-LR transfer | finished | wide_integration: 2.192802 @ 0.0004; hybrid_gdn_ev1: 2.160440 @ 0.0004; geometry_gdn_ev2: — | 2.191890 (0.0004) | wide_integration: -0.000911; hybrid_gdn_ev1: +0.031450; geometry_gdn_ev2: — |
| 810m | Cx8 | fixed-LR transfer | pending | wide_integration: 2.104939 @ 0.0004; hybrid_gdn_ev1: 2.095585 @ 0.0004; geometry_gdn_ev2: — | — | wide_integration: —; hybrid_gdn_ev1: —; geometry_gdn_ev2: — |
| 1p2b | Cx1 | fixed-LR transfer | pending | wide_integration: 2.273062 @ 0.0004; hybrid_gdn_ev1: 2.253953 @ 0.0004; geometry_gdn_ev2: — | — | wide_integration: —; hybrid_gdn_ev1: —; geometry_gdn_ev2: — |
| 1p2b | Cx2 | fixed-LR transfer | pending | wide_integration: 2.178332 @ 0.0006; hybrid_gdn_ev1: 2.163788 @ 0.0006; geometry_gdn_ev2: — | — | wide_integration: —; hybrid_gdn_ev1: —; geometry_gdn_ev2: — |
| 1p2b | Cx4 | fixed-LR transfer | pending | wide_integration: 2.094219 @ 0.0003; hybrid_gdn_ev1: —; geometry_gdn_ev2: — | — | wide_integration: —; hybrid_gdn_ev1: —; geometry_gdn_ev2: — |
| 1p2b | Cx8 | fixed-LR transfer | pending | wide_integration: 2.022641 @ 0.0004; hybrid_gdn_ev1: —; geometry_gdn_ev2: — | — | wide_integration: —; hybrid_gdn_ev1: —; geometry_gdn_ev2: — |

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
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 2 | 0.0004 | finished | 9.062 | 2.602154 | — | [wpbz1ar9](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/wpbz1ar9) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 2 | 0.0008 | finished | 9.062 | 2.586164 | — | [7u4epzt6](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7u4epzt6) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 2 | 0.0016 | finished | 9.062 | 2.585179 | — | [gjmz37ct](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/gjmz37ct) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 2 | 0.0032 | finished | 9.062 | 2.611176 | — | [xahm1pbt](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xahm1pbt) |
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
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 4 | 0.0004 | finished | 18.125 | 2.494023 | — | [pmfco9gy](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/pmfco9gy) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 4 | 0.0008 | finished | 18.125 | 2.477784 | — | [k5mjm4ev](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/k5mjm4ev) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 4 | 0.0016 | finished | 18.125 | 2.479370 | — | [4x00n8lj](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4x00n8lj) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 4 | 0.0032 | finished | 18.125 | 2.503428 | — | [z1lw0z2i](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/z1lw0z2i) |
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
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 8 | 0.0004 | finished | 36.250 | 2.406665 | — | [t76b5xjy](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/t76b5xjy) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 8 | 0.0008 | finished | 36.250 | 2.391953 | — | [d29gx1x9](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/d29gx1x9) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 8 | 0.0016 | finished | 36.250 | 2.394529 | — | [n8kkr1y8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/n8kkr1y8) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 8 | 0.0032 | finished | 36.250 | 2.418367 | — | [qhjvjwcu](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qhjvjwcu) |
| 275m | hybrid (GDN, expand_v=1) | 8 | 0.0004 | finished | 33.780 | 2.412233 | — | [b0z3qfmi](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/b0z3qfmi) |
| 275m | hybrid (GDN, expand_v=1) | 8 | 0.0008 | finished | 33.780 | 2.397966 | — | [rkxojd03](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rkxojd03) |
| 275m | hybrid (GDN, expand_v=1) | 8 | 0.0016 | finished | 33.780 | 2.396380 | — | [66aja50m](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/66aja50m) |
| 275m | hybrid (GDN, expand_v=1) | 8 | 0.0032 | finished | 33.780 | 2.414943 | — | [ntoo8vlo](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ntoo8vlo) |
| 275m | wide integration (SWA) | 8 | 0.0004 | finished | 32.502 | 2.435915 | — | [iv901lom](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/iv901lom) |
| 275m | wide integration (SWA) | 8 | 0.0008 | finished | 32.502 | 2.419273 | — | [qe052lo4](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qe052lo4) |
| 275m | wide integration (SWA) | 8 | 0.0016 | finished | 32.502 | 2.423303 | — | [qu2zaxr7](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qu2zaxr7) |
| 275m | wide integration (SWA) | 8 | 0.0032 | finished | 32.502 | 2.441430 | — | [235ye5lg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/235ye5lg) |
| 480m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 1 | 0.0012 | finished | 8.481 | 2.526546 | — | [i2bij623](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/i2bij623) |
| 480m | hybrid (GDN, expand_v=1) | 1 | 0.0012 | finished | 7.969 | 2.510874 | — | [wl8ebsd8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/wl8ebsd8) |
| 480m | wide integration (SWA) | 1 | 0.0012 | finished | 7.672 | 2.543281 | — | [z4wxvc6h](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/z4wxvc6h) |
| 480m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 2 | 0.0009 | finished | 16.963 | 2.419441 | — | [2d1t07dn](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/2d1t07dn) |
| 480m | hybrid (GDN, expand_v=1) | 2 | 0.0009 | finished | 15.939 | 2.412790 | — | [4vzmrld1](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4vzmrld1) |
| 480m | wide integration (SWA) | 2 | 0.0009 | finished | 15.344 | 2.423888 | — | [ywj13bkw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ywj13bkw) |
| 480m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 4 | 0.0008 | finished | 33.926 | 2.323768 | — | [ke2n42cm](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ke2n42cm) |
| 480m | hybrid (GDN, expand_v=1) | 4 | 0.0008 | finished | 31.878 | 2.305996 | — | [h06m5ls2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h06m5ls2) |
| 480m | wide integration (SWA) | 4 | 0.0008 | finished | 30.687 | 2.329976 | — | [rblv9hpr](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rblv9hpr) |
| 480m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 8 | 0.0008 | finished | 67.851 | 2.239326 | — | [pej34iwq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/pej34iwq) |
| 480m | hybrid (GDN, expand_v=1) | 8 | 0.0008 | finished | 63.755 | 2.236205 | — | [d34a9o4t](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/d34a9o4t) |
| 480m | wide integration (SWA) | 8 | 0.0008 | finished | 61.375 | 2.251305 | — | [vdcrgfy0](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vdcrgfy0) |
| 810m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 1 | 0.0006 | finished | 15.110 | 2.377107 | — | [8z7txpf8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8z7txpf8) |
| 810m | hybrid (GDN, expand_v=1) | 1 | 0.0006 | finished | 14.619 | 2.364345 | — | [h1rmcm2p](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h1rmcm2p) |
| 810m | wide integration (SWA) | 1 | 0.0006 | finished | 13.903 | 2.373197 | — | [w912irkq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/w912irkq) |
| 810m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 2 | 0.00056 | finished | 30.219 | 2.278701 | — | [upxsysuv](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/upxsysuv) |
| 810m | hybrid (GDN, expand_v=1) | 2 | 0.00056 | finished | 29.238 | 2.247185 | — | [1d5gxgjv](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1d5gxgjv) |
| 810m | wide integration (SWA) | 2 | 0.00056 | finished | 27.805 | 2.268948 | — | [jpbqhfvc](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jpbqhfvc) |
| 810m | geometry-matched hybrid (GDN, expand_v=2, NoPE) | 4 | 0.0004 | finished | 60.438 | 2.191890 | — | [8ewnju8z](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8ewnju8z) |
| 810m | hybrid (GDN, expand_v=1) | 4 | 0.0004 | finished | 58.476 | 2.160440 | — | [kye1c19u](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kye1c19u) |
| 810m | wide integration (SWA) | 4 | 0.0004 | finished | 55.610 | 2.192802 | — | [58ftjxmw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/58ftjxmw) |
| 810m | hybrid (GDN, expand_v=1) | 8 | 0.0004 | finished | 116.953 | 2.095585 | — | [s5gvyjiz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/s5gvyjiz) |
| 810m | wide integration (SWA) | 8 | 0.0004 | finished | 111.220 | 2.104939 | — | [kyti8h1y](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kyti8h1y) |
| 1p2b | hybrid (GDN, expand_v=1) | 1 | 0.0004 | finished | 22.691 | 2.253953 | — | [1d24xfx5](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1d24xfx5) |
| 1p2b | wide integration (SWA) | 1 | 0.0004 | finished | 21.417 | 2.273062 | — | [hww8eksq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/hww8eksq) |
| 1p2b | hybrid (GDN, expand_v=1) | 2 | 0.0006 | finished | 45.381 | 2.163788 | — | [4k1bh4k2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4k1bh4k2) |
| 1p2b | wide integration (SWA) | 2 | 0.0006 | finished | 42.835 | 2.178332 | — | [jfwntmwm](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jfwntmwm) |
| 1p2b | wide integration (SWA) | 4 | 0.0003 | finished | 85.670 | 2.094219 | — | [u7ab1tpb](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/u7ab1tpb) |
| 1p2b | wide integration (SWA) | 8 | 0.0004 | finished | 171.340 | 2.022641 | — | [bqjzmiqi](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/bqjzmiqi) |
