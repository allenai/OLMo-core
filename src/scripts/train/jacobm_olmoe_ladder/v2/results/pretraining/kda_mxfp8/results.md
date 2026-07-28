# KDA aggressive-MXFP8 comparison

Generated: `2026-07-28T16:35:36.772272+00:00`

Selection metric: final `250M`-token mean training CE. Only finished runs are eligible.
The optimal-LR summary includes only bracketed 275M sweeps with a valid quadratic fit.
The all-size figure uses observed-optimal points for 275M and fixed wide-LR transfer points for larger sizes; pending hybrid cells are labeled explicitly.
Fitted LR minima in the 275M U-plot are visual aids and are never used to select results.

## Completed results

| Model | Cx | Mode | Status | References (loss @ LR) | Intervention (loss @ LR) | Deltas vs references |
|---|---:|---|---|---|---:|---|
| 275m | Cx1 | LR sweep | complete | geometry_kda_ev2_neg_nope_gated: 2.692695 @ 0.0008 | 2.685399 (0.0016) | geometry_kda_ev2_neg_nope_gated: -0.007296 |
| 275m | Cx2 | LR sweep | complete | geometry_kda_ev2_neg_nope_gated: 2.562520 @ 0.0016 | 2.566948 (0.0016) | geometry_kda_ev2_neg_nope_gated: +0.004429 |
| 275m | Cx4 | LR sweep | complete | geometry_kda_ev2_neg_nope_gated: 2.464247 @ 0.0008 | 2.463998 (0.0016) | geometry_kda_ev2_neg_nope_gated: -0.000250 |
| 275m | Cx8 | LR sweep | provisional (0/4) | geometry_kda_ev2_neg_nope_gated: 2.380273 @ 0.0008 | — | geometry_kda_ev2_neg_nope_gated: — |
| 480m | Cx1 | fixed-LR transfer | finished | geometry_kda_ev2_neg_nope_gated: 2.492283 @ 0.0012 | 2.497288 (0.0012) | geometry_kda_ev2_neg_nope_gated: +0.005005 |
| 480m | Cx2 | fixed-LR transfer | pending | geometry_kda_ev2_neg_nope_gated: 2.382695 @ 0.0009 | — | geometry_kda_ev2_neg_nope_gated: — |
| 480m | Cx4 | fixed-LR transfer | pending | geometry_kda_ev2_neg_nope_gated: 2.291179 @ 0.0008 | — | geometry_kda_ev2_neg_nope_gated: — |
| 480m | Cx8 | fixed-LR transfer | pending | geometry_kda_ev2_neg_nope_gated: 2.216501 @ 0.0008 | — | geometry_kda_ev2_neg_nope_gated: — |

## Runs

| Model | Variant | Cx | LR | State | Tokens (B) | Final-window CE | Replay handling | W&B |
|---|---|---:|---:|---|---:|---:|---|---|
| 275m | BF16 KDA (expand_v=2, negative eigenvalues; transferred LR) | 1 | 0.0008 | finished | 4.526 | 2.692695 | — | [75dy08n9](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/75dy08n9) |
| 275m | aggressive MXFP8 KDA (32-aligned experts, fused_v2/FA4) | 1 | 0.0004 | finished | 4.553 | 2.712740 | — | [au00v1w9](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/au00v1w9) |
| 275m | aggressive MXFP8 KDA (32-aligned experts, fused_v2/FA4) | 1 | 0.0008 | finished | 4.553 | 2.695471 | local W&B recovery (SHA256 2400c27ea596…) | [uzg7z0t2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/uzg7z0t2) |
| 275m | aggressive MXFP8 KDA (32-aligned experts, fused_v2/FA4) | 1 | 0.0016 | finished | 4.553 | 2.685399 | — | [6hn2u009](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6hn2u009) |
| 275m | aggressive MXFP8 KDA (32-aligned experts, fused_v2/FA4) | 1 | 0.0032 | finished | 4.553 | 2.694743 | — | [os58up4x](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/os58up4x) |
| 275m | BF16 KDA (expand_v=2, negative eigenvalues; transferred LR) | 2 | 0.0016 | finished | 9.051 | 2.562520 | — | [ysswifrz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ysswifrz) |
| 275m | aggressive MXFP8 KDA (32-aligned experts, fused_v2/FA4) | 2 | 0.0004 | finished | 9.106 | 2.592246 | 0 reset(s); 1 duplicate token sample(s) removed | [wgns7v5v](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/wgns7v5v) |
| 275m | aggressive MXFP8 KDA (32-aligned experts, fused_v2/FA4) | 2 | 0.0008 | finished | 9.106 | 2.575318 | 0 reset(s); 1 duplicate token sample(s) removed | [n39xypjm](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/n39xypjm) |
| 275m | aggressive MXFP8 KDA (32-aligned experts, fused_v2/FA4) | 2 | 0.0016 | finished | 9.106 | 2.566948 | 0 reset(s); 1 duplicate token sample(s) removed | [x1fah0f5](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/x1fah0f5) |
| 275m | aggressive MXFP8 KDA (32-aligned experts, fused_v2/FA4) | 2 | 0.0032 | finished | 9.106 | 2.579905 | 0 reset(s); 1 duplicate token sample(s) removed | [mcehf9yi](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mcehf9yi) |
| 275m | BF16 KDA (expand_v=2, negative eigenvalues; transferred LR) | 4 | 0.0008 | finished | 18.103 | 2.464247 | 0 reset(s); 1 duplicate token sample(s) removed | [cyyeyven](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/cyyeyven) |
| 275m | aggressive MXFP8 KDA (32-aligned experts, fused_v2/FA4) | 4 | 0.0004 | finished | 18.213 | 2.482960 | 0 reset(s); 1 duplicate token sample(s) removed | [8tbwijr0](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8tbwijr0) |
| 275m | aggressive MXFP8 KDA (32-aligned experts, fused_v2/FA4) | 4 | 0.0008 | finished | 18.213 | 2.466508 | 0 reset(s); 1 duplicate token sample(s) removed | [usmgo5jo](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/usmgo5jo) |
| 275m | aggressive MXFP8 KDA (32-aligned experts, fused_v2/FA4) | 4 | 0.0016 | finished | 18.213 | 2.463998 | 0 reset(s); 1 duplicate token sample(s) removed | [lr2xex1r](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/lr2xex1r) |
| 275m | aggressive MXFP8 KDA (32-aligned experts, fused_v2/FA4) | 4 | 0.0032 | finished | 18.213 | 2.482777 | 0 reset(s); 1 duplicate token sample(s) removed | [u5lenacg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/u5lenacg) |
| 275m | BF16 KDA (expand_v=2, negative eigenvalues; transferred LR) | 8 | 0.0008 | finished | 36.205 | 2.380273 | 0 reset(s); 1 duplicate token sample(s) removed | [vrjssy6q](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vrjssy6q) |
| 480m | BF16 KDA (expand_v=2, negative eigenvalues; transferred LR) | 1 | 0.0012 | finished | 8.433 | 2.492283 | — | [sb4yqi8x](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sb4yqi8x) |
| 480m | aggressive MXFP8 KDA (32-aligned experts, fused_v2/FA4) | 1 | 0.0012 | finished | 8.384 | 2.497288 | 0 reset(s); 1 duplicate token sample(s) removed | [ei2f8ttc](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ei2f8ttc) |
| 480m | BF16 KDA (expand_v=2, negative eigenvalues; transferred LR) | 2 | 0.0009 | finished | 16.867 | 2.382695 | 0 reset(s); 1 duplicate token sample(s) removed | [k2zf4esa](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/k2zf4esa) |
| 480m | BF16 KDA (expand_v=2, negative eigenvalues; transferred LR) | 4 | 0.0008 | finished | 33.734 | 2.291179 | 0 reset(s); 1 duplicate token sample(s) removed | [lesw7ogm](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/lesw7ogm) |
| 480m | BF16 KDA (expand_v=2, negative eigenvalues; transferred LR) | 8 | 0.0008 | finished | 67.468 | 2.216501 | 0 reset(s); 1 duplicate token sample(s) removed | [nfgfhyv8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/nfgfhyv8) |
| 810m | BF16 KDA (expand_v=2, negative eigenvalues; transferred LR) | 1 | 0.0006 | finished | 14.730 | 2.352304 | 0 reset(s); 1 duplicate token sample(s) removed | [4k5dasv8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4k5dasv8) |
| 810m | BF16 KDA (expand_v=2, negative eigenvalues; transferred LR) | 2 | 0.00056 | finished | 29.459 | 2.241873 | 0 reset(s); 1 duplicate token sample(s) removed | [1e7z0xar](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1e7z0xar) |
| 810m | BF16 KDA (expand_v=2, negative eigenvalues; transferred LR) | 4 | 0.0004 | finished | 58.918 | 2.158207 | 0 reset(s); 1 duplicate token sample(s) removed | [gxgef1hf](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/gxgef1hf) |
