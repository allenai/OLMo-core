# 275M KDA aggressive-MXFP8 comparison

Generated: `2026-07-28T03:07:50.855457+00:00`

Selection metric: final `250M`-token mean training CE. Only finished runs are eligible.
The optimal-LR summary includes only bracketed 275M sweeps with a valid quadratic fit.
The all-size figure uses observed-optimal points for 275M and fixed wide-LR transfer points for larger sizes; pending hybrid cells are labeled explicitly.
Fitted LR minima in the 275M U-plot are visual aids and are never used to select results.

## Completed results

| Model | Cx | Mode | Status | References (loss @ LR) | Intervention (loss @ LR) | Deltas vs references |
|---|---:|---|---|---|---:|---|
| 275m | Cx1 | LR sweep | complete | geometry_kda_ev2_neg_nope_gated: 2.692695 @ 0.0008 | 2.685399 (0.0016) | geometry_kda_ev2_neg_nope_gated: -0.007296 |
| 275m | Cx2 | LR sweep | provisional (0/4) | geometry_kda_ev2_neg_nope_gated: 2.562520 @ 0.0016 | — | geometry_kda_ev2_neg_nope_gated: — |
| 275m | Cx4 | LR sweep | provisional (0/4) | geometry_kda_ev2_neg_nope_gated: 2.464247 @ 0.0008 | — | geometry_kda_ev2_neg_nope_gated: — |
| 275m | Cx8 | LR sweep | provisional (0/4) | geometry_kda_ev2_neg_nope_gated: 2.380273 @ 0.0008 | — | geometry_kda_ev2_neg_nope_gated: — |

## Runs

| Model | Variant | Cx | LR | State | Tokens (B) | Final-window CE | Replay handling | W&B |
|---|---|---:|---:|---|---:|---:|---|---|
| 275m | BF16 KDA (expand_v=2, negative eigenvalues; transferred LR) | 1 | 0.0008 | finished | 4.526 | 2.692695 | — | [75dy08n9](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/75dy08n9) |
| 275m | aggressive MXFP8 KDA (672 expert width, fused_v2/FA4) | 1 | 0.0004 | finished | 4.553 | 2.712740 | — | [au00v1w9](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/au00v1w9) |
| 275m | aggressive MXFP8 KDA (672 expert width, fused_v2/FA4) | 1 | 0.0008 | finished | 4.553 | 2.695471 | local W&B recovery (SHA256 2400c27ea596…) | [uzg7z0t2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/uzg7z0t2) |
| 275m | aggressive MXFP8 KDA (672 expert width, fused_v2/FA4) | 1 | 0.0016 | finished | 4.553 | 2.685399 | — | [6hn2u009](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6hn2u009) |
| 275m | aggressive MXFP8 KDA (672 expert width, fused_v2/FA4) | 1 | 0.0032 | finished | 4.553 | 2.694743 | — | [os58up4x](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/os58up4x) |
| 275m | BF16 KDA (expand_v=2, negative eigenvalues; transferred LR) | 2 | 0.0016 | finished | 9.051 | 2.562520 | — | [ysswifrz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ysswifrz) |
| 275m | BF16 KDA (expand_v=2, negative eigenvalues; transferred LR) | 4 | 0.0008 | finished | 18.103 | 2.464247 | 0 reset(s); 1 duplicate token sample(s) removed | [cyyeyven](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/cyyeyven) |
| 275m | BF16 KDA (expand_v=2, negative eigenvalues; transferred LR) | 8 | 0.0008 | finished | 36.205 | 2.380273 | 0 reset(s); 1 duplicate token sample(s) removed | [vrjssy6q](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vrjssy6q) |
