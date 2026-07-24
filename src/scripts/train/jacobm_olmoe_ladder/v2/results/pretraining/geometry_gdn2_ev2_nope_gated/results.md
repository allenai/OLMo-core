# Geometry-matched NoPE gated-attention GDN2 intervention

Generated: `2026-07-24T06:41:42.505649+00:00`

Selection metric: final `250M`-token mean training CE. Only finished runs are eligible.
The optimal-LR summary includes only bracketed 275M sweeps with a valid quadratic fit.
The all-size figure uses observed-optimal points for 275M and fixed wide-LR transfer points for larger sizes; pending hybrid cells are labeled explicitly.
Fitted LR minima in the 275M U-plot are visual aids and are never used to select results.

## Completed results

| Model | Cx | Mode | Status | References (loss @ LR) | Intervention (loss @ LR) | Deltas vs references |
|---|---:|---|---|---|---:|---|
| 275m | Cx1 | LR sweep | complete | wide_integration: 2.741044 @ 0.0016; geometry_gdn_ev2_nope_gated: 2.711104 @ 0.0008 | 2.646730 (0.0016) | wide_integration: -0.094314; geometry_gdn_ev2_nope_gated: -0.064374 |
| 275m | Cx2 | LR sweep | provisional (3/4) | wide_integration: 2.608295 @ 0.0016; geometry_gdn_ev2_nope_gated: 2.580768 @ 0.0016 | 2.544136 (0.0008) | wide_integration: -0.064159; geometry_gdn_ev2_nope_gated: -0.036632 |
| 275m | Cx4 | LR sweep | provisional (0/4) | wide_integration: 2.506019 @ 0.0008; geometry_gdn_ev2_nope_gated: 2.476065 @ 0.0008 | — | wide_integration: —; geometry_gdn_ev2_nope_gated: — |
| 275m | Cx8 | LR sweep | provisional (0/4) | wide_integration: 2.419273 @ 0.0008; geometry_gdn_ev2_nope_gated: 2.390397 @ 0.0008 | — | wide_integration: —; geometry_gdn_ev2_nope_gated: — |

## Runs

| Model | Variant | Cx | LR | State | Tokens (B) | Final-window CE | Replay handling | W&B |
|---|---|---:|---:|---|---:|---:|---|---|
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 1 | 0.0004 | finished | 4.839 | 2.676958 | 0 reset(s); 1 duplicate token sample(s) removed | [rsxmn720](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rsxmn720) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 1 | 0.0008 | finished | 4.839 | 2.657052 | 0 reset(s); 1 duplicate token sample(s) removed | [pqrdvu63](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/pqrdvu63) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 1 | 0.0016 | finished | 4.839 | 2.646730 | 0 reset(s); 1 duplicate token sample(s) removed | [5uzr9dva](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5uzr9dva) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 1 | 0.0032 | finished | 4.839 | 2.661486 | 0 reset(s); 1 duplicate token sample(s) removed | [j2t5c2jb](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/j2t5c2jb) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 1 | 0.0004 | finished | 4.557 | 2.724486 | 0 reset(s); 1 duplicate token sample(s) removed | [lg619wiz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/lg619wiz) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 1 | 0.0008 | finished | 4.557 | 2.711104 | 0 reset(s); 1 duplicate token sample(s) removed | [q81uxrxu](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/q81uxrxu) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 1 | 0.0016 | finished | 4.557 | 2.715845 | 0 reset(s); 1 duplicate token sample(s) removed | [sxuuwzzm](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sxuuwzzm) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 1 | 0.0032 | finished | 4.557 | 2.730489 | 0 reset(s); 1 duplicate token sample(s) removed | [1pr3blts](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1pr3blts) |
| 275m | wide integration (SWA) | 1 | 0.0008 | finished | 4.063 | 2.749142 | 0 reset(s); 1 duplicate token sample(s) removed | [kfua3dcq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kfua3dcq) |
| 275m | wide integration (SWA) | 1 | 0.0016 | finished | 4.063 | 2.741044 | 0 reset(s); 1 duplicate token sample(s) removed | [h86x1nv3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h86x1nv3) |
| 275m | wide integration (SWA) | 1 | 0.0032 | finished | 4.063 | 2.749132 | 0 reset(s); 1 duplicate token sample(s) removed | [afxq80js](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/afxq80js) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 2 | 0.0004 | finished | 9.679 | 2.564391 | 0 reset(s); 1 duplicate token sample(s) removed | [7yh4rfi1](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7yh4rfi1) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 2 | 0.0008 | finished | 9.679 | 2.544136 | 0 reset(s); 1 duplicate token sample(s) removed | [2egeqyvo](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/2egeqyvo) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 2 | 0.0032 | finished | 9.679 | 2.548521 | 0 reset(s); 1 duplicate token sample(s) removed | [xwtxd1pv](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xwtxd1pv) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 2 | 0.0004 | finished | 9.115 | 2.601429 | 0 reset(s); 1 duplicate token sample(s) removed | [sehjqtyk](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sehjqtyk) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 2 | 0.0008 | finished | 9.115 | 2.584133 | 0 reset(s); 1 duplicate token sample(s) removed | [bttby9r8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/bttby9r8) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 2 | 0.0016 | finished | 9.115 | 2.580768 | 0 reset(s); 1 duplicate token sample(s) removed | [ef4umox3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ef4umox3) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 2 | 0.0032 | finished | 9.115 | 2.597651 | 0 reset(s); 1 duplicate token sample(s) removed | [ofodwbzz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ofodwbzz) |
| 275m | wide integration (SWA) | 2 | 0.0008 | finished | 8.126 | 2.612348 | 0 reset(s); 1 duplicate token sample(s) removed | [o2bdr3gw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/o2bdr3gw) |
| 275m | wide integration (SWA) | 2 | 0.0016 | finished | 8.126 | 2.608295 | 0 reset(s); 1 duplicate token sample(s) removed | [6porpbo2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6porpbo2) |
| 275m | wide integration (SWA) | 2 | 0.0032 | finished | 8.126 | 2.614495 | 0 reset(s); 1 duplicate token sample(s) removed | [0f782vrw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0f782vrw) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 4 | 0.0004 | finished | 18.229 | 2.491557 | 0 reset(s); 1 duplicate token sample(s) removed | [fh9tl31v](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/fh9tl31v) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 4 | 0.0008 | finished | 18.229 | 2.476065 | 0 reset(s); 1 duplicate token sample(s) removed | [gwzx0ekc](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/gwzx0ekc) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 4 | 0.0016 | finished | 18.229 | 2.476188 | 0 reset(s); 1 duplicate token sample(s) removed | [jr74v01c](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jr74v01c) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 4 | 0.0032 | finished | 18.229 | 2.496595 | 0 reset(s); 1 duplicate token sample(s) removed | [2s5s1yw0](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/2s5s1yw0) |
| 275m | wide integration (SWA) | 4 | 0.0004 | finished | 16.251 | 2.520624 | 0 reset(s); 1 duplicate token sample(s) removed | [n1gjknwg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/n1gjknwg) |
| 275m | wide integration (SWA) | 4 | 0.0008 | finished | 16.251 | 2.506019 | 0 reset(s); 1 duplicate token sample(s) removed | [9n3xk8gs](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9n3xk8gs) |
| 275m | wide integration (SWA) | 4 | 0.0016 | finished | 16.251 | 2.508140 | 0 reset(s); 1 duplicate token sample(s) removed | [ttjquo05](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ttjquo05) |
| 275m | wide integration (SWA) | 4 | 0.0032 | finished | 16.251 | 2.522459 | 0 reset(s); 1 duplicate token sample(s) removed | [5u03fshf](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5u03fshf) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 8 | 0.0004 | finished | 36.459 | 2.405406 | 0 reset(s); 1 duplicate token sample(s) removed | [qehufcr5](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qehufcr5) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 8 | 0.0008 | finished | 36.459 | 2.390397 | 0 reset(s); 1 duplicate token sample(s) removed | [ouxblu4g](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ouxblu4g) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 8 | 0.0016 | finished | 36.459 | 2.392762 | 0 reset(s); 1 duplicate token sample(s) removed | [3xjjt5sa](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3xjjt5sa) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 8 | 0.0032 | finished | 36.459 | 2.413536 | 0 reset(s); 1 duplicate token sample(s) removed | [mbvin02a](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mbvin02a) |
| 275m | wide integration (SWA) | 8 | 0.0004 | finished | 32.502 | 2.435915 | 0 reset(s); 1 duplicate token sample(s) removed | [iv901lom](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/iv901lom) |
| 275m | wide integration (SWA) | 8 | 0.0008 | finished | 32.502 | 2.419273 | 0 reset(s); 1 duplicate token sample(s) removed | [qe052lo4](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qe052lo4) |
| 275m | wide integration (SWA) | 8 | 0.0016 | finished | 32.502 | 2.423303 | 0 reset(s); 1 duplicate token sample(s) removed | [qu2zaxr7](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qu2zaxr7) |
| 275m | wide integration (SWA) | 8 | 0.0032 | finished | 32.502 | 2.441430 | 0 reset(s); 1 duplicate token sample(s) removed | [235ye5lg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/235ye5lg) |
