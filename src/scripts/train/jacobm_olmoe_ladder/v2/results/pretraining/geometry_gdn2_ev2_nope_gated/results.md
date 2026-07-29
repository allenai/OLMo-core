# Geometry-matched NoPE gated-attention GDN2 intervention

Generated: `2026-07-29T06:54:34.139753+00:00`

Selection metric: final `250M`-token mean training CE. Only finished runs are eligible.
The optimal-LR summary includes only bracketed 275M sweeps with a valid quadratic fit.
The all-size figure uses observed-optimal points for 275M and fixed wide-LR transfer points for larger sizes; pending hybrid cells are labeled explicitly.
Fitted LR minima in the 275M U-plot are visual aids and are never used to select results.

## Completed results

| Model | Cx | Mode | Status | References (loss @ LR) | Intervention (loss @ LR) | Deltas vs references |
|---|---:|---|---|---|---:|---|
| 275m | Cx1 | LR sweep | complete | wide_integration: 2.741044 @ 0.0016; geometry_gdn_ev2_nope_gated: 2.711104 @ 0.0008 | 2.646730 (0.0016) | wide_integration: -0.094314; geometry_gdn_ev2_nope_gated: -0.064374 |
| 275m | Cx2 | LR sweep | complete | wide_integration: 2.608295 @ 0.0016; geometry_gdn_ev2_nope_gated: 2.580768 @ 0.0016 | 2.534116 (0.0016) | wide_integration: -0.074180; geometry_gdn_ev2_nope_gated: -0.046652 |
| 275m | Cx4 | LR sweep | complete | wide_integration: 2.506019 @ 0.0008; geometry_gdn_ev2_nope_gated: 2.476065 @ 0.0008 | 2.443132 (0.0016) | wide_integration: -0.062887; geometry_gdn_ev2_nope_gated: -0.032933 |
| 275m | Cx8 | LR sweep | provisional (3/4) | wide_integration: 2.419273 @ 0.0008; geometry_gdn_ev2_nope_gated: 2.390397 @ 0.0008 | 2.356985 (0.0008) | wide_integration: -0.062288; geometry_gdn_ev2_nope_gated: -0.033412 |
| 480m | Cx1 | fixed-LR transfer | finished | wide_integration: 2.543281 @ 0.0012; geometry_gdn_ev2_nope_gated: 2.519642 @ 0.0012 | 2.468555 (0.0012) | wide_integration: -0.074726; geometry_gdn_ev2_nope_gated: -0.051087 |
| 480m | Cx2 | fixed-LR transfer | finished | wide_integration: 2.423888 @ 0.0009; geometry_gdn_ev2_nope_gated: 2.414718 @ 0.0009 | 2.359149 (0.0009) | wide_integration: -0.064739; geometry_gdn_ev2_nope_gated: -0.055569 |
| 480m | Cx4 | fixed-LR transfer | finished | wide_integration: 2.329976 @ 0.0008; geometry_gdn_ev2_nope_gated: 2.315124 @ 0.0008 | 2.276454 (0.0008) | wide_integration: -0.053522; geometry_gdn_ev2_nope_gated: -0.038670 |
| 480m | Cx8 | fixed-LR transfer | finished | wide_integration: 2.251305 @ 0.0008; geometry_gdn_ev2_nope_gated: 2.239297 @ 0.0008 | 2.204316 (0.0008) | wide_integration: -0.046989; geometry_gdn_ev2_nope_gated: -0.034982 |
| 810m | Cx1 | fixed-LR transfer | finished | wide_integration: 2.373197 @ 0.0006; geometry_gdn_ev2_nope_gated: 2.373592 @ 0.0006 | 2.323505 (0.0006) | wide_integration: -0.049692; geometry_gdn_ev2_nope_gated: -0.050087 |
| 810m | Cx2 | fixed-LR transfer | pending | wide_integration: 2.268948 @ 0.00056; geometry_gdn_ev2_nope_gated: 2.277253 @ 0.00056 | — | wide_integration: —; geometry_gdn_ev2_nope_gated: — |
| 810m | Cx4 | fixed-LR transfer | finished | wide_integration: 2.192802 @ 0.0004; geometry_gdn_ev2_nope_gated: 2.191179 @ 0.0004 | 2.152382 (0.0004) | wide_integration: -0.040420; geometry_gdn_ev2_nope_gated: -0.038797 |
| 810m | Cx8 | fixed-LR transfer | pending | wide_integration: 2.104939 @ 0.0004; geometry_gdn_ev2_nope_gated: 2.114516 @ 0.0004 | — | wide_integration: —; geometry_gdn_ev2_nope_gated: — |

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
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 2 | 0.0016 | finished | 9.679 | 2.534116 | 1 reset(s); 79 duplicate token sample(s) removed | [gat5rtub](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/gat5rtub) / [8agi9zte](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8agi9zte) / [jhcmk80f](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jhcmk80f) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 2 | 0.0032 | finished | 9.679 | 2.548521 | 0 reset(s); 1 duplicate token sample(s) removed | [xwtxd1pv](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xwtxd1pv) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 2 | 0.0004 | finished | 9.115 | 2.601429 | 0 reset(s); 1 duplicate token sample(s) removed | [sehjqtyk](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sehjqtyk) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 2 | 0.0008 | finished | 9.115 | 2.584133 | 0 reset(s); 1 duplicate token sample(s) removed | [bttby9r8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/bttby9r8) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 2 | 0.0016 | finished | 9.115 | 2.580768 | 0 reset(s); 1 duplicate token sample(s) removed | [ef4umox3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ef4umox3) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 2 | 0.0032 | finished | 9.115 | 2.597651 | 0 reset(s); 1 duplicate token sample(s) removed | [ofodwbzz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ofodwbzz) |
| 275m | wide integration (SWA) | 2 | 0.0008 | finished | 8.126 | 2.612348 | 0 reset(s); 1 duplicate token sample(s) removed | [o2bdr3gw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/o2bdr3gw) |
| 275m | wide integration (SWA) | 2 | 0.0016 | finished | 8.126 | 2.608295 | 0 reset(s); 1 duplicate token sample(s) removed | [6porpbo2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6porpbo2) |
| 275m | wide integration (SWA) | 2 | 0.0032 | finished | 8.126 | 2.614495 | 0 reset(s); 1 duplicate token sample(s) removed | [0f782vrw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0f782vrw) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 4 | 0.0004 | finished | 19.358 | 2.462028 | 0 reset(s); 1 duplicate token sample(s) removed | [6b0vighm](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6b0vighm) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 4 | 0.0008 | finished | 19.358 | 2.446822 | 0 reset(s); 1 duplicate token sample(s) removed | [yq4mi5o0](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/yq4mi5o0) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 4 | 0.0016 | finished | 19.358 | 2.443132 | 0 reset(s); 1 duplicate token sample(s) removed | [0w6ezwgx](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0w6ezwgx) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 4 | 0.0032 | finished | 19.358 | 2.461867 | 0 reset(s); 1 duplicate token sample(s) removed | [kcig30ty](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kcig30ty) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 4 | 0.0004 | finished | 18.229 | 2.491557 | 0 reset(s); 1 duplicate token sample(s) removed | [fh9tl31v](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/fh9tl31v) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 4 | 0.0008 | finished | 18.229 | 2.476065 | 0 reset(s); 1 duplicate token sample(s) removed | [gwzx0ekc](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/gwzx0ekc) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 4 | 0.0016 | finished | 18.229 | 2.476188 | 0 reset(s); 1 duplicate token sample(s) removed | [jr74v01c](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jr74v01c) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 4 | 0.0032 | finished | 18.229 | 2.496595 | 0 reset(s); 1 duplicate token sample(s) removed | [2s5s1yw0](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/2s5s1yw0) |
| 275m | wide integration (SWA) | 4 | 0.0004 | finished | 16.251 | 2.520624 | 0 reset(s); 1 duplicate token sample(s) removed | [n1gjknwg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/n1gjknwg) |
| 275m | wide integration (SWA) | 4 | 0.0008 | finished | 16.251 | 2.506019 | 0 reset(s); 1 duplicate token sample(s) removed | [9n3xk8gs](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9n3xk8gs) |
| 275m | wide integration (SWA) | 4 | 0.0016 | finished | 16.251 | 2.508140 | 0 reset(s); 1 duplicate token sample(s) removed | [ttjquo05](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ttjquo05) |
| 275m | wide integration (SWA) | 4 | 0.0032 | finished | 16.251 | 2.522459 | 0 reset(s); 1 duplicate token sample(s) removed | [5u03fshf](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5u03fshf) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 8 | 0.0004 | finished | 38.715 | 2.372393 | 0 reset(s); 1 duplicate token sample(s) removed | [jewjx6yq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jewjx6yq) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 8 | 0.0008 | finished | 38.715 | 2.356985 | 0 reset(s); 1 duplicate token sample(s) removed | [1lpz9reu](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1lpz9reu) |
| 275m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 8 | 0.0032 | finished | 38.715 | 2.380649 | 0 reset(s); 1 duplicate token sample(s) removed | [e6n5iscu](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/e6n5iscu) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 8 | 0.0004 | finished | 36.459 | 2.405406 | 0 reset(s); 1 duplicate token sample(s) removed | [qehufcr5](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qehufcr5) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 8 | 0.0008 | finished | 36.459 | 2.390397 | 0 reset(s); 1 duplicate token sample(s) removed | [ouxblu4g](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ouxblu4g) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 8 | 0.0016 | finished | 36.459 | 2.392762 | 0 reset(s); 1 duplicate token sample(s) removed | [3xjjt5sa](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3xjjt5sa) |
| 275m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 8 | 0.0032 | finished | 36.459 | 2.413536 | 0 reset(s); 1 duplicate token sample(s) removed | [mbvin02a](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mbvin02a) |
| 275m | wide integration (SWA) | 8 | 0.0004 | finished | 32.502 | 2.435915 | 0 reset(s); 1 duplicate token sample(s) removed | [iv901lom](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/iv901lom) |
| 275m | wide integration (SWA) | 8 | 0.0008 | finished | 32.502 | 2.419273 | 0 reset(s); 1 duplicate token sample(s) removed | [qe052lo4](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qe052lo4) |
| 275m | wide integration (SWA) | 8 | 0.0016 | finished | 32.502 | 2.423303 | 0 reset(s); 1 duplicate token sample(s) removed | [qu2zaxr7](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qu2zaxr7) |
| 275m | wide integration (SWA) | 8 | 0.0032 | finished | 32.502 | 2.441430 | 0 reset(s); 1 duplicate token sample(s) removed | [235ye5lg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/235ye5lg) |
| 480m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 1 | 0.0012 | finished | 8.998 | 2.468555 | 0 reset(s); 1 duplicate token sample(s) removed | [6r2blwru](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6r2blwru) |
| 480m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 1 | 0.0012 | finished | 8.529 | 2.519642 | 0 reset(s); 1 duplicate token sample(s) removed | [9rltp47w](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9rltp47w) |
| 480m | wide integration (SWA) | 1 | 0.0012 | finished | 7.672 | 2.543281 | 0 reset(s); 1 duplicate token sample(s) removed | [z4wxvc6h](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/z4wxvc6h) |
| 480m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 2 | 0.0009 | finished | 17.997 | 2.359149 | 0 reset(s); 1 duplicate token sample(s) removed | [07rx8ez4](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/07rx8ez4) |
| 480m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 2 | 0.0009 | finished | 17.057 | 2.414718 | 0 reset(s); 1 duplicate token sample(s) removed | [0crj05wz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0crj05wz) |
| 480m | wide integration (SWA) | 2 | 0.0009 | finished | 15.344 | 2.423888 | 0 reset(s); 1 duplicate token sample(s) removed | [ywj13bkw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ywj13bkw) |
| 480m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 4 | 0.0008 | finished | 35.993 | 2.276454 | 0 reset(s); 1 duplicate token sample(s) removed | [9u4z0e36](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9u4z0e36) |
| 480m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 4 | 0.0008 | finished | 34.114 | 2.315124 | 0 reset(s); 1 duplicate token sample(s) removed | [ur7yonej](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ur7yonej) |
| 480m | wide integration (SWA) | 4 | 0.0008 | finished | 30.687 | 2.329976 | 0 reset(s); 1 duplicate token sample(s) removed | [rblv9hpr](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rblv9hpr) |
| 480m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 8 | 0.0008 | finished | 71.986 | 2.204316 | 0 reset(s); 1 duplicate token sample(s) removed | [7p8q3v6p](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7p8q3v6p) |
| 480m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 8 | 0.0008 | finished | 68.228 | 2.239297 | 0 reset(s); 1 duplicate token sample(s) removed | [4737op7s](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4737op7s) |
| 480m | wide integration (SWA) | 8 | 0.0008 | finished | 61.375 | 2.251305 | 0 reset(s); 1 duplicate token sample(s) removed | [vdcrgfy0](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vdcrgfy0) |
| 810m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 1 | 0.0006 | finished | 16.236 | 2.323505 | 0 reset(s); 1 duplicate token sample(s) removed | [3wukvwyl](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3wukvwyl) |
| 810m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 1 | 0.0006 | finished | 15.236 | 2.373592 | 0 reset(s); 1 duplicate token sample(s) removed | [027xoq0r](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/027xoq0r) |
| 810m | wide integration (SWA) | 1 | 0.0006 | finished | 13.903 | 2.373197 | 0 reset(s); 1 duplicate token sample(s) removed | [w912irkq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/w912irkq) |
| 810m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 2 | 0.00056 | finished | 30.471 | 2.277253 | 0 reset(s); 1 duplicate token sample(s) removed | [7ryj4klm](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7ryj4klm) |
| 810m | wide integration (SWA) | 2 | 0.00056 | finished | 27.805 | 2.268948 | 0 reset(s); 1 duplicate token sample(s) removed | [jpbqhfvc](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jpbqhfvc) |
| 810m | geometry-matched hybrid (GDN2, expand_v=2, NoPE, gated attention) | 4 | 0.0004 | finished | 64.943 | 2.152382 | 0 reset(s); 1 duplicate token sample(s) removed | [dzffl1jy](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/dzffl1jy) |
| 810m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 4 | 0.0004 | finished | 60.942 | 2.191179 | 0 reset(s); 1 duplicate token sample(s) removed | [l0u9gv52](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/l0u9gv52) |
| 810m | wide integration (SWA) | 4 | 0.0004 | finished | 55.610 | 2.192802 | 0 reset(s); 1 duplicate token sample(s) removed | [58ftjxmw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/58ftjxmw) |
| 810m | geometry-matched hybrid (GDN, expand_v=2, NoPE, gated attention) | 8 | 0.0004 | finished | 121.884 | 2.114516 | 0 reset(s); 1 duplicate token sample(s) removed | [pvoq0dq6](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/pvoq0dq6) |
| 810m | wide integration (SWA) | 8 | 0.0004 | finished | 111.220 | 2.104939 | 0 reset(s); 1 duplicate token sample(s) removed | [kyti8h1y](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kyti8h1y) |
