# GDN1 vs GDN2 vs KDA — our settings

Generated: `2026-07-31T21:38:49.686125+00:00`

Selection metric: final `250M`-token mean training CE. Only finished runs are eligible.
The optimal-LR summary includes only bracketed 275M sweeps with a valid quadratic fit.
The all-size figure uses observed-optimal points for 275M and fixed wide-LR transfer points for larger sizes; pending hybrid cells are labeled explicitly.
Fitted LR minima in the 275M U-plot are visual aids and are never used to select results.

## Completed results

| Model | Cx | Mode | Status | References (loss @ LR) | Intervention (loss @ LR) | Deltas vs references |
|---|---:|---|---|---|---:|---|
| 275m | Cx1 | fixed-LR transfer | finished | geometry_gdn_ev2_nope_gated: 2.711104 @ 0.0008; geometry_gdn2_ev2_nope_gated: 2.646730 @ 0.0016 | 2.692695 (0.0008) | geometry_gdn_ev2_nope_gated: -0.018409; geometry_gdn2_ev2_nope_gated: +0.045965 |
| 275m | Cx2 | fixed-LR transfer | finished | geometry_gdn_ev2_nope_gated: 2.580768 @ 0.0016; geometry_gdn2_ev2_nope_gated: 2.534116 @ 0.0016 | 2.562520 (0.0016) | geometry_gdn_ev2_nope_gated: -0.018248; geometry_gdn2_ev2_nope_gated: +0.028404 |
| 275m | Cx4 | fixed-LR transfer | finished | geometry_gdn_ev2_nope_gated: 2.476065 @ 0.0008; geometry_gdn2_ev2_nope_gated: 2.443132 @ 0.0016 | 2.464247 (0.0008) | geometry_gdn_ev2_nope_gated: -0.011817; geometry_gdn2_ev2_nope_gated: +0.021115 |
| 275m | Cx8 | fixed-LR transfer | finished | geometry_gdn_ev2_nope_gated: 2.390397 @ 0.0008; geometry_gdn2_ev2_nope_gated: 2.356985 @ 0.0008 | 2.380273 (0.0008) | geometry_gdn_ev2_nope_gated: -0.010124; geometry_gdn2_ev2_nope_gated: +0.023288 |
| 480m | Cx1 | fixed-LR transfer | finished | geometry_gdn_ev2_nope_gated: 2.519642 @ 0.0012; geometry_gdn2_ev2_nope_gated: 2.468555 @ 0.0012 | 2.492283 (0.0012) | geometry_gdn_ev2_nope_gated: -0.027360; geometry_gdn2_ev2_nope_gated: +0.023727 |
| 480m | Cx2 | fixed-LR transfer | finished | geometry_gdn_ev2_nope_gated: 2.414718 @ 0.0009; geometry_gdn2_ev2_nope_gated: 2.359149 @ 0.0009 | 2.382695 (0.0009) | geometry_gdn_ev2_nope_gated: -0.032023; geometry_gdn2_ev2_nope_gated: +0.023547 |
| 480m | Cx4 | fixed-LR transfer | finished | geometry_gdn_ev2_nope_gated: 2.315124 @ 0.0008; geometry_gdn2_ev2_nope_gated: 2.276454 @ 0.0008 | 2.291179 (0.0008) | geometry_gdn_ev2_nope_gated: -0.023945; geometry_gdn2_ev2_nope_gated: +0.014725 |
| 480m | Cx8 | fixed-LR transfer | finished | geometry_gdn_ev2_nope_gated: 2.239297 @ 0.0008; geometry_gdn2_ev2_nope_gated: 2.204316 @ 0.0008 | 2.216501 (0.0008) | geometry_gdn_ev2_nope_gated: -0.022797; geometry_gdn2_ev2_nope_gated: +0.012185 |
| 810m | Cx1 | fixed-LR transfer | finished | geometry_gdn_ev2_nope_gated: 2.373592 @ 0.0006; geometry_gdn2_ev2_nope_gated: 2.323505 @ 0.0006 | 2.352304 (0.0006) | geometry_gdn_ev2_nope_gated: -0.021289; geometry_gdn2_ev2_nope_gated: +0.028799 |
| 810m | Cx2 | fixed-LR transfer | finished | geometry_gdn_ev2_nope_gated: 2.277253 @ 0.00056; geometry_gdn2_ev2_nope_gated: — | 2.241873 (0.00056) | geometry_gdn_ev2_nope_gated: -0.035380; geometry_gdn2_ev2_nope_gated: — |
| 810m | Cx4 | fixed-LR transfer | finished | geometry_gdn_ev2_nope_gated: 2.191179 @ 0.0004; geometry_gdn2_ev2_nope_gated: 2.152382 @ 0.0004 | 2.158207 (0.0004) | geometry_gdn_ev2_nope_gated: -0.032972; geometry_gdn2_ev2_nope_gated: +0.005825 |
| 810m | Cx8 | fixed-LR transfer | finished | geometry_gdn_ev2_nope_gated: 2.114516 @ 0.0004; geometry_gdn2_ev2_nope_gated: — | 2.090719 (0.0004) | geometry_gdn_ev2_nope_gated: -0.023797; geometry_gdn2_ev2_nope_gated: — |
| 1p2b | Cx1 | fixed-LR transfer | finished | geometry_gdn_ev2_nope_gated: 2.273007 @ 0.0004; geometry_gdn2_ev2_nope_gated: — | 2.236574 (0.0004) | geometry_gdn_ev2_nope_gated: -0.036433; geometry_gdn2_ev2_nope_gated: — |
| 1p2b | Cx2 | fixed-LR transfer | finished | geometry_gdn_ev2_nope_gated: 2.188236 @ 0.0006; geometry_gdn2_ev2_nope_gated: — | 2.146299 (0.0006) | geometry_gdn_ev2_nope_gated: -0.041938; geometry_gdn2_ev2_nope_gated: — |
| 1p2b | Cx4 | fixed-LR transfer | finished | geometry_gdn_ev2_nope_gated: 2.108263 @ 0.0003; geometry_gdn2_ev2_nope_gated: — | 2.067653 (0.0003) | geometry_gdn_ev2_nope_gated: -0.040610; geometry_gdn2_ev2_nope_gated: — |
| 1p2b | Cx8 | fixed-LR transfer | finished | geometry_gdn_ev2_nope_gated: 2.037147 @ 0.0004; geometry_gdn2_ev2_nope_gated: — | 1.999736 (0.0004) | geometry_gdn_ev2_nope_gated: -0.037411; geometry_gdn2_ev2_nope_gated: — |

## Runs

| Model | Variant | Cx | LR | State | Tokens (B) | Final-window CE | Replay handling | W&B |
|---|---|---:|---:|---|---:|---:|---|---|
| 275m | GDN2 (our settings) | 1 | 0.0004 | finished | 4.839 | 2.676958 | 0 reset(s); 1 duplicate token sample(s) removed | [rsxmn720](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rsxmn720) |
| 275m | GDN2 (our settings) | 1 | 0.0008 | finished | 4.839 | 2.657052 | 0 reset(s); 1 duplicate token sample(s) removed | [pqrdvu63](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/pqrdvu63) |
| 275m | GDN2 (our settings) | 1 | 0.0016 | finished | 4.839 | 2.646730 | 0 reset(s); 1 duplicate token sample(s) removed | [5uzr9dva](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5uzr9dva) |
| 275m | GDN2 (our settings) | 1 | 0.0032 | finished | 4.839 | 2.661486 | 0 reset(s); 1 duplicate token sample(s) removed | [j2t5c2jb](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/j2t5c2jb) |
| 275m | GDN1 (our settings) | 1 | 0.0004 | finished | 4.557 | 2.724486 | 0 reset(s); 1 duplicate token sample(s) removed | [lg619wiz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/lg619wiz) |
| 275m | GDN1 (our settings) | 1 | 0.0008 | finished | 4.557 | 2.711104 | 0 reset(s); 1 duplicate token sample(s) removed | [q81uxrxu](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/q81uxrxu) |
| 275m | GDN1 (our settings) | 1 | 0.0016 | finished | 4.557 | 2.715845 | 0 reset(s); 1 duplicate token sample(s) removed | [sxuuwzzm](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sxuuwzzm) |
| 275m | GDN1 (our settings) | 1 | 0.0032 | finished | 4.557 | 2.730489 | 0 reset(s); 1 duplicate token sample(s) removed | [1pr3blts](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1pr3blts) |
| 275m | KDA (our settings) | 1 | 0.0008 | finished | 4.526 | 2.692695 | — | [75dy08n9](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/75dy08n9) |
| 275m | GDN2 (our settings) | 2 | 0.0004 | finished | 9.679 | 2.564391 | 0 reset(s); 1 duplicate token sample(s) removed | [7yh4rfi1](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7yh4rfi1) |
| 275m | GDN2 (our settings) | 2 | 0.0008 | finished | 9.679 | 2.544136 | 0 reset(s); 1 duplicate token sample(s) removed | [2egeqyvo](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/2egeqyvo) |
| 275m | GDN2 (our settings) | 2 | 0.0016 | finished | 9.679 | 2.534116 | 1 reset(s); 79 duplicate token sample(s) removed | [gat5rtub](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/gat5rtub) / [8agi9zte](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8agi9zte) / [jhcmk80f](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jhcmk80f) |
| 275m | GDN2 (our settings) | 2 | 0.0032 | finished | 9.679 | 2.548521 | 0 reset(s); 1 duplicate token sample(s) removed | [xwtxd1pv](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xwtxd1pv) |
| 275m | GDN1 (our settings) | 2 | 0.0004 | finished | 9.115 | 2.601429 | 0 reset(s); 1 duplicate token sample(s) removed | [sehjqtyk](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sehjqtyk) |
| 275m | GDN1 (our settings) | 2 | 0.0008 | finished | 9.115 | 2.584133 | 0 reset(s); 1 duplicate token sample(s) removed | [bttby9r8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/bttby9r8) |
| 275m | GDN1 (our settings) | 2 | 0.0016 | finished | 9.115 | 2.580768 | 0 reset(s); 1 duplicate token sample(s) removed | [ef4umox3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ef4umox3) |
| 275m | GDN1 (our settings) | 2 | 0.0032 | finished | 9.115 | 2.597651 | 0 reset(s); 1 duplicate token sample(s) removed | [ofodwbzz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ofodwbzz) |
| 275m | KDA (our settings) | 2 | 0.0016 | finished | 9.051 | 2.562520 | — | [ysswifrz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ysswifrz) |
| 275m | GDN2 (our settings) | 4 | 0.0004 | finished | 19.358 | 2.462028 | 0 reset(s); 1 duplicate token sample(s) removed | [6b0vighm](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6b0vighm) |
| 275m | GDN2 (our settings) | 4 | 0.0008 | finished | 19.358 | 2.446822 | 0 reset(s); 1 duplicate token sample(s) removed | [yq4mi5o0](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/yq4mi5o0) |
| 275m | GDN2 (our settings) | 4 | 0.0016 | finished | 19.358 | 2.443132 | 0 reset(s); 1 duplicate token sample(s) removed | [0w6ezwgx](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0w6ezwgx) |
| 275m | GDN2 (our settings) | 4 | 0.0032 | finished | 19.358 | 2.461867 | 0 reset(s); 1 duplicate token sample(s) removed | [kcig30ty](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kcig30ty) |
| 275m | GDN1 (our settings) | 4 | 0.0004 | finished | 18.229 | 2.491557 | 0 reset(s); 1 duplicate token sample(s) removed | [fh9tl31v](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/fh9tl31v) |
| 275m | GDN1 (our settings) | 4 | 0.0008 | finished | 18.229 | 2.476065 | 0 reset(s); 1 duplicate token sample(s) removed | [gwzx0ekc](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/gwzx0ekc) |
| 275m | GDN1 (our settings) | 4 | 0.0016 | finished | 18.229 | 2.476188 | 0 reset(s); 1 duplicate token sample(s) removed | [jr74v01c](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jr74v01c) |
| 275m | GDN1 (our settings) | 4 | 0.0032 | finished | 18.229 | 2.496595 | 0 reset(s); 1 duplicate token sample(s) removed | [2s5s1yw0](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/2s5s1yw0) |
| 275m | KDA (our settings) | 4 | 0.0008 | finished | 18.103 | 2.464247 | 0 reset(s); 1 duplicate token sample(s) removed | [cyyeyven](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/cyyeyven) |
| 275m | GDN2 (our settings) | 8 | 0.0004 | finished | 38.715 | 2.372393 | 0 reset(s); 1 duplicate token sample(s) removed | [jewjx6yq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jewjx6yq) |
| 275m | GDN2 (our settings) | 8 | 0.0008 | finished | 38.715 | 2.356985 | 0 reset(s); 1 duplicate token sample(s) removed | [1lpz9reu](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1lpz9reu) |
| 275m | GDN2 (our settings) | 8 | 0.0032 | finished | 38.715 | 2.380649 | 0 reset(s); 1 duplicate token sample(s) removed | [e6n5iscu](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/e6n5iscu) |
| 275m | GDN1 (our settings) | 8 | 0.0004 | finished | 36.459 | 2.405406 | 0 reset(s); 1 duplicate token sample(s) removed | [qehufcr5](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qehufcr5) |
| 275m | GDN1 (our settings) | 8 | 0.0008 | finished | 36.459 | 2.390397 | 0 reset(s); 1 duplicate token sample(s) removed | [ouxblu4g](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ouxblu4g) |
| 275m | GDN1 (our settings) | 8 | 0.0016 | finished | 36.459 | 2.392762 | 0 reset(s); 1 duplicate token sample(s) removed | [3xjjt5sa](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3xjjt5sa) |
| 275m | GDN1 (our settings) | 8 | 0.0032 | finished | 36.459 | 2.413536 | 0 reset(s); 1 duplicate token sample(s) removed | [mbvin02a](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mbvin02a) |
| 275m | KDA (our settings) | 8 | 0.0008 | finished | 36.205 | 2.380273 | 0 reset(s); 1 duplicate token sample(s) removed | [vrjssy6q](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vrjssy6q) |
| 480m | GDN2 (our settings) | 1 | 0.0012 | finished | 8.998 | 2.468555 | 0 reset(s); 1 duplicate token sample(s) removed | [6r2blwru](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6r2blwru) |
| 480m | GDN1 (our settings) | 1 | 0.0012 | finished | 8.529 | 2.519642 | 0 reset(s); 1 duplicate token sample(s) removed | [9rltp47w](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9rltp47w) |
| 480m | KDA (our settings) | 1 | 0.0012 | finished | 8.433 | 2.492283 | — | [sb4yqi8x](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sb4yqi8x) |
| 480m | GDN2 (our settings) | 2 | 0.0009 | finished | 17.997 | 2.359149 | 0 reset(s); 1 duplicate token sample(s) removed | [07rx8ez4](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/07rx8ez4) |
| 480m | GDN1 (our settings) | 2 | 0.0009 | finished | 17.057 | 2.414718 | 0 reset(s); 1 duplicate token sample(s) removed | [0crj05wz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0crj05wz) |
| 480m | KDA (our settings) | 2 | 0.0009 | finished | 16.867 | 2.382695 | 0 reset(s); 1 duplicate token sample(s) removed | [k2zf4esa](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/k2zf4esa) |
| 480m | GDN2 (our settings) | 4 | 0.0008 | finished | 35.993 | 2.276454 | 0 reset(s); 1 duplicate token sample(s) removed | [9u4z0e36](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9u4z0e36) |
| 480m | GDN1 (our settings) | 4 | 0.0008 | finished | 34.114 | 2.315124 | 0 reset(s); 1 duplicate token sample(s) removed | [ur7yonej](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ur7yonej) |
| 480m | KDA (our settings) | 4 | 0.0008 | finished | 33.734 | 2.291179 | 0 reset(s); 1 duplicate token sample(s) removed | [lesw7ogm](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/lesw7ogm) |
| 480m | GDN2 (our settings) | 8 | 0.0008 | finished | 71.986 | 2.204316 | 0 reset(s); 1 duplicate token sample(s) removed | [7p8q3v6p](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7p8q3v6p) |
| 480m | GDN1 (our settings) | 8 | 0.0008 | finished | 68.228 | 2.239297 | 0 reset(s); 1 duplicate token sample(s) removed | [4737op7s](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4737op7s) |
| 480m | KDA (our settings) | 8 | 0.0008 | finished | 67.468 | 2.216501 | 0 reset(s); 1 duplicate token sample(s) removed | [nfgfhyv8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/nfgfhyv8) |
| 810m | GDN2 (our settings) | 1 | 0.0006 | finished | 16.236 | 2.323505 | 0 reset(s); 1 duplicate token sample(s) removed | [3wukvwyl](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3wukvwyl) |
| 810m | GDN1 (our settings) | 1 | 0.0006 | finished | 15.236 | 2.373592 | 0 reset(s); 1 duplicate token sample(s) removed | [027xoq0r](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/027xoq0r) |
| 810m | KDA (our settings) | 1 | 0.0006 | finished | 14.730 | 2.352304 | 0 reset(s); 1 duplicate token sample(s) removed | [4k5dasv8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4k5dasv8) |
| 810m | GDN1 (our settings) | 2 | 0.00056 | finished | 30.471 | 2.277253 | 0 reset(s); 1 duplicate token sample(s) removed | [7ryj4klm](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7ryj4klm) |
| 810m | KDA (our settings) | 2 | 0.00056 | finished | 29.459 | 2.241873 | 0 reset(s); 1 duplicate token sample(s) removed | [1e7z0xar](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1e7z0xar) |
| 810m | GDN2 (our settings) | 4 | 0.0004 | finished | 64.943 | 2.152382 | 0 reset(s); 1 duplicate token sample(s) removed | [dzffl1jy](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/dzffl1jy) |
| 810m | GDN1 (our settings) | 4 | 0.0004 | finished | 60.942 | 2.191179 | 0 reset(s); 1 duplicate token sample(s) removed | [l0u9gv52](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/l0u9gv52) |
| 810m | KDA (our settings) | 4 | 0.0004 | finished | 58.918 | 2.158207 | 0 reset(s); 1 duplicate token sample(s) removed | [gxgef1hf](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/gxgef1hf) |
| 810m | GDN1 (our settings) | 8 | 0.0004 | finished | 121.884 | 2.114516 | 0 reset(s); 1 duplicate token sample(s) removed | [pvoq0dq6](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/pvoq0dq6) |
| 810m | KDA (our settings) | 8 | 0.0004 | finished | 117.837 | 2.090719 | 0 reset(s); 1 duplicate token sample(s) removed | [x3nar750](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/x3nar750) |
| 1p2b | GDN1 (our settings) | 1 | 0.0004 | finished | 23.430 | 2.273007 | 0 reset(s); 1 duplicate token sample(s) removed | [ojtvkjgk](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ojtvkjgk) |
| 1p2b | KDA (our settings) | 1 | 0.0004 | finished | 22.460 | 2.236574 | 0 reset(s); 1 duplicate token sample(s) removed | [yuv9m7p1](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/yuv9m7p1) |
| 1p2b | GDN1 (our settings) | 2 | 0.0006 | finished | 46.859 | 2.188236 | 1 reset(s); 131 duplicate token sample(s) removed | [kko6fe0y](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kko6fe0y) / [ama4a8s0](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ama4a8s0) / [u88pp5vm](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/u88pp5vm) |
| 1p2b | KDA (our settings) | 2 | 0.0006 | finished | 44.921 | 2.146299 | 0 reset(s); 1 duplicate token sample(s) removed | [86nmqw4e](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/86nmqw4e) |
| 1p2b | GDN1 (our settings) | 4 | 0.0003 | finished | 93.719 | 2.108263 | 0 reset(s); 1 duplicate token sample(s) removed | [bhr5mgpr](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/bhr5mgpr) |
| 1p2b | KDA (our settings) | 4 | 0.0003 | finished | 89.841 | 2.067653 | — | [8j8vgags](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8j8vgags) |
| 1p2b | GDN1 (our settings) | 8 | 0.0004 | finished | 187.437 | 2.037147 | 0 reset(s); 1 duplicate token sample(s) removed | [z4zmtqmu](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/z4zmtqmu) |
| 1p2b | KDA (our settings) | 8 | 0.0004 | finished | 179.682 | 1.999736 | — | [shy5gj1d](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/shy5gj1d) |
