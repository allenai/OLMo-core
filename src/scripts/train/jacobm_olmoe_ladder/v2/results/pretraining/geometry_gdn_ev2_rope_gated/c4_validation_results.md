# C4 validation comparison for selected gated-RoPE checkpoints

Generated: `2026-07-24T16:07:25.794416+00:00`

Metric: `eval/lm/c4_en-validation/CE loss` (lower is better).

The checkpoint selection is identical to the final-250M training-loss comparison: 
observed-best learning rates at 275M and wide-optimal LR transfers at larger sizes.
Running, missing, and unregistered validation cells remain pending.

| Model | Cx | Variant | Selection | LR | C4 validation CE | Validation state | W&B |
|---|---:|---|---|---:|---:|---|---|
| 275m | Cx1 | wide integration (SWA) | observed_train_loss_optimum | 0.0016 | 3.337888 | finished | [h86x1nv3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h86x1nv3) |
| 275m | Cx1 | hybrid (GDN, expand_v=1) | observed_train_loss_optimum | 0.0016 | 3.289717 | finished | [78prltpm](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/78prltpm) |
| 275m | Cx1 | geometry-matched GDN (expand_v=2) | observed_train_loss_optimum | 0.0016 | 3.289807 | finished | [9dkxpsfi](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9dkxpsfi) |
| 275m | Cx1 | geometry-matched GDN + NoPE | observed_train_loss_optimum | 0.0008 | 3.294925 | finished | [udrlzxwa](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/udrlzxwa) |
| 275m | Cx1 | geometry-matched GDN + NoPE + gated attention | observed_train_loss_optimum | 0.0008 | 3.293680 | finished | [a3002e0a](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/a3002e0a) |
| 275m | Cx1 | geometry-matched GDN + RoPE + gated attention | observed_train_loss_optimum | 0.0016 | 3.278518 | finished | [6n5svfb0](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6n5svfb0) |
| 275m | Cx2 | wide integration (SWA) | observed_train_loss_optimum | 0.0016 | 3.219680 | finished | [6porpbo2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6porpbo2) |
| 275m | Cx2 | hybrid (GDN, expand_v=1) | observed_train_loss_optimum | 0.0016 | 3.172174 | finished | [gus9q59t](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/gus9q59t) |
| 275m | Cx2 | geometry-matched GDN (expand_v=2) | observed_train_loss_optimum | 0.0016 | 3.169047 | finished | [4d1unh96](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4d1unh96) |
| 275m | Cx2 | geometry-matched GDN + NoPE | observed_train_loss_optimum | 0.0016 | 3.175375 | finished | [nfxu0rce](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/nfxu0rce) |
| 275m | Cx2 | geometry-matched GDN + NoPE + gated attention | observed_train_loss_optimum | 0.0016 | 3.169805 | finished | [180zzyzj](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/180zzyzj) |
| 275m | Cx2 | geometry-matched GDN + RoPE + gated attention | observed_train_loss_optimum | 0.0016 | 3.164707 | finished | [qy00n7y9](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qy00n7y9) |
| 275m | Cx4 | wide integration (SWA) | observed_train_loss_optimum | 0.0008 | 3.119446 | finished | [9n3xk8gs](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9n3xk8gs) |
| 275m | Cx4 | hybrid (GDN, expand_v=1) | observed_train_loss_optimum | 0.0016 | 3.082691 | finished | [rt2pcpht](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rt2pcpht) |
| 275m | Cx4 | geometry-matched GDN (expand_v=2) | observed_train_loss_optimum | 0.0016 | 3.076744 | finished | [nflctkj5](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/nflctkj5) |
| 275m | Cx4 | geometry-matched GDN + NoPE | observed_train_loss_optimum | 0.0008 | 3.079560 | finished | [r4foxlv8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/r4foxlv8) |
| 275m | Cx4 | geometry-matched GDN + NoPE + gated attention | observed_train_loss_optimum | 0.0008 | 3.080632 | finished | [6zkh05bf](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6zkh05bf) |
| 275m | Cx4 | geometry-matched GDN + RoPE + gated attention | observed_train_loss_optimum | 0.0008 | 3.075725 | finished | [xheu6zbz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xheu6zbz) |
| 275m | Cx8 | wide integration (SWA) | observed_train_loss_optimum | 0.0008 | 3.031144 | finished | [qe052lo4](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qe052lo4) |
| 275m | Cx8 | hybrid (GDN, expand_v=1) | observed_train_loss_optimum | 0.0016 | 3.005974 | finished | [zig9h01l](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/zig9h01l) |
| 275m | Cx8 | geometry-matched GDN (expand_v=2) | observed_train_loss_optimum | 0.0008 | 3.000070 | finished | [3w6pv014](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3w6pv014) |
| 275m | Cx8 | geometry-matched GDN + NoPE | observed_train_loss_optimum | 0.0008 | 3.000840 | finished | [fmmwu409](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/fmmwu409) |
| 275m | Cx8 | geometry-matched GDN + NoPE + gated attention | observed_train_loss_optimum | 0.0008 | 2.999882 | finished | [v8x3b6to](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/v8x3b6to) |
| 275m | Cx8 | geometry-matched GDN + RoPE + gated attention | observed_train_loss_optimum | 0.0016 | 2.994544 | finished | [plgdh22w](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/plgdh22w) |
| 480m | Cx1 | wide integration (SWA) | wide_lr_transfer | 0.0012 | 3.144330 | finished | [z4wxvc6h](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/z4wxvc6h) |
| 480m | Cx1 | hybrid (GDN, expand_v=1) | wide_lr_transfer | 0.0012 | — | not_registered | — |
| 480m | Cx1 | geometry-matched GDN (expand_v=2) | wide_lr_transfer | — | — | training_result_missing | — |
| 480m | Cx1 | geometry-matched GDN + NoPE | wide_lr_transfer | 0.0012 | 3.119133 | finished | [zpbio5dd](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/zpbio5dd) |
| 480m | Cx1 | geometry-matched GDN + NoPE + gated attention | wide_lr_transfer | 0.0012 | 3.108574 | finished | [s0g3a2dc](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/s0g3a2dc) |
| 480m | Cx1 | geometry-matched GDN + RoPE + gated attention | wide_lr_transfer | 0.0012 | 3.097183 | finished | [3g1z4mvc](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3g1z4mvc) |
| 480m | Cx2 | wide integration (SWA) | wide_lr_transfer | 0.0009 | 3.030859 | finished | [ywj13bkw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ywj13bkw) |
| 480m | Cx2 | hybrid (GDN, expand_v=1) | wide_lr_transfer | 0.0009 | — | not_registered | — |
| 480m | Cx2 | geometry-matched GDN (expand_v=2) | wide_lr_transfer | — | — | training_result_missing | — |
| 480m | Cx2 | geometry-matched GDN + NoPE | wide_lr_transfer | 0.0009 | 3.010991 | finished | [b3qim2gy](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/b3qim2gy) |
| 480m | Cx2 | geometry-matched GDN + NoPE + gated attention | wide_lr_transfer | 0.0009 | 3.004941 | finished | [6p6c5s63](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6p6c5s63) |
| 480m | Cx2 | geometry-matched GDN + RoPE + gated attention | wide_lr_transfer | 0.0009 | 2.994537 | finished | [1tntp2ns](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1tntp2ns) |
| 480m | Cx4 | wide integration (SWA) | wide_lr_transfer | 0.0008 | 2.941732 | finished | [rblv9hpr](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rblv9hpr) |
| 480m | Cx4 | hybrid (GDN, expand_v=1) | wide_lr_transfer | 0.0008 | 2.915676 | finished | [tl05mpku](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/tl05mpku) |
| 480m | Cx4 | geometry-matched GDN (expand_v=2) | wide_lr_transfer | — | — | training_result_missing | — |
| 480m | Cx4 | geometry-matched GDN + NoPE | wide_lr_transfer | 0.0008 | 2.917672 | finished | [fp741l2q](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/fp741l2q) |
| 480m | Cx4 | geometry-matched GDN + NoPE + gated attention | wide_lr_transfer | 0.0008 | 2.912261 | finished | [ztlwq68v](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ztlwq68v) |
| 480m | Cx4 | geometry-matched GDN + RoPE + gated attention | wide_lr_transfer | 0.0008 | 2.904977 | finished | [rzl1kcwq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rzl1kcwq) |
| 480m | Cx8 | wide integration (SWA) | wide_lr_transfer | 0.0008 | 2.868272 | finished | [vdcrgfy0](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vdcrgfy0) |
| 480m | Cx8 | hybrid (GDN, expand_v=1) | wide_lr_transfer | 0.0008 | 2.841840 | finished | [ou5web1c](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ou5web1c) |
| 480m | Cx8 | geometry-matched GDN (expand_v=2) | wide_lr_transfer | — | — | training_result_missing | — |
| 480m | Cx8 | geometry-matched GDN + NoPE | wide_lr_transfer | 0.0008 | 2.843065 | finished | [lp0xmebw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/lp0xmebw) |
| 480m | Cx8 | geometry-matched GDN + NoPE + gated attention | wide_lr_transfer | 0.0008 | 2.837496 | finished | [1f52ml11](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1f52ml11) |
| 480m | Cx8 | geometry-matched GDN + RoPE + gated attention | wide_lr_transfer | 0.0008 | 2.832544 | finished | [1toyuv66](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1toyuv66) |
| 810m | Cx1 | wide integration (SWA) | wide_lr_transfer | 0.0006 | 2.978391 | finished | [w912irkq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/w912irkq) |
| 810m | Cx1 | hybrid (GDN, expand_v=1) | wide_lr_transfer | 0.0006 | — | not_registered | — |
| 810m | Cx1 | geometry-matched GDN (expand_v=2) | wide_lr_transfer | — | — | training_result_missing | — |
| 810m | Cx1 | geometry-matched GDN + NoPE | wide_lr_transfer | 0.0006 | 2.968621 | finished | [3nq5my0d](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3nq5my0d) |
| 810m | Cx1 | geometry-matched GDN + NoPE + gated attention | wide_lr_transfer | 0.0006 | 2.962871 | finished | [m9rev3wx](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/m9rev3wx) |
| 810m | Cx1 | geometry-matched GDN + RoPE + gated attention | wide_lr_transfer | 0.0006 | 2.955805 | finished | [e7sak9iz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/e7sak9iz) |
| 810m | Cx2 | wide integration (SWA) | wide_lr_transfer | 0.00056 | 2.875221 | finished | [jpbqhfvc](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jpbqhfvc) |
| 810m | Cx2 | hybrid (GDN, expand_v=1) | wide_lr_transfer | 0.00056 | — | not_registered | — |
| 810m | Cx2 | geometry-matched GDN (expand_v=2) | wide_lr_transfer | — | — | training_result_missing | — |
| 810m | Cx2 | geometry-matched GDN + NoPE | wide_lr_transfer | 0.00056 | 2.870854 | finished | [vk3rix1e](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vk3rix1e) |
| 810m | Cx2 | geometry-matched GDN + NoPE + gated attention | wide_lr_transfer | 0.00056 | 2.868930 | finished | [atdeng8o](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/atdeng8o) |
| 810m | Cx2 | geometry-matched GDN + RoPE + gated attention | wide_lr_transfer | 0.00056 | 2.856514 | finished | [wa1i6gzq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/wa1i6gzq) |
| 810m | Cx4 | wide integration (SWA) | wide_lr_transfer | 0.0004 | 2.798253 | finished | [58ftjxmw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/58ftjxmw) |
| 810m | Cx4 | hybrid (GDN, expand_v=1) | wide_lr_transfer | 0.0004 | 2.771512 | finished | [zqf1pkd3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/zqf1pkd3) |
| 810m | Cx4 | geometry-matched GDN (expand_v=2) | wide_lr_transfer | — | — | training_result_missing | — |
| 810m | Cx4 | geometry-matched GDN + NoPE | wide_lr_transfer | 0.0004 | 2.793456 | finished | [9kvsbcfz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9kvsbcfz) |
| 810m | Cx4 | geometry-matched GDN + NoPE + gated attention | wide_lr_transfer | 0.0004 | 2.791039 | finished | [1cvea8dv](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1cvea8dv) |
| 810m | Cx4 | geometry-matched GDN + RoPE + gated attention | wide_lr_transfer | 0.0004 | 2.791158 | finished | [7vfy6rtf](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7vfy6rtf) |
| 810m | Cx8 | wide integration (SWA) | wide_lr_transfer | 0.0004 | 2.726456 | finished | [kyti8h1y](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kyti8h1y) |
| 810m | Cx8 | hybrid (GDN, expand_v=1) | wide_lr_transfer | 0.0004 | 2.705398 | finished | [niu69ade](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/niu69ade) |
| 810m | Cx8 | geometry-matched GDN (expand_v=2) | wide_lr_transfer | — | — | training_result_missing | — |
| 810m | Cx8 | geometry-matched GDN + NoPE | wide_lr_transfer | 0.0004 | 2.722415 | finished | [a4xjd6x4](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/a4xjd6x4) |
| 810m | Cx8 | geometry-matched GDN + NoPE + gated attention | wide_lr_transfer | 0.0004 | — | not_registered | — |
| 810m | Cx8 | geometry-matched GDN + RoPE + gated attention | wide_lr_transfer | 0.0004 | — | not_registered | — |
| 1p2b | Cx1 | wide integration (SWA) | wide_lr_transfer | 0.0004 | 2.873525 | finished | [hww8eksq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/hww8eksq) |
| 1p2b | Cx1 | hybrid (GDN, expand_v=1) | wide_lr_transfer | 0.0004 | 2.848454 | finished | [nrkr62f1](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/nrkr62f1) |
| 1p2b | Cx1 | geometry-matched GDN (expand_v=2) | wide_lr_transfer | — | — | training_result_missing | — |
| 1p2b | Cx1 | geometry-matched GDN + NoPE | wide_lr_transfer | 0.0004 | 2.869484 | finished | [va83vuvo](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/va83vuvo) |
| 1p2b | Cx1 | geometry-matched GDN + NoPE + gated attention | wide_lr_transfer | 0.0004 | 2.868846 | finished | [4vj2v7nx](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4vj2v7nx) |
| 1p2b | Cx1 | geometry-matched GDN + RoPE + gated attention | wide_lr_transfer | 0.0004 | 2.865893 | finished | [yjs5dt2d](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/yjs5dt2d) |
| 1p2b | Cx2 | wide integration (SWA) | wide_lr_transfer | 0.0006 | 2.781616 | finished | [jfwntmwm](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jfwntmwm) |
| 1p2b | Cx2 | hybrid (GDN, expand_v=1) | wide_lr_transfer | 0.0006 | 2.760263 | finished | [3f780ayn](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3f780ayn) |
| 1p2b | Cx2 | geometry-matched GDN (expand_v=2) | wide_lr_transfer | — | — | training_result_missing | — |
| 1p2b | Cx2 | geometry-matched GDN + NoPE | wide_lr_transfer | 0.0006 | 2.784089 | finished | [sky6cm3m](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sky6cm3m) |
| 1p2b | Cx2 | geometry-matched GDN + NoPE + gated attention | wide_lr_transfer | 0.0006 | — | not_registered | — |
| 1p2b | Cx2 | geometry-matched GDN + RoPE + gated attention | wide_lr_transfer | — | — | training_result_missing | — |
| 1p2b | Cx4 | wide integration (SWA) | wide_lr_transfer | 0.0003 | 2.701380 | finished | [u7ab1tpb](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/u7ab1tpb) |
| 1p2b | Cx4 | hybrid (GDN, expand_v=1) | wide_lr_transfer | 0.0003 | 2.688314 | finished | [4thd7tl1](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4thd7tl1) |
| 1p2b | Cx4 | geometry-matched GDN (expand_v=2) | wide_lr_transfer | — | — | training_result_missing | — |
| 1p2b | Cx4 | geometry-matched GDN + NoPE | wide_lr_transfer | 0.0003 | — | not_registered | — |
| 1p2b | Cx4 | geometry-matched GDN + NoPE + gated attention | wide_lr_transfer | 0.0003 | — | not_registered | — |
| 1p2b | Cx4 | geometry-matched GDN + RoPE + gated attention | wide_lr_transfer | 0.0003 | 2.707786 | finished | [h7n70dsq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h7n70dsq) |
| 1p2b | Cx8 | wide integration (SWA) | wide_lr_transfer | 0.0004 | 2.637577 | finished | [bqjzmiqi](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/bqjzmiqi) |
| 1p2b | Cx8 | hybrid (GDN, expand_v=1) | wide_lr_transfer | 0.0004 | — | not_registered | — |
| 1p2b | Cx8 | geometry-matched GDN (expand_v=2) | wide_lr_transfer | — | — | training_result_missing | — |
| 1p2b | Cx8 | geometry-matched GDN + NoPE | wide_lr_transfer | 0.0004 | — | not_registered | — |
| 1p2b | Cx8 | geometry-matched GDN + NoPE + gated attention | wide_lr_transfer | 0.0004 | — | not_registered | — |
| 1p2b | Cx8 | geometry-matched GDN + RoPE + gated attention | wide_lr_transfer | 0.0004 | — | running | [g7ur9ibf](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/g7ur9ibf) |
