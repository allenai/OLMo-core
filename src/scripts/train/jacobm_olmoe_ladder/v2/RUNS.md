# OLMoE ladder v2 run ledger

Record post-migration experiment waves here. Per-run rows must include Beaker
job IDs and W&B IDs once they exist. Detailed migration-era DDP jobs remain in
[`../v1/DDP_RUNS.md`](../v1/DDP_RUNS.md).

## Rolling status ledger

Rows are updated as their corresponding collectors run; detailed sections
retain the full launch and retry history. The canonical GDN2/KDA, new KDA
transfer, MXFP8, LatentMoE, and validation rows were refreshed at
2026-07-29 15:49 UTC.

| Stage | Family / cell | State | Progress / result | Current W&B |
|---|---|---|---|---|
| pretraining | first hybrid 480M Cx4 | finished | final-250M CE `2.305996` | [h06m5ls2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h06m5ls2) |
| pretraining | first hybrid 480M Cx8 | finished | final-250M CE `2.236205` | [d34a9o4t](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/d34a9o4t) |
| pretraining | first hybrid 810M Cx4 | finished | final-250M CE `2.160440` | [kye1c19u](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kye1c19u) |
| pretraining | first hybrid 810M Cx8 | finished | final-250M CE `2.095585` | [s5gvyjiz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/s5gvyjiz) |
| pretraining | first hybrid 1.2B Cx1 | finished | final-250M CE `2.253953`; validation finished | [1d24xfx5](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1d24xfx5) |
| pretraining | first hybrid 1.2B Cx2 | finished | final-250M CE `2.163788` | [4k1bh4k2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4k1bh4k2) |
| pretraining | first hybrid 1.2B Cx4 | finished | final-250M CE `2.081180` | [vc3c6gj6](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vc3c6gj6) |
| pretraining | first hybrid 1.2B Cx8 | finished | final-250M CE `2.016369`; all 16 scale cells complete | [7eemhu7g](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7eemhu7g) |
| pretraining | aligned geometry + NoPE 275M sweep | finished | 16/16; observed best LR is `8e-4`, `1.6e-3`, `8e-4`, `8e-4` at Cx1/2/4/8 | [results](results/pretraining/geometry_gdn_ev2_nope/results.md) |
| pretraining | aligned geometry + NoPE + gated attention 275M sweep | finished | 16/16; observed best LR is `8e-4`, `1.6e-3`, `8e-4`, `8e-4` at Cx1/2/4/8 | [results](results/pretraining/geometry_gdn_ev2_nope_gated/results.md) |
| pretraining | aligned geometry + RoPE + gated attention 275M sweep | finished | 16/16; observed best LR is `1.6e-3`, `1.6e-3`, `8e-4`, `1.6e-3` at Cx1/2/4/8 | [results](results/pretraining/geometry_gdn_ev2_rope_gated/results.md) |
| pretraining | larger aligned geometry + NoPE | finished | 12/12 formal cells; clean 1.2B Cx8 reproduction finished with final-250M CE `2.034305` | [results](results/pretraining/geometry_gdn_ev2_nope/results.md) |
| pretraining | larger aligned geometry + NoPE + gated attention | finished | 12/12; newly finished 1.2B Cx2 strict final-250M CE `2.188236` | [results](results/pretraining/geometry_gdn_ev2_nope_gated/results.md) |
| pretraining | larger aligned geometry + RoPE + gated attention | 11 finished / 1 failed | newly collected strict final-250M CE: 810M Cx8 `2.104806`, 1.2B Cx8 `2.029514`; 1.2B Cx2 remains failed | [results](results/pretraining/geometry_gdn_ev2_rope_gated/results.md) |
| pretraining | 275M geometry + NoPE + gated attention + original GDN2 | 15/16 formal cells finished; clean independent reproduction finished | The original Cx8 `1.6e-3` trajectory remains a deterministic failure at step 36,768, but its distinct `-fresh-r2` reproduction completed | [results](results/pretraining/geometry_gdn2_ev2_nope_gated/results.md) |
| pretraining | 275M GDN2 stability 2x2 | 3/3 finished | Fresh Cx8/LR `1.6e-3` ev1+negative, ev2+nonnegative, and canonical ev1+nonnegative controls all completed | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYBVM2N2D3DM67S8HWARJP6C) |
| pretraining | canonical GDN2 (`expand_v=1`, nonnegative) 275M sweep | 16/16 finished | All four Cx curves are complete and bracketed; observed-best LR is `1.6e-3` at every Cx | [results](results/pretraining/canonical_gdn2_kda/results.md) |
| pretraining | canonical KDA 275M sweep | 16/16 finished | All four Cx curves are complete and bracketed; observed-best LR is `1.6e-3` at every Cx | [results](results/pretraining/canonical_gdn2_kda/results.md) |
| pretraining | canonical GDN2 larger-scale transfer | 10 loss-collected / 2 failed | 1.2B Cx8 local W&B history was recovered through the verified final step; strict final-250M CE is `2.020529`; 480M Cx2 and 1.2B Cx2 remain failed | [results](results/pretraining/canonical_gdn2_kda/scale_results.md) |
| pretraining | canonical KDA 480M stability transfer | 4/4 finished | Cx1/2/4/8 strict final-250M CEs are `2.517826`, `2.412884`, `2.323228`, and `2.237558` | [launches](#canonical-kda-480m-stability-transfer) |
| pretraining | KDA `expand_v=2`, negative-eigenvalue transfer (275M/480M/810M/1.2B) | 14 finished / 2 running | 810M is 4/4; 1.2B Cx1/Cx2 finished at strict final-250M CE `2.236574`/`2.146299`. The distributed Cx4/Cx8 continuations are healthy at steps `146,270/171,359` (85.4%, ETA 6h08m) and `183,000/228,478` (80.1%, ETA 8h53m), respectively. No failures in this KDA family. | [results](results/pretraining/canonical_gdn2_kda/kda_ev2_neg_scale_results.md) / [launches](#kda-expand_v2-negative-eigenvalue-transfer) |
| qualification | LatentMoE PR #799, full-width-router controls, and 275M parameter-matched configs | 1,000-expert EP1 replacement qualified | Paper-matched 2× and exact 4× are 295.664M/296.770M active params. The selected 4× EP1 approximation uses exactly 1,000 experts/top-32 and is 296.632M active / 3.073B stored. Its exact-2-Mi result is 180.0K TPS/GPU and 280.4 TFLOPs/GPU; physical-max MB11 reaches 182.9K/284.9. Both 50-step runs have zero skips. | [plan and results](LATENT_MOE.md) / [capacity](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYP96YAGZ4CZCZBT87NHSA5X) / [throughput](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYPA1GYA09H78GCVT3PQ1TZH) |
| pretraining | LatentMoE 275M LR sweeps | 16/32 finished; 16 Cx4/Cx8 cells submitted | Both L2 and the 1,000-expert L4 replacement have bracketed Cx1/Cx2 curves selecting `1.6e-3`. Cx4 is allocated (`4h`, 32 GPUs total); Cx8 is unallocated (`0m`, 64 GPUs total). | [results](results/pretraining/kda_latent_moe/results.md) / [Cx4 work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYQS79P3JFX4SSB4AM7B8CDK) / [Cx8 work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYQS7J3NBH1TFX6Y1FGQ7WM2) |
| pretraining | aggressive MXFP8 KDA 275M LR sweep | 16/16 finished | All Cx curves are complete and bracketed. Observed best is `1.6e-3` throughout, with strict final-250M CE `2.685399`, `2.566948`, `2.463998`, and `2.383008` at Cx1/2/4/8. | [results](results/pretraining/kda_mxfp8/results.md) / [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYJPTZ3J4VHGBH0FSVAQRDGC) |
| pretraining | aggressive MXFP8 KDA 480M transferred-LR continuation | 1 finished / 3 intentionally canceled | Cx1 finished at strict final-250M CE `2.497288`, `+0.005005` versus matching BF16 KDA; Cx2/Cx4/Cx8 were paused after the matched throughput audit found a 42--62% regression; durable checkpoints retained | [results](results/pretraining/kda_mxfp8/results.md) / [launches](#480m-aggressive-mxfp8-kda-transferred-lr-continuation) |
| throughput | aggressive MXFP8 KDA 480M qualification | 2/2 finished | MB8/8-GPU: 192.7 TFLOPs/GPU, 69.8K TPS/GPU, 146.6/193.2 GiB active/reserved; MB6/16-GPU: 132.5 TFLOPs/GPU, 48.0K TPS/GPU, 111.1/155.7 GiB; zero skipped steps | [one node](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYKK27K7N1MVV702AKJ2DST8) / [two nodes](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYKK2A58MK03ANV6GAAZPAVG) |
| diagnostic | GDN2 production-shape PyTorch reference 2x2 | finished | All four `expand_v`/negative-eigenvalue cells passed forward, final-state, backward, packed-document, and recompute/retain comparisons | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYBY8DXT5BVM85WYKAT5TXQN) |
| diagnostic | Matched KDA/GDN2 numerical audit | finished | All 40 one/four-chunk output/state comparisons passed; GDN2 is broadly KDA-like, with localized 3.80% `A_log` relative-L2 error at T256/V256/negative eigvals | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYBZX8MHJ611ZSJD43SYS9HZ) / [results](results/diagnostics/matched_kda_gdn2_numerics.md) |
| diagnostic | Actual FLA `v0.5.2` GDN2 release | qualified; original replay matrix complete | Release commit `9c8e42e` passes the reference suite, but 4/6 reliably failing original checkpoints reproduce exactly; all four reproduced failures originate in GDN2 forward. The release is not a general fix. | [qualification](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYJSQRXFW1XH1Y1EQPEXVGM6) / [matrix](GDN2.md#fla-v052-release-qualification) |
| diagnostic | KDA reference + 50-step MB16 qualification | finished | Reference/packed checks passed; zero skipped steps; steady-state 404.7 TFLOPs/GPU and 290.5K TPS on one B300 | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYBX6WX46F9B3HV3W59G368R) / [3s14s676](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3s14s676) |
| throughput | 275M mixed 3-KDA/2-SWA/1-FA | finished | one B300, 2 Mi batch, MB16: 258.3K TPS/GPU, 412.2 TFLOPs/GPU, 243.4/247.6 GiB active/reserved; zero skipped steps; +1.1% TPS versus the matched KDA parent | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYQEMRRW27EDVEZ9VEVYHEPP) / [results](results/throughput/275m_kda_mixed6_single_gpu.csv) |
| throughput | 275M mixed 3-KDA/2-SWA/1-FA + paper-matched LatentMoE L=2 | finished | one B300, exact 2 Mi, MB8: 215.2K TPS/GPU, 351.55 TFLOPs/GPU, 167.5/169.2 GiB active/reserved; zero skipped steps; +0.25% TPS versus ordinary KDA L=2 at the identical protocol | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYQJT9HQ7F3P4WNR6NP60PCS) / [results](results/throughput/275m_kda_mixed6_latent2x_single_gpu.csv) |
| throughput | 275M 10-layer mixed-attention depth controls | 2/2 finished | non-latent MB16: 285.1K TPS/GPU and 447.40 TFLOPs/GPU; L=2 MB8: 232.8K and 372.55; exact 2 Mi on one B300, zero skips; +10.36%/+8.20% TPS versus respective 12-layer parents | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYQP504SNRMKGXZ7XSRJMFCV) / [results](results/throughput/275m_kda_mixed10_single_gpu.csv) |
| pretraining | larger geometry + NoPE + gated attention + original GDN2 | 6 finished / 5 numerical failures / 1 canceled | 480M is 4/4; 810M Cx1/Cx4 finished while Cx2/Cx8 failed; 1.2B Cx1/Cx2/Cx8 failed and Cx4 is stopped | [results](results/pretraining/geometry_gdn2_ev2_nope_gated/results.md) |
| pretraining | actual-FLA-v0.5.2 fresh original-GDN2 retrains | stopped | 810M Cx2 and 1.2B Cx1 failed numerically; the remaining 810M Cx1 was manually canceled at step 14,660 to avoid further compute | [810M Cx1](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYK1NKNB5MTWC30J72A019WK) / [810M Cx2](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYK1NPRCEPJ2160GJ59HQ6XB) / [1.2B Cx1](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYK1NSRZ0RBFMGN51775TB7H) |
| throughput | 275M 1:1 10-layer SWA depth control | finished | one B300, 2 Mi batch, MB16: 578.75 TFLOPs/GPU and 365.8K TPS/GPU; zero skipped steps | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYADSYYRHPYQCRVWJ27KV4KQ) |
| throughput | 275M KDA 672-wide EP1 fused-attention/MXFP8 qualification | finished | 6/6 50-step cells, zero skips. BF16/fused-v2/FA4/attention-MXFP8 tie at 397.9--399.0 TFLOPs/GPU; expert MXFP8 is 357.1--358.6 and does not lower peak memory | [qualification](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYJK6WM1C9A8PQ273XYV22T4) / [grid](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYJM4P6ZDD3QQRC6HWR4PCPN) / [results](results/throughput/275m_kda_672_ep1_fa4_mxfp8.csv) |
| midtraining | first hybrid 275M Cx8 | finished | 100B; final checkpoint `step95368`; validation finished | [1keo2hz6](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1keo2hz6) |
| midtraining | first hybrid 480M Cx8 | finished | 100.001B; final checkpoint `step95368`; validation finished | [mnp9rv5l](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mnp9rv5l) |
| validation | V2 historical full-suite backfills | 117/117 registered targets complete | The consolidated historical registry remains complete; newly finished winner-only KDA, MXFP8, and LatentMoE checkpoints are deliberately outside this frozen full-suite registry. | [results](results/validation/hybrid_full.md) |
| validation | 1.2B EP1 fast-path qualification | finished | Exit code 0. The EP8-trained checkpoint loaded under EP1 with `moe_mesh=None`; LM validation plus the downstream `fast` suite completed in 1,745 seconds of evaluation time. | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYND6K86VHPBBARK4KSV9TZQ) / [p5fp6bc5](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/p5fp6bc5) |

The winner-only validation wave is capacity-deferred. The previous 48-ready /
five-training inventory is now stale: four new LatentMoE Cx1/Cx2 winners are
selected, the 275M MXFP8 sweep is complete, and only the 1.2B KDA Cx4/Cx8
winners remain in training. Before launching, rebuild the inventory and
de-duplicate it against the 117 historical completed targets. Keep the new
policy at EP1 with LM validation and the `fast` downstream suite.

At the 2026-07-29 15:49 UTC audit, the only live project training is the
distributed 1.2B KDA Cx4/Cx8 continuation: two eight-GPU Cx4 replicas and four
eight-GPU Cx8 replicas, 48 B300s total. All six replicas are running without a
failed attempt. Cx4 is at step `146,270/171,359` with an in-process ETA of
6h08m; Cx8 is at `183,000/228,478` with an ETA of 8h53m. Both LatentMoE
Cx1/Cx2 sweeps and the 1.2B EP1 fast-validation qualification finished. No
V1 training or evaluation job is live.

The formal pretraining results and plots use finished runs only and enforce a
complete final-250M-token history. The gated-RoPE sweep is now complete. Its
observed best final-250M CEs at Cx1/2/4/8 are `2.691980`, `2.573449`,
`2.470110`, and `2.386206`. It beats the ungated RoPE geometry, ungated NoPE,
and gated NoPE controls at every Cx. It also beats the first hybrid at Cx1,
Cx4, and Cx8; at Cx2 it is `0.003461` worse.

All six corrected Cx2 MB3 validation retries finished in
[01KY09D3D872K0R03NHF5MGYD4](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY09D3D872K0R03NHF5MGYD4).
The 32 NoPE/gated 275M backfills and both first-hybrid midtraining backfills
also finished. The consolidated validation export contains 117 finished
targets with 498 metrics each. All 16 gated-RoPE 275M backfills and nine
larger gated-RoPE backfills are complete. The 1.2B gated-RoPE Cx4 evaluator
completed its full suite and finalized W&B successfully; its Beaker worker
then exited 127 on a post-eval wrapper typo, so the evaluation itself does not
need a retry. The 1.2B gated-RoPE Cx8 target finished in W&B run `ved01fli`
after earlier scheduler-preempted attempts, and the separately registered
first-hybrid 810M Cx8 target finished in `niu69ade`. No v2 validation job is
currently live, and the historical full-suite collector is complete for all
117 registered targets.

On 2026-07-28, new validation work changed to observed training-loss winners
only, using the `fast` task set plus LM validation. Eval-only jobs use EP1 at
every size, including checkpoints trained with EP8; the legacy `sync_1d`
training layout must not be inherited. Historical `_full` manifests and their
117 completed results remain unchanged. The newer GDN2/KDA/MXFP8 winners are
not yet registered. The first systems qualification of that policy uses the
finished 1.2B gated-RoPE Cx1 winner and a distinct W&B name so it cannot be
confused with the historical EP8/full evaluator; it was submitted alone at
urgent priority in
[01KYND6K86VHPBBARK4KSV9TZQ](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYND6K86VHPBBARK4KSV9TZQ).
The live qualification built `dense_mesh(dp=8)` with `moe_mesh=None`, loaded
the checkpoint successfully, and advanced through LM-eval batches without an
OOM or runtime error; W&B run
[`p5fp6bc5`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/p5fp6bc5)
tracks the full result.

The gated-RoPE scale comparison now also has a C4 validation-CE view at
[`plots/pretraining/geometry_gdn_ev2_rope_gated/c4_validation_fixed_lr_scale_comparison.png`](plots/pretraining/geometry_gdn_ev2_rope_gated/c4_validation_fixed_lr_scale_comparison.png),
with the underlying values and coverage states in
[`results/pretraining/geometry_gdn_ev2_rope_gated/c4_validation_results.md`](results/pretraining/geometry_gdn_ev2_rope_gated/c4_validation_results.md).
It uses exactly the checkpoints selected by the training-loss plot and leaves
unfinished evaluation cells blank.

The first formal GDN2 results use the same strict final-250M-token statistic.
Cx1/Cx2/Cx4 are complete and bracketed. Their observed best results are
`2.646730 @ 1.6e-3`, `2.534116 @ 1.6e-3`, and `2.443132 @ 1.6e-3`, respectively.
The completed Cx2 value combines its failed and final W&B segments and retains
the full final 250M-token window. Cx8 has three finished points and remains
provisional because `1.6e-3` stopped on a non-finite loss at step 36,768; its
current observed best is `2.356985 @ 8e-4`. The scale plot now includes all
four 480M results plus 810M Cx1/Cx4. Their strict final-window CEs are
`2.468555`, `2.359149`, `2.276454`, and `2.204316` for 480M Cx1/2/4/8, and
`2.323505` / `2.152382` for 810M Cx1/Cx4. The GDN2 plots compare only against
wide and matching gated-NoPE GDN1, as intended.

The 11 finished larger gated-RoPE points have strict final-250M CEs of
`2.506239`, `2.402917`, `2.307792`, and `2.233177` for 480M Cx1/2/4/8;
`2.368164`, `2.266516`, `2.191042`, and `2.104806` for 810M Cx1/2/4/8; and
`2.270124`, `2.105145`, and `2.029514` for 1.2B Cx1/4/8.
The 1.2B Cx2 failure is a training-path numerical failure, not an OOM: the
optimizer asserted on a non-finite total gradient at step 5,420, auto-resumed
from durable `step5000`, and hit the same assertion again at step 6,582. The
later CUDA device assertion and NCCL watchdog messages are distributed
teardown effects. The 810M Cx4 cell had one identical assertion at step 29,107,
then auto-resumed and finished cleanly.

## Canonical GDN2 and KDA 275M LR sweeps

Submitted 2026-07-25 as eight urgent, unallocated Holmes works in Cx1, Cx2,
Cx4, Cx8 order. Each architecture/Cx work contains the four LRs `4e-4`,
`8e-4`, `1.6e-3`, and `3.2e-3`, except the new GDN2 Cx8 work, which omits
`1.6e-3` and reuses the already-running canonical stability job
`01KYBVM3KP7NRWB21ZFGNK2K75`. KDA uses FLA 0.4.1; GDN2 uses the isolated
pinned FLA 0.5.2 overlay. All jobs use EP1, compilation, rolling ephemeral
checkpoints, and no in-loop evals. All 32 planned W&B identities initialized
uniquely and are resolved by exact display name in the collector.

| Architecture | Cx | Work | Job IDs in LR order (`4e-4`, `8e-4`, `1.6e-3`, `3.2e-3`) | State at launch |
|---|---:|---|---|---|
| KDA | 1 | [01KYC1F615GDDN5Z9W48W072CQ](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYC1F615GDDN5Z9W48W072CQ) | `01KYC1F64PN9B31DPS8BFJRWP6`, `01KYC1F688KMC4N9SYC0JS4VV2`, `01KYC1F6BMPXZPF4ZKXQDDCQ1G`, `01KYC1F6EWY0EMM8S318DC4F2Z` | scheduled |
| GDN2 | 1 | [01KYC1F7M7WWPG9TYV0F019ERB](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYC1F7M7WWPG9TYV0F019ERB) | `01KYC1F7RFXNKP07Z41JWEB2WC`, `01KYC1F7VZHPNHRSXN8M0D9TCE`, `01KYC1F7Z9Z8MX5MJ1CSG6PYCB`, `01KYC1F82NW9PPKAAEAKZ24NSS` | scheduled |
| KDA | 2 | [01KYC1F92YYNMSPH3SAQD5NXDJ](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYC1F92YYNMSPH3SAQD5NXDJ) | `01KYC1F96WKK9H6MH3MB6NBM33`, `01KYC1F9AH8029CKBSAXQW6A5E`, `01KYC1F9DS9GJYR5C8C02ZGV3B`, `01KYC1F9H0PM5RJGN14JXBW53R` | scheduled |
| GDN2 | 2 | [01KYC1FAG4R32BXSPBB57PVR2A](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYC1FAG4R32BXSPBB57PVR2A) | `01KYC1FAMB0WC7SPHHDKAGJB9T`, `01KYC1FAQWE7SGG5PZ936T6QDF`, `01KYC1FAVA3AWT1RQY1D90QYM4`, `01KYC1FAYNJNNW3VHZ816B6VND` | scheduled/queued |
| KDA | 4 | [01KYC1FBW5T3GC4XHEK0SHJG5J](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYC1FBW5T3GC4XHEK0SHJG5J) | `01KYC1FBZKF4QH5FDZ6ZQPX128`, `01KYC1FC31H2BK0TNSQVP06BEY`, `01KYC1FC6CN5RA4QSRFMTFNPNX`, `01KYC1FC9MH4EYA56PGB1VXH8E` | queued |
| GDN2 | 4 | [01KYC1FDDGC1RY528K4TZ776BV](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYC1FDDGC1RY528K4TZ776BV) | `01KYC1FDGV4XHKY0XQ0VYAW5ZY`, `01KYC1FDMBXM7573GFR0YC3MC0`, `01KYC1FDQTB4SGNPXRR4AQMQY6`, `01KYC1FDV9K59XBSAB4E2R85XZ` | queued |
| KDA | 8 | [01KYC1FER04GND05GA5F6EFH1N](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYC1FER04GND05GA5F6EFH1N) | `01KYC1FEVJ6076DS4F1M47EZH8`, `01KYC1FEYY3ZHP8N27BTDAZC07`, `01KYC1FF2A02YE62ZNTMK1M2X3`, `01KYC1FF5HRJR8S7ZZD1Z59RVP` | queued |
| GDN2 | 8 | [01KYC1FG3M4MCQT65NHW2FRKSY](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYC1FG3M4MCQT65NHW2FRKSY) | `01KYC1FG71AFWY17128KY9P1W3`, `01KYC1FGAFMNS4766MKHX80S8J`, reused `01KYBVM3KP7NRWB21ZFGNK2K75`, `01KYC1FGDRX9QHX0FR6XC9ARSR` | 3 queued + 1 running |

The submission ledger is
[`launchers/pretraining/generated/275m_canonical_gdn2_kda_lr_sweep_submissions.json`](launchers/pretraining/generated/275m_canonical_gdn2_kda_lr_sweep_submissions.json).
The 31 new jobs request 152 GPUs; the complete 32-cell comparison including
the reused eight-GPU cell is 160 GPUs.

Final status at 2026-07-25 17:22 UTC: KDA finished all 16/16 jobs without a failed
attempt. All four curves are bracketed and choose observed LR `1.6e-3`, with
strict final-250M CEs `2.717057`, `2.587990`, `2.486896`, and `2.405198` at
Cx1/2/4/8. Canonical GDN2 also finished all 16/16 jobs without a failed
attempt. Every curve selects observed LR `1.6e-3`, with strict final-250M CEs
`2.677515`, `2.557597`, `2.467207`, and `2.389725`. The completed Cx8 fit
predicts approximately `1.21e-3`; formal selection remains the observed
`1.6e-3` point. The shared best-of plot now includes canonical GDN2 Cx8 as a
fully completed, non-provisional curve.

The paired plotting entry point is `plot_canonical_gdn2_kda.py`. It resolves
all 32 paired 275M sweep cells and the 12 canonical GDN2 scale-transfer cells
by exact W&B display name; duplicate exact names fail closed. One command
produces separate baseline-free U-plots for canonical GDN2 and KDA plus a
single strict observed-best plot against wide integration, matching gated-NoPE
GDN1, and original `expand_v=2` GDN2. Every selected 275M point has a completed,
bracketed quadratic curve.

## Canonical GDN2 larger-scale transfer

Submitted 2026-07-25 as 12 distinct urgent, unallocated Holmes experiments
from commit `504872eae`. These are the canonical `expand_v=1`, nonnegative
GDN2 models with gated full attention and NoPE, not the earlier `expand_v=2`,
negative-eigenvalue family. All cells use their corresponding transferred wide
integration LR, normal backward recomputation, accumulation factor one,
rolling ephemeral checkpoints, and out-of-loop evaluation.

The manifest deliberately submits the longest cells first: 810M Cx8, 1.2B
Cx4/Cx8, all 480M cells, 810M Cx1/Cx2/Cx4, then 1.2B Cx1/Cx2. The balanced
layouts request 176 GPUs at full concurrency. The larger `expand_v=2` profile
already passed every reused capacity layout; canonical `expand_v=1` has lower
active counts and recurrent value-state memory.

| Submission | Cell | GPUs | EP | Rank MB | LR | Beaker |
|---:|---|---:|---:|---:|---:|---|
| 1 | 810M Cx8 | 16 | 1 | 6 | `4e-4` | [01KYD5XE46SZT0V03YD9K7BT93](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYD5XE46SZT0V03YD9K7BT93) |
| 2 | 1.2B Cx4 | 16 | 8 `sync_1d` | 4 | `3e-4` | [01KYD5XHT0PE00Q17A7MVVMMME](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYD5XHT0PE00Q17A7MVVMMME) |
| 3 | 1.2B Cx8 | 32 | 8 `sync_1d` | 3 | `4e-4` | [01KYD5XN1HVE5T6JRQG0XWD3WE](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYD5XN1HVE5T6JRQG0XWD3WE) |
| 4 | 480M Cx1 | 8 | 1 | 4 | `1.2e-3` | [01KYD5XR1Y35K6FA0KS9RPA6CW](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYD5XR1Y35K6FA0KS9RPA6CW) |
| 5 | 480M Cx2 | 8 | 1 | 6 | `9e-4` | [01KYD5XTYN7T4BEFMMT5946C9C](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYD5XTYN7T4BEFMMT5946C9C) |
| 6 | 480M Cx4 | 8 | 1 | 8 | `8e-4` | [01KYD5XY0PTA050SVGPKSB7NXE](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYD5XY0PTA050SVGPKSB7NXE) |
| 7 | 480M Cx8 | 16 | 1 | 6 | `8e-4` | [01KYD5Y0R1R166QDZXHK6FMVTZ](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYD5Y0R1R166QDZXHK6FMVTZ) |
| 8 | 810M Cx1 | 16 | 1 | 2 | `6e-4` | [01KYD5Y3WK8BFTVC5WD5ZA31QD](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYD5Y3WK8BFTVC5WD5ZA31QD) |
| 9 | 810M Cx2 | 16 | 1 | 3 | `5.6e-4` | [01KYD5Y77MP9JJ8PN7PS22XRT9](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYD5Y77MP9JJ8PN7PS22XRT9) |
| 10 | 810M Cx4 | 16 | 1 | 4 | `4e-4` | [01KYD5YBRDK3NWK0EDER5G7YY7](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYD5YBRDK3NWK0EDER5G7YY7) |
| 11 | 1.2B Cx1 | 8 | 8 `sync_1d` | 4 | `4e-4` | [01KYD5YEFCH20J94SKFH7YW435](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYD5YEFCH20J94SKFH7YW435) |
| 12 | 1.2B Cx2 | 16 | 8 `sync_1d` | 3 | `6e-4` | [01KYD5YK5YFKSJT0N3Y2HZ36W7](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYD5YK5YFKSJT0N3Y2HZ36W7) |

At the immediate post-submit audit, nine experiments were scheduled and three
were created/queued; none had failed. The submitted specs were checked across
all 22 replicas for the exact source commit, canonical model variant, urgent
priority, zero minimum runtime, Holmes-only placement, disabled in-loop evals,
and rolling ephemeral checkpoint policy. The complete immutable submission
ledger is
[`launchers/pretraining/generated/geometry_matched_scale_gdn2_ev1_noneg_nope_gated_balanced_submissions.json`](launchers/pretraining/generated/geometry_matched_scale_gdn2_ev1_noneg_nope_gated_balanced_submissions.json).

The same plotting entry point maintains the finished-only four-size comparison
at
[`plots/pretraining/canonical_gdn2_kda/gdn2_fixed_lr_scale_comparison.png`](plots/pretraining/canonical_gdn2_kda/gdn2_fixed_lr_scale_comparison.png)
and its machine-/human-readable ledgers at
[`results/pretraining/canonical_gdn2_kda/scale_results.json`](results/pretraining/canonical_gdn2_kda/scale_results.json)
and
[`scale_results.md`](results/pretraining/canonical_gdn2_kda/scale_results.md).
Live, queued, or unresolved canonical cells are labeled pending; only finished
runs with a complete strict final-250M-token window enter the plotted series.

Status at 2026-07-27 21:43 UTC: 480M Cx1/Cx4/Cx8 have strict final-250M CEs
`2.492882`, `2.293165`, and `2.226409`. All four 810M cells are now finished at
`2.346904`, `2.260980`, `2.181919`, and `2.109020`. The 1.2B Cx1 explicit
resume chain finished at `2.253605`, Cx4 finished at `2.089778`, and Cx8's
complete local W&B history was recovered at `2.020529`. The 480M Cx2 and 1.2B
Cx2 repeatable numerical failures remain pending. The collector registers
complete explicit W&B chains, verifies registered local recovery artifacts,
and enforces the full final-250M window before admitting any result.

## Canonical KDA 480M stability transfer

Submitted 2026-07-26 from commit `65fee545b` as four urgent, unallocated
Holmes experiments. This wave changes only the recurrent mixer from canonical
GDN2 to canonical KDA: the 15-layer geometry, 12:3 recurrent/full-attention
ratio, dense-first FFN, gated NoPE attention, MoE layout, transferred LR,
global batch, GPU count, and rank microbatch all match the corresponding 480M
canonical GDN2 cells. The KDA configuration uses `expand_v=1`, nonnegative
eigenvalues, and the base image's FLA 0.4.1 KDA kernel. The audited config has
471,153,504 active parameters, 394,083,168 active non-embedding parameters,
and 7,190,723,424 total parameters.

| Cell | LR | Global batch | GPUs | EP | Rank MB | Beaker |
|---|---:|---:|---:|---:|---:|---|
| 480M Cx1 | `1.2e-3` | 262,144 | 8 | 1 | 4 | [01KYEG1B2V572PFM8VD04ZAR58](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYEG1B2V572PFM8VD04ZAR58) |
| 480M Cx2 | `9e-4` | 393,216 | 8 | 1 | 6 | [01KYEG1DQVF32TQ06K0DSH60RT](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYEG1DQVF32TQ06K0DSH60RT) |
| 480M Cx4 | `8e-4` | 524,288 | 8 | 1 | 8 | [01KYEG1GAEDVYM5P7HVTTYJ2SE](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYEG1GAEDVYM5P7HVTTYJ2SE) |
| 480M Cx8 | `8e-4` | 786,432 | 16 | 1 | 6 | [01KYEG1JXY3XD7WR0Z7RBDMFHS](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYEG1JXY3XD7WR0Z7RBDMFHS) |

All four cells use accumulation factor one, rolling 500-step ephemeral
checkpoints, no in-loop evaluation, and distinct checkpoint/W&B identities.
The immutable submission ledger is
[`launchers/pretraining/generated/480m_geometry_kda_ev1_noneg_nope_gated_submissions.json`](launchers/pretraining/generated/480m_geometry_kda_ev1_noneg_nope_gated_submissions.json).

Status at 2026-07-27 04:31 UTC: all four cells finished cleanly without a
failed attempt. Their strict final-250M CEs at Cx1/2/4/8 are `2.517826`,
`2.412884`, `2.323228`, and `2.237558`. All exact display names are registered
in `plot_canonical_gdn2_kda.py`, and the fixed-LR comparison includes canonical
KDA as a finished-only series alongside wide integration, matching GDN1,
original GDN2, and canonical GDN2.

## KDA `expand_v=2`, negative-eigenvalue transfer

Submitted 2026-07-26 from commit `6af7b95f2` as eight urgent, unallocated
Holmes tasks. This family is explicitly named `kda-ev2-neg-nope-gated` in
Beaker, W&B, checkpoints, manifests, and plots so it cannot be merged with
canonical `expand_v=1`, nonnegative KDA.

All cells use accumulation factor one, compilation, rolling 500-step ephemeral
checkpoints, no in-loop evals, and out-of-loop validation after training. The
275M/480M/810M cells use EP1; the 1.2B cells use EP8 with the fixed rowwise
collective. The 275M cells use matching-GDN1 observed-best LRs; the larger
sizes use the usual transferred-wide LRs.

| Model | Cx | LR | GPUs | Rank MB | Beaker |
|---|---:|---:|---:|---:|---|
| 275M | 1 | `8e-4` | 4 | 8 | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYG9M9694N3TDCPM13KHCM8S) |
| 275M | 2 | `1.6e-3` | 4 | 12 | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYG9M9694N3TDCPM13KHCM8S) |
| 275M | 4 | `8e-4` | 4 | 16 | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYG9M9694N3TDCPM13KHCM8S) |
| 275M | 8 | `8e-4` | 8 | 12 | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYG9M9694N3TDCPM13KHCM8S) |
| 480M | 1 | `1.2e-3` | 8 | 4 | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYG9MQY83B794ZX1N0B9G336) |
| 480M | 2 | `9e-4` | 8 | 6 | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYG9MTKF140YDTHSB3EN5CNS) |
| 480M | 4 | `8e-4` | 8 | 8 | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYG9MX5723AQN3R2AH5G69B2) |
| 480M | 8 | `8e-4` | 16 | 6 | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYG9MZW2S5FP6PZQ15MW2YRY) |
| 810M | 1 | `6e-4` | 16 | 2 | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYGZYKHZJBMH0MS44N2VDGCB) |
| 810M | 2 | `5.6e-4` | 16 | 3 | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYGZYPM4T4TE78G24GR72Y2G) |
| 810M | 4 | `4e-4` | 16 | 4 | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYGZYSH64GPWVTCT7WTV7AMT) |
| 810M | 8 | `4e-4` | 16 | 6 | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYGZYW6R9CW9JTJ6CGPEK10D) |
| 1.2B | 1 | `4e-4` | 8 | 4 | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYKDA9Y1Z8MCH7AVBGYZ1KJ8) |
| 1.2B | 2 | `6e-4` | 16 | 3 | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYKDB9N96S6T5GPRCASXEVEK) |
| 1.2B | 4 | `3e-4` | 16 | 4 | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYKDBCHNWNHQB6EFN7DTZ07N) |
| 1.2B | 8 | `4e-4` | 32 | 3 | [work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYKDBFGHC70RPHRQ326P9ZBQ) |

The immutable launch ledgers are
[`275m_kda_ev2_neg_nope_gated_transfer_submissions.json`](launchers/pretraining/generated/275m_kda_ev2_neg_nope_gated_transfer_submissions.json),
[`480m_geometry_kda_ev2_neg_nope_gated_submissions.json`](launchers/pretraining/generated/480m_geometry_kda_ev2_neg_nope_gated_submissions.json),
and
[`810m_geometry_kda_ev2_neg_nope_gated_submissions.json`](launchers/pretraining/generated/810m_geometry_kda_ev2_neg_nope_gated_submissions.json).
The final 1.2B rowwise submissions are recorded in
[`1p2b_geometry_kda_ev2_neg_nope_gated_rowwise_ext3_submissions.json`](launchers/pretraining/generated/1p2b_geometry_kda_ev2_neg_nope_gated_rowwise_ext3_submissions.json).
The dedicated finished-only plot is
`plots/pretraining/canonical_gdn2_kda/kda_ev2_neg_fixed_lr_scale_comparison.png`;
its result ledgers are kept separate from the canonical KDA LR sweep.

Status at 2026-07-28 04:38 UTC: all eight 275M/480M cells and 810M Cx1/Cx2/Cx4
finished without a failed attempt. The 810M strict final-250M CEs collected so
far are `2.352304`, `2.241873`, and `2.158207`; Cx8 is healthy at 47% with an
approximately 19-hour ETA. All four 1.2B cells are running cleanly at 1--3%
complete, with current ETAs of roughly 23, 23, 40, and 44 hours for
Cx1/Cx2/Cx4/Cx8. No KDA cell in this family has failed.

The four 810M extensions were submitted on 2026-07-27 from commit
`395a61e85` as urgent, unallocated, non-preemptible Holmes jobs. Each uses two
8-GPU nodes (64 GPUs if all four run concurrently), the audited 839,239,616
active-parameter configuration, and the same balanced resource layout as the
canonical 810M GDN2 wave. At the first post-submit audit, Cx1/Cx2 were
initializing and Cx4/Cx8 were queued.

The four 1.2B extensions were submitted on 2026-07-28 from commit
`dac4353bf` as urgent, unallocated Holmes jobs. They preserve the balanced
8/16/16/32-GPU, EP8, MB4/3/4/3 layout, but use the fixed codebase default
`rowwise_nvshmem` collective rather than the legacy `sync_1d` workaround.
Because the current image predates the rowwise helper extension, each replica
builds `symm_mem_vdev2d` once in Gantry's post-checkout setup phase before
`torchrun`.
Earlier startup-only submissions were stopped and excluded from the result
registry; the final jobs have distinct W&B and checkpoint names. Accidental
duplicate rowwise works were canceled before any worker started, so they
created neither W&B runs nor checkpoint state.

## 275M geometry-matched gated-NoPE GDN2 sweep

- Manifest: [`launchers/pretraining/manifests/275m_geometry_gdn2_ev2_nope_gated.yaml`](launchers/pretraining/manifests/275m_geometry_gdn2_ev2_nope_gated.yaml)
- Beaker experiment: [01KY8TKEBSZHYBZYEC5NFB92YK](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8TKEBSZHYBZYEC5NFB92YK)
- Plots/results: [`plots/pretraining/geometry_gdn2_ev2_nope_gated/`](plots/pretraining/geometry_gdn2_ev2_nope_gated/)
  and [`results/pretraining/geometry_gdn2_ev2_nope_gated/results.md`](results/pretraining/geometry_gdn2_ev2_nope_gated/results.md)

All tasks are urgent and unallocated on Holmes. Status/progress below is the
2026-07-24 16:10 UTC snapshot.

| Cx | LR | Current job | W&B | State / strict result |
|---:|---:|---|---|---|
| 1 | `4e-4` | `01KY8TKEF6J4BK4G1A4YF6JGVM` | [rsxmn720](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rsxmn720) | finished; CE `2.676958` |
| 1 | `8e-4` | `01KY8TKEJQ2T4FE36P1SQJ4MEQ` | [pqrdvu63](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/pqrdvu63) | finished; CE `2.657052` |
| 1 | `1.6e-3` | `01KY8TKEP32D2476KWG92RTGQK` | [5uzr9dva](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5uzr9dva) | finished; CE `2.646730` |
| 1 | `3.2e-3` | `01KY8TKESHR6VS04WT0EXR8Z43` | [j2t5c2jb](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/j2t5c2jb) | finished; CE `2.661486` |
| 2 | `4e-4` | `01KY8TKEWX8RE3HZNRKEBRAYR5` | [7yh4rfi1](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7yh4rfi1) | finished; CE `2.564391` |
| 2 | `8e-4` | `01KY8TKF05JSXDAH47206QWNV4` | [2egeqyvo](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/2egeqyvo) | finished; CE `2.544136` |
| 2 | `1.6e-3` | `01KY9D9YN7SHRMKQ1DMB5P9V8W` | [final segment jhcmk80f](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jhcmk80f) | finished; combined strict CE `2.534116` |
| 2 | `3.2e-3` | `01KY8TKF6N03DQD0FPM3A360HJ` | [xwtxd1pv](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xwtxd1pv) | finished; CE `2.548521` |
| 4 | `4e-4` | `01KY8TKF9T8ZR612Z1418THVWY` | [6b0vighm](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6b0vighm) | finished; CE `2.462028` |
| 4 | `8e-4` | `01KY8TKFD1EPRHEY7TE4XSZNS4` | [yq4mi5o0](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/yq4mi5o0) | finished; CE `2.446822` |
| 4 | `1.6e-3` | `01KY8TKFG8HGZRF8PYTEA89SMY` | [0w6ezwgx](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0w6ezwgx) | finished; CE `2.443132`; observed best |
| 4 | `3.2e-3` | `01KY8TKFKCY8PWKN156K0YG3J6` | [kcig30ty](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kcig30ty) | finished; CE `2.461867` |
| 8 | `4e-4` | `01KY8TKFPMKP9MCCVNQM3P6KS9` | [jewjx6yq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jewjx6yq) | finished; CE `2.372393` |
| 8 | `8e-4` | `01KY8TKFST2JN575ZCHDXSRXG6` | [1lpz9reu](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1lpz9reu) | finished; CE `2.356985`; current observed best |
| 8 | `1.6e-3` | `01KY8TKFX02JSHWGT8X467S5E5` | [n48z3vh8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/n48z3vh8) | stopped at step 36,768 on non-finite loss; durable `step36500` |
| 8 | `3.2e-3` | `01KY8TKG0AJ8706NGJZN9GQC66` | [e6n5iscu](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/e6n5iscu) | finished; CE `2.380649` |

The Cx8 `1.6e-3` cell was retried four times from durable `step36500`; every
attempt failed deterministically on a non-finite loss at step 36,768. A fifth,
diagnostic continuation was queued urgent and unallocated on 2026-07-24 in
[01KYAY2X46RDRM00YEQT7E0VBS](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYAY2X46RDRM00YEQT7E0VBS).
It preserves the original run name and checkpoint directory and enables the
same all-rank non-finite and pre-failure gradient-norm diagnostics as the larger
GDN2 retries. If it reproduces without a preceding gradient warning, the next
diagnostic should target the loss/forward activations rather than performing
another blind checkpoint retry.

## Larger gated-NoPE GDN2 qualification

All three checkpoint-free 50-step production-layout smokes passed on
2026-07-24 at commit `4212d267b`. All five replicas exited 0 with finite
loss/gradients and zero skipped steps.

| Model / layout | Beaker | W&B | Final-10 TFLOPs/GPU | Final-10 TPS/GPU | Active / reserved GiB |
|---|---|---|---:|---:|---:|
| 480M MB8, 8 GPU EP1 | [01KY9GKF3MDHPFTKM9NJ9A0W81](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY9GKF3MDHPFTKM9NJ9A0W81) | [k9sw3u2k](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/k9sw3u2k) | 335.9 | 114.1K | 190.4 / 196.9 |
| 480M MB6, 16 GPU EP1 | [01KY9GKJ0BJ6ED4DKSKD3BHD3Y](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY9GKJ0BJ6ED4DKSKD3BHD3Y) | [db06dpbg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/db06dpbg) | 287.4 | 97.6K | 150.4 / 155.3 |
| 810M MB6, 16 GPU EP1 | [01KY9GKP0A1BW8XSF8C3W6G4J0](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY9GKP0A1BW8XSF8C3W6G4J0) | [fcuou24g](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/fcuou24g) | 358.0 | 66.8K | 224.5 / 234.1 |
| 1.2B MB4, 8 GPU EP8 `sync_1d` | [01KYAGA7P0W3HKZ8FDDGT80GEP](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYAGA7P0W3HKZ8FDDGT80GEP) | [f14cy64l](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/f14cy64l) | 367.2 | 45.1K | 189.7 / 198.8 |
| 1.2B MB4, 16 GPU EP8 `sync_1d` | [01KYAGAASYT6FHRJ86DB2NRQPK](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYAGAASYT6FHRJ86DB2NRQPK) | [qi7b76ad](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qi7b76ad) | 358.2 | 44.0K | 176.5 / 185.7 |
| 1.2B MB3, 32 GPU EP8 `sync_1d` | [01KYAGAERF6G60VSQQYNN8KR7X](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYAGAERF6G60VSQQYNN8KR7X) | [eqmvumxk](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/eqmvumxk) | 286.3 | 35.2K | 133.5 / 140.6 |

The smoke submission ledger is
`launchers/pretraining/generated/geometry_matched_scale_gdn2_nope_gated_smoke_submissions.json`.
All three balanced 1.2B layouts passed at commit `f886c7b79`; all seven
replicas exited 0 after 50 finite steps with zero skipped steps. All model
sizes are approved for the transferred-wide-LR production wave.

### Larger gated-NoPE GDN2 production wave

The eight qualified 480M/810M cells were submitted on 2026-07-24 from commit
`ed7accc25`. The four balanced 1.2B cells were subsequently submitted from
commit `cd40c04a5`. All are urgent, unallocated Holmes work. Every cell uses
accumulation factor 1, the transferred wide LR, normal backward recomputation,
rolling ephemeral checkpoints, and out-of-loop evaluation.

| Cell | GPUs | Rank MB | LR | Beaker |
|---|---:|---:|---:|---|
| 480M Cx1 | 8 | 4 | `1.2e-3` | [01KY9HQ0GHV1P6NYWVJRVQ39CH](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY9HQ0GHV1P6NYWVJRVQ39CH) |
| 480M Cx2 | 8 | 6 | `9e-4` | [01KY9HQ3EGGYYX1VDQP4SP3799](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY9HQ3EGGYYX1VDQP4SP3799) |
| 480M Cx4 | 8 | 8 | `8e-4` | [01KY9HQ67F0JPW80EK1WZHKJ9K](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY9HQ67F0JPW80EK1WZHKJ9K) |
| 480M Cx8 | 16 | 6 | `8e-4` | [01KY9HQ946E92SYPNWBMS697JY](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY9HQ946E92SYPNWBMS697JY) |
| 810M Cx1 | 16 | 2 | `6e-4` | [01KY9HQC8J0THDSVCN8YJDQA1V](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY9HQC8J0THDSVCN8YJDQA1V) |
| 810M Cx2 | 16 | 3 | `5.6e-4` | [01KY9HQFHP2D56FEBNMSMYQ6BT](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY9HQFHP2D56FEBNMSMYQ6BT) |
| 810M Cx4 | 16 | 4 | `4e-4` | [01KY9HQJY6G36KKDPSEFTVWCSB](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY9HQJY6G36KKDPSEFTVWCSB) |
| 810M Cx8 | 16 | 6 | `4e-4` | [01KY9HQNN4KG4C5FYG5JKB463K](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY9HQNN4KG4C5FYG5JKB463K) |
| 1.2B Cx1 | 8 | 4 | `4e-4` | [01KYAHJQC03QE92ZBQ1202EN8H](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYAHJQC03QE92ZBQ1202EN8H) |
| 1.2B Cx2 | 16 | 3 | `6e-4` | [01KYAHJTGPBX4S7FSE51AVWNX8](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYAHJTGPBX4S7FSE51AVWNX8) |
| 1.2B Cx4 | 16 | 4 | `3e-4` | [01KYAHJX8M013RG4VT5AVFAKEK](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYAHJX8M013RG4VT5AVFAKEK) |
| 1.2B Cx8 | 32 | 3 | `4e-4` | [01KYAHK0MCQ9EPJP0QZVAXMDN7](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYAHK0MCQ9EPJP0QZVAXMDN7) |

Submission ledger:
`launchers/pretraining/generated/geometry_matched_scale_gdn2_nope_gated_480m_810m_submissions.json`.
The 1.2B ledger is
`launchers/pretraining/generated/geometry_matched_scale_gdn2_nope_gated_1p2b_submissions.json`.

## 275M active hybrid GDN (`expand_v=1`)

- Manifest: [`launchers/pretraining/manifests/275m_hybrid_gdn_ev1.yaml`](launchers/pretraining/manifests/275m_hybrid_gdn_ev1.yaml)
- Training entrypoint: `src/scripts/train/jacobm_olmoe3_275m_hybrid_wide.py`
- W&B project/group: `ai2-llm/jacobm-olmoe-ladder` / `olmoe3-275m-integration-wide-hybrid`
- Initial Cx1 experiment: [01KXFR1KT408AWVN41NPKXS4F5](https://beaker.org/ex/01KXFR1KT408AWVN41NPKXS4F5)
- Initial Cx2/Cx4/Cx8 experiment: [01KXFSA0GP221T0X7V675XTSG2](https://beaker.org/ex/01KXFSA0GP221T0X7V675XTSG2)
- Bracketing extensions: [01KXHZNJ5FD5RA0BRC4ZF3DRKC](https://beaker.org/ex/01KXHZNJ5FD5RA0BRC4ZF3DRKC)
- Exact job/W&B table: [`../v1/DDP_RUNS.md`](../v1/DDP_RUNS.md#275m-integration-wide-hybrid-control)
- Plot/results registry: [`plot_pretraining_wave.py`](plot_pretraining_wave.py)
- Consolidated all-size artifacts:
  [`plots/pretraining/hybrid_gdn_ev1/`](plots/pretraining/hybrid_gdn_ev1/)
  and
  [`results/pretraining/hybrid_gdn_ev1/results.md`](results/pretraining/hybrid_gdn_ev1/results.md)

Status on 2026-07-16 01:59 UTC: the original grid and all four bracketing
extensions are complete. The Cx8 `3.2e-3` run was interrupted by a Holmes node
Xid 31 failure at step 29,498/42,954, resumed from its durable `step29000`
checkpoint, and finished cleanly at step 42,954. All four Cx cells now have an
observed interior best and a valid quadratic fit, so all four are present in the
observed-optimal summary.

| Cx | LR | Global batch | Rank microbatch | Accumulation | Job | W&B | Status |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | `4e-4` | 262,144 | 16 | 1 | [01KXHZNJGGRPHHN4PYYG90T9AP](https://beaker.org/ex/01KXHZNJGGRPHHN4PYYG90T9AP) | [fkm77yos](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/fkm77yos) | finished |
| 1 | `8e-4` | 262,144 | 16 | 1 | [01KXFR1M6K4M6SV4P2EBYWJYK3](https://beaker.org/ex/01KXFR1M6K4M6SV4P2EBYWJYK3) | [yo22u93q](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/yo22u93q) | finished |
| 1 | `1.6e-3` | 262,144 | 16 | 1 | [01KXFR1M9Y9966MHENKVDDS9TZ](https://beaker.org/ex/01KXFR1M9Y9966MHENKVDDS9TZ) | [moknw6oc](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/moknw6oc) | finished |
| 1 | `3.2e-3` | 262,144 | 16 | 1 | [01KXFR1MD63EEYTQRX1DW9611K](https://beaker.org/ex/01KXFR1MD63EEYTQRX1DW9611K) | [mettf0d3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mettf0d3) | finished |
| 2 | `4e-4` | 393,216 | 8 | 3 | [01KXHZNJKTW5ZFH2F7BT01GQPB](https://beaker.org/ex/01KXHZNJKTW5ZFH2F7BT01GQPB) | [s5qmhyb2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/s5qmhyb2) | finished |
| 2 | `8e-4` | 393,216 | 8 | 3 | [01KXFSA0WJCSDGKKB91EB5ZYK4](https://beaker.org/ex/01KXFSA0WJCSDGKKB91EB5ZYK4) | [07qo96gy](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/07qo96gy) | finished |
| 2 | `1.6e-3` | 393,216 | 8 | 3 | [01KXFSA0ZY3W8MDXCVW1DAGKJA](https://beaker.org/ex/01KXFSA0ZY3W8MDXCVW1DAGKJA) | [j12fk559](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/j12fk559) | finished |
| 2 | `3.2e-3` | 393,216 | 8 | 3 | [01KXFSA13248BP335W5EK8EX2G](https://beaker.org/ex/01KXFSA13248BP335W5EK8EX2G) | [mem73c7g](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mem73c7g) | finished |
| 4 | `4e-4` | 524,288 | 16 | 2 | [01KXFSA16XJC9XBNA74B2QTK7M](https://beaker.org/ex/01KXFSA16XJC9XBNA74B2QTK7M) | [socvue3a](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/socvue3a) | finished |
| 4 | `8e-4` | 524,288 | 16 | 2 | [01KXFSA1AA0H5WY9Y074TS604S](https://beaker.org/ex/01KXFSA1AA0H5WY9Y074TS604S) | [xvk92054](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xvk92054) | finished |
| 4 | `1.6e-3` | 524,288 | 16 | 2 | [01KXFSA1DMP6S8G7GAE0JW3DR4](https://beaker.org/ex/01KXFSA1DMP6S8G7GAE0JW3DR4) | [uhw9wfed](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/uhw9wfed) | finished |
| 4 | `3.2e-3` | 524,288 | 16 | 2 | [01KXHZNJQ84EGNZYZ8N7KBAR43](https://beaker.org/ex/01KXHZNJQ84EGNZYZ8N7KBAR43) | [sr1jgmao](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sr1jgmao) | finished |
| 8 | `4e-4` | 786,432 | 16 | 3 | [01KXFSA1GT8JJWTVMSNXF428XW](https://beaker.org/ex/01KXFSA1GT8JJWTVMSNXF428XW) | [b0z3qfmi](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/b0z3qfmi) | finished |
| 8 | `8e-4` | 786,432 | 16 | 3 | [01KXFSA1M6ZR3JCNXB2VQ9K04J](https://beaker.org/ex/01KXFSA1M6ZR3JCNXB2VQ9K04J) | [rkxojd03](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rkxojd03) | finished |
| 8 | `1.6e-3` | 786,432 | 16 | 3 | [01KXFSA1QH0T6YHPV54FMR88AA](https://beaker.org/ex/01KXFSA1QH0T6YHPV54FMR88AA) | [66aja50m](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/66aja50m) | finished |
| 8 | `3.2e-3` | 786,432 | 16 | 3 | [initial](https://beaker.org/ex/01KXHZNJTHJH9K7X88TJA5Q537) / [resume](https://beaker.org/ex/01KXKBZK6FKCM081WJH3YP82TX) | [initial](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/f7lbyrfl) / [resume](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ntoo8vlo) | finished, step 42,954 |

## Hybrid scale microbatch smokes

- Manifest: [`launchers/pretraining/manifests/hybrid_scale_mb_smokes.yaml`](launchers/pretraining/manifests/hybrid_scale_mb_smokes.yaml)
- Training entrypoint: `src/scripts/train/jacobm_olmoe3_hybrid_scale.py`
- Beaker attempt r1: [01KXJ3S4JTY4P120JF6Q0Y21G6](https://beaker.org/ex/01KXJ3S4JTY4P120JF6Q0Y21G6)
- Beaker attempt r2: [01KXJ4142N92YT68G4JEKVSGCB](https://beaker.org/ex/01KXJ4142N92YT68G4JEKVSGCB)
- 1.2B EP retry r3: [01KXJ56X8R0V5QN4C36NDB06QE](https://beaker.org/ex/01KXJ56X8R0V5QN4C36NDB06QE)
- 1.2B EP comparison r4: [01KXJ5QKRM0GDK1H7EC6TBKK77](https://beaker.org/ex/01KXJ5QKRM0GDK1H7EC6TBKK77)
- 1.2B single-checkpoint continuation r5: [01KXJ6KAHVVE48VZEKPYADJDWE](https://beaker.org/ex/01KXJ6KAHVVE48VZEKPYADJDWE)
- 1.2B EP1 Cx2 MB3 fallback r8: [01KXJ84SZE4HBW1686BA7VAA9H](https://beaker.org/ex/01KXJ84SZE4HBW1686BA7VAA9H)
- Destination: `ai2/OLMo-3-moe-experiments` on `ai2/holmes`, urgent
- Checkpoint root: `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/pretraining-smokes`

Attempt r1 was submitted 2026-07-15. The three allocated jobs failed before
step 1 because the Beaker callback had not been given `BEAKER_TOKEN`; the three
unallocated jobs were canceled. This was launcher infrastructure, not a memory
result. Attempt r2 adds `jacobm_BEAKER_TOKEN` and retains the same largest-MB
shapes. The 1.2B tasks use EP8 while the smaller sizes use EP1.

| Size | Cx | GPUs | EP | DP | Rank microbatch | Accumulation | Job | W&B | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 480M | 1 | 4 | 1 | 4 | 8 | 1 | [01KXJ3S4Z713DR7PVJAXZR07TT](https://beaker.org/ex/01KXJ3S4Z713DR7PVJAXZR07TT) | none | infra failed before step 1 |
| 480M | 2 | 4 | 1 | 4 | 12 | 1 | [01KXJ3S52PCW419ZX7YRB28GC1](https://beaker.org/ex/01KXJ3S52PCW419ZX7YRB28GC1) | [6c61hps9](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6c61hps9) | infra failed before step 1 |
| 810M | 1 | 8 | 1 | 8 | 4 | 1 | [01KXJ3S564EWDKY0F4MAH4TQ6K](https://beaker.org/ex/01KXJ3S564EWDKY0F4MAH4TQ6K) | none | infra failed before step 1 |
| 810M | 2 | 8 | 1 | 8 | 6 | 1 | [01KXJ3S59GY2J4PS4B6Z4V6DZC](https://beaker.org/ex/01KXJ3S59GY2J4PS4B6Z4V6DZC) | none | canceled before allocation |
| 1.2B | 1 | 8 | 8 | 1 | 8 | 4 | [01KXJ3S5D35RQYB134KY3NRM67](https://beaker.org/ex/01KXJ3S5D35RQYB134KY3NRM67) | none | canceled before allocation |
| 1.2B | 2 | 8 | 8 | 1 | 12 | 4 | [01KXJ3S5GFVN6ZED80B9QJ9CW4](https://beaker.org/ex/01KXJ3S5GFVN6ZED80B9QJ9CW4) | none | canceled before allocation |

Attempt r2:

| Size | Cx | GPUs | EP | DP | Rank microbatch | Accumulation | Job | W&B | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 480M | 1 | 4 | 1 | 4 | 8 | 1 | [01KXJ414EFESAP5Y9YXV44F8BJ](https://beaker.org/ex/01KXJ414EFESAP5Y9YXV44F8BJ) | [7mo5umkl](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7mo5umkl) | passed |
| 480M | 2 | 4 | 1 | 4 | 12 | 1 | [01KXJ414HY4J4THBE2PH7RXWBJ](https://beaker.org/ex/01KXJ414HY4J4THBE2PH7RXWBJ) | [zpg8791v](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/zpg8791v) | passed |
| 810M | 1 | 8 | 1 | 8 | 4 | 1 | [01KXJ414NBT6GSV2KGY0NJZWZE](https://beaker.org/ex/01KXJ414NBT6GSV2KGY0NJZWZE) | [wzrbehoi](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/wzrbehoi) | passed |
| 810M | 2 | 8 | 1 | 8 | 6 | 1 | [01KXJ414RNSK42G33RZ796REGW](https://beaker.org/ex/01KXJ414RNSK42G33RZ796REGW) | [dxweu5sj](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/dxweu5sj) | training/checkpoints/evals passed; post-training wrapper exit 127 |
| 1.2B | 1 | 8 | 8 | 1 | 8 | 4 | [01KXJ414W3P4A2DY8ZHEF0BMX5](https://beaker.org/ex/01KXJ414W3P4A2DY8ZHEF0BMX5) | none | infra failed before step 1 |
| 1.2B | 2 | 8 | 8 | 1 | 12 | 4 | [01KXJ41508ZYQNCE6706ZKZVZ5](https://beaker.org/ex/01KXJ41508ZYQNCE6706ZKZVZ5) | none | canceled before allocation |

The r2 1.2B startup failure came from the local source copy lacking the compiled
`symm_mem_vdev2d` extension required by EP8. The wrapper now runs the branch's
supported one-time CMake extension build before `torchrun` on EP jobs. Attempt
r3 retains the same MB shapes:

| Size | Cx | GPUs | EP | DP | Rank microbatch | Accumulation | Job | W&B | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 1.2B | 1 | 8 | 8 | 1 | 8 | 4 | [01KXJ56XMHDRTZZG78NTD33FW3](https://beaker.org/ex/01KXJ56XMHDRTZZG78NTD33FW3) | none | `rowwise_nvshmem` startup segfault before step 1 |
| 1.2B | 2 | 8 | 8 | 1 | 12 | 4 | [01KXJ56XR07C4WPM25V6CXE6WD](https://beaker.org/ex/01KXJ56XR07C4WPM25V6CXE6WD) | none | canceled after Cx1 exposed the shared startup failure |

The extension built successfully in r3, but `rowwise_nvshmem` then segfaulted
while creating symmetric-memory EP groups. Attempt r4 compares the supported
non-symmetric `sync_1d` EP8 path with EP1 on the same B300 hardware. This keeps
the EP8 MB candidates unchanged and adds the largest legal no-accumulation EP1
candidates.

| Size | Cx | GPUs | EP | EP path | DP | Rank microbatch | Accumulation | Job | W&B | Status |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---|
| 1.2B | 1 | 8 | 8 | `sync_1d` | 1 | 8 | 4 | [01KXJ5QM4X6Q5MBEJZTYEACNBP](https://beaker.org/ex/01KXJ5QM4X6Q5MBEJZTYEACNBP) | [rvypniye](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rvypniye) | passed |
| 1.2B | 2 | 8 | 8 | `sync_1d` | 1 | 12 | 4 | [01KXJ5QMAJ89B3758W8CVJFDEF](https://beaker.org/ex/01KXJ5QMAJ89B3758W8CVJFDEF) | none | canceled after dry run to switch to one final checkpoint |
| 1.2B | 1 | 8 | 1 | none | 8 | 4 | 1 | [01KXJ5QMF4J67XQDEV834BDS7D](https://beaker.org/ex/01KXJ5QMF4J67XQDEV834BDS7D) | none | canceled before allocation to switch checkpoint cadence |
| 1.2B | 2 | 8 | 1 | none | 8 | 6 | 1 | [01KXJ5QMJY5VY8JMPV8SB7ME9C](https://beaker.org/ex/01KXJ5QMJY5VY8JMPV8SB7ME9C) | none | canceled before allocation to switch checkpoint cadence |

The 1.2B checkpoint includes distributed optimizer state and is about 221 GB.
After that was measured on r4 Cx1, the remaining three jobs were moved to r5
with only the final hard-stop checkpoint. No existing checkpoint was removed.

| Size | Cx | GPUs | EP | EP path | DP | Rank microbatch | Accumulation | Job | W&B | Status |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---|
| 1.2B | 2 | 8 | 8 | `sync_1d` | 1 | 12 | 4 | [01KXJ6KAX3QPNHMYTKNF2CTJ49](https://beaker.org/ex/01KXJ6KAX3QPNHMYTKNF2CTJ49) | [cp8sywxn](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/cp8sywxn) | passed |
| 1.2B | 1 | 8 | 1 | none | 8 | 4 | 1 | [01KXJ6KB20Z9SJ4180XQ5S5ER0](https://beaker.org/ex/01KXJ6KB20Z9SJ4180XQ5S5ER0) | [g0ahrz9f](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/g0ahrz9f) | passed |
| 1.2B | 2 | 8 | 1 | none | 8 | 6 | 1 | [01KXJ6KB5EPPQ92M1SB9JK4Q11](https://beaker.org/ex/01KXJ6KB5EPPQ92M1SB9JK4Q11) | [75mi3xd0](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/75mi3xd0) | OOM in compiled dry run at 267.0/267.7 GiB |

The next legal Cx2 EP1 fallback is MB3 with accumulation 2:

| Size | Cx | GPUs | EP | EP path | DP | Rank microbatch | Accumulation | Job | W&B | Status |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---|
| 1.2B | 2 | 8 | 1 | none | 8 | 3 | 2 | [01KXJ84TTGCX9T3HA9AGRFYN42](https://beaker.org/ex/01KXJ84TTGCX9T3HA9AGRFYN42) | [il66xple](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/il66xple) | passed |

## Full hybrid scale Cx1/Cx2 runs

- Manifest: [`launchers/pretraining/manifests/hybrid_scale_full_cx1_cx2.yaml`](launchers/pretraining/manifests/hybrid_scale_full_cx1_cx2.yaml)
- Launcher: [`launchers/pretraining/launch_hybrid_scale_full_cx1_cx2.sh`](launchers/pretraining/launch_hybrid_scale_full_cx1_cx2.sh)
- Beaker experiment: [01KXJAPGGFPW9GMK6C50EE2FA7](https://beaker.org/ex/01KXJAPGGFPW9GMK6C50EE2FA7)
- W&B project/group: `ai2-llm/jacobm-olmoe-ladder` / `olmoe3-integration-wide-hybrid-scale`
- Checkpoint root: `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/pretraining`
- Checkpoint policy: rolling ephemeral save every 500 steps with
  `remove=ephemeral_only`; final checkpoint is permanent.
- Evaluation policy: no evaluator callbacks inside the training process,
  including on finish. Run validation afterward in separate eval-only jobs.

Submitted 2026-07-15 at urgent priority on `ai2/holmes` using 40 requested B300
GPUs. LRs are the observed optimal wide-intervention LRs for each matching size
and data multiple.

Status on 2026-07-17 02:46 UTC: both 480M Cx1/Cx2 runs, both 810M Cx1/Cx2
runs, and 1.2B Cx1 are finished. The eval-enabled 1.2B Cx1 attempt was stopped
at durable `step26000`.
The fifth 1.2B Cx2 attempt reached durable `step10000` and failed during an
evaluator transition with an illegal memory access. Both were requeued in
[no-eval resume experiment 01KXMPNWR2ZA53JZN7V4A6PRGS](https://beaker.org/ex/01KXMPNWR2ZA53JZN7V4A6PRGS)
with evaluator callbacks disabled and the same run/checkpoint directories.

The completed 480M Cx1/Cx2 and 810M Cx1/Cx2 losses are included in the
consolidated fixed-LR scale-transfer plot. At the transferred LR, hybrid 810M
Cx1 has final-250M CE 2.364345 versus 2.373197 for wide integration (delta
-0.008852), and hybrid 810M Cx2 has 2.247185 versus 2.268948 (delta -0.021762).
The remaining registered 810M and 1.2B runs will enter that plot only after W&B
marks them finished.

| Size | Cx | LR | Global batch | GPUs | EP/path | MB cap / effective | Accum | Job | W&B | Status |
|---|---:|---:|---:|---:|---|---:|---:|---|---|---|
| 480M | 1 | `1.2e-3` | 262,144 | 4 | EP1 | 8 | 1 | [01KXJAPH2DP3XSCHX1A637SN7K](https://beaker.org/ex/01KXJAPH2DP3XSCHX1A637SN7K) | [wl8ebsd8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/wl8ebsd8) | finished |
| 480M | 2 | `9e-4` | 393,216 | 4 | EP1 | 12 | 1 | [01KXJAPH62PWXK2B53PKT14M08](https://beaker.org/ex/01KXJAPH62PWXK2B53PKT14M08) | [4vzmrld1](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4vzmrld1) | finished |
| 810M | 1 | `6e-4` | 262,144 | 8 | EP1 | 4 | 1 | [01KXJAPH9V450M32SJC5G4KN93](https://beaker.org/ex/01KXJAPH9V450M32SJC5G4KN93) | [h1rmcm2p](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h1rmcm2p) | finished |
| 810M | 2 | `5.6e-4` | 393,216 | 8 | EP1 | 6 | 1 | [01KXJAPHDN7KN3TY7NXRVRZGNM](https://beaker.org/ex/01KXJAPHDN7KN3TY7NXRVRZGNM) | [1d5gxgjv](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1d5gxgjv) | finished, step 74,357 |
| 1.2B | 1 | `4e-4` | 262,144 | 8 | EP8 / `sync_1d` | 8 / 4 | 1 | [eval-enabled](https://beaker.org/ex/01KXJAPHHB8MBPD1B3E92QH89Y) / [no-eval resume](https://beaker.org/ex/01KXMPNX36KW85J97N8MAKBQEA) | [eval-enabled](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xapobmqb) / [no-eval resume](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1d24xfx5) | finished, final `step86558`; final-250M CE `2.253953` |
| 1.2B | 2 | `6e-4` | 393,216 | 8 | EP8 / `sync_1d` | 12 / 6 | 1 | [initial](https://beaker.org/ex/01KXJAPHN6SNXMG7X49M7HH17G) / [resume 1](https://beaker.org/ex/01KXK9R5PMR9GXBC0RMSDY2V13) / [resume 2](https://beaker.org/ex/01KXKEEFKYPGQAVWS024PTGCPK) / [resume 3](https://beaker.org/ex/01KXKSKC7K2KNQ9EN99EE5N11F) / [resume 4](https://beaker.org/ex/01KXMHBTR1D29J3SND9FZ13B8Z) / [no-eval resume](https://beaker.org/ex/01KXMPNX6QXBFKH2DRRYMPXDV6) | [initial](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/l4r1crzm) / [resume 1](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/bwvkwb9s) / [resume 2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jsb3obpq) / [resume 3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ezechghu) / [resume 4](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jybetzoc) / [no-eval resume](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vr2jfn4c) | running, 30.74B / 45.377B tokens (67.7%) |

## Full hybrid scale Cx4/Cx8 runs

- Manifest: [`launchers/pretraining/manifests/hybrid_scale_full_cx4_cx8.yaml`](launchers/pretraining/manifests/hybrid_scale_full_cx4_cx8.yaml)
- Launcher: [`launchers/pretraining/launch_hybrid_scale_full_cx4_cx8.sh`](launchers/pretraining/launch_hybrid_scale_full_cx4_cx8.sh)
- Beaker experiment: [01KXKTT3ZT5G4V9QTFBR6MKGEZ](https://beaker.org/ex/01KXKTT3ZT5G4V9QTFBR6MKGEZ)
- Corrected 1.2B retry: [01KXKY6RTR5ZSD1R1BS40SF7KR](https://beaker.org/ex/01KXKY6RTR5ZSD1R1BS40SF7KR)
- No-eval resume: [01KXMPNY9SWZSYGWB4585Z1YEH](https://beaker.org/ex/01KXMPNY9SWZSYGWB4585Z1YEH)
- W&B project/group: `ai2-llm/jacobm-olmoe-ladder` / `olmoe3-integration-wide-hybrid-scale`
- Checkpoint and evaluation policies match the Cx1/Cx2 production runs.

Submitted 2026-07-15 at urgent priority on `ai2/holmes`, requesting 32 B300
GPUs. LRs are the observed optimal wide-intervention LRs for the matching size
and data multiple.

| Size | Cx | LR | Global batch | GPUs | EP/path | MB cap / effective | Accum | Job | W&B | Status |
|---|---:|---:|---:|---:|---|---:|---:|---|---|---|
| 810M | 4 | `4e-4` | 524,288 | 8 | EP1 | 4 | 2 | [eval-enabled](https://beaker.org/ex/01KXKTT4G4QT56NGHKWSVXWEX6) / [no-eval resume](https://beaker.org/ex/01KXMPNYN9J377PP8F8YZ763NJ) | [eval-enabled](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/adi3mjy7) / [no-eval resume](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/bvlzu2c9) | running, 55.31B / 58.477B tokens (94.6%) |
| 810M | 8 | `4e-4` | 786,432 | 8 | EP1 | 6 | 2 | [eval-enabled](https://beaker.org/ex/01KXKTT4KPQRMJ0E1DF5GPS26A) / [no-eval resume](https://beaker.org/ex/01KXMPNYRK51C80XFNZNAP337M) | [eval-enabled](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sucwb1sc) / [no-eval resume](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/k1d1td9b) | running, 65.44B / 116.954B tokens (56.0%) |
| 1.2B | 4 | `3e-4` | 524,288 | 8 | EP8 / `sync_1d` | 4 / 4 | 2 | [failed](https://beaker.org/ex/01KXKTT4R8MANSKM43DEJB02GC) / [eval-enabled retry](https://beaker.org/ex/01KXKY6SXD7DFDDJDGK83NGQDF) / [no-eval resume](https://beaker.org/ex/01KXMPNYVY6A2RMTNW2E27H9GR) | [failed](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/f9wybz72) / [eval-enabled retry](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9c1fcuto) / [no-eval resume](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h5ft97x1) | running, 29.29B / 90.754B tokens (32.3%) |
| 1.2B | 8 | `4e-4` | 786,432 | 8 | EP8 / `sync_1d` | 6 / 6 | 2 | [failed](https://beaker.org/ex/01KXKTT4W01BXNZ5WRD40BT2QG) / [eval-enabled retry](https://beaker.org/ex/01KXKY6T9QR8PQDWRAN4Y76CVQ) / [no-eval resume](https://beaker.org/ex/01KXMPNYZ7Q1J3BXXD47KET92X) | [final segment](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7eemhu7g) | finished, final-250M CE `2.016369` |

### 480M Cx4/Cx8 completion runs

- Manifest: [`launchers/pretraining/manifests/hybrid_scale_480m_cx4_cx8.yaml`](launchers/pretraining/manifests/hybrid_scale_480m_cx4_cx8.yaml)
- Launcher: [`launchers/pretraining/launch_hybrid_scale_480m_cx4_cx8.sh`](launchers/pretraining/launch_hybrid_scale_480m_cx4_cx8.sh)
- Beaker experiment: [01KXMTAQPTG52EPEXMQN0Q1YJ7](https://beaker.org/ex/01KXMTAQPTG52EPEXMQN0Q1YJ7)
- Unallocated Cx8 replacement:
  [01KXPEF6MWN4AKPH6CRJNZ1GWE](https://beaker.org/ex/01KXPEF6MWN4AKPH6CRJNZ1GWE)
- Standalone allocated Cx4 post-maintenance continuation:
  [01KXS08WEZXJNAXAP6C8N2MA9S](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXS08WEZXJNAXAP6C8N2MA9S?taskId=01KXS08WF65TBMV4ARYA1AMA1C&jobId=01KXS08WJHAFA1BQZPTNCBYAWY)

Submitted 2026-07-16 at urgent priority on `ai2/holmes`. Both cells use the
observed-best wide-integration LR `8e-4`, EP1, the canonical global batch, and
no in-loop or on-finish evaluation. Cx4 uses the largest proven-safe legal
microbatch below the projected-over-capacity MB16 shape. Cx8 uses the already
validated MB12 shape. The allocated Cx8 task never started and produced no
checkpoint; it was canceled on 2026-07-16 and requeued with the same semantic
run/checkpoint identity as urgent unallocated, auto-resuming work
(`minRuntime: 0m`). The Cx4 task in the original experiment was initially left
untouched.
Maintenance later preempted Cx4. Resuming the paired work also duplicated the
already-running Cx8 task, so both paired attempts were canceled and Cx4 alone
was submitted in the standalone experiment above. It retains the same semantic
run/checkpoint directory and will resume from durable `step29000`. At
2026-07-17 21:40 UTC it is urgent and queued for four B300s; the workspace is
using all 64 allocated slots.

| Size | Cx | LR | Global batch | GPUs | EP | MB | Accum | Job | W&B | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 480M | 4 | `8e-4` | 524,288 | 4 | 1 | 8 | 2 | [initial](https://beaker.org/ex/01KXMTAR1ZB3ERY8JQ0MH4681B) / [standalone continuation](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXS08WEZXJNAXAP6C8N2MA9S?taskId=01KXS08WF65TBMV4ARYA1AMA1C&jobId=01KXS08WJHAFA1BQZPTNCBYAWY) | [h06m5ls2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h06m5ls2) | finished; final-250M CE `2.305996` |
| 480M | 8 | `8e-4` | 786,432 | 8 | 1 | 12 | 1 | [allocated, canceled](https://beaker.org/ex/01KXMTAR5C5JX0ATP038ECKNWS) / [unallocated replacement](https://beaker.org/ex/01KXPEF7J9GMGAJ1ZJNXMKJ11R) | [d34a9o4t](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/d34a9o4t) | finished; final-250M CE `2.236205` |

## 275M aligned-geometry GDN (`expand_v=2`)

- Model: `geometry_only` in
  [`models/geometry_matched_275m.py`](models/geometry_matched_275m.py)
- Full manifest:
  [`launchers/pretraining/manifests/275m_geometry_gdn_ev2.yaml`](launchers/pretraining/manifests/275m_geometry_gdn_ev2.yaml)
- Smoke manifest:
  [`launchers/pretraining/manifests/275m_geometry_gdn_ev2_smokes.yaml`](launchers/pretraining/manifests/275m_geometry_gdn_ev2_smokes.yaml)
- W&B project/group: `ai2-llm/jacobm-olmoe-ladder` /
  `olmoe3-275m-geometry-gdn-ev2`
- Scheduling exception: urgent, unallocated (`minRuntime: 0m`), and
  `autoResume: true` on two Holmes B300s per task.
- Smoke attempt r1:
  [01KXMY0DJA4W89BWK9SRNHR0WD](https://beaker.org/ex/01KXMY0DJA4W89BWK9SRNHR0WD)
- Smoke capacity attempt r2:
  [01KXMY5XE54S5R9ERXQHKVG5T0](https://beaker.org/ex/01KXMY5XE54S5R9ERXQHKVG5T0)
- Smoke fallback attempt r3:
  [01KXMYJQDN2MADN3AZRRSMFVNW](https://beaker.org/ex/01KXMYJQDN2MADN3AZRRSMFVNW)

Attempt r1 was stopped after model construction exposed that the Beaker token
secret had been omitted. It provides no capacity result. In r2, MB24, MB32,
and MB48 failed with genuine compiled-dry-run CUDA OOMs. Cx1 MB16 passed. The
r3 fallbacks established the remaining production shapes. Every passing row
completed compilation plus 12 optimizer steps with checkpointing and all
evaluator callbacks disabled.

| Cx | Global batch | MB | Accum | Job | W&B | Active memory | Mean TFLOPs/GPU | Status |
|---:|---:|---:|---:|---|---|---:|---:|---|
| 1 | 262,144 | 16 | 1 | [01KXMY5XS9V9XKPK9KEKQZ9NDS](https://beaker.org/ex/01KXMY5XS9V9XKPK9KEKQZ9NDS) | [4zeb0iah](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4zeb0iah) | 197.3 GiB | 368.0 | passed |
| 2 | 393,216 | 12 | 2 | [01KXMYJQSAE4B6SG06E36H196F](https://beaker.org/ex/01KXMYJQSAE4B6SG06E36H196F) | [hj0ip22r](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/hj0ip22r) | 156.8 GiB | 413.2 | passed |
| 4 | 524,288 | 16 | 2 | [01KXMYJQWNW5STRF7STH139N99](https://beaker.org/ex/01KXMYJQWNW5STRF7STH139N99) | [adpjvm8b](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/adpjvm8b) | 197.3 GiB | 387.4 | passed |
| 8 | 786,432 | 16 | 3 | [01KXMYJR0980PYVC9FTWBQD9RH](https://beaker.org/ex/01KXMYJR0980PYVC9FTWBQD9RH) | [rl5kz2u5](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rl5kz2u5) | 197.3 GiB | 471.1 | passed |

The TFLOPs values are arithmetic means of 11 reported step-level samples in
each short smoke. They are useful for detecting gross regressions, but are not
steady-state benchmarks.

### Four-LR sweep

- Beaker experiment:
  [01KXMZAGY82BW0FYT4S6J2TQH8](https://beaker.org/ex/01KXMZAGY82BW0FYT4S6J2TQH8)
- Submitted: 2026-07-16 at urgent priority as unallocated, auto-resuming work
- Scope: all four inherited LRs (`4e-4`, `8e-4`, `1.6e-3`, `3.2e-3`) at each
  of Cx1/Cx2/Cx4/Cx8; 16 tasks, two B300s per task
- Checkpoints: enabled only for production; rolling ephemeral every 500 steps,
  final permanent, `remove=ephemeral_only`
- Evaluation: no in-loop or on-finish evaluators; validation is post hoc

Status on 2026-07-17: all 12 Cx1/Cx2/Cx4 runs finished cleanly and have been
collected with strict final-250M-token windows. Cx1, Cx2, and Cx4 each have an
observed interior best at `1.6e-3` and a valid quadratic fit. Cx4 reaches
`2.474631`, beating the first `expand_v=1` hybrid by `0.003535` CE. The four
Cx8 runs are in their final few percent.

| Cx | LR | MB | Accum | Job | W&B | Status |
|---:|---:|---:|---:|---|---|---|
| 1 | `4e-4` | 16 | 1 | [01KXMZAHBMFQY0JT31JV3ATRXR](https://beaker.org/ex/01KXMZAHBMFQY0JT31JV3ATRXR) | [sa70hegz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sa70hegz) | finished |
| 1 | `8e-4` | 16 | 1 | [01KXMZAHEYW17EP0B3XHJH3RK7](https://beaker.org/ex/01KXMZAHEYW17EP0B3XHJH3RK7) | [3ddxwqks](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3ddxwqks) | finished |
| 1 | `1.6e-3` | 16 | 1 | [01KXMZAJ53C3YS6FNKPKPT0A5T](https://beaker.org/ex/01KXMZAJ53C3YS6FNKPKPT0A5T) | [8zx9zgnw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8zx9zgnw) | finished; observed best |
| 1 | `3.2e-3` | 16 | 1 | [01KXMZAJB819CC1WS154TZP24G](https://beaker.org/ex/01KXMZAJB819CC1WS154TZP24G) | [terfkng8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/terfkng8) | finished |
| 2 | `4e-4` | 12 | 2 | [01KXMZAJQ6BPCQ5TQX783EKPJ9](https://beaker.org/ex/01KXMZAJQ6BPCQ5TQX783EKPJ9) | [oaazdm2h](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/oaazdm2h) | finished |
| 2 | `8e-4` | 12 | 2 | [01KXMZAJXF4J0AMT8G6Q4E4EET](https://beaker.org/ex/01KXMZAJXF4J0AMT8G6Q4E4EET) | [u4cinuz5](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/u4cinuz5) | finished |
| 2 | `1.6e-3` | 12 | 2 | [01KXMZAK152MVAVD4QZS6C05C9](https://beaker.org/ex/01KXMZAK152MVAVD4QZS6C05C9) | [3oqkg24h](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3oqkg24h) | finished; observed best |
| 2 | `3.2e-3` | 12 | 2 | [01KXMZAK4T000W47TJNS1A9GQ6](https://beaker.org/ex/01KXMZAK4T000W47TJNS1A9GQ6) | [pz6377bu](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/pz6377bu) | finished |
| 4 | `4e-4` | 16 | 2 | [01KXMZAK8BPCBXJGES1K4DTXJS](https://beaker.org/ex/01KXMZAK8BPCBXJGES1K4DTXJS) | [7jzlrolc](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7jzlrolc) | finished |
| 4 | `8e-4` | 16 | 2 | [01KXMZAKBNSA46H7GXKVX90EXK](https://beaker.org/ex/01KXMZAKBNSA46H7GXKVX90EXK) | [gwve4pn6](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/gwve4pn6) | finished |
| 4 | `1.6e-3` | 16 | 2 | [01KXMZAKEXB7QDCG4CZ2PT04KH](https://beaker.org/ex/01KXMZAKEXB7QDCG4CZ2PT04KH) | [hwjvw532](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/hwjvw532) | finished; observed best |
| 4 | `3.2e-3` | 16 | 2 | [01KXMZAKJFKA9ERVNDSQ9FENE5](https://beaker.org/ex/01KXMZAKJFKA9ERVNDSQ9FENE5) | [hmjkig0r](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/hmjkig0r) | finished |
| 8 | `4e-4` | 16 | 3 | [node-cordoned attempt](https://beaker.org/ex/01KXMZAKNNNJ2YG3522S5K40YZ) / [automatic resume](https://beaker.org/ex/01KXQ0X75ZJEJGT629XA6W4XWB) | [initial](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7mlzc5x4) / [resume](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9k8mo2q5) | finished; strict combined final-250M CE `2.404090` |
| 8 | `8e-4` | 16 | 3 | [node-cordoned attempt](https://beaker.org/ex/01KXMZAKS5D3978QHT5D911M7C) / [automatic resume](https://beaker.org/ex/01KXQ0X7CRKS3XZ8HXEQ6KKZDN) | [initial](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/wo8raj1p) / [resume](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xdo7p86h) | finished; strict combined final-250M CE `2.389857`; observed best |
| 8 | `1.6e-3` | 16 | 3 | [01KXMZAKWCVBD3GMWHV0TK54NP](https://beaker.org/ex/01KXMZAKWCVBD3GMWHV0TK54NP) | [0x3i869n](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0x3i869n) | finished; final-250M CE `2.390902` |
| 8 | `3.2e-3` | 16 | 3 | [01KXMZAKZNQCX93WENA0DVAWBW](https://beaker.org/ex/01KXMZAKZNQCX93WENA0DVAWBW) | [aholwcgr](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/aholwcgr) | finished; final-250M CE `2.412928` |

The two original Cx8 workers were interrupted together when their Holmes node
was cordoned. Beaker's configured `autoResume` created the automatic resumes
above. Two redundant explicit resumes were detected before their next
checkpoint save and canceled; only the automatic resume for each LR remains a
writer to its checkpoint directory.

## Cx8 first-hybrid midtraining

- Manifest:
  [`launchers/midtraining/manifests/275m_hybrid_gdn_ev1_cx8.yaml`](launchers/midtraining/manifests/275m_hybrid_gdn_ev1_cx8.yaml)
- Launcher:
  [`launchers/midtraining/launch_midtraining.py`](launchers/midtraining/launch_midtraining.py)
- Beaker experiments: [r1](https://beaker.org/ex/01KXPEFRXGZSS05XXDAJ9PFSZ5),
  [credential-fixed r2](https://beaker.org/ex/01KXPEQBYPY9YA87QJ1YS8BTYV)
- Source: permanent final checkpoint `step42954` from the observed-best 275M
  `expand_v=1` hybrid Cx8 run at pretraining LR `1.6e-3`
- Destination:
  `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/midtraining/mt-275m-intwide-hybrid-gdn-ev1-cx8-lr1p6e-4-r1`
- Scheduling: urgent unallocated, `minRuntime: 0m`, `autoResume: true`, four
  Holmes B300s, EP1
- Recipe: 100B tokens at 8K; global batch 1,048,576 tokens / 128 sequences;
  rank MB8; four-way accumulation; weight-only initialization with a fresh
  optimizer; 2,000-step linear warmup into constant LR `1.6e-4`
- Checkpoints: rolling ephemeral every 500 steps, final permanent,
  `remove=ephemeral_only`; no automatic permanent-checkpoint cleanup
- Evaluation: no in-loop or on-finish evaluators; full validation is post hoc

The r1 task was canceled before checkpoint loading or step 1 after its GCS
source-mixture scan exposed a missing Google credential injection. It wrote no
checkpoint. The wrapper and manifest now use the established
`jacobm_GOOGLE_CREDENTIALS` path, and r2 keeps the same semantic run/checkpoint
identity.

| Model | Job | W&B | Status |
|---|---|---|---|
| 275M `expand_v=1` hybrid Cx8 | [r1 canceled](https://beaker.org/ex/01KXPEFSNRXYB5N3KAAEWJD2ZR) / r2 | [1keo2hz6](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1keo2hz6) | finished; 100B; final `step95368`; validation finished |
| 480M `expand_v=1` hybrid Cx8 | [01KXS955Q6RZQQM7PEGSWF3XDT](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXS955Q6RZQQM7PEGSWF3XDT?taskId=01KXS955QG0FG968ME9CAZ8RB2&jobId=01KXS955TSVZE9HEMCNDJP72QP) | [mnp9rv5l](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mnp9rv5l) | finished; 100.001B; final `step95368`; validation finished |

The 480M continuation loads the permanent first-hybrid Cx8 PT checkpoint
`step81069` weight-only, starts a fresh optimizer at LR `8e-5`, and otherwise
uses the same 100B-token midtraining recipe as 275M: 8K sequences, a
1,048,576-token global batch, EP1, rank MB8, accumulation 4, and a fixed
2,000-step warmup into constant LR. Its destination is
`/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/midtraining/mt-480m-intwide-hybrid-gdn-ev1-cx8-lr8e-5-r1`.

## 275M geometry + NoPE

- Model variant: `geometry_275m_gdn_ev2_nope`
- Dedicated Beaker wrapper:
  `src/scripts/train/jacobm_olmoe3_geometry_275m_nope_beaker.sh`
- Scaling-smoke manifest:
  [`launchers/pretraining/manifests/275m_geometry_gdn_ev2_nope_scaling_smokes.yaml`](launchers/pretraining/manifests/275m_geometry_gdn_ev2_nope_scaling_smokes.yaml)
- Scaling-smoke launcher:
  [`launchers/pretraining/launch_275m_geometry_gdn_ev2_nope_scaling_smokes.sh`](launchers/pretraining/launch_275m_geometry_gdn_ev2_nope_scaling_smokes.sh)
- Scaling-smoke Beaker work:
  [01KXS9BR4NKK06H7HJ5X73KQP2](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXS9BR4NKK06H7HJ5X73KQP2)
- W&B group: `ai2-llm/jacobm-olmoe-ladder` /
  `olmoe3-275m-geometry-gdn-ev2-nope`

The NoPE variant changes only the two full-attention blocks at layers 4 and 9.
It retains the geometry model's 290,782,080 active, 226,556,800 active
non-embedding, and 3,136,314,240 total parameters. Its first launch is a
checkpoint-free, evaluator-free, compiled 12-step DDP scaling study at the Cx1
and Cx8 endpoints on 2/4/8 B300s. All tasks use EP1 and exact canonical global
batches. The production LR-sweep GPU layout below was selected from measured
wall-clock throughput rather than assumed from per-GPU TFLOPs.

Submitted on 2026-07-18 at urgent priority as unallocated work
(`minRuntime: 0m`, `autoResume: true`) on Holmes. The six tasks request 28
B300s at full concurrency:

| Cx | GPUs | Global batch | Rank MB | Accum | Mean TFLOPs/GPU | Total-token speedup | Job | W&B | Status |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | 2 | 262,144 | 16 | 1 | 430.9 | 1.00x | [01KXS9BR89EBX1T7SVAC0XFGXA](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXS9BR4NKK06H7HJ5X73KQP2?taskId=01KXS9BR4XT8AZKXWQFPTR3GVA&jobId=01KXS9BR89EBX1T7SVAC0XFGXA) | [vow8stjm](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vow8stjm) | passed |
| 1 | 4 | 262,144 | 8 | 1 | 322.1 | 1.50x | [01KXS9BRCBFWA5TMYXPTYR7H8C](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXS9BR4NKK06H7HJ5X73KQP2?taskId=01KXS9BR8F979NJ9H3R8ATNFDC&jobId=01KXS9BRCBFWA5TMYXPTYR7H8C) | [mxghr6je](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mxghr6je) | passed; production choice |
| 1 | 8 | 262,144 | 4 | 1 | 210.5 | 1.95x | [01KXS9BRFNZ4AWAFHPSZ62SZ0M](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXS9BR4NKK06H7HJ5X73KQP2?taskId=01KXS9BRCEYMW0MYBFS9PNQDWR&jobId=01KXS9BRFNZ4AWAFHPSZ62SZ0M) | [lxovdnqd](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/lxovdnqd) | passed |
| 8 | 2 | 786,432 | 16 | 3 | 422.3 | 1.00x | [01KXS9BRK0ZDX6SXRRZTAJNE98](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXS9BR4NKK06H7HJ5X73KQP2?taskId=01KXS9BRFSF0Z4663HYRA65RE8&jobId=01KXS9BRK0ZDX6SXRRZTAJNE98) | [bjevyjbh](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/bjevyjbh) | passed |
| 8 | 4 | 786,432 | 12 | 2 | 384.9 | 1.82x | [01KXS9BRPEYT77MQFME8C8YH3H](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXS9BR4NKK06H7HJ5X73KQP2?taskId=01KXS9BRK4DVHSYBFZNMEPQ5ZE&jobId=01KXS9BRPEYT77MQFME8C8YH3H) | [5ie1n90w](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5ie1n90w) | passed |
| 8 | 8 | 786,432 | 12 | 1 | 362.0 | 3.43x | [01KXS9BRT1QBH9DCMVJ89NW0TF](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXS9BR4NKK06H7HJ5X73KQP2?taskId=01KXS9BRPJKPSNXKW5JESDARBQ&jobId=01KXS9BRT1QBH9DCMVJ89NW0TF) | [27qhyqgo](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/27qhyqgo) | passed; production choice |

Mean TFLOPs/GPU is the arithmetic mean of the 11 reported speed samples,
matching the earlier geometry-smoke reporting convention. Total-token speedup
uses mean per-device TPS multiplied by world size and is normalized within
each Cx. Cx1 uses four GPUs in production because eight GPUs provide only
1.31x more throughput for twice the GPUs. Cx8 uses eight because it is 1.88x
faster than four. Cx2/Cx4 use four GPUs, removing accumulation while retaining
the already validated MB12/MB16 per-rank shapes.

The independent production manifest is
[`launchers/pretraining/manifests/275m_geometry_gdn_ev2_nope.yaml`](launchers/pretraining/manifests/275m_geometry_gdn_ev2_nope.yaml);
its launcher is
[`launchers/pretraining/launch_275m_geometry_gdn_ev2_nope.sh`](launchers/pretraining/launch_275m_geometry_gdn_ev2_nope.sh).
It contains the four inherited LRs (`4e-4`, `8e-4`, `1.6e-3`, `3.2e-3`) at
Cx1/2/4/8, requesting 80 B300s at full concurrency.

The production sweep was submitted on 2026-07-18 as urgent unallocated work:
[01KXSABHBX2Z4G1JFV8W1PN6AN](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSABHBX2Z4G1JFV8W1PN6AN).

Final status: all 16 tasks exited 0 and all four Cx sweeps are bracketed.
Observed bests under the strict final-250M metric are Cx1 `8e-4` /
`2.712805`, Cx2 `1.6e-3` / `2.585179`, Cx4 `8e-4` / `2.477784`, and Cx8
`8e-4` / `2.391953`. Relative to otherwise-identical RoPE geometry, NoPE is
`+0.004924`, `+0.006217`, `+0.003153`, and `+0.002096` CE at Cx1/2/4/8.
The completed U-plot and observed-best summary are under
[`plots/pretraining/geometry_gdn_ev2_nope/`](plots/pretraining/geometry_gdn_ev2_nope/).

| Cx | LR | GPUs | MB | Accum | Job | W&B | Status |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | `4e-4` | 4 | 8 | 1 | [01KXSABHFQKSGQZJ523MSA68V1](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSABHBX2Z4G1JFV8W1PN6AN?taskId=01KXSABHC318G5N5Y7A8YNG5W2&jobId=01KXSABHFQKSGQZJ523MSA68V1) | [52ph1l67](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/52ph1l67) | finished |
| 1 | `8e-4` | 4 | 8 | 1 | [01KXSABHK67TJHNGDW33P7PT16](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSABHBX2Z4G1JFV8W1PN6AN?taskId=01KXSABHFWP0MFE5G037D9BXRS&jobId=01KXSABHK67TJHNGDW33P7PT16) | [epdjswap](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/epdjswap) | finished; observed best |
| 1 | `1.6e-3` | 4 | 8 | 1 | [01KXSABHPJMYFFT1T96H98D5J1](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSABHBX2Z4G1JFV8W1PN6AN?taskId=01KXSABHKB3DPXFCBAWNAZG115&jobId=01KXSABHPJMYFFT1T96H98D5J1) | [8mnuuecq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8mnuuecq) | finished |
| 1 | `3.2e-3` | 4 | 8 | 1 | [01KXSABHTFS04FEGPH1NAD92RY](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSABHBX2Z4G1JFV8W1PN6AN?taskId=01KXSABHPQAGSY0SFA0T26MCV1&jobId=01KXSABHTFS04FEGPH1NAD92RY) | [7gfls4r6](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7gfls4r6) | finished |
| 2 | `4e-4` | 4 | 12 | 1 | [01KXSABHZ85TAGR88MJYXKQBEX](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSABHBX2Z4G1JFV8W1PN6AN?taskId=01KXSABHTJM6X9EKGHDER3XEVD&jobId=01KXSABHZ85TAGR88MJYXKQBEX) | [wpbz1ar9](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/wpbz1ar9) | finished |
| 2 | `8e-4` | 4 | 12 | 1 | [01KXSABJ2YNA80PFV3FHAKMRZK](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSABHBX2Z4G1JFV8W1PN6AN?taskId=01KXSABHZEYSF5QY7MJP0DWCX9&jobId=01KXSABJ2YNA80PFV3FHAKMRZK) | [7u4epzt6](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7u4epzt6) | finished |
| 2 | `1.6e-3` | 4 | 12 | 1 | [01KXSABJ68WBX4TVSTR56BVZ1A](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSABHBX2Z4G1JFV8W1PN6AN?taskId=01KXSABJ34R4AE29AGHGXP65NN&jobId=01KXSABJ68WBX4TVSTR56BVZ1A) | [gjmz37ct](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/gjmz37ct) | finished; observed best |
| 2 | `3.2e-3` | 4 | 12 | 1 | [01KXSABJBS8QP3R5HEMG6JSWEN](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSABHBX2Z4G1JFV8W1PN6AN?taskId=01KXSABJ6A8DE73DQ51KM26C6P&jobId=01KXSABJBS8QP3R5HEMG6JSWEN) | [xahm1pbt](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xahm1pbt) | finished |
| 4 | `4e-4` | 4 | 16 | 1 | [01KXSABJFAQ0SMEQH6PR4Y8C19](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSABHBX2Z4G1JFV8W1PN6AN?taskId=01KXSABJBWZE65DRH4M21WB0T9&jobId=01KXSABJFAQ0SMEQH6PR4Y8C19) | [pmfco9gy](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/pmfco9gy) | finished; final-250M CE `2.494023` |
| 4 | `8e-4` | 4 | 16 | 1 | [01KXSABJKMVAAY3KZ7HDZ284M9](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSABHBX2Z4G1JFV8W1PN6AN?taskId=01KXSABJFCB5F0HWX0BR2ZAENY&jobId=01KXSABJKMVAAY3KZ7HDZ284M9) | [k5mjm4ev](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/k5mjm4ev) | finished; observed best |
| 4 | `1.6e-3` | 4 | 16 | 1 | [01KXSABJQM8P4QM4HNQBJP3CSV](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSABHBX2Z4G1JFV8W1PN6AN?taskId=01KXSABJKP9QQNCAM9YT0NWRNP&jobId=01KXSABJQM8P4QM4HNQBJP3CSV) | [4x00n8lj](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4x00n8lj) | finished |
| 4 | `3.2e-3` | 4 | 16 | 1 | [01KXSABJVYPN084NPHSDR2QV95](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSABHBX2Z4G1JFV8W1PN6AN?taskId=01KXSABJQPQRZSJSPTNMY0MRK5&jobId=01KXSABJVYPN084NPHSDR2QV95) | [z1lw0z2i](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/z1lw0z2i) | finished; final-250M CE `2.503428` |
| 8 | `4e-4` | 8 | 12 | 1 | [01KXSABK0Y35G8W2EBSNT97JBS](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSABHBX2Z4G1JFV8W1PN6AN?taskId=01KXSABJW0PR83WZ70FAB6W2H0&jobId=01KXSABK0Y35G8W2EBSNT97JBS) | [t76b5xjy](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/t76b5xjy) | finished |
| 8 | `8e-4` | 8 | 12 | 1 | [01KXSABK5B78Z8CTT7944NDYAF](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSABHBX2Z4G1JFV8W1PN6AN?taskId=01KXSABK10HX9VBTHCTKGHJ12P&jobId=01KXSABK5B78Z8CTT7944NDYAF) | [d29gx1x9](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/d29gx1x9) | finished; observed best |
| 8 | `1.6e-3` | 8 | 12 | 1 | [01KXSABK8R6YKTP2JQB0VG0SWR](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSABHBX2Z4G1JFV8W1PN6AN?taskId=01KXSABK5DR58R40YGF660FXPJ&jobId=01KXSABK8R6YKTP2JQB0VG0SWR) | [n8kkr1y8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/n8kkr1y8) | finished |
| 8 | `3.2e-3` | 8 | 12 | 1 | [01KXSABKC1YQXYYHJ2ECTYMGJW](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSABHBX2Z4G1JFV8W1PN6AN?taskId=01KXSABK8TBRFE0H7GKK82HRYT&jobId=01KXSABKC1YQXYYHJ2ECTYMGJW) | [qhjvjwcu](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qhjvjwcu) | finished |

## 275M geometry + NoPE + gated attention

- Model variant: `geometry_275m_gdn_ev2_nope_gated`
- Architecture and parameter audit:
  [`ATTENTION_GATING_275M.md`](ATTENTION_GATING_275M.md)
- Scheduling: urgent unallocated Holmes B300s, `minRuntime: 0m`,
  non-preemptible, auto-resuming
- Capacity smoke:
  [01KXSZKW55FZKJSD9CPFW4WZ82](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSZKW55FZKJSD9CPFW4WZ82)
- Full LR sweep:
  [01KXT07N6AGD1S0REJA3TH897G](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXT07N6AGD1S0REJA3TH897G)

The smoke passed all four production shapes on 2026-07-18: Cx1 MB8,
Cx2 MB12, and Cx4 MB16 on four GPUs, plus Cx8 MB12 on eight GPUs. Every task
reached step 11 and exited 0 without writing a checkpoint. The promoted sweep
contains `4e-4`, `8e-4`, `1.6e-3`, and `3.2e-3` for every Cx, requesting 80
GPUs at full concurrency with no in-loop or on-finish evaluators. All 16
production tasks subsequently exited 0.

All four curves are bracketed. Observed bests are Cx1 `8e-4` / `2.711104`,
Cx2 `1.6e-3` / `2.580768`, Cx4 `8e-4` / `2.476065`, and Cx8 `8e-4` /
`2.390397`. Gating improves over ungated NoPE by `0.001701`, `0.004411`,
`0.001719`, and `0.001556` CE at Cx1/2/4/8, respectively. It remains within
`0.000540`–`0.003223` CE of the otherwise-identical RoPE geometry control.
Plots and the exact run table are under
[`plots/pretraining/geometry_gdn_ev2_nope_gated/`](plots/pretraining/geometry_gdn_ev2_nope_gated/)
and
[`results/pretraining/geometry_gdn_ev2_nope_gated/`](results/pretraining/geometry_gdn_ev2_nope_gated/).

## 275M geometry + RoPE + gated attention

- Model variant: `geometry_275m_gdn_ev2_rope_gated`
- Parent: the completed `geometry_275m_gdn_ev2_nope_gated` sweep
- Isolated change: restore RoPE only on full-attention layers 4 and 9
- RoPE config: theta `500000`, full precision, no scaling
- Attention gate: elementwise, full precision; 8 Q / 4 KV heads retained
- Parameters: 292,092,800 active; 227,867,520 active non-embedding;
  3,137,624,960 total
- Smoke:
  [01KY0G559WGWP50DE05B8DJQGY](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0G559WGWP50DE05B8DJQGY)
- Full LR sweep:
  [01KY0GVX8SM5998GFMGAKR3AQ6](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0GVX8SM5998GFMGAKR3AQ6)

A strict local construction check normalized the new profile by removing RoPE
from layers 4 and 9 and then compared it for exact equality with the gated-NoPE
profile. The configs were identical after that normalization, including all
parameter counts, mixer placement, MoE widths, head geometry, norms, gating,
and initialization.

The checkpoint-free Cx4 MB16 smoke ran on four Holmes B300s, reached all 11
steps, and exited 0. Its final-five median was about 385.6 TFLOPs/GPU. The
subsequent production sweep reuses the gated-NoPE optimizer batches and launch
shapes: Cx1/Cx2/Cx4 use four GPUs with MB8/12/16, while Cx8 uses eight GPUs
with MB12. Every Cx runs `4e-4`, `8e-4`, `1.6e-3`, and `3.2e-3`; in-loop and
on-finish evaluation are disabled, and validation will be backfilled after
training. All 16 urgent unallocated Holmes tasks finished cleanly. Observed
best LRs at Cx1/2/4/8 are `1.6e-3`, `1.6e-3`, `8e-4`, and `1.6e-3`, with
final-250M CEs `2.691980`, `2.573449`, `2.470110`, and `2.386206`,
respectively.

| Cx | LR | GPUs | MB | Job | W&B | Launch status |
|---:|---:|---:|---:|---|---|---|
| 1 | `4e-4` | 4 | 8 | [01KY0GVXCD6YC1WRZZEH4EWPJ6](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0GVX8SM5998GFMGAKR3AQ6?taskId=01KY0GVX91YPCN1MKRYRX30QXK&jobId=01KY0GVXCD6YC1WRZZEH4EWPJ6) | [kd3fyszi](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kd3fyszi) | finished |
| 1 | `8e-4` | 4 | 8 | [01KY0GVXGN7TFVS60S6KKBJZSW](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0GVX8SM5998GFMGAKR3AQ6?taskId=01KY0GVXCJJZVWEA8TAN6X2R36&jobId=01KY0GVXGN7TFVS60S6KKBJZSW) | [ezdsfb9n](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ezdsfb9n) | finished |
| 1 | `1.6e-3` | 4 | 8 | [01KY0GVXM1TX7J3X2ZJRCWNEJR](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0GVX8SM5998GFMGAKR3AQ6?taskId=01KY0GVXGSENMC2XT3DA0F4ND1&jobId=01KY0GVXM1TX7J3X2ZJRCWNEJR) | [eo5bm8gw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/eo5bm8gw) | finished; observed best |
| 1 | `3.2e-3` | 4 | 8 | [01KY0GVXQG9C9CD1FKJMY842JH](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0GVX8SM5998GFMGAKR3AQ6?taskId=01KY0GVXM5H2W0P8X4SP124BPH&jobId=01KY0GVXQG9C9CD1FKJMY842JH) | [l4tp6qmo](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/l4tp6qmo) | finished |
| 2 | `4e-4` | 4 | 12 | [01KY0GVXTY4XCZ5QVCF5GMJBSY](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0GVX8SM5998GFMGAKR3AQ6?taskId=01KY0GVXQMHYW38WYKWR3ENTWD&jobId=01KY0GVXTY4XCZ5QVCF5GMJBSY) | [7gmi969q](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7gmi969q) | finished |
| 2 | `8e-4` | 4 | 12 | [01KY0GVXYJV563V32PGRHCB1SX](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0GVX8SM5998GFMGAKR3AQ6?taskId=01KY0GVXV3C83J7JE9JT1R6Z7D&jobId=01KY0GVXYJV563V32PGRHCB1SX) | [0ovig11c](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0ovig11c) | finished |
| 2 | `1.6e-3` | 4 | 12 | [01KY0GVY20SFVSGWB2TC5T2J06](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0GVX8SM5998GFMGAKR3AQ6?taskId=01KY0GVXYNY7331H1KTRPRQ3CJ&jobId=01KY0GVY20SFVSGWB2TC5T2J06) | [8mkt4xpz](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8mkt4xpz) | finished; observed best |
| 2 | `3.2e-3` | 4 | 12 | [01KY0GVY59R8FGMEZTBTKNG418](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0GVX8SM5998GFMGAKR3AQ6?taskId=01KY0GVY22MEKVE40PVGA83HXN&jobId=01KY0GVY59R8FGMEZTBTKNG418) | [66u6ekx2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/66u6ekx2) | finished |
| 4 | `4e-4` | 4 | 16 | [01KY0GVY97Y0EJEJ4Y5FBWF0SY](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0GVX8SM5998GFMGAKR3AQ6?taskId=01KY0GVY5BTCKCYKJQ2PBCKXN7&jobId=01KY0GVY97Y0EJEJ4Y5FBWF0SY) | [o1p6n2v7](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/o1p6n2v7) | finished |
| 4 | `8e-4` | 4 | 16 | [01KY0GVYCDXS3B4WKQTM711VME](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0GVX8SM5998GFMGAKR3AQ6?taskId=01KY0GVY980Z2S7X577M80N7DC&jobId=01KY0GVYCDXS3B4WKQTM711VME) | [iqxc5n9x](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/iqxc5n9x) | finished; observed best |
| 4 | `1.6e-3` | 4 | 16 | [01KY0GVYFXAK12CZ2FR2BHTMK8](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0GVX8SM5998GFMGAKR3AQ6?taskId=01KY0GVYCEH9XY6FBEFMVDV8YD&jobId=01KY0GVYFXAK12CZ2FR2BHTMK8) | [n6suaxul](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/n6suaxul) | finished |
| 4 | `3.2e-3` | 4 | 16 | [01KY0GVYK3FQN3CY90B8BJ4TN1](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0GVX8SM5998GFMGAKR3AQ6?taskId=01KY0GVYFYYMJSP2KDQ9N9J7F1&jobId=01KY0GVYK3FQN3CY90B8BJ4TN1) | [clfmsyx8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/clfmsyx8) | finished |
| 8 | `4e-4` | 8 | 12 | [01KY0GVYPDNRZH73D3PV9945KD](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0GVX8SM5998GFMGAKR3AQ6?taskId=01KY0GVYK43DFNPTYBR9BC34A5&jobId=01KY0GVYPDNRZH73D3PV9945KD) | [y1dh1cb5](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/y1dh1cb5) | finished |
| 8 | `8e-4` | 8 | 12 | [01KY0GVYSK43FX8NVFZ0BWCV02](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0GVX8SM5998GFMGAKR3AQ6?taskId=01KY0GVYPEWRV03209HY9N6NVE&jobId=01KY0GVYSK43FX8NVFZ0BWCV02) | [65bsc0wk](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/65bsc0wk) | finished |
| 8 | `1.6e-3` | 8 | 12 | [01KY0GVYWZ962WAHMNH6TP3DKT](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0GVX8SM5998GFMGAKR3AQ6?taskId=01KY0GVYSN9RQEPGQN05QXNCJD&jobId=01KY0GVYWZ962WAHMNH6TP3DKT) | [8rgf3myq](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8rgf3myq) | finished; observed best |
| 8 | `3.2e-3` | 8 | 12 | [01KY0GVZ0776A482S2CJ2XNZ3H](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0GVX8SM5998GFMGAKR3AQ6?taskId=01KY0GVYX0MDYP3X29ENBP97FY&jobId=01KY0GVZ0776A482S2CJ2XNZ3H) | [klgge8er](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/klgge8er) | finished |

## Larger geometry + RoPE + gated attention

- Model variant: `geometry_matched_gdn_ev2_rope_gated`
- Manifest:
  [`launchers/pretraining/manifests/geometry_matched_scale_rope_gated_full.yaml`](launchers/pretraining/manifests/geometry_matched_scale_rope_gated_full.yaml)
- Launcher:
  [`launchers/pretraining/launch_geometry_matched_scale_rope_gated_full.sh`](launchers/pretraining/launch_geometry_matched_scale_rope_gated_full.sh)
- Submission record:
  [`launchers/pretraining/generated/geometry_matched_scale_rope_gated_full_unallocated_submissions.json`](launchers/pretraining/generated/geometry_matched_scale_rope_gated_full_unallocated_submissions.json)
- Commit: `2242550320d8f48a28c532a078767d13dd0c3829`
- Scheduling: urgent unallocated Holmes B300s, `minRuntime: 0m`,
  non-preemptible, auto-resuming

The 12 transferred-LR runs were submitted on 2026-07-21 in the requested
Cx-major order: Cx1, Cx2, Cx4, then Cx8, with 480M, 810M, then 1.2B within
each Cx. This is the same architecture and training recipe as the completed
275M RoPE-plus-gate interaction control, scaled to the geometry-matched larger
models. A strict construction audit confirmed that each model differs from its
gated-NoPE counterpart only by restoring RoPE to full-attention layers.

The compact layout requests 124 GPUs. The 480M and 810M models use EP1; all
1.2B models use EP8 with `sync_1d`. Training retains rolling ephemeral
checkpoints every 500 steps and the final checkpoint. In-loop and on-finish
evaluation are disabled; validation will be backfilled after training. All
initialized W&B IDs are registered in the plotting collector.

| Cx | Size | LR | GPUs | EP | MB | Accum | Beaker work | State |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 480M | `1.2e-3` | 4 | 1 | 8 | 1 | [01KY1DKVJJ0ECA4QS0SENKH49E](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DKVJJ0ECA4QS0SENKH49E) | finished |
| 1 | 810M | `6e-4` | 8 | 1 | 4 | 1 | [01KY1DKYH8Q1HAX2922D7AJA4E](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DKYH8Q1HAX2922D7AJA4E) | finished |
| 1 | 1.2B | `4e-4` | 8 | 8 | 4 | 1 | [01KY1DM1J6C028TJNQART75FY1](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DM1J6C028TJNQART75FY1) | finished |
| 2 | 480M | `9e-4` | 4 | 1 | 12 | 1 | [01KY1DM576YRXHEVF13MTVN9VJ](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DM576YRXHEVF13MTVN9VJ) | finished |
| 2 | 810M | `5.6e-4` | 8 | 1 | 6 | 1 | [01KY1DM89PS7K4BCYTX0QFVFXE](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DM89PS7K4BCYTX0QFVFXE) | finished |
| 2 | 1.2B | `6e-4` | 16 | 8 | 3 | 1 | [01KY1DMBM1CBJTPK3W3W372B56](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DMBM1CBJTPK3W3W372B56) | failed |
| 4 | 480M | `8e-4` | 4 | 1 | 8 | 2 | [01KY1DMF80H4S3G45RCDE5D1TR](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DMF80H4S3G45RCDE5D1TR) | finished |
| 4 | 810M | `4e-4` | 8 | 1 | 4 | 2 | [01KY1DMJG0DPAN2M8E9YHEQPZT](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DMJG0DPAN2M8E9YHEQPZT) | finished |
| 4 | 1.2B | `3e-4` | 16 | 8 | 4 | 1 | [01KY1DMNKHZ8AND4FYA9WFHRYA](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DMNKHZ8AND4FYA9WFHRYA) | finished |
| 8 | 480M | `8e-4` | 8 | 1 | 12 | 1 | [01KY1DMRSHXB5RTZGD39NXD2G4](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DMRSHXB5RTZGD39NXD2G4) | finished |
| 8 | 810M | `4e-4` | 8 | 1 | 6 | 2 | [01KY1DMW31S6YVCDN479XYRE7S](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DMW31S6YVCDN479XYRE7S) | finished; final-250M CE `2.104806` |
| 8 | 1.2B | `4e-4` | 32 | 8 | 3 | 1 | [01KY1DN08V4T6HMJDDCPPKHGSN](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DN08V4T6HMJDDCPPKHGSN) | finished; final-250M CE `2.029514` |

The initial allocated submission was canceled on 2026-07-21 before any job
reached `started`; no checkpoint directory was created. Its 12 IDs remain in
[`geometry_matched_scale_rope_gated_full_submissions.json`](launchers/pretraining/generated/geometry_matched_scale_rope_gated_full_submissions.json)
as canceled history and must not be resumed.

## Larger geometry + NoPE capacity smokes

- Model variant: `geometry_matched_gdn_ev2_nope`
- Manifest:
  [`launchers/pretraining/manifests/geometry_matched_scale_nope_smokes.yaml`](launchers/pretraining/manifests/geometry_matched_scale_nope_smokes.yaml)
- Launcher:
  [`launchers/pretraining/launch_geometry_matched_scale_nope_smokes.py`](launchers/pretraining/launch_geometry_matched_scale_nope_smokes.py)
- Detailed parameter, performance, and ETA record:
  [`GEOMETRY_MATCHED_SCALE.md`](GEOMETRY_MATCHED_SCALE.md)
- Scheduling: urgent unallocated Holmes B300s, `minRuntime: 0m`,
  non-preemptible, auto-resuming
- Workload: compiled dry run plus 12 optimizer steps; checkpointing and all
  evaluators disabled

All ten valid settings passed. Performance below is the median of the final
five step-level samples.

| Size | Cx | GPUs | EP | MB | Accum | Beaker work | W&B | TFLOPs/GPU | Status |
|---|---:|---:|---:|---:|---:|---|---|---:|---|
| 480M | 1 | 4 | 1 | 8 | 1 | [01KXSCD0AFBF1YVYJPAKS4DRX6](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSCD0AFBF1YVYJPAKS4DRX6) | [wllr6m1g](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/wllr6m1g) | 379.8 | passed |
| 480M | 8 | 4 | 1 | 12 | 2 | [01KXSCD3EKY3RWEK20CHV7XC7N](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSCD3EKY3RWEK20CHV7XC7N) | [xch3s5bw](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xch3s5bw) | 460.8 | passed |
| 480M | 8 | 8 | 1 | 12 | 1 | [01KXSD1SSCA5NCGYDP9Z2XB5ER](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSD1SSCA5NCGYDP9Z2XB5ER) | [s8ubt7sh](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/s8ubt7sh) | 416.0 | passed |
| 810M | 1 | 8 | 1 | 4 | 1 | [01KXSCY02FZRQ84KRAQ25RXGRX](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSCY02FZRQ84KRAQ25RXGRX) | [0xfsc1vs](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0xfsc1vs) | 312.6 | passed |
| 810M | 8 | 8 | 1 | 6 | 2 | [01KXSCDA2JDTF73BJSRP7DMM3Y](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSCDA2JDTF73BJSRP7DMM3Y) | [kfm4ynjr](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kfm4ynjr) | 447.9 | passed |
| 810M | 8 | 16 | 1 | 6 | 1 | [01KXSD1WZ9FYXHTPV1C1JYJ1V1](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSD1WZ9FYXHTPV1C1JYJ1V1) | [o54vnqvg](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/o54vnqvg) | 380.3 | passed |
| 1.2B | 1 | 8 | 8 | 4 | 1 | [01KXSCDD4VE2HAV2T4Y8V1W5T2](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSCDD4VE2HAV2T4Y8V1W5T2) | [g1d4fcd7](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/g1d4fcd7) | 409.5 | passed |
| 1.2B | 8 | 8 | 8 | 6 | 2 | [01KXSCDGHHF6DKFZFADG1RQN0P](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSCDGHHF6DKFZFADG1RQN0P) | [c69nxyn3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/c69nxyn3) | 440.5 | passed |
| 1.2B | 8 | 16 | 8 | 6 | 1 | [01KXSD5GHS1MAFWTY1B20ZJ9XK](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSD5GHS1MAFWTY1B20ZJ9XK) | [kfm18iir](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kfm18iir) | 410.8 | passed |
| 1.2B | 8 | 32 | 8 | 3 | 1 | [01KXSD5KVEQNQ4J53CS6XPSGRA](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSD5KVEQNQ4J53CS6XPSGRA) | [owvnz62c](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/owvnz62c) | 315.6 | passed |

The first six submissions installed the checkout into the image and produced a
CUDA/TransformerEngine ABI mismatch before model execution. They wrote no
checkpoint and are excluded from capacity results. The corrected jobs use the
image environment unchanged. The first corrected 810M Cx1 worker then
segfaulted after optimizer initialization; its identical r3 retry above
passed, classifying that attempt as transient infrastructure rather than an
OOM or model failure.

The selected production layout uses eight GPUs for all four 480M points, 16
for all four 810M points, 16 for 1.2B Cx1/Cx2, and 32 for 1.2B Cx4/Cx8. The
12 urgent unallocated runs were submitted on 2026-07-18, requesting 192 peak
GPUs and using the transferred wide-integration LRs. Exact rank microbatches,
LRs, Beaker IDs, ETAs, and estimated GPU-hours are in
[`GEOMETRY_MATCHED_SCALE.md`](GEOMETRY_MATCHED_SCALE.md). All are pinned to
commit `fcf1c1b8828a3bddd0bad477a5c4055e63b0275f`, retain rolling ephemeral
checkpoints every 500 steps, and disable in-loop/on-finish evaluators.

Status at 2026-07-23 17:16 UTC: all 12 formal cells have finished results. The 810M Cx8 has strict final-250M CE
`2.119848`, versus `2.104939` for wide integration and `2.095585` for the
first hybrid. The 1.2B Cx4 strict final-250M CE is `2.107767`. The
user-requeued 1.2B Cx8 attempt resumed from the existing
checkpoint, reached step 17,644, and failed for a third time on the same
`Non-finite total grad norm` assertion. Its targeted
[diagnostic continuation](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY097ZF4F18F16P71XZWJVX0)
then reproduced a broad NaN at step 17,592: essentially every DP gradient was
NaN on every rank. This is not an isolated tensor or one safely skippable step.
For exact LR comparability, a new from-scratch reproduction with the identical
32-GPU/EP8/MB3 layout and a distinct checkpoint directory was trained in
[01KY0CM4HKG0R4H352N2SQV6P1](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0CM4HKG0R4H352N2SQV6P1).
It finished all 185.759B tokens at step 236,205 without reproducing the
collapse. Its strict final-250M CE is `2.034305`; the formal results registry
now points to W&B run `hiokrpag` rather than the failed original trajectory.

The equivalent larger NoPE-plus-gated-attention launcher and complete 12-cell
manifest were structurally and count validated, then submitted on 2026-07-18
after the completed 275M gated sweep improved on ungated NoPE at all four Cx.
At the 2026-07-23 17:16 UTC refresh, all 12 cells are finished. Strict
final-250M CE is `2.191179` for 810M Cx4, `2.114516` for 810M Cx8,
`2.273007` for 1.2B Cx1, `2.188236` for 1.2B Cx2, `2.108263` for 1.2B Cx4,
and `2.037147` for 1.2B Cx8. The 1.2B Cx2 continuation resumed from diagnostic
`step21500` and finished cleanly. All four 1.2B cells were initially
rejected before training by the shared 6% active-parameter-delta guard: their
audited gated delta is 6.1155%. The guard now permits up to 6.2% only for
gated variants while leaving the 6% limit unchanged for ungated models. The
1.2B Cx2 retry first stopped on a non-finite total grad norm at step 16,654.
The identical retry stopped again at step 20,969 on the same assertion. Its
targeted
[diagnostic continuation](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY098CNTQ5E0WTZTVJG8KTXR)
trained cleanly through the previous failure point and stopped as intended at
step 21,500. The durable `step21500` checkpoint then resumed normally in the
current production continuation.
Exact retry links are in
[`GEOMETRY_MATCHED_SCALE.md`](GEOMETRY_MATCHED_SCALE.md). The wave otherwise
retains the identical 192-GPU peak layout and transferred LRs.

## Post-training validation backfills

- Manifest: [`launchers/validation/manifests/275m_hybrid_geometry_full.yaml`](launchers/validation/manifests/275m_hybrid_geometry_full.yaml)
- Launcher: [`launchers/validation/launch_backfills.py`](launchers/validation/launch_backfills.py)
- Beaker experiment: [01KXNTZ24CK775H6E6AA2PBDW4](https://beaker.org/ex/01KXNTZ24CK775H6E6AA2PBDW4)
- Scope: all 16 finished first-hybrid checkpoints plus all eight finished
  geometry Cx1/Cx2 checkpoints; one eval-only task per final checkpoint, two
  Holmes B300s per task, full validation suite, no optimizer construction or
  optimizer-state load
- Final status on 2026-07-18: all 33 registered targets are finished with 498
  exported `eval/*` metrics each—16 geometry targets and 17 first-hybrid/scale
  targets. The collector selects successful retries over older crashed
  attempts. The complete metric export and compact coverage table are
  generated by [`collect_validation_results.py`](collect_validation_results.py)
  under [`results/validation/`](results/validation/).
- A direct Beaker audit at 2026-07-18 05:57 UTC found no live validation
  worker for the ten unfinished geometry targets. Seven latest jobs exited
  143: Cx1 at `4e-4`, `8e-4`, and `3.2e-3`; Cx2 at `4e-4` and `8e-4`; and
  Cx4 at `1.6e-3` and `3.2e-3`. The Cx1 `1.6e-3` and Cx2
  `1.6e-3`/`3.2e-3` tasks finalized without starting. Several of the exit-143
  workers also stopped before W&B initialization, so the W&B-only results
  dashboard reports seven `not_started` and three `crashed`; all ten require
  a future eval-only resubmission.
- The healthy terminated logs show active evaluator progress and no model or
  Python exception; the jobs received external SIGTERM. A targeted retry
  manifest containing exactly the ten missing checkpoints is prepared at
  `launchers/validation/manifests/275m_geometry_missing_full.yaml`. It renders
  10 two-GPU tasks (20 GPUs peak). The allocated urgent retry was submitted as
  [01KXSZKECGP966SZXJ8B19G36R](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXSZKECGP966SZXJ8B19G36R)
  on 2026-07-18.
- The first-hybrid 1.2B Cx1 final validation finished separately in
  [01KXPZ4WQ72WBVPNAD0KBT2WKY](https://beaker.org/ex/01KXPZ4WQ72WBVPNAD0KBT2WKY)
  using eight B300s and EP8 `sync_1d`.
- Geometry Cx4/Cx8 validation targets are registered in
  `launchers/validation/manifests/275m_geometry_cx4_cx8_full.yaml`. All four
  Cx4 targets finished in
  [01KXPZJC376C385MZ8JAZG7AW1](https://beaker.org/ex/01KXPZJC376C385MZ8JAZG7AW1);
  all four Cx8 targets finished across
  [the 1.6e-3/3.2e-3 pair](https://beaker.org/ex/01KXQ14ACQQV1YDXMNS8Z5QD7B),
  [8e-4](https://beaker.org/ex/01KXQ1T10BEXM057MHG0QA22WW), and
  [4e-4](https://beaker.org/ex/01KXQ2D2DNHZETFXJ01GH60K79).
- Existing larger hybrid coverage: finished 480M Cx1/Cx2 and 810M Cx1/Cx2
  runs already contain all 178 validation metrics and require no backfill
- On 2026-07-20, three additional allocated, urgent batches were submitted:
  [32 two-GPU NoPE/gated 275M tasks](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXZ1PVT69VQ0GBP9WWBAKWV0),
  [24 eight-GPU scale tasks](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXZ1Q0K8T9HM18534TQ405AR),
  and [two two-GPU midtraining tasks](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXZ1Q481A8VGT0EB9BX6JCCE).
  All 32 tasks in the first batch and both midtraining tasks finished. Six
  Cx2 scale targets in the second batch failed before evaluation because rank
  batch six was not divisible by MB4; all six corrected MB3 retries finished
  in
  [01KY09D3D872K0R03NHF5MGYD4](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY09D3D872K0R03NHF5MGYD4).
  The consolidated export now contains all 91 registered targets with 498
  metrics each.
- On 2026-07-23, full-suite validation was submitted for every completed
  RoPE-gated checkpoint: the initial allocated old-worktree submissions
  [16 two-GPU 275M tasks](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY85JPM1SE0XYX1RMFM5HXX7)
  and [10 eight-GPU larger-model tasks](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY85JY73XPE4223SKFNDDA7Y)
  were canceled before scheduling on 2026-07-23. They are superseded by
  high-priority unallocated backfills on `jacobm/moe-v2-core`:
  [16 two-GPU 275M tasks](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8CJNZW98DWVW4282SEHKWT)
  and [10 eight-GPU larger-model tasks](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8CK6Q5BFTEX09ZKRHWCCC5).
  The 275M batch finished 16/16. In the larger batch, 1.2B Cx1/Cx4/Cx8 are
  complete. The separately registered first-hybrid 810M Cx8 evaluation also
  finished, so the consolidated collector exports 117/117 finished targets
  with 498 metrics each and no live validation target. Failed/partial 1.2B
  Cx2 training is not evaluated.

## 275M aggressive MXFP8 KDA LR sweep

- Model variant: `geometry_275m_kda_ev2_neg_nope_gated_mxfp8_672`
- Manifest:
  [`launchers/pretraining/manifests/275m_kda_aggressive_mxfp8_lr_sweep.yaml`](launchers/pretraining/manifests/275m_kda_aggressive_mxfp8_lr_sweep.yaml)
- Rendered spec:
  [`launchers/pretraining/generated/275m_kda_aggressive_mxfp8_lr_sweep.yaml`](launchers/pretraining/generated/275m_kda_aggressive_mxfp8_lr_sweep.yaml)
- Beaker experiment:
  [01KYJPTZ3J4VHGBH0FSVAQRDGC](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYJPTZ3J4VHGBH0FSVAQRDGC)
- Scheduling: urgent unallocated Holmes B300s, `minRuntime: 0m`, auto-resuming
- Resources: 16 tasks, two B300s per task, 32 GPUs at full concurrency, EP1
- Precision: rowwise MXFP8 on all shared/routed FFNs; MXFP8 QKV/output
  projections; `moe_fused_v2`; fused attention with FlashAttention-4;
  `OLMO_MXFP8_SCALE_MODE=rceil`; KDA and other components retain their current
  higher-precision paths
- Training: Cx1/2/4/8 at `4e-4`, `8e-4`, `1.6e-3`, and `3.2e-3`; canonical
  global batches; MB16/12/16/16 with accumulation 1/2/2/3; rolling ephemeral
  checkpoints every 500 steps; no in-loop or on-finish evaluation
- Status at 2026-07-28 16:36 UTC: Cx1/Cx2/Cx4 are complete and bracketed. Their
  observed bests are `2.685399`, `2.566948`, and `2.463998`, all at `1.6e-3`.
  Relative to matching BF16 KDA, those points are `-0.007296`, `+0.004429`,
  and `-0.000250` CE. The Cx1 `8e-4` local W&B recovery remains hash-verified.
  All four Cx8 cells are running around 76% complete; the `4e-4`, `8e-4`, and
  `1.6e-3` restarts are registered as explicit predecessor/current W&B chains.
  The three predecessors were simultaneously canceled with exit 143 when
  Holmes cordoned node `01KV3W9DWEVF1JE98MQJXZ7Y05` after a health-check
  failure. Their final logs were finite and progressing normally, so this is
  classified as one infrastructure interruption rather than three KDA
  training crashes.

The exact architecture delta, parameter counts, token budgets, promotion
gates, and audited larger-size configurations are recorded in
[`MXFP8_LADDER.md`](MXFP8_LADDER.md). Add exact W&B IDs here after task
initialization and use the dedicated MXFP8 plotting family; do not mix these
runs into the existing BF16 KDA registry by broad display-name matching.

## 480M aggressive MXFP8 KDA transferred-LR continuation

- Model variant: `geometry_matched_kda_ev2_neg_nope_gated_mxfp8_aligned`
- Manifest:
  [`launchers/pretraining/manifests/480m_kda_aggressive_mxfp8_full.yaml`](launchers/pretraining/manifests/480m_kda_aggressive_mxfp8_full.yaml)
- Submission ledger:
  [`launchers/pretraining/generated/480m_kda_aggressive_mxfp8_full_submissions.json`](launchers/pretraining/generated/480m_kda_aggressive_mxfp8_full_submissions.json)
- Source commit: `785a87cf0aba6f12d81bf1d7b37b11cb49e6f9ab`
- Scheduling: urgent unallocated Holmes B300s, `minRuntime: 0m`,
  nonpreemptible, auto-resuming
- Precision and architecture: KDA `expand_v=2`, negative eigenvalues, NoPE,
  gated full attention, EP1, 832-wide aligned experts, rowwise MXFP8 on every
  shared/routed FFN, MXFP8 QKV/output projections, `moe_fused_v2`,
  FlashAttention-4, and `OLMO_MXFP8_SCALE_MODE=rceil`
- Parameters: 496,253,280 active; 419,182,944 active non-embedding;
  7,151,827,296 stored
- Checkpointing: rolling ephemeral saves every 500 steps, permanent final
  checkpoint, no in-loop or on-finish evaluation

| Cx | LR | Global batch | GPUs | Rank MB | Accumulation | Job | Launch state |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `1.2e-3` | 262,144 | 8 | 4 | 1 | [01KYKMN56Y5PWCE2173E50VV2J](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYKMN56Y5PWCE2173E50VV2J) | finished; CE `2.497288` |
| 2 | `9e-4` | 393,216 | 8 | 6 | 1 | [01KYKMN7YPTVTE5WB8T7S8734A](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYKMN7YPTVTE5WB8T7S8734A) | intentionally canceled at ~88%; durable `step38000` |
| 4 | `8e-4` | 524,288 | 8 | 8 | 1 | [01KYKMNAS6WKSTKE3S7QZBTXY4](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYKMNAS6WKSTKE3S7QZBTXY4) | intentionally canceled at ~60%; durable `step38500` (`step39000-tmp` ignored) |
| 8 | `8e-4` | 786,432 | 16 | 6 | 1 | [01KYKMNDDJS7D7D9DGC7Y32A43](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYKMNDDJS7D7D9DGC7Y32A43) | intentionally canceled at ~6%; durable `step5000` (`step5500-tmp` ignored) |

The four exact W&B display names are registered in `plot_kda_mxfp8.py`.
Consequently the next collection pass will add each finished 480M point to
the four-size best-of comparison without a plotting-code edit. These are
single transferred-LR cells, not four-point 480M sweeps, so they belong in the
best-of plot rather than a 480M U-plot.

Cx2/Cx4/Cx8 were manually stopped on 2026-07-28 to release 32 B300s while the
rowwise expert-MXFP8 path is optimized. This was a systems decision, not a
numerical failure. Resume only from the durable checkpoints recorded above;
the two `-tmp` directories were interrupted mid-save and are not checkpoints.

## New wave template

| Intervention | Manifest | Beaker experiment | W&B group | Status | Decision |
|---|---|---|---|---|---|
| `<name>` | `launchers/pretraining/manifests/<name>.yaml` | `<link>` | `<group>` | planned/running/complete | pending/reject/retain |

Append exact per-run rows below the wave summary:

| Cx | LR | Global batch | Rank microbatch | Accumulation | Job | W&B | Status |
|---:|---:|---:|---:|---:|---|---|---|
| `<Cx>` | `<LR>` | `<tokens>` | `<sequences>` | `<steps>` | `<link>` | `<link>` | `<state>` |
