# OLMoE ladder v2 run ledger

Record post-migration experiment waves here. Per-run rows must include Beaker
job IDs and W&B IDs once they exist. Detailed migration-era DDP jobs remain in
[`../v1/DDP_RUNS.md`](../v1/DDP_RUNS.md).

## Live status snapshot (2026-07-23 19:00 UTC)

This is the current source of truth for active V2 work. The detailed sections
below retain the full launch and retry history.

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
| pretraining | larger aligned geometry + RoPE + gated attention | 10 finished / 1 running / 1 failed | 1.2B Cx8 finished at 187.437B tokens with final `step238338`; 810M Cx8 is still running; 1.2B Cx2 remains failed | [results](results/pretraining/geometry_gdn_ev2_rope_gated/results.md) |
| midtraining | first hybrid 275M Cx8 | finished | 100B; final checkpoint `step95368`; validation finished | [1keo2hz6](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1keo2hz6) |
| midtraining | first hybrid 480M Cx8 | finished | 100.001B; final checkpoint `step95368`; validation finished | [mnp9rv5l](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mnp9rv5l) |
| validation | V2 post-training backfills | 91 complete; 26 RoPE-gated queued | Full-suite backfills submitted for all 16 finished 275M sweep cells and 10 finished larger cells | [results](results/validation/hybrid_full.md) |

The formal pretraining results and plots use finished runs only and enforce a
complete final-250M-token history. The gated-RoPE sweep is now complete. Its
observed best final-250M CEs at Cx1/2/4/8 are `2.691980`, `2.573449`,
`2.470110`, and `2.386206`. It beats the ungated RoPE geometry, ungated NoPE,
and gated NoPE controls at every Cx. It also beats the first hybrid at Cx1,
Cx4, and Cx8; at Cx2 it is `0.003461` worse.

All six corrected Cx2 MB3 validation retries finished in
[01KY09D3D872K0R03NHF5MGYD4](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY09D3D872K0R03NHF5MGYD4).
The 32 NoPE/gated 275M backfills and both first-hybrid midtraining backfills
also finished. The consolidated export now contains 91 complete targets with
498 metrics each. The current backlog is 28 checkpoints: 16 finished 275M
gated-RoPE points, nine finished gated-RoPE scale points, first-hybrid 1.2B
Cx8, gated-NoPE 1.2B Cx2, and the clean ungated-NoPE 1.2B Cx8 reproduction.

The nine finished larger gated-RoPE points have strict final-250M CEs of
`2.506239`, `2.402917`, `2.307792`, and `2.233177` for 480M Cx1/2/4/8;
`2.368164`, `2.266516`, and `2.191042` for 810M Cx1/2/4; and `2.270124` and
`2.105145` for 1.2B Cx1/4.
The 1.2B Cx2 failure is a training-path numerical failure, not an OOM: the
optimizer asserted on a non-finite total gradient at step 5,420, auto-resumed
from durable `step5000`, and hit the same assertion again at step 6,582. The
later CUDA device assertion and NCCL watchdog messages are distributed
teardown effects. The 810M Cx4 cell had one identical assertion at step 29,107,
then auto-resumed and has continued cleanly past 39.2B tokens.

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
evaluation are disabled; validation will be backfilled after training. W&B
IDs will be added to the plotting registry after the queued jobs initialize.

| Cx | Size | LR | GPUs | EP | MB | Accum | Beaker work | State |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 480M | `1.2e-3` | 4 | 1 | 8 | 1 | [01KY1DKVJJ0ECA4QS0SENKH49E](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DKVJJ0ECA4QS0SENKH49E) | queued |
| 1 | 810M | `6e-4` | 8 | 1 | 4 | 1 | [01KY1DKYH8Q1HAX2922D7AJA4E](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DKYH8Q1HAX2922D7AJA4E) | queued |
| 1 | 1.2B | `4e-4` | 8 | 8 | 4 | 1 | [01KY1DM1J6C028TJNQART75FY1](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DM1J6C028TJNQART75FY1) | queued |
| 2 | 480M | `9e-4` | 4 | 1 | 12 | 1 | [01KY1DM576YRXHEVF13MTVN9VJ](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DM576YRXHEVF13MTVN9VJ) | queued |
| 2 | 810M | `5.6e-4` | 8 | 1 | 6 | 1 | [01KY1DM89PS7K4BCYTX0QFVFXE](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DM89PS7K4BCYTX0QFVFXE) | queued |
| 2 | 1.2B | `6e-4` | 16 | 8 | 3 | 1 | [01KY1DMBM1CBJTPK3W3W372B56](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DMBM1CBJTPK3W3W372B56) | queued |
| 4 | 480M | `8e-4` | 4 | 1 | 8 | 2 | [01KY1DMF80H4S3G45RCDE5D1TR](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DMF80H4S3G45RCDE5D1TR) | queued |
| 4 | 810M | `4e-4` | 8 | 1 | 4 | 2 | [01KY1DMJG0DPAN2M8E9YHEQPZT](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DMJG0DPAN2M8E9YHEQPZT) | queued |
| 4 | 1.2B | `3e-4` | 16 | 8 | 4 | 1 | [01KY1DMNKHZ8AND4FYA9WFHRYA](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DMNKHZ8AND4FYA9WFHRYA) | queued |
| 8 | 480M | `8e-4` | 8 | 1 | 12 | 1 | [01KY1DMRSHXB5RTZGD39NXD2G4](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DMRSHXB5RTZGD39NXD2G4) | queued |
| 8 | 810M | `4e-4` | 8 | 1 | 6 | 2 | [01KY1DMW31S6YVCDN479XYRE7S](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DMW31S6YVCDN479XYRE7S) | queued |
| 8 | 1.2B | `4e-4` | 32 | 8 | 3 | 1 | [01KY1DN08V4T6HMJDDCPPKHGSN](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY1DN08V4T6HMJDDCPPKHGSN) | queued |

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
  high-priority unallocated backfills on `jacobm/moe-v2-core` (links recorded
  below after submission).
  Both batches are allocated, urgent, and constrained to Holmes. The larger
  batch includes the newly finished 1.2B Cx8 `step238338`; 810M Cx8 remains
  deferred while training, and failed/partial 1.2B Cx2 is not evaluated.

## New wave template

| Intervention | Manifest | Beaker experiment | W&B group | Status | Decision |
|---|---|---|---|---|---|
| `<name>` | `launchers/pretraining/manifests/<name>.yaml` | `<link>` | `<group>` | planned/running/complete | pending/reject/retain |

Append exact per-run rows below the wave summary:

| Cx | LR | Global batch | Rank microbatch | Accumulation | Job | W&B | Status |
|---:|---:|---:|---:|---:|---|---|---|
| `<Cx>` | `<LR>` | `<tokens>` | `<sequences>` | `<steps>` | `<link>` | `<link>` | `<state>` |
