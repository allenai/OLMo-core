# OLMoE ladder v2 run ledger

Record post-migration experiment waves here. Per-run rows must include Beaker
job IDs and W&B IDs once they exist. Detailed migration-era DDP jobs remain in
[`../v1/DDP_RUNS.md`](../v1/DDP_RUNS.md).

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

Status on 2026-07-16 05:35 UTC: both 480M runs and both 810M Cx1/Cx2 runs are
finished. The eval-enabled 1.2B Cx1 attempt was stopped at durable `step26000`.
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
| 1.2B | 1 | `4e-4` | 262,144 | 8 | EP8 / `sync_1d` | 8 / 4 | 1 | [eval-enabled](https://beaker.org/ex/01KXJAPHHB8MBPD1B3E92QH89Y) / [no-eval resume](https://beaker.org/ex/01KXMPNX36KW85J97N8MAKBQEA) | [eval-enabled](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xapobmqb) / [no-eval resume](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1d24xfx5) | no-eval job running; explicitly loaded `step26000` |
| 1.2B | 2 | `6e-4` | 393,216 | 8 | EP8 / `sync_1d` | 12 / 6 | 1 | [initial](https://beaker.org/ex/01KXJAPHN6SNXMG7X49M7HH17G) / [resume 1](https://beaker.org/ex/01KXK9R5PMR9GXBC0RMSDY2V13) / [resume 2](https://beaker.org/ex/01KXKEEFKYPGQAVWS024PTGCPK) / [resume 3](https://beaker.org/ex/01KXKSKC7K2KNQ9EN99EE5N11F) / [resume 4](https://beaker.org/ex/01KXMHBTR1D29J3SND9FZ13B8Z) / [no-eval resume](https://beaker.org/ex/01KXMPNX6QXBFKH2DRRYMPXDV6) | [initial](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/l4r1crzm) / [resume 1](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/bwvkwb9s) / [resume 2](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jsb3obpq) / [resume 3](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ezechghu) / [resume 4](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jybetzoc) / [no-eval resume](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vr2jfn4c) | no-eval job passed `step10001` and is running |

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
| 810M | 4 | `4e-4` | 524,288 | 8 | EP1 | 4 | 2 | [eval-enabled](https://beaker.org/ex/01KXKTT4G4QT56NGHKWSVXWEX6) / [no-eval resume](https://beaker.org/ex/01KXMPNYN9J377PP8F8YZ763NJ) | [eval-enabled](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/adi3mjy7) / [no-eval resume](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/bvlzu2c9) | no-eval job running from `step24500` |
| 810M | 8 | `4e-4` | 786,432 | 8 | EP1 | 6 | 2 | [eval-enabled](https://beaker.org/ex/01KXKTT4KPQRMJ0E1DF5GPS26A) / [no-eval resume](https://beaker.org/ex/01KXMPNYRK51C80XFNZNAP337M) | [eval-enabled](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sucwb1sc) / [no-eval resume](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/k1d1td9b) | no-eval job running from completed `step20000` |
| 1.2B | 4 | `3e-4` | 524,288 | 8 | EP8 / `sync_1d` | 4 / 4 | 2 | [failed](https://beaker.org/ex/01KXKTT4R8MANSKM43DEJB02GC) / [eval-enabled retry](https://beaker.org/ex/01KXKY6SXD7DFDDJDGK83NGQDF) / [no-eval resume](https://beaker.org/ex/01KXMPNYVY6A2RMTNW2E27H9GR) | [failed](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/f9wybz72) / [eval-enabled retry](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9c1fcuto) / [no-eval resume](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h5ft97x1) | no-eval job running; explicitly loaded `step4000` |
| 1.2B | 8 | `4e-4` | 786,432 | 8 | EP8 / `sync_1d` | 6 / 6 | 2 | [failed](https://beaker.org/ex/01KXKTT4W01BXNZ5WRD40BT2QG) / [eval-enabled retry](https://beaker.org/ex/01KXKY6T9QR8PQDWRAN4Y76CVQ) / [no-eval resume](https://beaker.org/ex/01KXMPNYZ7Q1J3BXXD47KET92X) | [failed](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/n5v3vewn) / [eval-enabled retry](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/48b58zfx) / [no-eval resume](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/zyeib8rb) | no-eval job running from `step4500` |

### 480M Cx4/Cx8 completion runs

- Manifest: [`launchers/pretraining/manifests/hybrid_scale_480m_cx4_cx8.yaml`](launchers/pretraining/manifests/hybrid_scale_480m_cx4_cx8.yaml)
- Launcher: [`launchers/pretraining/launch_hybrid_scale_480m_cx4_cx8.sh`](launchers/pretraining/launch_hybrid_scale_480m_cx4_cx8.sh)
- Beaker experiment: [01KXMTAQPTG52EPEXMQN0Q1YJ7](https://beaker.org/ex/01KXMTAQPTG52EPEXMQN0Q1YJ7)

Submitted 2026-07-16 at urgent priority on `ai2/holmes`. Both cells use the
observed-best wide-integration LR `8e-4`, EP1, the canonical global batch, and
no in-loop or on-finish evaluation. Cx4 uses the largest proven-safe legal
microbatch below the projected-over-capacity MB16 shape. Cx8 uses the already
validated MB12 shape.

| Size | Cx | LR | Global batch | GPUs | EP | MB | Accum | Job | W&B | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 480M | 4 | `8e-4` | 524,288 | 4 | 1 | 8 | 2 | [01KXMTAR1ZB3ERY8JQ0MH4681B](https://beaker.org/ex/01KXMTAR1ZB3ERY8JQ0MH4681B) | pending initialization | queued |
| 480M | 8 | `8e-4` | 786,432 | 8 | 1 | 12 | 1 | [01KXMTAR5C5JX0ATP038ECKNWS](https://beaker.org/ex/01KXMTAR5C5JX0ATP038ECKNWS) | pending initialization | queued |

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

## New wave template

| Intervention | Manifest | Beaker experiment | W&B group | Status | Decision |
|---|---|---|---|---|---|
| `<name>` | `launchers/pretraining/manifests/<name>.yaml` | `<link>` | `<group>` | planned/running/complete | pending/reject/retain |

Append exact per-run rows below the wave summary:

| Cx | LR | Global batch | Rank microbatch | Accumulation | Job | W&B | Status |
|---:|---:|---:|---:|---:|---|---|---|
| `<Cx>` | `<LR>` | `<tokens>` | `<sequences>` | `<steps>` | `<link>` | `<link>` | `<state>` |
