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

Status on 2026-07-15 17:11 UTC: the original grid and the added Cx1/Cx2 `4e-4`
and Cx4 `3.2e-3` points are complete. The Cx8 `3.2e-3` run was interrupted by
a Holmes node Xid 31 failure at step 29,498/42,954 and has now resumed from its
durable `step29000` checkpoint. Cx8 remains withheld from the optimal-LR summary
until that hot-side point finishes.

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
| 8 | `3.2e-3` | 786,432 | 16 | 3 | [initial](https://beaker.org/ex/01KXHZNJTHJH9K7X88TJA5Q537) / [resume](https://beaker.org/ex/01KXKBZK6FKCM081WJH3YP82TX) | [initial](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/f7lbyrfl) / [resume](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ntoo8vlo) | resumed from `step29000`; running |

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
- Evaluation policy: v1 `fast` task group and LM validation every 2,000 steps,
  plus both evaluations on finish.

Submitted 2026-07-15 at urgent priority on `ai2/holmes` using 40 requested B300
GPUs. LRs are the observed optimal wide-intervention LRs for each matching size
and data multiple.

Status on 2026-07-15 17:11 UTC: both 480M runs finished with final checkpoints
and final evals. The 810M Cx1/Cx2 and 1.2B Cx1 runs remain healthy. The 1.2B
Cx2 job was terminated when its Holmes node was cordoned for an unrecoverable
Xid 31; its replacement resumed from `step2500`, has passed step 3,500, and is
running normally.

| Size | Cx | LR | Global batch | GPUs | EP/path | Rank MB | Accum | Job | W&B | Status |
|---|---:|---:|---:|---:|---|---:|---:|---|---|---|
| 480M | 1 | `1.2e-3` | 262,144 | 4 | EP1 | 8 | 1 | [01KXJAPH2DP3XSCHX1A637SN7K](https://beaker.org/ex/01KXJAPH2DP3XSCHX1A637SN7K) | [wl8ebsd8](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/wl8ebsd8) | finished |
| 480M | 2 | `9e-4` | 393,216 | 4 | EP1 | 12 | 1 | [01KXJAPH62PWXK2B53PKT14M08](https://beaker.org/ex/01KXJAPH62PWXK2B53PKT14M08) | [4vzmrld1](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4vzmrld1) | finished |
| 810M | 1 | `6e-4` | 262,144 | 8 | EP1 | 4 | 1 | [01KXJAPH9V450M32SJC5G4KN93](https://beaker.org/ex/01KXJAPH9V450M32SJC5G4KN93) | [h1rmcm2p](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h1rmcm2p) | running, step 43,259/55,768 (77.6%) |
| 810M | 2 | `5.6e-4` | 393,216 | 8 | EP1 | 6 | 1 | [01KXJAPHDN7KN3TY7NXRVRZGNM](https://beaker.org/ex/01KXJAPHDN7KN3TY7NXRVRZGNM) | [1d5gxgjv](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1d5gxgjv) | running, step 35,999/74,357 (48.4%) |
| 1.2B | 1 | `4e-4` | 262,144 | 8 | EP8 / `sync_1d` | 8 | 4 | [01KXJAPHHB8MBPD1B3E92QH89Y](https://beaker.org/ex/01KXJAPHHB8MBPD1B3E92QH89Y) | [xapobmqb](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xapobmqb) | running, step 11,999/86,558 (13.9%) |
| 1.2B | 2 | `6e-4` | 393,216 | 8 | EP8 / `sync_1d` | 12 | 4 | [initial](https://beaker.org/ex/01KXJAPHN6SNXMG7X49M7HH17G) / [resume](https://beaker.org/ex/01KXK9R5PMR9GXBC0RMSDY2V13) | [initial](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/l4r1crzm) / [resume](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/bwvkwb9s) | resumed from `step2500`; running past step 3,500 |

## New wave template

| Intervention | Manifest | Beaker experiment | W&B group | Status | Decision |
|---|---|---|---|---|---|
| `<name>` | `launchers/pretraining/manifests/<name>.yaml` | `<link>` | `<group>` | planned/running/complete | pending/reject/retain |

Append exact per-run rows below the wave summary:

| Cx | LR | Global batch | Rank microbatch | Accumulation | Job | W&B | Status |
|---:|---:|---:|---:|---:|---|---|---|
| `<Cx>` | `<LR>` | `<tokens>` | `<sequences>` | `<steps>` | `<link>` | `<link>` | `<state>` |
