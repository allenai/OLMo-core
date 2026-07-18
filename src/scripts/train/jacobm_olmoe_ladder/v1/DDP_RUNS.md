# OLMoDDP run tracker

## Long-context Cx8 continuations

All runs use 64k sequences, a 100B-token budget, 8 Holmes B300s, DDP, no expert
parallelism, no context parallelism, block recomputation, and a fresh optimizer.
Learning rate is one half of the source midtraining LR.

The remaining V1 long-context runs deliberately retain the established V1
schedule: 2,000 fixed linear warmup steps followed by a constant LR. This keeps
them comparable to the completed 275M/480M continuations. The intended switch
to the pretraining-style 10%-of-training warmup plus cosine decay is recorded as
a controlled V2 experiment in `../v2/NEXT_EXPERIMENTS.md`; it must not be mixed
into the remaining V1 runs.

Evaluator callbacks are disabled in all resumed training jobs as of 2026-07-16.
Validation and RULER run afterward from final checkpoints in separate jobs.

| Size | Family | LR | Global batch | Rank microbatch | Beaker experiment | Job | W&B | Current state (2026-07-17 03:39 UTC) |
|---|---|---:|---:|---:|---|---|---|---|
| 275M | baseline | `1e-4` | 2 Mi tokens | 4 seq | [`01KXEW4KTBWXMY9XPYANZ6T7YD`](https://beaker.org/ex/01KXEW4KTBWXMY9XPYANZ6T7YD) | `01KXEW4M665ZHPG6SCV25N533Q` | [`e4hvrd33`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/e4hvrd33) | finished, step 47,684 / 100.001B tokens |
| 275M | integration deep | `8e-5` | 2 Mi tokens | 4 seq | [`01KXFP8ZKCB5BBMX5WY9MV5WWW`](https://beaker.org/ex/01KXFP8ZKCB5BBMX5WY9MV5WWW) | `01KXFP8ZYY404ABS636ANN09HX` | [`hq0yjd50`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/hq0yjd50) | finished, step 47,684 / 100.001B tokens |
| 275M | integration wide | `8e-5` | 2 Mi tokens | 4 seq | [`01KXFPTD55Y7JDRM27995GFX61`](https://beaker.org/ex/01KXFPTD55Y7JDRM27995GFX61) | `01KXFPTDKAZ85835R6QV1ABKC8` | [`zm8iut38`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/zm8iut38) | finished, step 47,684 / 100.001B tokens |
| 480M | baseline | `4e-5` | 3 Mi tokens | 6 seq | [`01KXFP8ZKCB5BBMX5WY9MV5WWW`](https://beaker.org/ex/01KXFP8ZKCB5BBMX5WY9MV5WWW) | `01KXFP902AXYVT672A1JMC5Z5E` | [`9lgu2exp`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9lgu2exp) | finished, step 31,790 / 100.003B tokens |
| 480M | integration deep | `4e-5` | 3 Mi tokens | 6 seq | [`01KXFP8ZKCB5BBMX5WY9MV5WWW`](https://beaker.org/ex/01KXFP8ZKCB5BBMX5WY9MV5WWW) | `01KXFP905NYT6E9AAYP288Q6YD` | [`32yjntyu`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/32yjntyu) | finished, step 31,790 / 100.003B tokens |
| 480M | integration wide | `4e-5` | 3 Mi tokens | 6 seq | [`01KXFQJ6M4HDHJ50WVS4ECB8KK`](https://beaker.org/ex/01KXFQJ6M4HDHJ50WVS4ECB8KK) | `01KXFQJ70Y3ZAV6KEPQJE59RSK` | [`3357zlh6`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3357zlh6) | finished, step 31,790 / 100.003B tokens |
| 810M | integration wide | `2e-5` | 4 Mi tokens | 4 seq x 2 accum | [current work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXMPP1AX400D5QTC07R8V5VB) | [current job](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXMPP1AX400D5QTC07R8V5VB?taskId=01KXMPP1K1MBVKANGMQMMJZ93W&jobId=01KXQ8KJTMHR3ERAVFFVFVB9D1) | [hxdiomdx](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/hxdiomdx) | running, step 14,769 / 23,842, 61.95B / 100B tokens (61.9%) |
| 1.2B | integration wide | `2e-5` | 4 Mi tokens | 2 seq x 4 accum | [current work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXMPP1AX400D5QTC07R8V5VB) | [current job](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXMPP1AX400D5QTC07R8V5VB?taskId=01KXMPP1PE5T3M7YFTXEYVM03E&jobId=01KXRZRA7PZ5FE1ZSY5NXW2MC3) | [vvofgf68](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vvofgf68) | running, step 9,799 / 23,842, 41.10B / 100B tokens (41.1%) |
| 810M | baseline | `2e-5` | 4 Mi tokens | 4 seq x 2 accum | [current work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXPYTPVH09F88PVR64G22HG3) | [current job](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXPYTPVH09F88PVR64G22HG3?taskId=01KXPYTPVPY296SME2TRMV2VPH&jobId=01KXQNNVFBS4MVCR7GSKWKANQP) | [vgqd5hij](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vgqd5hij) | running, step 6,029 / 23,842, 25.29B / 100B tokens (25.3%) |
| 1.2B | baseline | `2e-5` | 4 Mi tokens | 2 seq x 4 accum | [current work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXPYTPVH09F88PVR64G22HG3) | [current job](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXPYTPVH09F88PVR64G22HG3?taskId=01KXPYTPZ22EW6J07XQ7ZMAWQ&jobId=01KXRZRA3M46ZQEHA9NWV1H332) | [8yqbbo8n](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8yqbbo8n) | running from step 0; the preempted 759-step attempt had not reached the first ephemeral save at step 1,000, so no LC checkpoint existed to resume |

## 275M integration-wide hybrid control

`hybrid_wide_275m_model.py` derives the model from the canonical converted
integration-wide config and replaces only SWA layers 2, 4, 6, 8, and 10 with
GDN. Full-attention layers remain 0, 1, 3, 5, 7, 9, and 11.

The selected control uses 8 key/value heads, head dimension 128, and
`expand_v=1`: 288,194,512 active parameters versus 280,207,872 in the source
(+2.85%). This matches the existing MoE GDN implementation. The dense-ladder
`expand_v=2` reference would have 300,012,112 active parameters (+7.07%), so it
is retained as an explicit alternative rather than used silently in the isolated
control.

The two-B300 smoke completed ten optimizer steps without skips, saved its final
checkpoint, and exited 0: [`01KXFQA51P00W5GJHMNP318PVW`](https://beaker.org/ex/01KXFQA51P00W5GJHMNP318PVW),
job `01KXFQA5DT6BXVP2TH13TSPQ0P`, W&B run
[`k1tn9qyh`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/k1tn9qyh).
Peak active memory was 184.1 GiB per B300 at a rank microbatch of 16 sequences.

### Cx1 LR sweep

Each run trains from scratch for 4,222,483,520 target tokens (16,108 steps at a
262,144-token global batch) on two Holmes B300s, with no expert parallelism.
All three belong to Beaker experiment
[`01KXFR1KT408AWVN41NPKXS4F5`](https://beaker.org/ex/01KXFR1KT408AWVN41NPKXS4F5).

| LR | Run name | Job | W&B | Current state |
|---:|---|---|---|---|
| `8e-4` | `pt-275m-intwide-hybrid-gdn-ev1-cx1-lr8e-4-r1` | `01KXFR1M6K4M6SV4P2EBYWJYK3` | [`yo22u93q`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/yo22u93q) | finished |
| `1.6e-3` | `pt-275m-intwide-hybrid-gdn-ev1-cx1-lr1p6e-3-r1` | `01KXFR1M9Y9966MHENKVDDS9TZ` | [`moknw6oc`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/moknw6oc) | finished |
| `3.2e-3` | `pt-275m-intwide-hybrid-gdn-ev1-cx1-lr3p2e-3-r1` | `01KXFR1MD63EEYTQRX1DW9611K` | [`mettf0d3`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mettf0d3) | finished |

### Cx2/Cx4/Cx8 LR sweeps

All nine runs belong to Beaker experiment
[`01KXFSA0GP221T0X7V675XTSG2`](https://beaker.org/ex/01KXFSA0GP221T0X7V675XTSG2).
Each run uses two Holmes B300s, DDP, and no expert parallelism.

| Cx | LR | Global batch | Rank microbatch | Accumulation | Target tokens | Steps | Job | W&B | Current state |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 2 | `8e-4` | 393,216 | 8 seq | 3 | 8,444,967,040 | 21,477 | `01KXFSA0WJCSDGKKB91EB5ZYK4` | [`07qo96gy`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/07qo96gy) | finished |
| 2 | `1.6e-3` | 393,216 | 8 seq | 3 | 8,444,967,040 | 21,477 | `01KXFSA0ZY3W8MDXCVW1DAGKJA` | [`j12fk559`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/j12fk559) | finished |
| 2 | `3.2e-3` | 393,216 | 8 seq | 3 | 8,444,967,040 | 21,477 | `01KXFSA13248BP335W5EK8EX2G` | [`mem73c7g`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mem73c7g) | finished |
| 4 | `4e-4` | 524,288 | 16 seq | 2 | 16,889,934,080 | 32,215 | `01KXFSA16XJC9XBNA74B2QTK7M` | [`socvue3a`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/socvue3a) | finished |
| 4 | `8e-4` | 524,288 | 16 seq | 2 | 16,889,934,080 | 32,215 | `01KXFSA1AA0H5WY9Y074TS604S` | [`xvk92054`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xvk92054) | finished |
| 4 | `1.6e-3` | 524,288 | 16 seq | 2 | 16,889,934,080 | 32,215 | `01KXFSA1DMP6S8G7GAE0JW3DR4` | [`uhw9wfed`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/uhw9wfed) | finished |
| 8 | `4e-4` | 786,432 | 16 seq | 3 | 33,779,868,160 | 42,954 | `01KXFSA1GT8JJWTVMSNXF428XW` | [`b0z3qfmi`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/b0z3qfmi) | finished |
| 8 | `8e-4` | 786,432 | 16 seq | 3 | 33,779,868,160 | 42,954 | `01KXFSA1M6ZR3JCNXB2VQ9K04J` | [`rkxojd03`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rkxojd03) | finished |
| 8 | `1.6e-3` | 786,432 | 16 seq | 3 | 33,779,868,160 | 42,954 | `01KXFSA1QH0T6YHPV54FMR88AA` | [`66aja50m`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/66aja50m) | finished |

### Additional bracketing LRs

These four runs extend the first hybrid sweep so every reported optimum can be
an observed interior point. They belong to Beaker experiment
[`01KXHZNJ5FD5RA0BRC4ZF3DRKC`](https://beaker.org/ex/01KXHZNJ5FD5RA0BRC4ZF3DRKC),
use urgent priority on Holmes, and retain the original per-Cx batch settings.

| Cx | LR | Global batch | Rank microbatch | Accumulation | Target tokens | Steps | Job | W&B | Current state |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | `4e-4` | 262,144 | 16 seq | 1 | 4,222,483,520 | 16,108 | [`01KXHZNJGGRPHHN4PYYG90T9AP`](https://beaker.org/ex/01KXHZNJGGRPHHN4PYYG90T9AP) | [`fkm77yos`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/fkm77yos) | finished |
| 2 | `4e-4` | 393,216 | 8 seq | 3 | 8,444,967,040 | 21,477 | [`01KXHZNJKTW5ZFH2F7BT01GQPB`](https://beaker.org/ex/01KXHZNJKTW5ZFH2F7BT01GQPB) | [`s5qmhyb2`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/s5qmhyb2) | finished |
| 4 | `3.2e-3` | 524,288 | 16 seq | 2 | 16,889,934,080 | 32,215 | [`01KXHZNJQ84EGNZYZ8N7KBAR43`](https://beaker.org/ex/01KXHZNJQ84EGNZYZ8N7KBAR43) | [`sr1jgmao`](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sr1jgmao) | finished |
| 8 | `3.2e-3` | 786,432 | 16 seq | 3 | 33,779,868,160 | 42,954 | [initial](https://beaker.org/ex/01KXHZNJTHJH9K7X88TJA5Q537) / [resume](https://beaker.org/ex/01KXKBZK6FKCM081WJH3YP82TX) | [initial](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/f7lbyrfl) / [resume](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ntoo8vlo) | running, step 38,758/42,954 (90.2%) |
