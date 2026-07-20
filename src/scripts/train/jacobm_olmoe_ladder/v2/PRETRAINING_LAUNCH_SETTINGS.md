# Hybrid pretraining launch settings

This note records the promoted `expand_v=1` hybrid runs at 480M, 810M, and
1.2B. Holmes B300 smoke measurements established the production microbatch and
expert-parallel settings used below.

## Fixed scientific settings

- Train from initialization at sequence length 8,192 on `OLMo_mix_0925` with
  the Dolma 2 tokenizer.
- Use the integration-wide architecture at each size with only its SWA layers
  replaced by GDN, `head_dim=128`, `n_v_heads=n_heads`, and `expand_v=1`.
- Retain RoPE, initialization standard deviation `0.01`, the wide global-layer
  placement, MoE geometry, dense-first-FFN block, norms, and bias settings.
- Preserve the canonical optimizer batches: Cx1 is 32 sequences / 262,144
  tokens; Cx2 is 48 sequences / 393,216 tokens; Cx4 is 64 sequences / 524,288
  tokens; Cx8 is 96 sequences / 786,432 tokens.
- Use the existing optimizer and token-based schedule: 10% linear warmup,
  cosine decay to 10% of peak LR, weight decay `0.1`, betas `(0.9, 0.95)`, and
  the established no-decay/expert optimizer groups.
- Use the best observed per-width LR transferred by the wide-integration
  promotion. These are observed optima from the preceding width ladder, not
  newly measured hybrid optima at the larger sizes.
- The trainer must satisfy the v2 callback contract before launch: speed/MFU,
  Beaker progress, W&B, config saving, and checkpointing. Evaluators remain
  disabled; validation runs afterward in separate eval-only jobs.

## Promoted points and hybrid token budgets

The Chinchilla duration uses active non-embedding parameters from the hybrid
config. Steps are the first complete optimizer step at or beyond the target.

| Size | Cx | Transferred LR | Active non-embedding | Target tokens | Global batch | Steps | Tokens after final step |
|---|---:|---:|---:|---:|---:|---:|---:|
| 480M | 1 | `1.2e-3` | 398,468,336 | 7,969,366,720 | 262,144 | 30,401 | 7,969,439,744 |
| 480M | 2 | `9e-4` | 398,468,336 | 15,938,733,440 | 393,216 | 40,535 | 15,939,010,560 |
| 480M | 4 | `8e-4` | 398,468,336 | 31,877,466,880 | 524,288 | 60,802 | 31,877,758,976 |
| 480M | 8 | `8e-4` | 398,468,336 | 63,754,933,760 | 786,432 | 81,069 | 63,755,255,808 |
| 810M | 1 | `6e-4` | 730,950,232 | 14,619,004,640 | 262,144 | 55,768 | 14,619,246,592 |
| 810M | 2 | `5.6e-4` | 730,950,232 | 29,238,009,280 | 393,216 | 74,357 | 29,238,362,112 |
| 1.2B | 1 | `4e-4` | 1,134,521,920 | 22,690,438,400 | 262,144 | 86,558 | 22,690,660,352 |
| 1.2B | 2 | `6e-4` | 1,134,521,920 | 45,380,876,800 | 393,216 | 115,410 | 45,381,058,560 |
| 810M | 4 | `4e-4` | 730,950,232 | 58,476,018,560 | 524,288 | 111,535 | 58,476,462,080 |
| 810M | 8 | `4e-4` | 730,950,232 | 116,952,037,120 | 786,432 | 148,713 | 116,952,662,016 |
| 1.2B | 4 | `3e-4` | 1,134,521,920 | 90,761,753,600 | 524,288 | 173,115 | 90,762,117,120 |
| 1.2B | 8 | `4e-4` | 1,134,521,920 | 181,523,507,200 | 786,432 | 230,820 | 181,524,234,240 |

## Historical wide-integration launch reference

These are the actual source-run settings, not settings inferred from the
materialized DDP checkpoint config. The 480M source checkpoints predate saved
trainer configs, so their Beaker specs are authoritative.

| Size | Cx | Hardware | GPUs | EP | Rank MB (sequences) | Accumulation | Notes |
|---|---:|---|---:|---:|---:|---:|---|
| 480M | 1 | Titan B200 | 4 | 1 | 4 | 2 | Finished in about 5.3 hours. |
| 480M | 2 | Titan B200 | 4 | 1 | 4 | 3 | Finished in about 8.5 hours. |
| 810M | 1 | Titan B200 | 8 | 1 | 4 | 1 | Full-node, no accumulation overhead. |
| 810M | 2 | Titan B200 | 8 | 1 | 2 | 3 | `MB4` is illegal for a 48-sequence batch on eight DP ranks. |
| 1.2B | 1 | Titan B200 | 8 | 8 | 2 | 16 | Used full expert parallelism on the old hardware/code path. |
| 1.2B | 2 | Titan B200 | 8 | 8 | 3 | 16 | Used full expert parallelism on the old hardware/code path. |

## Holmes B300 microbatch exploration

With TP/PP/CP disabled, every candidate must satisfy:

```text
global_sequences = GPUs * rank_microbatch_sequences * accumulation
```

For every row, the data loader's DP world size is the full dense DP world size;
EP creates a separate MoE mesh and does not reduce the number of data shards.
The configured rank microbatch is a cap: when it exceeds the local rank batch,
the effective microbatch is the local rank batch. We should measure both
tokens/second per job and per GPU; the largest microbatch is not automatically
the cheapest or fastest allocation.

| Size | Cx | Primary allocation to test | Conservative start | Larger legal probes | Current target / implication |
|---|---:|---|---:|---|---:|
| 480M | 1 | 4 B300, EP1 | MB4 | MB8 | MB8 |
| 480M | 2 | 4 B300, EP1 | MB4 | MB6, MB12 | MB12 |
| 810M | 1 | 8 B300, EP1 | MB4 | none on eight GPUs; optionally compare 4 GPUs / MB8 | MB4 |
| 810M | 2 | 8 B300, EP1 | MB2 | MB3, MB6 | MB6 |
| 1.2B | 1 | 8 B300, compare EP8 and EP1 | EP8 MB4 / EP1 MB4 | EP8 cap MB8 | EP8 cap MB8 was effectively MB4 with no accumulation; EP1 MB4 fits but is slower |
| 1.2B | 2 | 8 B300, compare EP8 and EP1 | EP8 MB4 / EP1 MB3 | EP8 cap MB12 / EP1 MB6 | EP8 cap MB12 was effectively MB6 with no accumulation; EP1 MB6 OOMs and MB3 (accumulation 2) fits |

The 1.2B smoke starts with the historically proven full-node EP8 layout. With
eight GPUs, the dense data-parallel world size remains eight while the MoE mesh
has EP-DP one and EP-MP eight. Consequently, Cx1 has four sequences per rank and
Cx2 has six. The original cap-MB8 and cap-MB12 probes therefore tested effective
MB4 and MB6, respectively, with no accumulation. Keep TP/PP/CP at one and start
without activation recomputation so the throughput measurement reflects the
fast path.

On this branch/image, the production-oriented `rowwise_nvshmem` EP path builds
but segfaults during symmetric-memory group startup on Holmes B300s. The EP8
comparison therefore uses Olmo-core's supported synchronized all-to-all
`sync_1d` path, which explicitly does not require symmetric memory. EP1 is run
alongside it to determine whether B300 compute throughput removes the historical
reason to prefer EP8 at 1.2B.

Each smoke should cover optimizer construction, compiled forward/backward, at
least ten real optimizer steps, and a checkpoint write. Evaluator memory is
tested in separate eval-only jobs. Record peak active memory, TFLOPs/GPU, tokens/second per GPU,
tokens/second for the whole job, skipped steps, EP degree, and compile state.

## B300 smoke results

Completed 2026-07-15. Throughput numbers are medians over clean optimizer
steps 3, 4, 8, and 9, excluding graph compilation, checkpointing, and the
historical evaluator bookkeeping used in those smokes. Every passing row ran
12 optimizer steps, compiled forward/backward, and at least one full
checkpoint. Future smokes do not run evaluators inside the training process.

| Size | Cx | GPUs | EP / path | MB cap / effective | Accum | TFLOPs/GPU | MFU | tokens/s/GPU | tokens/s job | Peak device GiB | Result |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 480M | 1 | 4 | EP1 | 8 | 1 | 478.6 | 21.27% | 167,030 | 668,120 | 170.7 | pass |
| 480M | 2 | 4 | EP1 | 12 | 1 | 519.9 | 23.11% | 181,435 | 725,740 | 227.4 | pass |
| 810M | 1 | 8 | EP1 | 4 | 1 | 342.5 | 15.22% | 65,128 | 521,024 | 175.6 | pass |
| 810M | 2 | 8 | EP1 | 6 | 1 | 427.9 | 19.02% | 81,376 | 651,008 | 217.8 | pass; Beaker saw a post-training live-wrapper exit only |
| 1.2B | 1 | 8 | EP8 / `sync_1d` | 8 / 4 | 1 | 415.5 | 18.47% | 51,430 | 411,440 | 189.5 | pass |
| 1.2B | 1 | 8 | EP1 | 4 / 4 | 1 | 387.3 | 17.21% | 47,946 | 383,568 | 250.7 | pass |
| 1.2B | 2 | 8 | EP8 / `sync_1d` | 12 / 6 | 1 | 465.1 | 20.67% | 57,580 | 460,640 | 255.4 | pass |
| 1.2B | 2 | 8 | EP1 | 6 / 6 | 1 | — | — | — | — | 267.0 / 267.7 capacity | OOM in compiled dry run |
| 1.2B | 2 | 8 | EP1 | 3 / 3 | 2 | 409.1 | 18.18% | 50,637 | 405,096 | 251.1 | pass |

Promote the tested 480M and 810M EP1 rows unchanged. For 1.2B, promote EP8
with `sync_1d`; the effective Cx1/Cx2 microbatches are MB4/MB6. Against the
largest fitting EP1 layouts, EP8 is 7.3% faster per GPU at Cx1 and 13.7% faster
at Cx2, while using substantially less memory at Cx1. The faster
`rowwise_nvshmem` path remains a future optimization target, but it is not
launchable on this image because its symmetric-memory process-group startup
segfaults.

## 275M gated-RoPE Cx1 parallelism study

The checkpoint-free study in
[`launchers/pretraining/manifests/275m_rope_gated_parallelism_smokes.yaml`](launchers/pretraining/manifests/275m_rope_gated_parallelism_smokes.yaml)
holds the model, 262,144-token optimizer batch, sequence length, LR, compile
mode, and 20-step duration fixed. It varies only GPU count, expert parallelism,
EP communication path, and rank microbatch. The authoritative allocated,
urgent Holmes batch is
[01KY0K5T6YNRPMB3P9N3YBX9M3](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY0K5T6YNRPMB3P9N3YBX9M3).
Two earlier zero-minimum-runtime submissions were stopped after an immediate
preemption; they are not result-bearing runs.

Throughput below is the median of the final ten reported optimizer-step
samples. Aggregate TPS is per-GPU TPS multiplied by world size. All passing
cells used 292,092,800 active parameters and 3,137,624,960 total parameters.

| GPUs | EP / path | Rank MB | Accum | TFLOPs/GPU | TPS/GPU | TPS job | Peak active GiB | W&B | Result |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | EP1 | 8 | 4 | 407.9 | 271,156 | 271,156 | 134.3 | [y5hdpz9h](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/y5hdpz9h) | pass |
| 2 | EP1 | 8 | 2 | 384.0 | 255,260 | 510,519 | 116.8 | [6wn6bf60](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6wn6bf60) | pass |
| 4 | EP1 | 8 | 1 | 343.1 | 228,058 | 912,230 | 108.0 | [vhraoh7g](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vhraoh7g) | pass |
| 2 | EP2 / `sync_1d` | 8 | 2 | 344.0 | 228,703 | 457,406 | 114.6 | [49r9mybk](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/49r9mybk) | pass |
| 4 | EP2 / `sync_1d` | 8 | 1 | pending | pending | pending | pending | pending | queued |
| 4 | EP4 / `sync_1d` | 8 | 1 | pending | pending | pending | pending | pending | queued |
| 2 | EP2 / `rowwise_nvshmem` | 8 | 2 | pending | pending | pending | pending | pending | queued; current-image startup re-test |
| 4 | EP4 / `rowwise_nvshmem` | 8 | 1 | pending | pending | pending | pending | pending | queued; current-image startup re-test |
| 1 | EP1 | 16 | 2 | pending | pending | pending | pending | pending | queued |
| 1 | EP1 | 32 | 1 | pending | pending | pending | pending | pending | queued; maximum legal MB |
| 2 | EP1 | 16 | 1 | pending | pending | pending | pending | pending | queued; maximum legal MB |

The completed fixed-MB8 controls show the systems tradeoff directly: one GPU
has the best compute efficiency, while four GPUs deliver the best job-level
throughput. On two GPUs, EP2 is 10.4% slower per GPU than EP1 and saves only
about 2.2 GiB/GPU of active memory, so EP1 remains the default at this scale.
The larger-microbatch and current-image NVSHMEM conclusions remain pending.

Scale smokes now retain only their final hard-stop checkpoint. The initial r4
1.2B Cx1 job predated that change and retained step 5, 10, and 12; each is
about 221 GB because distributed optimizer state is included. No checkpoint
was automatically removed.

## Beaker settings held fixed

- Cluster: `ai2/holmes`
- Workspace: `ai2/OLMo-3-moe-experiments`
- Priority: `urgent`
- Image: `01KW8G8JC20H11Y60PPTE2VN4Q` (current validated hybrid image)
- Shared memory: 64 GiB
- Checkpoints:
  `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/pretraining/`
- Compile on. Match v1 checkpoint retention: save a rolling ephemeral
  checkpoint every 500 steps with `remove=ephemeral_only`, and retain the final
  checkpoint permanently.

The promoted six-run Cx1/Cx2 allocation requires 40 concurrent GPUs: two
four-GPU 480M jobs and four eight-GPU 810M/1.2B jobs.

## Promoted 810M/1.2B Cx4/Cx8 settings

The 810M jobs reuse the demonstrated-fit Cx1/Cx2 effective microbatches. The
initial 1.2B launch incorrectly treated EP8 as reducing the data-DP degree to
one, so its MB8/MB12 caps became effective MB8/MB12 at Cx4/Cx8 and OOMed in the
compiled dry run. The retries correct those caps to the proven MB4/MB6 shapes.

| Size | Cx | GPUs | EP / path | MB cap / effective | Accum | Transferred LR |
|---|---:|---:|---|---:|---:|---:|
| 810M | 4 | 8 | EP1 | 4 | 2 | `4e-4` |
| 810M | 8 | 8 | EP1 | 6 | 2 | `4e-4` |
| 1.2B | 4 | 8 | EP8 / `sync_1d` | 4 / 4 | 2 | `3e-4` |
| 1.2B | 8 | 8 | EP8 / `sync_1d` | 6 / 6 | 2 | `4e-4` |

These four jobs were submitted together on 2026-07-15 as Beaker experiment
[`01KXKTT3ZT5G4V9QTFBR6MKGEZ`](https://beaker.org/ex/01KXKTT3ZT5G4V9QTFBR6MKGEZ),
requesting 32 B300 GPUs at urgent priority.

The corrected 1.2B retries were submitted as Beaker experiment
[`01KXKY6RTR5ZSD1R1BS40SF7KR`](https://beaker.org/ex/01KXKY6RTR5ZSD1R1BS40SF7KR),
requesting 16 B300 GPUs at urgent priority.

## Promoted 480M Cx4/Cx8 settings

The omitted 480M cells use the observed-best wide-integration LRs and the
canonical Cx4/Cx8 global batches. Cx4 uses the largest proven-safe legal
microbatch divisor below MB16; Cx8 uses the already validated MB12 shape.

| Size | Cx | GPUs | EP | MB cap / effective | Accum | Transferred LR |
|---|---:|---:|---:|---:|---:|---:|
| 480M | 4 | 4 | EP1 | 8 / 8 | 2 | `8e-4` |
| 480M | 8 | 8 | EP1 | 12 / 12 | 1 | `8e-4` |

Submitted together on 2026-07-16 as Beaker experiment
[`01KXMTAQPTG52EPEXMQN0Q1YJ7`](https://beaker.org/ex/01KXMTAQPTG52EPEXMQN0Q1YJ7)
at urgent priority. Both tasks are queued pending Holmes allocation.
