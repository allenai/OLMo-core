# Gated DeltaNet 2 integration

GDN2 is implemented as a separate OLMo-core sequence mixer. Existing GDN1
configs, checkpoints, imports, and image references are unchanged.

## Dependency isolation

GDN2 currently runs through an ephemeral Python overlay installed by
`src/scripts/train/jacobm_olmoe3_gdn2_beaker.sh`. The wrapper installs exactly:

```text
flash-linear-attention[cuda] @ git+https://github.com/fla-org/flash-linear-attention.git@cbb0a72efb55c18ca0ef4f298298317573ad2cb3
```

It uses `pip --target /tmp/... --no-deps --no-build-isolation`, prepends that
temporary directory only for the GDN2 process, asserts FLA version 0.5.2 and
the presence of `fla.ops.gdn2`, and then calls the ordinary hybrid wrapper.
Normal GDN1 jobs do not set the overlay variable and continue to use the
existing image environment. A derived immutable image can replace this
temporary mechanism after the architecture is selected.

## Audited 275M candidate

The first variant is `geometry_275m_gdn2_ev2_rope_gated`. Relative to the
current gated-RoPE geometry model it changes only the eight GDN1 mixers to
GDN2. It retains:

- 640 model width, ten layers, and GDN layers 0--3 and 5--8;
- global gated RoPE attention at layers 4 and 9;
- eight QK/value heads, 128-dimensional keys, and `expand_v=2`;
- `allow_neg_eigval=true`, four-token short convolutions, all MoE dimensions,
  dense-first FFN placement, initialization, data, and optimization settings.

GDN2 adds channel-wise decay and independent channel-wise erase/write gates,
plus the canonical low-rank decay and sigmoid output-gate projections. This
adds 1,762,296 parameters per recurrent mixer.

| Variant | Active params | Active non-embedding | Total params |
|---|---:|---:|---:|
| GDN1 gated-RoPE geometry | 292,092,800 | 227,867,520 | 3,137,624,960 |
| GDN2 gated-RoPE geometry | 306,191,168 | 241,965,888 | 3,151,723,328 |
| Difference | +14,098,368 | +14,098,368 | +14,098,368 |

## Functional gate and throughput results

The compiled one-GPU MB1 smoke passed its 8,192-token forward/backward dry run
and six real optimizer steps with finite loss and gradients, zero skipped
updates, and exit code zero. Steady active/reserved memory was approximately
64.4/65.4 GiB. The result-bearing work is
[01KY8MNNDVT0BMFD20F7ZSS95P](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8MNNDVT0BMFD20F7ZSS95P).

The full systems comparison followed the completed GDN1/SWA protocol:

1. qualify MB8 and MB16 on one GPU at both 2 Mi and 4 Mi optimizer batches;
2. if MB16 passes, run the exact 16-cell 2/4/8-GPU EP1, code-default EP, and
   reduce-scatter matrix for 50 steps with compile enabled;
3. save no checkpoints or evals and record final-ten median TFLOPs/GPU,
   TPS/GPU, aggregate TPS, step time, MFU, peak active/reserved memory, and
   skipped-step count alongside the GDN1 and SWA rows.

The four-cell capacity work
[01KY8N6V09MWSNSY51BBWD4X33](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8N6V09MWSNSY51BBWD4X33)
passed 4/4 tasks. MB16 fit at both optimizer batches, peaking at approximately
253.7 GiB active and 257.0 GiB reserved memory. The full matrix work
[01KY8NMMR9AETXTGWR11Y51QWQ](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8NMMR9AETXTGWR11Y51QWQ)
passed 16/16 tasks. All 20 jobs completed with zero skipped optimizer updates.

The table reports final-ten median TFLOPs/GPU and thousands of tokens/s. All
cells use MB16.

| Batch | 1 GPU EP1 | 2 GPU EP1 | 4 GPU EP1 | 4 GPU RS | 8 GPU EP1 | 8 GPU RS |
|---:|---:|---:|---:|---:|---:|
| 2 Mi | 420.4 / 260.3 | 407.0 / 252.0 | 404.3 / 250.3 | 402.6 / 249.3 | 385.1 / 238.4 | 387.8 / 240.1 |
| 4 Mi | 413.9 / 256.3 | 413.7 / 256.2 | 407.5 / 252.3 | 409.0 / 253.3 | 404.0 / 250.1 | 399.5 / 247.3 |

Across these matched cells, GDN2 processes 11.1--13.4% fewer tokens/s than
GDN1 and 32.1--36.0% fewer than SWA. Its reported TFLOPs/GPU are 5.5--7.9%
below GDN1, a smaller gap than raw throughput because GDN2 performs more
modeled work per token. Code-default full EP is 12.6--17.6% slower than EP1 at
the same world size. Reduce-scatter ranges from 1.1% slower to 0.7% faster, so
it does not offer a consistent benefit. EP1/all-reduce remains the default.

The complete machine-readable GDN1/GDN2/SWA results, including aggregate TPS,
step time, MFU, memory, Beaker job IDs, and W&B links, are in
`results/throughput/275m_gdn_gdn2_swa_large_batch_parallelism.csv`. Re-run
`results/throughput/collect_275m_throughput.py` to merge finalized Beaker work
idempotently.

### Backward recomputation diagnostic

FLA's opt-in `disable_recompute=True` path retains forward WY/state
intermediates to avoid reconstructing them in the backward. The exact one-GPU,
2 Mi-token, MB16 A/B work
[01KY8RRSFNTBCVHJCEC439GW11](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8RRSFNTBCVHJCEC439GW11)
did not fit. After compiling the first dry-run microbatch, the second attempted
a 24.50 GiB allocation with only 17.70 GiB free; PyTorch already held 245.55
GiB allocated. This was a real capacity failure, not fragmentation (only 0.37
GiB was reserved but unallocated).

The earlier MB20 fit boundary belonged to GDN1. GDN2 with normal recomputation
already peaks at 253.7 GiB active / 257.0 GiB reserved at MB16. Keep
`disable_recompute=False` for production and the first LR sweep. The opt-in
flag remains available for a separate smaller-MB kernel diagnostic but must
not be enabled implicitly.

## 275M LR sweep

The first GDN2 quality experiment mirrors the completed 275M geometry-matched
GDN1 gated-NoPE sweep. It was submitted as 16 urgent, unallocated Holmes tasks
in [work 01KY8TKEBSZHYBZYEC5NFB92YK](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8TKEBSZHYBZYEC5NFB92YK).
It uses normal backward recomputation, EP1, compile, no in-loop evals, and the
established four-LR grid `4e-4`, `8e-4`, `1.6e-3`, `3.2e-3` at every Cx.
The model has NoPE on global-attention layers 4 and 9; every other model and
training setting matches the audited GDN2 gated-RoPE systems recipe.

The initially submitted gated-RoPE work
[01KY8SY728GAJN9MZ5B9VGZNP2](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY8SY728GAJN9MZ5B9VGZNP2)
was canceled during compilation/dry-run, before its first optimizer step or
checkpoint, and is superseded by the gated-NoPE work above.

| Cx | Target tokens | Approx. steps | Global batch | GPUs | Rank MB | Approx. wall time/run |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.839B | 18,461 | 262,144 | 4 | 8 | 1.4--1.6 h |
| 2 | 9.679B | 24,615 | 393,216 | 4 | 12 | 2.7--3.0 h |
| 4 | 19.357B | 36,922 | 524,288 | 4 | 16 | 5.4--5.8 h |
| 8 | 38.715B | 49,229 | 786,432 | 8 | 12 | 5.6--6.0 h |

This is 16 jobs and 80 concurrent GPUs, approximately 340--350 GPU-hours if
all cells run cleanly. Token budgets follow the usual `20 * active
non-embedding parameters * Cx` rule and are therefore about 6.2% larger than
the GDN1 gated-NoPE budgets. Use the ordinary 10%-of-tokens warmup followed by
cosine decay to 0.1x peak LR, the established ephemeral-checkpoint policy, and
out-of-loop validation. Per the corrected final comparison decision, plot one
GDN2 U-curve per Cx and its observed-best summary against only original wide
integration and the otherwise-matching geometry-matched gated-NoPE GDN1 model.

Status at 2026-07-24 16:10 UTC: Cx1/Cx2/Cx4 are complete and bracketed. Their
observed best LRs are `1.6e-3` at all three multiples, with strict final-250M
CEs `2.646730`, `2.534116`, and `2.443132`. The final Cx2 result combines the
failed and resumed W&B segments. Three Cx8 points finished; `8e-4` currently
leads with CE `2.356985`. Cx8 `1.6e-3` stopped on an explicit non-finite-loss
assertion at step 36,768 and has durable `step36500`, so Cx8 remains marked
provisional pending a resume.

## Larger-scale transfer

The 480M, 810M, and 1.2B candidates inherit the corresponding gated-NoPE GDN1
geometry exactly and replace only recurrent GDN1 mixers with GDN2. In
particular, dimensions, layer count and placement, MoE widths, dense-first FFN,
GQA, full-attention gates, NoPE, initialization, data, and optimizer recipe are
unchanged. Normal backward recomputation remains enabled.

| Size | Width | Layers | GDN2 / full | Q / KV | Expert hidden | Active | Active non-emb. | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 480M | 768 | 15 | 12 / 3 | 8 / 4 | 840 | 526,979,424 | 449,909,088 | 7,246,549,344 |
| 810M | 1,024 | 15 | 12 / 3 | 16 / 8 | 1,032 | 914,540,480 | 811,780,032 | 11,921,835,968 |
| 1.2B | 1,280 | 20 | 16 / 4 | 16 / 8 | 952 | 1,376,964,352 | 1,248,513,792 | 18,602,528,512 |

The full-attention layers are 4/9/14 for 480M and 810M and 4/9/14/19 for
1.2B. All other layers use GDN2 with `expand_v=2`; layer 0 retains the dense
FFN. The exact active increase over matching gated-NoPE GDN1 is 23.48M, 50.01M,
and 77.04M, respectively. The builder audits exact parameter counts and proves
that normalizing GDN2 back to GDN1 reproduces the parent config exactly through
`as_dict()`.

The proposed production transfer is one run per size/Cx at the observed wide
integration LR for that same cell:

| Size | Cx1 | Cx2 | Cx4 | Cx8 |
|---|---:|---:|---:|---:|
| 480M | `1.2e-3` | `9e-4` | `8e-4` | `8e-4` |
| 810M | `6e-4` | `5.6e-4` | `4e-4` | `4e-4` |
| 1.2B | `4e-4` | `6e-4` | `3e-4` | `4e-4` |

Token budgets continue to use `20 * active_non_embedding * Cx` and are rounded
up only to the next whole optimizer batch:

| Size | Cx1 tokens / steps | Cx2 tokens / steps | Cx4 tokens / steps | Cx8 tokens / steps |
|---|---:|---:|---:|---:|
| 480M | 8.998B / 34,326 | 17.996B / 45,768 | 35.993B / 68,651 | 71.985B / 91,535 |
| 810M | 16.236B / 61,934 | 32.471B / 82,579 | 64.942B / 123,868 | 129.885B / 165,158 |
| 1.2B | 24.970B / 95,255 | 49.941B / 127,006 | 99.881B / 190,509 | 199.762B / 254,011 |

The wall-clock candidate preserves the established accumulation-free scale
layout. It uses 8 GPUs for 480M Cx1/2/4, 16 for 480M Cx8, 16 for every 810M
cell, 16 for 1.2B Cx1/2, and 32 for 1.2B Cx4/8. EP stays at 1 except for the
1.2B cells, which retain EP8 with `sync_1d`.

| Size / Cx | GPUs | Rank MB | Global batch | Approx. time |
|---|---:|---:|---:|---:|
| 480M Cx1 / Cx2 / Cx4 / Cx8 | 8 / 8 / 8 / 16 | 4 / 6 / 8 / 6 | 262K / 393K / 524K / 786K | 5 / 8 / 14 / 14 h |
| 810M Cx1 / Cx2 / Cx4 / Cx8 | 16 / 16 / 16 / 16 | 2 / 3 / 4 / 6 | 262K / 393K / 524K / 786K | 10 / 15 / 26 / 42 h |
| 1.2B Cx1 / Cx2 / Cx4 / Cx8 | 16 / 16 / 32 / 32 | 2 / 3 / 2 / 3 | 262K / 393K / 524K / 786K | 28 / 27 / 42 / 61 h |

Those times extrapolate the matching GDN1 production runs using the measured
275M GDN2 token-throughput penalty and the larger GDN2 token budgets; they are
planning estimates, not measurements. The 12 jobs require 200 GPUs concurrently
and roughly 6.0K GPU-hours if every cell runs cleanly.

The 480M and 810M checkpoint-free compiled qualification completed successfully
on 2026-07-24. All five Beaker replicas exited 0, all three W&B runs finalized,
and every run completed 50 optimizer steps with finite loss/gradients and zero
skipped steps.

| Model / layout | Final-10 TFLOPs/GPU | Final-10 TPS/GPU | Aggregate TPS | Active / reserved memory |
|---|---:|---:|---:|---:|
| 480M MB8, 8 GPU EP1 | 335.9 | 114.1K | 912.5K | 190.4 / 196.9 GiB |
| 480M MB6, 16 GPU EP1 | 287.4 | 97.6K | 1.562M | 150.4 / 155.3 GiB |
| 810M MB6, 16 GPU EP1 | 358.0 | 66.8K | 1.069M | 224.5 / 234.1 GiB |

The qualification works are
[480M MB8/8 GPU](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY9GKF3MDHPFTKM9NJ9A0W81),
[480M MB6/16 GPU](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY9GKJ0BJ6ED4DKSKD3BHD3Y),
and [810M MB6/16 GPU](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KY9GKP0A1BW8XSF8C3W6G4J0).
The balanced 1.2B qualification also passed on 2026-07-24 at commit
`f886c7b79`. All seven replicas exited 0 after 50 finite optimizer steps with
zero skipped steps:

| 1.2B layout | Final-10 TFLOPs/GPU | Final-10 TPS/GPU | Aggregate TPS | Active / reserved memory |
|---|---:|---:|---:|---:|
| MB4, 8 GPU, EP8 `sync_1d` | 367.2 | 45.1K | 360.7K | 189.7 / 198.8 GiB |
| MB4, 16 GPU, EP8 `sync_1d` | 358.2 | 44.0K | 703.8K | 176.5 / 185.7 GiB |
| MB3, 32 GPU, EP8 `sync_1d` | 286.3 | 35.2K | 1.125M | 133.5 / 140.6 GiB |

The qualification works are
[8 GPU](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYAGA7P0W3HKZ8FDDGT80GEP),
[16 GPU](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYAGAASYT6FHRJ86DB2NRQPK),
and [32 GPU](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYAGAERF6G60VSQQYNN8KR7X).
All model sizes are now capacity-qualified. The launcher enforces this in:
`launchers/pretraining/manifests/geometry_matched_scale_gdn2_nope_gated_full_candidate.yaml`.

The eight qualified 480M/810M production cells were submitted on 2026-07-24
from commit `ed7accc25`, urgent and unallocated on Holmes. They use the
transferred wide LRs in the table above, accumulation factor 1, normal GDN2
backward recomputation, rolling ephemeral checkpoints, and no in-loop evals.
The submission ledger is
`launchers/pretraining/generated/geometry_matched_scale_gdn2_nope_gated_480m_810m_submissions.json`.
At this point the four 1.2B cells were not yet submitted.

After the balanced qualification passed, the four 1.2B production cells were
submitted urgent and unallocated on Holmes from commit `cd40c04a5`. They use
8/16/16/32 GPUs for Cx1/2/4/8, EP8 `sync_1d`, MB4/3/4/3, accumulation one,
and the transferred LRs `4e-4`/`6e-4`/`3e-4`/`4e-4`. Their submission ledger is
`launchers/pretraining/generated/geometry_matched_scale_gdn2_nope_gated_1p2b_submissions.json`.

At the 2026-07-24 16:10 UTC refresh, 480M Cx1/Cx2 finished with strict
final-250M CEs `2.468555` and `2.359149`. 810M Cx4 is running at 21.04B / 64.94B
tokens with an approximately 16-hour ETA. The remaining five cells stopped on
the optimizer's explicit non-finite checks after successful multi-thousand-step
training: 480M Cx4 at step 4,497 (total gradient), 480M Cx8 at 4,064 (total
gradient), 810M Cx1 at 3,524 (total gradient), 810M Cx2 at 43,979 (loss), and
810M Cx8 at 14,994 (loss). These are neither config failures nor OOMs; durable
checkpoints exist at steps 4,000, 4,000, 3,500, 43,500, and 14,500,
respectively. No automatic relaunch was performed during the refresh.

At the 2026-07-24 19:36 UTC refresh, the two completed results are unchanged:
480M Cx1/Cx2 have strict final-250M CEs `2.468555` and `2.359149`. Every other
production cell is running or queued under its original Beaker experiment ID.
The retry operation is `beaker experiment resume <experiment-id>`, which is the
CLI equivalent of the UI restart button: it creates another attempt for the
existing task and preserves the experiment ID, configuration, and checkpoint
path. It does not submit a duplicate experiment.
For a multi-task experiment it retries every failed task, so redundant attempts
must be canceled individually with `beaker job cancel <job-id>` when one task
has a nonzero wrapper exit despite already producing complete results.

Five stopped cells were restarted this way during the refresh: 480M Cx8 from
durable `step9500`, 810M Cx1 from `step10000`, 810M Cx2 from `step56500`, 1.2B
Cx1 from `step6500`, and 1.2B Cx2 from `step2500`. The latest failures were all
the same GDN2 numerical-instability family seen previously—an explicit
non-finite loss or total-gradient assertion followed by CUDA/NCCL teardown—not
configuration errors, OOMs, or infrastructure failures. 480M Cx4 also crossed
several in-place attempts after repeated non-finite-gradient assertions around
step 4662 and is now past step 11,900. At snapshot time, 480M Cx4/Cx8, 810M
Cx1/Cx4/Cx8, and 1.2B Cx4/Cx8 were running; 810M Cx2 and 1.2B Cx1/Cx2 were
queued or starting.

At the 2026-07-24 20:43 UTC audit, five cells had failed again after the prior
resume: 810M Cx1/Cx2 at steps 10,039/56,755 and 1.2B Cx1/Cx4/Cx8 at steps
8,029/8,456/7,125. Their durable checkpoints are `step10000`, `step56500`,
`step8000`, `step8000`, and `step7000`, respectively. Every failure is still
an explicit non-finite loss or total-gradient assertion rather than an OOM or
configuration failure.

Those five cells were stopped and relaunched from the same run names and
checkpoint directories at commit `e267bb23e`. Changing environment variables
requires a new Beaker experiment specification; an in-place Beaker retry would
reuse the old frozen environment. The diagnostic continuations are urgent,
unallocated Holmes jobs:

| Cell | Resume checkpoint | Diagnostic Beaker |
|---|---:|---|
| 810M Cx1 | `step10000` | [01KYAXSHTRHNHW1RR34B23RD61](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYAXSHTRHNHW1RR34B23RD61) |
| 810M Cx2 | `step56500` | [01KYAXSMKACHN3S9G81J9Q0QKT](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYAXSMKACHN3S9G81J9Q0QKT) |
| 1.2B Cx1 | `step8000` | [01KYAXSQ8S38PT84B7H9S7GCYT](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYAXSQ8S38PT84B7H9S7GCYT) |
| 1.2B Cx4 | `step8000` | [01KYAXSTD8W8ZDSC6RC377TPVM](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYAXSTD8W8ZDSC6RC377TPVM) |
| 1.2B Cx8 | `step7000` | [01KYAXSXACNR214DEBRSQ8X671](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYAXSXACNR214DEBRSQ8X671) |

The launcher passes `OLMO_DDP_DEBUG_NONFINITE_GRAD=1`, all-rank reporting,
top-50 offending parameters, gradient-norm checks every 20 steps on every rank,
and a warning threshold of 100. The branch also fixes the optimizer diagnostic
step label and wires the previously incomplete large-gradient warning path.
The other five unfinished production cells—480M Cx4/Cx8, 810M Cx4/Cx8, and
1.2B Cx2—remain running without a new failure in this audit.

The first diagnostic attempts produced results by 2026-07-24 21:10 UTC. 810M
Cx1 and 1.2B Cx1 reproduced non-finite total gradients at steps 10,039 and
8,029. On rank 0, every optimizer-visible parameter gradient was NaN
(`361/361` and `481/481`); the other ranks reported `358/361` and `478/481`.
For 810M Cx1 both replicated and sharded dense/DP components were NaN. For
1.2B Cx1 the dense/DP and expert-sharded components were NaN. The last logged
finite metrics were ordinary: CE/total-grad-norm `2.954/0.1952` at step 10,030
and `2.808/0.3100` at step 8,020.

810M Cx2 and 275M Cx8 failed earlier in the optimizer on non-finite reduced
loss at steps 56,755 and 36,768, so the per-parameter gradient diagnostic did
not run. Their last logged finite CE/grad-norm pairs were `2.296/0.1712` at
step 56,750 and `2.408/0.0901` at step 36,760. None of the four jobs emitted a
large-gradient warning at the configured threshold of 100; the transition is
abrupt rather than a visible gradual norm explosion.

These reports are collected after loss and gradient all-reduces, so a NaN from
one local rank contaminates every rank and nearly every parameter before the
current diagnostic observes it. The reports therefore rule out an isolated
optimizer parameter but do not identify the originating rank or layer. The
next useful deterministic replay needs per-rank checks before loss reduction
for the loss failures and pre-DDP-reduction activation/backward checks around
the GDN2 layers for the gradient failures. No further retry should be launched
until that instrumentation exists. The 1.2B Cx4 diagnostic crossed its prior
failure step 8,456 and reached step 8,690; 1.2B Cx8 was still starting from
`step7000` at this snapshot.

The non-diagnostic 1.2B Cx2 continuation also failed during this audit at step
5,097 on non-finite total gradients, with durable `step5000`. Its earlier
failure was at step 2,724, so unlike the checkpoint-deterministic Cx1 and loss
failures, this cell advanced through one unstable point and later encountered
another. It has no per-parameter report because it was launched before the
diagnostic environment was added. It should join the pre-reduction diagnostic
set rather than receive another ordinary retry.

At 2026-07-24 21:16 UTC, the five newly failed cells were explicitly restarted
in place to test whether they can advance to another durable checkpoint. The
diagnostic experiments retain their existing all-rank debug environment:
810M Cx1 from `step10000`, 810M Cx2 from `step56500`, 1.2B Cx1 from
`step8000`, and 275M Cx8 `1.6e-3` from `step36500`. The older 1.2B Cx2
experiment was restarted from `step5000`; because it predates the diagnostic
launches, this attempt does not have the new environment. All five new Beaker
attempts were accepted at urgent priority under the existing experiment IDs.
