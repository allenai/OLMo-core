# Small 3:1 hybrid pair: configuration and budget

Planning snapshot: September 18, 2026. No 3:1 training, sweep, bucket, uploader
registration, or controller has been launched. The architecture builder is
`olmoe3_small_hybrid_3to1.py`; this is not yet an end-to-end training launcher.
Base: qualified hero commit `b34d2972fcdc19a4bc0425cde43b88b5c9310cf5`.
Branch: `codex/small-hybrid-3to1-20260918`.

## Architecture: exactly two sequence-mixer substitutions

| Setting | Existing 7:1 | Proposed 3:1 |
|---|---:|---:|
| Layers | 16 | 16 |
| KDA / full-attention layers | 14 / 2 | 12 / 4 |
| Full-attention layers, one-based | 8, 16 | 4, 8, 12, 16 |
| Active parameters, including embeddings | 794,233,472 | 787,364,992 |
| Active non-embedding parameters | 691,473,024 | 684,604,544 |
| Total parameters | 12,496,341,632 | 12,489,473,152 |

Unchanged: d_model1024, latent512 (exactly half), head_dim128, Q/KV8/4,
512 routed experts, native top16, one shared expert, softmax routing with
renormalized top-K weights, first dense-FFN layer, KDA expand_v2 and negative
eigenvalues, gated NoPE full attention with scalable softmax and per-head QK
gains, vocabulary100352, norms and initialization. Full attention here has
fewer parameters than KDA; do not compensate by changing another dimension.
Its KV cache at a fixed context length is twice the 7:1 model's because there
are twice as many full-attention layers.

EMO is an explicit builder argument. Enabled uses the same document pool
min16/max512/eval512; disabled sets every routed block's EMO config to None.
For the paired experiment, keep EMO on throughout PT and its decay in the EMO
arm, and off throughout in the other. No MT/LC/SFT is part of this budget.

CPU verification: both configs validate, all parameter counts are computed by
the actual OLMo-core config classes, and a recursive comparison against the
canonical 7:1 config with hero QK gains confirms that only the sequence mixers
at zero-based blocks3 and11 differ. EMO-only differences between the two new
variants, cadence/fork counts, batch arithmetic and WSD boundary/midpoint LRs
also passed CPU checks. This is not a GPU memory/speed qualification.

## Matched ~2T experiment

Use the exact previous endpoint:120000 steps =2,013,265,920,000 tokens.
This is ~2T, not exactly2,000,000,000,000. At the existing batch size, an exact
decimal2T budget would be119209.29 steps; rounding to120000 preserves the old
comparison point and clean10% decay.

| Phase | Steps | Tokens |
|---|---|---:|
| Linear warmup | 0–2000 | 33.554B |
| Stable plateau | 2000–108000 | 1,778.385B |
| Linear decay to zero | 108000–120000 | 201.327B |
| Total | 120000 | 2,013.266B |

The decay fork is step108000 =1.811939328T. Stop the stable trunk at that point
and resume its complete optimizer/data state into a separately registered decay
lineage. Do not also train the stable tail unless an undecayed comparison is
explicitly requested: that would add201.327B per arm (~9.0–9.3h each).

Per arm:64B300 GPUs,8 nodes,16,777,216 global tokens,sequence8192,MB4,GA8,
DP64/EP1/PP1, BF16/FP32-optimizer, no MXFP8 or activation recomputation, same
qualified optimizations and pinned environment as the live heroes. Same local
Dolma3.5 manifest/order, init seed12536, data seed928543231, AdamW betas0.9/0.95,
weight decay0.1 with existing exceptions, grad clip1.0, synchronous checkpoints.
Peak LR is deliberately unset pending tuning/approval. No warmup restart at
the decay fork. Both arms should use the same selected LR for a controlled EMO
comparison. These would be fresh initializations, not converted 7:1 weights.

### Time and compute, conditional on unchanged throughput

Use94.2–96.8k effective TPS/GPU, or6.03–6.19M tokens/sec per64GPU job.

| Work | Elapsed on64GPUs | GPU-hours |
|---|---:|---:|
| One warmup | 1.5–1.55h | 96–99 |
| One trunk through1.812T (includes warmup) | 81.3–83.5h | 5202–5344 |
| One201.3B decay | 9.0–9.3h | 578–594 |
| One complete2.013T run | 90.3–92.8h (3.76–3.87days) | 5780–5938 |
| Both, serial on64GPUs | 180.6–185.6h (7.53–7.73days) | 11560–11876 |
| Both, concurrent on128GPUs | 90.3–92.8h | 11560–11876 |

Queueing and extra process startup are excluded. More frequent early saves
could lower effective speed versus the current500-step cadence. The 3:1 model's
speed and MB4 fit must be measured; changing attention ratio changes both work
and activation memory. Keeping the existing non-EMO hero on64GPUs means the
new pair concurrently would require192 training GPUs total, not128.

## Checkpoint storage

Use the revised hero cadence, including step0:

| Saved steps | Cadence | Checkpoints per arm |
|---|---|---:|
| 0–18000 (~302B) | step0 plus every100 | 181 |
| 18250–60000 (~1.007T) | every250 | 168 |
| 60500–120000 (~2.013T) | every500 | 120 |
| Total | | 469 |

The fork is already on cadence. Count it once, using the parent as a read-only
source. A launcher that also writes a new fork copy adds one checkpoint per
arm; do not count that as a new logical training point.

Measured current full-state64GPU checkpoint:150,078,748,181 bytes,1156 files.
Adjusting for the6,868,480 fewer parameters at approximately12bytes/parameter
gives149,996,326,421 bytes, i.e. budget **150GB per checkpoint**. This is an
estimate until the first3:1 save; file count and small metadata overhead must
be measured rather than assumed identical. Includes model/master weights and
optimizer/trainer state; excludes HF-inference conversions and eval artifacts.

- All saved checkpoints: **70.35TB per arm;140.70TB combined**, decimal units,
  before any HF/Xet deduplication. This is remote archival volume, not a required
  local working set when uploading and verified cleanup are enabled.
- Two retained local checkpoints per active run alone are0.60TB combined.
  With the existing1h deletion grace, early100-step cadence, verification and
  in-flight writes, expect roughly4–5TB for the pair **only if uploads keep up**;
  reserve6–8TB plus existing runs and any backlog. Finished parent lineages may
  retain additional checkpoints after the decay handoff (up to0.6TB for two
  parents with keep2); explicitly plan their eventual retention separately.
- Initial saving rate for the pair is1.08–1.11GB/sec (~150GB per arm every4.5–4.6min).
  Every250 steps:0.43–0.44GB/sec (~11.3–11.6min). Every500:0.216–0.221GB/sec
  (~22.6–23.2min). Existing non-EMO adds~0.108GB/sec at its500-step cadence.
- The early phase lasts~13.5–13.9h. Do not assume the uploader can sustain the
  initial rate: recent one-attempt150GB uploads+verification took~3.5–10min
  per checkpoint, and retries add delay. Aggregate capacity depends on worker
  concurrency and other campaigns. At a sustained1.1GB/sec production rate but
  only0.7GB/sec upload capacity, a14h interval creates~20TB of extra backlog.
  Use existing warning/stop guards and be prepared to stagger or pause trainers.
- If "previous cadence" instead means the original100-step policy all the way
  to1T, it is721 checkpoints/arm, ~108.15TB each/~216.3TB combined.

## LR tuning: reproduce or simplify the previous sweep

The previous sweep **used EMO**, the optimized small model,64GPUs,16Mi batch,
2000 warmup steps and6000 endpoint steps (100.663296B tokens, the rounded~Cx8
budget). Initial five LRs were0.000325,0.00065,0.0013,0.0026,0.0052;0.0104 was
added to bracket the minima. Each6000-step stable trunk spawned300/600/1200/1800
step linear decays (5/10/20/30%), all ending at the same6000-step endpoint.
Forks were5700/5400/4800/4200. Decay branches do not repeat warmup. Trunks kept7
checkpoints at300-step intervals; decay children kept1, with verified cleanup.
Score with the ladder's final128-point drift-corrected training-CE function.

| Option (one EMO setting) | Total trained tokens | GPU-hours | Time using one64GPU slot |
|---|---:|---:|---:|
| Original five LRs + four decays | 830.472B | 2384–2450 | 37.3–38.3h |
| Full final six LRs + four decays | 996.567B | 2861–2940 | 44.7–45.9h |
| Proposed five LRs,10% only | 503.316B | 1445–1485 | 22.6–23.2h |

The full final sweep is6x(6000+300+600+1200+1800)=59400 executed steps,30jobs.
Its cost is~25% of the new two-arm2T experiment; rerunning it separately with
and without EMO would double that tuning cost. More GPU slots reduce wall time
but not total GPU-hours; decay dependencies/startup/queues prevent perfect
scaling. A single100.66B point takes~4.5–4.6h of training on64GPUs.

Recommended cost-conscious starting proposal, not yet approved: five LRs
**6.5e-4,1.3e-3,2.6e-3,5.2e-3,1.04e-2**, EMO on,10% decay only since that is the
chosen production decay. Train each trunk to5400 then its600-step decay, not
to6000 plus a duplicate600-step tail. Five such jobs take~13.5–13.9h in three
waves on128GPUs, or~4.5–4.6h if all five have64GPUs each; startup/save/restore
adds overhead. Extend LR bounds if needed. Optionally validate the winning LR
and its neighbors without EMO before choosing one common LR for both2T arms.
If decay-fraction sensitivity remains a research question, use the full repeat.

Transfer the newly fitted100.66B optimum to2.013T using the previous heuristic:
`LR_2T = LR_100B * (120000 / 6000)^(-0.29) = 0.41947 * LR_100B`.
This is a heuristic, not a proven optimum. Do not silently reuse the old1.1e-3
chosen for a14T horizon, nor assume the7:1 optimum survives the architecture
change. If these new trunks should instead be optimized for future14T
continuation, explicitly choose that horizon before setting their LR.

Sweep archival storage at150GB per snapshot: a full six-LR repeat writes
6x(21 trunk saves +1+2+4+6 new decay saves)=204 snapshots,~30.6TB, excluding
unnecessary duplicated fork copies. Five-LR10%-only writes5x21=105,~15.75TB.
Existing frozen trunk keep7 policies need adjustment after branches start;
the6trunks+24decays alone retain66snapshots (~9.9TB), before grace/backlog.
These volumes are additional to the140.7TB hero pair archive.

## Before any launch

1. Confirm10%-only versus four-decay sweep, common LR policy, and GPU budget.
2. Integrate the builder into isolated campaign/lineage names; inherit the live
   hero's pinned package setup and optimizations, not a newer unqualified stack.
3. Perform a short64GPU MB4 correctness/memory/speed and save/resume smoke, with
   both EMO settings. Verify3:1 layer placement and inference-conversion mapping.
4. Register distinct trunk/decay upload prefixes; preserve model+optimizer state,
   step0, source fork, and verified-before-delete protections. No new bucket or
   cleanup policy changes are authorized by this planning request.
5. Queue only after approval; report measured speed/storage against these estimates.

## Requested operational change, completed separately

EMO experiment `01M2SSCRMSS386QSDJW4HT332H` was gracefully paused via its W&B
cancel tag. All8 tasks exited0. Final full-state checkpoint:step300411,
5,040,060,235,776 tokens, all64 resume-audit/trainer-state files present. Its
HF upload and payload verification are complete. Closest5T checkpoint,step298000
(4,999,610,368,000 tokens), is already verified in HF; its local copy has been
removed by the existing uploader retention policy, not this task.
Non-EMO experiment `01M2SSD1BED58RE6EZ7T599VM2` remains running unchanged on64GPUs.
