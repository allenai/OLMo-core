# Small hero EMO comparison

This branch freezes the qualified `optimized100b` small configuration. The two
production arms differ only in EMO routing, not in other performance flags.

| Setting | Both arms |
|---|---|
| Active / total parameters | 794,233,472 / 12,496,341,632 (including per-head QK gains) |
| Architecture | 16 layers, d1024 / latent512, 14 KDA + 2 FA, Q8 / KV4 |
| Experts | 512, top16, one shared; EMO pools16–512, eval512 when enabled |
| Systems | 64 B300 GPUs, PP1 / EP1 / DP64, MB4, GA8, sequence8192 |
| Precision | BF16 compute, FP32 optimizer/reductions; no MXFP8 or activation recomputation |
| Optimization | Qualified core-docpool-top16-wgrad-rs, kernel-fun7a6983b, KDA MIN_CTAS128, inverse scatter |
| Batch / LR | 16,777,216 tokens; 1.1e-3; 2,000-step warmup |
| Schedule | Constant-after-warmup WSD trunk; no decay at the initial stopping point |
| Initial stop | 179,000 steps = 3,003,121,664,000 tokens |
| Continuable horizon | 834,466 steps = 14,000,016,326,656 tokens |
| Checkpoints | Step0; every100 through60000, every500 thereafter; final/off-cadence interruption saves |
| Data | Complete local Dolma3.5 mirror `/weka/dolma-3p5/ai2-llm`; canonical manifest/order/filter unchanged |
| Seeds | Initialization12536; data928543231 |
| Evaluation | Same eleven held-out validation sets, every1000 steps and on finish; GCS credentials retained |
| Workspace | ai2/olmo3p5-training, urgent, allocated training (minRuntime1h) |

## Storage and uploader

A **new private** bucket `allenai/olmo-3p5-small` contains only the two real
trajectories under `emo/` and `non-emo/`. Smokes use the existing pilot bucket under
new isolated prefixes, never the hero bucket. All checkpoint payloads reside in
`/weka/olmo-3p5-checkpoints/production-hero-small/olmo35-small-hero-20260907/`.
Beaker results point at `/noop-results` and contain no training payloads.

The existing qualified four-GPU uploader accepts independent registrations for both
arms. Explicit `apply` deletion retains the highest two local checkpoint steps and
requires verified remote copies, a safe successor, and a one-hour grace. The trainer
never prunes checkpoints. The uploader does not delete HF copies.

The measured uploader rate is below two simultaneous 100-step streams. Both runs
check space every25 steps, warn below10TB free, and gracefully save/stop below5TB.
`STORAGE_PAUSED.json` latches this condition: no automatic relaunch or blind restart.
After investigating and allowing uploads to catch up, an operator may explicitly
archive that exact pause file and resume the same run. No shared-volume deletion
is performed by this implementation.

Each checkpoint includes sampled hashes of live BF16 parameters, FP32 optimizer
shards, persistent buffers, and skip-step histories under `resume_audit/`. These
files are written inside the temporary checkpoint before the completion marker;
they upload and restore with the full checkpoint. Saves verify that live state is
unchanged. Resumes fail closed on sampled disagreement. This is an exact sampled
check, not an assertion that every tensor element is hashed.

## Launch gates

The durable controller verifies the real mounts, completed-download manifest hash,
every listed object size and readability, uploader health, and free space. It creates
the bucket with `exist_ok=False` and records a durable receipt; a naming collision or
ambiguous creation fails rather than adopting a pre-existing bucket.

Next it runs config validation in the production image, then one64-GPU smoke
experiment: EMO0→2→4 and non-EMO0→2→4, using fresh processes for each restore.
It checks exact sampled save/restore state on all64 ranks, identical initial weights
and first data batches between arms, and the real immutable step4 checkpoints.
Only then are both ~3T jobs submitted. Durable submission intents and exact-name
reconciliation prevent duplicate submissions. Failures block; no retry storm.

## Continuation after selecting an arm

Do not create a fresh run ID, HF prefix, or optimizer. Reuse the selected run's root
and full checkpoint. Set `OLMO35_HERO_ALLOW_CONTINUATION=1`,
`OLMO35_HERO_STOP=834466`, and `OLMO35_HERO_EXPECTED_START=<latest saved step>`;
the latter forces checkpoint loading. This preserves absolute warmup, optimizer,
data position, and automatic checkpoint cadence. The final WSD decay policy still
needs to be chosen; a stable trunk is intentionally not an implicit final-decay
decision. The controller does **not** authorize or launch this continuation itself.

Entrypoints: `olmoe3_small_hero.py`, `olmoe3_small_hero_node.py`,
`olmoe3_small_hero_control.py`, all under `src/examples/olmo_ddp/`.

## Deployment receipt (2026-09-07 22:26 UTC)

- Runtime source: `ed9714c51` (the subsequent documentation commit does not change runtime).
- New private bucket created and verified empty at22:19UTC:
  https://huggingface.co/buckets/allenai/olmo-3p5-small
- Current controller: `01M1YZAJMNMQ5TVYS3PBCC0EYD`, running.
- Config validation: `01M1YZC0M6KV52Y1RMDWTBA5X9`, succeeded;
  all four production/smoke configs validated, model diff is only EMO.
- Save/restore smoke: `01M1YZFS2DFX36XFKVFS4KY425`, queued64GPUs.
  Scheduler reports insufficient free slots for the eight-node replica group.
- Both production jobs are **not yet submitted**. The live controller will submit
  them automatically only after smoke success and fingerprint/state checks.
- Superseded initial controller `01M1YZ76BZCT0DPRJMWAJXQFN6` was explicitly stopped
  before it could submit any GPU work. Its CPU-only validation passed too.
- Preflight verified2,334 objects against the completed Dolma inventory and observed
  37.47TB free. All four registrations are idempotently installed; production is in
  the new bucket, smoke is in the pilot bucket.
- Local tests: four dependency-free plan/spec tests passed, including exhaustive
  cadence checks through834466; lint/compile checks passed. An isolated mock test
  also verified rank0-only low-space cancellation and durable pause publication.
- The full64-GPU save/restore test is still pending; do not describe it as passed.

Do not rerun the one-shot launcher while the current controller is active. It owns
the durable lock and submission intents. Inspect its logs and the smoke experiment
first. Controller state is under the campaign's `uploader/automation/` directory;
`heroes-submitted.json` will contain the two production experiment IDs after gating.

## Hardware-gate failure and placement fix (2026-09-07)

The first smoke failed before any training agent or checkpoint save started. On
`holmes-cs-aus-534.reviz.ai2.in`, GPU6 had `SYS`/`NODE` connections instead of NVLink
to every peer. The existing runtime topology guard rejected it correctly; seven
other workers passed topology. Separately, the original leader on
`holmes-cs-aus-550.reviz.ai2.in` failed Beaker's interconnect ALLREDUCE healthcheck
and was automatically replaced. Neither is a model, EMO, or uploader failure.

Exclude534 and550 from this campaign's qualified hostname allowlist. No runtime
model/training settings or safety checks are changed. Both smoke and eventual hero
specs inherit the exclusions. The failed controller stopped without submitting
either hero. Its successor uses source-versioned validation/smoke names and the
same private bucket and registrations; no checkpoints are deleted or overwritten.

Replacement runtime pin: `171be9bef`. Controller `01M1Z16E1AHQ3QGR0YBXHR4S3F`
is running; real-image configuration gate `01M1Z17RJYEV0EPF49B2AQN1EM` passed.
Replacement smoke `01M1Z1AQSAGRZY6F3ZMC3S076Q` queued at22:55UTC, still64GPUs,
urgent/allocated, both faulty hosts excluded. Hero runs remain unsubmitted pending
the same save/restore and initialization/input-fingerprint gates.
