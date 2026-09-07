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
