---
experiment: weka_cleanup
title: INFRA — weka / S3 / node-local checkpoint storage cleanup (redundant stepN dirs, relay duplicates, ZFS snapshot lag)
status: active
updated: 2026-10-12
branch: prasann/landmark
launcher: debug/weka_cleanup/plan_deletions.sh
records:
  - records/weka-checkpoint-cleanup.md
results:
  - debug/storage_cleanup/DELETION_LOG.tsv
  - debug/storage_cleanup/CLEANUP_PLAN.md
  - debug/weka_cleanup/manifest_relaydup_20261012.tsv
  - debug/storage_cleanup/weka_stepn_audit.tsv
wandb_group:
next_step: Prasann to decide on long-term weka growth (~22 TB, mostly fp32 model-only finals) — S3 offload (blocked on AI2 S3 policy check) vs bf16 re-save (declined for now); optimizer-strip of the old top-level keepfinal stepN finals still open.
---

# weka_cleanup

## Question
Not an experiment: keep `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/` (and S3 relay copies and
Berkeley node-local `/data`) from filling, by deleting only provably redundant checkpoint copies with
auditable, two-stage (plan → literal manifest → validated apply) tooling.

## Decisions
- 2026-07-28 (Prasann): deletes name exact paths, never globs; two-stage plan/apply; manifests archived to `s3://ai2-llm/checkpoints/prasanns/_inventory/manifest_<phase>[_TAG].txt`.
- 2026-07-28 (agent): Phase 1 (all stepN in 74 `ctc_suite/ckpts` runs that have `model_and_optim/`) 11.22 TB + Phase 2 (keepfinal on 49 top-level runs) 3.04 TB → 21.37 → 7.2 TB. `export_olmo_to_hf.py:latest_step_dir()` falls back to `model_and_optim`.
- 2026-07-28 (agent): optimizer-strip of the 97 surviving top-level finals (~3.1 TB) NOT done — needs a GPU load-and-resave per run, irreversible, and outlier_datascale_sweep evals were owed.
- 2026-08-29 (agent): root cause of regrowth to 23 TB = `beaker_ctc_suite.py` never passed `--no-final-checkpoint` (weights stored twice); fixed as default-suppress with `--keep-final-checkpoint` opt-out (present in the launcher now).
- 2026-09-01 (agent): node-local deletes on mooney/horton/lorax/thidwick; ZFS weekly snapshots defer reclaim up to ~4 weeks — never promise immediate `df` gains; destroying snapshots on shared nodes is an infra decision (sysadmin request drafted for cubbins 2.63 TB USEDSNAP).
- 2026-09-08 (agent): S3 `ckpt_relay` cleanup (90 entries) + weka round logged in DELETION_LOG.tsv.
- 2026-10-12 (agent, via storage-audit skill): round 3 — weka had regrown to 28.8 TB; 96 stepN deleted (`outlier_lengthmix/ckpt_relay` 85 / ~4.5 TB, top-level keepfinal 10 / 0.40 TB incl. setA 32k/256k, `ckpt_slm_p8k4000` 1 / 0.05 TB) + relay-dedupe of 85 `ckpt_relay` runs that duplicated `ctc_suite/ckpts/lmx-*` (1.50 TiB); sneetches /data 58 stepN 2.20 TB + 6 approved dirs 1.20 TB. Weka ends ~22.3 TB.
- 2026-10-12 (Prasann): bf16 re-save of finals (would ~halve weka) declined for now; S3 offload tooling NOT built while Prasann checks AI2 S3 policy.

## Current best numbers
Storage state (no eval metrics):

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| weka prasanns | total | 21.37 TB (07-28) → 7.2 TB → 23 TB (08-29) → 28.8 TB → ~22.3 TB (10-12) | — | — | — | memory:weka-ckpt-cleanup, memory:beaker-launcher-missing-nofinalckpt |
| weka bucket | free | 3.8 T → 9.8 T after 10-12 round | — | — | — | memory:weka-ckpt-cleanup |
| DELETION_LOG 2026-10-12 | rows | weka:phase3 85, weka:relay-dedupe 85, weka:phase2 10, weka:phase4 1, sneetches:/data 64 | — | — | — | debug/storage_cleanup/DELETION_LOG.tsv |
| node /data USEDSNAP (09-01/09-08) | deferred reclaim | cubbins 2.63 TB (0 avail), mooney 5.81 TB, horton 9.36 TB | — | — | — | debug/storage_cleanup/sysadmin_request.md |
| horton → S3 | throughput | 73 MiB/s up / 81 MiB/s down (aws-cli v1) | — | — | — | memory:weka-ckpt-cleanup (s3_speedtest.sh) |

## Next step
Blocked on Prasann: (1) S3 offload of cold finals vs keep on weka (AI2 S3 policy); (2) bf16 re-save (declined for
now); (3) optimizer-strip of the old top-level keepfinal finals (~3.1 TB as of 07-28; re-measure first — some were
deleted since). Routine: run the `storage-audit` skill (`--refresh`) before large writes; repeat plan/apply rounds as needed.

## Open issues / traps
- Naming collision: memory index says "Phase 3 (optim-strip, ~3.1TB) still open", but the 2026-10-12 manifests also use `phase3` for the `outlier_lengthmix/ckpt_relay` stepN round. They are different things — the optim-strip has never run.
- Do NOT delete: `ctc_suite/bases/*`, `ctc_olmo3/bases/*-fixmark`, `xlong5_2k256k_qwen35/*` (weka is the only copy), `contra_mix_*_10k_128k` shards, `_eval_bundle*`, `ctc-olmo3-7b-hpqaret-*` (retrain owed; the 5 hpqaret runs keep stepN on purpose).
- Freshness-guard trap: deleting a stepN bumps its PARENT mtime; the applier pre-computes parent verdicts once. Resuming an interrupted apply needs a lowered `FRESH_MIN`.
- Plan-tool bug (fixed 10-12): a non-numeric `step*` dir put a bare RUN DIR in the manifest; all planners now use `-regex '.*/step[0-9]+'`.
- Weka is not mounted at Berkeley: plan/apply run as 0-GPU gantry jobs with the worker shipped inline (no pushed commit needed) — memory:gantry-needs-pushed-commit.
- ZFS: node-local deletes free nothing for up to ~4 weeks — memory:zfs-snapshots-delay-node-local-reclaim. `zfs destroy -nv` dry-runs fine; real destroys need sysadmins.
- Possible regrowth sources still writing stepN: the setA recipe (its 32k/256k runs had stepN deleted on 10-12) and any `--keep-final-checkpoint` launch.
- `/accounts` 20 GB quota is a separate problem — memory:accounts-quota-avoid-writes.

## Verified vs unverified
- Verified 2026-10-12: `beaker_ctc_suite.py` now passes `--no-final-checkpoint` unless `--keep-final-checkpoint` (lines ~191/281); DELETION_LOG.tsv row counts per pool/date (09-01: 330, 09-08: 369, 10-12: 245); the 10-12 tooling files (dedupe_relay.py, launch_dedupe_relay.py, dir_cleanup.sh, manifest_relaydup_20261012.tsv); sysadmin_request.md USEDSNAP figures; record's "Not done: optimizer strip" text.
- Read only in memory: all TB totals and bucket-free figures (DELETION_LOG `bytes` is blank on many rows, so it under-counts and cannot confirm them), sneetches 2.20 + 1.20 TB, S3 throughput.
- UNVERIFIED — confirm with Prasann: whether the cubbins/mooney/horton snapshot request was ever sent/acted on; whether the optimizer-strip is still wanted (and its current size); whether the setA launcher should also default to `--no-final-checkpoint`.
