---
name: storage-audit
description: >
  Use this skill in ANY repo whenever storage comes up: "how much am I using", "what's taking
  space", "is /data / weka / my quota full", "what can I clean up / delete", disk-full or EDQUOT
  errors, before writing large outputs (checkpoints, shards, eval dumps) to pick a location with
  room, and before any cleanup. Reports usage on weka, node-local /data on every Berkeley node,
  /accounts (20G quota) and /scratch, plus ranked cleanup candidates (large dirs untouched for
  N days, redundant stepN checkpoints) — in about a second, from pre-built indexes.
---

# Storage audit

```bash
python3 ~/.claude/skills/storage-audit/storage_report.py                 # ~1 s
python3 ~/.claude/skills/storage-audit/storage_report.py --refresh       # rebuild all indexes (bg)
python3 ~/.claude/skills/storage-audit/storage_report.py --refresh weka  # or: nodes | home
python3 ~/.claude/skills/storage-audit/storage_report.py --under /data/prasann/ctc_suite --nodes sneetches
python3 ~/.claude/skills/storage-audit/storage_report.py --idle-days 60 --min-gb 50 --json
python3 ~/.claude/skills/storage-audit/storage_report.py --html          # + HTML view
```

## Why it is split in two (don't "fix" this by walking live)

Walking these trees live is slow: a `du` over `/net/<node>` once sat for over an hour in NFS
D-state, weka isn't mounted at Berkeley at all, and even `du ~` takes >20 s. So:

- **`storage_index.py`** walks once *where the disk is local* and writes a compact JSON index:
  - node `/data/prasann` → 1-CPU slurm job per node (`--qos=preemptive`, niced; local walk of
    ~500k files takes ~10 s) → `/data/prasann/.storage_index.json`, read via `/net`.
  - weka (`ai2-llm/checkpoints/prasanns`, `oe-training-default/prasanns`, `prasanntestsingle`)
    → 0-GPU urgent Beaker job from ai2/flex2 (spec-only, no gantry/git, works from any repo); the
    index is the job's result dataset, fetched into `~/.cache/storage-audit/weka.json` on the
    next report run after it finishes (~5–30 min incl. scheduling).
  - `/accounts` home + `/scratch` → niced background walk on the login node → `~/.cache/storage-audit/`.
- **`storage_report.py`** never walks. Live: `quota` and `df` (3 s timeouts, so a hung node is
  skipped, not waited on). Everything else comes from the indexes, and each row shows the index age
  plus any running/pending index job. Indexes older than 7 days are flagged.

Refresh when the index is stale or before acting on a candidate. Re-running the report after a
`--refresh` picks up results as they land; you don't need to wait for all of them.

## Reading it

- **Locations table** — your usage (`mine`), filesystem fullness bar + free space, index age,
  candidate count/size, and notes (ZFS snapshot bytes, index job state). Warnings at the bottom:
  quota > 85 %, node `/data` > 95 %, missing/stale indexes.
- **CLEANUP CANDIDATES** — the *largest* directories whose newest mtime anywhere below is older
  than `--idle-days` (default 30) and size ≥ `--min-gb` (default 10). Listed maximally: if a
  parent qualifies, its children aren't repeated — use `--under <path>` to drill in.
- **REDUNDANT CHECKPOINTS** — `stepN` dirs (any age) in a run that also has a model-only
  `model_and_optim/.metadata`, or that aren't the run's highest step. Eval never reads these.
- **PROTECTED** tags mark things never to delete casually: bases, marker-fixed bases, `xlong5`
  (weka is the only copy), shards, eval bundles, envs/venvs, vLLM serving copies, `.claude`.

## Caveats

- "Idle" = mtime. Reading a checkpoint (eval) does not update mtime, and atime is unreliable on
  these filesystems, so an old-mtime checkpoint may still be in active *read* use. Check the
  run's eval status / ledger before treating it as dead.
- Node `/data` is ZFS with ~4 weekly snapshots: a delete drops `du` now but `df` only frees space
  when the snapshots roll (≤ 4 weeks). Don't promise immediate space from a node-local delete.
- The same run often exists on several nodes (copies made for eval); that duplication shows up as
  the same path under different `where` values.
- Nodes whose index job is pending (CPU-full node, `ReqNodeNotAvail`) simply show `?` until it
  runs; mcfuzz is often CPU-saturated.

## This skill never deletes

It only names candidates. To actually delete, follow the user's two-stage process: build an
explicit manifest of absolute paths, have Prasann approve it, then apply with re-validation. The
OLMo-core tooling for that lives in `debug/storage_cleanup/` (`nodelocal_cleanup.sh`,
`s3_cleanup.sh`, the weka plan/apply flow in the `weka-ckpt-cleanup` memory). Never `rm` straight
from this report, never glob.

Source of truth: OLMo-core `.claude/skills/storage-audit/` (`~/.claude/skills/storage-audit` and
`~/.claude-work/skills/storage-audit` are symlinks to it). User-specific constants (roots, nodes,
Beaker workspace/budget, protected patterns) are at the top of `storage_report.py`.
