# debug/weka_cleanup/ — reclaim redundant training checkpoints on weka

Tooling behind `records/weka-checkpoint-cleanup.md` (21.37 TB → 7.2 TB, 2026-07-28). Read that
record first: it explains *why* most `stepN/` dirs are redundant and which paths must never be
touched.

Weka is not mounted at Berkeley, so everything runs as a 0-GPU gantry job on jupiter.

## Two stages, on purpose

Deletion is split so that **no pattern expansion ever happens in the delete step**:

| Script | Role |
|---|---|
| `plan_deletions.sh` | **Discovery only** — contains no `rm` at all. Walks the run root and writes a manifest: one exact absolute path per line. |
| `apply_deletions.sh` | **Execution** — no globbing, no `find`, no wildcards in the delete path. Reads literal lines and re-derives every safety property from the filesystem before removing each one. |
| `run_cleanup_gantry.sh` | Driver. Ships both workers inline as base64, so it does **not** depend on a pushed commit (gantry otherwise clones the last pushed HEAD and would run a stale worker). |

The manifest is archived to `s3://ai2-llm/checkpoints/prasanns/_inventory/manifest_<phase>.txt`, so
what was removed stays auditable.

## Usage

```bash
bash run_cleanup_gantry.sh plan  phase1   # discovery -> manifest on S3. Deletes nothing.
# ...pull the manifest and read it before continuing...
bash run_cleanup_gantry.sh apply phase1   # deletes ONLY the exact paths in that manifest
```

- `phase1` → `ctc_suite/ckpts`, `MODE=modelonly`: drops every `stepN/`, keeping
  `<run>/model_and_optim/` (the model-only final save the eval path loads).
- `phase2` → top-level per-run dirs, `MODE=keepfinal`: these have no model-only save, so the
  highest `stepN/` is kept and only earlier ones are removed.

`FRESH_MIN=<min>` overrides the 90-minute in-flight window. Lower it **only** to resume an
interrupted apply — see the trap below.

## Safety properties enforced per path (all must pass, or the path is refused)

1. absolute path under `.../checkpoints/prasanns/` — namespace containment
2. basename matches `^step[0-9]+$` — only training-step dirs, ever
3. is a real directory and not a symlink; no `..` anywhere in the path
4. parent run name does not match `KEEP_REGEX` (bases, shards, eval bundles, tokenizer, hpqaret)
5. parent run untouched for `FRESH_MIN` minutes — nothing in flight
6. mode-dependent **"the weights survive"** invariant:
   - `modelonly` — parent has a non-empty `model_and_optim/.metadata`
   - `keepfinal` — a strictly higher-numbered `stepN` survives on disk **and is not itself in the
     manifest**. If the survivor *is* listed, the whole run is refused, not just that path.

Both modes were exercised against adversarial fixtures (path outside the namespace, a symlink to
`/etc`, a `model_and_optim` dir, a protected run, a run with no final model, a manifest naming a
run's own final checkpoint, and the `step9`-vs-`step10` numeric-sort trap). All were refused.

## Trap: the freshness guard can trip on its own writes

Deleting a `stepN` updates its **parent's** mtime. When the in-flight check was evaluated inline
per path, the first delete in a run made every remaining path in that run look "modified <90m ago"
— the first Phase 1 apply deleted 74 of 210 paths and refused 136. `apply_deletions.sh` now runs a
freshness **pre-pass** that resolves each parent's verdict once, before any deletion.

Consequence when resuming an interrupted apply: the tooling itself has just touched those dirs, so
a fresh 90-minute window will refuse everything. Lower `FRESH_MIN` — but only once the plan stage
has already certified those runs idle over the full window.
