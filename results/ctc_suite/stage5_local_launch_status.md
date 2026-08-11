# CTC-suite Stage-5 local launch status (jsteinhardt H200, since lambda is quota-blocked)

Owner: prasann + Claude. Started 2026-07-19/20 local time (job clock reads ~2026-07-20 early AM).

## Goal
12 joint training runs = 6 tasks x 2 arms (full, chunked-mix), Qwen3.5 0.8B, 1 epoch,
seq_len 40960, on the jsteinhardt partition (H200 nodes cubbins/mcfuzz/mooney/sneetches).
Launcher: `src/scripts/train/memexpress/ctc_suite/run_ctc_local.sbatch` ->
`train_ctc_suite.py`. Replaces the lambda runs cancelled for the same 6 tasks (lambda NFS
quota exhausted).

## Data audit (Berkeley /scratch/users/prasann/ctc_suite_staged/shards/<task>_train)

All 6 shards present, `marker_set=qwen3_5`, instance counts match expectation, all
`max_example_len` <= 40960 (fits seq_len with no truncation):

| task | num_instances | expected | max_example_len | status |
|---|---|---|---|---|
| hotpotqa | 20000 | 20000 | 25742 | OK |
| grouping | 20000 | 20000 | 36785 | OK |
| oolong | 21000 | 21000 | 39971 | OK |
| outlier | 19986 | 19986 | 40558 | OK |
| reorder | 19944 | 19944 | 40957 | OK |
| rerank | 20000 | 20000 | 31135 | OK |

No task skipped.

## Launcher change

`run_ctc_local.sbatch` had no activation-checkpointing / shard-degree passthrough (unlike
`run_ctc_lambda.sbatch`). Added backward-compatible `ACT_CKPT` / `SHARD_DEGREE` env knobs ->
`${ACT_CKPT:+--activation-checkpointing "$ACT_CKPT"}` / `${SHARD_DEGREE:+--shard-degree
"$SHARD_DEGREE"}`, default empty (no behavior change for existing callers).

## Trap hit + fixed: WANDB_API_KEY

First validation submission (job 3338930's predecessor, 3338922) failed in ~3 min:
`OLMoEnvironmentError: missing env var 'WANDB_API_KEY'` (the callback requires the env var
explicitly; it does not fall back to `~/.netrc` even though `wandb` itself would). Fixed by
extracting the key from `~/.netrc` (`machine api.wandb.ai`) and passing
`WANDB_API_KEY=...` through `--export=ALL` on every sbatch submission. Note: **shell state
does not persist between tool calls** in this environment, so the key must be re-exported in
every submission command (sourced from a small scratchpad helper).

## Fit validation

Pending / in progress -- see table below for the validated NGPU/ACT_CKPT config once confirmed.

## Launch table

| task | arm | jobid | run-name | host | status |
|---|---|---|---|---|---|
| hotpotqa | full | 3338922 | ctc-s5-hotpotqa-full-08b-loc | mooney | FAILED (missing WANDB_API_KEY, 3 min) |
| hotpotqa | full | 3338930 | ctc-s5-hotpotqa-full-08b-loc | mooney | validating |

(table updated as runs launch/complete/fail)
