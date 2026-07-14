# Using Beaker (AI2 cluster) from Berkeley

The Beaker counterpart to `local_cluster.md`. Everything here is validated from this machine
(`beaker` CLI at `~/.local/bin/beaker`, authed as **prasanns**; `gantry` lives in the
`corpus-reasoning-olmo` env — prepend its bin dir or use `source src/scripts/local_env.sh`).

**Constants (use these unless you have a reason not to):** workspace `ai2/flex2`, budget
`ai2/oe-other`, weka bucket `oe-training-default`, baked image
`tylerr/olmo-core-tch291cu128-2025-11-25` (== `OLMoCoreBeakerImage.stable`), secrets prefixed
`PRASANNS_` (wandb, AWS, beaker token, claude oauth).

## The four golden rules

1. **Commit AND push before launching.** Both the internal `launch` command and gantry clone the
   repo at your *pushed* HEAD — a dirty tree errors, and an unpushed commit either fails
   (`RemoteBranchNotFoundError`, sometimes silently after "Launched") or ships stale code.
2. **Priority is ALWAYS `urgent`** (or `immediate`) — never normal/high; lower pends for hours
   behind capacity. Bump a queued job in place: `beaker job update-priority <JOB-id> urgent`.
3. **GPU jobs → `ai2/jupiter`** (H100s; gantry `--cluster` may need the full
   `ai2/jupiter-cirrascale-2` form). Never neptune for training — L40S, ~5x slower (MFU 4.5%).
4. **CPU jobs → multi-cluster, eager-first:** `--cluster ai2/neptune,ai2/ceres,ai2/saturn,ai2/jupiter`.
   Jupiter alone is strict-priority backfill and can queue a 0-GPU job 10+ min; multi-cluster
   schedules in seconds. (`ai2/holmes` is reserved for other budgets — unusable.)

## Launching a training run

Training scripts (the `memexpress/` Beaker families) use the internal `launch` pattern:

```bash
cd $REPO && git push origin prasann/landmark            # rule 1
python -u src/scripts/train/memexpress/sft_5task/Qwen3-4B-dense-5task-32k-nocpt-SFT.py \
    launch <run-name> ai2/jupiter-cirrascale-2 --launch.priority=urgent
```

- `python -u` matters: `launch` submits then **streams logs until the run finishes** (the process
  staying alive is normal, not a hang), and without `-u` the
  `Experiment: https://beaker.org/ex/<ID>` line sits in the block buffer and never appears in a
  piped/captured shell. Grab the ID, then it's safe to kill the follower — the job runs
  independently. (`--launch.follow=false` does NOT work: soft-timeout requires follow.)
- Dry-run first when touching config: `SCRIPT.py dry_run <run-name> ai2/jupiter-cirrascale-2`.
- Surface the **wandb group link** as soon as the run starts.
- The auto-resume trap applies on Beaker too: relaunching a run name whose `--save-folder` already
  has a `stepN` checkpoint silently RESUMES it. Fresh experiment ⇒ fresh run name.

## One-off jobs with gantry (data staging, transfers, evals)

Template (this exact shape is validated — it's the S3→weka sync from the eval bundle staging):

```bash
gantry run --name <job-name> -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/neptune,ai2/ceres,ai2/saturn,ai2/jupiter --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c '<the work>'
```

- **The baked image is the speed trick**: deps (transformers/numpy/etc.) are pre-installed and the
  image is cached on AI2 nodes → near-zero startup. With `--beaker-image` pass `--install true`
  if the job needs `olmo_core` importable (the literal `true` no-op is REQUIRED — omitting
  `--install` triggers a slow `pip install -e .`); use `--no-python` when it doesn't.
- `--timeout 0` returns immediately — never let gantry block the shell (killing the shell cancels
  the job).
- AWS creds on-node come from the beaker secrets as above (`AWS_PROFILE=S3` = real AWS).

## Monitoring, canceling, priorities

```bash
beaker experiment get <EXP-id> --format json   # jobs[-1].status is the ONLY truth:
   # canceled→CANCELED · finalized→DONE (check exitCode) · started→RUNNING · scheduled→SCHEDULED
beaker experiment logs <EXP-id>
beaker job cancel <JOB-id>                     # JOB-id = jobs[-1].id from the JSON
beaker job update-priority <JOB-id> urgent
```

- The `beaker experiment get` **table view lies** (shows "running" for canceled jobs) and
  `beaker experiment stop` **silently no-ops** (exit 0, job keeps running). JSON + `job cancel`
  only, then re-verify `jobs[-1].status`.
- This CLI (v1.5.317) has no `beaker experiment list` — track experiment IDs at launch time.
- The shell here is **zsh: unquoted `$VAR` does not word-split**, so `for id in $IDS` loops over
  one newline-joined blob and every cancel/status loop silently breaks. Wrap list loops in
  `bash -c '...'`.
- `exitCode=0` ≠ success for evals — **grep the logs for `MISSING`** (see next section).

## Data: weka vs S3 (weka is NOT mounted at Berkeley)

- Jobs read/write **weka** (`/weka/oe-training-default/ai2-llm/checkpoints/prasanns/...`); this
  machine can only reach **S3** (`s3://ai2-llm/checkpoints/prasanns/...`, real AWS, the `S3`
  profile; the weka S3-gateway is firewalled from here).
- **Staging new data for on-beaker jobs is a TWO-step**: (1) `aws s3 sync <local> s3://...` from
  here, (2) a gantry job (template above) to `aws s3 sync s3://... /weka/...` on a weka node.
  S3-push alone = every input logs `MISSING`, the job skips everything and **exits 0** looking
  successful.
- Checkpoints weka→here: gantry job does a model-only resave (drop optimizer) + `aws s3 sync` to
  S3, then pull from S3 locally. See the `weka-s3-checkpoint-transfer` pattern
  (`deprecated/scratch_transfer_cptmix32k.sh` is a worked example to adapt).

## Interactive + agent sessions

- `scripts/interactive_gpu_session.sh` — live dev shell on an AI2 GPU with weka (GPUS=N, LOCAL=1).
- `scripts/launch_remote_control_agent.sh` — Claude Code agent on a Beaker GPU, steerable from
  your phone; OAuth token via masked beaker secret.
- `scripts/setup_dev_env.sh` — bootstrap the toolchain + sibling repos on a new machine.

## Lessons index (each burned real time)

| Lesson | Symptom if ignored |
|---|---|
| Push before launch | `RemoteBranchNotFoundError` (sometimes silent after "Launched"), or job runs stale code |
| `python -u` + kill follower | Launch "hangs"; experiment URL never appears in captured output |
| urgent priority, always | 1h+ pending; use `job update-priority` to rescue a queued job |
| Multi-cluster for CPU | 0-GPU job unscheduled 10+ min on jupiter (1007/1008 slots) |
| Baked image + `--install true` / `--no-python` | ~3 min image pull + pip per job, dominating short jobs |
| JSON status, `job cancel` | Canceled jobs look "running"; `experiment stop` no-ops; wrong-base jobs kept running |
| S3→weka gantry sync + grep `MISSING` | Eval exits 0 with zero results |
| Fresh run name per config | Silent auto-resume from a stale checkpoint (chimera runs, garbage timings) |
| zsh loops via `bash -c` | Cancel/status loops silently operate on one blob argument |
