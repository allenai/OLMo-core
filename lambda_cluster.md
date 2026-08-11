# Lambda cluster

Third compute pool, alongside the Berkeley slurm H200s (`local_cluster.md`) and AI2 Beaker
(`beaker.md`). Lambda is a **separate SLURM cluster of A100 nodes reached over ssh** — it shares no
filesystem, no git remote, and no network with the Berkeley cluster. Read this before launching
anything here or debugging a Lambda run.

**Use it for**: long fan-outs of independent training runs when the Berkeley per-user GPU quota is
saturated. Lambda's GPUs are frequently idle while jsteinhardt is contended, so moving a queue here
is usually the fastest way to unblock a stalled sweep.

**Don't use it for**: anything needing internet (HF/W&B/GitHub downloads), anything needing a fresh
git checkout, or a one-off run that would cost more in staging than in compute.

---

## Access and hardware

```bash
ssh lambda          # -> lambda-headnode02
```

| | |
|---|---|
| Partition | `lambda` (the only one) |
| Account | `site` |
| Nodes | `lambda-hyperplane01/02/03/05` up, **`04` is DOWN+MAINTENANCE** (long-standing) |
| Per node | 8 × A100-SXM4-**80GB**, 256 CPUs, ~1TB RAM |
| Usable pool | **32 GPUs** across 4 live nodes |

Check what's actually free before planning a wave — a single other user can hold whole nodes for
days:

```bash
ssh lambda 'sinfo -N -o "%.22N %.6t %.10G %.30C"; squeue -o "%.8i %.12u %.20j %.10q %.8T %.12M %N"'
```

`AllocTRES=...,gres/gpu=8` in `scontrol show node` means that node is fully committed even if its
CPUs look idle — GPU allocation is what matters, and `sinfo`'s `mix` state does not distinguish
"6 GPUs free" from "0 GPUs free".

### QOS, and the two-node fair-share rule

Same three-tier hierarchy as jsteinhardt:

| QOS | Preempts | Scheduler cap | **Our policy cap** |
|---|---|---|---|
| `normal` | — | none | **none — use freely** |
| `preemptive` | `normal` | none | **8 GPUs (1 node)** |
| `preemptive_high` | `normal`, `preemptive` | `gres/gpu=8` | 8 GPUs (1 node) |

**RULE: never hold more than 8 GPUs on `preemptive_high` and 8 on `preemptive` at the same time —
i.e. cap your *preempting* footprint at two nodes. Take any capacity beyond that at `normal`.**

The cluster has 4 live nodes shared with one other regular user (`rhys_gould`). Capping each user's
preempting footprint at 2 nodes is what guarantees **both users can always get 2 nodes**. `normal`
is exempt from the cap precisely because it cannot preempt anything and is itself preemptible — a
`normal` job soaks up genuinely idle GPUs and gets out of the way the moment the other user submits,
so it costs nobody anything.

Practical consequence: `preemptive_high` is already capped at 8 by the scheduler, so **escalating
QOS to preempt someone buys you nothing beyond one node's worth.** Take idle capacity instead of
preempting. This is the opposite of the Berkeley cluster, where the standing directive is to default
to the highest QOS and preempt freely — do **not** carry that habit over here.

Scaling past 2 nodes, in order: fill `preemptive_high` (8) → `preemptive` (8) → everything else at
`normal`, and expect the `normal` jobs to be killed at any time (checkpoint accordingly, and use a
fresh run name on relaunch).

---

## Storage — the usual blocker

**The quota is per-USER across all of `/accounts`, not per-project or per-filesystem.** The
filesystem has terabytes free; that is irrelevant. Hitting the cap fails every write with
`Disk quota exceeded (122)`, including partial checkpoint writes that then look like training bugs.

```bash
ssh lambda 'quota -s'
#   FILESYSTEM  USE     QUOTA   %
#     accounts  1.10 T  1.30 T  84.5
```

⚠ **Quota accounting lags deletes by ~a minute.** Immediately after freeing 209G it still reported
`1.30T 100.0%`; the correct `1.10T 84.5%` appeared shortly after. Don't conclude a cleanup failed —
re-check, or just try a `dd` write test.

Budget before a fan-out: **a 4B distcp checkpoint is 17-52G**, so ~200G of headroom is about one
wave of four runs. Plan to eval and delete each wave before launching the next.

### Node-local disk is the right default for checkpoints

Compute nodes have large **node-local NVMe that is not quota'd** (verified on
`lambda-hyperplane03`):

| Mount | Size | Free | Writable by me? |
|---|---|---|---|
| `/data` | 14T | 4.3T | ❌ **no** — see below |
| `/tmp` | 2.3T | 2.2T | ✅ yes — use this |
| `/dev/shm` | 504G | 504G | ✅ tmpfs |

⚠ **`/data` is NOT usable on Lambda, unlike on the Berkeley cluster.** It is `root:root` with an ACL
granting only `group:sudo`, and each user's `/data/<user>` is created by an admin. There is no
`/data/prasann`, so `mkdir` fails with `Permission denied` and the run dies at startup. Don't copy
the `SAVE_ROOT=/data/...` pattern over from `local_cluster.md`.

Use `/tmp` instead. **Verified: `/tmp` is shared, not per-job-private** — a file written by one job
is readable by a later job on the same node, so a node-pinned eval can read a training run's
checkpoint. (The doubled `/tmp` line in `mount` output looks like a `job_container/tmpfs` private
mount but is not one; test rather than assume.)

```bash
SAVE_ROOT=/tmp/prasann/ctc_suite/ckpts WORK_ROOT=/tmp/prasann/ctc_suite/work
```

This keeps NFS usage flat — only shards, bases, and result JSONs live on the quota'd filesystem.
The cost is the familiar one: **node-local checkpoints die with preemption, and eval must be
node-pinned (`-w`) to the training host.** On Lambda that cost is small, because the 8-GPU
`preemptive_high` cap means you are working within a single node anyway, so you were going to pin
the job regardless. Only `preemptive_high` can preempt `preemptive_high`, so the preemption risk is
low.

Data is read-only from NFS and does not grow your quota; only checkpoints and caches do.

### Where the space actually goes

A representative breakdown of a 1.10T footprint, for deciding what to reap:

| Path | Size |
|---|---|
| `~/.cache` (HF datasets — **holds the tokenizer eval needs**) | 484G |
| `projects/corpus-reasoning` | 333G |
| `~/.conda` (`conda clean --all` is the safe easy win) | 151G |
| `projects/prasannfirstphd` | 70G |
| `ctc_suite` (bases 23G + shards + ckpts) | 47G |

Reap your own `ctc_suite/ckpts/` and `cache-<jobid>/` first — that is usually enough, and it costs
nothing you can't regenerate.

Find what's using space (note `*/` misses dotfiles — the HF cache is a dotdir and is often the
largest single item):

```bash
ssh lambda 'du -sh /accounts/projects/sewonm/prasann/*/ /accounts/projects/sewonm/prasann/.[!.]* | sort -h'
```

### LROOT layout

```
LROOT=/accounts/projects/sewonm/prasann/ctc_suite
├── OLMo-core/src/     # rsync'd source (NOT a git clone — no network)
├── bases/             # converted distcp bases: q35-08b-base-modelonly, q35-4b-base-modelonly
├── data/<DATA_SRC>/   # tokenized shards; DATA_SRC is the dir NAME under here
├── eval_rungs/<task>/ # eval ladders (rung_*.jsonl)
├── ckpts/<RUN>/       # training output
├── cache-<jobid>/     # per-job work dir; safe to reap after a run finishes
└── logs/              # ctc_suite_<name>_<jobid>.log
```

The env is a **conda-pack'd relocatable env** at `/accounts/projects/sewonm/prasann/olmo_test/env`
(torch 2.9.1+cu128, flash_attn 2.8.3, `fla` present, olmo_core via `PYTHONPATH`). The flash-attn
2.8.3 varlen SIGSEGV that forced a 2.8.2 pin on H200 is **sm90-only — A100 (sm80) is unaffected**.

---

## Staging (there is no shared filesystem and no network)

Lambda's `/accounts` is a *different* filesystem that merely uses the same path convention — the
Berkeley repo is **not** visible from Lambda. Everything must be rsync'd in over ssh.

```bash
L=/accounts/projects/sewonm/prasann/ctc_suite

# code — after ANY local edit, or the run trains stale logic
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
  src/ lambda:$L/OLMo-core/src/

# tokenized shards (the dir name becomes DATA_SRC)
rsync -a debug/<...>/shards_<x>/ lambda:$L/data/<name>/

# eval ladders
rsync -a debug/<...>/eval_rungs/ lambda:$L/eval_rungs/<task>/
```

There is **zero internet** on the head node and every compute node — `curl huggingface.co` fails
with `Could not resolve host`. So `WANDB_MODE=offline` and `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`
are load-bearing, not hygiene, and no `git clone` / `pip install` from a remote will ever work.

A shard is only valid if `metadata.json` is present; the launcher hard-fails on a missing one rather
than silently training on nothing.

---

## Launching

`src/scripts/train/memexpress/ctc_suite/run_ctc_lambda.sbatch` mirrors `run_ctc_local.sbatch`'s knobs.

⚠ **The `#SBATCH` header hardcodes `--gres=gpu:A100:8`, and the self-submit path does NOT forward a
`--gres` override.** Executing the script directly therefore grabs **all 8 GPUs = your entire
`preemptive_high` quota**, serializing everything behind it. `NGPU` only sets `torchrun --nproc_per_node`;
it does not change the allocation. This is the same trap as `run_ctc_local.sbatch`.

**Always submit via explicit `sbatch` with the gres you want** (setting `SLURM_JOB_ID` implicitly by
going through `sbatch` skips the self-submit and honours your flags):

```bash
ssh lambda 'cd /accounts/projects/sewonm/prasann/ctc_suite/OLMo-core/src/scripts/train/memexpress/ctc_suite && \
  sbatch --partition=lambda --account=site --qos=preemptive_high \
         --gres=gpu:A100:2 --nodes=1 --time=06:00:00 --job-name=<RUN> \
         --export=ALL,TASK=<task>,DATA_SRC=<dir>,VARIANT=full,SCALE=4b,RUN=<RUN>,\
NGPU=2,SEQ_LEN=<n>,EPOCHS=1,BASE_SRC=$LROOT/bases/q35-4b-base-modelonly \
         run_ctc_lambda.sbatch'
```

Key knobs: `TASK DATA_SRC VARIANT(full|chunked|chunked-mix) SCALE(0.8b|4b) RUN EPOCHS SEQ_LEN LR
GLOBAL_BATCH MICRO_BATCH BASE_SRC NGPU SEED MAX_STEPS ACT_CKPT SHARD_DEGREE ATTN_BACKEND`
plus `SAVE_ROOT`/`WORK_ROOT` (see above).

Model family is auto-detected from the shard's `metadata.json:marker_set`; a wrong-family shard
produces plausible numbers rather than a crash, so let auto-detection do it.

Measured 4B throughput: **MFU ~36%, ~3,300-3,900 tokens/s/device** at `SEQ_LEN` 10240-19968 on 8 A100s.

---

## Eval on Lambda

Historically Lambda was a **train-only** pool and results were harvested as loss curves
(`src/scripts/eval/ctc_suite/harvest_lambda_pilot.py`). The reason was a latent bug, now fixed:

> `run_ctc_lambda.sbatch` exported `HF_HOME=$LROOT/hf-cache`, **a path that never existed**. Training
> never noticed because shards are pre-tokenized and no tokenizer is ever loaded. Eval *does* load a
> tokenizer, and under `HF_HUB_OFFLINE=1` a cache miss is a hard failure with no fallback download.

`HF_HOME` now points at the populated cache,
`/accounts/projects/sewonm/prasann/.cache/huggingface`, which holds the `Qwen/Qwen3.5-0.8B-Base`
tokenizer shared by the whole `qwen3_5` family (override with `HF_HOME_OVERRIDE`). Because
checkpoints land on LROOT **NFS**, eval does not need node-pinning here — a real advantage over the
Berkeley node-local `/data` setup.

---

## Traps index

- **`sbatch` exit code lies.** SLURM routinely reports `FAILED`/exit 1 on a *fully successful* run —
  teardown races after the async checkpoint write, no traceback. Verify via the loss curve, a
  `.metadata` file in the checkpoint, and the shard count. Also seen: `--task-epilog failed status=9`
  and `Container ... has N processes, giving up after 63 sec` on healthy runs.
- **`set -u` kills the conda-pack activate** (`CONDA_PREFIX: unbound variable`). Use `set -o pipefail`
  only. `run_ctc_lambda.sbatch` deliberately omits `-u`, unlike its local counterpart.
- **`SEQ_LEN` must be ≥ the shard's `max_example_len`**, which varies a lot per task (xabsence 4451,
  obliq 8192, cycle 19884, groups4 24960, grouping 36598). Too small and `PadToLength` silently drops
  long examples. Too large wastes A100 time — size it snugly per task, and give long-sequence tasks
  more GPUs.
- **Always relaunch under a FRESH run name** (`-r2`/`-r3`). Reusing `RUN` reuses the save folder, and
  the trainer *silently auto-resumes* stale dataloader state; if `SEQ_LEN` changed you get
  `RuntimeError: Restoring data loader state from different dataset source`. See the
  trainer-auto-resume trap in `local_cluster.md`.
- **ssh drops mid-session** (`Connection closed by 199.255.18.176`) — just retry. Long `du`/`find`
  over NFS can also exceed a short client timeout; run them with a generous timeout or in background.
- **Destructive deletes over ssh are blocked** by the agent classifier — a human must run `rm -rf`
  against Lambda paths (in Claude Code, via the `!` prefix).
- Don't delete other users' data to make room. Reap your own `ckpts/` and `cache-<jobid>/` first.
