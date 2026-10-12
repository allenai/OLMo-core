# Hero mt20 midtraining: how to launch a run

This is a handoff for an agent asked to midtrain a checkpoint of the OLMo 3.5 small hero
(OLMoE3: 16 layers, 7:1 KDA/full-attention hybrid, latent MoE with 512 experts / top-16, EMO
routing) with the "mt20" recipe. Read it end to end before changing anything. Every run so far
used the exact same recipe; only the source checkpoint and the run identity change. When the user
asks for a run that "looks exactly like" an earlier one, that is what they mean: diff configs,
specs and data against the earlier run and explain every difference.

## Current state (2026-10-12, 17:30 UTC)

All five tanushy training runs have finished and passed every Monitoring check. Each had one
2-step pass and one reload from step 2 (no task retries), and all 8 Beaker jobs exited 0. The four
runs launched on 2026-10-12 were checked against the 12T + 400B run with
`/weka/oe-training-default/tanushy/tmp/mt_postrun_check.py` (see Monitoring):

| Run | Beaker | CE, last 100 steps | vs 12T + 400B, same batches |
|---|---|---|---|
| `olmo35-small-12t400b-mt20-20261011-emo` | [01M4MW7D878HDD9354MEJD8EY2](https://beaker.org/ex/01M4MW7D878HDD9354MEJD8EY2) | 1.1738 | 0 |
| `olmo35-small-12p4t-mt20-20261012-emo` | [01M4PRN4Z3SZZP2MA9799E1SCB](https://beaker.org/ex/01M4PRN4Z3SZZP2MA9799E1SCB) | 1.2024 | +0.0287 |
| `olmo35-small-14t200b-mt20-20261012-emo` | [01M4QEY8TT9KY8GE0M1HRPWM1E](https://beaker.org/ex/01M4QEY8TT9KY8GE0M1HRPWM1E) | 1.1801 | +0.0063 |
| `olmo35-small-14t400b-mt20-20261012-emo` | [01M4QDH1FBZQRXZ4TGEBCA14QW](https://beaker.org/ex/01M4QDH1FBZQRXZ4TGEBCA14QW) | 1.1732 | -0.0006 |
| `olmo35-small-14t600b-mt20-20261012-emo` | [01M4QE60HTFAS7E0YS3Y0481SJ](https://beaker.org/ex/01M4QE60HTFAS7E0YS3Y0481SJ) | 1.1696 | -0.0042 |

- All runs are in workspace ai2/olmo3p5-training, use EMO, and have a 6h Beaker `minRuntime`
  (inherited from the 12T spec). The user asked about 8h after the 14T jobs were submitted but did
  not ask to change it.
- At 17:30 UTC the 14T + 200B `step5961` upload was at `payload_verified`; every other checkpoint
  of every run is `remote_verified`. Rerun the uploads check before evaluating that run.
- Evals: the 12T + 400B gen_mc, math and code evals finished (success markers and
  `metrics.json` under `scratch/olmo35-small-12t400b-mt20-20261011/emo/step5961/pilot-olmobase-*`)
  but have not been compared with the 2T/4T runs; the user deferred that. No evals have been
  launched for the 12.4T or 14T runs yet.
- The 14T + 600B first `beaker experiment create` failed with "a retryable database conflict
  occurred" and created nothing; the retry succeeded. Check for a partial experiment with
  `beaker workspace experiments <workspace> --text <name>` before retrying (the
  `<workspace>/<name>` reference form is invalid; use `<author>/<name>`).

## Why mt20, not the scaling-ladders midtraining script

The user's earlier midtraining used `allenai/scaling-ladders` (`midtraining.py`; local worktrees
in `/weka/oe-training-default/tanushy/scaling`). That script cannot run the hero: it builds
GDN-hybrid models from the run name, and its pinned olmo-core predates KDA, latent MoE and EMO.
The hero midtraining code lives here instead, ported from jacobm's 2T hero scripts.

mt20 is also not the current ladder recipe. Both use peak LR = pretraining peak / 5 = 2.2e-4, a
2000-step linear warmup, 100B tokens, the same mix and seed, and weights-only loading. They differ
in two choices:

| | Current scaling-ladders | mt20 (hero) |
|---|---|---|
| Global batch | 8Mi tokens (half of pretraining) | 16Mi tokens (same as pretraining) |
| Decay after warmup | linear to 0 | cosine to 0 |
| Steps for 100B | 11,921 | 5,961 |
| Warmup share | 16.8B tokens (17%) | 33.6B tokens (34%) |

The batch size also changes the shuffle, so only mt20 reproduces the hero runs' exact data
stream. The user chose mt20 for comparability with the earlier hero midtraining runs. Don't
switch recipes without asking.

## Runs so far

| Campaign | Source checkpoint | Source GPUs | EMO in MT | Code (branch @ commit) | W&B |
|---|---|---|---|---|---|
| `olmo35-small-2t-mt20-20260912` (jacobm) | 2T decay endpoints `olmo35-small-decay2t-20260911-{emo,non-emo}` step120000 | 64 | as pretrained | `codex/small-hero-2t-midtrain-20260912` (scripts at `05d8adba0`) | |
| `olmo35-small-2t-mt20-noemo-20260914` (jacobm) | 2T decay endpoint, emo arm, step120000 | 64 | off | jacobm branches | |
| `olmo35-small-4t-mt20-noemo-20260916` (jacobm) | 4T decay endpoints `olmo35-small-decay4t-20260916-{emo,non-emo}` step240000 | 64 | off | `codex/hero-4t-pipeline-20260916` | |
| `olmo35-small-12t400b-mt20-20261011` | 12T + 400B fixed PT decay endpoint `production-fixed-pt-decays/olmo35-small-12t-plus400bdecay-20261006-emo/step739342` | 64 | on | `tanushy/hero-12t400b-mt20-20261011` @ `6564a1d11` (train), `3aa13aca2` (evals) | [c78dfa02](https://wandb.ai/ai2-llm/olmo3p5-hero/runs/c78dfa02) |
| `olmo35-small-12p4t-mt20-20261012` | non-decayed hero `protected/olmo35-small-hero-20260907/emo/step739000/olmo-core` (alias `emo/12.4T`) | 128 | on | `tanushy/hero-12p4t-mt20-20261012` @ `e0ef65f44` | [308abca6](https://wandb.ai/ai2-llm/olmo3p5-hero/runs/308abca6) |
| `olmo35-small-14t400b-mt20-20261012` | 14T + 400B fixed PT decay endpoint `production-fixed-pt-decays/olmo35-small-14t-plus400bdecay-20261005-emo/step858308` | 64 | on | `tanushy/hero-14t400b-mt20-20261012` @ `705cd38ce` (worktree `core/hero-14t400b-mt20`) | [0fd59dbf](https://wandb.ai/ai2-llm/olmo3p5-hero/runs/0fd59dbf) |
| `olmo35-small-14t600b-mt20-20261012` | 14T + 600B fixed PT decay endpoint `production-fixed-pt-decays/olmo35-small-14t-plus600bdecay-20261008-emo/step870229` | 64 | on | `tanushy/hero-14t600b-mt20-20261012` @ `02f7ee3fe` (worktree `core/hero-14t600b-mt20`) | [31fad926](https://wandb.ai/ai2-llm/olmo3p5-hero/runs/31fad926) |
| `olmo35-small-14t200b-mt20-20261012` | 14T + 200B fixed PT decay endpoint `production-fixed-pt-decays/olmo35-small-14t-plus200bdecay-20261008-emo/step846387` | 64 | on | `tanushy/hero-14t200b-mt20-20261012` @ `5cf3a5558` (worktree `core/hero-14t200b-mt20`) | [2b7da1a8](https://wandb.ai/ai2-llm/olmo3p5-hero/runs/2b7da1a8) |

"emo"/"non-emo" arms name how the source was pretrained. The tanushy runs keep EMO on and are
ports of jacobm's 2T scripts (imported verbatim in `fb61d2cc7`) onto this branch, which carries
the 128-GPU hero runtime and the fixed PT decay tooling (`olmoe3_hero_fixed_decay_example.py`).
The 2T EMO run is the closest earlier comparison for EMO-on runs.

## The recipe (do not change unless explicitly asked)

| Setting | Value |
|---|---|
| GPUs | 64 (8 nodes x 8 B300 on ai2/holmes), DDP, microbatch 4 x 8192, grad accumulation 8 |
| Global batch | 16,777,216 tokens (2048 x 8192) |
| Peak LR | `1.1e-3 * 0.2` = 2.2e-4 |
| Schedule | `CosWithWarmup(warmup=2000, alpha_f=0.0)`, ending at step `END` |
| Tokens | 100B requested -> `END = 5961` steps = 100,008,984,576 tokens, exactly one pass over the 12,208,128 selected sequences |
| Data | `OLMo3-32B-midtraining-modelnamefilter.yaml`, SHA-256 `1ed52c81...25ba`, seed `1387822106`, read from GCS |
| Init | source model weights (fp32 master weights) and buffers only; optimizer moments, skip-step history, step counter and data stream all start fresh |
| Model | unchanged from pretraining, EMO kept on; `validate()` asserts the built model config equals the source `config.json` |
| Checkpoints | step 2, then every 500 steps, synchronous; in-loop perplexity evals every 1000 steps |
| W&B | project `ai2-llm/olmo3p5-hero`, group = campaign, run ID = first 8 hex characters of `sha256(run_id)` |

Because the mix, seed and batch are fixed, every mt20 run sees byte-identical batches in the same
order. Use that as a check (see Monitoring).

## Code map (`src/examples/olmo_ddp/`)

| File | Role |
|---|---|
| `olmoe3_hero_mt_plan.py` | All run constants (`CAMPAIGN`, `SOURCE`, `PT_STEP`, `SOURCE_GPUS`, ...), the `MTRun` identity (run ID, save root, uploader prefix, registration, `as_dict` provenance), `validate_checkpoint`, `ready`, `verified_copy` |
| `olmoe3_hero_mt.py` | Config builders, `MTAudit` callback (transfer/restore proofs, exact cosine LR check at every step), `--validate-only`, `--data-probe` |
| `olmoe3_hero_mt_node.py` | Beaker entrypoint run on each of the 8 nodes |
| `olmoe3_hero_mt_cache.py` | Puts `cached_path` document metadata on Weka, not the container disk |
| `olmoe3_hero_reshard_audit.py` | `verify_halved_transfer`: proves each 64-rank shard holds both saved 128-rank shards |
| `olmoe3_hero_mt_eval.py` | Post-training conversion and OLMoBase eval wrapper (see Evals) |
| `test_hero_mt_plan.py`, `test_olmoe3_hero_reshard_audit.py` | CPU tests |

Everything else (`olmoe3_small_hero.py`, `olmoe3_small_hero_plan.py`, runtime/kernel policy) is
the hero's production code and is reused as is.

## How a run executes

`olmoe3_hero_mt_node.py <run_id>` runs on all 8 replicas:

1. Checks the metadata cache (on the Weka mount, at least 12 TB free), the pinned runtime, and runs
   `olmoe3_hero_mt.py --validate-only`.
2. Rendezvous through files in `AUTOMATION/rendezvous/<experiment>/`; rank 0 picks the latest
   complete checkpoint in the run root, or the source if there is none, and validates its audits.
3. Pass 1 (only on a fresh start): `torchrun` from the source, weights-only, stopping at step 2.
   `MTAudit` writes `audit/initial-mt-transfer-rank{r}.json` and `audit/mt-restore-step0-rank{r}.json`
   on all 64 ranks; rank 0 then writes `audit/mt-smoke-saved.json`.
4. Pass 2: a new `torchrun` that reloads the full state from `step2` and trains to step 5961.
   `MTAudit` writes `audit/mt-restore-step2-rank{r}.json` (sampled state, RNG and data position
   exact). Rank 0 writes `audit/mt-success.json` at the end.

On a Beaker task retry the runner resumes from the latest complete checkpoint in the run root and
writes `mt-restore-step<N>-rank*.json` for it.

## Launching a new run, step by step

### 0. Rules and prerequisites

- Training jobs go in workspace **ai2/olmo3p5-training**. Eval jobs that run on ai2/jupiter or
  ai2/saturn go in **ai2/OLMo-3-moe-experiments** (the user was explicit about both). Ask before
  using any other workspace or cluster.
- Reference secrets by name only; never write literal secret values. If a needed secret is missing
  from a workspace, tell the user rather than copying one in.
- Keep scratch scripts and logs under `/weka/oe-training-default/<user>/tmp/`. `/tmp` on the dev
  machine gets wiped, which has killed watchers before.
- Never use `pkill -f <pattern>` with a pattern that matches your own shell command; kill by PID.
- Needs the Weka mounts, an authenticated `beaker` CLI, `uv`, and git push access to
  `allenai/OLMo-core`. The training node asserts at least 12 TB free on
  `/weka/olmo-3p5-checkpoints`, and the eval copy needs 11 TB after staging; check `df -h` first.
- If the user says they'll alert you when a job finishes, stop any watcher and don't poll.

### 1. Choose and inspect the source

- Use a native OLMo-core checkpoint with `model_and_optim/`, `train/rank*.pt` and
  `resume_audit/rank*.json`, not an HF export.
- Use the real path. Aliases such as `protected/olmo35-small-hero-20260907/emo/12.4T` are
  symlinks (to `step739000/olmo-core`), and `inventory()` rejects symlinks.
- Read `resume_audit/rank0.json` for `step`, `tokens` (must equal `step * 16777216`) and `gpus`.
- Read the source `config.json`: its `run_name` is `PARENT`, and its scheduler tells you whether
  it is decayed (fixed PT decays) or constant-LR (raw hero checkpoints).
- Protected checkpoints are read-only; the run only reads them.

### 2. Pick the base branch by source GPU count

Branch from the most recent run whose source has the same GPU count, in its own worktree so other
checkouts stay untouched:

```bash
git -C <any worktree> fetch origin
git -C <any worktree> worktree add -b tanushy/hero-<src>-mt20-<YYYYMMDD> \
  /weka/oe-training-default/tanushy/core/hero-<src>-mt20 origin/<base branch>
```

- **64-GPU source** (fixed PT decay endpoints, which were trained on 64 GPUs): base
  `tanushy/hero-14t200b-mt20-20261012` (`5cf3a5558`). Its code equals the 12T branch
  (`3aa13aca2`) except for source constants. `MTAudit` compares each rank directly against the
  same saved rank, and the parent is recorded through `DECAY_JOBS`.
- **128-GPU source** (non-decayed hero checkpoints under `protected/`): base
  `tanushy/hero-12p4t-mt20-20261012` (`e0ef65f44`). Its first load is checked by
  `verify_halved_transfer` (world size 64, two saved ranks `2r` and `2r+1` per live rank). DCP
  reshards the `.main` master weights and loads the replicated buffers; optimizer moments, RNG and
  data state are not loaded, so 128 -> 64 needs no other special handling. This branch uses
  `SOURCE_GPUS` and `ARMS` instead of `DECAY_JOBS`, and calls `verify_halved_transfer`
  unconditionally, so it fails on a 64-GPU source.

### 3. Change everything that depends on the starting checkpoint

Every 64-GPU source change so far (12T + 400B -> 14T + 400B -> 14T + 600B -> 14T + 200B) touched
exactly the items below and nothing else; the switch to a 128-GPU source also added the halved
audit code. Reference diffs:
`git diff 705cd38ce 02f7ee3fe` or `git diff 02f7ee3fe 5cf3a5558` (minimal 64-GPU retargets) and
`git diff 3aa13aca2 e0ef65f44 -- src/examples/olmo_ddp` (64-GPU 12T -> 128-GPU 12.4T, which also
adds the halved audit). `<src>` is a short lowercase source tag such as `14t600b` or `12p4t`.

**Gather the source values first:**

| Value | Where to get it |
|---|---|
| `PT_STEP` | `resume_audit/rank0.json` `step` (for fixed PT decays also `fixed-decay-plan.json` `end_step`) |
| `PARENT` | source `config.json` `run_name` (for fixed PT decays also `fixed-decay-plan.json` `run_name`) |
| Source GPUs | `resume_audit/rank0.json` `gpus` |
| Parent tokens | `PT_STEP * 16777216`; must equal `resume_audit/rank0.json` `tokens` |
| Total tokens after MT | `(PT_STEP + 5961) * 16777216` |
| Decay experiment (64-GPU) | list `<decay root>/audit/session-<job id>-<start step>.json`, take the job with the highest start step, then `beaker job get <job id> --format json` and read its experiment ID. A decay can span several experiments; record the one that wrote the final checkpoint |

**Code, in `src/examples/olmo_ddp/`:**

| # | File | What | 14T + 400B -> 14T + 600B example |
|---|---|---|---|
| 1 | `olmoe3_hero_mt_plan.py` | module docstring (names the source) | `14T + 400B EMO fixed PT decay endpoint` -> `14T + 600B ...` |
| 2 | `olmoe3_hero_mt_plan.py` | `CAMPAIGN` | `olmo35-small-14t400b-mt20-20261012` -> `olmo35-small-14t600b-mt20-20261012` |
| 3 | `olmoe3_hero_mt_plan.py` | `BRANCH` | `tanushy/hero-14t400b-...` -> `tanushy/hero-14t600b-...` |
| 4 | `olmoe3_hero_mt_plan.py` | `PARENT` | `olmo35-small-14t-plus400bdecay-20261005-emo` -> `olmo35-small-14t-plus600bdecay-20261008-emo` |
| 5 | `olmoe3_hero_mt_plan.py` | `PT_STEP` | `858308` -> `870229` |
| 6 | `olmoe3_hero_mt_plan.py` | `SOURCE` formula: keep `MOUNT / "production-fixed-pt-decays" / PARENT / f"step{PT_STEP}"` for fixed PT decays; for raw hero checkpoints use `MOUNT / "protected/olmo35-small-hero-20260907/emo" / f"step{PT_STEP}" / "olmo-core"`. Never point it at an alias symlink | unchanged |
| 7 | `olmoe3_hero_mt_plan.py` | `DECAY_JOBS = {"emo": ...}` (64-GPU branch), or `SOURCE_GPUS` plus `source_stage` in `as_dict` (128-GPU branch) | `01M48Y69B6EBJT6XS25CWWQMQW` -> `01M4D727Z0M27GDCZ6RQQNGXFA` |
| 8 | `olmoe3_hero_mt_plan.py` | `MTRun.prefix` (HF bucket path; must not collide with any registration) | `mt20-after-14t400b/{arm}` -> `mt20-after-14t600b/{arm}` |
| 9 | `olmoe3_hero_mt.py` | source tag in `trainer_config` `wb.tags` (`decayed-<src>-source`; the 128-GPU branch uses `constant-lr-<src>-source` and `128to64-source`) | `decayed-14t400b-source` -> `decayed-14t600b-source` |
| 10 | `olmoe3_hero_mt_eval.py` | module docstring | `14T + 400B mt20 endpoint` -> `14T + 600B ...` |
| 11 | `test_hero_mt_plan.py` | `test_recipe_totals`: run ID, `parent_tokens`, `total_seen_tokens`. These are plain numbers that a grep for the old name will not find | `14_400_018_710_528` / `14_500_027_695_104` -> `14_600_019_902_464` / `14_700_028_887_040` |
| 12 | `test_hero_mt_plan.py` | `test_isolated_child_identity`: expected prefix and the `-non-emo` run ID | `mt20-after-14t400b/emo` -> `mt20-after-14t600b/emo` |
| 13 | `test_hero_mt_plan.py` | source test skip message (names the source); on the 128-GPU branch also the source assertions (scheduler class `ConstantWithWarmup` vs `WSD`, `SOURCE_GPUS` gate) | `14T + 400B decay endpoint` -> `14T + 600B ...` |
| 14 | `test_hero_mt_plan.py` | `test_final_checkpoint_is_ready` run ID path | `olmo35-small-14t400b-...-emo` -> `olmo35-small-14t600b-...-emo` |

**Do not change** `LR`, `WARMUP`, `REQUESTED_TOKENS`, `END`, `SMOKE_END`, `SEED`, `MIX_SHA256`,
`CORE_REF`, `HELPER_REF`, the arm (`emo`), `PRIOR_MT20` in the test (a fixed reference to the 2T
mt20 session) and, on the 128-GPU branch only, `PRIOR_SOURCE` (a fixed reference to the 12T
step739342 model config), or anything outside the four files above.

**Then prove nothing stale is left.** Sweep the whole repo for the old campaign tag, old
`PARENT`, old `PT_STEP`, old parent/total tokens (both `14_400_018` and `14400018` forms), old
`DECAY_JOBS` ID and old commit prefix, e.g.
`rg -n -i "14t400b|plus400b|858308|14_400_018|14400018|14_500_027|01M48Y69|705cd38c|14T \+ 400B" .`,
which must print nothing. Then `git diff <base commit> -- src/examples/olmo_ddp` must show only
the 14 items.

**Derived automatically** from `CAMPAIGN` (none may exist yet): the campaign dir
`production-hero-small-midtrain/<CAMPAIGN>` (`ROOT`) with the run root `<CAMPAIGN>-emo` and
`data-work/` inside it, the automation folder `uploader/automation/<CAMPAIGN>`, the eval scratch
`scratch/<CAMPAIGN>`, the W&B group and the W&B run ID (`sha256(run_id)[:8]`).

**Outside the code, per checkpoint** (covered in steps 6-9 and Evals):

| Artifact | What changes |
|---|---|
| `uploader/automation/<CAMPAIGN>/plan.json` | written fresh from the new commit (step 6) |
| uploader registration `<run_id>.json` | written fresh from the plan (step 7) |
| train spec | exactly five fields: `GIT_REF`, `GIT_BRANCH`, `arguments[2]` (run ID), `CACHED_PATH_CACHE_ROOT`, `description` (step 8) |
| experiment name | `<run_id>-train` (step 9) |
| submission receipt | `uploader/automation/<CAMPAIGN>/submissions/<run_id>-train.json` (step 9) |
| local helper scripts | the status script hard-codes its campaign and the validate script hard-codes its worktree path; copy and edit them. `mt_dump_configs.py` takes the worktree as an argument |
| eval specs (later) | campaign name everywhere, wrapper commit in the command and `GIT_REF`, the qualify HF path |
| this doc | add the run to "Runs so far" and "Current state" |

### 4. Test and lint

```bash
cd <repo>
PYTHONPATH=src:src/examples/olmo_ddp python -m unittest discover -s src/examples/olmo_ddp -p test_hero_mt_plan.py
PYTHONPATH=src:src/examples/olmo_ddp python -m pytest -q src/examples/olmo_ddp/test_olmoe3_hero_reshard_audit.py
isort <changed files> && black <changed files>
<venv>/bin/ruff check <changed files>   # ruff 0.15.9; newer ruff reports spurious C408 errors
```

The source test runs only where Weka is mounted and must pass (it validates every saved rank).
Install ruff into a venv on Weka (`pip install ruff==0.15.9`), not `/tmp`.

### 5. Commit, push, validate the config

Gantry clones the pushed commit, so commit and push first. Then build both pass configs locally.
`kernel_fun` is not installed on the dev machine, so stub it:

```python
# /weka/oe-training-default/<user>/tmp/mt_validate_local.py
import os, sys, types
sys.path.insert(0, "<repo>/src/examples/olmo_ddp")
kf = types.ModuleType("kernel_fun"); common = types.ModuleType("kernel_fun._common")
common.support = types.SimpleNamespace(MIN_CTAS=None); kf._common = common
sys.modules["kernel_fun"] = kf; sys.modules["kernel_fun._common"] = common
import olmoe3_hero_mt as mt
os.environ.setdefault("OLMO35_HERO_EXPECTED_START", "0")
os.environ.setdefault("OLMO35_HERO_STOP", str(mt.END))
os.environ["OLMO35_HERO_ALLOW_CONTINUATION"] = "1"
mt.hero.qualified.apply_policy()
mt.validate()
```

`cd <repo> && PYTHONPATH=src python3 <that file>` must print `HERO_MT_CONFIG_VALIDATED` for
start 0 and start 2. This also asserts the model equals the source `config.json`.

Then diff the new run's configs against the previous run's, field by field.
`/weka/oe-training-default/tanushy/tmp/mt_dump_configs.py` runs the same stubbed `validate()` and
dumps both pass configs as JSON:

```bash
cd <worktree> && PYTHONPATH=src python3 /weka/oe-training-default/tanushy/tmp/mt_dump_configs.py <worktree> <out prefix>
# writes <out prefix>-start0.json and <out prefix>-start2.json; flatten both runs' dumps and compare
```

Against a previous 64-GPU-source run, exactly these 21 of 1,121 fields may differ (anything else
is a recipe change and must be explained): `dataset.work_dir`, `launch.cmd[2]`,
`launch.git.branch`, `launch.git.ref`, `launch.name`, `run_name`,
`trainer.callbacks.checkpoint_ready.{lineage_id,run_id}`,
`trainer.callbacks.hero_audit.{output_dir,run_id}`, `trainer.callbacks.hero_complete.run_id`,
`trainer.callbacks.hero_storage.run_id`, `trainer.callbacks.lm_evaluator.eval_dataset.work_dir`,
`trainer.callbacks.slack_notifier.name`, `trainer.callbacks.wandb.{group,name,notes,tags[2]}`,
`trainer.load_path`, `trainer.save_folder`, `trainer.work_dir`. Against the 12.4T run, the W&B tag
list also shifts because that run has an extra `128to64-source` tag (28 fields). Comparing a
locally built config with one saved in a running job's checkpoint differs only in 6
environment-dependent fields: `launch.git.branch` (absent in the saved one),
`launch.git.{repo,repo_url}` (`OLMo-core` vs `Olmo-core`), the random `launch.name` suffix,
`trainer.callbacks.beaker.enabled` (set only inside Beaker) and
`trainer.callbacks.hero_audit._CLASS_` (`__main__.MTAudit` vs `olmoe3_hero_mt.MTAudit`).

### 6. Write the provenance plan

```bash
cd <repo>/src/examples/olmo_ddp && test -z "$(git status --porcelain)" && python3 - "$(git rev-parse HEAD)" <<'EOF'
import sys
from olmoe3_hero_mt_plan import AUTOMATION, ROOT, MOUNT, runs, validate_checkpoint, SOURCE, PT_STEP, SOURCE_GPUS
from olmoe3_lr_sweep_watch import atomic_json
assert MOUNT.is_mount() and not AUTOMATION.exists() and not ROOT.exists()
validate_checkpoint(SOURCE, PT_STEP, SOURCE_GPUS)
AUTOMATION.mkdir(parents=True)
atomic_json(AUTOMATION / "plan.json", dict(source_commit=sys.argv[1], runs=[r.as_dict() for r in runs()]))
EOF
```

On the 64-GPU branch, drop `SOURCE_GPUS` from both the import and the call. The clean-tree check
also fails on untracked files, so commit or move them first.

### 7. Register the uploader lineage

The hero's `pre_train` refuses to start without a registration for the run, and the uploader
copies each checkpoint to the private HF bucket `allenai/olmo-3p5-small` under the run's prefix.

```bash
cd <repo>/src/examples/olmo_ddp && uv run -q --no-project --python 3.12 \
  --with 'olmo-checkpoint-uploader @ git+https://github.com/allenai/olmo-checkpoint-uploader.git@c49ca1f9eb0b4308bf8e32f170d13bcdff05bbee' \
  python - <<'EOF'
import glob, json
from olmo_checkpoint_uploader.models import Registration
from olmo_checkpoint_uploader.state import StateStore
from olmoe3_small_hero_plan import CONTROL, STATE
from olmoe3_hero_mt_plan import AUTOMATION, runs
(run,) = runs()
assert json.loads((AUTOMATION / "plan.json").read_text())["runs"] == [run.as_dict()]
reg, store = run.registration(), StateStore(CONTROL, STATE)
regs = [json.load(open(f)) for f in glob.glob(str(CONTROL / "registrations" / "*.json"))]
assert not any(r["lineage_id"] == reg["lineage_id"] or r["remote_prefix"] == reg["remote_prefix"]
               or r["checkpoint_root"] == reg["checkpoint_root"] for r in regs)
run.root.mkdir(parents=True, exist_ok=True)
print(store.register(Registration(**reg)))
EOF
```

Never replace an existing registration. The registration uses `deletion_mode: apply`, so once a
checkpoint is verified remotely the uploader deletes older local copies (it keeps at least 2);
finished run roots hold only the last two checkpoints. `HuggingFaceBucketBackend().assert_private(...)`
returns 403 locally with a non-fine-grained token; the uploader service checks privacy itself.

### 8. Build the Beaker spec from the previous run's spec

Copy the latest train spec, e.g.
`uploader/automation/olmo35-small-14t200b-mt20-20261012/specs/olmo35-small-14t200b-mt20-20261012-emo-train-5cf3a555.json`,
and change only:

- env `GIT_REF` (the pushed commit) and `GIT_BRANCH`
- `arguments[2]` (the new run ID)
- env `CACHED_PATH_CACHE_ROOT` (the plan's `METADATA_CACHE`, `<ROOT>/data-work/cached-path-metadata`)
- `description` (put the plan's `as_dict()` there)

Assert that nothing else differs from the previous spec, and that the task body and the
description's run fields no longer mention the old campaign, source step, decay experiment or
commit. Scope that check to those parts: a description note such as "identical to <previous
campaign> except the source" legitimately names the old run. Save it as
`AUTOMATION/specs/<run_id>-train-<commit8>.json`. What the spec carries:
8 replicas x 8 GPUs on a pinned list of 65 ai2/holmes hostnames, urgent priority, `minRuntime 6h`,
`autoResume`, 10 task retries, host networking, leader selection, image
`01KY8RWDARD3PR0F1T4ANGFH95`, the gantry entrypoint `bash /gantry/entrypoint.sh` with
`olmoe3_profile_setup.sh` as post-setup, Weka mounts `olmo-3p5-checkpoints` and `dolma-3p5`,
`OLMO35_HERO_EXPECTED_START=0` / `OLMO35_HERO_STOP=5961`, and secrets
`TANUSHY_{BEAKER_TOKEN,WANDB_API_KEY,GITHUB_TOKEN,GOOGLE_CREDENTIALS,AWS_*}` (the Google
credentials are what read the data from GCS). The leftover `OLMOE3_INTEGRATION_*` env vars come
from the hero spec and are expected. Do not edit the hostname list.

### 9. Pre-check, submit, record

Before `beaker experiment create <spec> --name <run_id>-train --workspace ai2/olmo3p5-training`:

- no experiment with that name exists: `beaker experiment get <author>/<name>` reports "not found"
  and `beaker workspace experiments ai2/olmo3p5-training --text <name>` lists nothing
- the run root is empty and the registration file exists in `uploader/control/registrations/`
- the branch head on origin equals `GIT_REF`
- every pinned hostname appears in `beaker cluster nodes ai2/holmes`

Then write `AUTOMATION/submissions/<run_id>-train.json` with the experiment ID, workspace, spec
path and SHA-256, plan path, commit, branch, expected start/stop and the time.

## Monitoring

Expected timeline (12.4T run): queued 72 min because holmes was full of other teams' urgent jobs
(`beaker cluster free-slots ai2/holmes` shows capacity); then about 25 min of setup, data
preparation and weight loading; about 6 min of kernel compilation; steps 1-2 and the save; about
13 min to restart for pass 2; then about 2.7 s/step, roughly 4.5 h for the remaining steps.

Reusable scripts:

- While running: `/weka/oe-training-default/tanushy/tmp/mt12p4t_train_status.py` (job states,
  proof counts, markers, checkpoints, last step/CE/LR). It hard-codes its campaign; copy and edit.
- After finishing: `cd <any hero mt20 worktree> && PYTHONPATH=src python3
  /weka/oe-training-default/tanushy/tmp/mt_postrun_check.py <ref_campaign> <campaign>...` runs
  every check in the table below against a finished reference campaign (use
  `olmo35-small-12t400b-mt20-20261011`). It reads the source step and Beaker experiment from
  `uploader/automation/<campaign>/{plan.json,submissions/}` and takes about 20 s per campaign.

Leader logs: `beaker job logs <job id>`.

| What | Where | Expect |
|---|---|---|
| Transfer proofs | `audit/initial-mt-transfer-rank*.json` | 64 files, all flags true; for a 128-GPU source each rank lists `source_ranks [2r, 2r+1]` with 543 replicated and 287 resharded tensors |
| Same data as earlier runs | `data-work/dataset-common/`, `data-work/global_indices_dataset_size12208128_epoch1_seed1387822106_v1.npy`, `audit/input-step1-rank*.sha256` | byte-identical to the previous mt20 campaign (12,263 files; 64/64 step-1 hashes) |
| Gate | `audit/mt-restore-step0-rank*.json`, `audit/mt-smoke-saved.json`, `audit/mt-restore-step2-rank*.json` | present; 64 of each, all exact |
| LR and loss | `audit/metrics.jsonl` (`optim/LR (group 0/1)`, `train/CE loss`, `optim/total grad norm`, `optim/step skipped`) | LR equals `CosWithWarmup(2000, 0).get_lr(2.2e-4, step, 5961)` (the run aborts otherwise); finite losses; `step skipped` is 1 on a few isolated steps (12 to 27 per run so far, see below) |
| Throughput | `throughput/device/TFLOPs_per_GPU` | about 440 |
| Config | `step*/config.json` vs the previous run | only identity, path and pass-specific load fields differ: 20 fields between two 64-GPU-source runs' `step5961` (the 21 at build time minus `launch.git.branch`, which saved configs record as null), 27 for the 12.4T run because its extra `128to64-source` W&B tag shifts the tag list (`step2` also differs in `hard_stop`, `load_path`, `load_optim_state`, `load_trainer_state`, `reset_optimizer_states_on_load` and no evals, because it was saved by pass 1) |
| Uploads | `uploader/state/checkpoints/<run_id>/step-*.json` | `status: remote_verified` |
| Done | `audit/mt-success.json`, `audit/complete-step5961.json`, `uploader/control/inbox/<run_id>/step-000000005961.ready.json`, step5961 upload `remote_verified` | all present; all 8 Beaker jobs exit 0 |

Expected and harmless:

- A step-1 grad norm in the millions (12T: 1.3M, 4T: 5.9M, 12.4T: 7.1M); step 2 is back to tens.
  The LR at step 1 is 1.1e-7. One-step grad-norm blips during warmup also happen (12T: 4.2 at
  step 29; 12.4T: 10.6 at step 32).
- `Ignoring load path (...step2) since checkpoint was found in save folder` at the start of pass 2:
  it loads the same `step2` from the save folder.
- The Beaker experiment description shows live progress (e.g. "33.6M tokens" right after the
  2-step pass): olmo-core's Beaker callback overwrites it. The submitted description is in the spec.
- The log's warning set: diff it against the previous run's leader log instead of judging each
  warning alone; the 12T and 12.4T sets were identical. Saved leader logs and warning extracts
  to diff against: `/weka/oe-training-default/tanushy/tmp/{mt12t,mt12p4t}-r0.{log,warn}`.
- A one-step throughput dip at step 64 (both runs).
- `optim/step skipped` = 1 on a few isolated steps: 17 (12T + 400B), 27 (12.4T), 16 (14T + 200B),
  13 (14T + 400B), 12 (14T + 600B), spread from step ~330 to the end. The optimizer
  (`OLMoDDPOptimizerConfig`, `rolling_interval_length=128`, `sigma_factor=6`) skips an update when
  the loss or grad norm is more than 6 sigma above its rolling window. The grad norm at those steps
  is small (0.09 to 0.92), and the skipped steps differ between runs even though the batches are
  identical. Investigate only if skips cluster or the count is far above these.
- Loss offsets between sources on identical batches: the 12.4T (non-decayed) run was about 0.10
  nats above the 12T + 400B (decayed) run over the first 100 steps, 0.045 over steps 1000-1999,
  and a steady 0.026 to 0.029 from step 2000 to the end. The 14T sources end within 0.01 of the
  12T + 400B run (table under Current state).
- Each campaign rebuilds its own metadata cache from GCS (about 3.5 min, 36,789 files, 77 GB).
  The data is identical anyway, because selection depends only on mix, seed and batch.

## Evals after training

`olmoe3_hero_mt_eval.py` reproduces jacobm's 2T mt20 eval pipeline so results are comparable.
Submit convert only after training is done: it asserts `ready(run, END)` (the uploader's inbox
event for step 5961) and fails otherwise.

1. `convert`: copies `step5961` to `scratch/<CAMPAIGN>/emo/step5961/olmo-core` (verified copy),
   converts to HF with OLMo-core at `CORE_REF` and checks parity (`conversion-success.json`).
2. `qualify` (plain spec, no wrapper; the HF path is hard-coded in it): HF vs vLLM precise parity
   and an eval smoke test (`vllm-parity-success.json`, `eval-smoke-success.json`).
3. `gen_mc`, `math`, `code`: OLMoBase suites run from scaling-ladders at `HELPER_REF`; each writes
   `pilot-olmobase-<suite>-*/full-eval-pilot-success.json` and `metrics.json`. The suites refuse
   to start unless the conversion marker and both qualify markers passed.

The specs are Beaker JSON specs whose command is a bash script, built from jacobm's
`uploader/automation/olmo35-small-2t-mt20-20260912/specs/olmo35-small-2t-mt20-20260912-emo-<stage>.json`,
changing only the wrapper commit (in the command and `GIT_REF`), the qualify model path, the
GitHub token secret and the description. The 12T versions are in
`uploader/automation/olmo35-small-12t400b-mt20-20261011/specs/` (`-jcs` variants allow
jupiter/ceres/saturn; receipts in `submissions/`). For a new campaign, also change the campaign
name everywhere and use a commit whose `olmoe3_hero_mt_plan.py` has the new `CAMPAIGN`.

- Clusters: the originals target ai2/jupiter and ai2/ceres (H100). The 12T qualify sat queued for
  about 40 minutes with no free slots, so with the user's approval convert, qualify and the suites
  ran on ai2/saturn (A100-80GB) instead. Submit them to ai2/OLMo-3-moe-experiments.
- Secrets: `HF_CHECKPOINT_BUCKET_TOKEN` and a GitHub token (scaling-ladders is private). The code
  suite used `jacobm_MODAL_TOKEN_ID`/`_SECRET` and `jacobm_CODE_EVAL_GOOGLE_CREDENTIALS` with
  the user's approval for the 12T campaign; confirm again before reusing them.
- Run convert, then qualify, then the three suites in parallel. On saturn, gen_mc took about
  2 h, code about 3.5 h and math about 4.3 h, within the 6 h timeout.
- Compare with earlier runs using `/weka/oe-training-default/tanushy/tmp/mt12t_compare.py`
  (reads each run's `pilot-olmobase-*/metrics.json`); `mt12t_eval_status.py` in the same folder
  shows eval job states.
