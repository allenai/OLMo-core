# MoE ladder handoff

This is the starting context for a fresh Codex session working on Jacob's OLMoE ladder experiments.

## Workspace

- Repo: `/weka/oe-adapt-default/jacobm/olmoe3/OLMo-core`
- Branch: `jacobm/olmoe-dev-v2`
- GitHub repo: `git@github.com:allenai/OLMo-core.git`
- Main experiment folder: `src/scripts/train/jacobm_olmoe_ladder/`

On a fresh session, start with:

```bash
cd /weka/oe-adapt-default/jacobm/olmoe3/OLMo-core
git status --short
git pull --ff-only
```

Do not revert unrelated local changes. `beaker-docs/` may exist as an untracked directory on the laptop checkout; it was intentionally left uncommitted.

## Remote control session

Hammond is being used as a CPU-only control plane for launching and analyzing Beaker/W&B experiments.

- Host: `hammond-cs-aus-452.reviz.ai2.in`
- Current Beaker session: `01KT7NS3EZMTR1CE38K1RJMTCR`
- Current Codex remote environment ID: `env_e_6a209d70cad08331a7ffb20891a66edb`
- Control root: `/weka/oe-adapt-default/jacobm/olmoe3`
- Repo root: `/weka/oe-adapt-default/jacobm/olmoe3/OLMo-core`

Details and recovery commands are in:

- `src/scripts/train/jacobm_olmoe_ladder/HAMMOND_CONTROL.md`
- `src/scripts/train/jacobm_olmoe_ladder/launch_hammond_control_session.sh`

If the session dies, recreate it from a machine with Beaker/GitHub/Codex auth:

```bash
src/scripts/train/jacobm_olmoe_ladder/launch_hammond_control_session.sh
```

Use `REFRESH_SECRETS=1` only when local secrets need to be re-synced into Beaker.

## Beaker and W&B defaults

Training jobs use:

- Beaker workspace: `ai2/OLMo-3-moe-experiments`
- Budget: `ai2/oe-other`
- Training cluster: `ai2/titan`
- Training image: `tianhuat/olmo-core-torch211-2404-cu128`
- W&B: `ai2-llm/jacobm-olmoe-ladder`
- Checkpoint root: `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3`
- Data root: `s3://ai2-llm`

Required Beaker secrets:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `jacobm_WANDB_API_KEY`

Hammond control-session secrets are listed in `HAMMOND_CONTROL.md`.
Use `jacobm_git_config` for remote git identity; do not use the shared `git-config`
secret for this project.

## Training script and model sizes

The main train script is:

- `src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py`

Despite the filename, this script now supports all current baseline sizes:

- `275m`: current tiny MoE baseline, about 278M active including embeddings
  and about 1.13B total params.
- `mid_480m`: planned midpoint baseline rung, not yet implemented/smoked. The
  intended shape is 16 layers at `d_model=1024`, with the same 48E/top-4 MoE A0
  recipe. Estimated counts before smoke-test confirmation are about 480M active
  including embeddings/head and about 2.6B total params.
- `810m`: about 817M active including embeddings and about 4.93B total params.
- `1p2b`: about 1.22B active including embeddings and about 7.76B total params.

Important arguments:

- `--model-size`
- `--chinchilla-multiple`
- `--global-batch-size-seq`
- `--lr`
- `--gpus-per-node`
- `--micro-batch-size`
- `--ep-dim`
- `--ladder-evals`
- `--eval-task-set=fast`
- `--eval-interval`

Checkpoint policy for current/future training jobs:

- Permanent checkpoints: final only, using `--save-interval=999999999`.
- Resume checkpoints: ephemeral every 500 steps, using
  `--ephemeral-save-interval=500`.

This avoids filling Weka with intermediate permanent checkpoints.

Throughput policy:

- Prefer `EP_DIM=1` for these model sizes. Tianhua indicated EP only becomes
  necessary at substantially larger total parameter counts, roughly 5B+ in the
  relevant regime, and our smoke tests confirmed EP=1 is faster where it fits.
- Increase microbatch as much as memory allows, then use GPU count primarily to
  improve wall clock when the cluster is idle.
- The old slow setting was `EP_DIM=8`, `MICRO_BSZ=1`; avoid using it except for
  explicit sanity checks.
- Aim for `throughput/device/TFLOPs_per_GPU` around 600+ when possible.

Canonical settings currently in use:

| Model | Cx | Batch tokens | `global_batch_size_seq` | GPUs | EP | Microbatch | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 275M | 1 | 262,144 | 32 | 1-2 | 1 | 16 | Canonical final family for Cx1. |
| 275M | 2 | 393,216 | 48 | 2 | 1 | 8 | Canonical Cx2 repair family as of 2026-06-12. Old `b256k` baseline and `b512k` EG runs are diagnostic. |
| 275M | 4 | 524,288 | 64 | 4 | 1 | 16 | Canonical final family. |
| 275M | 8 | 786,432 | 96 | 8 | 1 | 4 | Forward policy. Historical baseline used 4 GPUs / mb8. |
| 275M | 16 | 1,048,576 | 128 | 8 | 1 | 16 | Used to finish Cx16 faster. |
| Midpoint | 1 | 262,144 | 32 | 4 | 1 | 8 | Completed/validated family. |
| Midpoint | 2 | 393,216 | 48 | 4 | 1 | 4 | Canonical Cx2 repair family as of 2026-06-12. Old `b512k` runs are diagnostic. |
| Midpoint | 4 | 524,288 | 64 | 4 | 1 | 8 | Completed/validated family. |
| Midpoint | 8 | 786,432 | 96 | 8 | 1 | 4 | Completed/validated family. |
| 810M | 1 | 262,144 | 32 | 4-8 | 1 | 4 | Completed Cx1 family. |
| 810M | 2 | 393,216 | 48 | 8 | 1 | 2 | Canonical Cx2 repair family as of 2026-06-12. Old `b512k` runs are diagnostic. |
| 810M | 4 | 524,288 | 64 | 8 | 1 | 4 | Completed Cx4 family. |
| 810M | 8 | 786,432 | 96 | 8-16 | 1 | 4 | Completed at 8 GPUs; forward policy prefers 16 GPUs when launching fresh. |
| 1.2B | 1 | 262,144 | 32 | 8 | 1 | 2 | Completed Cx1 family. |
| 1.2B | 2 | 524,288 | 64 | 8 | 1 | 2 | Not launched yet; low priority. |
| 1.2B | 4 | 524,288 | 64 | 8-16 | 1 | 2 | Completed at 8 GPUs; forward policy prefers 16 GPUs when launching fresh. |
| 1.2B | 8 | 786,432 | 96 | 8 | 1 | 4 | Preferred one-node replacement setting after 32-GPU/mb1 underperformed. |

## Run tracking and analysis

Primary docs:

- `src/scripts/train/jacobm_olmoe_ladder/RUNS.md`
- `src/scripts/train/jacobm_olmoe_ladder/ANALYSIS.md`

W&B tooling:

- `src/scripts/train/jacobm_olmoe_ladder/summarize_wandb_losses.py`
- `src/scripts/train/jacobm_olmoe_ladder/analyze_wandb_ladder.py`
- `src/scripts/train/jacobm_olmoe_ladder/plot_wandb_ladder.py`
- `src/scripts/train/jacobm_olmoe_ladder/plot_cx1_uplot.py`
- `src/scripts/train/jacobm_olmoe_ladder/wandb_cache.py`

Loss metric:

- `train/CE loss`

LR selection metric:

- Use final-window **training** CE loss only for LR selection for now.
- Validation losses and downstream evals are logged/backfilled, but are
  observational only until Jacob explicitly changes the policy.
- Prefer `avg250M` as the primary decision column. Also inspect `avg100M` and
  `avg500M` for noise/sensitivity.
- Do not use a 512-step smoke test, eval-only run, sanity run, or partial run as
  a completed U-plot point.

The W&B scripts cache run histories, especially for completed runs, because
there will eventually be many ladder points. Prefer cached plotting for routine
updates. For newly completed runs or runs whose first W&B history scan was
short, use `--refresh-stale-cache` with the narrowest possible `--name-regex`.
Only use `--refresh-cache` when intentionally re-downloading every selected
finished run.

Useful pattern:

```bash
uv run --with wandb python src/scripts/train/jacobm_olmoe_ladder/summarize_wandb_losses.py \
  --name-regex 'olmoe3-tiny-275m'
```

Useful current summaries:

```bash
uv run --with wandb python src/scripts/train/jacobm_olmoe_ladder/analyze_wandb_ladder.py \
  --name-regex 'olmoe3-(tiny-275m|moe-a0-810m|moe-a0-1p2b)-cx' \
  --mode final --finished-only --windows-m 100 250 500
```

Regenerate plots from cached histories:

```bash
uv run --with wandb python src/scripts/train/jacobm_olmoe_ladder/plot_wandb_ladder.py \
  --name-regex 'olmoe3-(tiny-275m|moe-a0-810m|moe-a0-1p2b)-cx' \
  --finished-only --window-m 250
```

Refresh only when needed:

```bash
uv run --with wandb python src/scripts/train/jacobm_olmoe_ladder/analyze_wandb_ladder.py \
  --name-regex '<specific-finished-run-family>' \
  --mode final --finished-only --windows-m 100 250 500 --refresh-stale-cache
```

## Laddering philosophy

We are building a MoE ladder foundation before committing to larger architecture choices.

The v0 baseline goal is to establish comparable LR-tuned baselines across three
active-parameter rungs (`275m`, `810m`, `1p2b`) and useful Chinchilla multiples.
This gives us a baseline ladder for judging later architecture interventions
such as alternate sparsity choices.

Dense-ladder context from coworkers:

- Their batch-size rule was roughly `B = 0.6 * D ** 0.6`.
- Example batch schedule:
  - Cx1: 262,144 tokens
  - Cx2: 524,288 tokens
  - Cx4: 524,288 tokens
  - Cx8: 786,432 tokens
  - Cx16: 1,048,576 tokens
- They fit U-plots over log LR vs final-stage train loss.
- They sometimes fit linear trends over the final 1k steps, but averaging over
  final token windows is acceptable for our current use.

Our LR sweep policy:

- Start with a coarse factor-spaced sweep around a transfer estimate, not a
  dense grid.
- For larger/expensive rungs, use exactly four points when possible.
- If the curve is still descending at the high edge, launch a farther-out
  sentinel point rather than walking outward by only one small multiple.
- If the best point is bracketed, fit a local quadratic in log LR. Use three
  points around the basin or five points if the local neighborhood is clean.
- Treat fitted optima as decision aids, not exact truth. Prefer robust bracketed
  basins over tiny final-window differences.
- Do not stop full training jobs early unless a run is clearly invalid or
  uninformative. Jacob prefers finishing full runs for robust plots.

Transfer-rule policy:

- The initial prior `alpha=-0.25` for model-size transfer was too hot for this
  MoE ladder.
- Real 275M -> 810M evidence implies a much cooler size transfer, roughly
  `alpha ~= -1.0` when using active params including embeddings, or `alpha ~= -0.9`
  with active non-embedding params.
- After observing 810M Cx1, calibrating the 275M Cx rule predicted 810M Cx4
  well: predicted about `4.5e-4` to `5.0e-4`, observed/fitted about `5.0e-4`.
- Use completed larger-model points to refine transfer before launching dependent
  rungs. Do not pre-decide LRs for dependent rungs before the relevant completed
  evidence lands.

Current fitted/observed LR centers, training avg250M:

| Model / Cx | Pre-run prediction / center | Observed or current expectation |
| --- | ---: | ---: |
| 810M Cx1 | initial `~1.0e-3`, later widened colder | fitted `~6.2e-4` |
| 810M Cx4 | `4.5e-4` without 275M Cx2, `5.0e-4` with Cx2 | fitted `~5.0e-4` |
| 1.2B Cx1 | `~4.0e-4` from updated transfer | fitted `~4.8e-4` to `5.0e-4` |
| 1.2B Cx4 | `~3.3e-4` via size transfer, `~3.9e-4` via 1.2B Cx1 times 810M Cx4/Cx1 ratio | completed; bracketed around `4e-4`, with `3e-4`/`6e-4` effectively tied |
| 810M Cx2 | interpolation/extrapolation around `5.1e-4` to `5.6e-4` | pending |
| 810M Cx8 | extrapolation around `3.5e-4` to `4.5e-4` | pending |

Validation/eval policy:

- Current in-loop eval setting is `--ladder-evals --eval-task-set=fast
  --eval-interval=2000`.
- Evals include C4 plus downstream BPB/BPBv2 style metrics such as MMLU RC BPBv2
  and ARC Challenge RC BPBv2.
- Final checkpoint eval backfills can be copied onto source W&B runs using
  `copy_eval_backfills_to_wandb.py`.
- We still need a future discussion on how validation losses should affect
  checkpoint/LR selection. For now, ignore evals for LR choice.

## 2026-06-12 current state

Use `CURRENT_PLAN.md` as the active source of truth. The short version:

- Do not queue more jobs right now.
- Re-enter the loop with a long cadence, about 4 hours unless a job is near
  completion or a just-started job needs startup/OOM checks.
- Monitor existing baseline repair runs, remaining baseline runs, and expert
  granularity runs. Monitor total-sparsity jobs only if new tracked IDs appear.
- Cx2 repair is now the main cleanup thread:
  - 275M baseline + `eg24e2k` + `eg96e8k`: `b384k`, 2 GPUs, EP=1, mb8,
    LRs `9e-4`, `1.8e-3`, `3.6e-3`.
  - `mid_480m`: `b384k`, 4 GPUs, EP=1, mb4, LRs `4.5e-4`, `9e-4`, `1.8e-3`.
  - 810M: `b384k`, 8 GPUs, EP=1, mb2, LRs `2.8e-4`, `5.6e-4`, `1.12e-3`.
  - 1.2B Cx2 has not been launched and remains low priority.
- Expert granularity currently has Cx1/Cx4 completed for the main two variants,
  Cx8 in progress/queued, and repaired Cx2 queued.
- Total sparsity currently has only one confirmed completed full run:
  `sp96e4k` Cx1 `1e-3`. The other tracked Cx1/Cx4 sparsity jobs were manually
  canceled, and no tracked Cx8 sparsity IDs were found. Do not queue more from
  this session without explicit confirmation.
- When full runs finish, refresh only their stale/missing W&B histories,
  regenerate plots, update docs, commit, and push.
- LR selection still uses training loss only. Evals are observational.

## 2026-06-07 historical pause state

This was the state at pause, after commit `4ddca365` was pushed.
It is retained for provenance and is superseded by the 2026-06-12 current state
above.

Recently completed:

- 1.2B Cx1, `gpu8-ep1mb2`, `global_batch_size_seq=32`: `1e-4`, `2e-4`,
  `4e-4`, `8e-4` all finished. Final-window avg250M favors `4e-4`; local
  quadratic fits put `lr*` around `4.8e-4` to `5.0e-4`.

Currently running:

- 810M Cx8, `gpu8-ep1mb4`, `global_batch_size_seq=96`: `1e-4`, `2e-4`,
  `4e-4`, `8e-4`.
- 1.2B Cx4, `gpu8-ep1mb2`, `global_batch_size_seq=64`: `1.5e-4`, `3e-4`,
  `6e-4` are running.

Queued/not yet started:

- 1.2B Cx4 `1.2e-3`.
- 810M Cx2, `gpu8-ep1mb4`, `global_batch_size_seq=64`: `1.5e-4`, `3e-4`,
  `6e-4`, `1.2e-3`.

Beaker IDs:

| Family | LR | Beaker ID |
| --- | ---: | --- |
| 810M Cx8 | `1e-4` | `01KTHQWMSQ0A4P6RCNKPS7YPYD` |
| 810M Cx8 | `2e-4` | `01KTHQX04RMEK7C7V6DZRZVXM6` |
| 810M Cx8 | `4e-4` | `01KTHQXB575GS84FBP4SNZ1GAA` |
| 810M Cx8 | `8e-4` | `01KTHQXNN4MFDBAP490ACJTJ07` |
| 1.2B Cx4 | `1.5e-4` | `01KTHW5XZXCNW9VV7FAMCS1C8F` |
| 1.2B Cx4 | `3e-4` | `01KTHW68C59T1XE9WNFW3EP3G1` |
| 1.2B Cx4 | `6e-4` | `01KTHW6KH3XFR790J6J4G8ZAJ6` |
| 1.2B Cx4 | `1.2e-3` | `01KTHW6ZSXGD1P8NEA7S3KM198` |
| 810M Cx2 | `1.5e-4` | `01KTHW7HB59AMPSZBP8FJHS5QG` |
| 810M Cx2 | `3e-4` | `01KTHW7WY6Z2NFAP8FNT1HP3XN` |
| 810M Cx2 | `6e-4` | `01KTHW88Q43J8M8CRCDN9VZDHV` |
| 810M Cx2 | `1.2e-3` | `01KTHW8MCKRJH3PW0W58KRVXA4` |

Latest ETA check before pausing:

- Shortest ETA was about 44 hours, from the running 810M Cx8 jobs.
- 810M Cx8 progress was about 3.9% for `1e-4`/`2e-4`/`4e-4`, and about 2.2%
  for `8e-4`.
- Running 1.2B Cx4 jobs were about 1.0% through with estimated 56-57 hours left.
- All running jobs had `optim/step skipped = 0`.

ETA command:

```bash
uv run --with wandb python - <<'PY'
import time, math
from datetime import datetime
import wandb

api = wandb.Api(timeout=60)
project = "ai2-llm/jacobm-olmoe-ladder"
families = (
    "olmoe3-moe-a0-1p2b-cx4-b512k-gpu8-ep1mb2",
    "olmoe3-moe-a0-810m-cx8-b768k-gpu8-ep1mb4",
    "olmoe3-moe-a0-810m-cx2-b512k-gpu8-ep1mb4",
)

def target_tokens(name: str) -> int | None:
    if "810m-cx8" in name:
        return 20 * 817_000_000 * 8
    if "810m-cx2" in name:
        return 20 * 817_000_000 * 2
    if "1p2b-cx4" in name:
        return 20 * 1_220_000_000 * 4
    return None

now = time.time()
rows = []
for run in api.runs(project, per_page=200):
    if not any(family in run.name for family in families):
        continue
    tokens = run.summary.get("throughput/total tokens")
    target = target_tokens(run.name)
    created_at = run.created_at
    start_ts = None
    if isinstance(created_at, str):
        start_ts = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
    elapsed = now - start_ts if start_ts else None
    rate = float(tokens) / elapsed if tokens and elapsed and elapsed > 0 else None
    eta = (target - float(tokens)) / rate if target and tokens and rate and float(tokens) < target else None
    rows.append((eta if eta is not None else float("inf"), run.state, run.name, tokens, target, rate))

for eta, state, name, tokens, target, rate in sorted(rows):
    eta_s = "n/a" if math.isinf(eta) else f"{eta / 3600:.1f}h"
    progress = "n/a" if not tokens or not target else f"{100 * float(tokens) / target:.1f}%"
    print(f"{eta_s}\t{progress}\t{state}\t{name}")
PY
```

Next loop:

1. Check Beaker/W&B for started/failed jobs.
2. Check only occasionally. There is no reason to loop tightly while the shortest
   ETA is roughly two days.
3. Once a full run finishes, update final-window training loss summaries and
   regenerate plots.
4. Push plot/doc updates to GitHub when new completed results are added.
5. Do not choose larger Cx rungs until the current 810M Cx2/Cx8 and 1.2B Cx4
   evidence lands.
6. If a fresh loop is started while jobs are still far from completion, use a
   long cadence such as 4-8 hours.

Current pushed plot files:

- `src/scripts/train/jacobm_olmoe_ladder/plots/275m_all_cx_uplot.png`
- `src/scripts/train/jacobm_olmoe_ladder/plots/810m_all_cx_uplot.png`
- `src/scripts/train/jacobm_olmoe_ladder/plots/1p2b_all_cx_uplot.png`
- Per-Cx plot files in the same directory.

## Recent important implementation commits

- `4ddca365` launch 1.2B Cx4 and 810M Cx2 sweeps; add 1.2B plotting support
- `2d7917a0` queue 810M Cx8 ladder sweep
- `11d1c46a` record 810M Cx4 fit and 1.2B Cx1 launch
- `3f0ac56a` prepare 1.2B Cx1 ladder launcher
- `49b53f25` cache W&B histories
- `fbf429d0` improve throughput settings
- `699cdbd9` add 2-GPU smoke launcher
- `a822a0f7` record throughput smoke runs
- `c15994a9` use partial-node settings for relaunches
- `267c4b2d` record partial-node tiny MoE relaunches
- `f94b0ae1` add high-LR follow-up launcher
- `fb98d59b` record high-LR follow-up runs
- `42fdcb2d` add Hammond control session launcher
- `353e7ca5` document Hammond Codex remote connection

## Practical cautions

- Never print secret contents. A prior Beaker config inspection exposed sensitive output in tool logs; do not repeat that in user-visible text.
- Beaker remote exec from the laptop may try local username `jacob`; use direct SSH as `jacobm` when needed:

```bash
ssh jacobm@hammond-cs-aus-452.reviz.ai2.in \
  "beaker session exec <SESSION_ID> -- bash -l"
```

- When launching training, preserve the current naming convention: experiment names encode Cx, batch, EP/MB, LR, and a short git hash.
- Checkpoint paths are derived from experiment names under `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3`.
- Use `uv run ...` locally and remotely unless there is a strong reason not to.
- Use `rg` first for search.
- Use `apply_patch` for manual file edits.
