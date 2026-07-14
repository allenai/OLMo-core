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

## Immediate Resume Snapshot

Last updated: 2026-06-12 after the plot/cache update commit `b05441de`.
The documentation-only handoff commit may be newer.

The loop is currently paused for handoff. Prefer explicit queue bundles and poll on request or at natural milestones. If Jacob asks for monitoring, use a real 4-hour cadence unless a just-started job needs a short startup check.

Currently active/queued jobs to watch:

| Area | LR / variant | Beaker | Status at handoff |
| --- | --- | --- | --- |
| 810M Cx2 `b384k` repair | `2.8e-4`, `r3` | `01KTYEJH9S58TDA837YZANCJ9C` | started; W&B `uh4el1df`; reached step 100+ cleanly |
| 810M Cx2 `b384k` repair | `5.6e-4`, `r3` | `01KTYEJWXR7X0KFBFQ57G9YC9V` | started; W&B `v5puakhq`; reached step 29+ cleanly |
| 810M Cx2 `b384k` repair | `1.12e-3`, `r3` | `01KTYEK7S5MXJYSJMGB3BNR1HE` | queued/created |
| 810M expert granularity | coarse Cx1 `6e-4` | `01KTX8DY64MRW5DCWAJZPQY1YR` | started |
| 810M expert granularity | coarse Cx4 `4e-4` | `01KTXR4J7FN4ERB9BYDKC261F5` | started |
| 810M expert granularity | fine Cx1 `6e-4` | `01KTXR7563GGMW6FE57TTVACSY` | started |
| 810M expert granularity | fine Cx4 `4e-4` | `01KTXR9YA2QAR7HB1XS1R0FTBW` | started |
| 480M Cx2 `b384k` repair | `4.5e-4`, `9e-4`, `1.8e-3` | `01KTWV11...`, `01KTWV1E...`, `01KTWV1T...` | all started |
| 275M expert-granularity Cx2 repair | coarse/fine `9e-4`, `1.8e-3`, `3.6e-3` | see `RUNS.md` | all started |
| 1.2B Cx8 canonical replacements | `2e-4`, `8e-4` | `01KTWB5V3...`, `01KTWB65Y...` | both started |

Known failed/ignored recent attempts:

- 810M Cx2 `r1`: failed before training with distributed startup/checkpointer
  errors. Treat as infrastructure/startup failure.
- 810M Cx2 `r2`: failed before training because W&B rejected an overlong group
  name: `invalid parameters: 128 limit exceeded for GroupName`.
- Commit `07df5ad` fixed W&B group names in the ladder train script; the `r3` jobs use
  that commit and have passed W&B init.
- Accidental 480M Cx2 `r2` duplicate jobs were stopped immediately; ignore.

Latest plotting state:

- Baseline plotter: `plot_wandb_ladder.py`.
- Expert-granularity plotter: `experiments/expert_granularity/plot_expert_granularity.py`.
- Both use W&B history caching and exclude running jobs by default.
- Use `--include-running` only for debugging live runs.
- `275m_all_cx_uplot.png` now includes the repaired 275M Cx2 curve and has
  direct line-end labels.
- Expert-granularity plots now include 810M output files as well as 275M:
  `plots/expert_granularity/810m_cx1_uplot.png`,
  `810m_cx4_uplot.png`, and `810m_cx8_uplot.png`.

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

The main train script for new launches is:

- `src/scripts/train/jacobm_olmoe_ladder/moe_a0_ladder.py`

The old `tiny_275m.py` path is kept as a historical copy for reproducing older launch records. `mid_480m` remains accepted as a compatibility alias for `480m`.

This script supports all current baseline sizes:

- `275m`: current tiny MoE baseline, about 278M active including embeddings
  and about 1.13B total params.
- `480m`: implemented 480M baseline rung, about 480M active including
  embeddings/head and about 2.6B total params. Smoke passed at `gpu4-ep1mb8`.
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
| 480M | 1 | 262,144 | 32 | 4 | 1 | 8 | Completed/validated family. |
| 480M | 2 | 393,216 | 48 | 4 | 1 | 4 | Canonical Cx2 repair family as of 2026-06-12. Old `b512k` runs are diagnostic. |
| 480M | 4 | 524,288 | 64 | 4 | 1 | 8 | Completed/validated family. |
| 480M | 8 | 786,432 | 96 | 8 | 1 | 4 | Completed/validated family. |
| 810M | 1 | 262,144 | 32 | 4-8 | 1 | 4 | Completed Cx1 family. |
| 810M | 2 | 393,216 | 48 | 8 | 1 | 2 | Canonical Cx2 repair family as of 2026-06-12. Old `b512k` runs are diagnostic. |
| 810M | 4 | 524,288 | 64 | 8 | 1 | 4 | Completed Cx4 family. |
| 810M | 8 | 786,432 | 96 | 8-16 | 1 | 4 | Completed at 8 GPUs; forward policy prefers 16 GPUs when launching fresh. |
| 1.2B | 1 | 262,144 | 32 | 8 | 1 | 2 | Completed Cx1 family. |
| 1.2B | 2 | 393,216 | 48 | 8 | 1 | 2 | Not launched yet; next baseline gap after settings confirmation. |
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
- `src/scripts/train/jacobm_olmoe_ladder/experiments/expert_granularity/plot_expert_granularity.py`
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
  --name-regex '(olmoe3-tiny-275m-cx|m480-cx|olmoe3-moe-a0-810m-cx|olmoe3-moe-a0-1p2b-cx)' \
  --mode final --finished-only --windows-m 100 250 500
```

Regenerate plots from cached histories:

```bash
uv run --with wandb python src/scripts/train/jacobm_olmoe_ladder/plot_wandb_ladder.py \
  --window-m 250
```

Regenerate expert-granularity plots from cached histories:

```bash
uv run --with wandb --with matplotlib python \
  src/scripts/train/jacobm_olmoe_ladder/experiments/expert_granularity/plot_expert_granularity.py \
  --window-m 250
```

Both plotters exclude running jobs by default to avoid distorted axes. Add
`--include-running` only when deliberately debugging in-flight runs.

Refresh only when needed:

```bash
uv run --with wandb python src/scripts/train/jacobm_olmoe_ladder/analyze_wandb_ladder.py \
  --name-regex '<specific-finished-run-family>' \
  --mode final --finished-only --windows-m 100 250 500 --refresh-stale-cache
```

## Laddering philosophy

We are building a MoE ladder foundation before committing to larger architecture choices.

The v0 baseline goal is to establish comparable LR-tuned baselines across the
active-parameter rungs (`275m`, `480m`, `810m`, `1p2b`) and useful Chinchilla multiples.
This gives us a baseline ladder for judging later architecture interventions
such as alternate sparsity choices.

Dense-ladder context from coworkers:

- Their batch-size rule was roughly `B = 0.6 * D ** 0.6`.
- Historical example batch schedule before the repaired Cx2 policy:
  - Cx1: 262,144 tokens
  - Cx2: 524,288 tokens
  - Cx4: 524,288 tokens
  - Cx8: 786,432 tokens
  - Cx16: 1,048,576 tokens
- Current canonical MoE Cx2 policy uses `b384k`: 393,216 tokens /
  `global_batch_size_seq=48` for all model sizes.
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
| 810M Cx8 | extrapolation around `3.5e-4` to `4.5e-4` | completed; bracketed around `4e-4` |

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
- Use explicit queue bundles rather than an autonomous monitoring loop. Poll on request, near completion, or for startup/OOM checks.
- Monitor existing baseline repair runs, remaining baseline runs, and expert
  granularity runs. Monitor total-sparsity jobs only if new tracked IDs appear.
- Cx2 repair is now the main cleanup thread:
  - 275M baseline + `eg24e2k` + `eg96e8k`: `b384k`, 2 GPUs, EP=1, mb8,
    LRs `9e-4`, `1.8e-3`, `3.6e-3`.
  - `480m`: `b384k`, 4 GPUs, EP=1, mb4, LRs `4.5e-4`, `9e-4`, `1.8e-3`.
  - 810M: `b384k`, 8 GPUs, EP=1, mb2, LRs `2.8e-4`, `5.6e-4`, `1.12e-3`.
  - 1.2B Cx2 has not been launched; prepare a `b384k` queue after settings confirmation.
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

- `b05441de` expand cached expert-granularity plots to 810M and add line-end
  labels to all-Cx baseline plots
- `e943eec4` include repaired 275M Cx2 in aggregate plots
- `f7d1788e` exclude running jobs from ladder plots by default
- `07df5ad7` shorten W&B group names to avoid the 128-character `GroupName`
  failure
- `ff4ef56a` track Cx2 repair retries and add launch selectors for 480M vs
  810M Cx2 repairs
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

- For new launches, use semantic resume-stable names: model, variant, Cx/data scale, batch policy when relevant, LR, and attempt id. Do not encode node count, GPU count, EP, microbatch, cluster, or other systems-only settings in new names; put those in W&B/Beaker tags and config.
- Checkpoint paths are derived from experiment names under `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3`; keeping names stable is what lets systems-only resumes work.
- Use `uv run ...` locally and remotely unless there is a strong reason not to.
- Use `rg` first for search.
- Use `apply_patch` for manual file edits.
