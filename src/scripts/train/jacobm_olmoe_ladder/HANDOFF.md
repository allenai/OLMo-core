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

## Tiny 275M MoE scripts

The main train script is:

- `src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py`

It supports MoE ladder runs with arguments for:

- `--chinchilla-multiplier`
- `--global-batch-size`
- `--learning-rate`
- `--gpus-per-node`
- `--micro-batch-size`
- `--ep-dim`

Current throughput-oriented settings:

- Cx1/Cx2 256k batch runs: 2 GPUs, `--gpus-per-node=2`, `--micro-batch-size=16`, `--ep-dim=1`
- Cx4 512k batch runs: 4 GPUs, `--gpus-per-node=4`, `--micro-batch-size=16`, `--ep-dim=1`

This replaced the older slow setting `EP_DIM=8`, `MICRO_BSZ=1`. Aim for `throughput/device/TFLOPs_per_GPU` around 600+.

## Run tracking and analysis

Primary docs:

- `src/scripts/train/jacobm_olmoe_ladder/RUNS.md`
- `src/scripts/train/jacobm_olmoe_ladder/ANALYSIS.md`

W&B tooling:

- `src/scripts/train/jacobm_olmoe_ladder/summarize_wandb_losses.py`
- `src/scripts/train/jacobm_olmoe_ladder/plot_cx1_uplot.py`
- `src/scripts/train/jacobm_olmoe_ladder/wandb_cache.py`

Loss metric:

- `train/CE loss`

The W&B scripts cache run histories, especially for completed runs, because there will eventually be many ladder points.

Useful pattern:

```bash
uv run --with wandb python src/scripts/train/jacobm_olmoe_ladder/summarize_wandb_losses.py \
  --name-regex 'olmoe3-tiny-275m'
```

## Current experiment direction

We are building a MoE ladder foundation before committing to larger architecture choices.

Near-term target:

1. Finish and analyze 275M-ish MoE learning-rate sweeps.
2. Pick good LRs for Cx1, Cx2, Cx4, and later higher chinchilla multiples.
3. Move to larger MoE model sizes, likely around 810M and 1.2B-ish active/ladder rungs.

Dense-ladder context from coworkers:

- Their batch-size rule was roughly `B = 0.6 * D ** 0.6`.
- For 275M-ish dense/hybrid, example batch schedule:
  - Cx1: 262,144 tokens
  - Cx2: 524,288 tokens
  - Cx4: 524,288 tokens
  - Cx8: 786,432 tokens
  - Cx16: 1M tokens
- They fit U-plots over log LR vs final-stage train loss.
- They sometimes fit linear trends over the final 1k steps, but averaging over final token windows is acceptable for our current use.

For MoE runs, we have been using final-token-window summaries rather than only final step loss.

## Recent important implementation commits

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
