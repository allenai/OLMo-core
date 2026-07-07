# Midtraining And Long-Context Plan

Last refreshed: 2026-07-06.

This note records how the dense `scaling-ladders` repo stages midtraining and
long-context training, and what is required to apply the same process to the MoE
ladder.

## Dense `scaling-ladders` Recipe

The dense ladder generator wires each rung as:

1. `pretrain`
2. `midtrain`, loading from the rung's `pretrain` save folder
3. `long-context`, loading from the rung's `midtrain` save folder

Artifacts are stored rung-first:

`/weka/oe-training-default/ai2-llm/ladders/mainline/<user>/<run-id>/<size>-Cx<multiple>/<stage>/`

Midtraining:

- Sequence length: 8192.
- Token budget: 100B tokens.
- Data: `src/olmo_core/data/source_mixtures/OLMo3-32B-midtraining-modelnamefilter.yaml`.
- Dataset: `SourceMixtureList` + `NumpyFSLDatasetConfig.from_src_mix`.
- Load behavior: `load_strategy=always`, `load_trainer_state=False`,
  `load_optim_state=False`.
- Scheduler: constant LR after 2000 warmup steps (`ConstantWithWarmup` in the dense repo).
- Optimizer state is intentionally fresh so LR decisions are not coupled to the
  pretraining optimizer moments.

Long context:

- Sequence length: 65536.
- Token budget: 100B tokens in the dense ladder reference.
- Data: packed long-context Weka/GCS glob under
  `preprocessed/tylerr/lc-reshard-final-cleaned/...`.
- Dataset: `NumpyPackedFSLDatasetConfig.glob`.
- RoPE: YaRN-style scaling from 8192 to 65536.
- Larger dense runs use context parallelism.

## MoE Branch Readiness

Midtraining is close to ready. The branch already has:

- The midtraining source-mixture YAML.
- `SourceMixtureList`, `SourceMixtureDatasetConfig`, and
  `NumpyFSLDatasetConfig.from_src_mix`.
- Trainer support for weight-only loading via `load_trainer_state=False` and
  `load_optim_state=False`.
- The MoE architecture and optimizer configs used by our ladder experiments.

The new script
`src/scripts/train/jacobm_olmoe_ladder/experiments/midtraining/midtraining_ladder.py`
reuses `moe_a0_ladder.py` for architecture, batch, optimizer, W&B, and checkpoint
callbacks, but replaces:

- pretraining data with the OLMo 3 midtraining source mixture,
- Chinchilla duration with a fixed `--midtrain-max-tokens` budget,
- cosine pretraining LR schedule with an equivalent composable warmup-then-constant scheduler,
- checkpoint loading with explicit weight-only continuation from `--load-path`.

Long context is not ready for MoE promotion without more work:

- `MoEV2TransformerTrainModule` currently raises `NotImplementedError` when
  `cp_config` is set, so dense-style context parallelism cannot be reused.
- The MoE model config currently uses default RoPE at theta 500k with no scaling;
  long context should add YaRN or another agreed scaling policy before launch.
- 65k sequence memory and throughput need separate smoke tests for each target
  model size, likely starting with context-parallel-free small models only.

## Midtraining LR Search Proposal

For today's LR-search smoke path:

1. Pick the pretrained checkpoint to continue from. Start with one optimal 275M
   Cx8 checkpoint, unless the question is specifically Cx1 transfer.
2. Launch a short smoke test with the exact architecture variant and batch
   settings:
   - `--midtrain-max-tokens` can be reduced for smoke tests.
   - `--save-interval=999999999` and normal ephemeral checkpoints are fine.
   - Keep `--midtrain-load-optim-state=false`.
3. If the smoke reaches training with sane MFU/loss, launch a three-point LR
   bracket around the pretrained baseline LR shifted cooler by the dense ladder
   midtraining prior.

## First 275M Launcher

The first concrete launcher is:

`src/scripts/train/jacobm_olmoe_ladder/experiments/midtraining/launch_275m_lr_search.sh`

Required environment variables:

- `LOAD_PATH`: OLMo-core checkpoint folder to continue from.
- `SOURCE_TAG`: short stable label for the source checkpoint, used in run names.

Defaults:

| Setting | Value | Notes |
| --- | ---: | --- |
| Model size | `275m` | Can be overridden, but the launcher defaults are tuned for 275M. |
| LRs | `2e-4 4e-4 8e-4 1.6e-3` | Broad first pass between the MoE pretraining-ratio prior and dense absolute prior. |
| Max tokens | `100B` | Override `MIDTRAIN_MAX_TOKENS` for smoke tests. |
| Global batch seq | `128` | `1,048,576` tokens; larger than our pretraining batches without jumping to dense-ladder batch sizes. |
| GPUs | `4` for the first real 275M grid | The 4-GPU smoke scaled well enough to use for the first Cx1/Cx8 LR transfer check. |
| Microbatch | `8` | Legal with global batch 128 for 2, 4, or 8 GPU smoke/full launches. |
| EP | `1` | Keep the first midtraining path simple and comparable. |
| Scheduler | warmup 2000 steps, then constant | Implemented with the MoE-compatible composable scheduler. |
| Optimizer state | fresh | `--midtrain-load-optim-state=false` by default. |

Example smoke invocation:

```bash
LOAD_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/<run>/stepXXXXX \
SOURCE_TAG=baseline-cx8 \
MIDTRAIN_MAX_TOKENS=2000000000 \
LRS=8e-4 \
  src/scripts/train/jacobm_olmoe_ladder/experiments/midtraining/launch_275m_lr_search.sh
```

## Initial 275M Baseline LR-Transfer Grid

The first full midtraining data points continue from the optimal observed
pretraining baseline checkpoints at the two data-scale extremes, Cx1 and Cx8.
This tests whether the same midtraining LR bracket works when the source model
has seen much less or much more pretraining data.

Shared settings:

| Setting | Value |
| --- | --- |
| Script | `src/scripts/train/jacobm_olmoe_ladder/experiments/midtraining/midtraining_ladder.py` |
| Run prefix | `mt-275m` |
| Data | `OLMo3-32B-midtraining-modelnamefilter.yaml` |
| Sequence length | `8192` |
| Max tokens | `100B` |
| Global batch seq | `128` (`1,048,576` tokens) |
| GPUs | `4` |
| Nodes | `1` |
| EP | `1` |
| Microbatch | `8` |
| Scheduler | 2000-step warmup, then constant LR |
| Optimizer state | fresh; load model weights only |
| Cluster / workspace | `ai2/titan`, `ai2/OLMo-3-moe-experiments` |
| Priority | urgent |
| Checkpoint root | `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/midtraining` |

Source checkpoints:

| Source tag | Pretraining best observed LR | 10% MT LR | Source checkpoint |
| --- | ---: | ---: | --- |
| `baseline-cx1` | `2e-3` | `2e-4` | `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr2e-3-r2/step15365` |
| `baseline-cx2` | `1.8e-3` | `1.8e-4` | `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx2-b384k-gpu2-ep1mb8-lr1.8e-3-r3/step20486` |
| `baseline-cx4` | `1.5e-3` | `1.5e-4` | `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr1.5e-3/step30729` |
| `baseline-cx8` | `1.6e-3` | `1.6e-4` | `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr1.6e-3-r2/step40971` |

Initial Cx1/Cx8 LR grid for each source: `2e-4`, `4e-4`, `8e-4`, `1.6e-3`.
After the 2026-07-07 eval backfill summary, use `10%` of the canonical baseline
best observed pretraining LR as the first single-point transfer rule. The first
queued single-point follow-ups are Cx2 at `1.8e-4` and Cx4 at `1.5e-4`.

Run names intentionally omit systems settings so future restarts can adjust
GPU count or microbatch without changing the semantic run identity:

- `mt-275m-baseline-cx1-lr{2e-4,4e-4,8e-4,1.6e-3}-r1`
- `mt-275m-baseline-cx8-lr{2e-4,4e-4,8e-4,1.6e-3}-r1`

## Tentative Larger-Model Midtraining Batch Plan

These are planning targets, not yet smoke-tested settings. Keep batch choices in
this document and W&B tags rather than in run names. The larger-model LR rule is
`0.1 * canonical baseline best observed PT LR` for the matching model size and
pretraining Cx multiple.

| Model size | Planned global batch seq | Tokens / optimizer step | Initial systems target | Smoke status | Notes |
| --- | ---: | ---: | --- | --- | --- |
| 275M | `128` | `1,048,576` | 4 GPUs, EP1, MB8 | passed | Cx1/Cx8 grid complete, Cx2/Cx4 single points queued. |
| 480M | `192` | `1,572,864` | 4 GPUs, EP1, MB8 | [passed, stopped](https://beaker.org/ex/01KWZ9B2HS4RVK6ZFC5V80YGNT) | Cx8 smoke at `8e-5` for `2B` tokens stepped successfully. |
| 810M | `256` | `2,097,152` | 8 GPUs, EP1, MB4 | [passed, stopped](https://beaker.org/ex/01KWZ9B9KHRJTW139P8PMV6A4F) | Cx8 smoke at `4e-5` for `2B` tokens stepped successfully. |
| 1.2B | `384` | `3,145,728` | 8 GPUs, EP1, MB4 | [passed, stopped](https://beaker.org/ex/01KWZ9B4C3P04P6R5K63W8TMDF) | Cx8 smoke at `4e-5` for `2B` tokens stepped successfully. |

Full one-LR larger-model grid after smoke validation:

| Model size | Cx1 LR | Cx2 LR | Cx4 LR | Cx8 LR | GPUs / job | Concurrent GPUs for 4 Cx jobs | Launch state |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 480M | `1.2e-4` | `9e-5` | `8e-5` | `8e-5` | 4 | 16 | launched 2026-07-07 |
| 810M | `6e-5` | `5.6e-5` | `4e-5` | `4e-5` | 8 | 32 | launched 2026-07-07 |
| 1.2B | `4e-5` | `6e-5` | `3e-5` | `4e-5` | 8 | 32 | launched 2026-07-07 |

The 12 larger midtraining jobs require `80` concurrent GPUs if all run at once.
Including the two 275M Cx2/Cx4 follow-ups would make `88` concurrent GPUs, but
those were already running before this sweep launch. Individual Beaker links are
recorded in `RUN_TRACKER.md` and `RUNS.md`.

Open decisions after the initial baseline grid:

- Confirm the larger-model smoke tests are stable and fast enough with the planned global batch settings.
- Whether to add Qwen-like or integration-candidate source checkpoints after the baseline transfer check.
