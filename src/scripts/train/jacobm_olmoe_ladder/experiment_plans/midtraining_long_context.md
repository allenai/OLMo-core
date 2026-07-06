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
| GPUs | `8` | One Titan node by default; smoke 2 and 4 GPUs first. |
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

Open decisions before a real sweep:

- Which source checkpoints are the first targets: baseline only, Qwen-like, or
  integration candidates?
- What token budget defines a useful midtraining LR search: full 100B, a shorter
  pilot, or a fixed number of optimizer steps?
- Whether to run in-loop evals during midtraining or keep the first sweep loss
  only.
- Whether midtraining LR should be centered on the pretraining best observed LR
  for the checkpoint family, or on the dense midtraining LR scale.
