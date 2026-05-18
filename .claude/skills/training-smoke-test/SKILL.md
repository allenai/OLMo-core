---
name: training-smoke-test
description: >
  This skill should be used when the user asks to "create a verification script",
  "write a test training run", "make a quick training job", "verify my change with a
  short run", "launch a smoke test", or wants a small Beaker job to validate that a
  feature works end-to-end on GPUs. Also triggers when the user mentions creating a
  modified 190M script to test a specific behavior.
---

# Training job smoke test

Create short, single-node training scripts that exercise a specific feature in a
real GPU run. These scripts are derived from the 190M base config
(`src/scripts/train/OLMo3/OLMo-3-190M.py`) and train for only ~20 steps with an
eval callback to confirm the feature works.

## When to use

- After implementing a new training feature (parallelism mode, eval type, data
  pipeline change) that needs GPU validation beyond unit tests.
- When a CI test cannot cover the behavior (e.g., multi-GPU distributed features).
- To produce a quick smoke-test script that can be launched on Beaker.

## Workflow

### 1. Identify what to verify

Determine the feature under test and what "success" looks like. Examples:

| Feature | Success signal |
|---------|---------------|
| CP perplexity evals | LM evaluator reports CE loss and PPL without error |
| TP training | Training completes 20 steps, loss decreases |
| New data mix | Data loader produces batches without errors |
| New attention backend | Training completes 20 steps with the new backend, no kernel errors |

### 2. Copy and modify the 190M base script

Start from `src/scripts/train/OLMo3/OLMo-3-190M.py` and place the new script in
`src/scripts/train/smoketests/`.

Key modifications:

- **File name**: `<feature>-test.py` in `src/scripts/train/smoketests/`.
- **Docstring**: Describe what is being verified and how to run it.
- **Duration**: `Duration.steps(20)` — just enough to confirm the feature works.
- **Batch size**: `SEQ_LENGTH * 16` — small but enough for distributed training.
- **Metrics/cancel interval**: 5 steps (frequent enough to catch issues early).
- **Remove unused features**: Drop `Float8Config`, `InstanceFilterConfig`,
  `CheckpointerCallback`, `FOR_BENCHMARKING`, `CHINCHILLA_MULTIPLE`,
  `estimate_lr()` none of these are needed for a quick verification.
- **WandB**: Keep but set `enabled=False`.

### 3. Add the feature-specific config

Depending on what is being verified, add the relevant config. Common patterns:

**Context Parallelism (CP):**
```python
from olmo_core.train.train_module import TransformerContextParallelConfig

# In train_module_config:
cp_config=TransformerContextParallelConfig.ulysses(degree=2),
```

**Tensor Parallelism (TP):**
```python
from olmo_core.train.train_module import TransformerTensorParallelConfig

# In train_module_config:
tp_config=TransformerTensorParallelConfig(degree=2),
```

**New attention backend:**
```python
from olmo_core.nn.attention import AttentionBackendName

# In model_config — override the backend selection:
model_config = TransformerConfig.olmo3_190M(
    vocab_size=tokenizer_config.padded_vocab_size(),
    attn_backend=AttentionBackendName.flash_4,  # or .te, .torch, etc.
)
```
Available backends: `torch`, `flash_2`, `flash_3`, `flash_4`, `te`.
The base script selects between `flash_2` and `flash_3` based on GPU type;
a verification script can hardcode a specific backend to test it.


**PPL evals (LMEvaluator):**
```python
from olmo_core.data import NumpyPaddedFSLDatasetConfig
from olmo_core.train.callbacks import LMEvaluatorCallbackConfig

# In trainer_config:
.with_callback(
    "lm_evaluator",
    LMEvaluatorCallbackConfig(
        eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
            DataMix.v3_small_ppl_validation,
            mix_base_dir=get_root_dir(cli_context.cluster),
            sequence_length=SEQ_LENGTH,
            tokenizer=tokenizer_config,
            work_dir=work_dir,
        ),
        eval_interval=10,
    ),
)
```


**Full recommended evals (PPL + downstream):**
```python
trainer_config = trainer_config.with_recommended_evals(
    tokenizer_config, SEQ_LENGTH, cli_context.cluster, task_set="fast"
)
```
Note: downstream evals require full logits and are incompatible with CP or TP.

### 4. Launch and verify

Before launching, **ask the user what priority** to use for the Beaker job.
The options are: `low`, `normal`, `high`, `urgent`. Not all workspaces support
all priorities (e.g., `ai2/OLMo_3` caps at `normal`).

```bash
# Dry run first to check the config renders:
python src/scripts/train/smoketests/<feature>-test.py \
    dry_run test-<feature> ai2/jupiter

# Launch on Beaker:
python src/scripts/train/smoketests/<feature>-test.py \
    launch test-<feature> ai2/jupiter \
    --launch.priority=<priority> \
    --launch.follow=false
```

Always launch with `--launch.follow=false` (or set `--launch.launch_timeout=<seconds>`
to cap how long to wait). This avoids blocking the terminal indefinitely.

By default the `launch` command follows the job logs in the terminal until
completion, which requires `step_soft_timeout` and blocks the session. Using
`follow=false` lets the job run asynchronously.

After launching, **report the Beaker experiment link** to the user so they can
monitor the job.

To reconnect to a running job later:
```bash
gantry follow <experiment-id>
```

### 5. Confirm success

Check the Beaker logs for:
- Training steps completing without errors.
- Throughput / MFU at expected levels.
- Eval metrics being reported (CE loss, PPL for LM evals).
- No NCCL or distributed errors.

## Naming convention

Scripts live in `src/scripts/train/smoketests/` and are named `<feature-slug>-test.py`.

Examples:
- `src/scripts/train/smoketests/cp-ppl-eval-test.py`
- `src/scripts/train/smoketests/tp-training-test.py`
- `src/scripts/train/smoketests/cp-tp-combined-test.py`
- `src/scripts/train/smoketests/new-data-mix-test.py`
- `src/scripts/train/smoketests/flash4-attn-test.py`
