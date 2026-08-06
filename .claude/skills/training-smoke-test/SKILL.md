---
name: training-smoke-test
description: >
  This skill should be used when the user asks to "create a verification script",
  "write a test training run", "make a quick training job", "verify my change with a
  short run", "launch a smoke test", or wants a small GPU job to validate that a
  feature works end-to-end. Also triggers when the user mentions creating a
  modified 190M script to test a specific behavior.
---

# Training job smoke test

Create short, single-node training scripts that exercise a specific feature in a
real GPU run. These scripts are derived from the 190M base config
(`src/scripts/train/OLMo3/OLMo-3-190M.py`) and train for only ~20 steps with an
eval callback to confirm the feature works.

Runs launch on the eduLLM platform through `edullm`. Beaker and the `ai2/*`
clusters named elsewhere in this repository belong to AI2 and cannot be reached
from here — see `CLAUDE.md`, "Running on GPUs".

## When to use

- After implementing a new training feature (parallelism mode, eval type, data
  pipeline change) that needs GPU validation beyond unit tests.
- When a CI test cannot cover the behavior (e.g., multi-GPU distributed features).
- To produce a quick smoke-test script that can be submitted to the platform.

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

### 4. Point the run spec at the script

`.edullm/run.yaml` holds the command. `edullm check` writes a first one if the
repository has none. Edit `command` to name the smoke-test script, and say the
dtype explicitly:

```yaml
command: >-
  bash -lc 'python src/scripts/train/smoketests/<feature>-test.py train_single
  smoke-<feature> --train_module.dp_config.param_dtype=bfloat16'
```

**Name the dtype even when it is the default.** The platform's precision guard
reads the words of the command and cannot see a dtype set in code, so a command
that omits it is accepted on a T4 — Turing, which has no bfloat16 in the hardware
— and dies on the first kernel that needs the format after the machine is billed.
A command that names it is refused at `check` time for free. Use `float32` if you
mean to stay on a T4.

Commit the spec and the script, and push to a branch named `edullm/<something>`.
That push is what builds the image.

### 5. Check, then submit

```bash
# Prices the run and lists every refusal. No network, no queue, nothing dispatched.
edullm check --json --experiment smoke-<feature> --dataset none

# Same checks, then dispatch.
edullm submit --experiment smoke-<feature> --dataset none
```

Read stdout on its own and branch on the exit code before anything else: 0 stands,
1 is refused on the merits, 2 means the command was wrong, 3 is worth retrying.
Match on each refusal's `code`, never on its prose.

`--dataset none` is right for a smoke test that reads no registered corpus, and it
is a different answer from omitting the flag. Use `--compute` to override the shape
in the spec; `check` prices whichever wins.

**Tell the user the total and whether a person has to release it** before you
submit. Read `maximum_compute_cost_usd` and `approval_class` out of
`check --json` — never from memory and never from this file.

### 6. Confirm success

```bash
edullm status --json <run-id>   # answers from GitHub, costs nothing, safe in a loop
edullm logs <run-id>            # the last lines the run printed, costs a workflow
```

Check the logs for:
- Training steps completing without errors.
- Throughput / MFU at expected levels.
- Eval metrics being reported (CE loss, PPL for LM evals).
- No NCCL or distributed errors.

Nothing you can run releases a submission waiting at an approval gate. A person has to.

## Naming convention

Scripts live in `src/scripts/train/smoketests/` and are named `<feature-slug>-test.py`.

Examples:
- `src/scripts/train/smoketests/cp-ppl-eval-test.py`
- `src/scripts/train/smoketests/tp-training-test.py`
- `src/scripts/train/smoketests/cp-tp-combined-test.py`
- `src/scripts/train/smoketests/new-data-mix-test.py`
- `src/scripts/train/smoketests/flash4-attn-test.py`

## The full platform skill

This covers smoke tests. For the other verbs, every refusal code, and what a clean
check does not promise, install the platform's own skill — it is maintained there
and deliberately not copied into this repository.

```bash
mkdir -p .claude/skills/edullm-platform && curl -fsSL \
  https://raw.githubusercontent.com/edu-llm/platform/main/skills/edullm-platform/SKILL.md \
  -o .claude/skills/edullm-platform/SKILL.md
```

Lines for Cursor and Codex are in
[`skills/README.md`](https://github.com/edu-llm/platform/blob/main/skills/README.md).
