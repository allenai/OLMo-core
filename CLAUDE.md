@AGENTS.md

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OLMo-core is AI2's training library for the Open Language Model (OLMo) series. It provides modular components for transformer architectures, distributed training, data loading, and evaluation.

This repository is a hard fork maintained by `edu-llm`. It does not sync with `allenai/OLMo-core`. GPU work runs on the eduLLM platform, not on Beaker — see [Running on GPUs](#running-on-gpus) before launching anything.

## Commands

```bash
# Install (development, for local tests and linting only — not how a GPU run starts)
pip install -e '.[all]'

# Run all tests (GPU tests auto-skip without GPU)
pytest -v src/

# Run a specific test file
pytest -v src/test/path/to/test_file.py

# Filter to specific tests by keywords
pytest -v src/test/path/to/test_file.py -k 'keyword'

# Auto-format code
make style

# Check formatting, lint, and types
make checks          # all three at once
make style-check     # isort + black
make lint-check      # ruff
make type-check      # mypy
```

## Running on GPUs

Every GPU run goes through the eduLLM platform's `edullm` CLI. Beaker, `ai2/jupiter` and the other `ai2/*` clusters named throughout this repository belong to AI2 and are unreachable from here.

```bash
# Install the CLI. Re-run to upgrade; `uv tool upgrade` does not work on a git install.
uv tool install --force git+https://github.com/edu-llm/platform

# Price a run and list every refusal. No network, no queue, nothing dispatched.
edullm check --json --experiment <slug> --dataset <release-or-none>

# Same checks, then dispatch. Keep the run id it prints.
edullm submit --experiment <slug> --dataset <release-or-none>
edullm status --json <run-id>
```

- `check` writes a first `.edullm/run.yaml` if there is none, and that file holds the command. Edit it, commit it, push to a branch named `edullm/<something>`.
- Read `check --json` on stdout and branch on the exit code: 0 stands, 1 is refused on the merits, 2 means the command was wrong, 3 means retry. Match on the `code` of each refusal, never on its prose.
- Never write a script that calls AWS. No `boto3`, no `aws` CLI. The credentials live in workflows and a laptop cannot get one.
- Never quote a price, a runtime bound or an approver count from this file. Read them out of `check --json`.

### bfloat16 is set in code, and the platform cannot see it

`.edullm/train_on_corpus.py` builds its data-parallel config in `bfloat16` unless told otherwise. The platform's precision guard reads the words of the command, so a command that merely runs that program carries no bfloat16 token and is accepted on any shape — including a T4, which is Turing and has no bfloat16 in the hardware at all. `torch.cuda.is_bf16_supported()` returns true there, so nothing warns you.

Put the dtype in the command. A run that names it is refused for free at `check` time instead of dying on the first kernel that needs the format, after the machine has been billed.

```bash
# In .edullm/run.yaml, either spelling is read by the guard:
python .edullm/train_on_corpus.py "$EDULLM_RUN_ID" train_module.dp_config.param_dtype=bfloat16
python .edullm/train_on_corpus.py "$EDULLM_RUN_ID" --param-dtype bfloat16   # or float32 on a T4
```

`train_on_corpus.py` also checks its own built config against the visible devices and exits 73 in the first seconds when they cannot do the requested format. That check only protects commits that carry it, so a research branch that has not merged it has nothing in front of the failure but the command text.

### What your coding agent reads about the platform

There is nothing to install. The first line of this file imports `AGENTS.md`, which carries the binary, its verbs, the exit codes, `--json` and the rule against calling AWS directly. That is what an agent needs for anything on the platform, and it is read every session without being invoked. It is held byte-identical to [edu-llm/platform](https://github.com/edu-llm/platform) by a workflow that runs daily, so do not edit the marked block: the edit is reverted and turns that workflow red in the meantime. Everything outside the markers is this repository's own.

The curl line that used to be here has gone, along with the gitignore entry under it. A copy each person installs for themselves is a copy nobody can see, nobody updates and nothing compares, and the observed result was that most people never installed one at all and their agents wrote AWS calls.

There is deliberately no skill here for submitting a run. `AGENTS.md` and the `detail` printed beside every refusal are the whole of it. The platform's one skill, `registering-a-repository`, is not here either and that is not an omission: it fires when a codebase is *not* on the platform, so this repository is one of the places it can never be needed. It installs once per person at user level, and [`skills/README.md`](https://github.com/edu-llm/platform/blob/main/skills/README.md) says where each host reads one from.

## Code Style

- Line length: 100
- Formatting: `isort` (profile=black) + `black`
- Linting: `ruff` (ignores F403, F405, E501; F401 ignored in `__init__.py`)
- Type checking: `mypy` with `ignore_missing_imports = true`

## Docstrings

- Docstrings should be included on all public classes, methods, and functions.
- We use Sphinx to automatically build API docs by pulling from those docstrings.
- The syntax of the docstrings is a superset of reStructuredText with additional Sphinx-specific syntax for things like:
  - Cross-document links, e.g.:
    ```
    :class:`foo.Foo`  <- links to the class named 'Foo' in the module 'foo'
    :mod:`foo`        <- links to the module named 'foo'
    :func:`foo.bar`   <- links to the function named 'bar' in the module named 'foo'
    ```
  - Documenting parameters (`:param ...:`), return values (`:returns:`), or expected exceptions (`:raises ...:`).

Here's a toy example for a function:

```python
def read_file(path: str) -> str:
    """
    Read a file from disk.

    :param path: The path to the file.

    :returns: The contents of the file.

    :raises FileNotFoundError: If the file doesn't exist.
    """
    pass
```

## Architecture

### Configuration System (`src/olmo_core/config.py`)

Everything is configured via `@dataclass` classes inheriting from `Config`. This is the central design pattern:
- Configs support YAML/JSON serialization, command-line overrides via dot notation (`--train_module.optim.lr=6e-3`), and `merge()` with dotlists.
- The `Registrable` mixin (from `dataclass-extensions`) enables polymorphic config fields — a base config class can resolve to different subclasses at runtime based on a `type` field. Used in optimizers, schedulers, attention backends, and data loaders.
- Nested configs compose modularly: `TrainerConfig` contains `CheckpointerConfig`, `OptimConfig`, etc.

### Training Pipeline (`src/olmo_core/train/`)

- `Trainer` / `TrainerConfig`: Core training loop with checkpointing, evaluation, and an extensible callback system (`callbacks/`).
- `TrainModule`: Wraps the model with forward/backward logic and optimizer. The main concrete implementation is `TransformerTrainModule` / `TransformerTrainModuleConfig`, which handles parallelism setup (DP, TP, PP, CP, EP configs all live here).

### Model Architecture (`src/olmo_core/nn/`)

- `transformer/`: Core transformer with configurable blocks. `TransformerConfig` has factory methods like `olmo2_32B()` for predefined architectures.
- `attention/`: Multi-head attention with backends (flash attention, ring attention, etc.).
- `moe/`: Mixture of Experts with expert parallelism.
- `feed_forward.py`, `layer_norm.py`, `rope.py`, `lm_head.py`: Standard components.

### Data Loading (`src/olmo_core/data/`)

- `NumpyDataset` variants: Memory-mapped numpy datasets for pre-tokenized data (`.npy` files).
- `composable/`: The preferred data loading API, built on a pipeline of `TokenSource` -> `InstanceSource` -> `ComposableDataLoader`. Sources can be sliced, sampled, mixed with ratios, and split for curriculum learning. Use `InstanceSource.visualize()` to inspect the source tree. See the module docstring in `src/olmo_core/data/composable/__init__.py` for detailed examples.
- `mixes/`: Predefined data mixture configs (dolma17, OLMoE-mix-0824, etc.) with paths to tokenized data by source and tokenizer.
- Training data is stored on AI2 infrastructure (Weka filesystem, GCS). For local development, use small validation sets or synthetic data.

### Distributed Training (`src/olmo_core/distributed/`)

- `parallel/`: Implementations of data (FSDP/HSDP/DDP), tensor, pipeline, context (ring attention), and expert parallelism. These can be combined for multi-dimensional parallelism.
- `checkpoint/`: Distributed checkpointing with various filesystem backends.

### Optimization (`src/olmo_core/optim/`)

- Optimizer configs (`AdamWConfig`, `SkipStepAdamWConfig`, `LionConfig`) and LR schedulers (`CosWithWarmup`, etc.).
- `SkipStepOptimizer`: Wrapper for gradient clipping with loss spike detection.

### Examples (src/examples)

Runnable, self-contained examples and reference scripts.

### Training Scripts

Two patterns exist:

**Official scripts** (`src/scripts/official/`): Use `ExperimentConfig` + `main()` from `src/olmo_core/script_utils.py`. Launched with `torchrun` or Beaker. These reproduce published model runs.

```bash
torchrun --nproc-per-node=8 src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py \
  --save-folder=/path/to/checkpoints
```

**Internal scripts** (`src/scripts/train/`): Use `prepare_cli_environment()` with commands (`launch`, `train`, `train_single`, `prep`, `dry_run`). See `template.py` for the starting point.

```bash
python src/scripts/train/OLMo2-1B.py dry_run test-run ai2/titan-cirrascale
python src/scripts/train/OLMo2-1B.py launch olmo2-1b-test ai2/jupiter-cirrascale-2 --launch.num_nodes=4
```

The cluster arguments above are AI2's and cannot be reached from this organization. The `launch` command needs Beaker; `dry_run` and `train_single` do not. See [Running on GPUs](#running-on-gpus) for how a run actually starts here.

## Docker and Beaker Launch (upstream AI2 — not usable here)

Everything in this section describes `allenai/OLMo-core`'s own launch path. The code is still in `src/olmo_core/launch/beaker.py` and still imports, so it is documented rather than deleted, but nothing in this organization can authenticate to Beaker or pull a Beaker image. Do not follow it, and do not offer it to a researcher. The eduLLM route is [Running on GPUs](#running-on-gpus).

The Docker image (`src/Dockerfile`) is a two-stage build: a `build` stage compiles GPU-specific dependencies (flash-attn, TransformerEngine, grouped_gemm, ring-flash-attn, etc.) on an NVIDIA CUDA devel image, and a `release` stage copies the conda environment into a lighter Ubuntu base with AWS CLI, Google Cloud SDK, and MLNX OFED drivers. The image contains all dependencies but *not* the OLMo-core package itself — source code is cloned at runtime.

```bash
# Build locally (versions configured in Makefile)
make docker-image

# Push to GHCR
make ghcr-image

# Create Beaker image
make beaker-image
```

**How launch works**: When a training script uses the `launch` command (or `BeakerLaunchConfig.launch()`), it creates a Gantry recipe that:
1. Starts a container from a pre-built Beaker image (default: `OLMoCoreBeakerImage.stable` in `src/olmo_core/launch/beaker.py`)
2. Clones the git repo at the current commit into the container (requires clean working tree unless `allow_dirty=True`)
3. Installs the package from source (`pip install -e .`)
4. Runs the training command, optionally wrapped with `torchrun` for multi-GPU/multi-node

Pre-built images are listed in the `OLMoCoreBeakerImage` enum in `src/olmo_core/launch/beaker.py`, tagged by torch version and CUDA version (e.g., `tch2100cu128`). The `stable` image tracks the default torch/CUDA versions. When updating default images, also update `.github/workflows/main.yml`.

`BeakerLaunchConfig` also supports `pre_setup` and `post_setup` hooks for running commands before/after the package install step, Weka bucket mounts, and multi-node settings (replicas, leader selection, host networking).

## Testing

- Tests in `src/test/` mirror the source structure.
- Name individual test functions `test_*` and prefer `pytest.mark.parametrize` to cover multiple inputs or configurations without duplicating code.
- GPU tests use `@pytest.mark.gpu` and are skipped without a GPU.
- Distributed tests use helpers in `src/olmo_core/testing/distributed.py`.
