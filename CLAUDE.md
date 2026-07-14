# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OLMo-core is AI2's training library for the Open Language Model (OLMo) series. It provides modular components for transformer architectures, distributed training, data loading, and evaluation.

## Commands

```bash
# Install (development)
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

### Document-chunked / landmark attention: REPAIR THE BASE CHECKPOINT FIRST

Qwen3 never trains the embedding rows for the reserved marker tokens that the document-chunked and
landmark data paths are built on (`<|box_start|>`/`<|box_end|>`, plus the landmark/pad ids past the
real vocab). Their embeddings are **bit-identical** (cosine similarity 1.0000), so the model cannot
distinguish an "open document" marker from a "close document" one. Marker-dense runs then train to
chance and the failure looks exactly like a modeling result.

**Before any document-chunked / landmark training from a fresh base, run
`src/scripts/data/fix_marker_embeddings.py` on the checkpoint and train from the repaired copy.** The
tokenized shards are correct — do NOT rebuild data. See `records/document-chunked-marker-embeddings.md` for
the full diagnosis, the one-line check for whether a base is affected, and the validation numbers.

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

`src/scripts/train/` is also the hub for ALL of our own training code, organized by experiment
family (`cpt/`, `sft_5task/`, `sft_docchunk/`, `attn_explore/`, ...) — see the index in
`src/scripts/train/README.md` and each family's README. A gitignored repo-root symlink `training/`
points there for quick inspection. New training scripts go in the matching family folder (or a new
one with a README), never loose in `sft/` (upstream-only).

```bash
python src/scripts/train/OLMo2-1B.py dry_run test-run ai2/titan-cirrascale
python src/scripts/train/OLMo2-1B.py launch olmo2-1b-test ai2/jupiter-cirrascale-2 --launch.num_nodes=4
```

## Deprecated scripts

Retired scripts live in `deprecated/` (mirroring their old repo-relative path) with a README row
saying why and what replaces them — see `deprecated/README.md` for the convention. Never use or
reference anything in there; when you retire a script, `git mv` it in and log it.

## Records (standalone writeups)

Experiment diagnoses, task briefs, and setup notes live in `records/` (see its README for the
index). New writeups of this kind go there — the repo root keeps only
README/CHANGELOG/CONTRIBUTING/CLAUDE.md/local_cluster.md.

## Local (Berkeley) cluster

We also train/eval on Berkeley slurm H200 nodes without AI2 infra (no weka, no Beaker). The full
pipeline — nodes/QOS, the NFS-vs-`/data` rule, env setup, the weka-free training recipe, and the
known traps — is documented in `local_cluster.md`. Read it before touching any `*local*.sbatch`
launcher or debugging a hung/crashed local run.

## Docker and Beaker Launch

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

**Job priority — ALWAYS `urgent`, never lower.** Every Beaker/gantry job (training, eval, data build) must launch at `priority="urgent"` (or `immediate`); never `normal`/`high`/`low`. Lower-priority jobs pend behind capacity; `urgent` preempts to get nodes. Set `BeakerLaunchConfig.priority = "urgent"`, pass `--priority urgent` to launchers, or bump a live job with `beaker job update-priority <job-id> urgent`. If you add or edit a launcher, make `urgent` its default.

## Testing

- Tests in `src/test/` mirror the source structure.
- Name individual test functions `test_*` and prefer `pytest.mark.parametrize` to cover multiple inputs or configurations without duplicating code.
- GPU tests use `@pytest.mark.gpu` and are skipped without a GPU.
- Distributed tests use helpers in `src/olmo_core/testing/distributed.py`.

## Reporting experimental results

**`n` means CORPUS SIZE — never eval-set size.** In this project `n` is reserved for the number of
documents/claims in an example's context (`n=20`, `n=100`, the `..._n100_k3` files, `--ndocs`). It is
a property of the *task*. Using the same letter for how many examples an eval ran on has already
caused real confusion, so:

- For eval-set size write **`eval_size`** (or spell it out: "488 eval examples"). Never `n=488`.
- New eval scripts must emit the field `eval_size` in their result JSON. Older results wrote this as
  `n`, which is exactly the collision being retired — when reading them, translate to `eval_size`
  rather than propagating the old name.
- The same applies in prose, tables, run names, and anything ingested into results-hub.

**Eval sets should have at least 500 examples, and a smaller one must be flagged.** If a number came
from fewer than 500 eval examples, say so *inline, next to the number* — give the size and its error
bar, e.g. `f1=0.83  ⚠ eval_size=100 only (±0.038)`. Never present a sub-500 result bare; a small eval
inflates noise into apparent findings. (The contradiction held-out set is 488 examples, which is the
entire file and is accepted as-is.)

**Quote a resolution, not three decimals.** On a right/wrong-graded eval, the binomial standard error
is ±0.021 at f1≈0.70 and ±0.010 at f1≈0.95 for 488 examples. Before calling a difference real, check
it against that — and remember eval noise is only half the story, since run-to-run seed variation adds
more.
