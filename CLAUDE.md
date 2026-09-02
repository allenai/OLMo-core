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
- **Our data-generation pipeline** (task JSONL → tokenized SFT shards → weka/local staging) is
  mapped in `src/scripts/data/README.md`: layer 1 = `src/corpus_reasoning/data/` (task
  generators), layer 2 = `src/scripts/data/` (converters + gantry/sbatch staging).

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

⚠ **Fixing the marker cosine is NOT enough — the marker NORM matters too.** The first version of
`fix_marker_embeddings.py` made the markers mutually distinguishable but left them at ~1/3.6 the norm
of a real token, which RMSNorm amplifies into full-strength noise. On the leak-free (label-inside-chunk)
shards this flatlines training at CE ≈ 0.79 for **every** mask, including plain causal — an
unrestricted model cannot even memorize the data, so it reads as "the mask is too restrictive" when it
is not. The script now seeds each marker from a real trained delimiter row (`«`/`»`/…) and asserts the
norm is in-distribution. Any base repaired before 2026-07-14 is affected: re-run the script.
See `records/n100-chunked-marker-position-bug.md`.

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

ALL of our own training code lives in `src/scripts/train/memexpress/`, organized by experiment
family (`cpt/`, `sft_5task/`, `sft_docchunk/`, `attn_explore/`, ...) — see the index in
`src/scripts/train/memexpress/README.md` and each family's README. Two shortcuts point there:
`scripts/memexpress` (tracked symlink) and the gitignored repo-root `training/`. New training
scripts go in the matching family folder (or a new one with a README), never loose in
`src/scripts/train/sft/` (upstream-only).

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
README/CHANGELOG/CONTRIBUTING/CLAUDE.md/local_cluster.md/lambda_cluster.md/beaker.md.

## Local (Berkeley) cluster

We also train/eval on Berkeley slurm H200 nodes without AI2 infra (no weka, no Beaker). The full
pipeline — nodes/QOS, the NFS-vs-`/data` rule, env setup, the weka-free training recipe, and the
known traps — is documented in `local_cluster.md`. Read it before touching any `*local*.sbatch`
launcher or debugging a hung/crashed local run.

**`/scratch` IS NFS — it is not a fast local disk.** Both `/accounts` (home, this repo) and
`/scratch` are shared NFS: `/scratch` writes at roughly 5 MB/s, and importing from an NFS conda env
costs a ~5–60 s tax per process. The ONLY node-local fast storage is the target node's `/data`
(and `/tmp`). So moving job I/O from `/accounts` to `/scratch` is **not** a mitigation for an NFS
problem — it is the same class of storage, and it will still deadlock concurrent readers/writers in
`nfs_wait_bit_killable`. Job logs, work dirs, JIT/index caches (triton, flashinfer, pyserini) and
checkpoints all belong on `/data`; `/scratch` is fine for code and small artifacts you read once.

**`/net/<node>/...` reads any node's local disk from anywhere — for INSPECTION ONLY.** Every
compute node is exported at `/net/<hostname>`, so `/net/horton/data/prasann/...` reads horton's
node-local `/data` from the login node or from another node. Nodes today: `balrog`, `cubbins`,
`feanor`, `horton`, `lorax`, `mcfuzz`, `mooney`, `rainbowquartz`, `saruman`, `shadowfax`, `smaug`,
`smokyquartz`, `sneetches`, `sunstone`, `thidwick`.

This is genuinely useful for auditing — checking what data or checkpoints exist on a node, reading a
job log, confirming a file's schema — without an `srun`. Use it that way and keep it light.

**It does NOT make `/data` remotely fast, and it is not a way to share data between nodes.** `/net`
*is* the NFS layer this section is about: a `/net` path has every property that makes `/scratch`
slow, so pointing job I/O, an interpreter, a checkpoint read or a cache directory at one recreates
exactly the `nfs_wait_bit_killable` deadlock described above. A job running on horton must use
`/data/...`, never `/net/horton/data/...` — the same bytes, one over the local disk and one over a
~5 MB/s link. Recursive `find` across `/net` is also slow enough to hit command timeouts; go
straight to the directory you want.

**The INTERPRETER itself must be node-local for any torch/vLLM job.** The "~5–60 s import tax"
above is the *best* case (small scripts). A vLLM or torch job launched with
`/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python` has to page multiple GB of
shared objects over a ~5 MB/s link and parks in `D` state / `nfs_wait_bit_killable` at ~0% CPU for
many minutes — it looks like a hung GPU or a slow model load, and it is neither. Use the
node-local venv **`/data/prasann/ctc_vllm_venv/bin/python`** (vllm 0.25.1 / torch 2.11.0+cu130)
for vLLM work, together with `export CUDA_HOME=/usr/local/cuda-12.8` and
`export PATH=$CUDA_HOME/bin:$PATH` for the GDN JIT. Diagnose with
`ps -o stat,wchan,%cpu` — `Dl` + `nfs_wait_bit_killabl` + ~0% CPU is this bug, not a compile.

### Loading an olmo-exported Qwen3.5 in vLLM: use the serving copy

`export_olmo_to_hf.py` emits a **text-only** checkpoint (`model_type: qwen3_5_text`, flat config,
no `vision_config`). vLLM 0.25.1 resolves *any* `Qwen3_5*` architecture to a **multimodal** class
whose `__init__` reads `config.vision_config`, so pointing vLLM at the raw export — even with
`hf_overrides={"architectures": ["Qwen3_5ForCausalLM"]}` — dies at model construction with
`AttributeError: 'Qwen3_5TextConfig' object has no attribute 'vision_config'`, before any memory
is touched.

**FIRST, look for an existing serving copy — do not rebuild one.** Prebuilt, already-validated
copies live at `/data/prasann/ctc_suite/vllm_serving_4b{,_v2,_v3}/<ckpt-name>/` on horton (`_v3`
is current). Point vLLM straight at that directory.

Building one from scratch takes **three** scripts in `debug/ctc_vllm_validation/`, not one — miss
any and the load dies:
1. `make_vllm_serving_copy.py` — wrapper `config.json` (base VL config with our `text_config`).
2. `make_vl_weights.py` — key rename `model.*` → `model.language_model.*`. Without it:
   `ValueError: There is no module or parameter named 'model' in Qwen3_5ForConditionalGeneration`.
3. `add_dummy_visual.py` — ~297 synthesized `visual.*` params. Without them vLLM's
   `track_weights_loading` errors on the uninitialized vision tower.

Sanity-check a serving dir before launching: it should have **~426 `model.language_model.*` and
~297 `visual.*`** keys, and `vision_config` in `config.json`.

The arch override and `limit_mm_per_prompt={"image": 0, "video": 0}` are *part* of the recipe, not
the whole of it — see [[qwen35-4b-vllm-load-recipe]] for all seven pieces. Copying two lines out of
`run_vllm_eval.py` without the serving copy is a repeat failure; that driver is always pointed at
a serving copy someone else already built.

**`hf_overrides` REPLACES a nested sub-config, it does not merge.** Passing
`hf_overrides={"text_config": {"rope_scaling": ...}}` wipes every other `text_config` field and
fails pydantic validation with *"text_config … does not have `num_attention_heads`"*. To change
RoPE/YaRN or any other nested field, patch `config.json` on disk in a symlinked copy (see
`debug/ctx_ceiling_4b/make_yarn_copy.py`) rather than fighting override semantics.

## Lambda cluster

A third pool: a **separate A100 SLURM cluster reached via `ssh lambda`**, sharing no filesystem, no
git remote, and no internet with the Berkeley cluster. Use it to drain a queue of independent
training runs when the Berkeley per-user GPU quota is saturated — its GPUs are often idle while
jsteinhardt is contended.

**Fair-share rule (Lambda only): cap your *preempting* footprint at two nodes — at most 8 GPUs on
`preemptive_high` and 8 on `preemptive` at any time. Take anything beyond that at `normal`, which
can't preempt and is itself preemptible.** The cluster is shared with one other regular user, and
this is what guarantees both of you can always get 2 nodes. Do NOT carry over the Berkeley
"default to highest QOS and preempt freely" directive.

**`lambda_cluster.md` (repo root)** documents access, the QOS table and that policy, the per-user
1.30T `/accounts` quota (the usual blocker) vs. node-local disk, the rsync staging recipe,
`run_ctc_lambda.sbatch` knobs, and the trap index. Read it before touching any `*lambda*.sbatch`
launcher. Two traps worth knowing up front: the launcher header hardcodes
`--gres=gpu:A100:8` so a direct execution eats your whole GPU quota, and **`sbatch` routinely
reports `FAILED` on runs that fully succeeded** — verify via the loss curve and checkpoint, never
the exit code.

## Docker and Beaker Launch

**Practical how-to + hard-won lessons for launching anything on Beaker from this machine live in
`beaker.md` (repo root)** — launch/gantry templates, monitoring/cancel commands that actually
work, the weka-vs-S3 staging two-step, and the traps index. This section covers the underlying
mechanics.

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

**Job priority — ALWAYS `urgent`, never lower.** Every Beaker/gantry job (training, eval, data build) must launch at `priority="urgent"` — exactly `urgent`: never `normal`/`high`/`low`, and not `immediate` either. Lower-priority jobs pend behind capacity; `urgent` preempts to get nodes. Set `BeakerLaunchConfig.priority = "urgent"`, pass `--priority urgent` to launchers, or bump a live job with `beaker job update-priority <job-id> urgent`. If you add or edit a launcher, make `urgent` its default.

**Allocation — ALWAYS unallocated.** Beaker capacity is split into per-budget *allocations*
(`beaker allocation list ai2/jupiter-cirrascale-2 --budget ai2/oe-other`: 14%/168 slots on
jupiter, 15%/88 on ceres, all of it granted to the `ai2/oe-scaling` workspace group; nothing on
saturn/neptune). A job is *allocated* only when its workspace is in that group, and allocated
jobs consume the team's guaranteed quota. Our jobs must stay **unallocated + urgent**: launch
from workspace `ai2/flex2` under budget `ai2/oe-other` and never from an oe-scaling workspace.
There is no per-job flag; the workspace decides. Verify with
`beaker report gpu-usage --organizations ai2 --users prasanns --since 7d -g allocated,cluster`
(non-admins must pass the org filter). Status 2026-09-02: 100% of our GPU hours were unallocated.

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
