# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Added `DownstreamEvaluatorCallbackConfig` class for running in-loop downstream eval via [OLMo-in-loop-evals](https://github.com/allenai/OLMo-in-loop-evals).

### Fixed

- Made GCS client more robust by automatically retrying timeout errors for most operations.

## [v1.5.0](https://github.com/allenai/OLMo-core/releases/tag/v1.5.0) - 2024-10-23

### Added

- Added Google Cloud support for `list_directory()` and `clear_directory()`.
- Added `CometCallback` for logging training runs to Comet.ml.
- Added `DataMixBase` class, to allow extending to new data mix groups.
- Added support for MoE-based models.
- Added method `DataLoaderBase.get_mock_batch()`.
- Trainer now starts with a dry-run of a fake batch created by `DataLoaderBase.get_mock_batch()`.
- Added `Callback.pre_backward()`, `.pre_eval_batch()`, and `.post_eval_batch()` methods.
- Added `Trainer.model_forward()`, `.get_losses()`, and `.eval_batch()` methods.
- Added a new `TransformerActivationCheckpointingMode`, "selected_ops" (requires torch 2.5 or newer).

### Changed

- `BeakerLaunchConfig.setup_steps` should now include steps to clone your repo (which it will by default). This change allows support for private repos.

### Fixed

- `prepare_cli_environment()` now calls `add_cached_path_clients()`.
- Removed an unnecessary host-device sync.

## [v1.4.0](https://github.com/allenai/OLMo-core/releases/tag/v1.4.0) - 2024-10-02

### Changed

- Updated default layer norm epsilon for OLMo models from `1e-5` to `1e-6` to match latest model.
- Renamed `FSLDataLoader` to `NumpyFSLDataLoader`.
- Renamed `VSLDataLoader` to `NumpyVSLDataLoader`.
- The trainer now takes a `data_loader: DataLoaderBase` instead of a `dataset: NumpyDatasetBase`.

## [v1.3.2](https://github.com/allenai/OLMo-core/releases/tag/v1.3.2) - 2024-09-27

### Added

- Added `Config.validate()`, `Config.replace()`, and `Config.apply()` methods.
- Trainer now records sequence length as a metric.

### Fixed

- Ensure additional cached-path clients are added in the process pool workers from some dataset preparation methods.
- Fixed `label_mask` tensor created by `NumpyPaddedFSLDataset`.
- Removed redundant warning messages about CUDA alloc retries.
- Fixed non-deterministic deadlock bug with async checkpointing.

## [v1.3.1](https://github.com/allenai/OLMo-core/releases/tag/v1.3.1) - 2024-09-26

### Fixed

- Fixed the name given to evaluator metrics logged.

## [v1.3.0](https://github.com/allenai/OLMo-core/releases/tag/v1.3.0) - 2024-09-26

### Added

- Added `torchao` to the Docker/Beaker images.
- Added support for `torchao` `float8` training via the `Float8HandlerCallback`.
- Added `Callback.post_attach()` method.

## [v1.2.0](https://github.com/allenai/OLMo-core/releases/tag/v1.2.0) - 2024-09-25

### Added

- Added support for wildcards in `OptimGroupOverride.params`.
- Added `NumpyPaddedFSLDataset` variant.
- Added `Evaluator` class and `EvaluatorCallback` for in-loop evals.
- Added `v3-small-ppl-validation` data mix.

### Fixed

- Fixed bug with data loader when using threading.

## [v1.1.0](https://github.com/allenai/OLMo-core/releases/tag/v1.1.0) - 2024-09-18

### Added

- Added support for changing train sequence length when loading a checkpoint.
- Added support for sequence length warm-up during training via the callback `SequenceLengthSchedulerCallback`.
- Added support for variable sequence length (VSL) datasets and VSL curriculums as introduced in ["Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum"](https://arxiv.org/pdf/2405.13226).
- Added `Lion` and `SkipStepLion` optimizers.
- Added `init_seed` argument to `Transformer` and `TransformerConfig`.

### Changed

- Renamed `MemMapDataset` to `NumpyFSLDataset`.
- Batch size is now specified in tokens, not instances.

## [v1.0.6](https://github.com/allenai/OLMo-core/releases/tag/v1.0.6) - 2024-09-05

### Added

- Added "selected_modules" transformer activation checkpointing mode.
- Added `OLMo-1B.py` official training script.
- Added `OLMo-13B.py` official training script.
- Added `Trainer.get_metric()`, `.get_loss()`, and `.get_zloss()` methods.
- Added `io.copy_file()` function.
- Added `ProfilerCallback` for profiling/tracing the training loop with PyTorch `profiler` module.
- Added an "L2 norm" metric reduce type.

### Fixed

- Made reducing metrics more numerically stable with large world sizes.

## [v1.0.5](https://github.com/allenai/OLMo-core/releases/tag/v1.0.5) - 2024-09-03

### Fixed

- Fixed bug with checkpointer callback searching for existing ephemeral checkpoints when the checkpoint folder doesn't exist.
- Checkpointer callback won't collect existing ephemeral checkpoints that were saved after the checkpoint that was loaded from.

## [v1.0.4](https://github.com/allenai/OLMo-core/releases/tag/v1.0.4) - 2024-09-01

### Added

- Added `Trainer.save_checkpoint()` and `Trainer.save_checkpoint_async()` methods.
- Added `Callback.post_checkpoint_saved()` and `Callback.post_checkpoint_loaded()` methods.
- Added `ConfigSaverCallback`.
- Added `MemMapDataset.fingerprint` property.

### Changed

- The `work_dir` argument to `TrainerConfig` now defaults to `save_folder` is `save_folder` is a local path, otherwise a temporary directory with the same name as the basename of the `save_folder`.
- The `seed` argument to `prepare_training_environment()` is now optional.

### Fixed

- Fixed setting the right env vars for single node training on Jupiter.

## [v1.0.3](https://github.com/allenai/OLMo-core/releases/tag/v1.0.3) - 2024-08-30

### Added

- Add `Trainer.hard_stop` field.
- The trainer now catches `SIGTERM` and marks the run as canceled.
- Added `CheckpointerCallback.remove` strategy for configuring which old checkpoints found in the save folder are removed.
- Added `ReorderedNormTransformerBlock` implementation.
- Added `WandBCallback.notes` field.

### Fixed

- Fixed bug with how command arguments were expanded by `BeakerLaunchConfig`.

## [v1.0.2](https://github.com/allenai/OLMo-core/releases/tag/v1.0.2) - 2024-08-29

### Added

- Added support for unsharding model state into `safetensors` format with `olmo_core.distributed.checkpoint.unshard_checkpoint(..., use_safetensors=True)`.
- Added `data.TokenizerConfig` config class and `data.TokenizerName` enumeration.
- Added data mixes with `data.DataMix` API.
- Added `block_idx` attribute to the `TransformerBlock` class.
- Added `init_method` option to `Transformer` for controlling how the weights are initialized.

### Fixed

- Fixed `list_directory` for remote folders.

### Changed

- Callbacks now have to have a name assigned.

## [v1.0.1](https://github.com/allenai/OLMo-core/releases/tag/v1.0.1) - 2024-08-26

### Fixed

- Fixed a bug with resetting the initial LR in optimizers after a loading a checkpoint.

## [v1.0.0](https://github.com/allenai/OLMo-core/releases/tag/v1.0.0) - 2024-08-26

### Added

- Ported, refactored, and optimized the modeling and training from the OLMo repo while fixing several bugs. Introduces a new highly efficient yet customizable trainer and a standard API for launching jobs directly to Beaker from a Python script.

## [v0.1.0](https://github.com/allenai/OLMo-core/releases/tag/v0.1.0) - 2024-06-11

### Added

- Initial release.
