# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Added option to set LR scheduler based on tokens instead of steps (e.g. `--train_module.scheduler.units=tokens`).
- Added a "packed" numpy FSL variant that packs documents into sequences using the best-fit-decreasing bin packing algorithm following the work from [Fewer Truncates Improve Language Modeling](https://arxiv.org/pdf/2404.10830).
- Added module `olmo_core.testing`.
- Added a "interleaved" numpy FSL variant that interleaves several documents into sequences following the work from [LongSkywork: A Training Recipe for Efficiently Extending Context Length in Large Language Models](https://arxiv.org/pdf/2406.00605).

### Changed

- Output of `LMHead` when `labels` is passed as input is now a 4-tuple instead of a 3-tuple, with `(logits, loss, ce_loss, z_loss)`, where `loss` is the combined loss (`ce_loss + z_loss`).

### Fixed

- Modify `TokenizerConfig.from_hf()` to fallback to tokenizer_config.json if config.json is not found.
- Fixed loading checkpoints with missing keys from transformer train modules using torch 2.7.
- Made MoE load balancing loss more robust.
- Fixed a bug with `ReorderedNormTransformerBlock` when using fine-grained FSDP wrapping and activation checkpointing together.
- Fixed an issue preventing tensor parallelism from working with `LMHead` when using the "fused_linear" loss implementation.
- Fixed a bug with `LMHead` when using "fused_linear" loss implementation where the `ce_loss` output included the `z_loss` added to it.
- Fixed training on single GPU when using a `SkipStepOptimizer`.
- Fixed the initialization of the `CosWithWarmupAndLinearDecay` learning rate scheduler


## [v2.1.0](https://github.com/allenai/OLMo-core/releases/tag/v2.1.0) - 2025-04-14

### Added

- Added 50B Dolmino 11/24 mix.
- Added support for auxiliary-loss-free MoE load-balancing, similar to DeepSeek-v3. You can activate this by setting `bias_gamma` to a non-zero float in your `MoERouter` config.
- Added support for sequence-level MoE load balancing loss.
- Compatibility with B200s.
- Added support for `warmup_fraction` as an alternative to `warmup_steps` in all schedulers, allowing warmup to be specified as a fraction of total training steps.
- A better config for the 1B model, ported from the old OLMo trainer.
- Added `auto_resume` option to `CometCallback` for resume an existing run.
- (BETA) Added methods `load_hf_model` and `save_hf_model` for saving supported OLMo Core models to HF transformers format.
Also added lower-level methods for converting state between the formats.
- Added the ability to run the evaluator callback on `.pre_train()` by setting `eval_on_startup=True`, and to cancel the run after the first time evals run by setting `cancel_after_first_eval=True`.
- Added support for label mask files with numpy FSL datasets.
- Added a `git` configuration to `BeakerLaunchConfig`.

### Changed

- `TransformerTrainModuleConfig` can now be used to build a `TransformerPipelineTrainModule` by adding a `pp_config` spec. This makes the `TransformerPipelineTrainModuleConfig` redundant, but it will be kept around for backwards compatibility until the next major release.
- Several state dict methods in `TrainModule` now take an `optim` option, which can disable the use of optimizer state.
- Updated `Float8Config` for latest version of `torchao`.
- Undo a fix applied to `olmo_core.data.numpy_dataset.NumpyFSLDatasetMixture` that was generating a mismatch between the shape of instances in the dataset and the shape of instances in the data loader.
- Made the 1B and 7B scripts more similar to each other.
- Changed underlying logic and top-level arguments of `convert_checkpoint_from_hf.py` and `convert_checkpoint_to_hf.py`.
- Beaker experiments launched with the `BeakerLaunchConfig` will now log with ANSI colors enabled.

### Fixed

- Fixed calculation of total steps based on epochs at the end of a training job.
- Fixed a bug where the trainer might try to save a duplicate final checkpoint if the run that already completed was restarted.
- When submitting a Beaker job from a branch that's tracking a GitHub fork, OLMo-core now instructs Beaker to pull from the fork instead of from the main repo.
- Made Beaker image resolution more robust.
- Having `t_max` overrides in the default model configs is confusing and error prone, so we removed them.
- Beaker launcher will only clone a single branch at runtime when possible, which can be much faster.


## [v2.0.1](https://github.com/allenai/OLMo-core/releases/tag/v2.0.1) - 2025-03-18

### Added

- Added information about the official 32B training run.
- Added information about the official 32B anneal training run.
- Added automatic support for LL128 when running on Augusta.
- Added information about 32B training logs.

### Fixed

- The official config for the 32B had unrealistic batch size settings.
- Ignore `group_overrides` for frozen parameters instead of throwing an error.
- Bump `ai2-olmo-eval==0.7.1`, which fixes makes the in-loop evaluation consistent with OLMES by removing [a bias](https://github.com/allenai/OLMo-in-loop-evals/pull/6)

### Removed

- Removed the "fused" cross-entropy loss variant. It had a bug and consistently under-performed the native PyTorch version when compiled. See [Post Incident Report: bug with fused CE loss](https://docs.google.com/document/d/1IK6q2gX6mH7eQO_IItCZAYYlm4g4htL4mNWbTQuPKf4/edit?usp=sharing) for more information.

## [v2.0.0](https://github.com/allenai/OLMo-core/releases/tag/v2.0.0) - 2025-03-12

This major release introduces a few breaking changes. We've provided more information here: [OLMo-core v2 design and upgrade guide](https://docs.google.com/document/d/1LvANhNzA-MdtiD2pLniLTqB9wxSSuqY435WuJIADeFM/edit?usp=sharing).

### Added

- Added `TrainModule` abstraction with `TransformerTrainModule` implementation, which encapsulates both a model and optimizer.
- Added `namespace` argument to `Trainer.record_metric()`.
- Added support for context parallelism.
- Added support for expert parallelism with MoE models.
- Added in-loop evals for Minerva, GSM, HumanEval, MBPP (`ai2-olmo-eval==0.7.0`)
- Added `CosWithWarmupAndLinearDecay` learning rate scheduler
- Added `WSD` learning rate scheduler
- Added `RunDuration` in `model_ladder` to configure training durations in terms of Chinchilla multipliers.

### Changed

- The `Trainer` now takes a `TrainModule` instead of a model and optimizer, and several configuration options have been moved to `TransformerTrainModule`, including `rank_microbatch_size`, `fused_loss`, `compile_loss`, `z_loss_multiplier`, and `autocast_precision`.
- Several `TransformerModelConfig` options have been to `TransformerTrainModule` / `TransformerTrainModuleConfig`, including `dp_config`, `tp_config`, `float8_config`, and `compile`.

### Removed

- Removed the following callbacks: `MoEHandlerCallback`, `SchedulerCallback`, `MatrixNormalizerCallback`, `GradClipperCallback`, and `Float8HandlerCallback`.
  The functionality from all of those callbacks has been moved to the `TransformerTrainModule` class.
- Removed the callback methods `.pre_eval_batch()` and `.post_eval_batch()`.

### Fixed

- Fixed the model ladder code when training on mps or cpu device

## [v1.9.0](https://github.com/allenai/OLMo-core/releases/tag/v1.9.0) - 2025-03-10

### Fixed

- Ensure certain optimizer param group fields are not overridden by the values in a checkpoint.

### Added

- Added `instance_filter_config` field to `NumpyDatasetConfig`.
- Added conversion script for OLMo 2 checkpoints to Huggingface format.
- Added `BeakerCallback`.
- Added logging for in-loop eval throughput

### Fixed

- Ensure certain optimizer param group fields are not overridden by the values in a checkpoint.
- Fixed issue where non-zero ranks would report partially-reduced values for training metrics.

## [v1.8.0](https://github.com/allenai/OLMo-core/releases/tag/v1.8.0) - 2025-01-29

### Added

- Added support for tensor parallelism. See the `TransformerConfig` class for usage.
- Added more downstream tasks from the model ladder.
- Added `io.copy_dir()` function.
- Added new LR schedulers: `LinearWithWarmup`, `InvSqrtWithWarmup`, `ConstantWithWarmup`, `SequentialScheduler`.
- Added option to pre-download checkpoint files from remote storage before trying to load a checkpoint.
- Added a callback for sending Slack notifications.
- Makes the MPS device work on Apple Silicon
- Added `SkipStepAdamW` optimizer.
- The trainer can load model-only checkpoints now.
- Added the option to throttle checkpoint uploads to one rank from each node at a time.
- Added support for logging rich Table objects as text in source mixture datasets.
- Added `unshard_strategy` parameter to `unshard_checkpoint()` function in `olmo_core.distributed.checkpoint`.
- Added function `load_keys()` to `olmo_core.distributed.checkpoint`.
- Added support for low precision optim state in `SkipStepAdamW`.

### Changed

- Changed storage of shared shard state in sharded checkpoints from smallest shard to lowest rank (normally 0).
- Changed how the trainer handles loading a checkpoint when `load_path` is provided. Now `load_path` is only used if no checkpoint is found in the `save_folder`.

### Fixed

- Added missing `weights_only=False` argument to fix loading train checkpoints with newer versions of PyTorch.
- Fixed bug where GCS upload does not retry on transient failures.
- Fixed bug where source mixture datasets were truncating source files instead of randomly sampling.
- Fixed bug in source mixture datsets where sampling from small npy files raised an mmap exception due to 0 instances in the sampled index.

## [v1.7.0](https://github.com/allenai/OLMo-core/releases/tag/v1.7.0) - 2024-11-27

### Added

- Added `key_mapping` argument to `olmo_core.distributed.checkpoint.load_model_and_optim_state()`
  for loading checkpoints with different key names.
- Added `load_key_mapping` field to the trainer, same idea as the new `key_mapping` argument above.
- Added an implementation of nGPT called `NormalizedTransformer`.
- Added an example showing how to convert a HuggingFace Llama 3.2 checkpoint into the right format for OLMo-core.
- Added an API for scaling RoPE embeddings.
- Added a `ModelLadder` API.

### Changed

- The `w_out` and `norm` top-level children of the `Transformer` model are now wrapped together in an `lm_head` module. Training scripts will have backwards compatibility with older checkpoints due to the `load_key_mapping` explained above.

### Fixed

- (Optimization) Mark model input sizes as dynamic for `torch.compile()` to avoid recompile during evals or variable-sequence / batch size training. This doesn't seem to hurt throughput.
- Made HTTPS and GCS IO functions more robust.
- Fixed a bug where we were always getting dolma2 tokenized validation data when generating config with DataMix.v3_small_ppl_validation.

## [v1.6.3](https://github.com/allenai/OLMo-core/releases/tag/v1.6.3) - 2024-11-15

### Added

- Added `olmo_core.distributed.checkpoint.get_checkpoint_metadata()` function.
- (BETA) Added flag to compile the optimizer step. So far only tested with AdamW. May not work with other optimizers.

### Fixed

- Old ephemeral checkpoints won't be removed until after the latest ephemeral checkpoint is saved successfully.
- Made GCS uploads more robust.
- Fixed single-node training on Google Augusta cluster.
- `numpy.random.dirichlet()` does not always sum to 1.0, so allow for a small tolerance in validating domain weights.

## [v1.6.2](https://github.com/allenai/OLMo-core/releases/tag/v1.6.2) - 2024-11-08

### Added

- Added option to disable `GarbageCollectorCallback`, not that you'd want to do this usually, but I needed to run an experiment to show how important that callback is.

### Fixed

- Fixed a bug where some default callbacks could be added twice if given a different name by the user.
- Fixed a bug where some `Trainer` bookkeeping tasks may not complete before `.fit()` returns.

## [v1.6.1](https://github.com/allenai/OLMo-core/releases/tag/v1.6.1) - 2024-11-06

### Added

- Added `retries` field to `BeakerLaunchConfig`.
- Allow running on Augusta cluster with existing train scripts.
- Added `olmo_core.utils.logging_configured()` function to check if logging has been configured.

### Fixed

- Fixed a potential distributed deadlock bug when training without a separate CPU-only bookkeeping backend.
- Removed some unnecessary host-device syncs in `olmo_core.distributed.utils`.
- Added `Trainer(Config).async_bookkeeping` field to toggle async bookkeeping.

## [v1.6.0](https://github.com/allenai/OLMo-core/releases/tag/v1.6.0) - 2024-11-01

### Added

- Added option to compile the trainer's loss function (`Trainer.compile_loss`).
- Added `SourceMixtureDataset` for composing a training mixture based on ratios of source datasets.
- Added `NumpyFSLDatasetMixture` for constructing a `NumpyDatasetBase` from a `SourceMixtureDataset`. Note this is only supported for FSL datasets.
- Added tests for `SourceMixture*` and `NumpyFSLDatasetMixture`.
- Added `DownstreamEvaluatorCallbackConfig` class for running in-loop downstream eval via [OLMo-in-loop-evals](https://github.com/allenai/OLMo-in-loop-evals).

### Changed

- Moved some types into `olmo_core.data.types` to avoid some circular dependencies.

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
