import concurrent.futures
import json
import logging
import typing
from abc import ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar

import rich
from cached_path import cached_path

import olmo_core.distributed.utils as dist_utils
import olmo_core.io as io
import olmo_core.train.callbacks as callbacks
from olmo_core.aliases import PathOrStr
from olmo_core.config import Config
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    InstanceSourceConfig,
    set_composable_seed,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.config import ModelConfig
from olmo_core.optim import OptimConfig, Scheduler
from olmo_core.train import (
    Checkpointer,
    Duration,
    DurationUnit,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.train_module import TrainModule

from .utils import format_count, format_tokens

if TYPE_CHECKING:
    from pandas import DataFrame

log = logging.getLogger(__name__)


class DeviceMeshSpec(NamedTuple):
    """
    Describes the relevant dimensions of a device mesh needed to train a model of a certain size.
    """

    world_size: int
    """The mininum numbers of devices required."""
    dp_world_size: int | None
    """
    The mininum size of the data parallel group. This can be set to ``None`` if the data parallel
    world size should equal the world size. This, along with the per-device micro-batch size, is
    needed to determine the right global batch size.
    """


@dataclass(frozen=True)
class RunCheckpointInfo:
    """Describes a checkpoint from a model run."""

    name: str
    """A descriptive name for the checkpoint, assigned by the :class:`RunConfigurator`."""
    step: int
    """The training step number of the checkpoint."""
    tokens: int
    """The number of training tokens processed up to this checkpoint."""
    checkpoint_path: PathOrStr
    """A path to the checkpoint directory."""
    metrics_path: PathOrStr | None
    """A path to the metrics JSON file for this checkpoint, if it exists."""
    exists: bool
    """Whether the checkpoint actually exists."""

    def display(self) -> str:
        """Get a rich-formatted string representation of the checkpoint info."""
        info = f"Step {self.step:,d} ({format_tokens(self.tokens)}) [b cyan]{self.name}[/]"
        if self.exists:
            out = f"[b green]✔[/] {info}\n  ↳ checkpoint: [u blue]{self.checkpoint_path}[/]"
            if self.metrics_path is not None:
                out += f"\n  ↳ metrics:    [u blue]{self.metrics_path}[/]"
            return out
        else:
            return f"[b yellow]✘[/] {info}"


M = TypeVar("M", bound=ModelConfig)


@dataclass(kw_only=True)
class ModelConfigurator(Config, Generic[M], metaclass=ABCMeta):
    """
    Defines how to configure a model of a particular size.
    """

    @abstractmethod
    def configure_model(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        tokenizer: TokenizerConfig,
        device_type: str,
    ) -> M:
        """Configure the model for the given size spec."""
        raise NotImplementedError

    @abstractmethod
    def configure_device_microbatch_size(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        device_type: str,
    ) -> int:
        """Configure the training per-device micro-batch size in tokens for a model of this size."""
        raise NotImplementedError

    @abstractmethod
    def configure_minimal_device_mesh_spec(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        device_type: str,
    ) -> DeviceMeshSpec:
        """Configure the minimal device mesh spec needed to train a model of this size."""
        raise NotImplementedError

    @abstractmethod
    def build_train_module(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        device_microbatch_size: int,
        model_config: M,
        optim_config: OptimConfig,
        scheduler: Scheduler,
        device_type: str,
    ) -> TrainModule:
        """Build the train module for the given model and optimizer configs."""
        raise NotImplementedError


@dataclass(kw_only=True)
class RunConfigurator(Config, metaclass=ABCMeta):
    """
    Defines how to configure a run for a model of a particular size.
    """

    @abstractmethod
    def configure_duration(self, num_params: int) -> Duration:
        """Get the training duration for a model of this size."""
        raise NotImplementedError

    @abstractmethod
    def configure_target_batch_size(self, num_params: int) -> int:
        """
        Get the target global batch size in tokens for a model of this size.
        The actual batch size used may be slightly different to ensure it's a multiple of
        the data parallel world size times the device micro-batch size.
        """
        raise NotImplementedError

    @abstractmethod
    def configure_optimizer(self, num_params: int) -> OptimConfig:
        """Get the optimizer config for a model of this size."""
        raise NotImplementedError

    @abstractmethod
    def configure_lr_scheduler(self, num_params: int) -> Scheduler:
        """Get the learning rate scheduler for a model of this size."""
        raise NotImplementedError

    @abstractmethod
    def configure_checkpoint_intervals(self, num_params: int) -> list[tuple[Duration, str]]:
        """
        Get the checkpoint intervals for a model of this size.
        Returns a list of (checkpoint interval, checkpoint description) tuples.
        """
        raise NotImplementedError

    @abstractmethod
    def plot_lr_schedule(
        self, num_params: int, *, batch_size: int | None = None, save_path: PathOrStr | None = None
    ):
        """Render a plot of the learning rate schedule."""
        raise NotImplementedError


@dataclass(kw_only=True)
class ModelLadder(Config):
    """
    Represents a complete model ladder of runs.
    """

    name: str
    """A name to assign to the ladder."""
    dir: str
    """A unique directory where ladder run results and intermediate checkpoints should be saved."""
    sizes: list[str]
    """A list of model size specs to run as part of the ladder."""
    max_devices: int
    """The number of accelerator devices available to use for each run."""
    device_type: str
    """The type of accelerator device available to use for each run (e.g. "NVIDIA H100 80GB HBM3")."""
    model_configurator: ModelConfigurator
    """The model configurator to use."""
    run_configurator: RunConfigurator
    """The run configurator to use."""
    data_loader: ComposableDataLoaderConfig
    """The data loader configuration to use for each run."""
    instance_sources: list[InstanceSourceConfig]
    """The instance sources to use for each run."""
    sequence_length: int = 8192
    """The sequence length to train each run on."""
    tokenizer: TokenizerConfig
    """The tokenizer to use."""
    seed: int = 42
    """The initial random seed to use for all runs in the ladder."""
    backend: str = "cpu:gloo,cuda:nccl"
    """The distributed backend to use for each run."""

    def __post_init__(self):
        if self.max_devices <= 0:
            raise OLMoConfigurationError("max_devices must be a positive integer.")
        for size_spec in self.sizes:
            min_devices, _ = self.model_configurator.configure_minimal_device_mesh_spec(
                size_spec=size_spec,
                sequence_length=self.sequence_length,
                device_type=self.device_type,
            )
            if min_devices > self.max_devices:
                raise OLMoConfigurationError(
                    f"Model of size {size_spec} requires at least {min_devices} devices, "
                    f"but max_devices is set to {self.max_devices}."
                )

    @property
    def work_dir(self) -> PathOrStr:
        return "./cache" if io.is_url(self.dir) else str(io.join_path(self.dir, "cache"))

    def dry_run(self, size_spec: str):
        """
        Do a dry-run, which prints relevant hyperparameters, the required number of devices,
        and a displays a plot of the learning rate schedule.
        """
        if size_spec not in self.sizes:
            raise ValueError(f"Invalid size_spec '{size_spec}', must be one of {self.sizes}")

        num_params = self.get_num_params(size_spec)

        # Configure global batch size, make sure request number of devices matches the number
        # of devices available.
        target_global_batch_size = self.run_configurator.configure_target_batch_size(num_params)
        (
            global_batch_size,
            device_microbatch_size,
            requested_devices,
            dp_world_size,
        ) = self._configure_batch_size_and_num_devices(size_spec, num_params)
        assert device_microbatch_size % self.sequence_length == 0
        assert global_batch_size % device_microbatch_size == 0
        assert global_batch_size % self.sequence_length == 0
        assert device_microbatch_size % self.sequence_length == 0
        assert global_batch_size % (device_microbatch_size * dp_world_size) == 0
        num_grad_accum_steps = global_batch_size // (device_microbatch_size * dp_world_size)

        rich.get_console().print(
            f"Dry run for model size {size_spec}:\n"
            f" ❯ Actual number of non-embedding params is {format_count(num_params)}\n"
            f" ❯ Target batch size is {target_global_batch_size:,d} tokens\n"
            f" ❯ Actual batch size is {global_batch_size:,d} tokens, which is "
            f"{global_batch_size // self.sequence_length:,d} instance(s)\n"
            f" ❯ Micro-batch size per device size {device_microbatch_size:,d} tokens, which is "
            f"{device_microbatch_size // self.sequence_length} instance(s)\n"
            f" ❯ So there will be {num_grad_accum_steps:,d} grad accumulation step(s) per batch\n"
            f" ❯ And the run requires {requested_devices} out of {self.max_devices} devices, "
            f"with a data-parallel world size of {dp_world_size:,d}.",
            highlight=False,
        )
        log.info("Plotting LR schedule...")
        self.run_configurator.plot_lr_schedule(num_params, batch_size=global_batch_size)

    def run(self, size_spec: str, for_benchmarking: bool = False):
        """
        Execute a particular model run of the experiment locally and store the results.
        """
        if size_spec not in self.sizes:
            raise ValueError(f"Invalid size_spec '{size_spec}', must be one of {self.sizes}")
        prepare_training_environment(seed=self.seed, backend=self.backend)
        set_composable_seed(self.seed)

        # Configure model.
        model_config = self.get_model_config(size_spec)
        num_params = model_config.num_non_embedding_params

        # Configure global batch size, make sure request number of devices matches the number
        # of devices available.
        (
            global_batch_size,
            device_microbatch_size,
            requested_devices,
            _,
        ) = self._configure_batch_size_and_num_devices(size_spec, num_params)
        if requested_devices != dist_utils.get_world_size():
            raise OLMoConfigurationError(
                f"Requested {requested_devices} devices for model of size '{size_spec}', "
                f"but {dist_utils.get_world_size()} are available."
            )

        # Configure optimizer and scheduler.
        optim_config = self.run_configurator.configure_optimizer(num_params)
        scheduler = self.run_configurator.configure_lr_scheduler(num_params)

        # Configure trainer.
        trainer_config = self._configure_trainer(size_spec, for_benchmarking=for_benchmarking)

        # Build instance sources and data loader.
        instance_sources = [
            source.build(work_dir=self.work_dir) for source in self.instance_sources
        ]
        data_loader = self.data_loader.build(
            *instance_sources,
            work_dir=self.work_dir,
            global_batch_size=global_batch_size,
            tokenizer=self.tokenizer,
        )
        if data_loader.sequence_length != self.sequence_length:
            raise OLMoConfigurationError(
                f"Data loader sequence of {data_loader.sequence_length} does not match "
                f"configured sequence length of {self.sequence_length}."
            )

        # Build train module.
        train_module = self.model_configurator.build_train_module(
            size_spec=size_spec,
            sequence_length=self.sequence_length,
            device_microbatch_size=device_microbatch_size,
            model_config=model_config,
            optim_config=optim_config,
            scheduler=scheduler,
            device_type=self.device_type,
        )

        # Build trainer.
        trainer = trainer_config.build(train_module, data_loader)

        # Record all configs.
        config_dict = {
            "seed": self.seed,
            "size": str(size_spec),
            "model": model_config.as_config_dict(),
            "optim": optim_config.as_config_dict(),
            "scheduler": scheduler.as_config_dict(),
            "data_loader": self.data_loader.as_config_dict(),
            "instance_sources": [s.as_config_dict() for s in self.instance_sources],
        }
        typing.cast(
            callbacks.ConfigSaverCallback, trainer.callbacks["config_saver"]
        ).config = config_dict

        # Train.
        trainer.fit()

        teardown_training_environment()

    def run_benchmark(self, size_spec: str):
        """
        Do a bench-marking run for a model of the given size spec. This is just like
        :meth:`run`, but with benchmarking-specific settings (no checkpoints, no evals, hard stop).
        """
        self.run(size_spec, for_benchmarking=True)

    def get_model_config(self, size_spec: str) -> ModelConfig:
        """Get the model config for a model of the given size spec."""
        return self.model_configurator.configure_model(
            size_spec=size_spec,
            sequence_length=self.sequence_length,
            tokenizer=self.tokenizer,
            device_type=self.device_type,
        )

    def get_num_params(self, size_spec: str):
        """Get the actual number of non-embedding parameters for a model of the given size spec."""
        return self.get_model_config(size_spec).num_non_embedding_params

    def get_num_devices(self, size_spec: str) -> int:
        """Get the number of devices that would be used for a run of the given size spec."""
        _, _, num_devices, _ = self._configure_batch_size_and_num_devices(
            size_spec, self.get_num_params(size_spec)
        )
        return num_devices

    def get_save_folder(self, size_spec: str) -> str:
        """Get the training save folder for a run of the given size spec."""
        return str(io.join_path(self.dir, size_spec))

    def get_checkpoints(
        self, size_spec: str, download_metrics: bool = False
    ) -> list[RunCheckpointInfo]:
        """
        Get the list of checkpoints from the run of the given size spec, at the intervals
        defined by :meth:`RunConfigurator.configure_checkpoint_intervals()`.
        """

        def _get_checkpoint_info(step: int, name: str) -> RunCheckpointInfo:
            dirname = Checkpointer.checkpoint_dirname(step)
            dir = io.join_path(save_folder, dirname)
            exists = Checkpointer.dir_is_checkpoint(dir)
            metrics_path: PathOrStr | None = io.join_path(save_folder, f"metrics_step{step}.json")
            if not io.file_exists(metrics_path):
                metrics_path = None
            elif download_metrics:
                metrics_path = cached_path(metrics_path, quiet=True)
            return RunCheckpointInfo(
                name=name,
                step=step,
                tokens=step * global_batch_size,
                checkpoint_path=dir,
                metrics_path=metrics_path,
                exists=exists,
            )

        save_folder = self.get_save_folder(size_spec)
        io.init_client(save_folder)

        num_params = self.get_num_params(size_spec)
        global_batch_size, *_ = self._configure_batch_size_and_num_devices(size_spec, num_params)

        checkpoints_to_check: dict[int, str] = {0: "initialization"}
        for step, (_, checkpoint_name) in zip(
            self._get_checkpoint_intervals(
                num_params=num_params, global_batch_size=global_batch_size
            ),
            self.run_configurator.configure_checkpoint_intervals(num_params),
        ):
            checkpoints_to_check[step] = checkpoint_name

        step_to_checkpoint_info: dict[int, RunCheckpointInfo] = {}
        with ThreadPoolExecutor() as executor:
            futures = []
            for step, name in checkpoints_to_check.items():
                futures.append(executor.submit(_get_checkpoint_info, step, name))
            for future in concurrent.futures.as_completed(futures):
                info = future.result()
                step_to_checkpoint_info[info.step] = info

        return [step_to_checkpoint_info[step] for step in sorted(step_to_checkpoint_info.keys())]

    def get_metrics(self, size_spec: str, prefix: str = "eval/") -> "DataFrame":
        """
        Get the metrics from the run of the given size spec, at the intervals
        defined by :meth:`RunConfigurator.configure_checkpoint_intervals()`.
        """
        import pandas as pd

        checkpoints = self.get_checkpoints(size_spec, download_metrics=True)
        num_params = self.get_num_params(size_spec)
        all_metrics = []
        for checkpoint in checkpoints:
            if checkpoint.metrics_path is not None:
                with open(checkpoint.metrics_path, "r") as f:
                    metrics = {k: v for k, v in json.load(f).items() if k.startswith(prefix)}
                    metrics["name"] = checkpoint.name
                    metrics["step"] = checkpoint.step
                    metrics["tokens"] = checkpoint.tokens
                    metrics["size"] = size_spec
                    metrics["num_params"] = num_params
                    all_metrics.append(metrics)
        df = pd.DataFrame(all_metrics)
        return df

    def _get_checkpoint_intervals(self, *, num_params: int, global_batch_size: int) -> list[int]:
        return [
            self._duration_to_steps(d, global_batch_size)
            for d, _ in self.run_configurator.configure_checkpoint_intervals(num_params)
        ]

    def _duration_to_steps(self, d: Duration, global_batch_size: int) -> int:
        if d.unit == DurationUnit.steps:
            return d.value
        elif d.unit == DurationUnit.tokens:
            steps = d.value // global_batch_size
            return steps
        else:
            raise ValueError(f"Unsupported checkpoint interval duration unit: {d.unit}.")

    def _configure_batch_size_and_num_devices(
        self, size_spec: str, num_params: int
    ) -> tuple[int, int, int, int]:
        # Configure global batch size and device micro-batch size.
        global_batch_size = self.run_configurator.configure_target_batch_size(num_params)
        device_microbatch_size = self.model_configurator.configure_device_microbatch_size(
            size_spec=size_spec,
            sequence_length=self.sequence_length,
            device_type=self.device_type,
        )
        device_microbatch_size = min(device_microbatch_size, global_batch_size)

        # Configure minimal device mesh spec, i.e. the minimum number of devices needed and the
        # corresponding minimum data parallel world size.
        (
            min_world_size,
            min_dp_world_size,
        ) = self.model_configurator.configure_minimal_device_mesh_spec(
            size_spec=size_spec,
            sequence_length=self.sequence_length,
            device_type=self.device_type,
        )
        if min_dp_world_size is None:
            min_dp_world_size = min_world_size
        if min_world_size % min_dp_world_size != 0:
            raise OLMoConfigurationError(
                f"Invalid device mesh spec for model of size '{size_spec}': "
                f"minimum world size {min_world_size} is not divisible by "
                f"the minimum data parallel world size {min_dp_world_size}."
            )
        if self.max_devices < min_world_size:
            raise OLMoConfigurationError(
                f"Not enough devices ({self.max_devices}) to run model of size '{size_spec}' "
                f"which requires at least {min_world_size} devices."
            )

        # And from that we adjust the global batch size to be a multiple of
        # `device_microbatch_size x min_dp_world_size`.
        gbz_factor = device_microbatch_size * min_dp_world_size
        global_batch_size = round(global_batch_size / gbz_factor) * gbz_factor

        # Then we can determine the actual number of devices to allocate to the run. In particular
        # we can expand `min_world_size` up to the number of devices available (`self.max_devices`)
        # by a factor that's just the number of gradient accumulation steps needed with the minimum
        # requested number of devices.
        max_num_grad_accum_steps = global_batch_size // gbz_factor
        expansion_factor = min(self.max_devices // min_world_size, max_num_grad_accum_steps)
        num_devices = min_world_size * expansion_factor
        dp_world_size = min_dp_world_size * expansion_factor

        # Finally we ensure `global_batch_size` is divisible by the micro-batch size.
        microbatch_size = device_microbatch_size * dp_world_size
        global_batch_size = round(global_batch_size / microbatch_size) * microbatch_size

        return global_batch_size, device_microbatch_size, num_devices, dp_world_size

    def _configure_trainer(
        self,
        size_spec: str,
        for_benchmarking: bool = False,
    ) -> TrainerConfig:
        run_name = f"{self.name}-{size_spec}"
        save_folder = self.get_save_folder(size_spec)
        num_params = self.get_num_params(size_spec)
        global_batch_size, *_ = self._configure_batch_size_and_num_devices(size_spec, num_params)
        duration = self.run_configurator.configure_duration(num_params)
        checkpoint_interval_steps = self._get_checkpoint_intervals(
            num_params=num_params, global_batch_size=global_batch_size
        )
        return TrainerConfig(
            save_folder=save_folder,
            work_dir=str(self.work_dir),
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=duration,
            hard_stop=Duration.steps(100) if for_benchmarking else None,
            no_checkpoints=for_benchmarking,
            no_evals=for_benchmarking,
            save_overwrite=for_benchmarking,
            callbacks={
                "gpu_monitor": callbacks.GPUMemoryMonitorCallback(),
                "config_saver": callbacks.ConfigSaverCallback(),
                "garbage_collector": callbacks.GarbageCollectorCallback(),
                "checkpointer": callbacks.CheckpointerCallback(
                    ephemeral_save_interval=1000,
                    ephemeral_cooldown=250,
                    save_interval=None,
                    save_async=True,
                    fixed_steps=checkpoint_interval_steps,
                    enabled=not for_benchmarking,
                ),
                "profiler": callbacks.ProfilerCallback(enabled=for_benchmarking),
                "gap_monitor": callbacks.GAPMonitorCallback(enabled=False),
                "slack_notifier": callbacks.SlackNotifierCallback(name=run_name, enabled=False),
                "beaker": callbacks.BeakerCallback(),
                "wandb": callbacks.WandBCallback(
                    name=run_name,
                    group=run_name,
                    project=self.name,
                    cancel_check_interval=50,
                    enabled=not for_benchmarking,
                ),
                "downstream_evaluator": callbacks.DownstreamEvaluatorCallbackConfig(
                    tokenizer=self.tokenizer,
                    tasks=self._get_in_loop_eval_tasks(),
                    eval_interval=None,
                    fixed_steps=checkpoint_interval_steps,
                    enabled=not for_benchmarking,
                ),
                "metric_saver": callbacks.MetricSaverCallback(
                    fixed_steps=checkpoint_interval_steps,
                    enabled=not for_benchmarking,
                ),
            },
        )

    def _get_in_loop_eval_tasks(self) -> list[str]:
        # For training runs where we don't expect the model to acquire MC (e.g., 1B-5xC, short 7B training runs).
        tasks_small_compute = [
            # OLMES Core 9(-ish) RC
            "arc_challenge_test_rc_5shot",
            "arc_easy_test_rc_5shot",
            "hellaswag_rc_5shot",  # 1K subset of HellaSwag
            "winogrande_val_rc_5shot",  # Helpful after 750M-5xC scale
            "csqa_val_rc_5shot",
            "piqa_val_rc_5shot",
            "socialiqa_val_rc_5shot",
            # Too noisy to be worth tracking
            # "boolq_val_rc_5shot",
            # "openbookqa_test_rc_5shot",
            # MMLU RC
            "mmlu_stem_val_rc_5shot",
            "mmlu_humanities_val_rc_5shot",
            "mmlu_social_sciences_val_rc_5shot",
            "mmlu_other_val_rc_5shot",
            "mmlu_stem_test_rc_5shot",
            "mmlu_humanities_test_rc_5shot",
            "mmlu_social_sciences_test_rc_5shot",
            "mmlu_other_test_rc_5shot",
            # Gen tasks BPB
            "gsm8k_gold_bpb_5shot",
            "minerva_math_algebra_gold_bpb_0shot",
            "minerva_math_counting_and_probability_gold_bpb_0shot",
            "minerva_math_geometry_gold_bpb_0shot",
            "minerva_math_intermediate_algebra_gold_bpb_0shot",
            "minerva_math_number_theory_gold_bpb_0shot",
            "minerva_math_prealgebra_gold_bpb_0shot",
            "minerva_math_precalculus_gold_bpb_0shot",
            "codex_humaneval_gold_bpb_0shot",
            "codex_mbpp_gold_bpb_0shot",
            # Sanity check for MCQA ability
            "copycolors_10way",
        ]

        # For training runs where we expect the model to acquire MC
        tasks_large_compute = [
            # OLMES Core 9(-ish) MC
            "arc_challenge_test_mc_5shot",
            "arc_easy_test_mc_5shot",
            "hellaswag_rc_5shot",  # 1K subset of HellaSwag
            "csqa_val_mc_5shot",
            "piqa_val_mc_5shot",
            "socialiqa_val_mc_5shot",
            "winogrande_val_rc_5shot",
            # Too noisy to be worth tracking
            # "boolq_val_mc_5shot",
            # "openbookqa_test_mc_5shot",
            # MMLU MC BPB
            "mmlu_stem_val_mc_5shot",
            "mmlu_humanities_val_mc_5shot",
            "mmlu_social_sciences_val_mc_5shot",
            "mmlu_other_val_mc_5shot",
            "mmlu_stem_test_mc_5shot",
            "mmlu_humanities_test_mc_5shot",
            "mmlu_social_sciences_test_mc_5shot",
            "mmlu_other_test_mc_5shot",
            # Gen tasks BPB
            "gsm8k_gold_bpb_5shot",
            "minerva_math_algebra_gold_bpb_0shot",
            "minerva_math_counting_and_probability_gold_bpb_0shot",
            "minerva_math_geometry_gold_bpb_0shot",
            "minerva_math_intermediate_algebra_gold_bpb_0shot",
            "minerva_math_number_theory_gold_bpb_0shot",
            "minerva_math_prealgebra_gold_bpb_0shot",
            "minerva_math_precalculus_gold_bpb_0shot",
            "codex_humaneval_gold_bpb_0shot",
            "codex_mbpp_gold_bpb_0shot",
            # Sanity check for MCQA ability
            "copycolors_10way",
        ]

        # Unfortunately we need the same metrics for everything, so we run them all.
        tasks = list(set(tasks_small_compute + tasks_large_compute))
        tasks.sort()
        return tasks
