"""
Configuration classes for defining model ladder scaling ablations.
"""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from torch.distributed.device_mesh import DeviceMesh

from olmo_core.config import Config, StrEnum
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
)
from olmo_core.distributed.utils import get_num_nodes, init_hybrid_shard_mesh
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import join_path
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    GradClipperCallback,
    LMEvaluatorCallbackConfig,
    SchedulerCallback,
    WandBCallback,
)

__all__ = ["ModelSize", "ModelLadder"]


class ModelSize(StrEnum):
    """
    An enumeration of the standard model sizes in the ladder.
    :class:`ModelLadder` implementations should produce models that match these sizes
    as close as possible, ignoring embeddings.
    """

    size_190M = "190M"
    """
    190M parameters.
    """
    size_370M = "370M"
    """
    370M parameters.
    """
    size_600M = "600M"
    """
    600M parameters.
    """
    size_760M = "760M"
    """
    760M parameters.
    """
    size_1B = "1B"
    """
    1B parameters.
    """
    size_3B = "3B"
    """
    3B parameters.
    """
    size_7B = "7B"
    """
    7B parameters.
    """
    size_13B = "13B"
    """
    13B parameters.
    """

    @property
    def num_params(self) -> int:
        value, unit = int(self[:-1]), self[-1]
        if unit == "M":
            return value * int(1e6)
        elif unit == "B":
            return value * int(1e9)
        else:
            raise NotImplementedError(self)


@dataclass
class ModelLadder(Config, metaclass=ABCMeta):
    """
    Base class for defining model ladder experiments.

    At a minimum subclasses must implement:

    - :meth:`get_model_config()`
    - :meth:`get_optim_config()`
    - :meth:`get_rank_microbatch_size()`

    for every model size defined by :class:`ModelSize`.
    """

    name: str
    """
    The name of the ladder runs.
    """

    project: str
    """
    The name of the W&B/Comet project to save run data to.
    """

    mix_base_dir: str
    """
    The base directory of the training data.
    """

    work_dir: str
    """
    The local working directory used for dataset caching.
    """

    save_folder: str
    """
    The local or remote folder to save checkpoints to.
    """

    sequence_length: int = 2048
    """
    The target sequence length to train the ladder on.
    """

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig.dolma2)
    """
    Get the tokenizer config to use throughput the ladder.
    """

    init_seed: int = 2352
    """
    The seed to use when first initializing RNG states.
    """

    data_mix: DataMix = DataMix.OLMoE_mix_0824
    """
    The data mix to train on.
    """

    data_seed: int = 34521
    """
    The seed to use for shuffling the data.
    """

    def get_save_folder(self, size: ModelSize) -> str:
        return str(join_path(self.save_folder, f"checkpoints/{self.name}-{size}"))

    @abstractmethod
    def get_model_config(self, *, size: ModelSize) -> TransformerConfig:
        """
        Get the model config for a given model size.

        :param size: The target model size.
        """
        raise NotImplementedError

    @abstractmethod
    def get_optim_config(self, *, size: ModelSize) -> OptimConfig:
        """
        Get the optimizer config for a given model size.

        :param size: The target model size.
        """
        raise NotImplementedError

    def get_dataset_config(self) -> NumpyDatasetConfig:
        """
        Get the train dataset config.

        :param kwargs: Extra kwargs to pass to the dataset config constructor.
        """
        return NumpyDatasetConfig.from_data_mix(
            self.data_mix,
            tokenizer=self.tokenizer,
            mix_base_dir=self.mix_base_dir,
            sequence_length=self.sequence_length,
            work_dir=self.work_dir,
        )

    def get_data_loader_config(
        self, *, size: ModelSize, dp_world_size: int
    ) -> NumpyDataLoaderConfig:
        """
        Get the data loader config.

        :param size: The target model size.
        :param dp_world_size: The data parallel world size for training.
        """
        return NumpyDataLoaderConfig(
            global_batch_size=self.get_global_batch_size(size=size, dp_world_size=dp_world_size),
            seed=self.data_seed,
            num_workers=4,
        )

    @abstractmethod
    def get_rank_microbatch_size(self, *, size: ModelSize, gpu_type: str) -> int:
        """
        Returns the micro-batch size in tokens per device that should be used for the given
        model size.

        :param size: The target model size.
        :param gpu_type: The type of GPU as given by ``torch.cuda.get_device_name()``.
        """
        raise NotImplementedError

    def get_global_batch_size(self, *, size: ModelSize, dp_world_size: int = 64) -> int:
        """
        Get the global batch size in tokens for a given model size.

        :param size: The target model size.
        :param dp_world_size: The data parallel world size for training.
        """
        # Calculate batch size according to https://api.semanticscholar.org/CorpusID:270764838,
        # which assumes a sequence length of 2048. So adjust from there accordingly.
        assert self.sequence_length in {2048, 4096, 8192}
        seq_len_divisor = self.sequence_length // 2048

        global_batch_size = 160 * (size.num_params / 108000000) ** (2 / 3)
        global_batch_size /= seq_len_divisor
        global_batch_size /= dp_world_size
        global_batch_size = round(global_batch_size)
        global_batch_size *= dp_world_size

        return self.sequence_length * global_batch_size

    def get_duration(self, size: ModelSize) -> Duration:
        """
        Get the duration to train for given the model size. Defaults to 2 x Chinchilla optimal.

        :param size: The target model size.
        """
        return Duration.tokens(2 * 20 * size.num_params)

    def get_dp_mesh(self, *, size: ModelSize) -> Optional[DeviceMesh]:
        """
        Get the data parallel device mesh. Could be a 2D mesh for HSDP, or just none or FSDP/DDP.
        """
        if get_num_nodes() == 1 or size.num_params < 1e9:
            return None
        else:
            return init_hybrid_shard_mesh()

    def get_trainer_config(
        self,
        *,
        size: ModelSize,
        gpu_type: str,
    ) -> TrainerConfig:
        """
        Build the trainer config.

        :param size: The target model size.
        :param gpu_type: The type of GPU as given by ``torch.cuda.get_device_name()``.
        """
        rank_mbz = self.get_rank_microbatch_size(size=size, gpu_type=gpu_type)
        if rank_mbz % self.sequence_length != 0:
            raise OLMoConfigurationError(
                f"rank micro-batch size ({rank_mbz:,d} tokens) must be divisible "
                f"by the sequence length ({self.sequence_length:,d})"
            )

        return (
            TrainerConfig(
                save_folder=self.get_save_folder(size),
                rank_microbatch_size=rank_mbz,
                metrics_collect_interval=10,
                cancel_check_interval=1,
                compile_loss=True,
                max_duration=self.get_duration(size),
            )
            .with_callback(
                "lr_scheduler", SchedulerCallback(scheduler=CosWithWarmup(warmup_steps=2000))
            )
            .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
            .with_callback("grad_clipper", GradClipperCallback(max_grad_norm=1.0))
            .with_callback("config_saver", ConfigSaverCallback())
            .with_callback("garbage_collector", GarbageCollectorCallback())
            .with_callback(
                "lm_evaluator",
                LMEvaluatorCallbackConfig(
                    eval_dataset=NumpyDatasetConfig.from_data_mix(
                        DataMix.v3_small_ppl_validation,
                        name=NumpyDatasetType.padded_fsl,
                        mix_base_dir=self.mix_base_dir,
                        sequence_length=self.sequence_length,
                        tokenizer=self.tokenizer,
                        work_dir=self.work_dir,
                    ),
                    eval_interval=1000,
                ),
            )
            .with_callback(
                "downstream_evaluator",
                DownstreamEvaluatorCallbackConfig(
                    tasks=["hellaswag"],  # TODO: which other tasks?
                    tokenizer=self.tokenizer,
                    eval_interval=250,
                ),
            )
            .with_callback(
                "checkpointer",
                CheckpointerCallback(
                    save_interval=100_000,  # large enough value that we won't save until the end
                    ephemeral_save_interval=250,
                    save_async=True,
                ),
            )
            .with_callback(
                "comet",
                CometCallback(
                    name=f"{self.name}-{size}",
                    workspace="ai2",
                    project=self.project,
                    enabled=True,
                    cancel_check_interval=5,
                ),
            )
            .with_callback(
                "wandb",
                WandBCallback(
                    name=f"{self.name}-{size}",
                    entity="ai2",
                    project=self.project,
                    enabled=False,
                    cancel_check_interval=5,
                ),
            )
        )

    def validate(self):
        """
        Validate the ladder configuration.

        :raises OLMoConfigurationError: If the ladder has any issues.
        """
        for size in ModelSize:
            target_size = int(size[:-1])
            if size.endswith("M"):
                target_size = target_size * 10**6
            elif size.endswith("B"):
                target_size = target_size * 10**9
            else:
                raise NotImplementedError(size)

            model_config = self.get_model_config(size=size)

            # Make sure actual model size is close to target size.
            num_params = model_config.num_non_embedding_params
            if abs(num_params - target_size) / target_size > 0.05:
                raise OLMoConfigurationError(
                    f"Model size of {num_params:,d} for sequence length {self.sequence_length} is "
                    f"too far from target size of {size}: {model_config}"
                )

            self.get_optim_config(size=size)
            self.get_rank_microbatch_size(size=size, gpu_type="H100")
            bz_tokens = self.get_global_batch_size(size=size)
            if bz_tokens % self.sequence_length != 0:
                raise OLMoConfigurationError(
                    f"Batch size of {bz_tokens:,d} tokens for model size {size} "
                    f"must be divisible by the sequence length ({self.sequence_length})"
                )
