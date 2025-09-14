"""
Configuration classes for defining model ladder scaling ablations.
"""

import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, List

from olmo_core.data.numpy_dataset import InstanceFilterConfig
from olmo_core.train.callbacks.monkey_patcher import MonkeyPatcherCallback

from .config import Config, StrEnum
from .data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
)
from .doc_utils import beta_feature
from .exceptions import OLMoConfigurationError
from .io import join_path
from .nn.transformer import TransformerConfig
from .optim import CosWithWarmup, OptimConfig
from .train import Duration, TrainerConfig
from .train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    WandBCallback,
)
from .train.train_module import TransformerTrainModuleConfig

__all__ = ["ModelSize", "ModelLadder", "RunDuration"]

log = logging.getLogger(__name__)


class ModelSize(StrEnum):
    """
    An enumeration of the standard model sizes in the ladder.
    :class:`ModelLadder` implementations should produce models that match these sizes
    as close as possible, ignoring embeddings.
    """

    size_60M = "60M"
    """
    60M parameters.
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
    size_970M = "970M"
    """
    970M parameters.
    """
    size_1B = "1B"
    """
    1B parameters.
    """
    size_2140M = "2140M"
    """
    2140M parameters.
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
    size_32B = "32B"
    """
    32B parameters.
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


class RunDuration(StrEnum):
    """
    An enumeration of the standard training durations for the ladder, in terms of Chinchilla multipliers.
    """

    Cx0_5 = "0.5xC"
    """
    Multiplier of 0.5.
    """

    Cx1 = "1xC"
    """
    Multiplier of 1.
    """
    Cx2 = "2xC"
    """
    Multiplier of 2.
    """
    Cx5 = "5xC"
    """
    Multiplier of 5.
    """

    Cx10 = "10xC"
    """
    Multiplier of 10.
    """

    @property
    def multiplier(self) -> float:
        return float(self.split("xC")[0])


@beta_feature
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

    max_dp_world_size: int = 64
    """
    The maximum data parallel world size that you intent to run with. This is used to set the batch size.
    """

    beaker_workspace: str = "ai2/OLMo-core"
    """
    Beaker workspace.
    """

    intra_document_masking: bool = False

    SUPPORTED_MODEL_SIZES: ClassVar[List[ModelSize]] = list(ModelSize)

    @property
    def model_size(self) -> int:
        """
        The size of the model in terms of non-embedding parameters.
        """
        return self._model_size

    def get_save_folder(self, size: ModelSize, run_duration: RunDuration) -> str:
        return str(join_path(self.save_folder, f"checkpoints/{self.name}-{size}-{run_duration}"))

    @abstractmethod
    def _get_model_config(self, *, size: ModelSize) -> TransformerConfig:
        """
        Get the model config for a given model size.

        :param size: The target model size.
        """
        raise NotImplementedError

    def get_model_config(self, *, size: ModelSize) -> TransformerConfig:
        """
        Get the model config for a given model size.

        :param size: The target model size.
        """
        model_config = self._get_model_config(size=size)
        self._model_size = model_config.num_non_embedding_params
        return model_config

    @abstractmethod
    def get_optim_config(self) -> OptimConfig:
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
            generate_doc_lengths=self.intra_document_masking,
            instance_filter_config=InstanceFilterConfig(
                repetition_max_period=13,
                repetition_min_period=1,
                repetition_max_count=32,
            ),
        )

    def get_data_loader_config(self) -> NumpyDataLoaderConfig:
        """
        Get the data loader config.

        :param size: The target model size.
        """
        return NumpyDataLoaderConfig(
            global_batch_size=self.get_global_batch_size(),
            seed=self.data_seed,
            num_workers=4,
        )

    @abstractmethod
    def get_rank_microbatch_size(self, *, size: ModelSize, gpu_type: str) -> int:
        """
        Returns the maximum micro-batch size in tokens per device that should be used for the given
        model size.

        :param size: The target model size.
        :param gpu_type: The type of GPU as given by ``torch.cuda.get_device_name()``.
        """
        raise NotImplementedError

    def get_global_batch_size(self) -> int:
        """
        Get the global batch size in tokens for a given model size.

        :param size: The target model size.
        """
        # Calculate batch size according to https://api.semanticscholar.org/CorpusID:270764838,
        # which assumes a sequence length of 2048. So adjust from there accordingly.
        assert self.sequence_length in {2048, 4096, 8192}
        seq_len_divisor = self.sequence_length // 2048

        global_batch_size = 160 * (self.model_size / 108000000) ** (2 / 3)
        global_batch_size /= seq_len_divisor
        global_batch_size /= self.max_dp_world_size
        global_batch_size = round(global_batch_size)
        global_batch_size *= self.max_dp_world_size

        return self.sequence_length * global_batch_size

    def get_duration(self, run_duration: RunDuration = RunDuration.Cx2) -> Duration:
        """
        Get the duration to train for given the model size. Defaults to 2 x Chinchilla optimal.

        :param size: The target model size.
        """
        return Duration.tokens(int(run_duration.multiplier * 20) * self.model_size)

    def get_train_module_config(
        self,
        *,
        size: ModelSize,
        run_duration: RunDuration,
        gpu_type: str,
        dp_world_size: int,
    ) -> TransformerTrainModuleConfig:
        """
        Build the train module config.

        :param size: The target model size.
        :param gpu_type: The type of GPU as given by ``torch.cuda.get_device_name()``.
        :param dp_world_size: The data parallel world size.
        """
        if dp_world_size > self.max_dp_world_size:
            raise OLMoConfigurationError(
                f"max_dp_world_size ({self.max_dp_world_size}) must be at least as big as current dp "
                f"world size ({dp_world_size})"
            )

        rank_mbz = self.get_rank_microbatch_size(size=size, gpu_type=gpu_type)
        if rank_mbz % self.sequence_length != 0:
            raise OLMoConfigurationError(
                f"rank micro-batch size ({rank_mbz:,d} tokens) must be divisible "
                f"by the sequence length ({self.sequence_length:,d})"
            )

        rank_mbz_instances = rank_mbz // self.sequence_length

        global_bz = self.get_global_batch_size()
        if global_bz % self.sequence_length != 0:
            raise OLMoConfigurationError(
                f"global batch size ({rank_mbz:,d} tokens) must be divisible "
                f"by the sequence length ({self.sequence_length:,d})"
            )

        global_bz_instances = self.get_global_batch_size() // self.sequence_length

        if global_bz_instances % (rank_mbz_instances * dp_world_size) != 0:
            new_rank_mbz_instances = global_bz_instances // dp_world_size
            if new_rank_mbz_instances > rank_mbz_instances:
                for divisor in range(2, new_rank_mbz_instances + 1):
                    if (
                        new_rank_mbz_instances % divisor == 0
                        and new_rank_mbz_instances // divisor < rank_mbz_instances
                    ):
                        new_rank_mbz_instances //= divisor
                        break
                else:
                    raise RuntimeError("shouldn't get here")

            assert new_rank_mbz_instances <= rank_mbz_instances
            assert global_bz_instances % (new_rank_mbz_instances * dp_world_size) == 0

            new_rank_mbz = new_rank_mbz_instances * self.sequence_length
            log.warning(
                f"Adjusting rank micro-batch size from {rank_mbz:,d} tokens ({rank_mbz_instances:,d} instances) "
                f"down to {new_rank_mbz:,d} tokens ({new_rank_mbz_instances:,d} instances) to be compatible "
                "with data parallel world size"
            )
            rank_mbz = new_rank_mbz
        return TransformerTrainModuleConfig(
            rank_microbatch_size=rank_mbz,
            max_sequence_length=self.sequence_length,
            optim=self.get_optim_config(),
            max_grad_norm=1.0,
            scheduler=CosWithWarmup(warmup_steps=round(self.model_size / global_bz)),
        )

    def get_trainer_config(
        self,
        *,
        size: ModelSize,
        run_duration: RunDuration,
        gpu_type: str,
        dp_world_size: int,
        cluster: str,
    ) -> TrainerConfig:
        """
        Build the trainer config.

        :param size: The target model size.
        :param gpu_type: The type of GPU as given by ``torch.cuda.get_device_name()``.
        :param dp_world_size: The data parallel world size.
        """
        del dp_world_size

        from olmo_eval import list_tasks

        if size not in self.SUPPORTED_MODEL_SIZES:
            raise OLMoConfigurationError(
                f"Size {size} not supported by this ladder (supported sizes: {self.SUPPORTED_MODEL_SIZES})."
            )

        # For training runs where we don't expect the model to acquire MC (e.g., 1B-5xC, short 7B training runs)
        tasks = [
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

        config = TrainerConfig(
            save_folder=self.get_save_folder(size, run_duration),
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=self.get_duration(run_duration),
        )
        if gpu_type not in ("cpu", "mps"):
            config = config.with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        config = config.with_callback("config_saver", ConfigSaverCallback())
        config = config.with_callback("garbage_collector", GarbageCollectorCallback())
        config = config.with_callback("monkey_patcher", MonkeyPatcherCallback())
        config = config.with_recommended_evals(
            self.tokenizer,
            self.sequence_length,
            cluster,
            task_set="fast",
            eval_interval=1000,
        )
        config = config.with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,  # large enough value that we won't save until the end
                ephemeral_save_interval=250,
                save_async=True,
            ),
        )
        config = config.with_callback(
            "comet",
            CometCallback(
                name=f"{self.name}-{size}-{run_duration}",
                workspace="ai2",
                project=self.project,
                enabled=False,
                cancel_check_interval=5,
            ),
        )
        config = config.with_callback(
            "wandb",
            WandBCallback(
                name=f"{self.name}-{size}-{run_duration}",
                entity="ai2",
                project=self.project,
                enabled=True,
                cancel_check_interval=5,
            ),
        )
        return config

    def validate(self):
        """
        Validate the ladder configuration.

        :raises OLMoConfigurationError: If the ladder has any issues.
        """
        for size in self.SUPPORTED_MODEL_SIZES:
            # validating to match old ladder sizes.
            if size == ModelSize.size_1B:
                target_size = 1.3
            elif size == ModelSize.size_3B:
                target_size = 3.2
            else:
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

            self.get_optim_config()
            self.get_rank_microbatch_size(size=size, gpu_type="H100")
            bz_tokens = self.get_global_batch_size()
            if bz_tokens % self.sequence_length != 0:
                raise OLMoConfigurationError(
                    f"Batch size of {bz_tokens:,d} tokens for model size {size} "
                    f"must be divisible by the sequence length ({self.sequence_length})"
                )
