"""Generation module for autoregressive text generation."""

import contextlib
import logging
import tempfile
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from torch.distributed import DeviceMesh, ProcessGroup
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.stateful import Stateful

from olmo_core.aliases import PathOrStr
from olmo_core.config import Config, DType
from olmo_core.distributed.checkpoint import (
    get_checkpoint_metadata,
    load_state_dict,
)
from olmo_core.distributed.parallel import (
    build_world_mesh,
    get_dp_process_group,
)
from olmo_core.distributed.utils import (
    get_fs_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    scatter_object,
)
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.generate.selection import temperature_sampling
from olmo_core.internal.experiment import ExperimentConfig
from olmo_core.io import is_url, normalize_path
from olmo_core.nn.transformer import Transformer, TransformerConfig
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.train.train_module.transformer.config import (
    TransformerDataParallelConfig,
    TransformerTensorParallelConfig,
)
from olmo_core.utils import get_default_device, move_to_device

log = logging.getLogger(__name__)


@dataclass
class GenerationConfig(Config):
    """Configuration for text generation."""

    max_length: int = 8192
    """Maximum length of generated sequences."""

    pad_token_id: Optional[int] = None
    """Padding token ID."""

    eos_token_id: Optional[int] = None
    """End of sequence token ID."""

    temperature: float = 0.0
    """Temperature for sampling. If 0, this is equivalent to greedy selection."""


class GenerationModule(Stateful, metaclass=ABCMeta):
    @property
    def dp_process_group(self) -> Optional[dist.ProcessGroup]:
        """
        Should return the data parallel process group if it's anything other than the default
        process group.
        """
        return None

    def state_dict_to_load(self, metadata: Metadata) -> Dict[str, Any]:
        del metadata
        return self.state_dict()

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load a state dict.
        """
        raise NotImplementedError


class TransformerGenerationModule(GenerationModule):
    """Module for autoregressive text generation with transformer models."""

    def __init__(
        self,
        model: Transformer,
        generation_config: GenerationConfig,
        compile_model: bool = False,
        float8_config: Optional[Float8Config] = None,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        tp_config: Optional[TransformerTensorParallelConfig] = None,
        autocast_precision: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        state_dict_load_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        state_dict_save_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        load_key_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__()

        # Build world mesh.
        self.device = device or get_default_device()
        self.world_mesh: Optional[DeviceMesh] = None
        if is_distributed():
            self.world_mesh = build_world_mesh(
                dp=dp_config, tp=tp_config, device_type=self.device.type
            )
            log.info(f"Data parallel world size = {get_world_size(self.dp_process_group):,d}")
        elif dp_config is not None or tp_config is not None:
            raise OLMoConfigurationError(
                "Training parallelism configs are only valid for distributed training"
            )

        # Parallelize model.
        self.model = parallelize_model(
            model,
            world_mesh=self.world_mesh,
            device=self.device,
            compile_model=compile_model,
            float8_config=float8_config,
            dp_config=dp_config,
            tp_config=tp_config,
        )

        self._dp_config = dp_config
        self._tp_config = tp_config
        self.autocast_precision = autocast_precision
        self.state_dict_save_opts = state_dict_save_opts or dist_cp_sd.StateDictOptions(strict=True)
        self.state_dict_load_opts = state_dict_load_opts or dist_cp_sd.StateDictOptions(strict=True)
        self.load_key_mapping = load_key_mapping
        self.generation_config = generation_config

    @property
    def dp_process_group(self) -> Optional[dist.ProcessGroup]:
        return None if self.world_mesh is None else get_dp_process_group(self.world_mesh)

    @property
    def tp_enabled(self) -> bool:
        return self._tp_config is not None

    @cached_property
    def world_size(self) -> int:
        return get_world_size()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": dist_cp_sd.get_model_state_dict(self.model, options=self.state_dict_save_opts),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        dist_cp_sd.set_model_state_dict(
            self.model, state_dict["model"], options=self.state_dict_load_opts
        )

    @contextlib.contextmanager
    def _model_forward_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if self.autocast_precision is not None:
                stack.enter_context(torch.autocast(self.device.type, dtype=self.autocast_precision))
            yield

    def model_forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Run a forward pass on a micro-batch, returning the logits.
        """
        with self._model_forward_context():
            logits = self.model(input_ids, **kwargs)
            return logits

    @torch.no_grad()
    def generate_batch(
        self,
        input_ids: torch.Tensor,
        **generation_kwargs,
    ) -> torch.Tensor:
        """
        Generate text using greedy decoding.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            **generation_kwargs: Generation c onfiguration overrides
        Returns:
            Generated token IDs of shape (batch_size, output_length)
        """
        self.model.eval()

        # Move input_ids to the right device.
        input_ids = move_to_device(input_ids, self.device)

        generation_config = self.generation_config.replace(**generation_kwargs)

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Track which sequences are finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Start with the input
        generated = input_ids

        while generated.shape[1] < generation_config.max_length:
            logits = self.model_forward(generated)  # (batch_size, seq_len, vocab_size)
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

            next_tokens = temperature_sampling(  # TODO: make this modular
                next_token_logits, temperature=self.generation_config.temperature
            )

            # Handle finished sequences
            if generation_config.eos_token_id is not None:
                # Check which sequences have generated EOS
                just_finished = next_tokens == generation_config.eos_token_id
                finished = finished | just_finished

                # Replace tokens for finished sequences with padding
                if generation_config.pad_token_id is not None:
                    next_tokens = torch.where(
                        finished,
                        generation_config.pad_token_id,
                        next_tokens,
                    )

            # Append next tokens
            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=1)

            # Stop if all sequences are finished
            if generation_config.eos_token_id is not None and finished.all():
                break

        return generated

    def load_checkpoint(
        self,
        checkpoint_dir: PathOrStr,
        work_dir: PathOrStr,
        process_group: Optional[ProcessGroup] = None,
        pre_download: bool = True,
        load_thread_count: Optional[int] = None,
    ):
        """
        Load model checkpoint.

        Args:
            checkpoint_dir: Path to checkpoint directory
            work_dir: Working directory for caching remote checkpoints
            process_group: Process group for distributed loading
            pre_download: Whether to pre-download remote checkpoints
            load_thread_count: Number of threads to use for loading the checkpoint

        Raises:
            FileNotFoundError: If checkpoint directory doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        work_dir = Path(work_dir)
        if get_fs_local_rank() == 0:
            work_dir.mkdir(exist_ok=True, parents=True)

        checkpoint_dir = normalize_path(checkpoint_dir)

        train_module_dir = f"{checkpoint_dir}/model_and_optim"
        metadata: Optional[Metadata] = None
        if get_rank(process_group) == 0:
            try:
                metadata = get_checkpoint_metadata(train_module_dir)
            except FileNotFoundError:
                # Try base directory, which could be the case if user is trying to
                # load model weights and not an actual train checkpoint.
                log.warning(
                    f"Checkpoint metadata not found in '{train_module_dir}', "
                    f"trying base directory '{checkpoint_dir}'"
                )
                metadata = get_checkpoint_metadata(checkpoint_dir)
                train_module_dir = checkpoint_dir

        train_module_dir = scatter_object(train_module_dir)
        if metadata is None:
            metadata = get_checkpoint_metadata(train_module_dir)

        state_dict = self.state_dict_to_load(metadata)
        load_state_dict(
            train_module_dir,
            state_dict,  # state is loaded into here, in-place
            process_group=process_group,
            pre_download=is_url(checkpoint_dir) and pre_download,
            work_dir=work_dir,
            thread_count=load_thread_count,
        )
        self.load_state_dict(state_dict)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: PathOrStr,
        *,
        transformer_config: Optional[TransformerConfig] = None,
        generation_config: Optional[GenerationConfig] = None,
        process_group: Optional[ProcessGroup] = None,
        pre_download: bool = True,
        load_thread_count: Optional[int] = None,
        **kwargs,
    ) -> "TransformerGenerationModule":
        """
        Create a GenerationModule from a checkpoint.

        This is a convenience method that combines model initialization and checkpoint loading.

        Args:
            checkpoint_dir: Path to checkpoint directory.
            transformer_config: Configuration for the transformer model. If not provided,
                will be loaded from the checkpoint's config.json file.
            generation_config: Configuration for generation. If not provided, uses default
                GenerationConfig.
            process_group: Process group for distributed checkpoint loading.
            pre_download: Whether to pre-download remote checkpoints.
            load_thread_count: Number of threads to use for loading checkpoint.
            **kwargs: Additional keyword arguments passed to the TransformerGenerationModule
                constructor.

        Returns:
            TransformerGenerationModule instance with loaded checkpoint.

        Raises:
            FileNotFoundError: If checkpoint directory doesn't exist.
            OLMoConfigurationError: If transformer config cannot be determined.
            RuntimeError: If checkpoint loading fails.
        """
        checkpoint_dir = Path(normalize_path(checkpoint_dir))
        generation_config = generation_config or GenerationConfig()

        # Load transformer config from checkpoint if not provided
        if transformer_config is None and get_rank(process_group) == 0:
            experiment_config = ExperimentConfig.from_file(checkpoint_dir / "config.json")
            transformer_config = experiment_config.model

        # Create work directory on rank 0
        work_dir = Path(tempfile.mkdtemp()) if get_rank(process_group) == 0 else Path("/tmp")

        # Broadcast config and work_dir to all ranks
        transformer_config = scatter_object(transformer_config)
        work_dir = scatter_object(work_dir)

        if transformer_config is None:
            raise OLMoConfigurationError("Transformer config is required")

        # Build model and generation module
        model = transformer_config.build()
        generation_module = cls(model, generation_config, **kwargs)

        # Load checkpoint
        generation_module.load_checkpoint(
            checkpoint_dir=checkpoint_dir,
            process_group=process_group,
            work_dir=work_dir,
            pre_download=pre_download,
            load_thread_count=load_thread_count,
        )

        return generation_module


@beta_feature
@dataclass
class TransformerGenerationModuleConfig(Config):
    """
    A configuration class for building :class:`TransformerGenerationModule` instances.
    """

    # Generation settings.
    generation_config: GenerationConfig = field(default_factory=lambda: GenerationConfig())

    # Model settings.
    compile_model: bool = False
    float8_config: Optional[Float8Config] = None
    dp_config: Optional[TransformerDataParallelConfig] = None
    tp_config: Optional[TransformerTensorParallelConfig] = None

    # Checkpoint settings.
    state_dict_load_opts: Optional[Dict[str, Any]] = None
    load_key_mapping: Optional[Dict[str, str]] = None

    # Other settings.
    autocast_precision: Optional[DType] = None

    def build(
        self,
        checkpoint_dir: Optional[PathOrStr] = None,
        device: Optional[torch.device] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        work_dir: Optional[Path] = None,
        pre_download: bool = True,
        load_thread_count: Optional[int] = None,
    ) -> "TransformerGenerationModule":
        """
        Build the corresponding :class:`TransformerGenerationModule`.

        :param model: The :class:`~olmo_core.nn.transformer.Transformer` model to use for generation.
        :param device: The device to use for generation.
        :param checkpoint_dir: Optional checkpoint directory to load from.
        :param process_group: The process group for distributed loading.
        :param work_dir: Working directory for temporary files during loading.
        :param pre_download: Whether to pre-download remote checkpoints.
        :param load_thread_count: Number of threads to use for loading.
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)
        if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)

        return TransformerGenerationModule.from_checkpoint(
            checkpoint_dir=checkpoint_dir,
            process_group=process_group,
            work_dir=work_dir,
            pre_download=pre_download,
            load_thread_count=load_thread_count,
            **kwargs,
        )
