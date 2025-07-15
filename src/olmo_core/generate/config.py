from dataclasses import dataclass, field
from typing import Any, Dict, Optional, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd

from olmo_core.aliases import PathOrStr
from olmo_core.config import Config, DType
from olmo_core.doc_utils import beta_feature
from olmo_core.float8 import Float8Config
from olmo_core.generate.generation import TransformerGenerationModule
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train.train_module.transformer.config import TransformerDataParallelConfig


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

    # Checkpoint settings.
    state_dict_load_opts: Optional[Dict[str, Any]] = None
    load_key_mapping: Optional[Dict[str, str]] = None

    # Other settings.
    autocast_precision: Optional[DType] = None

    def build(
        self,
        checkpoint_dir: PathOrStr,
        transformer_config: Optional[TransformerConfig] = None,
        device: Optional[torch.device] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        work_dir: Optional[PathOrStr] = None,
        pre_download: bool = True,
        load_thread_count: Optional[int] = None,
    ) -> "TransformerGenerationModule":
        """
        Build the corresponding :class:`TransformerGenerationModule`.

        :param checkpoint_dir: Checkpoint directory to load from.
        :param transformer_config: The :class:`~olmo_core.nn.transformer.TransformerConfig` to use for generation.
        :param device: The device to use for generation.
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
            transformer_config=transformer_config,
            device=device,
            process_group=process_group,
            work_dir=work_dir,
            pre_download=pre_download,
            load_thread_count=load_thread_count,
            **kwargs,
        )
