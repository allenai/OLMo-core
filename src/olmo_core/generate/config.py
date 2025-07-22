from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd

from olmo_core.aliases import PathOrStr
from olmo_core.config import Config, DType
from olmo_core.doc_utils import beta_feature
from olmo_core.float8 import Float8Config
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train.train_module.transformer.config import (
    TransformerDataParallelConfig,
)

if TYPE_CHECKING:
    from .generation import TransformerGenerationModule


@dataclass
class GenerationConfig(Config):
    """Configuration for text generation."""

    pad_token_id: int
    """Padding token ID."""

    eos_token_id: int
    """End of sequence token ID."""

    max_length: int = 8192
    """Maximum length of generated sequences."""

    do_sample: bool = True
    """Whether to use sampling for generation. If False, greedy decoding is used. This overrides temperature, top_k, and top_p."""

    temperature: float = 0.0
    """Temperature for sampling. If 0, this is equivalent to greedy selection."""

    top_k: int = -1
    """Top-k sampling. Only consider the top k tokens with the highest probabilities. -1 means no filtering."""

    top_p: float = 1.0
    """Top-p (nucleus) sampling. Only consider the smallest set of tokens whose cumulative probability exceeds this threshold. 1.0 means no filtering."""

    use_cache: bool = True
    """Whether to use a kv-cache for generation. If True, the model will cache past key-value pairs to speed up generation."""

    stop_sequences: Optional[List[List[int]]] = None
    """Stop sequences. If provided, generation will stop when any of these sequences of tokens
    are generated (in addition to the EOS token)."""

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate the generation configuration."""
        if self.pad_token_id < 0:
            raise ValueError(f"pad_token_id must be non-negative, got {self.pad_token_id}")
        if self.eos_token_id < 0:
            raise ValueError(f"eos_token_id must be non-negative, got {self.eos_token_id}")
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        if self.top_k <= 0 and self.top_k != -1:
            raise ValueError(f"top_k must be positive or -1, got {self.top_k}")
        if self.top_p <= 0.0 or self.top_p > 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")


@beta_feature
@dataclass
class TransformerGenerationModuleConfig(Config):
    """
    A configuration class for building :class:`TransformerGenerationModule` instances.
    """

    # Generation settings.
    generation_config: GenerationConfig

    # Model settings.
    compile_model: bool = False
    float8_config: Optional[Float8Config] = None
    dp_config: Optional[TransformerDataParallelConfig] = None

    # Checkpoint settings.
    state_dict_load_opts: Optional[Dict[str, Any]] = None
    load_key_mapping: Optional[Dict[str, str]] = None

    # Other settings.
    dtype: Optional[DType] = None

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
        from olmo_core.generate.generation import TransformerGenerationModule

        config_dict = self.as_dict(exclude_none=True, recurse=False)
        if (dtype := config_dict.pop("dtype", None)) is not None:
            dtype = DType(dtype)
        if (state_dict_save_opts := config_dict.pop("state_dict_save_opts", None)) is not None:
            config_dict["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(
                **state_dict_save_opts
            )
        if (state_dict_load_opts := config_dict.pop("state_dict_load_opts", None)) is not None:
            config_dict["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(
                **state_dict_load_opts
            )
        print(config_dict)

        return TransformerGenerationModule.from_checkpoint(
            checkpoint_dir=checkpoint_dir,
            transformer_config=transformer_config,
            device=device,
            process_group=process_group,
            work_dir=work_dir,
            pre_download=pre_download,
            load_thread_count=load_thread_count,
            dtype=dtype,
            **config_dict,
        )
