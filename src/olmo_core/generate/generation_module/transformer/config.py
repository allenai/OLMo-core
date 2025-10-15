import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd

from olmo_core.aliases import PathOrStr
from olmo_core.config import Config, DType
from olmo_core.doc_utils import beta_feature
from olmo_core.float8 import Float8Config
from olmo_core.generate.generation_module.config import GenerationConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train.train_module.transformer.config import (
    TransformerDataParallelConfig,
)
from olmo_core.utils import log_or_print

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from . import TransformerGenerationModule


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
    """The dtype to build the model in."""

    def build(
        self,
        checkpoint_dir: PathOrStr | List[PathOrStr],
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
        from olmo_core.generate.generation_module import TransformerGenerationModule

        config_dict = self.as_dict(exclude_none=True, recurse=False)
        if (dtype := config_dict.pop("dtype", None)) is not None:
            dtype = DType(dtype)
        if (autocast_precision := config_dict.pop("autocast_precision", None)) is not None:
            config_dict["autocast_precision"] = DType(autocast_precision).as_pt()
        if (state_dict_save_opts := config_dict.pop("state_dict_save_opts", None)) is not None:
            config_dict["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(
                **state_dict_save_opts
            )
        if (state_dict_load_opts := config_dict.pop("state_dict_load_opts", None)) is not None:
            config_dict["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(
                **state_dict_load_opts
            )
        log_or_print(log, f"TransformerGenerationModuleConfig: {config_dict}")

        return TransformerGenerationModule.from_checkpoints(
            checkpoint_dirs=[checkpoint_dir] if isinstance(checkpoint_dir, PathOrStr) else checkpoint_dir,  # type: ignore  # mypy bug with Union isinstance
            transformer_config=transformer_config,
            device=device,
            process_group=process_group,
            work_dir=work_dir,
            pre_download=pre_download,
            load_thread_count=load_thread_count,
            dtype=dtype,
            **config_dict,
        )
