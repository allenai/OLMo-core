"""
Callback for converting the final checkpoint to HuggingFace format at the end of training.
"""

import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd

from olmo_core.config import DType
from olmo_core.distributed.utils import barrier, get_rank

from .callback import Callback
from .checkpointer import CheckpointerCallback

log = logging.getLogger(__name__)


@dataclass
class HFConverterCallback(Callback):
    """
    Converts the final saved checkpoint to HuggingFace format at the end of a training job.

    This callback runs after training completes and uses
    :func:`olmo_core.nn.hf.convert_checkpoint_to_hf` to convert the final OLMo Core
    checkpoint to a HuggingFace-compatible format.

    .. note::
        This callback requires the ``transformers`` library to be installed.

    .. warning::
        In distributed training, ALL ranks must participate in this callback because
        gathering the full model state dict from FSDP requires collective operations.
        Only rank 0 performs the actual HF conversion and saving.
    """

    priority: ClassVar[int] = -1  # Run after checkpointer callback.

    enabled: bool = True
    """
    Whether this callback is enabled. Set to ``False`` to disable HF conversion.
    """

    output_folder: Optional[str] = None
    """
    The folder to save the HuggingFace checkpoint to. If not specified, defaults to
    ``{checkpoint_path}-hf`` where ``checkpoint_path`` is the final checkpoint path.
    """

    dtype: Optional[DType] = DType.bfloat16
    """
    The dtype to save the HuggingFace model weights as. Defaults to bfloat16.
    """

    validate: bool = False
    """
    Whether to validate the converted model against the original model.
    Validation loads both models and compares their outputs.
    """

    debug: bool = False
    """
    Whether to output debug information during validation.
    Only has an effect if ``validate`` is ``True``.
    """

    tokenizer_id: Optional[str] = None
    """
    The HuggingFace tokenizer identifier to save with the model.
    If not specified, uses the tokenizer from the experiment config.
    """

    max_sequence_length: Optional[int] = None
    """
    The maximum sequence length for the model. If not specified, uses the tokenizer's
    default max length.
    """

    device: Optional[str] = None
    """
    The device to use for conversion. Defaults to CPU.
    """

    moe_capacity_factor: Optional[float] = None
    """
    The MoE capacity factor. Higher values can decrease validation false negatives
    but may cause OOM errors. Only relevant for MoE models.
    """

    def _get_checkpointer_callback(self) -> Optional[CheckpointerCallback]:
        for callback in self.trainer.callbacks.values():
            if isinstance(callback, CheckpointerCallback):
                return callback
        return None

    def _get_latest_checkpoint_path(self) -> Optional[str]:
        checkpointer = self._get_checkpointer_callback()
        if checkpointer is None:
            log.warning("CheckpointerCallback not found, cannot determine latest checkpoint path")
            return None

        if checkpointer._latest_checkpoint_path:
            return checkpointer._latest_checkpoint_path

        if checkpointer._checkpoints:
            return checkpointer._checkpoints[-1]

        return None

    def _get_full_model_state_dict(self) -> Dict[str, Any]:
        """
        Get the full model state dict from the trainer's model.

        This is a collective operation - ALL ranks must call this method.
        The full state dict is gathered to rank 0.
        """
        model = self.trainer.train_module.model
        # full_state_dict=True gathers the complete model state to rank 0.
        # cpu_offload=True avoids GPU OOM for large models.
        sd_options = dist_cp_sd.StateDictOptions(full_state_dict=True, cpu_offload=True)
        return dist_cp_sd.get_model_state_dict(model, options=sd_options)

    def post_train(self):
        # NOTE: In distributed training with FSDP, getting the full model state dict requires
        # ALL ranks to participate in the collective operation. Only rank 0 performs the actual
        # HF conversion; all ranks synchronize at a barrier before returning.

        if not self.enabled:
            log.info("HFConverterCallback is disabled, skipping conversion")
            barrier()
            return

        checkpoint_path = self._get_latest_checkpoint_path()
        if checkpoint_path is None:
            log.warning("No checkpoint found, skipping HF conversion")
            barrier()
            return

        try:
            from olmo_core.nn.hf import convert_checkpoint_to_hf, load_config
        except ImportError:
            log.error(
                "Failed to import HF conversion utilities. "
                "Make sure the 'transformers' library is installed."
            )
            barrier()
            return

        experiment_config: Optional[dict] = None
        if get_rank() == 0:
            try:
                experiment_config = load_config(checkpoint_path)
            except Exception as e:
                log.error(f"Failed to load config from checkpoint: {e}")

        # ALL ranks must participate in gathering the full state dict (FSDP collective).
        log.info("Gathering full model state dict (collective operation)...")
        try:
            model_state_dict = self._get_full_model_state_dict()
        except Exception as e:
            log.error(f"Failed to get model state dict: {e}")
            barrier()
            raise

        if get_rank() == 0:
            log.info(f"Converting checkpoint at '{checkpoint_path}' to HuggingFace format")

            if experiment_config is None:
                log.error("Experiment config not found in checkpoint, cannot convert to HF format")
                barrier()
                return

            transformer_config_dict = experiment_config.get("model")
            tokenizer_config_dict = experiment_config.get("dataset", {}).get("tokenizer")

            if transformer_config_dict is None:
                log.error(
                    "Model config not found in experiment config, cannot convert to HF format"
                )
                barrier()
                return

            if tokenizer_config_dict is None:
                log.warning(
                    "Tokenizer config not found in experiment config, "
                    "conversion will proceed without tokenizer"
                )
                tokenizer_config_dict = {}

            if self.output_folder is not None:
                output_path = self.output_folder
            else:
                output_path = checkpoint_path + "-hf"

            device = torch.device(self.device) if self.device else None

            try:
                convert_checkpoint_to_hf(
                    original_checkpoint_path=checkpoint_path,
                    output_path=output_path,
                    transformer_config_dict=transformer_config_dict,
                    tokenizer_config_dict=tokenizer_config_dict,
                    model_state_dict=model_state_dict,
                    dtype=self.dtype,
                    tokenizer_id=self.tokenizer_id,
                    max_sequence_length=self.max_sequence_length,
                    validate=self.validate,
                    debug=self.debug,
                    device=device,
                    moe_capacity_factor=self.moe_capacity_factor,
                )
                log.info(
                    f"Successfully converted checkpoint to HuggingFace format at '{output_path}'"
                )
            except Exception as e:
                log.error(f"Failed to convert checkpoint to HuggingFace format: {e}")
                barrier()
                raise

        barrier()
