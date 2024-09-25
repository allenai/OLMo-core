"""
Utilities for training in Float8 via `torchao <https://github.com/pytorch/ao>`_.
"""

import logging
from typing import List, Union

import torch
import torch.nn as nn

from .config import StrEnum

__all__ = ["Float8Handler", "Float8ScalingType"]

log = logging.getLogger(__name__)


class Float8ScalingType(StrEnum):
    """
    Float8 scaling type.
    """

    dynamic = "dynamic"
    """
    Dynamic scaling.
    """
    delayed = "delayed"
    """
    Delayed scaling.
    """


class Float8Handler:
    """
    Enables Float8 training with linear layers.

    .. seealso::
        See :class:`~olmo_core.train.callbacks.Float8HandlerCallback` for enabling Float8 training
        with the :class:`~olmo_core.train.Trainer`.

    :param scaling_type_input: Float8 scaling for inputs.
    :param scaling_type_weight: Float8 scaling for weights.
    :param scaling_type_grad_output: Float8 scaling for grad outputs.
    :param enable_fsdp_float8_all_gather: Cast ``Float8Linear.weight`` from high precision to
        float8 before FSDP all-gather so we can communicate in float8 to save bandwidth.
    :param precompute_float8_dynamic_scale_for_fsdp: Communicate AMAX/scales efficiently in a single
        all-reduce for all parameters instead of doing many small all-reduce for each parameter.
    :param compile: If using ``torch.compile``.
    """

    def __init__(
        self,
        *,
        scaling_type_input: Float8ScalingType = Float8ScalingType.dynamic,
        scaling_type_weight: Float8ScalingType = Float8ScalingType.dynamic,
        scaling_type_grad_output: Float8ScalingType = Float8ScalingType.dynamic,
        enable_fsdp_float8_all_gather: bool = True,
        precompute_float8_dynamic_scale_for_fsdp: bool = True,
        compile: bool = True,
    ):
        if not _is_sm89_or_later():
            raise RuntimeError("Float8 training is only supported on SM89 or later")

        try:
            from torchao.float8 import (  # type: ignore
                CastConfig,
                Float8LinearConfig,
                ScalingType,
            )
        except ImportError as e:
            raise ImportError(
                "torchao is not installed. Please install it to use float8 linear layers."
            ) from e

        self.enable_fsdp_float8_all_gather = enable_fsdp_float8_all_gather
        self.compile = compile

        scaling_type_input = ScalingType(scaling_type_input)
        scaling_type_weight = ScalingType(scaling_type_weight)
        scaling_type_grad_output = ScalingType(scaling_type_grad_output)
        self.config = Float8LinearConfig(
            enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
            cast_config_input=CastConfig(scaling_type=scaling_type_input),
            cast_config_weight=CastConfig(scaling_type=scaling_type_weight),
            cast_config_grad_output=CastConfig(scaling_type=scaling_type_grad_output),
            enable_pre_and_post_forward=False,
        )

        # for precompute_float8_dynamic_scale_for_fsdp
        self.precompute_scale = (
            enable_fsdp_float8_all_gather and precompute_float8_dynamic_scale_for_fsdp
        )

        # for sync_float8_amax_and_scale_history
        self.delayed_scaling = (
            scaling_type_input == Float8ScalingType.delayed
            or scaling_type_weight == Float8ScalingType.delayed
            or scaling_type_grad_output == Float8ScalingType.delayed
        )
        self._sync_float8_amax_and_scale_history = None

    def convert_to_float8_training(self, model: nn.Module):
        """
        This method converts the linear layers of ``model`` to ``Float8Linear``.
        Note that today, only dynamic tensor scaling (the default) is supported.
        This will mutate the model in place.
        """
        from torchao.float8 import convert_to_float8_training  # type: ignore

        # Mutates the model in place, replacing instances of nn.Linear with Float8Linear.
        convert_to_float8_training(
            model,
            config=self.config,
            module_filter_fn=lambda _, fqn: fqn != "output",
        )
        log.info("Swapped to Float8Linear layers")

    def precompute_float8_dynamic_scale_for_fsdp(self, model: Union[nn.Module, List[nn.Module]]):
        if not self.precompute_scale:
            return

        from torchao.float8 import (  # type: ignore
            precompute_float8_dynamic_scale_for_fsdp,
        )

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            precompute_float8_dynamic_scale_for_fsdp(m)

    def sync_float8_amax_and_scale_history(self, model: Union[nn.Module, List[nn.Module]]):
        if not self.delayed_scaling:
            return

        from torchao.float8 import sync_float8_amax_and_scale_history  # type: ignore

        if self._sync_float8_amax_and_scale_history is None:
            if self.compile:
                self._sync_float8_amax_and_scale_history = torch.compile(
                    sync_float8_amax_and_scale_history
                )
            else:
                self._sync_float8_amax_and_scale_history = sync_float8_amax_and_scale_history

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            self._sync_float8_amax_and_scale_history(m)  # type: ignore


def _is_sm89_or_later():
    # Float8 is only supported on SM89 or later (H100+ GPUs)
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
