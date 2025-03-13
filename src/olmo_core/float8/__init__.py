"""
Utilities for training in Float8 via `torchao <https://github.com/pytorch/ao>`_.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Set, Union

import torch
import torch.nn as nn

from ..config import Config, StrEnum
from ..exceptions import OLMoConfigurationError

__all__ = ["Float8Handler", "Float8ScalingType", "Float8Config"]

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

    :param config: The handler config.
    """

    def __init__(self, config: "Float8Config"):
        self.config = config
        if not self.config.enabled:
            return

        if not _is_sm89_or_later():
            raise RuntimeError("Float8 training is only supported on SM89 or later")

        # for precompute_float8_dynamic_scale_for_fsdp
        self.precompute_scale = (
            self.config.enable_fsdp_float8_all_gather
            and self.config.precompute_float8_dynamic_scale_for_fsdp
        )

        # for sync_float8_amax_and_scale_history
        self.delayed_scaling = (
            self.config.scaling_type_input == Float8ScalingType.delayed
            or self.config.scaling_type_weight == Float8ScalingType.delayed
            or self.config.scaling_type_grad_output == Float8ScalingType.delayed
        )
        self._sync_float8_amax_and_scale_history = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def convert_to_float8_training(
        self, model: nn.Module, modules_to_ignore: Optional[Set[str]] = None
    ):
        """
        This just calls out to :meth:`Float8Config.convert_to_float8_training()`.
        """
        if not self.enabled:
            return

        # NOTE: there's a bug with `Float8Linear.from_float()` where it will override `requires_grad=False`
        # when `enable_fsdp_float8_all_gather=True`. So we have to reset frozen params after the fact.
        # https://github.com/pytorch/ao/issues/1871
        frozen_params: Set[str] = set()
        for n, p in model.named_parameters():
            if not p.requires_grad:
                frozen_params.add(n)

        self.config.convert_to_float8_training(model, modules_to_ignore=modules_to_ignore)

        for n in frozen_params:
            p = model.get_parameter(n)
            p.requires_grad = False

    def precompute_float8_dynamic_scale_for_fsdp(self, model: Union[nn.Module, List[nn.Module]]):
        if not self.enabled:
            return

        if not self.precompute_scale:
            return

        from torchao.float8 import (  # type: ignore
            precompute_float8_dynamic_scale_for_fsdp,
        )

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            precompute_float8_dynamic_scale_for_fsdp(m)

    def sync_float8_amax_and_scale_history(self, model: Union[nn.Module, List[nn.Module]]):
        if not self.enabled:
            return

        if not self.delayed_scaling:
            return

        from torchao.float8 import sync_float8_amax_and_scale_history  # type: ignore

        if self._sync_float8_amax_and_scale_history is None:
            if self.config.compile:
                self._sync_float8_amax_and_scale_history = torch.compile(
                    sync_float8_amax_and_scale_history
                )
            else:
                self._sync_float8_amax_and_scale_history = sync_float8_amax_and_scale_history

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            self._sync_float8_amax_and_scale_history(m)  # type: ignore


@dataclass
class Float8Config(Config):
    """
    A configuration class for building :class:`Float8Handler` objects.

    :param scaling_type_input: Float8 scaling for inputs.
    :param scaling_type_weight: Float8 scaling for weights.
    :param scaling_type_grad_output: Float8 scaling for grad outputs.
    :param enable_fsdp_float8_all_gather: Cast ``Float8Linear.weight`` from high precision to
        float8 before FSDP all-gather so we can communicate in float8 to save bandwidth.
    :param precompute_float8_dynamic_scale_for_fsdp: Communicate AMAX/scales efficiently in a single
        all-reduce for all parameters instead of doing many small all-reduce for each parameter.
    :param compile: If using ``torch.compile``.
    :param enabled: If ``False`` this will be a no-op.
    """

    scaling_type_input: Float8ScalingType = Float8ScalingType.dynamic
    scaling_type_weight: Float8ScalingType = Float8ScalingType.dynamic
    scaling_type_grad_output: Float8ScalingType = Float8ScalingType.dynamic
    enable_fsdp_float8_all_gather: bool = True
    precompute_float8_dynamic_scale_for_fsdp: bool = True
    force_recompute_fp8_weight_in_bwd: bool = True
    compile: Optional[bool] = None
    enabled: bool = True

    @property
    def float8_linear_config(self):
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

        scaling_type_input = ScalingType(self.scaling_type_input)
        scaling_type_weight = ScalingType(self.scaling_type_weight)
        scaling_type_grad_output = ScalingType(self.scaling_type_grad_output)
        return Float8LinearConfig(
            enable_fsdp_float8_all_gather=self.enable_fsdp_float8_all_gather,
            force_recompute_fp8_weight_in_bwd=self.force_recompute_fp8_weight_in_bwd,
            cast_config_input=CastConfig(scaling_type=scaling_type_input),
            cast_config_weight=CastConfig(scaling_type=scaling_type_weight),
            cast_config_grad_output=CastConfig(scaling_type=scaling_type_grad_output),
            enable_pre_and_post_forward=False,
        )

    def convert_to_float8_training(
        self, model: nn.Module, modules_to_ignore: Optional[Set[str]] = None
    ):
        """
        This method converts the linear layers of ``model`` to ``Float8Linear``.

        .. note::
            Note only dynamic tensor scaling (the default) is supported at the moment.

        .. warning::
            This will mutate the model in place.

        .. warning::
            This should be called before compiling the model, applying activation checkpointing,
            or wrapping it with FSDP(2) or any other parallel wrapper.

        .. hint::
            If you're using a :class:`~olmo_core.nn.transformer.Transformer` model the config
            builder (:class:`~olmo_core.nn.transformer.TransformerConfig`) will call this
            automatically from :meth:`~olmo_core.nn.transformer.TransformerConfig.build()` if
            the :data:`~olmo_core.nn.transformer.TransformerConfig.float8_config` field is set.
        """
        if not self.enabled:
            return

        from torchao.float8 import convert_to_float8_training  # type: ignore

        ignored_modules_found = set()

        def module_filter_fn(m: nn.Module, fqn: str) -> bool:
            nonlocal ignored_modules_found
            if modules_to_ignore is not None and fqn in modules_to_ignore:
                ignored_modules_found.add(fqn)
                return False

            # Linear layers must have all dimensions divisible by 16.
            if isinstance(m, nn.Linear):
                for d in m.weight.shape:
                    if d % 16 != 0:
                        return False

            return True

        # Mutates the model in place, replacing instances of nn.Linear with Float8Linear.
        convert_to_float8_training(
            model,
            config=self.float8_linear_config,
            module_filter_fn=module_filter_fn,
        )

        if modules_to_ignore is not None and modules_to_ignore != ignored_modules_found:
            raise OLMoConfigurationError(
                f"invalid module name(s) in 'modules_to_ignore': {list(modules_to_ignore - ignored_modules_found)}"
            )

    def build(self) -> Float8Handler:
        """
        Build the Float8 handler.
        """
        return Float8Handler(config=self)


def _is_sm89_or_later():
    # Float8 is only supported on SM89 or later (H100+ GPUs)
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
