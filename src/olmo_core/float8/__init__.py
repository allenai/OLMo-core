"""
Utilities for training in Float8 via `torchao <https://github.com/pytorch/ao>`_.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Set

import torch.nn as nn
from torchao.utils import torch

from olmo_core.utils import has_compute_capability

from ..config import Config
from ..exceptions import OLMoConfigurationError
from .ao import AOFloat8LinearConfig, AOFloat8LinearRecipe, AOMXLinearConfig

__all__ = ["Float8Config", "AOFloat8LinearConfig", "AOFloat8LinearRecipe", "AOMXLinearConfig"]

log = logging.getLogger(__name__)


@dataclass
class Float8Config(Config):
    """
    A configuration class for specifying Float8 options.

    :param ao: A torchao ``Float8Linear`` linear configuration.
    :param ao_recipe: Alternatively you can specify a recipe name from torchao.
    :param ao_mx: A torchao ``MXLinearConfig`` configuration for MX formats (MXFP8/MXFP4).
    :param enabled: If ``False`` this will be a no-op.
    """

    ao: Optional[AOFloat8LinearConfig] = None
    ao_recipe: Optional[AOFloat8LinearRecipe] = None
    ao_mx: Optional[AOMXLinearConfig] = None

    modules_to_ignore: Optional[List[str]] = None
    """A set of fully-qualified module names to ignore for Float8 conversion."""

    enabled: bool = True

    def __post_init__(self):
        self.validate()

    def validate(self):
        config_count = sum(
            [self.ao is not None, self.ao_recipe is not None, self.ao_mx is not None]
        )
        if config_count > 1:
            raise OLMoConfigurationError(
                "'ao', 'ao_recipe', and 'ao_mx' configs are mutually exclusive"
            )

    @property
    def should_precompute_float8_dynamic_scale_for_fsdp(self):
        if self.ao_recipe is not None or self.ao_mx is not None:
            return False

        float8_linear_config = (
            self.ao if self.ao is not None else AOFloat8LinearConfig()
        ).to_ao_type()
        return float8_linear_config.enable_fsdp_float8_all_gather

    def apply_float8_linear(
        self, model: nn.Module, *, modules_to_ignore: Optional[Set[str]] = None
    ):
        """
        This method converts the linear layers of ``model`` to ``Float8Linear`` or ``MXLinear``.

        .. warning::
            This will mutate the model in place.

        .. warning::
            This should be called before compiling the model, applying activation checkpointing,
            or wrapping it with FSDP(2) or any other parallel wrapper.
        """
        if not self.enabled:
            return

        self.validate()

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

        def quantize_filter_fn(m: nn.Module, fqn: str) -> bool:
            nonlocal ignored_modules_found
            if modules_to_ignore is not None and fqn in modules_to_ignore:
                ignored_modules_found.add(fqn)
                return False
            if isinstance(m, torch.nn.Linear) and hasattr(m, "weight"):
                return True
            return False

        # NOTE: there's a bug with `Float8Linear.from_float()` where it will override `requires_grad=False`
        # when `enable_fsdp_float8_all_gather=True`. So we have to reset frozen params after the fact.
        # https://github.com/pytorch/ao/issues/1871
        frozen_params: Set[str] = set()
        for n, p in model.named_parameters():
            if not p.requires_grad:
                frozen_params.add(n)

        # Handle MX format conversion
        if self.ao_mx is not None:
            if not has_compute_capability(10, 0):
                raise RuntimeError("MX format training is only supported on SM100 or later")

            from torchao.quantization import quantize_ as ao_quantize_

            mx_linear_config = self.ao_mx.to_ao_type()

            ao_quantize_(
                model,
                config=mx_linear_config,
                filter_fn=quantize_filter_fn,  # !!! Opposite semantics of the module_filter_fn below
            )

        else:
            from torchao.float8 import Float8LinearConfig, convert_to_float8_training

            # Mutates the model in place, replacing instances of nn.Linear with Float8Linear.
            float8_linear_config: Float8LinearConfig
            if self.ao_recipe is not None:
                float8_linear_config = Float8LinearConfig.from_recipe_name(
                    self.ao_recipe.to_ao_type()
                )
            else:
                float8_linear_config = (
                    self.ao if self.ao is not None else AOFloat8LinearConfig()
                ).to_ao_type()

            convert_to_float8_training(
                model,
                config=float8_linear_config,
                module_filter_fn=module_filter_fn,
            )

        if modules_to_ignore is not None and modules_to_ignore != ignored_modules_found:
            raise OLMoConfigurationError(
                f"invalid module name(s) in 'modules_to_ignore': {list(modules_to_ignore - ignored_modules_found)}"
            )

        if ignored_modules_found:
            log.info(f"Ignored modules for Float8 conversion: {sorted(ignored_modules_found)}")

        for n in frozen_params:
            p = model.get_parameter(n)
            p.requires_grad = False
