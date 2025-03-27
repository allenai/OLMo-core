"""
Utilities for training in Float8 via `torchao <https://github.com/pytorch/ao>`_.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Set

import torch
import torch.nn as nn

from ..config import Config
from ..exceptions import OLMoConfigurationError
from .ao import AOFloat8LinearConfig, AOFloat8LinearRecipe

__all__ = ["Float8Config", "AOFloat8LinearConfig", "AOFloat8LinearRecipe"]

log = logging.getLogger(__name__)


@dataclass
class Float8Config(Config):
    """
    A configuration class for specifying Float8 options.

    :param ao: A torchao ``Float8Linear`` linear configuration.
    :param ao_recipe: Alternatively you can specify a recipe name from torchao.
    :param enabled: If ``False`` this will be a no-op.
    """

    ao: Optional[AOFloat8LinearConfig] = None
    ao_recipe: Optional[AOFloat8LinearRecipe] = None

    enabled: bool = True

    @property
    def should_precompute_float8_dynamic_scale_for_fsdp(self):
        if self.ao_recipe is not None:
            return False

        float8_linear_config = (
            self.ao if self.ao is not None else AOFloat8LinearConfig()
        ).to_ao_type()
        return float8_linear_config.enable_fsdp_float8_all_gather

    def apply_float8_linear(
        self, model: nn.Module, *, modules_to_ignore: Optional[Set[str]] = None
    ):
        """
        This method converts the linear layers of ``model`` to ``Float8Linear``.

        .. warning::
            This will mutate the model in place.

        .. warning::
            This should be called before compiling the model, applying activation checkpointing,
            or wrapping it with FSDP(2) or any other parallel wrapper.
        """
        if not self.enabled:
            return

        if not (torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)):
            raise RuntimeError("Float8 training is only supported on SM89 or later")

        if self.ao_recipe is not None and self.ao is not None:
            raise OLMoConfigurationError("'ao_recipe' and 'ao' config are mutually exclusive")

        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

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

        # NOTE: there's a bug with `Float8Linear.from_float()` where it will override `requires_grad=False`
        # when `enable_fsdp_float8_all_gather=True`. So we have to reset frozen params after the fact.
        # https://github.com/pytorch/ao/issues/1871
        frozen_params: Set[str] = set()
        for n, p in model.named_parameters():
            if not p.requires_grad:
                frozen_params.add(n)

        # Mutates the model in place, replacing instances of nn.Linear with Float8Linear.
        float8_linear_config: Float8LinearConfig
        if self.ao_recipe is not None:
            float8_linear_config = Float8LinearConfig.from_recipe_name(self.ao_recipe.to_ao_type())
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

        for n in frozen_params:
            p = model.get_parameter(n)
            p.requires_grad = False
