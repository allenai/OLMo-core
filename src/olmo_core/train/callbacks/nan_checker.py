import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch
import torch.nn as nn

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class NaNCheckerCallback(Callback):
    """
    Checks for NaN values in the output of every layer during training.

    This callback registers forward hooks on all modules in the model to detect
    NaN values in layer outputs. When NaN values are detected, it raises a
    :class:`RuntimeError` and halts training.
    """

    steps: Optional[List[int]] = None
    """
    List of step numbers at which to check for NaNs. If ``None`` (default), checks every step.
    """

    enabled: bool = True
    """
    Set to ``False`` to disable NaN checking.
    """

    def pre_train(self):
        """
        Register forward hooks on all modules before training starts.
        """
        if not self.enabled:
            return

        # Get the model from the train module
        model = self.trainer.train_module.model

        # Register hooks on all modules
        count = 0
        for name, module in model.named_modules():
            # Skip the root module to avoid redundant checks
            if name == "":
                continue
            module.register_forward_hook(self._make_nan_check_hook(name))
            count += 1

        log.info(f"NaN checker registered hooks on {count} modules")

    def _make_nan_check_hook(self, module_name: str):
        """
        Create a forward hook that checks for NaN values in the output.

        :param module_name: The name of the module for logging purposes.
        """

        def hook(module: nn.Module, input: Any, output: Any):
            del module
            del input
            if not self.enabled:
                return

            # Check if we should run at this step
            if self.steps is not None and self.trainer.global_step not in self.steps:
                return

            # Check different output types
            has_nan = False
            if isinstance(output, torch.Tensor):
                has_nan = torch.isnan(output).any().item()
            elif isinstance(output, (tuple, list)):
                for i, tensor in enumerate(output):
                    if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any().item():
                        has_nan = True
                        break
            elif isinstance(output, dict):
                for key, tensor in output.items():
                    if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any().item():
                        has_nan = True
                        break

            if has_nan:
                raise RuntimeError(
                    f"NaN detected in layer '{module_name}' output at step {self.trainer.global_step}"
                )

        return hook
