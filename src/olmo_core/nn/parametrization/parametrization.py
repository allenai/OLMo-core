import math
from abc import abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...config import StrEnum
from .config import MupScalingStrategy, ParametrizationOptimizerType

__all__ = ["ParametrizationBase"]


class ParameterScalingType(StrEnum):
    """
    Possible ways a parameter can scale with model ``width``.
    """

    constant = "constant"
    """
    The parameter's input and output are independent of width, so the parametrization does not adjust it.
    """

    scaling_input = "scaling_input"
    """
    The input of the parameter scales with width but the output does not. This is called an *output
    weight* in the Maximal Update Parametrization paper.
    """

    scaling_output = "scaling_output"
    """
    The output of the parameter scales with width but the input does not. This is called an *input
    weight* in the Maximal Update Parametrization paper. This also applies to biases.
    """

    scaling_input_output = "scaling_input_output"
    """
    Both the input and output of the parameter scale with width. This is called a *hidden weight* in
    the Maximal Update Parametrization paper.
    """


class ParametrizationBase:
    """
    Base class for parametrization implementations.
    """

    def __init__(
        self,
        *,
        optimizer: ParametrizationOptimizerType,
        input_dim: int,
        output_dim: int,
    ):
        self.optimizer = optimizer
        self.input_dim = input_dim
        self.output_dim = output_dim

    @property
    @abstractmethod
    def input_multiplier(self) -> Optional[float]:
        pass

    @property
    @abstractmethod
    def init_std_multiplier(self) -> Optional[float]:
        pass

    @property
    @abstractmethod
    def lr_multiplier(self) -> Optional[float]:
        pass

    @property
    @abstractmethod
    def softmax_scale_multiplier(self) -> Optional[float]:
        pass

    def set_base_dims(
        self,
        base_input_dim: Optional[int] = None,
        base_output_dim: Optional[int] = None,
    ) -> None:
        """
        Passes in the input and output dimensions of the base model's corresponding parameter,
        which are used to compute scaling multipliers.
        """
        del base_input_dim, base_output_dim  # Default implementation does nothing.

    @classmethod
    def scale_input(
        cls, parametrization: Optional["ParametrizationBase"], x: torch.Tensor
    ) -> torch.Tensor:
        """
        Convenient wrapper for applying the parametrization input multiplier.
        This just performs ``x * parametrization_input_multiplier``, if the multiplier exists.
        """
        if parametrization is None or parametrization.input_multiplier is None:
            return x

        return (x * parametrization.input_multiplier).to(x.dtype)

    @classmethod
    def scale_init_std(cls, parametrization: Optional["ParametrizationBase"], std: float) -> float:
        """
        Convenient wrapper for applying the parametrization init std multiplier.
        This just performs ``std * parametrization_std_multiplier``, if the multiplier exists.
        """
        if parametrization is None or parametrization.init_std_multiplier is None:
            return std

        return std * parametrization.init_std_multiplier

    @classmethod
    def scale_lr(cls, parametrization: Optional["ParametrizationBase"], lr: float) -> float:
        """
        Convenient wrapper for applying the parametrization LR multiplier.
        This just performs ``lr * parametrization_lr_multiplier``, if the multiplier exists.
        """
        if parametrization is None or parametrization.lr_multiplier is None:
            return lr

        return lr * parametrization.lr_multiplier

    @classmethod
    def scale_coupled_wd(
        cls, parametrization: Optional["ParametrizationBase"], weight_decay: float
    ) -> float:
        """
        Convenient wrapper for un-applying parametrization LR multiplier to coupled weight decay.
        This just performs ``weight_decay / parametrization_lr_multiplier``, if the multiplier exists.
        """
        if parametrization is None or parametrization.lr_multiplier is None:
            return weight_decay

        if not parametrization.optimizer.coupled_weight_decay:
            raise RuntimeError(
                f"Scaling coupled weight decay is not supported for optimizer {parametrization.optimizer} with uncoupled weight decay"
            )

        return weight_decay / parametrization.lr_multiplier

    @classmethod
    def named_parametrizations(cls, model: nn.Module) -> Dict[str, "ParametrizationBase"]:
        """
        An analogue of ``model.named_parameters()`` for getting all the Parametrizations of the model.
        """
        named_parametrizations: Dict[str, "ParametrizationBase"] = {}
        for module_name, module in model.named_modules():
            if not hasattr(module, "parametrizations"):
                continue

            parametrizations = getattr(module, "parametrizations")
            assert isinstance(parametrizations, dict)
            for param_name, parametrization in parametrizations.items():
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                named_parametrizations[full_name] = parametrization

        return named_parametrizations

    @classmethod
    def set_base_model_dims(cls, model: nn.Module, base_model: nn.Module) -> None:
        named_parametrizations = cls.named_parametrizations(model)
        base_named_parametrizations = cls.named_parametrizations(base_model)

        if named_parametrizations.keys() != base_named_parametrizations.keys():
            raise ValueError("Model and base model do not have the same parametrization keys")

        for key in named_parametrizations.keys():
            parametrization = named_parametrizations[key]
            base_parametrization = base_named_parametrizations[key]

            parametrization.set_base_dims(
                base_parametrization.input_dim, base_parametrization.output_dim
            )


class StandardParametrization(ParametrizationBase):
    """
    The standard parametrization, where no scaling is applied.
    """

    @property
    def input_multiplier(self) -> Optional[float]:
        return None

    @property
    def init_std_multiplier(self) -> Optional[float]:
        return None

    @property
    def lr_multiplier(self) -> Optional[float]:
        return None

    @property
    def softmax_scale_multiplier(self) -> Optional[float]:
        return None


class MaximalUpdateParametrization(ParametrizationBase):
    """
    Encapsulates scaling factors for applying the parametrization to a single parameter.
    """

    def __init__(
        self,
        *,
        optimizer: ParametrizationOptimizerType,
        scaling_strategy: MupScalingStrategy,
        input_dim: int,
        output_dim: int,
        scaling_type: Optional[ParameterScalingType] = None,
    ):
        super().__init__(optimizer=optimizer, input_dim=input_dim, output_dim=output_dim)
        self.scaling_strategy = scaling_strategy
        self.scaling_type_override = scaling_type

        self.set_base_dims(base_input_dim=None, base_output_dim=None)

    def set_base_dims(
        self,
        base_input_dim: Optional[int] = None,
        base_output_dim: Optional[int] = None,
    ) -> None:
        base_input_dim = base_input_dim if base_input_dim is not None else 1
        base_output_dim = base_output_dim if base_output_dim is not None else 1
        self.input_dim_ratio = self.input_dim / base_input_dim
        self.output_dim_ratio = self.output_dim / base_output_dim
        self.scaling_type = self.scaling_type_override or self._get_scaling_type()

        input_multiplier, init_std_multiplier, lr_multiplier = self._get_parametrization_scalars()
        self._input_multiplier = (
            input_multiplier if not math.isclose(input_multiplier, 1.0) else None
        )
        self._init_std_multiplier = (
            init_std_multiplier if not math.isclose(init_std_multiplier, 1.0) else None
        )
        self._lr_multiplier = lr_multiplier if not math.isclose(lr_multiplier, 1.0) else None

        # We want softmax scaling to be such that `softmax_scale * multiplier = sqrt(base_input_dim) / input_dim`.
        # The normal softmax scale is `1 / sqrt(input_dim)`, so we have `multiplier = sqrt(base_input_dim / input_dim)`.
        self._softmax_scale_multiplier = math.sqrt(base_input_dim / self.input_dim)

    def _get_scaling_type(self) -> ParameterScalingType:
        scaling_input = not math.isclose(self.input_dim_ratio, 1)
        scaling_output = not math.isclose(self.output_dim_ratio, 1)

        if scaling_input and scaling_output:
            return ParameterScalingType.scaling_input_output
        elif scaling_input:
            return ParameterScalingType.scaling_input
        elif scaling_output:
            return ParameterScalingType.scaling_output
        else:
            return ParameterScalingType.constant

    def _get_parametrization_scalars(self) -> Tuple[float, float, float]:
        """
        Returns the trio of (input multiplier, init std multiplier, optimizer LR multiplier)
        """

        # These scalars come from Table 3 of the Maximal Update Parametrization paper.
        base_scalars_map: Dict[
            ParametrizationOptimizerType,
            Dict[ParameterScalingType, Tuple[float, float, float]],
        ] = {
            ParametrizationOptimizerType.adam: {
                ParameterScalingType.scaling_input: (
                    1.0,
                    1.0 / self.input_dim_ratio,
                    1.0 / self.input_dim_ratio,
                ),
                ParameterScalingType.scaling_input_output: (
                    1.0,
                    1.0 / math.sqrt(self.input_dim_ratio),
                    1.0 / self.input_dim_ratio,
                ),
            },
            ParametrizationOptimizerType.adam_coupled_wd: {
                ParameterScalingType.scaling_input: (
                    1.0,
                    1.0 / self.input_dim_ratio,
                    1.0 / self.input_dim_ratio,
                ),
                ParameterScalingType.scaling_input_output: (
                    1.0,
                    1.0 / math.sqrt(self.input_dim_ratio),
                    1.0 / self.input_dim_ratio,
                ),
            },
        }

        input_multiplier, init_std_multiplier, lr_multiplier = base_scalars_map.get(
            self.optimizer, {}
        ).get(self.scaling_type, (1.0, 1.0, 1.0))

        # Lemma J.1 of the Maximal Update Parametrization paper allows for scaling the input multiplier up by a factor K by
        # scaling down the init std and LR a corresponding amount. The LR is scaled down by
        # K for Adam variants since Adam normalizes gradients, and K^2 for SGD variants since SGD
        # updates are proportional to gradient.
        scale_lr_twice = self.optimizer not in (
            ParametrizationOptimizerType.adam,
            ParametrizationOptimizerType.adam_coupled_wd,
        )
        if self.scaling_strategy == MupScalingStrategy.constant_inputs:
            input_rescaling = 1 / input_multiplier
            init_std_rescaling = input_multiplier
            lr_rescaling = input_multiplier**2 if scale_lr_twice else input_multiplier
        elif self.scaling_strategy == MupScalingStrategy.constant_init_std:
            input_rescaling = init_std_multiplier
            init_std_rescaling = 1 / init_std_multiplier
            lr_rescaling = (
                (1 / init_std_multiplier) ** 2 if scale_lr_twice else 1 / init_std_multiplier
            )
        elif self.scaling_strategy == MupScalingStrategy.constant_lr:
            input_rescaling = lr_multiplier
            init_std_rescaling = 1 / lr_multiplier
            lr_rescaling = 1 / lr_multiplier
            if scale_lr_twice:
                input_rescaling = math.sqrt(input_rescaling)
                init_std_rescaling = math.sqrt(init_std_rescaling)
        else:
            raise NotImplementedError(self.scaling_strategy)

        input_multiplier *= input_rescaling
        init_std_multiplier *= init_std_rescaling
        lr_multiplier *= lr_rescaling

        return (input_multiplier, init_std_multiplier, lr_multiplier)

    @property
    def input_multiplier(self) -> Optional[float]:
        return self._input_multiplier

    @property
    def init_std_multiplier(self) -> Optional[float]:
        return self._init_std_multiplier

    @property
    def lr_multiplier(self) -> Optional[float]:
        return self._lr_multiplier

    @property
    def softmax_scale_multiplier(self) -> Optional[float]:
        return self._softmax_scale_multiplier
