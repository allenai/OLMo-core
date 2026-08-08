from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn

from olmo_core.config import Config, DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import (
    BFLOAT16_MIN_COMPUTE_CAPABILITY,
    get_devices_without_bfloat16,
)

if TYPE_CHECKING:
    from .train_module import TrainModule

__all__ = ["TrainModuleConfig", "validate_precision_support"]


PRECISION_FIELD_MARKERS = ("dtype", "precision")
"""
What makes a config field a precision setting: ``param_dtype``, ``reduce_dtype``,
``autocast_precision``, ``dtype``. Required in addition to the value, because there is no way
past this refusal -- bfloat16 on hardware without bfloat16 does not run, waiver or not -- so a
field whose value merely happens to be the string "bfloat16" and whose name says nothing about
precision must not stop a run that would have worked.
"""


def _bfloat16_fields(config: Config) -> List[str]:
    """
    The dotted paths in ``config`` that ask for bfloat16, or an empty list.

    Read off the serialized config rather than off whatever the caller was given, so that a
    default nobody typed, a keyword argument and a dotted override are one fact by the time we
    look. ``as_config_dict`` is also the serialization the config saver writes beside the
    checkpoint, so a path named here is a path a reader can find in that file.
    """
    found: List[str] = []
    bfloat16 = DType.bfloat16.value

    def walk(node, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, (list, tuple)):
            for position, value in enumerate(node):
                walk(value, f"{path}[{position}]")
        elif isinstance(node, str) and node == bfloat16:
            name = path.rsplit(".", 1)[-1]
            if any(marker in name for marker in PRECISION_FIELD_MARKERS):
                found.append(path)

    walk(config.as_config_dict(), "")
    return sorted(found)


def validate_precision_support(config: Config, model: Optional[nn.Module] = None) -> None:
    """
    Raise if ``config`` (or ``model``, when given) asks for bfloat16 on a device whose silicon
    has no bfloat16 arithmetic. A no-op otherwise, including on CPU and ROCm.

    Called from :meth:`TrainModuleConfig.build` implementations so that the refusal lands before
    the model is placed on the device, before the data loader is built and long before the first
    kernel that would want the format. Everything up to that kernel succeeds on a Turing card --
    the model builds, the process group forms, the run starts -- so without this a job dies
    minutes in, on hardware somebody is already paying for.

    :param config: The config to inspect. Any :class:`~olmo_core.config.Config` will do; a train
        module config is the usual one, since that is where the precision fields live.
    :param model: The built model, if there is one. Catches a model whose parameters are already
        bfloat16, which no field of a train module config would show.

    :raises OLMoConfigurationError: If the hardware cannot do the precision that was asked for.
    """
    unusable = get_devices_without_bfloat16()
    if not unusable:
        return

    asked_at = _bfloat16_fields(config)
    if model is not None and any(p.dtype == torch.bfloat16 for p in model.parameters()):
        asked_at.append("the model's parameters")
    if not asked_at:
        return

    _, name, (major, minor) = unusable[0]
    if len(unusable) == torch.cuda.device_count() and len({n for _, n, _ in unusable}) == 1:
        devices = f"all {len(unusable)} of this host's devices are {name}"
    else:
        devices = f"device {', '.join(str(index) for index, _, _ in unusable)} is a {name}"

    raise OLMoConfigurationError(
        f"{devices}, compute capability {major}.{minor}. bfloat16 arithmetic starts at compute "
        f"capability {BFLOAT16_MIN_COMPUTE_CAPABILITY[0]}.{BFLOAT16_MIN_COMPUTE_CAPABILITY[1]} "
        f"(Ampere), so there is none in this hardware, and this run asks for bfloat16 at "
        f"{', '.join(asked_at)}.\n\n"
        "Note that torch.cuda.is_bf16_supported() returns True on this device, which is why "
        "nothing before this caught it: when its capability test fails it falls back to "
        "allocating a bfloat16 tensor, and that succeeds here because PyTorch implements the "
        "storage and the conversions in software on every architecture. It is the arithmetic "
        "that is missing, so the capability above is read straight off the device.\n\n"
        "Either move to a device of compute capability "
        f"{BFLOAT16_MIN_COMPUTE_CAPABILITY[0]}.{BFLOAT16_MIN_COMPUTE_CAPABILITY[1]} or newer, or "
        "ask for a precision this one has. float32 is what a pre-Ampere card is for, and "
        "float16 is the half precision it does have -- though OLMo-core ships no gradient "
        "scaler, so fp16 here is fp16 without loss scaling and small gradients underflow to zero."
    )


@dataclass
class TrainModuleConfig(Config):
    @abstractmethod
    def build(self, *args, **kargs) -> "TrainModule":
        raise NotImplementedError
