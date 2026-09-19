"""HuggingFace transformers implementation of MainlineLadder.

Two registration entry points, with different weights of import:

- ``register_config()`` — registers only the HF *config* with ``AutoConfig``.
  This is all the vLLM path needs (vLLM parses ``config.json`` via
  ``AutoConfig.from_pretrained`` and uses its own model executor).
  ``register()`` — full registration (config + ``AutoModel`` + ``AutoModelForCausalLM``)
  for the HuggingFace provider path and the weight converter. Imports the torch
  modeling module.
"""

from typing import TYPE_CHECKING


MODEL_TYPE = "mainline_ladder"

_CONFIG_REGISTERED = False
_FULL_REGISTERED = False


def _safe_register(register_fn, *args):
    """Call a transformers ``Auto*.register`` tolerating older signatures.

    Newer transformers supports ``exist_ok=True`` (idempotent re-registration);
    older versions don't. Try with it first, fall back without, and treat an
    "already registered" ValueError as success.
    """

    def _is_already_registered_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "already" in msg and "register" in msg

    try:
        return register_fn(*args, exist_ok=True)
    except TypeError:
        # Older transformers: no exist_ok parameter.
        pass
    except ValueError as exc:
        if _is_already_registered_error(exc):
            return None
        raise

    try:
        return register_fn(*args)
    except ValueError as exc:
        if _is_already_registered_error(exc):
            return None
        raise


def register_config() -> None:
    """Register the HF config with ``AutoConfig`` (lightweight; all vLLM needs)."""
    global _CONFIG_REGISTERED
    if _CONFIG_REGISTERED:
        return
    from transformers import AutoConfig

    from .configuration_mainline_ladder import MainlineLadderConfig

    _safe_register(AutoConfig.register, MODEL_TYPE, MainlineLadderConfig)
    _CONFIG_REGISTERED = True


def register() -> None:
    """Full HF registration: config + ``AutoModel`` + ``AutoModelForCausalLM``.

    Use this for the transformers (``hf``) provider path or the weight converter.
    Idempotent.
    """
    global _FULL_REGISTERED
    if _FULL_REGISTERED:
        return
    register_config()
    from transformers import AutoModel, AutoModelForCausalLM

    from .configuration_mainline_ladder import MainlineLadderConfig
    from .modeling_mainline_ladder import (
        MainlineLadderForCausalLM,
        MainlineLadderModel,
    )

    _safe_register(AutoModel.register, MainlineLadderConfig, MainlineLadderModel)
    _safe_register(
        AutoModelForCausalLM.register, MainlineLadderConfig, MainlineLadderForCausalLM
    )
    _FULL_REGISTERED = True


__all__ = [
    "MODEL_TYPE",
    "MainlineLadderConfig",
    "MainlineLadderForCausalLM",
    "MainlineLadderModel",
    "MainlineLadderPreTrainedModel",
    "register",
    "register_config",
]

if TYPE_CHECKING:
    from .configuration_mainline_ladder import MainlineLadderConfig
    from .modeling_mainline_ladder import (
        MainlineLadderForCausalLM,
        MainlineLadderModel,
        MainlineLadderPreTrainedModel,
    )


def __getattr__(name: str):
    # Lazy class exports: avoid importing torch modeling unless actually accessed.
    if name == "MainlineLadderConfig":
        from .configuration_mainline_ladder import MainlineLadderConfig

        return MainlineLadderConfig
    if name in (
        "MainlineLadderForCausalLM",
        "MainlineLadderModel",
        "MainlineLadderPreTrainedModel",
    ):
        from . import modeling_mainline_ladder as _m

        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
