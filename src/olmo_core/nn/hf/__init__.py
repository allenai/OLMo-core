"""
Utilities for converting models between OLMo Core and Hugging Face formats. To configure the
mappings between OLMo Core and Hugging Face, you may change the variables in
:mod:`olmo_core.nn.hf.convert` (e.g. :data:`olmo_core.nn.hf.convert.HF_TO_OLMO_CORE_WEIGHT_MAPPINGS`).
"""

from .checkpoint import load_hf_model, save_hf_hybrid_model, save_hf_model
from .config import (
    get_hf_config,
    get_hybrid_hf_config,
    get_hybrid_layer_types,
    is_olmo_hybrid_model,
)
from .convert import (
    convert_hybrid_state_to_hf,
    convert_state_from_hf,
    convert_state_to_hf,
    get_converter_from_hf,
    get_converter_to_hf,
)
from .convert_checkpoint import (
    convert_checkpoint_to_hf,
    load_config,
    validate_conversion,
)

__all__ = [
    "convert_checkpoint_to_hf",
    "convert_hybrid_state_to_hf",
    "convert_state_from_hf",
    "convert_state_to_hf",
    "get_converter_from_hf",
    "get_converter_to_hf",
    "get_hf_config",
    "get_hybrid_hf_config",
    "get_hybrid_layer_types",
    "is_olmo_hybrid_model",
    "load_config",
    "load_hf_model",
    "save_hf_hybrid_model",
    "save_hf_model",
    "validate_conversion",
]
