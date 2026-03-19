"""
Custom Gemma3 model with gated attention support for OLMo-core's Gemma-like models.

This module provides a custom HuggingFace model that extends Gemma3 with an attention
gate projection (g_proj), matching the GateConfig in OLMo-core's attention module.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Check if we have the required transformers version
HAS_GEMMA3 = False
try:
    from transformers import Gemma3TextConfig
    from transformers.models.gemma3.modeling_gemma3 import (
        Gemma3ForCausalLM,
        Gemma3TextModel,
    )

    HAS_GEMMA3 = True
except ImportError:
    Gemma3TextConfig = object
    Gemma3ForCausalLM = nn.Module
    Gemma3TextModel = nn.Module


class Gemma3GatedTextConfig(Gemma3TextConfig):
    """
    Configuration for Gemma3 with gated attention.

    Adds `attention_gate_dim` to control the gate projection output dimension.
    If None, defaults to num_attention_heads * head_dim (elementwise gating).
    """

    model_type = "gemma3_gated_text"

    def __init__(
        self,
        attention_gate_dim: Optional[int] = None,
        **kwargs,
    ):
        if not HAS_GEMMA3:
            raise RuntimeError(
                "Gemma3GatedTextConfig requires transformers >= 4.50.0 with Gemma3 support"
            )
        super().__init__(**kwargs)
        # Default to elementwise gating (n_heads * head_dim)
        if attention_gate_dim is None:
            attention_gate_dim = self.num_attention_heads * self.head_dim
        self.attention_gate_dim = attention_gate_dim


class Gemma3GatedForCausalLM(Gemma3ForCausalLM):
    """
    Gemma3 with gated attention for causal language modeling.

    This extends the standard Gemma3ForCausalLM by adding a gate projection (g_proj)
    to each attention layer. The gate is applied element-wise to the attention output.
    """

    config_class = Gemma3GatedTextConfig

    def __init__(self, config: Gemma3GatedTextConfig):
        if not HAS_GEMMA3:
            raise RuntimeError(
                "Gemma3GatedForCausalLM requires transformers >= 4.50.0 with Gemma3 support"
            )
        super().__init__(config)

        # Add gate projection to each attention layer
        for layer in self.model.layers:
            layer.self_attn.g_proj = nn.Linear(
                config.hidden_size,
                config.attention_gate_dim,
                bias=config.attention_bias,
            )

    def _init_weights(self, module):
        """Initialize weights including the gate projection."""
        super()._init_weights(module)
        # Gate projections are initialized by the parent class's _init_weights
        # since they are nn.Linear modules


class Gemma3GatedTextModel(Gemma3TextModel):
    """
    Gemma3 text model with gated attention (without LM head).
    """

    config_class = Gemma3GatedTextConfig

    def __init__(self, config: Gemma3GatedTextConfig):
        if not HAS_GEMMA3:
            raise RuntimeError(
                "Gemma3GatedTextModel requires transformers >= 4.50.0 with Gemma3 support"
            )
        super().__init__(config)

        # Add gate projection to each attention layer
        for layer in self.layers:
            layer.self_attn.g_proj = nn.Linear(
                config.hidden_size,
                config.attention_gate_dim,
                bias=config.attention_bias,
            )


def register_gemma3_gated():
    """Register Gemma3Gated models with transformers Auto classes."""
    if not HAS_GEMMA3:
        log.warning("Cannot register Gemma3Gated: transformers >= 4.50.0 required")
        return False

    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        AutoConfig.register("gemma3_gated_text", Gemma3GatedTextConfig)
        AutoModelForCausalLM.register(Gemma3GatedTextConfig, Gemma3GatedForCausalLM)
        log.info("Registered Gemma3Gated models with transformers Auto classes")
        return True
    except Exception as e:
        log.warning(f"Failed to register Gemma3Gated models: {e}")
        return False


# Auto-register when module is imported
_registered = register_gemma3_gated()
