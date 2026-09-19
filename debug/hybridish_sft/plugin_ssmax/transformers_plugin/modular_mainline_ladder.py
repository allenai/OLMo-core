"""
MainlineLadder: A hybrid model with GatedDeltaNet + gated NoPE attention,
peri-norm (pre+post norm), embedding scaling/norm.

Inherits from OlmoHybrid and overrides:
- Config: embed_scale, embedding_norm, NoPE, gated attention, peri_norm, per-head QK norm
- Attention: elementwise gate, per-head QK norm, always NoPE
- DecoderLayers: peri_norm (pre+post norm)
- Model: embedding scaling + norm, no rotary embeddings
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..olmo_hybrid.configuration_olmo_hybrid import OlmoHybridConfig
from ..olmo_hybrid.modeling_olmo_hybrid import (
    OlmoHybridAttention,
    OlmoHybridAttentionDecoderLayer,
    OlmoHybridDynamicCache,
    OlmoHybridForCausalLM,
    OlmoHybridGatedDeltaNet,
    OlmoHybridLinearAttentionDecoderLayer,
    OlmoHybridMLP,
    OlmoHybridModel,
    OlmoHybridPreTrainedModel,
    OlmoHybridRMSNorm,
    OlmoHybridRMSNormGated,
    eager_attention_forward,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="allenai/MainlineLadder")
@strict
class MainlineLadderConfig(OlmoHybridConfig):
    r"""
    embed_scale (`float`, *optional*):
        Scaling factor applied to embeddings. Defaults to `sqrt(hidden_size)` if not set.
    embedding_norm_eps (`float`, *optional*, defaults to 1e-6):
        Epsilon for the RMSNorm applied after embedding scaling.
    use_attention_gate (`bool`, *optional*, defaults to `True`):
        Whether to apply an elementwise sigmoid gate on the full attention output (before o_proj).
    use_head_qk_norm (`bool`, *optional*, defaults to `True`):
        Whether to use per-head QK normalization (norm of size head_dim per head) instead of
        full-dim norm (norm of size num_heads * head_dim).
    linear_num_key_heads (`int`, *optional*):
        Number of key heads for the linear attention layers. Defaults to `num_attention_heads`.
    linear_num_value_heads (`int`, *optional*):
        Number of value heads for the linear attention layers. Defaults to `num_attention_heads`.
    linear_key_head_dim (`int`, *optional*):
        Dimension of each key head in linear attention layers. Defaults to `head_dim`.
    linear_value_head_dim (`int`, *optional*):
        Dimension of each value head in linear attention layers. Defaults to `2 * linear_key_head_dim`.
    linear_a_log_min (`float`, *optional*, defaults to 0.0):
        Minimum value for uniform initialization of A_log in GatedDeltaNet layers.
    linear_a_log_max (`float`, *optional*, defaults to 16.0):
        Maximum value for uniform initialization of A_log in GatedDeltaNet layers.
    linear_dt_min (`float`, *optional*, defaults to 0.001):
        Minimum value for dt initialization in GatedDeltaNet layers.
    linear_dt_max (`float`, *optional*, defaults to 0.1):
        Maximum value for dt initialization in GatedDeltaNet layers.
    linear_dt_init_floor (`float`, *optional*, defaults to 0.0001):
        Floor value for clamping dt during initialization in GatedDeltaNet layers.
    linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
        Kernel size for the short convolution applied to queries, keys, and values in linear attention layers.
    linear_allow_neg_eigval (`bool`, *optional*, defaults to `True`):
        Whether to allow negative eigenvalues in the GatedDeltaNet recurrence.

    Example:

    ```python
    >>> from transformers import MainlineLadderModel, MainlineLadderConfig

    >>> # Initializing an MainlineLadder style configuration
    >>> configuration = MainlineLadderConfig()

    >>> # Initializing a model from the MainlineLadder style configuration
    >>> model = MainlineLadderModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "mainline_ladder"

    # New attributes
    embed_scale: float | None = None
    embedding_norm_eps: float = 1e-6
    use_attention_gate: bool = True
    use_head_qk_norm: bool = True

    # Override defaults from parent
    hidden_size: int = 640
    intermediate_size: int = 5120
    num_hidden_layers: int = 10
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 100352
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 8192

    # NoPE: no positional embeddings used anywhere in this model.
    # rope_theta and rope_scaling are inherited but unused; we override the
    # model to never construct rotary embeddings regardless of config values.

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            # Default: 4 GDN layers + 1 full attention, repeating (4:1 ratio)
            self.layer_types = ["linear_attention"] * int(self.num_hidden_layers)
            for i in range(int(self.num_hidden_layers)):
                if i % 5 == 4:
                    self.layer_types[i] = "full_attention"
            # Ensure at least one full attention layer
            if "full_attention" not in self.layer_types:
                self.layer_types[-1] = "full_attention"

        if self.embed_scale is None:
            self.embed_scale = math.sqrt(self.hidden_size)

        if self.linear_num_key_heads is None:
            self.linear_num_key_heads = self.num_attention_heads
        if self.linear_num_value_heads is None:
            self.linear_num_value_heads = self.num_attention_heads
        if self.linear_key_head_dim is None:
            self.linear_key_head_dim = self.head_dim
        if self.linear_value_head_dim is None:
            self.linear_value_head_dim = 2 * self.linear_key_head_dim

        from ...configuration_utils import PreTrainedConfig

        PreTrainedConfig.__post_init__(**kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        # NoPE model: skip RoPE parameter standardization entirely to prevent
        # Transformers from filling in default rope_theta/rope_type values.
        return kwargs

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if "linear_attention" not in self.layer_types:
            raise ValueError(
                "MainlineLadder expects at least one 'linear_attention' layer."
            )
        if all(t == "linear_attention" for t in self.layer_types):
            raise ValueError("MainlineLadder expects at least one attention layer.")


class MainlineLadderRMSNorm(OlmoHybridRMSNorm):
    pass


class MainlineLadderRMSNormGated(OlmoHybridRMSNormGated):
    pass


class MainlineLadderMLP(OlmoHybridMLP):
    pass


class MainlineLadderDynamicCache(OlmoHybridDynamicCache):
    pass


class MainlineLadderGatedDeltaNet(OlmoHybridGatedDeltaNet):
    pass


class MainlineLadderAttention(OlmoHybridAttention):
    """
    Full attention for MainlineLadder with:
    - Per-head QK normalization (RMSNorm of size head_dim applied after reshape)
    - Elementwise sigmoid gate before o_proj
    - Always NoPE (no rotary embeddings)
    """

    def __init__(self, config: MainlineLadderConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.head_dim = (
            config.head_dim or config.hidden_size // config.num_attention_heads
        )

        # Per-head QK norm (size = head_dim, applied per head)
        if config.use_head_qk_norm:
            self.q_norm = MainlineLadderRMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = MainlineLadderRMSNorm(self.head_dim, config.rms_norm_eps)

        # Elementwise attention gate
        if config.use_attention_gate:
            self.attn_gate = nn.Linear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim,
                bias=False,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        # Per-head QK norm (applied after reshape to head dimensions)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # NoPE: never apply rotary embeddings (position_embeddings is always None)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # Elementwise gate: sigmoid(linear(x)) * attn_output
        if hasattr(self, "attn_gate"):
            gate = self.attn_gate(hidden_states).float()
            gate = torch.sigmoid(gate).to(attn_output.dtype)
            attn_output = attn_output.reshape(*input_shape, -1) * gate
        else:
            attn_output = attn_output.reshape(*input_shape, -1)

        attn_output = attn_output.contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MainlineLadderAttentionDecoderLayer(OlmoHybridAttentionDecoderLayer):
    """
    Full attention decoder layer with peri-norm (pre+post norm on both sub-blocks).

    OlmoHybrid uses reordered_norm (post-norm only):
        residual + post_norm(attn(x))

    MainlineLadder uses peri_norm (pre+post):
        residual + post_norm(attn(pre_norm(x)))
    """

    def __init__(self, config: MainlineLadderConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MainlineLadderAttention(config=config, layer_idx=layer_idx)
        # peri_norm: pre-norm (input_layernorm) + post-norm (post_attention_layernorm)
        self.input_layernorm = MainlineLadderRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MainlineLadderRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # peri_norm for FFN: pre-norm (ffn_layernorm) + post-norm (post_feedforward_layernorm)
        self.ffn_layernorm = MainlineLadderRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = MainlineLadderRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = MainlineLadderMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # Peri-norm attention block: residual + post_norm(attn(pre_norm(x)))
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=None,  # Always NoPE
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Peri-norm FFN block: residual + post_norm(mlp(pre_norm(x)))
        residual = hidden_states
        hidden_states = self.ffn_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MainlineLadderLinearAttentionDecoderLayer(OlmoHybridLinearAttentionDecoderLayer):
    """
    Linear attention (GatedDeltaNet) decoder layer with peri-norm.

    OlmoHybrid uses pre-norm only:
        residual + attn(pre_norm(x))

    MainlineLadder uses peri_norm (pre+post):
        residual + post_norm(attn(pre_norm(x)))
    """

    def __init__(self, config: MainlineLadderConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.linear_attn = MainlineLadderGatedDeltaNet(config, layer_idx=layer_idx)
        self.input_layernorm = MainlineLadderRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MainlineLadderRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.ffn_layernorm = MainlineLadderRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = MainlineLadderRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = MainlineLadderMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # Peri-norm linear attention block: residual + post_norm(linear_attn(pre_norm(x)))
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Peri-norm FFN block: residual + post_norm(mlp(pre_norm(x)))
        residual = hidden_states
        hidden_states = self.ffn_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MainlineLadderPreTrainedModel(OlmoHybridPreTrainedModel):
    _no_split_modules = [
        "MainlineLadderAttentionDecoderLayer",
        "MainlineLadderLinearAttentionDecoderLayer",
    ]
    _can_record_outputs = {
        "hidden_states": (
            MainlineLadderAttentionDecoderLayer,
            MainlineLadderLinearAttentionDecoderLayer,
        ),
        "attentions": MainlineLadderAttention,
    }

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        # This model uses MainlineLadderDynamicCache, which stores split q/k/v conv states, so it
        # cannot consume the plain DynamicCache that GenerationMixin builds by default. Returning
        # False makes generate() skip cache construction and lets our forward build the right cache.
        # Upstream excludes OlmoHybrid the same way, but by a hardcoded class-name match that our
        # subclass name does not trigger (transformers GenerationMixin._supports_default_dynamic_cache).
        return False

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, MainlineLadderGatedDeltaNet):
            from ... import initialization as init

            cfg = self.config
            init.copy_(
                module.A_log,
                torch.empty_like(module.A_log)
                .uniform_(cfg.linear_a_log_min, cfg.linear_a_log_max)
                .log_(),
            )
            dt = torch.exp(
                torch.rand_like(module.dt_bias)
                * (math.log(cfg.linear_dt_max) - math.log(cfg.linear_dt_min))
                + math.log(cfg.linear_dt_min)
            )
            dt = torch.clamp(dt, min=cfg.linear_dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            init.copy_(module.dt_bias, inv_dt)


class MainlineLadderModel(OlmoHybridModel):
    def __init__(self, config: MainlineLadderConfig):
        super().__init__(config)
        # Replace layers with peri-norm variants
        self.layers = nn.ModuleList(
            [
                MainlineLadderLinearAttentionDecoderLayer(config, layer_idx)
                if config.layer_types[layer_idx] == "linear_attention"
                else MainlineLadderAttentionDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        # Embedding norm (applied after embed_scale)
        self.embed_norm = MainlineLadderRMSNorm(
            config.hidden_size, eps=config.embedding_norm_eps
        )
        # Final norm before LM head
        self.norm = MainlineLadderRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # No rotary embeddings (NoPE everywhere)
        self.rotary_emb = None
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = MainlineLadderDynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + past_seen_tokens
            )
            position_ids = position_ids.unsqueeze(0)

        # Embedding scaling + norm
        hidden_states = inputs_embeds * self.config.embed_scale
        hidden_states = self.embed_norm(hidden_states)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(
            attention_mask, past_key_values
        )

        for i, decoder_layer in enumerate(self.layers):
            layer_mask = (
                linear_attn_mask
                if self.config.layer_types[i] == "linear_attention"
                else causal_mask
            )

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=None,  # Always NoPE
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class MainlineLadderForCausalLM(OlmoHybridForCausalLM):
    pass


__all__ = [
    "MainlineLadderConfig",
    "MainlineLadderForCausalLM",
    "MainlineLadderModel",
    "MainlineLadderPreTrainedModel",
]
