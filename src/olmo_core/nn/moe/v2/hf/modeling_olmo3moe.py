
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, cast
from transformers.utils.generic import TransformersKwargs

from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.utils.deprecation import deprecate_kwarg
from transformers.activations import ACT2FN
from transformers.utils.auto_docstring import auto_docstring
from transformers.utils.generic import can_return_tuple
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from transformers.models.olmo3.modeling_olmo3 import (
    eager_attention_forward,
)
from .configuration_olmo3moe import Olmo3MoeConfig


class Olmo3MoeRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Olmo3MoeConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: Olmo3MoeConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor
    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
            return cos, sin


class Olmo3MoeDenseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.dense_mlp_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class Olmo3MoeExpert(nn.Module):
    def __init__(self, hidden_size, moe_intermediate_size, hidden_act):
        super().__init__()
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.moe_intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj



class Olmo3MoeExperts(nn.ModuleList):
    """Container for routed experts.

    vLLM detects a child module named ``experts`` that is a ``ModuleList`` and
    replaces it with a fused implementation at load time (TransformersFusedMoE).
    This class provides an eager reference implementation, and a compile-safe
    fallback (very slow) to avoid TorchDynamo graph breaks if it is ever traced.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,  # (N, H)
        topk_ids: torch.Tensor,  # (N, K)
        topk_weights: torch.Tensor,  # (N, K)
    ) -> torch.Tensor:
        # Use a compile-safe fallback if TorchDynamo is tracing this module.
        # NOTE: This is extremely slow because it runs every expert on every token.
        try:
            is_compiling = torch._dynamo.is_compiling()
        except Exception:
            is_compiling = False

        if is_compiling:
            out = hidden_states.new_zeros(hidden_states.shape)
            for expert_id, expert in enumerate(self):
                # Aggregate the routing weights for this expert across the K slots.
                w = (topk_weights * (topk_ids == expert_id).to(topk_weights.dtype)).sum(
                    dim=1, keepdim=True
                )  # (N, 1)
                out = out + expert(hidden_states) * w
            return out

        # Eager reference routing (faster), but uses data-dependent shapes.
        N, H = hidden_states.shape
        out = hidden_states.new_zeros((N, H))
        for expert_id, expert in enumerate(self):
            mask = (topk_ids == expert_id)  # (N, K) bool
            if not mask.any():
                continue
            token_ids, k_ids = mask.nonzero(as_tuple=True)  # both (M,)
            x_sel = hidden_states.index_select(0, token_ids)  # (M, H)
            y_sel = expert(x_sel)  # (M, H)
            w_sel = topk_weights[token_ids, k_ids].unsqueeze(-1).to(dtype=hidden_states.dtype)  # (M, 1)
            out.index_add_(0, token_ids, y_sel * w_sel)
        return out


class Olmo3MoeSparseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.router = Olmo3MoeRouter(config)
        self.experts = Olmo3MoeExperts()
        for _ in range(config.n_routed_experts):
            expert = Olmo3MoeExpert(
                hidden_size=config.hidden_size,
                moe_intermediate_size=config.moe_intermediate_size,
                hidden_act=config.hidden_act,
            )
            self.experts.append(expert)
        if config.shared_expert_intermediate_size is not None:
            self.shared_expert = Olmo3MoeExpert(
                hidden_size=config.hidden_size,
                moe_intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
            )
        else:
            self.shared_expert = None

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)
        B, S, H = x.shape

        # Compute gating weights and expert indices
        # expert_weights: (batch_size, seq_len, top_k)
        # expert_indices: (batch_size, seq_len, top_k)
        expert_weights, expert_indices = self.router(x)
        K = expert_indices.size(-1)
        # Flatten tokens: N = B*S. vLLM's fused experts expects (N, H), (N, K), (N, K).
        x_flat = x.reshape(B * S, H)  # (N, H)
        idx_flat = expert_indices.reshape(B * S, K)  # (N, K)
        w_flat = expert_weights.reshape(B * S, K).to(dtype=x.dtype)  # (N, K)

        out_flat = self.experts(x_flat, topk_ids=idx_flat, topk_weights=w_flat)  # (N, H)
        routed_expert_out = out_flat.view(B, S, H)

        # shared expert
        if self.shared_expert is None:
            out = routed_expert_out
        else:
            shared_expert_out = self.shared_expert(x)
            out = routed_expert_out + shared_expert_out

        return out


class Olmo3MoeRouter(nn.Module):
    def __init__(self, config: Olmo3MoeConfig):
        super().__init__()
        self.config = config
        self.gating_function = config.gating_function
        self.hidden_size = config.hidden_size
        self.num_experts_per_tok = config.num_experts_per_tok
        self.gate = nn.Linear(self.hidden_size, config.n_routed_experts, bias=False)
        self.normalize_expert_weights = config.normalize_expert_weights
        self.restore_weight_scale = config.restore_weight_scale

    def forward(self, x):
        logits = self.gate(x)

        if self.gating_function == 'softmax':
            scores = logits.softmax(dim=-1)
        elif self.gating_function == 'sigmoid':
            scores = torch.sigmoid(logits)
            # to avoid NaNs in the load balancing loss
            # if all logits of a token are very negative for all experts, sigmoid gives 0 for all experts, causing NaNs when we div by the sum.
            scores = scores + 1e-7  
        else:
            raise NotImplementedError(self.gating_function)

        expert_weights, expert_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)

        if self.normalize_expert_weights is not None:
            expert_weights = expert_weights.div(
                torch.norm(
                    expert_weights,
                    p=self.normalize_expert_weights,
                    dim=-1,
                    keepdim=True,
                )
            )

        if self.restore_weight_scale:
            expert_weights = expert_weights * self.num_experts_per_tok

        return expert_weights, expert_indices

class Olmo3MoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Olmo3MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Olmo3MoeAttention(config=config, layer_idx=layer_idx)

        if layer_idx in config.dense_layers_indices:
            self.mlp = Olmo3MoeDenseMLP(config)
        else:
            self.mlp = Olmo3MoeSparseMLP(config)

        self.post_attention_layernorm = Olmo3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Olmo3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states




def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class Olmo3MoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Olmo3MoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_head_qk_norm = config.use_head_qk_norm

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        if config.use_head_qk_norm:
            self.q_norm = Olmo3MoeRMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = Olmo3MoeRMSNorm(self.head_dim, config.rms_norm_eps)
        else:
            self.q_norm = Olmo3MoeRMSNorm(config.num_attention_heads * self.head_dim, config.rms_norm_eps)
            self.k_norm = Olmo3MoeRMSNorm(config.num_key_value_heads * self.head_dim, config.rms_norm_eps)
        assert config.layer_types is not None
        self.attention_type = config.layer_types[layer_idx]
        self.sliding_window = config.sliding_window if self.attention_type == "sliding_attention" else None

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

         # QK norm behavior:
         # - use_head_qk_norm=False: normalize over the flattened projection dim (matches File 1 "full-dim" norm path)
         # - use_head_qk_norm=True: reshape into heads first, then normalize per head over head_dim (matches File 1 head-wise path)
        if not self.use_head_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)   # (B, n_heads, T, head_dim)
        key_states   = key_states.view(hidden_shape).transpose(1, 2)     # (B, n_kv_heads, T, head_dim)
        value_states = value_states.view(hidden_shape).transpose(1, 2)   # (B, n_kv_heads, T, head_dim)

        if self.use_head_qk_norm:
            query_states = self.q_norm(query_states.contiguous())
            key_states = self.k_norm(key_states.contiguous())

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights



@auto_docstring
class Olmo3MoePreTrainedModel(PreTrainedModel):
    config: Olmo3MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Olmo3MoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Olmo3MoeDecoderLayer,
        "attentions": Olmo3MoeAttention,
    }
    
class Olmo3MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
 
@auto_docstring
class Olmo3MoeModel(Olmo3MoePreTrainedModel):
    def __init__(self, config: Olmo3MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Olmo3MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Olmo3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.rotary_embs = Olmo3MoeRotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            assert inputs_embeds is not None
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_embs(hidden_states, position_ids)

        for decoder_layer in self.layers:
            # if used in vllm with PP, a few layers will be replaced by PPMissingLayer(), which just passes the inputs through, so we need to skip the attention mask and position embeddings in that case
            if not isinstance(decoder_layer, Olmo3MoeDecoderLayer):
                hidden_states = decoder_layer(hidden_states)
                continue

            decoder_layer = cast(Olmo3MoeDecoderLayer, decoder_layer)
            attention_mask = causal_mask_mapping[decoder_layer.self_attn.attention_type]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

@auto_docstring
class Olmo3MoeForCausalLM(Olmo3MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Olmo3MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



__all__ = [
    "Olmo3MoeConfig",
    "Olmo3MoeForCausalLM",
    "Olmo3MoeModel",
    "Olmo3MoePreTrainedModel",
]