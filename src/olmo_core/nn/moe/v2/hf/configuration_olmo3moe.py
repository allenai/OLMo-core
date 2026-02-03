
from typing import List, Optional
from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation


class Olmo3MoeConfig(PretrainedConfig):
    model_type = "olmo3moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.k_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.v_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.o_proj": "rowwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=4096,
        dense_mlp_intermediate_size=11008,
        moe_intermediate_size=2048,
        shared_expert_intermediate_size=2048,
        n_routed_experts =64,
        num_experts_per_tok=4,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        gating_function="softmax",
        normalize_expert_weights=1.0,
        restore_weight_scale=True,
        max_position_embeddings=2048,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=50279,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-5,
        sliding_window=4096,
        use_head_qk_norm=False,
        layer_types: Optional[List[str]] = None,
        dense_layers_indices: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size

        # for dense MLP layers
        self.dense_mlp_intermediate_size = dense_mlp_intermediate_size 

        # for sparse MLP layers
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size # if None, no shared experts
        self.n_routed_experts  = n_routed_experts 
        self.num_experts_per_tok = num_experts_per_tok
        assert gating_function in ["softmax", "sigmoid"], "supported gating function: 'softmax' or 'sigmoid'"
        self.gating_function = gating_function 
        self.normalize_expert_weights = normalize_expert_weights
        self.restore_weight_scale = restore_weight_scale


        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.rms_norm_eps = rms_norm_eps
        self.use_head_qk_norm = use_head_qk_norm

        self.sliding_window = sliding_window
        if layer_types is None:
            self.layer_types: List[str] = [
                "sliding_attention" if (i + 1) % 4 != 0 else "full_attention" for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types: List[str] = layer_types

        layer_type_validation(self.layer_types)

        self.dense_layers_indices = dense_layers_indices if dense_layers_indices is not None else [0,]

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        rope_config_validation(self)


__all__ = ["Olmo3MoeConfig"]
