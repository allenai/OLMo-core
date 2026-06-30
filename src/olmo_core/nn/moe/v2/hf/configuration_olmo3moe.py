from typing import List, Optional

from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation


class Olmo3MoeConfig(PretrainedConfig):
    model_type = "olmo3moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    ignore_keys_at_rope_validation = {"truncate"}
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
        attention_hidden_size=4096,
        head_dim=None,
        dense_mlp_intermediate_size=11008,
        moe_intermediate_size=2048,
        shared_expert_intermediate_size=2048,
        n_routed_experts=64,
        num_experts_per_tok=4,
        original_num_experts_per_tok=None,
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
        embed_scale=1.0,
        embed_norm=False,
        use_peri_ln=False,
        **kwargs,
    ):
        rope_parameters = kwargs.pop("rope_parameters", rope_scaling)
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
        self.attention_hidden_size = (
            attention_hidden_size if attention_hidden_size is not None else hidden_size
        )
        self.head_dim = (
            head_dim if head_dim is not None else self.attention_hidden_size // num_attention_heads
        )

        # for dense MLP layers
        self.dense_mlp_intermediate_size = dense_mlp_intermediate_size

        # for sparse MLP layers
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = (
            shared_expert_intermediate_size  # if None, no shared experts
        )
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.original_num_experts_per_tok = original_num_experts_per_tok
        assert gating_function in [
            "softmax",
            "sigmoid",
        ], "supported gating function: 'softmax' or 'sigmoid'"
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

        if layer_types is None:
            self.layer_types: List[str] = [
                "sliding_attention" if (i + 1) % 2 != 0 else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types: List[str] = layer_types

        layer_type_validation(self.layer_types)

        # Transformers validates nested per-layer RoPE configs against
        # `self.layer_types`, so set layer metadata before assigning RoPE.
        self.rope_theta = rope_theta
        self.rope_scaling = rope_parameters
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.rms_norm_eps = rms_norm_eps
        self.use_head_qk_norm = use_head_qk_norm

        self.embed_scale = embed_scale
        self.embed_norm = embed_norm
        self.use_peri_ln = use_peri_ln

        self.sliding_window = sliding_window

        self.dense_layers_indices = (
            dense_layers_indices if dense_layers_indices is not None else [0]
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        rope_config_validation(self)

    def _validate_yarn_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type", "factor", "rope_theta", "original_max_position_embeddings"}
        optional_keys = {
            "attention_factor",
            "beta_fast",
            "beta_slow",
            "mscale",
            "mscale_all_dim",
            "truncate",
        }
        self._check_received_keys(
            rope_parameters["rope_type"],
            set(rope_parameters.keys()),
            required_keys,
            optional_keys,
            ignore_keys=ignore_keys,
        )

        factor = rope_parameters["factor"]
        if factor is None or not isinstance(factor, (float, int)) or factor < 1.0:
            raise ValueError(f"`rope_parameters`'s factor field must be a float >= 1, got {factor}")
        factor = float(factor)

        original_max_position_embeddings = rope_parameters["original_max_position_embeddings"]
        implicit_factor = self.max_position_embeddings / original_max_position_embeddings
        if abs(implicit_factor - factor) > 1e-12 and implicit_factor != 1:
            raise ValueError(
                "The explicitly set RoPE scaling factor does not match "
                "max_position_embeddings / original_max_position_embeddings: "
                f"{factor} vs {implicit_factor}"
            )


__all__ = ["Olmo3MoeConfig"]
