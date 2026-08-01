"""
Forward/parity test for the ``olmo3moe`` HF model + its conversion.

Builds a small ``Olmo3MoeForCausalLM`` in memory, runs its forward, then does a full
``HF -> OLMo-core -> HF`` conversion roundtrip and checks the reloaded model produces the same
logprobs. This exercises the ``modeling_olmo3moe`` forward and validates the conversion against the
model's real parameter set (strict ``load_state_dict``). Requires ``transformers``.
"""

import pytest
import torch


def _has_olmo3moe() -> bool:
    try:
        from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import (  # noqa: F401
            Olmo3MoeForCausalLM,
        )

        return True
    except ImportError:
        return False


requires_olmo3moe = pytest.mark.skipif(
    not _has_olmo3moe(), reason="requires transformers (for the olmo3moe HF model)"
)


def _small_config():
    from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig

    # Layer 0 is dense, layer 1 is MoE (dense_layers_indices=[0]); head qk-norm on so the
    # q_norm/k_norm params the converter maps exist.
    return Olmo3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        dense_mlp_intermediate_size=16,
        shared_expert_intermediate_size=16,
        max_position_embeddings=32,
        use_head_qk_norm=True,
        dense_layers_indices=[0],
    )


@requires_olmo3moe
def test_olmo3moe_logprobs_match_after_conversion_roundtrip():
    from olmo_core.nn.hf.convert import convert_state_from_hf, convert_state_to_hf
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM

    config = _small_config()
    model = Olmo3MoeForCausalLM(config)
    model.eval()

    input_ids = torch.randint(
        0, config.vocab_size, (1, 8), generator=torch.Generator().manual_seed(0)
    )
    with torch.no_grad():
        ref_logits = model(input_ids).logits
    ref_logprobs = torch.log_softmax(ref_logits, dim=-1)

    hf_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    oc_state = convert_state_from_hf(config, hf_state, model_type="olmo3moe")
    hf_roundtrip = convert_state_to_hf(config, oc_state)

    model.load_state_dict(hf_roundtrip, strict=True)
    model.eval()
    with torch.no_grad():
        rt_logits = model(input_ids).logits
    rt_logprobs = torch.log_softmax(rt_logits, dim=-1)

    torch.testing.assert_close(rt_logprobs, ref_logprobs, rtol=1e-4, atol=1e-4)


@requires_olmo3moe
def test_olmo3moe_experts_grouped_mm_matches_reference_loop():
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeExpert, Olmo3MoeExperts

    torch.manual_seed(0)
    num_experts, hidden_size, intermediate_size, n_tokens, top_k = 4, 32, 16, 7, 2
    experts = Olmo3MoeExperts(
        Olmo3MoeExpert(
            hidden_size=hidden_size, moe_intermediate_size=intermediate_size, hidden_act="silu"
        )
        for _ in range(num_experts)
    )

    hidden_states = torch.randn(n_tokens, hidden_size)
    topk_ids = torch.randint(0, num_experts, (n_tokens, top_k))
    topk_weights = torch.rand(n_tokens, top_k)

    # The dims satisfy the 16-byte alignment guard, but grouped_mm itself needs a new enough torch
    # (the package supports torch>=2.6, while grouped_mm's `offs=` API lands in 2.10). Skip rather
    # than fail where the op is unavailable; the forward falls back to `_forward_loop` there anyway.
    if not experts._can_use_grouped_mm(hidden_states):
        pytest.skip("torch grouped_mm unavailable")
    reference = experts._forward_loop(hidden_states, topk_ids, topk_weights)
    grouped = experts._forward_grouped_mm(hidden_states, topk_ids, topk_weights)
    torch.testing.assert_close(grouped, reference, rtol=1e-5, atol=1e-5)


@requires_olmo3moe
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA KDA kernels")
def test_olmo3moe_kda_cached_decode_matches_full_forward():
    pytest.importorskip("fla")
    from transformers.cache_utils import DynamicCache

    from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM

    config = Olmo3MoeConfig(
        vocab_size=64,
        hidden_size=64,
        attention_hidden_size=64,
        head_dim=16,
        dense_mlp_intermediate_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        use_head_qk_norm=True,
        use_rope=False,
        attention_gate_type="elementwise",
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        linear_allow_neg_eigval=True,
        layer_types=["linear_attention", "full_attention"],
        dense_layers_indices=[0],
        embed_norm=True,
        use_peri_ln=True,
    )
    model = Olmo3MoeForCausalLM(config).cuda().to(torch.bfloat16).eval()
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 16),
        generator=torch.Generator().manual_seed(0),
        device="cuda",
    )

    with torch.no_grad():
        full_logits = model(input_ids, use_cache=False).logits
        cache = DynamicCache(config=config)
        cached_logits = [model(input_ids[:, :12], past_key_values=cache, use_cache=True).logits]
        for position in range(12, input_ids.shape[1]):
            cached_logits.append(
                model(
                    input_ids[:, position : position + 1],
                    past_key_values=cache,
                    use_cache=True,
                ).logits
            )
        cached_logits = torch.cat(cached_logits, dim=1)

    torch.testing.assert_close(cached_logits.float(), full_logits.float(), rtol=2e-2, atol=2e-2)
    assert cache.get_seq_length() == input_ids.shape[1]
