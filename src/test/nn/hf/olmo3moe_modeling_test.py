"""
Forward/parity test for the ``olmo3moe`` HF model + its conversion.

Builds a small ``Olmo3MoeForCausalLM`` in memory, runs its forward, then does a full
``HF -> OLMo-core -> HF`` conversion roundtrip and checks the reloaded model produces the same
logprobs. This exercises the ``modeling_olmo3moe`` forward and validates the conversion against the
model's real parameter set (strict ``load_state_dict``). Requires ``transformers``.
"""

import pytest
import torch

from olmo_core.testing.utils import requires_fla, requires_gpu, requires_triton


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


def _small_kda_emo_config():
    from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig

    # Exercise the ladder's important architectural choices together: NoPE KDA plus gated full
    # attention, a dense first layer represented by the shared-expert module, and full-width EMo
    # routing (latent_moe_dim=None). At inference the EMo pool spans every expert, so its routing is
    # faithfully representable as ordinary global top-k routing in the HF implementation.
    return Olmo3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        attention_hidden_size=32,
        head_dim=8,
        dense_mlp_intermediate_size=24,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=24,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        use_head_qk_norm=True,
        use_rope=False,
        attention_gate_type="elementwise",
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        linear_allow_neg_eigval=True,
        latent_moe_dim=None,
        layer_types=["linear_attention", "full_attention"],
        dense_layers_indices=[0],
        dense_layers_use_shared_expert=True,
        embed_norm=True,
        use_peri_ln=True,
        emo_min_document_expert_pool=2,
        emo_max_document_expert_pool=4,
        emo_eval_document_expert_pool=4,
        emo_eos_token_id=63,
        global_load_balancing=True,
        use_cache=False,
    )


def _assert_logprobs_match_after_conversion_roundtrip(
    config, device: torch.device = torch.device("cpu")
):
    from olmo_core.nn.hf.convert import convert_state_from_hf, convert_state_to_hf
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM

    model = Olmo3MoeForCausalLM(config).to(device)
    model.eval()

    input_ids = torch.randint(
        0, config.vocab_size, (1, 8), generator=torch.Generator().manual_seed(0)
    ).to(device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        ref_logits = model(input_ids, use_cache=False).logits
    ref_logprobs = torch.log_softmax(ref_logits, dim=-1)

    hf_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    oc_state = convert_state_from_hf(config, hf_state, model_type="olmo3moe")
    hf_roundtrip = convert_state_to_hf(config, oc_state)

    model.load_state_dict(hf_roundtrip, strict=True)
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        rt_logits = model(input_ids, use_cache=False).logits
    rt_logprobs = torch.log_softmax(rt_logits, dim=-1)

    torch.testing.assert_close(rt_logprobs, ref_logprobs, rtol=1e-4, atol=1e-4)


@requires_olmo3moe
def test_olmo3moe_logprobs_match_after_conversion_roundtrip():
    config = _small_config()
    _assert_logprobs_match_after_conversion_roundtrip(config)


@requires_olmo3moe
def test_olmo3moe_kda_rejects_padding_mask_before_attention():
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import _validate_linear_attention_mask

    attention_mask = torch.tensor([[1, 1, 1, 0]])

    with pytest.raises(NotImplementedError, match="attention-mask support"):
        _validate_linear_attention_mask(attention_mask)


@requires_olmo3moe
@requires_gpu
@requires_fla
@requires_triton
def test_olmo3moe_kda_emo_logprobs_match_after_conversion_roundtrip():
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import (
        Olmo3MoeForCausalLM,
        Olmo3MoeKimiDeltaAttention,
    )

    config = _small_kda_emo_config()

    assert "linear_attention" in config.layer_types
    assert config.latent_moe_dim is None
    assert config.dense_layers_use_shared_expert is True
    assert config.emo_eval_document_expert_pool == config.n_routed_experts
    assert config.global_load_balancing is True

    model = Olmo3MoeForCausalLM(config)
    kda = next(module for module in model.modules() if isinstance(module, Olmo3MoeKimiDeltaAttention))
    assert torch.isfinite(kda.A_log).all()
    assert torch.equal(kda.dt_bias, torch.zeros_like(kda.dt_bias))

    _assert_logprobs_match_after_conversion_roundtrip(config, torch.device("cuda"))


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
