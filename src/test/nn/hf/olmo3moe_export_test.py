"""Export actual core-built hybrid models, not states synthesized by the HF implementation."""

from copy import deepcopy

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.attention import (
    AttentionConfig,
    GateConfig,
    GateGranularity,
    KimiDeltaAttentionConfig,
)
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.hf.checkpoint import save_hf_model
from olmo_core.nn.hf.config import get_hf_config
from olmo_core.nn.hf.convert import convert_state_from_hf, convert_state_to_hf
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe.emo import EmoRouterConfig
from olmo_core.nn.moe.moe import LatentMoEConfig
from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.testing.utils import has_fla, requires_gpu

requires_fla = pytest.mark.skipif(not has_fla, reason="Requires flash-linear-attention")


def build_hybrid(*, latent=True, emo=False, per_head=True, scalable=True, device="cpu"):
    """Small generic fixture with a dense KDA layer and both sparse attention types."""
    width, experts = 128, 4
    norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False)
    kda = KimiDeltaAttentionConfig(
        n_heads=2,
        n_v_heads=2,
        head_dim=64,
        expand_v=2.0,
        allow_neg_eigval=True,
        use_experimental_kernels=False,
    )
    attention = AttentionConfig(
        n_heads=2,
        n_kv_heads=1,
        head_dim=64,
        bias=False,
        rope=None,
        qk_norm=deepcopy(norm),
        use_head_qk_norm=True,
        qk_norm_per_head_gains=per_head,
        scalable_softmax=scalable,
        backend=AttentionBackendName.torch,
        gate=GateConfig(granularity=GateGranularity.elementwise),
    )
    shared = SharedExpertsConfig(
        d_model=width, hidden_size=128, num_experts=1, bias=False, dtype=DType.float32
    )
    sparse = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        sequence_mixer=kda,
        layer_norm=norm,
        shared_experts=shared,
        routed_experts=RoutedExpertsConfig(
            d_model=64 if latent else width,
            hidden_size=128,
            num_experts=experts,
            bias=False,
            dtype=DType.float32,
        ),
        routed_experts_router=MoERouterConfigV2(
            d_model=width,
            num_experts=experts,
            top_k=2,
            emo=EmoRouterConfig(
                eos_token_id=63,
                min_document_expert_pool=2,
                max_document_expert_pool=experts,
                eval_document_expert_pool=experts,
            )
            if emo
            else None,
        ),
        latent_moe=LatentMoEConfig(latent_dim=64) if latent else None,
        use_peri_norm=True,
        use_pre_norm=False,
    )
    dense = deepcopy(sparse)
    dense.routed_experts = dense.routed_experts_router = dense.latent_moe = None
    full = deepcopy(sparse)
    full.sequence_mixer = attention
    model = OLMoDDPModelConfig(
        name=TransformerType.moe_fused_v2,
        d_model=width,
        n_layers=3,
        vocab_size=64,
        block=sparse,
        block_overrides={0: dense, 2: full},
        lm_head=LMHeadConfig(layer_norm=deepcopy(norm), bias=False),
        embedding_norm=deepcopy(norm),
        embed_scale=width**0.5,
        init_seed=42,
    ).build(init_device=device)
    model.init_weights(max_seq_len=32)
    # Non-uniform gains catch head/sequence broadcasting errors hidden by all-ones initialization.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".q_norm.weight" in name or ".k_norm.weight" in name or name.endswith("ssmax_scale"):
                parameter.copy_(
                    torch.linspace(0.5, 1.5, parameter.numel()).reshape(parameter.shape)
                )
    return model.eval()


@requires_fla
@pytest.mark.parametrize("latent,emo", [(False, True), (True, True), (True, False)])
@pytest.mark.parametrize(
    "per_head,scalable", [(False, False), (True, False), (False, True), (True, True)]
)
def test_core_hybrid_exports_and_reloads(tmp_path, latent, emo, per_head, scalable):
    model = build_hybrid(latent=latent, emo=emo, per_head=per_head, scalable=scalable)
    config = get_hf_config(model)
    assert config.latent_moe_dim == (64 if latent else None)
    assert config.qk_norm_per_head_gains == per_head and config.scalable_softmax == scalable
    assert config.emo_eos_token_id == (63 if emo else None)
    state = model.state_dict()
    converted = convert_state_to_hf(config, state)
    hf = Olmo3MoeForCausalLM(config)
    hf.load_state_dict(converted, strict=True, assign=True)
    restored = convert_state_from_hf(config, hf.state_dict(), model_type="olmo3moe")
    assert restored.keys() == state.keys()
    for key in state:
        torch.testing.assert_close(restored[key], state[key], rtol=0, atol=0, msg=key)
    save_hf_model(tmp_path / "hf", state, model)
    reloaded = Olmo3MoeForCausalLM.from_pretrained(tmp_path / "hf")
    assert reloaded.state_dict().keys() == converted.keys()
    for key, value in reloaded.state_dict().items():
        torch.testing.assert_close(
            value, converted[key], rtol=0, atol=0, check_dtype=False, msg=key
        )


@requires_fla
@requires_gpu
@pytest.mark.parametrize("emo", [False, True])
def test_core_hybrid_hf_forward_parity(emo):
    model = build_hybrid(emo=emo, device="cuda")
    config = get_hf_config(model)
    config._attn_implementation = "eager"
    hf = Olmo3MoeForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
    hf.load_state_dict(convert_state_to_hf(config, model.state_dict()), strict=True)
    tokens = torch.randint(0, 63, (2, 32), device="cuda")
    with torch.no_grad():
        expected = model(tokens).float()
        actual = hf(tokens, use_cache=False).logits.float()
    # Different attention/expert kernels round in BF16; compare the model's predictive distribution.
    torch.testing.assert_close(
        actual.log_softmax(-1), expected.log_softmax(-1), atol=0.02, rtol=0.005
    )


@requires_fla
@pytest.mark.parametrize("per_head", [False, True])
@pytest.mark.parametrize("scalable", [False, True])
def test_core_hf_attention_forward_parity(per_head, scalable):
    model = build_hybrid(per_head=per_head, scalable=scalable).float()
    config = get_hf_config(model)
    config._attn_implementation = "eager"
    hf = Olmo3MoeForCausalLM(config).eval()
    hf.load_state_dict(convert_state_to_hf(config, model.state_dict()), strict=True)
    x = torch.randn(2, 7, model.d_model)
    # Sequence length differs from either head count; gains are deliberately non-uniform.
    mask = torch.full((7, 7), float("-inf")).triu(1)[None, None]
    with torch.no_grad():
        expected = model.blocks["2"].attention(x)
        actual, _ = hf.model.layers[2].self_attn(
            x,
            position_embeddings=None,
            attention_mask=mask,
            position_ids=torch.arange(7)[None],
        )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@requires_fla
def test_hybrid_export_rejects_restricted_emo_inference_pool():
    model = build_hybrid(emo=True)
    model.blocks["1"].routed_experts_router.emo.eval_document_expert_pool = 2
    with pytest.raises(NotImplementedError, match="eval_document_expert_pool"):
        get_hf_config(model)


@requires_fla
def test_hybrid_hf_rejects_packed_positions():
    model = build_hybrid()
    hf = Olmo3MoeForCausalLM(get_hf_config(model))
    with pytest.raises(NotImplementedError, match="Packed/reset"):
        hf(torch.ones(1, 4, dtype=torch.long), position_ids=torch.tensor([[0, 1, 0, 1]]))
