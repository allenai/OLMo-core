import torch

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe import MoERouterGatingFunction
from olmo_core.nn.moe.v2.block import (
    MoEFusedV2TransformerBlock,
    MoEFusedV2TransformerBlockConfig,
)
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.transformer import (
    MoEFusedV2TransformerConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.testing import requires_gpu, requires_grouped_gemm


def test_v2_no_ep_module_names_importable():
    """The no-EP core ships ``no_ep`` / ``checkpointing`` / ``tbo_state``; the EP families land later."""
    from olmo_core.nn.moe.v2 import checkpointing, no_ep, tbo_state

    assert hasattr(no_ep, "combined_forward_no_ep")
    assert hasattr(checkpointing, "checkpoint_recompute_context_fn")
    assert hasattr(tbo_state, "SyncedTboPendingContext")


def _build_block(
    *,
    d_model: int = 512,
    hidden_size: int = 1024,
    num_experts: int = 4,
    top_k: int = 1,
    uniform_expert_assignment: bool = True,
    init_device: str = "cuda",
) -> MoEFusedV2TransformerBlock:
    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=DType.float32,
    )
    return MoEFusedV2TransformerBlock(
        d_model=d_model,
        block_idx=0,
        n_layers=1,
        sequence_mixer=AttentionConfig(
            name=AttentionType.default,
            n_heads=2,
            n_kv_heads=2,
            bias=False,
            use_flash=False,
            dtype=DType.float32,
        ),
        attention_norm=layer_norm,
        routed_experts_router=MoERouterConfigV2(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            gating_function=MoERouterGatingFunction.softmax,
            uniform_expert_assignment=uniform_expert_assignment,
            lb_loss_weight=None,
            z_loss_weight=None,
            dtype=DType.float32,
        ),
        shared_experts_router=None,
        shared_experts=None,
        routed_experts=RoutedExpertsConfig(
            d_model=d_model,
            hidden_size=hidden_size,
            num_experts=num_experts,
            bias=False,
            dtype=DType.float32,
        ),
        feed_forward_norm=layer_norm,
        ep_no_sync=False,
        ep_no_sync_major_align=1,
        init_device=init_device,
    )


def _init_block_params(block: MoEFusedV2TransformerBlock):
    torch.manual_seed(1234)
    with torch.no_grad():
        for p in block.parameters():
            if p.is_floating_point():
                p.normal_(mean=0.0, std=0.02)


def _install_forced_router(block: MoEFusedV2TransformerBlock):
    """Force all tokens to expert 0 so the forward is deterministic and doesn't depend on routing."""

    def _make_forced_forward(router):
        def _forced_forward(local_x, scores_only, loss_div_factor=None):
            del loss_div_factor
            B, S, _ = local_x.shape
            if scores_only:
                return (
                    torch.ones(
                        B, S, router.num_experts, device=local_x.device, dtype=local_x.dtype
                    ),
                    None,
                    None,
                    None,
                )

            expert_weights = torch.ones(
                B, S, router.top_k, device=local_x.device, dtype=local_x.dtype
            )
            expert_indices = torch.zeros(
                B, S, router.top_k, device=local_x.device, dtype=torch.long
            )
            batch_size_per_expert = torch.zeros(
                router.num_experts, device=local_x.device, dtype=torch.long
            )
            batch_size_per_expert[0] = B * S * router.top_k
            return expert_weights, expert_indices, batch_size_per_expert, None

        return _forced_forward

    assert block.routed_experts_router is not None
    block.routed_experts_router.forward = _make_forced_forward(block.routed_experts_router)
    if block.shared_experts_router is not None:
        block.shared_experts_router.forward = _make_forced_forward(block.shared_experts_router)


@requires_gpu
@requires_grouped_gemm
def test_v2_no_ep_forward_backward_smoke():
    block = _build_block(init_device="cuda")
    _init_block_params(block)
    _install_forced_router(block)
    block.train()

    x = torch.randn(1, 8, block.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
    y = block(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for p in block.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


@requires_gpu
@requires_grouped_gemm
def test_v2_no_ep_apply_compile_forward_smoke():
    block = _build_block(
        d_model=128,
        hidden_size=256,
        num_experts=4,
        top_k=1,
        init_device="cuda",
    )
    _init_block_params(block)
    block.to(dtype=torch.bfloat16)
    _install_forced_router(block)
    block.train()
    block.apply_compile()

    x = torch.randn(1, 4, block.d_model, device="cuda", dtype=torch.bfloat16)
    y = block(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def _build_model_config(*, d_model: int = 128, n_layers: int = 2) -> MoEFusedV2TransformerConfig:
    dtype = DType.float32
    layer_norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=dtype)
    return MoEFusedV2TransformerConfig(
        init_seed=0,
        d_model=d_model,
        recompute_each_block=False,
        vocab_size=128,
        n_layers=n_layers,
        name=TransformerType.moe_fused_v2,
        block=MoEFusedV2TransformerBlockConfig(
            name=TransformerBlockType.moe_fused_v2,
            attention=AttentionConfig(
                name=AttentionType.default,
                n_heads=4,
                bias=False,
                use_flash=False,
                dtype=dtype,
            ),
            routed_experts=RoutedExpertsConfig(
                d_model=d_model, hidden_size=256, num_experts=4, bias=False, dtype=dtype
            ),
            routed_experts_router=MoERouterConfigV2(
                d_model=d_model, num_experts=4, top_k=2, dtype=dtype
            ),
            shared_experts=None,
            layer_norm=layer_norm,
        ),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
    )


@requires_gpu
@requires_grouped_gemm
def test_v2_transformer_config_builds_and_initializes():
    """End-to-end check of the transformer integration: config dispatch + ``init_moe_v2``."""
    config = _build_model_config()
    model = config.build(init_device="cuda")
    assert type(model).__name__ == "MoEFusedV2Transformer"

    model.init_weights(device=torch.device("cuda"))
    for p in model.parameters():
        assert torch.isfinite(p).all()
