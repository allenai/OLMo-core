"""Per-token block skipping on GatedDeltaNet (hybrid) blocks: the mixer honours ``block_keep``."""

import torch

from olmo_core.nn.attention import GatedDeltaNetConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.testing import requires_gpu
from olmo_core.testing.utils import requires_fla


def _gdn(d_model=256, device="cuda"):
    torch.manual_seed(0)
    m = GatedDeltaNetConfig(n_heads=8).build(d_model, layer_idx=0, n_layers=2, init_device=device)
    m.init_weights(generator=torch.Generator(device=device).manual_seed(0))
    return m


def _hybrid_model(device="cuda"):
    torch.manual_seed(0)
    ln = LayerNormConfig(name=LayerNormType.rms, bias=False)
    cfg = TransformerConfig(
        d_model=256,
        vocab_size=1024,
        n_layers=2,
        block=TransformerBlockConfig(
            name=TransformerBlockType.reordered_norm,
            sequence_mixer=GatedDeltaNetConfig(n_heads=8),
            layer_norm=ln,
            feed_forward=FeedForwardConfig(hidden_size=512, bias=False),
        ),
        lm_head=LMHeadConfig(layer_norm=ln, bias=False),
    )
    model = cfg.build(init_device=device)
    model.init_weights()
    return model


@requires_gpu
@requires_fla
def test_block_keep_all_true_is_a_noop():
    m = _gdn()
    x = torch.randn(2, 32, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = m(x)
        out = m(x, block_keep=torch.ones(2, 32, dtype=torch.bool, device="cuda"))
    torch.testing.assert_close(out, ref)


@requires_gpu
@requires_fla
def test_skipped_token_does_not_write_the_state_or_touch_earlier_tokens():
    """Prefill up to token j leaves state S_j; prefilling through a SKIPPED token j+1 must leave
    the recurrent state at S_j, and the outputs before j+1 unchanged."""
    m = _gdn()
    B, T, j = 1, 24, 10
    x = torch.randn(B, T, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = m(x)
        m.init_state_cache(B, T)
        m(x[:, : j + 1])
        state_after_j = m.state_cache.recurrent_state.clone()
        m.init_state_cache(B, T)
        keep = torch.ones(B, j + 2, dtype=torch.bool, device="cuda")
        keep[:, j + 1] = False
        out = m(x[:, : j + 2], block_keep=keep)
        state_after_skip = m.state_cache.recurrent_state.clone()
    torch.testing.assert_close(state_after_skip, state_after_j)
    torch.testing.assert_close(out[:, : j + 1], ref[:, : j + 1], atol=2e-2, rtol=2e-2)
    m.state_cache = None


@requires_gpu
@requires_fla
def test_hybrid_model_routes_gdn_blocks_and_reproduces_base_at_init():
    model = _hybrid_model()
    ids = torch.randint(0, 1024, (2, 32), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = model(ids)
    model.enable_block_skip(target=0.5)
    assert model._block_skip["routed"] == [0, 1]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(ids)
    torch.testing.assert_close(out, ref)
    # skip everything: the block stack is the identity
    for r in model.bskip_routers.values():
        r.w.bias.data.fill_(-10.0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(ids)
        h = model.embeddings(ids)
        ref_id = model.lm_head(h)
    torch.testing.assert_close(out, ref_id)
    assert model._block_skip["holder"].mean_keep(last_forward=False) == 0.0


@requires_gpu
@requires_fla
def test_joint_budget_prices_gdn_blocks_as_skippable_and_router_gets_gradient():
    from olmo_core.nn.joint_budget import install_joint_budget

    model = _hybrid_model()
    model.enable_block_skip(target=0.25)
    jb = install_joint_budget(model, target=0.5, seq_len=32)
    # every block is a GDN block here: all of it is skippable, nothing is "attention score"
    assert jb["s_attn"] == 0.0
    assert abs(jb["s_fixed"] + jb["s_skip"] - 1.0) < 1e-6
    ids = torch.randint(0, 1024, (1, 32), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(ids, labels=ids.clone())
    out.loss.backward()
    g = model.bskip_routers["0"].w.bias.grad
    assert g is not None and g.item() > 0  # run-all at init, the budget pushes run prob down
