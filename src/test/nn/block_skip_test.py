import torch

from olmo_core.nn.transformer import TransformerConfig


def _tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    cfg = TransformerConfig.llama_like(
        d_model=64, n_layers=3, n_heads=4, n_kv_heads=2, vocab_size=128
    )
    model = cfg.build(init_device="cpu")
    model.init_weights()
    return model


def test_run_all_matches_base():
    model = _tiny_model()
    ids = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        ref = model(ids)
    model.enable_block_skip(target=0.5)
    assert model._block_skip["routed"] == [0, 1, 2]
    with torch.no_grad():
        out = model(ids)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_skip_all_is_identity_through_blocks():
    model = _tiny_model()
    model.enable_block_skip(target=0.5)
    for blk in model.blocks.values():
        blk._bskip_router.w.bias.data.fill_(-10.0)
    ids = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        out = model(ids)
        h = model.embeddings(ids)
        ref = model.lm_head(h)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    assert model._block_skip["holder"].mean_keep(last_forward=False) == 0.0


def test_skipped_tokens_are_not_keys():
    """Skipping token j in block 1 must change later tokens' outputs (it left the key set) but not
    earlier ones."""
    model = _tiny_model()
    model.enable_block_skip(target=0.5)
    ids = torch.randint(0, 128, (1, 10))
    with torch.no_grad():
        ref = model(ids)
    blk = model.blocks["1"]
    # steer token 3 to skip block 1 via a huge weight on a one-hot-ish direction: simpler to
    # monkeypatch the router output
    orig = blk._bskip_router.forward

    def routed(x):
        out = orig(x)
        out[:, 3] = -10.0
        return out

    blk._bskip_router.forward = routed
    with torch.no_grad():
        out = model(ids)
    torch.testing.assert_close(out[:, :3], ref[:, :3])
    assert not torch.allclose(out[:, 3], ref[:, 3])  # token 3 lost block 1 itself
    assert not torch.allclose(out[:, 4:], ref[:, 4:])  # later tokens lost token 3 as a key
    assert abs(model._block_skip["holder"].mean_keep(last_forward=False) - (29 / 30)) < 1e-6


def test_budget_gradient_and_joint_budget():
    from olmo_core.nn.joint_budget import install_joint_budget

    model = _tiny_model()
    model.enable_block_skip(target=0.25)
    model.enable_kv_route(target=0.5)
    model.enable_nested_ffn_moe(start_layer=1, divisors=(1, 4), width_multiple=1, target_cost=0.5)
    jb = install_joint_budget(model, target=0.5, seq_len=32)
    assert abs(jb["s_fixed"] + jb["s_skip"] - 1.0) < 1e-6  # every block is skippable here
    ids = torch.randint(0, 128, (1, 24))
    out = model(ids, labels=ids.clone())
    out.loss.backward()
    g = model.blocks["0"]._bskip_router.w.bias.grad
    assert g is not None and g.item() > 0  # run-all at init, budget pushes the run prob DOWN
    assert model.blocks["0"].attention._kvr_router.w.bias.grad is not None
    assert 0.9 < jb["last_cost"] <= 1.0 + 1e-6
