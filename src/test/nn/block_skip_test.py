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
    for r in model.bskip_routers.values():
        r.w.bias.data.fill_(-10.0)
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
    import olmo_core.nn.router_fn as rf

    orig = rf.router_logits

    def routed(h, router, eps=1e-6):  # steer token 3 to skip block 1
        out = orig(h, router, eps)
        if router is model.bskip_routers["1"]:
            out = out.clone()
            out[:, 3] = -10.0
        return out

    rf.router_logits = routed
    try:
        with torch.no_grad():
            out = model(ids)
    finally:
        rf.router_logits = orig
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
    g = model.bskip_routers["0"].w.bias.grad
    assert g is not None and g.item() > 0  # run-all at init, budget pushes the run prob DOWN
    assert model.kvr_routers["0"].w.bias.grad is not None
    assert 0.9 < jb["last_cost"] <= 1.0 + 1e-6


def test_activation_checkpointing_matches_and_frees():
    """With full AC the skip wrapper is checkpointed as one region: same outputs/grads as without."""
    from olmo_core.nn.transformer import TransformerActivationCheckpointingMode

    ids = torch.randint(0, 128, (1, 24))
    outs = []
    for ac in (False, True):
        model = _tiny_model()
        model.enable_block_skip(target=0.5)
        for r in model.bskip_routers.values():  # a real skip pattern
            r.w.weight.data.normal_(0, 0.5)
            r.w.bias.data.fill_(0.0)
        if ac:
            model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)
        out = model(ids, labels=ids.clone())
        out.loss.backward()
        outs.append((out.loss.detach().clone(), model.bskip_routers["1"].w.weight.grad.clone(),
                     model._block_skip["holder"].mean_keep(last_forward=False)))
    torch.testing.assert_close(outs[0][0], outs[1][0])
    torch.testing.assert_close(outs[0][1], outs[1][1], atol=1e-5, rtol=1e-4)
    assert outs[0][2] == outs[1][2] and 0.0 < outs[0][2] < 1.0


def test_budget_attach_matches_loss_term_gradients():
    """budget_attach delivers the same router gradients as the separate loss term once the lagged
    coefficients exist (second forward), for the single-router two-sided budget."""
    ids = torch.randint(0, 128, (1, 24))
    grads = []
    for attach in (False, True):
        model = _tiny_model()
        model.enable_block_skip(target=0.25, budget_weight=1.0, target_anneal_calls=0)
        model.budget_attach = attach
        for _ in range(2):  # first pass primes the lagged expectations
            model.zero_grad(set_to_none=True)
            out = model(ids, labels=ids.clone())
            out.loss.backward()
        grads.append(model.bskip_routers["0"].w.bias.grad.clone())
    # with attach the loss excludes the budget, but the router gradient must be the same sign and
    # magnitude (the coefficient is the exact derivative of |mean - target| at the lagged point,
    # which equals the current point here: the router did not move between the two forwards)
    torch.testing.assert_close(grads[0], grads[1], atol=1e-6, rtol=1e-4)
