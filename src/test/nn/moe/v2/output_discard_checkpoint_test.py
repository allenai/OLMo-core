import copy

import torch

from olmo_core.nn import OutputDiscardCheckpoint


def test_output_discard_checkpoint_discards_and_restores_storage():
    torch.manual_seed(11)
    ckpt = OutputDiscardCheckpoint()

    submodule = torch.nn.Linear(512, 512)
    x = torch.randn(8, 512, requires_grad=True)

    y = ckpt.checkpoint(submodule, x)
    assert isinstance(y, torch.Tensor)

    # Make a grad-producing tensor that does not require reading y in backward.
    # This exercises hook-triggered recompute + storage restoration without
    # relying on full graph stitching.
    add_bias = torch.nn.Parameter(torch.randn_like(y))
    z = y + add_bias

    ckpt.discard_output_and_register_recompute(z)
    assert ckpt.outputs is not None
    assert ckpt.outputs[0].untyped_storage().nbytes() == 0

    loss = z.square().sum()
    loss.backward()

    assert y.untyped_storage().nbytes() > 0
    assert ckpt.outputs is None
    assert ckpt._ctx is None
    assert add_bias.grad is not None


def test_output_discard_checkpoint_allows_backward_through_linear_chain():
    torch.manual_seed(7)

    submodule_ref = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 512),
    )
    next_layer_ref = torch.nn.Linear(512, 256)

    submodule_ckpt = copy.deepcopy(submodule_ref)
    next_layer_ckpt = copy.deepcopy(next_layer_ref)

    x_ref = torch.randn(8, 512, requires_grad=True)
    x_ckpt = x_ref.detach().clone().requires_grad_(True)

    # Baseline.
    y_ref = submodule_ref(x_ref)
    z_ref = next_layer_ref(y_ref)
    loss_ref = z_ref.square().mean()
    loss_ref.backward()

    # Output-discard path.
    ckpt = OutputDiscardCheckpoint()
    y_ckpt = ckpt.checkpoint(submodule_ckpt, x_ckpt)
    z_ckpt = next_layer_ckpt(y_ckpt)
    ckpt.discard_output_and_register_recompute(z_ckpt)
    loss_ckpt = z_ckpt.square().mean()
    loss_ckpt.backward()

    assert x_ref.grad is not None
    assert x_ckpt.grad is not None
    torch.testing.assert_close(x_ckpt.grad, x_ref.grad, atol=1e-6, rtol=1e-6)

    for p_ref, p_ckpt in zip(submodule_ref.parameters(), submodule_ckpt.parameters()):
        assert p_ref.grad is not None
        assert p_ckpt.grad is not None
        torch.testing.assert_close(p_ckpt.grad, p_ref.grad, atol=1e-6, rtol=1e-6)

    for p_ref, p_ckpt in zip(next_layer_ref.parameters(), next_layer_ckpt.parameters()):
        assert p_ref.grad is not None
        assert p_ckpt.grad is not None
        torch.testing.assert_close(p_ckpt.grad, p_ref.grad, atol=1e-6, rtol=1e-6)
