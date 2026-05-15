import copy

import torch

from olmo_core.nn import OutputDiscardCheckpoint
from olmo_core.nn import output_discard_checkpoint as odc_module


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


def test_output_discard_checkpoint_python_fallback(monkeypatch):
    """
    Force the Python fallback for share_storage and verify the checkpoint still
    produces correct grads and restored storage. Exercises the path used on
    machines without a working C++ toolchain even when one is available in CI.
    """
    monkeypatch.setattr(odc_module, "_get_share_storage", lambda: None)

    torch.manual_seed(13)

    submodule_ref = torch.nn.Sequential(
        torch.nn.Linear(256, 512),
        torch.nn.GELU(),
        torch.nn.Linear(512, 256),
    )
    next_layer_ref = torch.nn.Linear(256, 128)

    submodule_ckpt = copy.deepcopy(submodule_ref)
    next_layer_ckpt = copy.deepcopy(next_layer_ref)

    x_ref = torch.randn(4, 256, requires_grad=True)
    x_ckpt = x_ref.detach().clone().requires_grad_(True)

    y_ref = submodule_ref(x_ref)
    z_ref = next_layer_ref(y_ref)
    z_ref.square().mean().backward()

    ckpt = OutputDiscardCheckpoint()
    y_ckpt = ckpt.checkpoint(submodule_ckpt, x_ckpt)
    z_ckpt = next_layer_ckpt(y_ckpt)
    ckpt.discard_output_and_register_recompute(z_ckpt)
    assert y_ckpt.untyped_storage().nbytes() == 0
    z_ckpt.square().mean().backward()

    assert y_ckpt.untyped_storage().nbytes() > 0
    torch.testing.assert_close(x_ckpt.grad, x_ref.grad, atol=1e-6, rtol=1e-6)
    for p_ref, p_ckpt in zip(submodule_ref.parameters(), submodule_ckpt.parameters()):
        torch.testing.assert_close(p_ckpt.grad, p_ref.grad, atol=1e-6, rtol=1e-6)
