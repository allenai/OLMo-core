import torch

from olmo_core.config import DType
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig


def _build(*, d_model=16, hidden_size=32, num_experts=2):
    module = SharedExpertsConfig(
        d_model=d_model,
        hidden_size=hidden_size,
        num_experts=num_experts,
        bias=False,
        dtype=DType.float32,
    ).build(init_device="cpu")
    # Weights are allocated with torch.empty; initialize them for a deterministic forward.
    with torch.no_grad():
        module.w_up_gate.normal_()
        module.w_down.normal_()
    return module


def test_shared_experts_num_params():
    cfg = SharedExpertsConfig(
        d_model=16, hidden_size=32, num_experts=2, bias=False, dtype=DType.float32
    )
    # up + gate + down weights, per expert (no bias).
    assert cfg.num_params() == 3 * 16 * 32 * 2


def test_shared_experts_forward_shape():
    torch.manual_seed(0)
    B, S, D, H, E = 2, 4, 16, 32, 3
    module = _build(d_model=D, hidden_size=H, num_experts=E)
    out = module(torch.randn(B, S, D))
    assert out.shape == (E, B, S, D)


def test_shared_experts_single_expert():
    module = _build(num_experts=1)
    out = module(torch.randn(2, 4, 16))
    assert out.shape == (1, 2, 4, 16)


def test_shared_experts_split_forward_matches_full():
    torch.manual_seed(0)
    B, S, D, H, E = 2, 4, 16, 32, 2
    module = _build(d_model=D, hidden_size=H, num_experts=E)
    x = torch.randn(B, S, D)

    full = module(x)
    up, gate = module.forward1(x)
    split = module.forward2(up, gate, x.shape)

    torch.testing.assert_close(full, split)
