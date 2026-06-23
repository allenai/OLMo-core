import torch
import torch.nn.functional as F

from olmo_core.config import DType
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig


def _build(*, d_model=16, hidden_size=32, num_experts=2):
    # build() initializes the weights, so no manual init is needed.
    return SharedExpertsConfig(
        d_model=d_model,
        hidden_size=hidden_size,
        num_experts=num_experts,
        bias=False,
        dtype=DType.float32,
    ).build(init_device="cpu")


def test_shared_experts_build_initializes_weights():
    # Regression: build() must initialize the parameters (they were previously left as raw
    # torch.empty storage), so a freshly built module yields finite weights and output.
    module = SharedExpertsConfig(
        d_model=16, hidden_size=32, num_experts=2, bias=False, dtype=DType.float32
    ).build(init_device="cpu")

    assert torch.isfinite(module.w_up_gate).all()
    assert torch.isfinite(module.w_down).all()
    assert torch.isfinite(module(torch.randn(2, 4, 16))).all()


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


def test_shared_experts_forward_matches_per_expert_reference():
    torch.manual_seed(0)
    B, S, D, H, E = 2, 4, 16, 32, 3
    module = _build(d_model=D, hidden_size=H, num_experts=E)
    x = torch.randn(B, S, D)

    # Explicit per-expert SwiGLU + down-projection reference for the vectorized forward.
    # w_up_gate is column-packed as [expert, {up, gate}, hidden]; w_down is (E, H, D).
    x2 = x.reshape(B * S, D)
    expected = []
    for e in range(E):
        up_gate = x2 @ module.w_up_gate[:, e * 2 * H : (e + 1) * 2 * H]  # (BS, 2H)
        up, gate = up_gate[:, :H], up_gate[:, H:]
        hidden = F.silu(gate) * up
        expected.append(hidden @ module.w_down[e])  # (BS, D)
    expected = torch.stack(expected, dim=0).view(E, B, S, D)

    torch.testing.assert_close(module(x), expected)
