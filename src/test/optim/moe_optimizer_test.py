import math

import pytest
import torch
import torch.nn as nn

from olmo_core.optim.config import OptimGroupOverride
from olmo_core.optim.moe_optimizer import (
    MUON_DEFAULT_EPS,
    MUON_DEFAULT_NS_COEFFICIENTS,
    MUON_DEFAULT_NS_STEPS,
    MoEFusedV2OptimizerConfig,
    _adjust_muon_lr,
    _is_fp8_weight_store,
    _zeropower_via_newtonschulz,
)


def test_zeropower_via_newtonschulz_orthogonalizes():
    torch.manual_seed(0)
    g = torch.randn(48, 32)  # a realistic (wide-spectrum) gradient
    in_svals = torch.linalg.svdvals(g)
    out = _zeropower_via_newtonschulz(
        g, MUON_DEFAULT_NS_COEFFICIENTS, MUON_DEFAULT_NS_STEPS, MUON_DEFAULT_EPS
    )
    assert out.shape == g.shape
    # Muon computes the orthogonalization in bfloat16.
    svals = torch.linalg.svdvals(out.float())
    # Newton-Schulz pushes all singular values toward 1, collapsing the input's spread.
    assert 0.6 < svals.min().item()
    assert svals.max().item() < 1.3
    assert svals.max() / svals.min() < in_svals.max() / in_svals.min()


def test_zeropower_via_newtonschulz_requires_2d():
    with pytest.raises(ValueError, match="at least 2 dims"):
        _zeropower_via_newtonschulz(
            torch.randn(8), MUON_DEFAULT_NS_COEFFICIENTS, MUON_DEFAULT_NS_STEPS, MUON_DEFAULT_EPS
        )


def test_zeropower_via_newtonschulz_nd_preserves_shape():
    torch.manual_seed(0)
    g = torch.randn(4, 16, 16)  # batched (>2 dims) — grouped-expert weight layout
    out = _zeropower_via_newtonschulz(
        g, MUON_DEFAULT_NS_COEFFICIENTS, MUON_DEFAULT_NS_STEPS, MUON_DEFAULT_EPS
    )
    assert out.shape == g.shape


def test_adjust_muon_lr():
    shape = torch.Size([8, 2])
    assert _adjust_muon_lr(1.0, None, shape) == pytest.approx(math.sqrt(4.0))
    assert _adjust_muon_lr(1.0, "original", shape) == pytest.approx(math.sqrt(4.0))
    assert _adjust_muon_lr(1.0, "match_rms_adamw", shape) == pytest.approx(0.2 * math.sqrt(8))
    with pytest.raises(ValueError, match="Unsupported Muon lr adjustment"):
        _adjust_muon_lr(1.0, "bogus", shape)


def test_build_groups_applies_overrides():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    config = MoEFusedV2OptimizerConfig(
        lr=1e-3,
        group_overrides=[OptimGroupOverride(params=["*bias*"], opts={"weight_decay": 0.0})],
    )
    groups = config.build_groups([model])
    assert isinstance(groups, list)

    # Every parameter lands in exactly one group.
    grouped = [p for g in groups for p in g["named_params"].values()]
    assert len(grouped) == len(list(model.parameters()))

    # The bias parameters land in a group carrying the override option.
    bias_ids = {id(p) for n, p in model.named_parameters() if "bias" in n}
    override_group = next(g for g in groups if g.get("weight_decay") == 0.0)
    assert {id(p) for p in override_group["named_params"].values()} == bias_ids


def test_is_fp8_weight_store_false_for_plain_param():
    assert _is_fp8_weight_store(nn.Parameter(torch.zeros(2, 2))) is False
    assert _is_fp8_weight_store(torch.zeros(2, 2)) is False
