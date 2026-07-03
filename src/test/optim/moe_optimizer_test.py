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
    _zeropower_via_newtonschulz,
)


@pytest.mark.parametrize("shape", [(48, 32), (4, 24, 16)])
def test_zeropower_via_newtonschulz_orthogonalizes(shape):
    # Covers both dispatch branches: the 2D kernel and the batched n-d (grouped-expert) kernel.
    torch.manual_seed(0)
    g = torch.randn(*shape)
    out = _zeropower_via_newtonschulz(
        g, MUON_DEFAULT_NS_COEFFICIENTS, MUON_DEFAULT_NS_STEPS, MUON_DEFAULT_EPS
    )
    # Newton-Schulz drives every singular value toward 1 (Muon computes it in bfloat16), collapsing
    # the input's wide spectrum — so a no-op / broken kernel would leave values outside this band.
    svals = torch.linalg.svdvals(out.float())
    assert svals.min().item() > 0.6
    assert svals.max().item() < 1.3


def test_zeropower_via_newtonschulz_requires_2d():
    with pytest.raises(ValueError, match="at least 2 dims"):
        _zeropower_via_newtonschulz(
            torch.randn(8), MUON_DEFAULT_NS_COEFFICIENTS, MUON_DEFAULT_NS_STEPS, MUON_DEFAULT_EPS
        )


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
