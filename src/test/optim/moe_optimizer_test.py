import pytest
import torch.nn as nn

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.transformer import CompletedPConfig, TransformerConfig
from olmo_core.optim.completedp import CompletedPOptimConfig, completedp_multipliers
from olmo_core.optim.config import OptimGroupOverride
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizerConfig
from olmo_core.optim.mup import MuPConfig


def test_build_groups_applies_overrides():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    config = OLMoDDPOptimizerConfig(
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


def test_build_groups_applies_mup_with_wrapper_prefix():
    class Wrapper(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

    model = Wrapper(nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 8)))
    config = OLMoDDPOptimizerConfig(
        lr=1e-3,
        weight_decay=0.1,
        mup=MuPConfig(base_width=8),
        group_overrides=[OptimGroupOverride(params=["*bias*"], opts={"weight_decay": 0.0})],
    )
    groups = config.build_groups([model])

    by_name = {name: g for g in groups for name in g["named_params"]}
    assert set(by_name) == {name for name, _ in model.named_parameters()}
    assert by_name["module.0.weight"]["lr"] == pytest.approx(1e-3)
    assert by_name["module.1.weight"]["lr"] == pytest.approx(0.5e-3)
    for name in ("module.0.bias", "module.1.bias"):
        assert by_name[name]["lr"] == pytest.approx(1e-3)
        assert by_name[name]["weight_decay"] == 0.0
    # The default group's splits still come first, and leave weight decay to the optimizer default.
    assert "weight_decay" not in groups[0]
    assert "weight_decay" not in by_name["module.1.weight"]


def test_build_groups_applies_completedp_with_wrapper_prefix():
    class Wrapper(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

    model = Wrapper(
        TransformerConfig.llama_like(
            d_model=128,
            vocab_size=256,
            n_layers=4,
            n_heads=2,
            feed_forward=FeedForwardConfig(hidden_size=512, bias=False),
            completedp=CompletedPConfig(base_width=64, base_depth=2),
        ).build(init_device="meta")
    )
    completedp = CompletedPOptimConfig(base_batch_size=1, batch_size=4)
    config = OLMoDDPOptimizerConfig(
        lr=1e-3,
        eps=1e-8,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        completedp=completedp,
        group_overrides=[
            OptimGroupOverride(params=["*embeddings.weight"], opts={"weight_decay": 0.0})
        ],
    )
    groups = config.build_groups([model])

    by_name = {name: g for g in groups for name in g["named_params"]}
    assert set(by_name) == {name for name, _ in model.named_parameters()}
    multipliers = completedp_multipliers(model, completedp)
    for name, group in by_name.items():
        assert name.startswith("module.")
        assert group["lr"] == pytest.approx(1e-3 * multipliers[name]["lr"])
        assert group["eps"] == pytest.approx(1e-8 * multipliers[name]["eps"])
        assert group["betas"] == pytest.approx((0.6, 0.8))
    assert by_name["module.embeddings.weight"]["weight_decay"] == 0.0
    w2 = "module.blocks.0.feed_forward.w2.weight"
    assert by_name[w2]["weight_decay"] == pytest.approx(0.1 * multipliers[w2]["weight_decay"])


def test_build_groups_requires_completedp_for_completedp_model():
    model = TransformerConfig.llama_like(
        d_model=128,
        vocab_size=256,
        n_layers=2,
        n_heads=2,
        completedp=CompletedPConfig(base_width=64, base_depth=2),
    ).build(init_device="meta")
    with pytest.raises(OLMoConfigurationError, match="needs 'completedp' set"):
        OLMoDDPOptimizerConfig(lr=1e-3).build_groups([model])
