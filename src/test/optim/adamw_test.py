import torch
import torch.nn as nn

from olmo_core.optim import AdamWConfig, OptimGroupOverride


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(1024, 16)
        self.out = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.wte(x))


def test_adamw_config_to_optim():
    config = AdamWConfig()

    model = MyModel()
    optim = config.build(model)

    assert isinstance(optim, torch.optim.AdamW)
    assert len(optim.param_groups) == 1

    assert config.merge(["lr=1e-1"]).lr == 0.1


def test_adamw_config_to_optim_with_group_overrides():
    config = AdamWConfig(
        group_overrides=[OptimGroupOverride(params=["wte.weight"], opts=dict(weight_decay=0.0))]
    )

    model = MyModel()
    optim = config.build(model)
    assert isinstance(optim, torch.optim.AdamW)
    assert len(optim.param_groups) == 2
    assert optim.param_groups[0]["weight_decay"] == 0.0
    assert len(optim.param_groups[0]["params"]) == 1
    assert len(optim.param_groups[1]["params"]) == len(list(model.parameters())) - 1

    assert config.merge(["lr=1e-1"]).lr == 0.1

    for group in optim.param_groups:
        assert "initial_lr" in group
