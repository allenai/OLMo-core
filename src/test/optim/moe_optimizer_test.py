import torch
import torch.nn as nn

from olmo_core.optim import moe_optimizer
from olmo_core.optim.config import OptimGroupOverride
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizerConfig


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


def test_copy_model_params_to_main_params(monkeypatch):
    optimizer = object.__new__(moe_optimizer.OLMoDDPOptimizer)
    optimizer.should_maintain_fp32_main_param = True
    model_param = nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    main_param = object()
    optimizer.param_groups = [{"named_params": {"weight": model_param}}]
    optimizer.states = {"weight.main": main_param}
    copied = []

    def record_copy(*, dst, src):
        copied.append((dst, src.clone()))

    monkeypatch.setattr(moe_optimizer, "assign_full_tensor_to_dtensor", record_copy)
    optimizer._copy_model_params_to_main_params()

    assert len(copied) == 1
    assert copied[0][0] is main_param
    torch.testing.assert_close(copied[0][1], model_param.float())
