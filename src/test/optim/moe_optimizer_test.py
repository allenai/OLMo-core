import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from olmo_core.optim.config import OptimGroupOverride
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizer, OLMoDDPOptimizerConfig
from olmo_core.testing.distributed import run_distributed_test


def _run_refresh_main_params():
    mesh = init_device_mesh("cpu", (2,))
    param = nn.Parameter(torch.arange(12, dtype=torch.bfloat16).reshape(3, 4))
    optim = object.__new__(OLMoDDPOptimizer)
    optim.should_maintain_fp32_main_param = True
    optim.param_groups = [{"named_params": {"weight": param}}]
    for placement in (Replicate(), Shard(0)):
        main = distribute_tensor(torch.full((12,), -1.0), mesh, [placement])
        moment = distribute_tensor(torch.full((12,), 0.5), mesh, [placement])
        optim.states = {"weight.main": main, "weight.exp_avg": moment}
        optim._copy_model_params_to_main_params()
        assert optim.states["weight.main"] is main
        torch.testing.assert_close(main.full_tensor(), torch.arange(12).float(), rtol=0, atol=0)
        torch.testing.assert_close(moment.full_tensor(), torch.full((12,), 0.5), rtol=0, atol=0)
        torch.testing.assert_close(
            param, torch.arange(12, dtype=torch.bfloat16).reshape(3, 4), rtol=0, atol=0
        )


def test_refresh_main_params_after_model_load():
    run_distributed_test(
        _run_refresh_main_params, world_size=2, backend="gloo", start_method="spawn"
    )


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
