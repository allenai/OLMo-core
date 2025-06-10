import sys

import pytest
import torch

from olmo_core.distributed.parallel import (
    build_world_mesh,
    get_cp_mesh,
    get_dp_mesh,
    get_dp_model_mesh,
    get_pp_mesh,
    get_tp_mesh,
    get_world_mesh,
)
from olmo_core.distributed.parallel.context_parallel import ContextParallelConfig
from olmo_core.distributed.parallel.data_parallel import DataParallelConfig, DataParallelType
from olmo_core.distributed.parallel.expert_parallel import ExpertParallelConfig
from olmo_core.distributed.parallel.pipeline_parallel import PipelineParallelConfig
from olmo_core.distributed.parallel.tensor_parallel import TensorParallelConfig
from olmo_core.distributed.utils import get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.testing import BACKENDS, run_distributed_test


def _build_and_check_world_mesh(dp_degree, tp_degree, cp_degree, pp_degree, ep_degree):
    """Helper that is executed on each rank under ``run_distributed_test``."""

    dp_cfg = DataParallelConfig(name=DataParallelType.fsdp) if dp_degree > 0 else None
    tp_cfg = TensorParallelConfig(degree=tp_degree) if tp_degree > 0 else None
    cp_cfg = ContextParallelConfig(degree=cp_degree) if cp_degree > 0 else None
    pp_cfg = PipelineParallelConfig(degree=pp_degree) if pp_degree > 0 else None
    ep_cfg = ExpertParallelConfig(degree=ep_degree) if ep_degree > 0 else None

    # Expert and tensor parallelism are incompatible
    if tp_cfg is not None and ep_cfg is not None:
        with pytest.raises(OLMoConfigurationError):
            build_world_mesh(dp=dp_cfg, tp=tp_cfg, cp=cp_cfg, pp=pp_cfg, ep=ep_cfg)
        return

    # Build the mesh
    mesh = build_world_mesh(dp=dp_cfg, tp=tp_cfg, cp=cp_cfg, pp=pp_cfg, ep=ep_cfg)
    assert get_world_mesh() is mesh
    world_size = get_world_size()

    # Calculate expected DP world size
    divisor = 1
    if tp_degree > 0:
        divisor *= tp_degree
    if cp_degree > 0:
        divisor *= cp_degree
    if pp_degree > 0:
        divisor *= pp_degree
    if ep_degree > 0:
        divisor *= ep_degree
    expected_dp_world = world_size // divisor

    assert mesh.mesh_dim_names is not None
    dim_names = [
        n.decode() if isinstance(n, (bytes, bytearray)) else n for n in mesh.mesh_dim_names
    ]
    dims_map = dict(zip(dim_names, mesh.shape))

    # Check dimensions exist and have correct values
    if dp_degree > 0:
        assert dims_map.get("dp", dims_map.get("dp_ep", 0)) == expected_dp_world
    if cp_degree > 0:
        assert dims_map["cp"] == cp_degree
    if tp_degree > 0:
        assert dims_map["tp"] == tp_degree
    if pp_degree > 0:
        assert dims_map["pp"] == pp_degree
    if ep_degree > 0:
        assert dims_map.get("ep", dims_map.get("dp_ep", 0)) == ep_degree or "dp_ep" in dims_map

    # Check sub-meshes can be accessed
    if dp_degree > 0:
        dp_mesh = get_dp_mesh(mesh)
        assert dp_mesh.shape == (expected_dp_world,)

    if cp_degree > 0:
        cp_mesh = get_cp_mesh(mesh)
        assert cp_mesh.shape == (cp_degree,)

    if tp_degree > 0:
        tp_mesh = get_tp_mesh(mesh)
        assert tp_mesh.shape == (tp_degree,)

    if pp_degree > 0:
        pp_mesh = get_pp_mesh(mesh)
        assert pp_mesh.shape == (pp_degree,)

    # Model DP mesh flattens DP and CP dims
    if dp_degree > 0:
        dp_model_mesh = get_dp_model_mesh(mesh)
        expected_model_dp = expected_dp_world
        if cp_degree > 0:
            expected_model_dp *= cp_degree
        assert dp_model_mesh.shape == (expected_model_dp,)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "dp_degree,tp_degree,cp_degree,pp_degree,ep_degree,world_size",
    [
        pytest.param(1, 0, 0, 0, 0, 4, id="dp_only"),
        pytest.param(1, 2, 0, 0, 0, 4, id="dp_tp"),
        pytest.param(1, 0, 2, 0, 0, 4, id="dp_cp"),
        pytest.param(1, 0, 0, 2, 0, 4, id="dp_pp"),
        pytest.param(1, 0, 0, 0, 2, 4, id="dp_ep"),
        pytest.param(1, 2, 2, 0, 0, 4, id="dp_tp_cp"),
        pytest.param(1, 2, 0, 2, 0, 8, id="dp_tp_pp"),
        pytest.param(1, 0, 2, 2, 0, 8, id="dp_cp_pp"),
        pytest.param(1, 0, 0, 2, 2, 8, id="dp_ep_pp"),
        pytest.param(1, 2, 2, 2, 0, 8, id="dp_tp_cp_pp"),
        pytest.param(1, 2, 0, 0, 2, 4, id="ep_tp_error"),
    ],
)
def test_build_world_mesh_parameterized(
    backend: str,
    dp_degree: int,
    tp_degree: int,
    cp_degree: int,
    pp_degree: int,
    ep_degree: int,
    world_size: int,
):
    # Skip NCCL backend if there aren't enough GPUs for the desired world size
    if "nccl" in backend and torch.cuda.device_count() < world_size:
        pytest.skip("Not enough GPUs available for this test.")

    run_distributed_test(
        _build_and_check_world_mesh,
        backend=backend,
        world_size=world_size,
        func_args=(dp_degree, tp_degree, cp_degree, pp_degree, ep_degree),
        start_method="spawn",
    )
