import math

import pytest
import torch

from olmo_core.distributed.parallel import (
    build_world_mesh,
    get_cp_mesh,
    get_dp_mesh,
    get_dp_model_mesh,
    get_ep_mesh,
    get_pp_mesh,
    get_tp_mesh,
    get_world_mesh,
)
from olmo_core.distributed.parallel.context_parallel import ContextParallelConfig
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.distributed.parallel.expert_parallel import ExpertParallelConfig
from olmo_core.distributed.parallel.pipeline_parallel import PipelineParallelConfig
from olmo_core.distributed.parallel.tensor_parallel import TensorParallelConfig
from olmo_core.distributed.utils import get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.testing import run_distributed_test
from olmo_core.testing.utils import requires_multi_gpu


def _get_expected_dp_world_size(world_size, tp_degree, cp_degree, pp_degree, ep_degree, dp_type):
    """Get the expected DP world size for a given configuration."""
    divisor = 1
    if tp_degree > 0:
        divisor *= tp_degree
    if cp_degree > 0:
        divisor *= cp_degree
    if pp_degree > 0:
        divisor *= pp_degree
    # For HSDP expert parallel shares the dp_shard dimension, so do not divide by ep_degree.
    if ep_degree > 0 and not (dp_type == DataParallelType.hsdp):
        divisor *= ep_degree
    expected_dp_world = world_size // divisor
    return expected_dp_world


def _build_and_check_world_mesh(dp_degree, tp_degree, cp_degree, pp_degree, ep_degree, dp_type):
    """Helper that is executed on each rank under ``run_distributed_test``."""

    dp_cfg = DataParallelConfig(name=dp_type) if dp_degree > 0 else None
    tp_cfg = TensorParallelConfig(degree=tp_degree) if tp_degree > 0 else None
    cp_cfg = ContextParallelConfig(degree=cp_degree) if cp_degree > 0 else None
    pp_cfg = PipelineParallelConfig(degree=pp_degree) if pp_degree > 0 else None
    ep_cfg = ExpertParallelConfig(degree=ep_degree) if ep_degree > 0 else None

    # Check for invalid configurations that should raise errors
    if ep_cfg is not None:
        # Expert and tensor parallelism are incompatible
        if tp_cfg is not None:
            with pytest.raises(OLMoConfigurationError):
                build_world_mesh(dp=dp_cfg, tp=tp_cfg, cp=cp_cfg, pp=pp_cfg, ep=ep_cfg)
            return

        # Expert parallelism is (currently) only compatible with HSDP
        if dp_cfg is not None and dp_cfg.name != DataParallelType.hsdp:
            with pytest.raises(OLMoConfigurationError):
                build_world_mesh(dp=dp_cfg, tp=tp_cfg, cp=cp_cfg, pp=pp_cfg, ep=ep_cfg)
            return
        elif dp_cfg is not None and dp_cfg.name == DataParallelType.hsdp:
            # Expert parallelism + HSDP requires the same sharding degree
            dp_world_for_hsdp = get_world_size()
            for degree in [pp_degree, cp_degree, tp_degree]:
                if degree > 0:
                    dp_world_for_hsdp //= degree
            _, shard_degree = dp_cfg.get_replicate_and_shard_degree(dp_world_for_hsdp)
            if ep_cfg.degree != shard_degree:
                with pytest.raises(OLMoConfigurationError):
                    build_world_mesh(dp=dp_cfg, tp=tp_cfg, cp=cp_cfg, pp=pp_cfg, ep=ep_cfg)
                return

    # Build the mesh
    mesh = build_world_mesh(dp=dp_cfg, tp=tp_cfg, cp=cp_cfg, pp=pp_cfg, ep=ep_cfg)
    assert get_world_mesh() is mesh

    expected_dp_world_size = _get_expected_dp_world_size(
        get_world_size(), tp_degree, cp_degree, pp_degree, ep_degree, dp_type
    )

    assert mesh.mesh_dim_names is not None
    dims_map = dict(zip(mesh.mesh_dim_names, mesh.shape))

    # Check root device_mesh dimensions exist and have correct values
    if dp_degree > 0:
        # HSDP splits DP into `dp_replicate` and `dp_shard` dimensions.
        if dp_cfg is not None and dp_cfg.name == DataParallelType.hsdp:
            # Expect the product of replicate and shard dims to equal the DP world size.
            assert dims_map["dp_replicate"] * dims_map["dp_shard"] == expected_dp_world_size
        else:
            assert dims_map.get("dp", dims_map.get("dp_ep", 0)) == expected_dp_world_size

    for degree, dim_name in [(cp_degree, "cp"), (tp_degree, "tp"), (pp_degree, "pp")]:
        if degree > 0:
            assert dims_map[dim_name] == degree

    if ep_degree > 0:
        if dp_cfg is not None and dp_cfg.name == DataParallelType.hsdp:
            # EP is folded into dp_shard
            assert dims_map["dp_shard"] == ep_degree
        else:
            assert dims_map.get("ep", dims_map.get("dp_ep", 0)) == ep_degree or "dp_ep" in dims_map

    # Check sub-meshes can be accessed
    if dp_degree > 0:  # special case for DP mesh
        dp_mesh = get_dp_mesh(mesh)
        assert math.prod(dp_mesh.shape) == expected_dp_world_size

        # Model DP mesh flattens DP and CP dims, and also EP dims
        dp_model_mesh = get_dp_model_mesh(mesh)
        expected_model_dp = expected_dp_world_size
        if cp_degree > 0:
            expected_model_dp *= cp_degree
        if ep_degree > 0 and not (dp_cfg is not None and dp_cfg.name == DataParallelType.hsdp):
            expected_model_dp *= ep_degree
        assert math.prod(dp_model_mesh.shape) == expected_model_dp

    for degree, dim_name, get_mesh_fn in [
        (cp_degree, "cp", get_cp_mesh),
        (tp_degree, "tp", get_tp_mesh),
        (pp_degree, "pp", get_pp_mesh),
        (ep_degree, "ep", get_ep_mesh),
    ]:
        if degree > 0:
            sub_mesh = get_mesh_fn(mesh)
            assert sub_mesh.shape == (degree,)


PARALLEL_MESH_TESTCONFIGS = [
    pytest.param(2, 0, 0, 0, 0, 2, id="dp_only"),
    pytest.param(1, 2, 0, 0, 0, 2, id="dp_tp"),
    pytest.param(1, 0, 2, 0, 0, 2, id="dp_cp"),
    pytest.param(1, 0, 0, 2, 0, 2, id="dp_pp"),
    pytest.param(1, 0, 0, 0, 2, 2, id="dp_ep"),
    pytest.param(1, 1, 2, 0, 0, 2, id="dp_tp_cp"),
    pytest.param(1, 1, 1, 2, 0, 2, id="dp_tp_cp_pp"),
    pytest.param(1, 0, 1, 2, 2, 4, id="dp_cp_ep_pp"),
    pytest.param(1, 2, 0, 0, 2, 4, id="dp_tp_ep_configerror"),
]


@pytest.mark.parametrize("dp_type", [DataParallelType.fsdp, DataParallelType.hsdp])
@pytest.mark.parametrize("dp,tp,cp,pp,ep,world_size", PARALLEL_MESH_TESTCONFIGS)
def test_build_world_mesh_cpu(
    dp_type: DataParallelType, dp: int, tp: int, cp: int, pp: int, ep: int, world_size: int
):
    run_distributed_test(
        _build_and_check_world_mesh,
        backend="gloo",
        world_size=world_size,
        func_args=(dp, tp, cp, pp, ep, dp_type),
        start_method="spawn",
    )


@requires_multi_gpu
@pytest.mark.parametrize("dp_type", [DataParallelType.fsdp, DataParallelType.hsdp])
@pytest.mark.parametrize("dp,tp,cp,pp,ep,world_size", PARALLEL_MESH_TESTCONFIGS)
def test_build_world_mesh_gpu(
    dp_type: DataParallelType, dp: int, tp: int, cp: int, pp: int, ep: int, world_size: int
):
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            "Not enough GPUs available for this test (req: {}, avail: {})".format(
                world_size, torch.cuda.device_count()
            )
        )

    run_distributed_test(
        _build_and_check_world_mesh,
        backend="nccl",
        world_size=world_size,
        func_args=(dp, tp, cp, pp, ep, dp_type),
    )
