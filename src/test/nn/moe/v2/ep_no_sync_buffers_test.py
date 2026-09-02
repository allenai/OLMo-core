from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from olmo_core.nn.ddp import model as ddp_model
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig
from olmo_core.nn.moe.v2.ep_no_sync_buffers import (
    _alloc_ep_symm_tensor,
    _cached_symm_tensor_covers,
    _parse_bool_env,
    _view_cached_symm_tensor,
    compute_ep_no_sync_rank_capacity,
)
from olmo_core.train.globals import set_global_arg


def test_parse_bool_env():
    assert _parse_bool_env("auto", env_name="X") is None
    assert _parse_bool_env("", env_name="X") is None
    for v in ("1", "true", "YES", "y", "on"):
        assert _parse_bool_env(v, env_name="X") is True
    for v in ("0", "false", "No", "n", "off"):
        assert _parse_bool_env(v, env_name="X") is False
    with pytest.raises(RuntimeError, match="X must be one of"):
        _parse_bool_env("maybe", env_name="X")


def test_cached_symm_tensor_covers():
    cpu = torch.device("cpu")
    cached = torch.empty(8, 4, dtype=torch.float32, device=cpu)
    # Same shape, and a smaller leading dim, are both covered.
    assert _cached_symm_tensor_covers(cached, (8, 4), torch.float32, cpu)
    assert _cached_symm_tensor_covers(cached, (5, 4), torch.float32, cpu)
    # A larger leading dim, mismatched trailing dim, or mismatched dtype are not.
    assert not _cached_symm_tensor_covers(cached, (9, 4), torch.float32, cpu)
    assert not _cached_symm_tensor_covers(cached, (8, 5), torch.float32, cpu)
    assert not _cached_symm_tensor_covers(cached, (8, 4), torch.bfloat16, cpu)


def test_view_cached_symm_tensor():
    cached = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    # Exact shape returns the tensor itself.
    assert _view_cached_symm_tensor(cached, (8, 4)) is cached
    # Smaller leading dim returns a contiguous prefix view aliasing the same storage.
    view = _view_cached_symm_tensor(cached, (5, 4))
    assert tuple(view.shape) == (5, 4)
    assert view.data_ptr() == cached.data_ptr()
    torch.testing.assert_close(view, cached[:5])


def test_runtime_symm_allocation_guard(monkeypatch):
    monkeypatch.setenv("OLMO_EP_NO_SYNC_FORBID_RUNTIME_SYMM_ALLOC", "1")
    set_global_arg("ep_no_sync_symm_initialization_complete", True)
    try:
        with pytest.raises(RuntimeError, match="after initialization completed"):
            _alloc_ep_symm_tensor(
                shape=(8, 4),
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
                group=cast(Any, object()),
            )
    finally:
        set_global_arg("ep_no_sync_symm_initialization_complete", False)


def test_compute_ep_no_sync_rank_capacity():
    block: Any = SimpleNamespace(ep=ExpertParallelConfig(capacity_factor=1.25))
    assert compute_ep_no_sync_rank_capacity(block, 10) == 13  # ceil(1.25 * 10)
    assert compute_ep_no_sync_rank_capacity(block, 0) == 1  # floored to at least 1
    block2: Any = SimpleNamespace(ep=ExpertParallelConfig(capacity_factor=2.0))
    assert compute_ep_no_sync_rank_capacity(block2, 4) == 8


def test_ep_no_sync_prewarm_uses_runtime_bf16_dtype(monkeypatch):
    recorded_dtypes = []
    recorded_d_models = []

    def record_dtype(*args, dtype, d_model, **kwargs):
        del args, kwargs
        recorded_dtypes.append(dtype)
        recorded_d_models.append(d_model)

    monkeypatch.setattr(ddp_model, "get_ep_no_sync_buffers", record_dtype)
    monkeypatch.setattr(
        ddp_model,
        "prewarm_ep_no_sync_rowwise_lifetime_leases",
        record_dtype,
    )
    monkeypatch.setattr(
        ddp_model,
        "use_ep_no_sync_rowwise_symm_dispatch_in",
        lambda block: False,
    )
    monkeypatch.setattr(
        ddp_model,
        "use_ep_no_sync_rowwise_symm_combine_out",
        lambda block: False,
    )
    monkeypatch.setattr(
        ddp_model,
        "use_ep_no_sync_rowwise_symm_combine_gather",
        lambda block: False,
    )

    block = SimpleNamespace(
        routed_experts_router=SimpleNamespace(top_k=2),
        routed_experts=SimpleNamespace(d_model=2),
        ep_pg=object(),
        ep=SimpleNamespace(
            uses_rowwise_buffers=True,
            rowwise_transport="pytorch",
            shared_slots=1,
            capacity_factor=1.25,
        ),
        rowwise_fp8=None,
        checkpoint_attn=False,
        checkpoint_permute_moe_unpermute=False,
    )
    model = SimpleNamespace(
        named_ep_no_sync_blocks=lambda: [("0", block)],
        parameters=lambda: iter([torch.nn.Parameter(torch.empty(1, dtype=torch.float32))]),
        tbo=False,
        d_model=4,
        recompute_all_blocks_by_chunk=False,
        recompute_each_block=False,
        recompute_block_keys=None,
        compile_enabled=False,
        _compile_requested=False,
        _ep_no_sync_dummy_symm_tensors=[],
    )

    ddp_model.OLMoDDPModel.prewarm_ep_no_sync_symm_buffers(
        model,
        max_local_microbatch_size=8,
        pad_to_block_count=1,
        rowwise_lifetime_lease_slots=4,
    )

    assert recorded_dtypes == [torch.bfloat16, torch.bfloat16]
    assert recorded_d_models == [2, 2]
