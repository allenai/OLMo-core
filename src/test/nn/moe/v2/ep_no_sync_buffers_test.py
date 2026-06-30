from types import SimpleNamespace

import pytest
import torch

from olmo_core.nn.moe.v2.ep_no_sync_buffers import (
    _cached_symm_tensor_covers,
    _parse_bool_env,
    _view_cached_symm_tensor,
    compute_ep_no_sync_rank_capacity,
)


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


def test_compute_ep_no_sync_rank_capacity():
    block = SimpleNamespace(ep_no_sync_capacity_factor=1.25)
    assert compute_ep_no_sync_rank_capacity(block, 10) == 13  # ceil(1.25 * 10)
    assert compute_ep_no_sync_rank_capacity(block, 0) == 1  # floored to at least 1
    assert compute_ep_no_sync_rank_capacity(SimpleNamespace(ep_no_sync_capacity_factor=2.0), 4) == 8
