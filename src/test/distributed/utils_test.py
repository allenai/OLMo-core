import pytest
import torch.distributed as dist

import olmo_core.distributed.utils as dist_utils

from .utils import BACKENDS, run_distributed_test


def scatter_object():
    if dist.get_rank() == 0:
        x = ("abc", "def")
    else:
        x = ("abc", "abc")
    x = dist_utils.scatter_object(x)
    assert x == ("abc", "def")


@pytest.mark.parametrize("backend", BACKENDS)
def test_scatter_object(backend: str):
    run_distributed_test(scatter_object, backend=backend)


def run_get_fs_local_rank(tmp_path):
    # Test with a shared global dir.
    assert dist_utils.get_fs_local_rank(dir=tmp_path) == dist_utils.get_rank()

    # Test with a non-shared directory.
    rank_specific_dir = tmp_path / f"rank{dist_utils.get_rank()}"
    rank_specific_dir.mkdir()
    assert dist_utils.get_fs_local_rank(dir=rank_specific_dir) == 0


def test_get_fs_local_rank(tmp_path):
    run_distributed_test(run_get_fs_local_rank, func_args=(tmp_path,))
