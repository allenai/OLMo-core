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
