from dataclasses import dataclass

import pytest
import torch

from olmo_core.utils import apply_to_tensors


@dataclass
class Foo:
    x: torch.Tensor


@pytest.mark.parametrize(
    "container, tensor_count",
    [
        (Foo(x=torch.rand(2, 2)), 1),
        ({"x": torch.rand(2, 2)}, 1),
        ((torch.rand(2, 2),), 1),
        ([torch.rand(2, 2)], 1),
        ({torch.rand(2, 2)}, 1),
        ({"x": {"x": torch.rand(2, 2), "y": torch.rand(1, 1)}}, 2),
        ((torch.rand(1) for _ in range(2)), 2),
    ],
)
def test_apply_to_tensors(container, tensor_count):
    count = 0

    def count_tensors(x):
        nonlocal count
        if isinstance(x, torch.Tensor):
            count += 1

    apply_to_tensors(count_tensors, container)

    assert count == tensor_count
