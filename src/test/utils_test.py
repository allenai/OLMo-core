from dataclasses import dataclass

import pytest
import torch

from olmo_core.utils import apply_to_tensors, flatten_dict, format_float


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


def test_flatten_dict():
    assert flatten_dict(
        {
            "a": {"foo": 1, "bar": {"baz": 2}},
            "b": 2,
        }
    ) == {
        "a.foo": 1,
        "a.bar.baz": 2,
        "b": 2,
    }


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.0, "0.0"),
        (1e-5, "1.00E-05"),
        (1234.0, "1,234"),
        (1234.56, "1,234"),
        (1_234_567.0, "1,234,567"),
        (2_500_000_000.0, "2.500 B"),
        (1_000_000_000_000.0, "1.0000 T"),
        (123_456_789_000_000_000_000.0, "123.5 E"),
        (-1_234_567.0, "-1,234,567"),
        (float("inf"), "inf"),
        (float("-inf"), "-inf"),
        (float("nan"), "nan"),
    ],
)
def test_format_float(value, expected):
    assert format_float(value) == expected
