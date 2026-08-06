from dataclasses import dataclass

import pytest
import torch

from olmo_core.utils import (
    apply_to_tensors,
    flatten_dict,
    format_float,
    get_devices_without_bfloat16,
)


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
        (2_500_000_000.0, "2.500B"),
        (1_000_000_000_000.0, "1.0000T"),
        (123_456_789_000_000_000_000.0, "123.5E"),
        (-1_234_567.0, "-1,234,567"),
        (float("inf"), "inf"),
        (float("-inf"), "-inf"),
        (float("nan"), "nan"),
    ],
)
def test_format_float(value, expected):
    assert format_float(value) == expected


@pytest.mark.parametrize(
    "capability, has_bfloat16",
    [
        ((6, 1), False),  # Pascal, the P100 and the T4's predecessor
        ((7, 0), False),  # Volta, the V100
        ((7, 5), False),  # Turing, the T4: tensor cores, no bfloat16
        ((8, 0), True),  # Ampere, the A100, where the format arrives
        ((8, 6), True),  # Ampere, the A10G
        ((9, 0), True),  # Hopper
    ],
)
def test_get_devices_without_bfloat16_reads_the_threshold_off_the_die(
    capability, has_bfloat16, monkeypatch
):
    """Mutation: make the comparison ``<=`` so that 8.0 itself is reported as lacking it.

    8.0 is the first capability with the arithmetic, not the last without it, and getting the
    boundary wrong the safe-looking way refuses the A100 -- which is a card people run on.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=None: "a card")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)

    assert (get_devices_without_bfloat16() == []) is has_bfloat16


def test_get_devices_without_bfloat16_reads_every_visible_device(monkeypatch):
    """Mutation: read device 0 and assume the rest of the host matches it.

    This is called before any rank has run ``torch.cuda.set_device``, so every rank would read
    device 0. On a homogeneous node that is the same answer eight times; on anything else it is
    the wrong answer seven times.
    """
    cards = {0: ("NVIDIA A10G", (8, 6)), 1: ("Tesla T4", (7, 5))}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(cards))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: cards[index][0])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: cards[index][1])

    assert get_devices_without_bfloat16() == [(1, "Tesla T4", (7, 5))]
