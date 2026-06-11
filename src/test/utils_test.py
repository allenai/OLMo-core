import gc
import threading
import time
from dataclasses import dataclass

import pytest
import torch

from olmo_core.exceptions import OLMoThreadError
from olmo_core.utils import (
    apply_to_tensors,
    flatten_dict,
    format_float,
    threaded_generator,
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


def test_threaded_generator():
    assert list(threaded_generator(iter(range(100)), maxsize=4)) == list(range(100))


def test_threaded_generator_raises_on_failure():
    def bad_generator():
        yield 0
        raise ValueError("oh no")

    with pytest.raises(OLMoThreadError):
        list(threaded_generator(bad_generator()))


def test_threaded_generator_abandoned_consumer_does_not_leak_thread():
    def infinite_source():
        i = 0
        while True:
            yield i
            i += 1

    thread_names = [f"test_threaded_generator_leak {i}" for i in range(8)]
    for name in thread_names:
        gen = threaded_generator(infinite_source(), maxsize=2, thread_name=name)
        # Take a single item, then abandon the generator before it's exhausted.
        assert next(gen) == 0
        gen.close()

    del gen
    gc.collect()

    # The producer threads should shut down promptly once their consumers are gone.
    deadline = time.monotonic() + 5.0
    while True:
        leaked = [t.name for t in threading.enumerate() if t.name in thread_names]
        if not leaked or time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    assert not leaked, f"threaded_generator leaked threads: {leaked}"


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
