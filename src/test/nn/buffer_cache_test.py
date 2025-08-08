import torch

from olmo_core.nn.buffer_cache import BufferCache


def test_buffer_cache():
    cache = BufferCache()
    cache["a"] = torch.tensor(1)
    cache["b"] = torch.tensor(2)
    assert set(cache) == {"a", "b"}
    assert len(cache) == 2
    assert cache["a"].item() == 1
    assert (x := cache.get("a")) is not None and x.item() == 1

    cache2 = cache.with_namespace("foo")
    assert len(cache2) == 0
    cache2["a"] = torch.tensor(3)
    assert len(cache2) == 1
    assert cache2["a"].item() == 3
    assert cache2["a"].item() == 3
    assert (x := cache2.get("a")) is not None and x.item() == 3
    assert len(cache) == 2
    assert cache["a"].item() == 1
