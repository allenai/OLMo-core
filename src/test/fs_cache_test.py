import os

import olmo_core.fs_cache as fs_cache


def test_fs_cache(tmp_path):
    _CACHE_MISSES = 0

    @fs_cache.maybe_cache()
    def foo(x: int) -> int:
        nonlocal _CACHE_MISSES
        _CACHE_MISSES += 1
        return x * 2

    os.environ[fs_cache.CACHE_DIR_ENV_VAR] = str(tmp_path)
    try:
        foo(x=2)
        assert _CACHE_MISSES == 1
        foo(x=2)
        assert _CACHE_MISSES == 1

        foo(x=4)
        assert _CACHE_MISSES == 2
    finally:
        os.environ.pop(fs_cache.CACHE_DIR_ENV_VAR)

    foo(x=2)
    assert _CACHE_MISSES == 3
