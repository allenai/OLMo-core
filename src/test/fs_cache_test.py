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


def test_dedicated_cache_does_not_enable_global_remote_metadata_cache(tmp_path, monkeypatch):
    import olmo_core.io as io

    dedicated_env = "OLMO_CORE_DATA_VERIFICATION_CACHE_DIR"
    monkeypatch.delenv(fs_cache.CACHE_DIR_ENV_VAR, raising=False)
    monkeypatch.setenv(dedicated_env, str(tmp_path / "verification"))
    calls = []

    @fs_cache.maybe_cache(cache_dir_env_var=dedicated_env)
    def verified(value):
        calls.append(value)
        return value

    assert verified(1) == verified(1) == 1
    assert calls == [1]
    sizes = iter([10, 20])
    monkeypatch.setattr(io, "_http_file_size", lambda path: next(sizes))
    assert io.get_file_size("https://example.test/data.arrow") == 10
    assert io.get_file_size("https://example.test/data.arrow") == 20
    monkeypatch.delenv(dedicated_env)
    assert verified(1) == 1
    assert calls == [1, 1]
