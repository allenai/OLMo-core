import os
import time
from unittest.mock import Mock

import pytest

from olmo_core.kernels import cuda_extension_utils
from olmo_core.kernels.cuda_extension_utils import (
    LazyCudaExtension,
    _cuda_arch_tag,
    _env_float,
    _force_rebuild_build_directory,
    _maybe_remove_stale_build_lock,
    _torch_extension_abi_tag,
)


def test_env_float(monkeypatch):
    monkeypatch.setenv("OLMO_TEST_F", "1.5")
    assert _env_float(["OLMO_TEST_F"], 0.0) == 1.5
    # Unparseable value falls through to the default.
    monkeypatch.setenv("OLMO_TEST_F", "notafloat")
    assert _env_float(["OLMO_TEST_F"], 2.0) == 2.0
    monkeypatch.delenv("OLMO_TEST_F", raising=False)
    assert _env_float(["OLMO_TEST_F"], 3.0) == 3.0


def test_arch_and_abi_tags():
    arch = _cuda_arch_tag()
    assert isinstance(arch, str)
    assert arch == "cpu" or arch.startswith("sm")

    abi = _torch_extension_abi_tag()
    assert abi.startswith("torch")
    assert "cxxabi" in abi
    # Should be a filesystem-safe tag (no separators that would break a build dir name).
    assert "/" not in abi and " " not in abi


def test_maybe_remove_stale_build_lock(tmp_path):
    # No lock present -> no-op (and no error).
    _maybe_remove_stale_build_lock(tmp_path, timeout_seconds=0.0)

    lock = tmp_path / "lock"
    lock.touch()
    # Backdate the lock so the mtime fallback (used when /proc is unavailable) also treats it as
    # stale; where /proc is available it's removed because no process holds it open.
    old = time.time() - 3600
    os.utime(lock, (old, old))

    _maybe_remove_stale_build_lock(tmp_path, timeout_seconds=1.0)
    assert not lock.exists()


def test_force_rebuild_build_directory_non_distributed(tmp_path):
    build_dir = tmp_path / "ext_build"
    build_dir.mkdir()
    stale = build_dir / "stale.o"
    stale.touch()

    # Disabled -> no-op.
    _force_rebuild_build_directory(str(build_dir), enabled=False)
    assert stale.exists()

    # Enabled, not distributed -> fs-local rank 0 wipes and recreates the directory.
    _force_rebuild_build_directory(str(build_dir), enabled=True)
    assert build_dir.is_dir()
    assert not stale.exists()


def test_lazy_extension_defers_and_caches_build(monkeypatch):
    loaded = object()
    build = Mock(return_value=loaded)
    discover = Mock(return_value={"extra_ldflags": ["-lnccl"]})
    monkeypatch.setattr(cuda_extension_utils, "load_cuda_extension", build)
    extension = LazyCudaExtension(
        name="test",
        base_name="test_ext",
        sources=("test.cpp",),
        dynamic_build_kwargs=discover,
    )
    build.assert_not_called()
    discover.assert_not_called()

    assert extension.load() is loaded
    assert extension.load() is loaded
    build.assert_called_once()
    discover.assert_called_once()
    assert build.call_args.kwargs["extra_ldflags"] == ["-lnccl"]
    assert build.call_args.kwargs["sources"][0].name == "test.cpp"


@pytest.mark.parametrize("fail_discovery", [False, True])
def test_lazy_extension_caches_original_failure(monkeypatch, fail_discovery):
    original = RuntimeError("dependency unavailable")
    discover = Mock(side_effect=original if fail_discovery else None, return_value={})
    build = Mock(side_effect=original)
    monkeypatch.setattr(cuda_extension_utils, "load_cuda_extension", build)
    extension = LazyCudaExtension(
        name="test",
        base_name="test_ext",
        sources=("test.cpp",),
        dynamic_build_kwargs=discover,
    )

    with pytest.raises(RuntimeError, match="Failed to build/load CUDA test") as first:
        extension.load()
    with pytest.raises(RuntimeError, match="CUDA test extension is unavailable") as repeated:
        extension.load()

    assert first.value.__cause__ is original
    assert repeated.value.__cause__ is original
    discover.assert_called_once()
    assert build.call_count == (0 if fail_discovery else 1)
