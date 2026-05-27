import os
import time

from olmo_core.kernels.cuda_extension_utils import (
    _env_bool,
    _env_float,
    cuda_arch_tag,
    maybe_remove_stale_build_lock,
    torch_extension_abi_tag,
)


def test_env_bool(monkeypatch):
    monkeypatch.setenv("OLMO_TEST_FLAG", "true")
    assert _env_bool(["OLMO_TEST_FLAG"]) is True
    monkeypatch.setenv("OLMO_TEST_FLAG", "off")
    assert _env_bool(["OLMO_TEST_FLAG"]) is False
    monkeypatch.delenv("OLMO_TEST_FLAG", raising=False)
    assert _env_bool(["OLMO_TEST_FLAG"], default=True) is True
    # First set name wins.
    monkeypatch.setenv("OLMO_TEST_FLAG_2", "yes")
    assert _env_bool(["OLMO_TEST_FLAG", "OLMO_TEST_FLAG_2"]) is True


def test_env_float(monkeypatch):
    monkeypatch.setenv("OLMO_TEST_F", "1.5")
    assert _env_float(["OLMO_TEST_F"], 0.0) == 1.5
    # Unparseable value falls through to the default.
    monkeypatch.setenv("OLMO_TEST_F", "notafloat")
    assert _env_float(["OLMO_TEST_F"], 2.0) == 2.0
    monkeypatch.delenv("OLMO_TEST_F", raising=False)
    assert _env_float(["OLMO_TEST_F"], 3.0) == 3.0


def test_arch_and_abi_tags():
    arch = cuda_arch_tag()
    assert isinstance(arch, str)
    assert arch == "cpu" or arch.startswith("sm")

    abi = torch_extension_abi_tag()
    assert abi.startswith("torch")
    assert "cxxabi" in abi
    # Should be a filesystem-safe tag (no separators that would break a build dir name).
    assert "/" not in abi and " " not in abi


def test_maybe_remove_stale_build_lock(tmp_path):
    # No lock present -> no-op (and no error).
    maybe_remove_stale_build_lock(tmp_path, timeout_seconds=0.0)

    lock = tmp_path / "lock"
    lock.touch()
    # Backdate the lock so the mtime fallback (used when /proc is unavailable) also treats it as
    # stale; where /proc is available it's removed because no process holds it open.
    old = time.time() - 3600
    os.utime(lock, (old, old))

    maybe_remove_stale_build_lock(tmp_path, timeout_seconds=1.0)
    assert not lock.exists()
