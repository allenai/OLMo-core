import pytest

import olmo_core.kernels.symm_mem_vdev2d as symm_mod
from olmo_core.kernels import olmo_symm_mem


def test_nvshmem_world_barrier_calls_extension(monkeypatch):
    class _Ext:
        called = False

        def olmo_symm_world_barrier(self):
            self.called = True

    ext = _Ext()
    monkeypatch.setattr(symm_mod, "_load_cuda_extension", lambda: ext)

    symm_mod.nvshmem_world_barrier()

    assert ext.called


def test_bootstrap_world_barrier_calls_extension_for_bootstrap_group(monkeypatch):
    class _Ext:
        called = False

        def olmo_symm_world_barrier(self):
            self.called = True

    ext = _Ext()
    group = object()
    monkeypatch.setattr(olmo_symm_mem, "_BOOTSTRAP_GLOBAL_RANKS", (0, 1))
    monkeypatch.setattr(olmo_symm_mem, "_group_global_ranks", lambda current_group: (0, 1))
    monkeypatch.setattr(olmo_symm_mem, "_load_cuda_extension", lambda: ext)

    olmo_symm_mem.barrier(group)  # type: ignore[arg-type]

    assert ext.called


def test_bootstrap_world_barrier_rejects_inner_subgroup(monkeypatch):
    def _raise_if_loaded():
        raise AssertionError("subgroup barrier should fail before loading the extension")

    group = object()
    monkeypatch.setattr(olmo_symm_mem, "_BOOTSTRAP_GLOBAL_RANKS", (0, 1, 2, 3))
    monkeypatch.setattr(olmo_symm_mem, "_group_global_ranks", lambda current_group: (0, 1))
    monkeypatch.setattr(olmo_symm_mem, "_load_cuda_extension", _raise_if_loaded)

    with pytest.raises(RuntimeError, match="bootstrap world"):
        olmo_symm_mem.barrier(group)  # type: ignore[arg-type]
