import pytest
import torch

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


def test_peer_base_ptrs_registers_group_and_calls_extension(monkeypatch):
    class _Ext:
        def __init__(self):
            self.called_with = None

        def olmo_symm_peer_base_ptrs(self, tensor, group_name):
            self.called_with = (tensor, group_name)
            return torch.tensor([int(tensor.data_ptr())], dtype=torch.long)

    class _Group:
        group_name = "test_group"

    ext = _Ext()
    group = _Group()
    tensor = torch.empty(4)
    registered = {}
    barriers = []

    def _register_group(current_group, *, device=None):
        registered["group"] = current_group
        registered["device"] = device

    monkeypatch.setattr(olmo_symm_mem, "register_group", _register_group)
    monkeypatch.setattr(olmo_symm_mem, "_load_cuda_extension", lambda: ext)
    monkeypatch.setattr(olmo_symm_mem.dist, "barrier", lambda *, group: barriers.append(group))

    ptrs = olmo_symm_mem.peer_base_ptrs(tensor, group=group)  # type: ignore[arg-type]

    assert registered == {"group": group, "device": tensor.device}
    assert barriers == [group]
    assert ext.called_with == (tensor, "test_group")
    assert ptrs.tolist() == [tensor.data_ptr()]
