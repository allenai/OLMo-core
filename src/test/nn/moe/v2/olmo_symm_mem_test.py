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


def test_rowwise_collective_preflight_forwards_and_caches_all_settings(monkeypatch):
    class _Ext:
        def __init__(self):
            self.calls = []

        def preflight_rowwise_collective_launches(
            self,
            get_nblocks,
            put_nblocks,
            weighted_put_nblocks,
        ):
            self.calls.append((get_nblocks, put_nblocks, weighted_put_nblocks))

    ext = _Ext()
    current_device = 0
    monkeypatch.setattr(symm_mod, "_load_cuda_extension", lambda: ext)
    monkeypatch.setattr(symm_mod.torch.cuda, "current_device", lambda: current_device)
    monkeypatch.setattr(symm_mod, "_PREFLIGHTED_ROWWISE_COLLECTIVE_LAUNCHES", set())

    symm_mod.preflight_rowwise_collective_launches(256, 256, 128)
    symm_mod.preflight_rowwise_collective_launches(256, 256, 128)
    symm_mod.preflight_rowwise_collective_launches(257, 256, 128)
    symm_mod.preflight_rowwise_collective_launches(256, 257, 128)
    symm_mod.preflight_rowwise_collective_launches(256, 256, 129)
    current_device = 1
    symm_mod.preflight_rowwise_collective_launches(256, 256, 128)

    assert ext.calls == [
        (256, 256, 128),
        (257, 256, 128),
        (256, 257, 128),
        (256, 256, 129),
        (256, 256, 128),
    ]
    assert symm_mod._PREFLIGHTED_ROWWISE_COLLECTIVE_LAUNCHES == {
        (0, 256, 256, 128),
        (0, 257, 256, 128),
        (0, 256, 257, 128),
        (0, 256, 256, 129),
        (1, 256, 256, 128),
    }


@pytest.mark.parametrize(
    ("nblocks", "invalid_setting"),
    [
        ((0, 256, 128), "rowwise_get_nblocks"),
        ((-1, 256, 128), "rowwise_get_nblocks"),
        ((256, 0, 128), "rowwise_put_nblocks"),
        ((256, -1, 128), "rowwise_put_nblocks"),
        ((256, 256, 0), "rowwise_weighted_put_nblocks"),
        ((256, 256, -1), "rowwise_weighted_put_nblocks"),
    ],
)
def test_rowwise_collective_preflight_rejects_nonpositive_settings(
    monkeypatch,
    nblocks,
    invalid_setting,
):
    def _unexpected_cuda_call():
        raise AssertionError("invalid settings should fail before accessing CUDA")

    monkeypatch.setattr(symm_mod.torch.cuda, "current_device", _unexpected_cuda_call)

    with pytest.raises(ValueError, match=invalid_setting):
        symm_mod.preflight_rowwise_collective_launches(*nblocks)


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

    ptrs = olmo_symm_mem.peer_base_ptrs(tensor, group=group)  # type: ignore[attr-defined]

    assert registered == {"group": group, "device": tensor.device}
    assert barriers == [group]
    assert ext.called_with == (tensor, "test_group")
    assert ptrs.tolist() == [tensor.data_ptr()]
