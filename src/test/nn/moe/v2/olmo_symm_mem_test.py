from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import olmo_core.kernels.symm_mem_vdev2d as symm_mod
from olmo_core.kernels import olmo_symm_mem


@pytest.fixture
def local_only_bootstrap(monkeypatch):
    group = object()
    ext = SimpleNamespace(
        olmo_symm_get_unique_id=Mock(return_value=[1, 2]),
        olmo_symm_init=Mock(),
    )
    state = SimpleNamespace(
        group=group,
        ext=ext,
        load_extension=Mock(return_value=ext),
        visible_uuids=["GPU-B", "GPU-A"],
        peer=("host0", "GPU-B", {"GPU-A", "GPU-B"}, ""),
        topology=[],
        p2p=Mock(return_value=True),
    )

    def gather(output, value, *, group):
        assert group is state.group
        if isinstance(value, tuple):
            state.topology.append(value)
            output[:] = [value, state.peer]
        else:
            output[:] = [value, [3, 4]]

    monkeypatch.setenv("NVSHMEM_REMOTE_TRANSPORT", "none")
    monkeypatch.setattr(olmo_symm_mem, "_BOOTSTRAP_GLOBAL_RANKS", None)
    monkeypatch.setattr(olmo_symm_mem, "_group_global_ranks", lambda group: (8, 9))
    monkeypatch.setattr(olmo_symm_mem, "_load_cuda_extension", state.load_extension)
    monkeypatch.setattr(olmo_symm_mem, "get_node_hostname", lambda: "host0")
    monkeypatch.setattr(olmo_symm_mem.dist, "is_available", lambda: True)
    monkeypatch.setattr(olmo_symm_mem.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(olmo_symm_mem.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(olmo_symm_mem.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(olmo_symm_mem.dist, "all_gather_object", gather)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(state.visible_uuids))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(uuid=state.visible_uuids[index]),
    )
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", state.p2p)
    return state


def test_local_only_init_checks_uuid_topology_once_with_reordered_devices(local_only_bootstrap):
    state = local_only_bootstrap

    olmo_symm_mem.init(state.group, device="cuda:1")
    olmo_symm_mem.init(state.group, device="cuda:1")

    assert state.topology == [("host0", "GPU-A", {"GPU-A", "GPU-B"}, "")]
    state.p2p.assert_called_once_with(1, 0)
    state.ext.olmo_symm_get_unique_id.assert_called_once_with()
    state.ext.olmo_symm_init.assert_called_once_with([[1, 2], [3, 4]], 0, 2, 1)
    assert olmo_symm_mem._BOOTSTRAP_GLOBAL_RANKS == (8, 9)


@pytest.mark.parametrize("transport", [None, "", "ibrc", "ucx"])
def test_remote_transport_init_does_not_inspect_local_topology(
    local_only_bootstrap, monkeypatch, transport
):
    state = local_only_bootstrap
    if transport is None:
        monkeypatch.delenv("NVSHMEM_REMOTE_TRANSPORT")
    else:
        monkeypatch.setenv("NVSHMEM_REMOTE_TRANSPORT", transport)
    inspect_devices = Mock(side_effect=AssertionError("must not inspect local topology"))
    monkeypatch.setattr(torch.cuda, "device_count", inspect_devices)

    olmo_symm_mem.init(state.group, device="cuda:1")

    assert state.topology == []
    inspect_devices.assert_not_called()
    state.ext.olmo_symm_init.assert_called_once_with([[1, 2], [3, 4]], 0, 2, 1)


@pytest.mark.parametrize(
    ("peer", "message"),
    [
        (("host1", "GPU-B", {"GPU-A", "GPU-B"}, ""), "multiple hosts"),
        (("host0", "GPU-A", {"GPU-A", "GPU-B"}, ""), "same GPU UUID"),
        (("host0", "GPU-B", {"GPU-B"}, ""), "rank pairs.*1, 0"),
        (("host0", "GPU-C", {"GPU-A", "GPU-C"}, ""), "rank pairs.*0, 1"),
        (("", "", set(), "peer inspection failed"), "group rank 1: peer inspection failed"),
        (None, "missing peer topology"),
    ],
)
def test_local_only_init_rejects_peer_topology_before_extension(
    local_only_bootstrap, peer, message
):
    state = local_only_bootstrap
    state.peer = peer

    with pytest.raises(RuntimeError, match=message):
        olmo_symm_mem.init(state.group, device="cuda:1")

    assert len(state.topology) == 1
    state.load_extension.assert_not_called()
    assert olmo_symm_mem._BOOTSTRAP_GLOBAL_RANKS is None


@pytest.mark.parametrize("missing_device", [0, 1])
def test_local_only_init_gathers_missing_uuid_error(local_only_bootstrap, missing_device):
    state = local_only_bootstrap
    state.visible_uuids[missing_device] = None

    with pytest.raises(RuntimeError, match=f"group rank 0:.*device {missing_device}.*no GPU UUID"):
        olmo_symm_mem.init(state.group, device="cuda:1")

    assert "no GPU UUID" in state.topology[0][3]
    state.load_extension.assert_not_called()


@pytest.mark.parametrize(
    "failure", ["device_count", "get_device_properties", "can_device_access_peer"]
)
def test_local_only_init_gathers_cuda_inspection_errors(local_only_bootstrap, monkeypatch, failure):
    state = local_only_bootstrap
    monkeypatch.setattr(torch.cuda, failure, Mock(side_effect=RuntimeError("inspection failed")))

    with pytest.raises(RuntimeError, match="group rank 0: RuntimeError: inspection failed"):
        olmo_symm_mem.init(state.group, device="cuda:1")

    assert state.topology[0][3] == "RuntimeError: inspection failed"
    state.load_extension.assert_not_called()


def test_local_only_init_rejects_local_p2p_failure_before_extension(local_only_bootstrap):
    state = local_only_bootstrap
    state.p2p.return_value = False

    with pytest.raises(RuntimeError, match="rank pairs.*0, 1"):
        olmo_symm_mem.init(state.group, device="cuda:1")

    state.load_extension.assert_not_called()


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
