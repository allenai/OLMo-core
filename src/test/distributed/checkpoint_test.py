from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from cached_path import cached_path
from torch.distributed._tensor import Shard, distribute_tensor, init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from olmo_core.distributed.checkpoint import (
    Checkpointer,
    OptimStateDict,
    OverlapType,
    SafeTensorsLoader,
    TensorShardSpec,
    _flatten_optimizer_state,
    _get_local_tensor_data,
    _get_model_state_dict_for_checkpoint,
    _offsets_overlap,
    _unflatten_optimizer_state,
    init_optimizer_state,
    load_model_and_optim_state,
    save_model_and_optim_state,
    unshard_model_state,
    unshard_optim_state,
)
from olmo_core.distributed.fsdp import FSDP
from olmo_core.distributed.tensors import (
    ShardedFlatParameter,
    ShardedFlatTensor,
    ShardingSpec,
)

from .utils import (
    BACKENDS,
    DEVICES,
    get_default_device,
    requires_multi_gpu,
    run_distributed_test,
)


def test_offsets_overlap():
    assert _offsets_overlap((0, 3), (1, 4))
    assert _offsets_overlap((0, 6), (0, 6))
    assert _offsets_overlap((0, 6), (0, 12))
    assert _offsets_overlap((1, 6), (0, 12))
    assert _offsets_overlap((0, 6), (2, 4))
    assert _offsets_overlap((0, 6), (5, 6))
    assert _offsets_overlap((0, 6), (0, 2))

    assert _offsets_overlap((1, 4), (0, 3))
    assert _offsets_overlap((0, 6), (0, 6))
    assert _offsets_overlap((0, 12), (0, 6))
    assert _offsets_overlap((0, 12), (1, 6))
    assert _offsets_overlap((2, 4), (0, 6))
    assert _offsets_overlap((5, 6), (0, 6))
    assert _offsets_overlap((0, 2), (0, 6))

    assert not _offsets_overlap((2, 5), (7, 9))


def test_tensor_shard_spec_get_merged_flattened_offsets():
    assert list(TensorShardSpec(flattened_offsets=((0, 3),)).get_merged_flattened_offsets((128, 256))) == [(0, 3)]

    assert list(TensorShardSpec(flattened_offsets=((0, 3), (3, 6))).get_merged_flattened_offsets((128, 256))) == [
        (0, 6)
    ]

    assert list(
        TensorShardSpec(flattened_offsets=((0, 3), (6, 9), (9, 12), (15, 18))).get_merged_flattened_offsets(
            (128, 256)
        )
    ) == [(0, 3), (6, 12), (15, 18)]


def test_tensor_shard_spec_compute_overlap_with_flattened_offsets():
    assert (
        TensorShardSpec(flattened_offsets=((0, 3), (3, 6))).compute_overlap_with(
            TensorShardSpec(flattened_offsets=((0, 6),)), (128, 256)
        )
        == OverlapType.EQUAL
    )

    assert (
        TensorShardSpec(flattened_offsets=((0, 3), (6, 12))).compute_overlap_with(
            TensorShardSpec(flattened_offsets=((0, 3), (6, 9))), (128, 256)
        )
        == OverlapType.SUPERSET
    )

    assert (
        TensorShardSpec(flattened_offsets=((0, 3), (6, 12))).compute_overlap_with(
            TensorShardSpec(flattened_offsets=((0, 15),)), (128, 256)
        )
        == OverlapType.SUBSET
    )

    assert (
        TensorShardSpec(flattened_offsets=((0, 3), (6, 12))).compute_overlap_with(
            TensorShardSpec(flattened_offsets=((2, 5),)), (128, 256)
        )
        == OverlapType.MIXED
    )

    assert (
        TensorShardSpec(flattened_offsets=((0, 3), (6, 12))).compute_overlap_with(
            TensorShardSpec(flattened_offsets=((12, 15),)), (128, 256)
        )
        is None
    )


def test_tensor_shard_spec_compute_overlap_with_dtensor_fields():
    assert (
        TensorShardSpec(local_shape=(2, 8), global_offset=(0, 0)).compute_overlap_with(
            TensorShardSpec(local_shape=(2, 8), global_offset=(0, 0)), (16, 8)
        )
        == OverlapType.EQUAL
    )

    assert (
        TensorShardSpec(local_shape=(4, 8), global_offset=(0, 0)).compute_overlap_with(
            TensorShardSpec(local_shape=(2, 8), global_offset=(1, 0)), (16, 8)
        )
        == OverlapType.SUPERSET
    )

    assert (
        TensorShardSpec(local_shape=(2, 8), global_offset=(1, 0)).compute_overlap_with(
            TensorShardSpec(local_shape=(4, 8), global_offset=(0, 0)), (16, 8)
        )
        == OverlapType.SUBSET
    )

    assert (
        TensorShardSpec(local_shape=(2, 8), global_offset=(0, 0)).compute_overlap_with(
            TensorShardSpec(local_shape=(4, 4), global_offset=(0, 0)), (16, 8)
        )
        == OverlapType.MIXED
    )

    assert (
        TensorShardSpec(local_shape=(2, 4), global_offset=(1, 2)).compute_overlap_with(
            TensorShardSpec(local_shape=(4, 8), global_offset=(0, 0)), (16, 8)
        )
        == OverlapType.SUBSET
    )

    assert (
        TensorShardSpec(local_shape=(2, 4), global_offset=(1, 2)).compute_overlap_with(
            TensorShardSpec(local_shape=(2, 4), global_offset=(0, 0)), (16, 8)
        )
        == OverlapType.MIXED
    )


def test_tensor_shard_spec_for_dtensor_1D():
    full_shape = (16,)
    shard_spec = TensorShardSpec(local_shape=(8,), global_offset=(0,))
    assert list(shard_spec.get_flattened_offsets(full_shape)) == [(0, 8)]


def test_tensor_shard_spec_for_dtensor_2D_colwise():
    # For example:
    #  from torch.distributed._tensor import Shard, distribute_tensor, init_device_mesh
    #  mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    #  distribute_tensor(torch.randn(16, 8), mesh, [Shard(dim=0)])
    full_shape = (16, 8)
    shard_spec = TensorShardSpec(local_shape=(4, 8), global_offset=(4, 0))
    assert list(shard_spec.get_flattened_offsets(full_shape)) == [(32, 40), (40, 48), (48, 56), (56, 64)]


def test_tensor_shard_spec_for_dtensor_2D_rowwise():
    # For example:
    #  from torch.distributed._tensor import Shard, distribute_tensor, init_device_mesh
    #  mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    #  distribute_tensor(torch.randn(16, 8), mesh, [Shard(dim=1)])
    full_shape = (16, 8)
    shard_spec = TensorShardSpec(local_shape=(16, 2), global_offset=(0, 2))
    assert list(shard_spec.get_flattened_offsets(full_shape)) == [
        (2, 4),  # row 0
        (10, 12),  # row 1
        (18, 20),  # row 2
        (26, 28),  # row 3
        (34, 36),  # row 4
        (42, 44),  # row 5
        (50, 52),  # row 6
        (58, 60),  # row 7
        (66, 68),  # row 8
        (74, 76),  # row 9
        (82, 84),  # row 10
        (90, 92),  # row 11
        (98, 100),  # row 12
        (106, 108),  # row 13
        (114, 116),  # row 14
        (122, 124),  # row 15
    ]


def save_and_load_checkpoint_with_regular_and_sharded_tensors(dir):
    checkpointer = Checkpointer()

    state_dict_to_save = {
        "x": torch.tensor([[1, 2, 3], [2, 2, 2]], device=get_default_device()),
        "y": ShardedFlatParameter.shard(torch.rand(2, 3, device=get_default_device())),
    }

    state_dict_to_load = {
        "x": torch.zeros_like(state_dict_to_save["x"]),
        "y": ShardedFlatParameter.shard(torch.zeros(2, 3, device=get_default_device())),
    }

    _, files = checkpointer.save(dir, state_dict_to_save)
    if dist.get_rank() == 0:
        assert len(files) == 2  # will include metadata
    else:
        assert len(files) == 1
    for path in files:
        assert isinstance(path, Path)
        assert path.exists()
    checkpointer.load(dir, state_dict_to_load)

    torch.testing.assert_close(state_dict_to_save["x"], state_dict_to_load["x"])
    torch.testing.assert_close(state_dict_to_save["y"], state_dict_to_load["y"])

    # Test loading unsharded checkpoint.
    full_state_dict = checkpointer.unshard(dir)
    assert full_state_dict["x"].shape == (2, 3)
    assert full_state_dict["y"].shape == (2, 3)

    # Now from rank 0 only.
    full_state_dict = checkpointer.unshard(dir, rank0_only=True)
    if dist.get_rank() == 0:
        assert full_state_dict["x"].shape == (2, 3)
        assert full_state_dict["y"].shape == (2, 3)
    else:
        assert len(full_state_dict) == 0

    # Now from rank 1 only.
    if dist.get_rank() == 1:
        full_state_dict = checkpointer.unshard(dir, no_dist=True)
        assert full_state_dict["x"].shape == (2, 3)
        assert full_state_dict["y"].shape == (2, 3)


def run_get_local_tensor_data_with_dtensor():
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    dtensor = distribute_tensor(torch.randn(16, device=get_default_device()), mesh, [Shard(dim=0)])

    # Make sure modifying the data returned from `_get_local_tensor_data` will modify the data
    # in the actual tensor.
    _get_local_tensor_data(dtensor).fill_(torch.nan)
    assert _get_local_tensor_data(dtensor).isnan().all()
    assert dtensor.full_tensor().isnan().all()


@requires_multi_gpu
def test_get_local_tensor_data_with_dtensor():
    run_distributed_test(run_get_local_tensor_data_with_dtensor, backend="nccl")


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_checkpoint_with_regular_and_sharded_tensors(backend, tmp_path):
    run_distributed_test(
        save_and_load_checkpoint_with_regular_and_sharded_tensors, backend=backend, func_args=(tmp_path,)
    )
    # We should be able to load unsharded checkpoint from a non-distributed context.
    full_state_dict = Checkpointer().unshard(tmp_path)
    assert full_state_dict["x"].shape == (2, 3)
    assert full_state_dict["y"].shape == (2, 3)


def save_and_load_checkpoint_with_dtensors(dir):
    checkpointer = Checkpointer()

    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    state_dict_to_save = {
        "1d": distribute_tensor(torch.randn(16, device=get_default_device()), mesh, [Shard(dim=0)]),
        "2d_colwise": distribute_tensor(torch.randn(16, 8, device=get_default_device()), mesh, [Shard(dim=0)]),
        "2d_rowwise": distribute_tensor(torch.randn(16, 8, device=get_default_device()), mesh, [Shard(dim=1)]),
    }

    state_dict_to_load = {
        "1d": distribute_tensor(torch.randn(16, device=get_default_device()), mesh, [Shard(dim=0)]),
        "2d_colwise": distribute_tensor(torch.randn(16, 8, device=get_default_device()), mesh, [Shard(dim=0)]),
        "2d_rowwise": distribute_tensor(torch.randn(16, 8, device=get_default_device()), mesh, [Shard(dim=1)]),
    }

    checkpointer.save(dir, state_dict_to_save)  # type: ignore[arg-type]
    checkpointer.load(dir, state_dict_to_load)  # type: ignore[arg-type]

    for key in state_dict_to_load:
        torch.testing.assert_close(state_dict_to_save[key], state_dict_to_load[key])


@requires_multi_gpu
def test_save_and_load_checkpoint_with_dtensors(tmp_path):
    run_distributed_test(save_and_load_checkpoint_with_dtensors, backend="nccl", func_args=(tmp_path,))


def save_and_load_checkpoint_with_different_dtensor_topology(dir):
    checkpointer = Checkpointer()

    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    og_tensor = torch.randn(8, 6, device=get_default_device())

    # Ensure tensor matches on all ranks (could use scatter here too, but whatever).
    dist.all_reduce(og_tensor)

    state_dict_to_save = {
        "x": distribute_tensor(og_tensor, mesh, [Shard(dim=0)]),
    }
    checkpointer.save(dir, state_dict_to_save)  # type: ignore[arg-type]

    state_dict_to_load = {
        "x": distribute_tensor(torch.randn(8, 6, device=get_default_device()), mesh, [Shard(dim=1)]),
    }
    checkpointer.load(dir, state_dict_to_load)  # type: ignore[arg-type]

    # Gather full tensor from the state dict to load and make sure it matches the full OG tensor.
    full_loaded_tensor = state_dict_to_load["x"].full_tensor()
    torch.testing.assert_close(og_tensor, full_loaded_tensor)


@requires_multi_gpu
def test_save_and_load_checkpoint_with_different_dtensor_topology(tmp_path):
    run_distributed_test(
        save_and_load_checkpoint_with_different_dtensor_topology, backend="nccl", func_args=(tmp_path,)
    )


def save_and_unshard_dtensor(dir):
    checkpointer = Checkpointer()

    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    og_tensor = torch.randn(8, 6, device=get_default_device())

    # Ensure tensor matches on all ranks (could use scatter here too, but whatever).
    dist.all_reduce(og_tensor)

    state_dict_to_save = {
        "x": distribute_tensor(og_tensor, mesh, [Shard(dim=0)]),
    }
    checkpointer.save(dir, state_dict_to_save)  # type: ignore[arg-type]

    full_state_dict = checkpointer.unshard(dir, device=get_default_device())
    torch.testing.assert_close(og_tensor, full_state_dict["x"])


@requires_multi_gpu
def test_save_and_unshard_dtensor(tmp_path):
    run_distributed_test(save_and_unshard_dtensor, backend="nccl", func_args=(tmp_path,))


def save_and_load_checkpoint_with_different_sharding_spec(dir):
    for idx, (offsets_to_save, offsets_to_load) in enumerate(
        [
            # save_tensor: |x x x|x x x
            # load_tensor: |x x|x x x x
            ((((0, 3),), ((3, 6),)), (((0, 2),), ((2, 6),))),
            # save_tensor: |x x x|x x x
            # load_tensor: |x x x x|x x
            ((((0, 3),), ((3, 6),)), (((0, 4),), ((4, 6),))),
            # save_tensor: |x x x x x x|
            # load_tensor: |x x x x|x x
            ((((0, 6),), ((6, 6),)), (((0, 4),), ((4, 6),))),
        ]
    ):
        checkpointer = Checkpointer()

        state_dict_to_save = {
            "x": ShardedFlatParameter.shard(
                torch.rand(2, 3, device=get_default_device()),
                ShardingSpec(unsharded_shape=(2, 3), unsharded_flattened_offsets=offsets_to_save),
            ),
        }

        state_dict_to_load = {
            "x": ShardedFlatParameter.shard(
                torch.rand(2, 3, device=get_default_device()),
                ShardingSpec(unsharded_shape=(2, 3), unsharded_flattened_offsets=offsets_to_load),
            ),
        }

        checkpointer.save(dir / f"checkpoint{idx}", state_dict_to_save)  # type: ignore
        checkpointer.load(dir / f"checkpoint{idx}", state_dict_to_load)  # type: ignore

        og_x_unsharded = state_dict_to_save["x"].gather()
        loaded_x_unsharded = state_dict_to_load["x"].gather()

        torch.testing.assert_close(og_x_unsharded, loaded_x_unsharded)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_checkpoint_with_different_sharding_spec(backend, tmp_path):
    run_distributed_test(
        save_and_load_checkpoint_with_different_sharding_spec, backend=backend, func_args=(tmp_path,)
    )


def save_and_load_checkpoint_from_regular_to_sharded_tensor(dir):
    checkpointer = Checkpointer()

    state_dict_to_save = {
        "x": torch.rand(2, 3, device=get_default_device()),
    }

    state_dict_to_load = {
        "x": ShardedFlatParameter.shard(torch.zeros(2, 3, device=get_default_device())),
    }

    checkpointer.save(dir, state_dict_to_save)  # type: ignore
    checkpointer.load(dir, state_dict_to_load)  # type: ignore

    gathered = state_dict_to_load["x"].gather()
    if dist.get_rank() == 0:
        torch.testing.assert_close(state_dict_to_save["x"], gathered)
    torch.testing.assert_close(gathered, checkpointer.unshard(dir, device=get_default_device())["x"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_checkpoint_from_regular_to_sharded_tensor(backend, tmp_path):
    run_distributed_test(
        save_and_load_checkpoint_from_regular_to_sharded_tensor, backend=backend, func_args=(tmp_path,)
    )


def save_and_load_checkpoint_from_sharded_to_regular_tensor(dir):
    checkpointer = Checkpointer()

    state_dict_to_save = {
        "x": ShardedFlatParameter.shard(torch.zeros(2, 3, device=get_default_device())),
    }

    state_dict_to_load = {
        "x": torch.rand(2, 3, device=get_default_device()),
    }

    checkpointer.save(dir, state_dict_to_save)  # type: ignore
    checkpointer.load(dir, state_dict_to_load)  # type: ignore

    torch.testing.assert_close(state_dict_to_save["x"].gather(), state_dict_to_load["x"])
    torch.testing.assert_close(state_dict_to_load, checkpointer.unshard(dir, device=get_default_device()))


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_checkpoint_from_sharded_to_regular_tensor(backend, tmp_path):
    run_distributed_test(
        save_and_load_checkpoint_from_sharded_to_regular_tensor, backend=backend, func_args=(tmp_path,)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_save_and_load_non_distributed(device, tmp_path):
    checkpointer = Checkpointer()

    state_dict_to_save = {
        "x": torch.tensor([[1, 2, 3], [2, 2, 2]], device=device),
    }

    state_dict_to_load = {
        "x": torch.zeros_like(state_dict_to_save["x"]),
    }

    checkpointer.save(tmp_path, state_dict_to_save)
    checkpointer.load(tmp_path, state_dict_to_load)

    torch.testing.assert_close(state_dict_to_save, state_dict_to_load)


@pytest.mark.parametrize("device", DEVICES)
def test_save_and_load_remote_non_distributed(device, s3_checkpoint_dir):
    checkpointer = Checkpointer()

    state_dict_to_save = {
        "x": torch.tensor([[1, 2, 3], [2, 2, 2]], device=device),
    }

    state_dict_to_load = {
        "x": torch.zeros_like(state_dict_to_save["x"]),
    }

    _, files = checkpointer.save(s3_checkpoint_dir, state_dict_to_save)
    assert len(files) == 2  # will include metadata
    for path in files:
        assert isinstance(path, str)
        assert path.startswith("s3://")
    checkpointer.load(s3_checkpoint_dir, state_dict_to_load)

    torch.testing.assert_close(state_dict_to_save, state_dict_to_load)


def save_and_load_remote_checkpoint(remote_dir):
    checkpointer = Checkpointer()

    state_dict_to_save = {
        "x": torch.tensor([[1, 2, 3], [2, 2, 2]], device=get_default_device()),
        "y": ShardedFlatParameter.shard(torch.rand(2, 3, device=get_default_device())),
    }

    state_dict_to_load = {
        "x": torch.zeros_like(state_dict_to_save["x"]),
        "y": ShardedFlatParameter.shard(torch.zeros(2, 3, device=get_default_device())),
    }

    checkpointer.save(remote_dir, state_dict_to_save)
    checkpointer.load(remote_dir, state_dict_to_load)

    torch.testing.assert_close(state_dict_to_save, state_dict_to_load)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_remote_checkpoint(backend, s3_checkpoint_dir):
    run_distributed_test(save_and_load_remote_checkpoint, backend=backend, func_args=(s3_checkpoint_dir,))


def run_save_and_load_with_different_data_across_ranks(dir):
    checkpointer = Checkpointer()

    state_dict_to_save: Dict[str, torch.Tensor] = {
        "x": ShardedFlatParameter.shard(torch.rand(2, 3, device=get_default_device()))
    }
    y_to_save = ShardedFlatParameter.shard(torch.rand(2, 3, device=get_default_device()))
    if dist.get_rank() == 1:
        state_dict_to_save["y"] = y_to_save
        state_dict_to_save["z"] = torch.rand(2, 3)

    state_dict_to_load: Dict[str, torch.Tensor] = {
        "x": ShardedFlatParameter.shard(torch.rand(2, 3, device=get_default_device()))
    }
    y_to_load = ShardedFlatParameter.shard(torch.rand(2, 3, device=get_default_device()))
    if dist.get_rank() == 0:
        state_dict_to_load["z"] = torch.rand(2, 3)
    else:
        state_dict_to_load["y"] = y_to_load

    checkpointer.save(dir, state_dict_to_save)
    checkpointer.load(dir, state_dict_to_load)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_with_different_data_across_ranks(backend, tmp_path):
    run_distributed_test(
        run_save_and_load_with_different_data_across_ranks, backend=backend, func_args=(tmp_path,)
    )


def run_save_and_load_with_sharded_tensors_in_process_group(dir):
    checkpointer = Checkpointer()

    pg1 = dist.new_group([0, 1])
    pg2 = dist.new_group([2, 3])

    state_dict_to_save: Dict[str, torch.Tensor] = {"x": ShardedFlatTensor.shard(torch.rand(2, 3))}
    if dist.get_rank(pg1) >= 0:
        state_dict_to_save["y"] = ShardedFlatTensor.shard(torch.rand(2, 3), process_group=pg1)
    if dist.get_rank(pg2) >= 0:
        state_dict_to_save["y"] = ShardedFlatTensor.shard(torch.rand(2, 3), process_group=pg2)

    state_dict_to_load: Dict[str, torch.Tensor] = {
        "x": ShardedFlatTensor.shard(torch.rand(2, 3)),
        "y": ShardedFlatTensor.shard(torch.rand(2, 3)),
    }

    checkpointer.save(dir, state_dict_to_save)
    checkpointer.load(dir, state_dict_to_load)

    loaded_y = state_dict_to_load["y"].gather()  # type: ignore
    if dist.get_rank(pg1) >= 0:
        saved_y = state_dict_to_save["y"].gather()  # type: ignore
        torch.testing.assert_close(loaded_y, saved_y)


def test_save_and_load_with_sharded_tensors_in_process_group(tmp_path):
    run_distributed_test(
        run_save_and_load_with_sharded_tensors_in_process_group,
        backend="gloo",
        func_args=(tmp_path,),
        world_size=4,
    )


def test_safe_tensors_loader():
    url = "https://huggingface.co/stas/tiny-random-llama-2/resolve/main/model.safetensors"
    key = "model.layers.0.post_attention_layernorm.weight"
    path = cached_path(url)

    for start_idx, end_idx in [(0, None), (7, 13), (13, None)]:
        with SafeTensorsLoader(url) as loader:
            tensor_from_url = loader.get_flat_slice(key, start_idx, end_idx)

        with SafeTensorsLoader(path) as loader:
            tensor_from_path = loader.get_flat_slice(key, start_idx, end_idx)

        try:
            torch.testing.assert_close(tensor_from_path, tensor_from_url)
        except AssertionError:
            print(f"start_idx={start_idx}, end_idx={end_idx}")
            raise


def assert_optim_state_close(optim_state1: OptimStateDict, optim_state2: OptimStateDict):
    assert optim_state1.keys() == optim_state2.keys()

    # Validate param groups.
    assert len(optim_state1["param_groups"]) == len(optim_state2["param_groups"])
    for i in range(len(optim_state2["param_groups"])):
        assert optim_state1["param_groups"][i] == optim_state2["param_groups"][i]

    # Validate state tensors.
    assert optim_state1["state"].keys() == optim_state2["state"].keys()
    for param_id in optim_state2["state"].keys():
        assert optim_state1["state"][param_id].keys() == optim_state2["state"][param_id].keys()
        for key in optim_state2["state"][param_id].keys():
            torch.testing.assert_close(optim_state1["state"][param_id][key], optim_state2["state"][param_id][key])


def test_flatten_optimizer_state(tiny_model, tiny_model_data):
    # Do a step to ensure optimizer state is initialized.
    optim = torch.optim.AdamW(tiny_model.parameters())
    tiny_model(tiny_model_data).sum().backward()
    optim.step()

    flat_optim_state = _flatten_optimizer_state(
        tiny_model,
        optim,
        _get_model_state_dict_for_checkpoint(tiny_model),
        optim.state_dict(),  # type: ignore[arg-type]
    )
    unflattened_optim_state = _unflatten_optimizer_state(flat_optim_state)

    # Make sure unflattened state matches what we'd get from `optim.state_dict()`.
    assert_optim_state_close(optim.state_dict(), unflattened_optim_state)  # type: ignore

    # Lastly, make sure we can load it.
    optim.load_state_dict(unflattened_optim_state)  # type: ignore


def flatten_optimizer_state_with_sharded_flat_params(model_factory, model_data_factory):
    model = model_factory().to(get_default_device())
    model_data = model_data_factory().to(get_default_device())

    # Do a step to ensure optimizer state is initialized.
    optim = torch.optim.AdamW(model.parameters())
    model(model_data).sum().backward()
    optim.step()

    # Now shard part of the model and the corresponding optimizer state.
    og_param = model.fc[0].weight
    flat_param = ShardedFlatParameter.shard(og_param)
    optim.state[flat_param] = {
        k: v if k == "step" else ShardedFlatParameter.shard(v, requires_grad=False).data
        for k, v in optim.state.pop(og_param).items()
    }
    param_id = optim.param_groups[0]["params"].index(og_param)
    optim.param_groups[0]["params"][param_id] = flat_param
    setattr(model.fc[0], "weight", flat_param)

    model_state = _get_model_state_dict_for_checkpoint(model)
    assert model_state["fc.0.weight"].shape == flat_param.shape

    flat_optim_state = _flatten_optimizer_state(
        model,
        optim,
        model_state,
        optim.state_dict(),  # type: ignore[arg-type]
    )
    unflattened_optim_state = _unflatten_optimizer_state(flat_optim_state)

    # Make sure unflattened state matches what we'd get from `optim.state_dict()`.
    assert_optim_state_close(optim.state_dict(), unflattened_optim_state)  # type: ignore

    # Lastly, make sure we can load it.
    optim.load_state_dict(unflattened_optim_state)  # type: ignore


@pytest.mark.parametrize("backend", BACKENDS)
def test_flatten_optimizer_state_with_sharded_flat_params(backend, tiny_model_factory, tiny_model_data_factory):
    run_distributed_test(
        flatten_optimizer_state_with_sharded_flat_params,
        backend=backend,
        start_method="spawn",
        func_args=(tiny_model_factory, tiny_model_data_factory),
    )


def run_save_and_load_fsdp_model(dir, model_factory, model_data_factory, pre_init_optim_state_to_load):
    fsdp_model = FSDP(model_factory())
    optim = torch.optim.AdamW(fsdp_model.parameters())

    # Take a train step to initialize optimizer state.
    fsdp_model(model_data_factory().to(fsdp_model.device)).sum().backward()
    optim.step()

    # Save checkpoint.
    save_model_and_optim_state(dir, fsdp_model, optim)
    dist.barrier()

    # Now create a new fsdp model and load that state.
    fsdp_model2 = FSDP(model_factory())
    optim2 = torch.optim.AdamW(fsdp_model2.parameters())
    if pre_init_optim_state_to_load:
        init_optimizer_state(optim2)
    load_model_and_optim_state(dir, fsdp_model2, optim2)

    # Check model parameters.
    with fsdp_model.summon_full_params(recurse=True), fsdp_model2.summon_full_params(recurse=True):
        torch.testing.assert_close(fsdp_model.state_dict(), fsdp_model2.state_dict())

    # Check optimizer state.
    for p1, p2 in zip(fsdp_model.parameters(), fsdp_model2.parameters()):
        if p1.numel() > 0:
            torch.testing.assert_close(optim.state[p1], optim2.state[p2])
        else:
            for key in ("exp_avg", "exp_avg_sq"):
                assert key not in optim.state or optim.state[p1][key].numel() == 0
                assert key not in optim2.state or optim2.state[p2][key].numel() == 0

    # Check unsharding model state.
    full_model_state = unshard_model_state(dir)
    assert full_model_state
    for name, param in fsdp_model.named_parameters():
        assert isinstance(param, ShardedFlatParameter)
        assert name in full_model_state
        assert full_model_state[name].shape == param.unsharded_shape

    # Check unsharding optim state.
    full_optim_state = unshard_optim_state(dir)
    assert full_optim_state
    assert len(full_optim_state["param_groups"]) == len(optim.param_groups)
    for i, param in enumerate(fsdp_model.parameters()):
        assert isinstance(param, ShardedFlatParameter)
        assert i in full_optim_state["state"]
        state = full_optim_state["state"][i]
        assert state["step"].numel() == 1
        assert state["exp_avg"].shape == param.unsharded_shape
        assert state["exp_avg_sq"].shape == param.unsharded_shape


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "pre_init_optim_state_to_load",
    (pytest.param(True, id="initialized_optim"), pytest.param(False, id="uninitialized_optim")),
)
def test_save_and_load_fsdp_model(
    backend, tmp_path, tiny_model_factory, tiny_model_data_factory, pre_init_optim_state_to_load
):
    run_distributed_test(
        run_save_and_load_fsdp_model,
        backend=backend,
        start_method="spawn",
        func_args=(tmp_path, tiny_model_factory, tiny_model_data_factory, pre_init_optim_state_to_load),
    )


def run_save_and_load_torch_fsdp_model(
    dir, model_factory, model_data_factory, pre_init_optim_state_to_load, use_orig_params
):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    fsdp_model = FSDP(model_factory().cuda(), use_orig_params=use_orig_params)
    optim = torch.optim.AdamW(fsdp_model.parameters())

    # Take a train step to initialize optimizer state.
    fsdp_model(model_data_factory().cuda()).sum().backward()
    optim.step()

    # Save checkpoint.
    files = save_model_and_optim_state(dir, fsdp_model, optim)
    if dist.get_rank() == 0:
        assert len(files) == 4
    else:
        assert len(files) == 2
    dist.barrier()

    # Now create a new fsdp model and load that state.
    fsdp_model2 = FSDP(model_factory().cuda(), use_orig_params=use_orig_params)
    optim2 = torch.optim.AdamW(fsdp_model2.parameters())
    if pre_init_optim_state_to_load:
        init_optimizer_state(optim2)
    load_model_and_optim_state(dir, fsdp_model2, optim2)

    # Check unsharding state.
    full_model_state = unshard_model_state(dir, device=torch.device("cuda"))

    # Check model parameters.
    with FSDP.summon_full_params(fsdp_model, recurse=True), FSDP.summon_full_params(fsdp_model2, recurse=True):
        torch.testing.assert_close(fsdp_model.state_dict(), fsdp_model2.state_dict())
        torch.testing.assert_close(
            full_model_state,
            {k.replace("_fsdp_wrapped_module.", ""): v for k, v in fsdp_model.state_dict().items()},
        )

    # Check optimizer state.
    for (p1_name, p1), (p2_name, p2) in zip(fsdp_model.named_parameters(), fsdp_model2.named_parameters()):
        assert p1_name == p2_name
        torch.testing.assert_close(
            optim.state[p1], optim2.state[p2], msg=lambda m: f"State for '{p1_name}' does not match. {m}"
        )


@requires_multi_gpu
@pytest.mark.parametrize(
    "pre_init_optim_state_to_load",
    (pytest.param(True, id="initialized_optim"), pytest.param(False, id="uninitialized_optim")),
)
@pytest.mark.parametrize(
    "use_orig_params",
    (
        pytest.param(True, id="use_orig_params=True"),
        pytest.param(False, id="use_orig_param=False", marks=pytest.mark.skip(reason="Not implemented")),
    ),
)
def test_save_and_load_torch_fsdp_model(
    tmp_path, tiny_model_factory, tiny_model_data_factory, pre_init_optim_state_to_load, use_orig_params
):
    run_distributed_test(
        run_save_and_load_torch_fsdp_model,
        backend="nccl",
        start_method="spawn",
        func_args=(
            tmp_path,
            tiny_model_factory,
            tiny_model_data_factory,
            pre_init_optim_state_to_load,
            use_orig_params,
        ),
    )


def run_save_and_load_tensor_parallel_model(dir, take_step_before_checkpoint):
    tp_mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    class FeedForward(nn.Module):
        def __init__(self, dim: int = 16):
            super().__init__()
            self.dim = dim
            self.w1 = nn.Linear(dim, dim)
            self.w2 = nn.Linear(dim, dim)
            self.w3 = nn.Linear(dim, dim)

        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    feed_forward = FeedForward().cuda()
    parallelize_module(
        feed_forward,
        tp_mesh,
        {
            # by default ColwiseParallel input layouts is replicated
            # and RowwiseParallel output layouts is replicated
            "w1": ColwiseParallel(),
            "w2": RowwiseParallel(),
            "w3": ColwiseParallel(),
        },
    )
    optim = torch.optim.AdamW(feed_forward.parameters())

    # Take a forward and backward pass.
    feed_forward(torch.rand((2, feed_forward.dim), device="cuda")).sum().backward()

    if take_step_before_checkpoint:
        # Take an optimizer step.
        optim.step()

    # Save checkpoint.
    save_model_and_optim_state(dir, feed_forward, optim)

    # Now load the checkpoint with a different topology, in this case an unsharded model.
    unsharded_feed_forward = FeedForward().cuda()
    unsharded_optim = torch.optim.AdamW(unsharded_feed_forward.parameters())
    load_model_and_optim_state(dir, unsharded_feed_forward, unsharded_optim)


@requires_multi_gpu
@pytest.mark.parametrize(
    "take_step_before_checkpoint", [pytest.param(True, id="after-step"), pytest.param(False, id="pre-step")]
)
def test_save_and_load_tensor_parallel_model(tmp_path, take_step_before_checkpoint):
    run_distributed_test(
        run_save_and_load_tensor_parallel_model,
        backend="nccl",
        start_method="spawn",
        func_args=(tmp_path, take_step_before_checkpoint),
    )
