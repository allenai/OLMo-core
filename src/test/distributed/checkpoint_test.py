import pytest
import torch
from cached_path import cached_path

from olmo_core.distributed.checkpoint import (
    Checkpointer,
    SafeTensorsLoader,
    flatten_optimizer_state,
    unflatten_optimizer_state,
)
from olmo_core.distributed.sharded_flat_parameter import (
    ShardedFlatParameter,
    ShardingSpec,
)

from .utils import BACKENDS, DEVICES, get_default_device, run_distributed_test


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

    checkpointer.save(dir, state_dict_to_save)
    checkpointer.load(dir, state_dict_to_load)

    torch.testing.assert_close(state_dict_to_save, state_dict_to_load)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_checkpoint_with_regular_and_sharded_tensors(backend, tmp_path):
    run_distributed_test(
        save_and_load_checkpoint_with_regular_and_sharded_tensors, backend=backend, func_args=(tmp_path,)
    )


def save_and_load_checkpoint_with_different_sharding_spec(dir):
    for idx, (offsets_to_save, offsets_to_load) in enumerate(
        [
            # save_tensor: |x x x|x x x
            # load_tensor: |x x|x x x x
            (((0, 3), (3, 6)), ((0, 2), (2, 6))),
            # save_tensor: |x x x|x x x
            # load_tensor: |x x x x|x x
            (((0, 3), (3, 6)), ((0, 4), (4, 6))),
            # save_tensor: |x x x x x x|
            # load_tensor: |x x x x|x x
            (((0, 6), (6, 6)), ((0, 4), (4, 6))),
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

    torch.testing.assert_close(state_dict_to_save["x"], state_dict_to_load["x"].gather())
    torch.testing.assert_close(state_dict_to_save, checkpointer.unshard(dir))


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
    torch.testing.assert_close(state_dict_to_load, checkpointer.unshard(dir))


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

    checkpointer.save(s3_checkpoint_dir, state_dict_to_save)
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


def test_flatten_optimizer_state(tiny_model, tiny_model_data):
    # Do a step to ensure optimizer state is initialized.
    optim = torch.optim.AdamW(tiny_model.parameters())
    tiny_model(tiny_model_data).sum().backward()
    optim.step()

    flat_optim_state = flatten_optimizer_state(tiny_model, optim)
    unflattened_optim_state = unflatten_optimizer_state(flat_optim_state)

    # Make sure unflattened state matches what we'd get from `optim.state_dict()`.
    optim_state = optim.state_dict()
    assert unflattened_optim_state.keys() == optim_state.keys()
    # Validate param groups.
    assert len(unflattened_optim_state["param_groups"]) == len(optim_state["param_groups"])
    for i in range(len(optim_state["param_groups"])):
        assert unflattened_optim_state["param_groups"][i] == optim_state["param_groups"][i]
    # Validate state tensors.
    assert unflattened_optim_state["state"].keys() == optim_state["state"].keys()
    for param_id in optim_state["state"].keys():
        assert unflattened_optim_state["state"][param_id].keys() == optim_state["state"][param_id].keys()
        for key in optim_state["state"][param_id].keys():
            torch.testing.assert_close(
                unflattened_optim_state["state"][param_id][key], optim_state["state"][param_id][key]
            )

    # Lastly, make sure we can load it.
    optim.load_state_dict(unflattened_optim_state)  # type: ignore
