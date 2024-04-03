from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.distributed as dist
from cached_path import cached_path

from olmo_core.distributed.checkpoint import (
    Checkpointer,
    OptimStateDict,
    SafeTensorsLoader,
    _flatten_optimizer_state,
    _get_model_state_dict_for_checkpoint,
    _unflatten_optimizer_state,
    init_optimizer_state,
    load_model_and_optim_state,
    save_model_and_optim_state,
    unshard_model_state,
    unshard_optim_state,
)
from olmo_core.distributed.fsdp import FSDP
from olmo_core.distributed.sharded_flat_parameter import ShardedFlatParameter
from olmo_core.distributed.sharded_flat_tensor import ShardedFlatTensor, ShardingSpec

from .utils import (
    BACKENDS,
    DEVICES,
    get_default_device,
    requires_multi_gpu,
    run_distributed_test,
)


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


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_checkpoint_with_regular_and_sharded_tensors(backend, tmp_path):
    run_distributed_test(
        save_and_load_checkpoint_with_regular_and_sharded_tensors, backend=backend, func_args=(tmp_path,)
    )
    # We should be able to load unsharded checkpoint from a non-distributed context.
    full_state_dict = Checkpointer().unshard(tmp_path)
    assert full_state_dict["x"].shape == (2, 3)
    assert full_state_dict["y"].shape == (2, 3)


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
        torch.testing.assert_close(optim.state[p1], optim2.state[p2])

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

    # Check model parameters.
    with FSDP.summon_full_params(fsdp_model, recurse=True), FSDP.summon_full_params(fsdp_model2, recurse=True):
        torch.testing.assert_close(fsdp_model.state_dict(), fsdp_model2.state_dict())

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
