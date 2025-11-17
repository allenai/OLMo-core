import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as distcp
from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

from olmo_core.distributed.checkpoint.filesystem import (
    RemoteFileSystemReader,
    RemoteFileSystemWriter,
)
from olmo_core.io import dir_is_empty
from olmo_core.testing import BACKENDS, run_distributed_test
from olmo_core.utils import get_default_device


def run_save_and_load_with_dtensors(dir, throttle: bool = False):
    mesh = init_device_mesh(get_default_device().type, (dist.get_world_size(),))

    x_full = torch.randn(4, 4, device=get_default_device())
    y_full = torch.randn(4, 8, device=get_default_device())
    # Make sure these tensors are the same across all ranks. We could scatt
    dist.broadcast(x_full, 0)
    dist.broadcast(y_full, 0)

    # Shard the tensors.
    x = distribute_tensor(x_full, mesh, [Shard(dim=0)])
    y = distribute_tensor(y_full, mesh, [Shard(dim=0)])

    # Save the sharded tensors.
    distcp.state_dict_saver.save(
        {"x": x, "y": y},
        checkpoint_id=dir,
        storage_writer=RemoteFileSystemWriter(dir, thread_count=2, throttle_uploads=throttle),
    )

    # Now create new sharded copies with a different sharding strategy and load the checkpoint.
    x_loaded = distribute_tensor(torch.zeros_like(x_full), mesh, [Shard(dim=1)])
    y_loaded = distribute_tensor(torch.zeros_like(y_full), mesh, [Shard(dim=1)])
    distcp.state_dict_loader.load(
        {"x": x_loaded, "y": y_loaded},
        checkpoint_id=dir,
        storage_reader=RemoteFileSystemReader(dir, thread_count=2),
    )

    # Make sure the loaded tensors match the original tensors.
    x_full_loaded = x_loaded.full_tensor()
    y_full_loaded = y_loaded.full_tensor()
    torch.testing.assert_close(x_full, x_full_loaded)
    torch.testing.assert_close(y_full, y_full_loaded)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_locally_with_dtensors(backend, tmp_path):
    run_distributed_test(
        run_save_and_load_with_dtensors,
        backend=backend,
        func_args=(tmp_path,),
        start_method="spawn",
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("throttle", [True, False])
def test_save_and_load_remotely_to_s3_with_dtensors(backend, s3_checkpoint_dir, throttle):
    from botocore.exceptions import NoCredentialsError

    try:
        dir_is_empty(s3_checkpoint_dir)
    except NoCredentialsError:
        pytest.skip("Requires AWS credentials")

    run_distributed_test(
        run_save_and_load_with_dtensors,
        backend=backend,
        func_args=(s3_checkpoint_dir, throttle),
        start_method="spawn",  # NOTE: forking causes a crash with boto3
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("throttle", [True, False])
def test_save_and_load_remotely_to_gcs_with_dtensors(backend, gcs_checkpoint_dir, throttle):
    from google.auth.exceptions import DefaultCredentialsError

    try:
        dir_is_empty(gcs_checkpoint_dir)
    except DefaultCredentialsError:
        pytest.skip("Requires authentication with Google Cloud")

    run_distributed_test(
        run_save_and_load_with_dtensors,
        backend=backend,
        func_args=(gcs_checkpoint_dir, throttle),
        start_method="spawn",  # NOTE: forking causes a crash with boto3
    )
