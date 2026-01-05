import pytest
import torch

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.parallel import (
    DataParallelType,
    build_world_mesh,
)
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.transformer.model import Transformer
from olmo_core.optim.muon import MuonConfig
from olmo_core.testing import DEVICES, requires_multi_gpu, run_distributed_test
from olmo_core.testing.utils import requires_dion
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.train.train_module.transformer.config import (
    TransformerDataParallelConfig,
)
from olmo_core.utils import get_default_device, seed_all


def build_transformer_model() -> Transformer:
    config = TransformerConfig.olmo2_30M(vocab_size=1024)
    model = config.build()
    return model


@requires_dion
def test_muon_config_to_optim():
    from dion import Muon  # type: ignore[reportMissingImports]

    config = MuonConfig()

    model = build_transformer_model()
    optim = config.build(model)

    assert isinstance(optim, Muon)
    assert len(optim.param_groups) == 4  # emb, matrix, vector, lm_head

    assert config.merge(["lr=1e-1"]).lr == 0.1


@requires_dion
@pytest.mark.parametrize("device", DEVICES)
def test_muon(device: torch.device, tmp_path):
    config = MuonConfig()

    model = build_transformer_model().train().to(device)
    optim = config.build(model)

    for group in optim.param_groups:
        assert "initial_lr" in group

    optim.zero_grad(set_to_none=True)
    model(torch.randint(0, 1024, (2, 8), device=device).int()).sum().backward()
    optim.step()

    # Save and then restore a checkpoint, and make sure fixed fields reset.
    # Store original initial_lr values before modifying them.
    # TODO: double check this test.
    original_initial_lrs = [group["initial_lr"] for group in optim.param_groups]
    for group in optim.param_groups:
        group["initial_lr"] = 1e-8
    save_model_and_optim_state(tmp_path, model, optim)
    load_model_and_optim_state(tmp_path, model, optim)
    for group, original_lr in zip(optim.param_groups, original_initial_lrs):
        assert group["initial_lr"] == original_lr


def _run_hsdp_muon(shard_degree: int, num_replicas: int):
    device = get_default_device()

    # HSDP Transformer
    dp_config = TransformerDataParallelConfig(
        name=DataParallelType.hsdp, shard_degree=shard_degree, num_replicas=num_replicas
    )
    world_mesh = build_world_mesh(dp=dp_config, device_type=device.type)
    config = TransformerConfig.olmo2_30M(vocab_size=1024)
    model = config.build(init_device=device.type)
    model.train()
    model = parallelize_model(model, world_mesh=world_mesh, device=device, dp_config=dp_config)

    # Create the Muon optimizer
    optim_config = MuonConfig()
    optim = optim_config.create_optimizer(model)

    # Fwd-bwd
    bs, seq_len = 2, 8
    input_ids = torch.randint(0, 1024, (bs, seq_len), device=device)
    logits = model(input_ids)
    logits.sum().backward()

    # Take optimizer step to test Muon with HSDP
    optim.step()


@requires_dion
@requires_multi_gpu
@pytest.mark.parametrize(
    "shard_degree,num_replicas",
    [
        pytest.param(2, 1, id="shard2_replica1"),
        pytest.param(1, 2, id="shard1_replica2"),
    ],
)
def test_hsdp_muon(shard_degree: int, num_replicas: int):
    seed_all(0)
    run_distributed_test(
        _run_hsdp_muon,
        backend="nccl",
        start_method="spawn",
        world_size=2,
        func_args=(shard_degree, num_replicas),
    )


def _run_fsdp_muon():
    device = get_default_device()

    # FSDP Transformer
    dp_config = TransformerDataParallelConfig(name=DataParallelType.fsdp)
    world_mesh = build_world_mesh(dp=dp_config, device_type=device.type)
    config = TransformerConfig.olmo2_30M(vocab_size=1024)
    model = config.build(init_device=device.type)
    model.train()
    model = parallelize_model(model, world_mesh=world_mesh, device=device, dp_config=dp_config)

    # Create the Muon optimizer
    optim_config = MuonConfig()
    optim = optim_config.create_optimizer(model)

    # Fwd-bwd
    bs, seq_len = 2, 8
    input_ids = torch.randint(0, 1024, (bs, seq_len), device=device)
    logits = model(input_ids)
    logits.sum().backward()

    # Take optimizer step to test Muon with FSDP
    optim.step()


@requires_dion
@requires_multi_gpu
def test_fsdp_muon():
    seed_all(0)
    run_distributed_test(
        _run_fsdp_muon,
        backend="nccl",
        start_method="spawn",
        world_size=2,
    )
