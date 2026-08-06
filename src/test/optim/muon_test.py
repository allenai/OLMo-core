import pytest
import torch
import torch.nn as nn

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.parallel import DataParallelType, build_world_mesh
from olmo_core.nn.attention import FusedAttention
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
    config = TransformerConfig.olmo2_30M(vocab_size=1024, n_layers=2)
    model = config.build()
    return model


def _parameter_options(config: MuonConfig, model: Transformer) -> dict[str, dict]:
    return {
        param_name: override.opts
        for override in config.default_group_overrides(model)
        for param_name in override.params
    }


def test_muon_splits_attention_projections_by_head():
    model = TransformerConfig.llama_like(
        d_model=64,
        vocab_size=128,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        hidden_size_multiple_of=8,
    ).build()

    parameter_options = _parameter_options(MuonConfig(), model)

    assert parameter_options["blocks.0.attention.w_q.weight"]["num_heads"] == 4
    assert parameter_options["blocks.0.attention.w_k.weight"]["num_heads"] == 2
    assert parameter_options["blocks.0.attention.w_v.weight"]["num_heads"] == 2
    assert "num_heads" not in parameter_options["blocks.0.attention.w_out.weight"]


def test_muon_splits_fused_qkv_projection_by_head():
    model = TransformerConfig.llama_like(
        d_model=64,
        vocab_size=128,
        n_layers=1,
        n_heads=4,
        hidden_size_multiple_of=8,
    ).build()

    # Construct a FusedAttention module without initializing its optional flash-attention backend;
    # parameter grouping only needs its projection weights and head count.
    attention = FusedAttention.__new__(FusedAttention)
    nn.Module.__init__(attention)
    attention.n_heads = 4
    attention.w_qkv = nn.Linear(64, 3 * 64, bias=False)
    attention.w_out = nn.Linear(64, 64, bias=False)
    model.blocks["0"].attention = attention

    parameter_options = _parameter_options(MuonConfig(), model)

    assert parameter_options["blocks.0.attention.w_qkv.weight"]["num_heads"] == 12
    assert "num_heads" not in parameter_options["blocks.0.attention.w_out.weight"]


@requires_dion
def test_muon_config_to_optim():
    from dion import Muon  # type: ignore[reportMissingImports]

    config = MuonConfig()

    model = build_transformer_model()
    optim = config.build(model)

    assert isinstance(optim, Muon)
    assert len(optim.param_groups) == 5  # matrix, attention, vector, embedding, lm_head

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

    # Test that initial_lr is a "fixed field" that gets reset on checkpoint load.
    # Corrupt initial_lr, save, then load—initial_lr should be restored to original, not loaded from checkpoint.
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
