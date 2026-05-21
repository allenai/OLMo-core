from test.nn.attention.attention_test import BF16_ATOL, BF16_RTOL
from importlib import import_module
from typing import Any, Dict

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from olmo_core.config import Config
from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.utils import get_full_tensor, get_rank, get_world_size
from olmo_core.nn.attention import AttentionConfig, GatedDeltaNetConfig
from olmo_core.nn.attention.recurrent import GatedDeltaNet
from olmo_core.nn.attention.ring import UlyssesContextParallelStyle
from olmo_core.testing import requires_gpu, run_distributed_test
from olmo_core.testing.utils import requires_fla, requires_multi_gpu
from olmo_core.utils import get_default_device, seed_all


def test_gated_delta_net_config_decodes_legacy_fla_module_path():
    data = GatedDeltaNetConfig(n_heads=8, n_v_heads=16).as_config_dict()
    data[Config.CLASS_NAME_FIELD] = "olmo_core.nn.fla.GatedDeltaNetConfig"

    config = GatedDeltaNetConfig.from_dict(data)

    assert config == GatedDeltaNetConfig(n_heads=8, n_v_heads=16)


def test_gated_delta_net_config_decodes_legacy_fla_layer_module_path():
    legacy_module = import_module("olmo_core.nn.fla.layer")
    assert legacy_module.GatedDeltaNetConfig is GatedDeltaNetConfig

    data = GatedDeltaNetConfig(n_heads=8, n_v_heads=16).as_config_dict()
    data[Config.CLASS_NAME_FIELD] = "olmo_core.nn.fla.layer.GatedDeltaNetConfig"

    config = GatedDeltaNetConfig.from_dict(data)

    assert config == GatedDeltaNetConfig(n_heads=8, n_v_heads=16)


def test_gated_delta_net_config_decodes_legacy_fla_config():
    data = {
        Config.CLASS_NAME_FIELD: "olmo_core.nn.fla.layer.FLAConfig",
        "name": "GatedDeltaNet",
        "n_heads": 8,
        "fla_layer_kwargs": {
            "head_dim": 64,
            "num_v_heads": 16,
            "allow_neg_eigval": True,
        },
    }

    config = GatedDeltaNetConfig.from_dict(data)

    assert isinstance(config, GatedDeltaNetConfig)
    assert config.n_heads == 8
    assert config.n_v_heads == 16
    assert config.head_dim == 64
    assert config.allow_neg_eigval is True
    assert config.fla_layer_kwargs == {}


def test_gated_delta_net_config_decodes_legacy_n_kv_heads():
    data = GatedDeltaNetConfig(n_heads=8).as_config_dict()
    data[Config.CLASS_NAME_FIELD] = "olmo_core.nn.fla.GatedDeltaNetConfig"
    data["n_kv_heads"] = 16

    config = GatedDeltaNetConfig.from_dict(data)

    assert config.n_v_heads == 16
    assert "n_kv_heads" not in config.as_config_dict()


@requires_fla
@pytest.mark.parametrize(
    "recurrent_config",
    [
        pytest.param(GatedDeltaNetConfig(n_heads=8), id="default"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, n_v_heads=16), id="GVA"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, head_dim=32), id="head_dim=32"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, expand_v=1.0), id="expand_v=1.0"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, conv_size=8, conv_bias=True), id="conv_bias"),
        pytest.param(
            GatedDeltaNetConfig(n_heads=8, allow_neg_eigval=False), id="allow_neg_eigval=False"
        ),
    ],
)
def test_gated_delta_net_config_num_params(recurrent_config: GatedDeltaNetConfig):
    d_model = 512
    module = recurrent_config.build(d_model, layer_idx=0, n_layers=12, init_device="meta")

    # Make sure the estimated number of params matches the actual number of params.
    n_params = sum(p.numel() for p in module.parameters())
    assert recurrent_config.num_params(d_model) == n_params


@requires_fla
@requires_gpu
def test_gated_delta_net_fwd_bwd():
    device = "cuda"
    dtype = torch.bfloat16

    d_model, seq_len, batch_size = 256, 32, 2

    config = GatedDeltaNetConfig(n_heads=8)
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device=device)

    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

    with torch.autocast(device_type=device, dtype=dtype):
        y = module(x)
        assert y.shape == x.shape

        loss = y.sum()
        loss.backward()
    assert x.grad is not None


@requires_fla
def test_gated_delta_net_num_flops_per_token():
    d_model, n_heads, seq_len = 256, 2, 8192

    gdn = GatedDeltaNetConfig(n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1, init_device="meta"
    )
    attn = AttentionConfig(n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1, init_device="meta"
    )

    # At long sequence lengths, recurrent layers use fewer FLOPs than quadratic attention.
    gdn_flops = gdn.num_flops_per_token(seq_len)
    attn_flops = attn.num_flops_per_token(seq_len)  # type: ignore
    assert 0 < gdn_flops < attn_flops


def _run_context_parallel_gdn_ulysses(
    checkpoint_dir: str,
    inputs_path: str,
    outputs_path: str,
    gdn_kwargs: Dict[str, Any],
):
    device = get_default_device()
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("cp",))

    gdn = GatedDeltaNet(init_device=device.type, **gdn_kwargs)
    gdn.apply_cp(mesh["cp"], uly=UlyssesContextParallelStyle())
    load_model_and_optim_state(checkpoint_dir, gdn)

    # Load the input and split it across ranks on the sequence dimension.
    x = torch.load(inputs_path, map_location=device)
    rank, world_size = get_rank(), get_world_size()
    chunk_size = x.size(1) // world_size
    x_local = x[:, rank * chunk_size : (rank + 1) * chunk_size, :]

    with torch.autocast(device.type, dtype=x_local.dtype):
        local_y = gdn(x_local)
    y = DTensor.from_local(local_y, mesh, (Shard(1),))

    og_y = torch.load(outputs_path, map_location=device)
    tol_scale = 2  # requires slightly more tolerance than default
    torch.testing.assert_close(
        og_y, get_full_tensor(y), rtol=BF16_RTOL * tol_scale, atol=BF16_ATOL * tol_scale
    )


@requires_multi_gpu
@requires_fla
def test_context_parallel_gdn_ulysses(tmp_path):
    seed_all(0)
    device = get_default_device()

    # n_heads must be divisible by CP degree (world_size=2).
    gdn_kwargs: Dict[str, Any] = {"d_model": 128, "n_heads": 8}
    gdn = GatedDeltaNet(init_device=device.type, **gdn_kwargs)

    bs, seq_len = 2, 64
    x = torch.randn(bs, seq_len, gdn_kwargs["d_model"], device=device, dtype=torch.bfloat16)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        y = gdn(x)

    outputs_path = tmp_path / "gdn_y.pt"
    torch.save(y, outputs_path)
    inputs_path = tmp_path / "gdn_x.pt"
    torch.save(x, inputs_path)
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, gdn)

    run_distributed_test(
        _run_context_parallel_gdn_ulysses,
        backend="nccl",
        start_method="spawn",
        func_args=(checkpoint_dir, inputs_path, outputs_path, gdn_kwargs),
    )
