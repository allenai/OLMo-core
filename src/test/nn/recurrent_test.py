from typing import Any, Dict, Optional

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh

from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_model_and_optim_state
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.nn.attention import AttentionConfig, GatedDeltaNetConfig
from olmo_core.nn.attention.recurrent import GatedDeltaNet
from olmo_core.nn.attention.ring import UlyssesContextParallelStyle
from olmo_core.testing import run_distributed_test
from olmo_core.testing.utils import requires_fla, requires_multi_gpu
from olmo_core.utils import get_default_device, seed_all

BF16_RTOL = 1e-5
BF16_ATOL = 5e-3


@requires_fla
@pytest.mark.parametrize(
    "recurrent_config",
    [
        pytest.param(GatedDeltaNetConfig(n_heads=8), id="MHA"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, n_kv_heads=4), id="GQA-4"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, n_kv_heads=2), id="GQA-2"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, n_kv_heads=1), id="MQA"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, conv_size=8), id="conv_size=8"),
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
@pytest.mark.parametrize("n_kv_heads", [pytest.param(4, id="GQA-4")])
def test_gated_delta_net_fwd_bwd(n_kv_heads: Optional[int]):
    seed_all(0)

    d_model = 512
    seq_len = 32

    config = GatedDeltaNetConfig(n_heads=8, n_kv_heads=n_kv_heads)
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device="cuda")

    x1 = torch.randn(1, seq_len, d_model, dtype=torch.bfloat16, device="cuda")
    x2 = torch.randn(1, seq_len, d_model, dtype=torch.bfloat16, device="cuda")
    x = torch.cat([x1, x2])

    # Make sure batch outputs match individual outputs.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y1 = module(x1)
        y2 = module(x2)
        y = module(x)

    torch.testing.assert_close(y[0:1, :, :], y1)
    torch.testing.assert_close(y[1:, :, :], y2)

    # Test backward pass.
    x = torch.randn(2, seq_len, d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = module(x)
        loss = y.sum()

    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


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


def _run_context_parallel_gated_delta_net(
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
        y_local = gdn(x_local)

    # Gather outputs from all ranks.
    y_gathered = [torch.empty_like(y_local) for _ in range(world_size)]
    torch.distributed.all_gather(y_gathered, y_local)
    y_cp = torch.cat(y_gathered, dim=1)

    # Compare with non-CP output.
    y = torch.load(outputs_path, map_location=device)
    torch.testing.assert_close(y_cp, y, rtol=BF16_RTOL, atol=BF16_ATOL)


@requires_multi_gpu
@requires_fla
def test_context_parallel_gated_delta_net(tmp_path):
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
        _run_context_parallel_gated_delta_net,
        backend="nccl",
        start_method="spawn",
        func_args=(checkpoint_dir, inputs_path, outputs_path, gdn_kwargs),
    )
