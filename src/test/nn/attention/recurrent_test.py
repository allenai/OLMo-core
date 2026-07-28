from test.nn.attention.attention_test import BF16_ATOL, BF16_RTOL
from typing import Any, Dict

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.utils import get_full_tensor, get_rank, get_world_size
from olmo_core.nn.attention import (
    AttentionConfig,
    GatedDeltaNet2Config,
    GatedDeltaNetConfig,
)
from olmo_core.nn.attention.recurrent import GatedDeltaNet, GatedDeltaNet2
from olmo_core.nn.attention.ring import UlyssesContextParallelStyle
from olmo_core.testing import requires_gpu, run_distributed_test
from olmo_core.testing.utils import requires_fla, requires_multi_gpu
from olmo_core.utils import get_default_device, seed_all


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


@requires_fla
@pytest.mark.parametrize(
    "recurrent_config",
    [
        pytest.param(GatedDeltaNet2Config(n_heads=8), id="default"),
        pytest.param(GatedDeltaNet2Config(n_heads=8, n_v_heads=16), id="GVA"),
        pytest.param(GatedDeltaNet2Config(n_heads=8, head_dim=32), id="head_dim=32"),
        pytest.param(GatedDeltaNet2Config(n_heads=8, expand_v=2.0), id="expand_v=2.0"),
        pytest.param(GatedDeltaNet2Config(n_heads=8, conv_size=8, conv_bias=True), id="conv_bias"),
        pytest.param(
            GatedDeltaNet2Config(n_heads=8, allow_neg_eigval=True), id="allow_neg_eigval=True"
        ),
    ],
)
def test_gated_delta_net_2_config_num_params(recurrent_config: GatedDeltaNet2Config):
    d_model = 512
    module = recurrent_config.build(d_model, layer_idx=0, n_layers=12, init_device="meta")

    # Make sure the estimated number of params matches the actual number of params.
    n_params = sum(p.numel() for p in module.parameters())
    assert recurrent_config.num_params(d_model) == n_params


@requires_fla
@requires_gpu
def test_gated_delta_net_2_fwd_bwd():
    device = "cuda"
    dtype = torch.bfloat16

    d_model, seq_len, batch_size = 256, 32, 2

    config = GatedDeltaNet2Config(n_heads=8)
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device=device)

    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

    with torch.autocast(device_type=device, dtype=dtype):
        y = module(x)
        assert y.shape == x.shape

        loss = y.sum()
        loss.backward()
    assert x.grad is not None


@requires_fla
@requires_gpu
def test_gated_delta_net_2_fwd_bwd_with_doc_lens():
    device = "cuda"
    dtype = torch.bfloat16

    # cu_seqlens-style document packing requires batch size 1.
    d_model, seq_len = 256, 128

    config = GatedDeltaNet2Config(n_heads=8)
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device=device)

    x = torch.randn(1, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    cu_doc_lens = torch.tensor([0, 40, 64, seq_len], dtype=torch.int32, device=device)

    with torch.autocast(device_type=device, dtype=dtype):
        y = module(x, cu_doc_lens=cu_doc_lens)
        assert y.shape == x.shape

        loss = y.sum()
        loss.backward()
    assert x.grad is not None


# Weight-name mapping from our GatedDeltaNet2 to fla's reference implementation,
# used to verify the port is numerically faithful.
_GDN2_FLA_WEIGHT_MAPPING = {
    "w_q.weight": "q_proj.weight",
    "w_k.weight": "k_proj.weight",
    "w_v.weight": "v_proj.weight",
    "w_f.0.weight": "f_proj.0.weight",
    "w_f.1.weight": "f_proj.1.weight",
    "w_b.weight": "b_proj.weight",
    "w_w.weight": "w_proj.weight",
    "w_g.0.weight": "g_proj.0.weight",
    "w_g.1.weight": "g_proj.1.weight",
    "w_g.1.bias": "g_proj.1.bias",
    "w_out.weight": "o_proj.weight",
    "A_log": "A_log",
    "dt_bias": "dt_bias",
    "q_conv1d.weight": "q_conv1d.weight",
    "k_conv1d.weight": "k_conv1d.weight",
    "v_conv1d.weight": "v_conv1d.weight",
    "o_norm.weight": "o_norm.weight",
}


@requires_fla
@requires_gpu
@pytest.mark.parametrize(
    "n_v_heads",
    [pytest.param(None, id="default"), pytest.param(16, id="GVA")],
)
def test_gated_delta_net_2_matches_fla_reference(n_v_heads):
    from fla.layers.gdn2 import GatedDeltaNet2 as FLAGatedDeltaNet2

    device = "cuda"
    dtype = torch.bfloat16

    d_model, n_heads, head_dim = 256, 8, 32
    # seq_len > 64 keeps the fla layer on the chunk kernel (matching our dispatch)
    # rather than its short-prefill fused-recurrent path.
    batch_size, seq_len = 2, 128

    seed_all(0)
    ref = FLAGatedDeltaNet2(
        hidden_size=d_model,
        num_heads=n_heads,
        num_v_heads=n_v_heads,
        head_dim=head_dim,
    ).to(device)

    module = GatedDeltaNet2(
        d_model=d_model,
        n_heads=n_heads,
        n_v_heads=n_v_heads,
        head_dim=head_dim,
        init_device=device,
    )
    ref_state = ref.state_dict()
    module.load_state_dict(
        {ours: ref_state[theirs] for ours, theirs in _GDN2_FLA_WEIGHT_MAPPING.items()}
    )

    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    with torch.autocast(device_type=device, dtype=dtype):
        y = module(x)
        y_ref, _, _ = ref(x)
    torch.testing.assert_close(y, y_ref, rtol=BF16_RTOL, atol=BF16_ATOL)


@requires_fla
def test_gated_delta_net_2_num_flops_per_token():
    d_model, n_heads, seq_len = 256, 2, 8192

    gdn2 = GatedDeltaNet2Config(n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1, init_device="meta"
    )
    attn = AttentionConfig(n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1, init_device="meta"
    )

    # At long sequence lengths, recurrent layers use fewer FLOPs than quadratic attention.
    gdn2_flops = gdn2.num_flops_per_token(seq_len)
    attn_flops = attn.num_flops_per_token(seq_len)  # type: ignore
    assert 0 < gdn2_flops < attn_flops


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


def _run_context_parallel_gdn2_ulysses(
    checkpoint_dir: str,
    inputs_path: str,
    outputs_path: str,
    gdn2_kwargs: Dict[str, Any],
):
    device = get_default_device()
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("cp",))

    gdn2 = GatedDeltaNet2(init_device=device.type, **gdn2_kwargs)
    gdn2.apply_cp(mesh["cp"], uly=UlyssesContextParallelStyle())
    load_model_and_optim_state(checkpoint_dir, gdn2)

    # Load the input and split it across ranks on the sequence dimension.
    x = torch.load(inputs_path, map_location=device)
    rank, world_size = get_rank(), get_world_size()
    chunk_size = x.size(1) // world_size
    x_local = x[:, rank * chunk_size : (rank + 1) * chunk_size, :]

    with torch.autocast(device.type, dtype=x_local.dtype):
        local_y = gdn2(x_local)
    y = DTensor.from_local(local_y, mesh, (Shard(1),))

    og_y = torch.load(outputs_path, map_location=device)
    tol_scale = 2  # requires slightly more tolerance than default
    torch.testing.assert_close(
        og_y, get_full_tensor(y), rtol=BF16_RTOL * tol_scale, atol=BF16_ATOL * tol_scale
    )


@requires_multi_gpu
@requires_fla
@pytest.mark.parametrize(
    "gdn2_kwargs",
    [
        pytest.param({"d_model": 128, "n_heads": 8}, id="default"),
        pytest.param({"d_model": 128, "n_heads": 8, "n_v_heads": 16}, id="GVA"),
    ],
)
def test_context_parallel_gdn2_ulysses(tmp_path, gdn2_kwargs: Dict[str, Any]):
    seed_all(0)
    device = get_default_device()

    # n_heads must be divisible by CP degree (world_size=2).
    gdn2 = GatedDeltaNet2(init_device=device.type, **gdn2_kwargs)

    bs, seq_len = 2, 64
    x = torch.randn(bs, seq_len, gdn2_kwargs["d_model"], device=device, dtype=torch.bfloat16)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        y = gdn2(x)

    outputs_path = tmp_path / "gdn2_y.pt"
    torch.save(y, outputs_path)
    inputs_path = tmp_path / "gdn2_x.pt"
    torch.save(x, inputs_path)
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, gdn2)

    run_distributed_test(
        _run_context_parallel_gdn2_ulysses,
        backend="nccl",
        start_method="spawn",
        func_args=(checkpoint_dir, inputs_path, outputs_path, gdn2_kwargs),
    )
