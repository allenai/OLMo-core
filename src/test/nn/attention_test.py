from typing import Any, Dict, Optional

import pytest
import torch
from torch.distributed.tensor import Shard, init_device_mesh

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    Attention,
    AttentionConfig,
    AttentionType,
    FusedAttention,
    RingAttentionLoadBalancerType,
    RingAttentionZigZagLoadBalancer,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.testing import (
    BACKENDS,
    DEVICES,
    FLASH_MARKS,
    GPU_MARKS,
    requires_flash_attn,
    requires_gpu,
    requires_multi_gpu,
    run_distributed_test,
)
from olmo_core.utils import get_default_device, seed_all

BF16_RTOL = 1e-5
BF16_ATOL = 5e-3


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, id="bf16", marks=GPU_MARKS),
        pytest.param(torch.float32, id="fp32"),
    ],
)
@pytest.mark.parametrize(
    "n_kv_heads",
    [pytest.param(None, id="MHA"), pytest.param(1, id="MQA"), pytest.param(2, id="GQA")],
)
@pytest.mark.parametrize(
    "use_flash",
    [pytest.param(True, id="flash", marks=FLASH_MARKS), pytest.param(False, id="torch-SDPA")],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"clip_qkv": 8.0}, id="QKV-clip"),
        pytest.param({"rope": RoPEConfig()}, id="rope"),
        pytest.param({"rope": RoPEConfig(name=RoPEType.complex)}, id="complex-rope"),
        pytest.param({"qk_norm": LayerNormConfig()}, id="qk-norm"),
        pytest.param({"qk_norm": LayerNormConfig(), "use_head_qk_norm": True}, id="head-qk-norm"),
    ],
)
def test_attention(
    dtype: torch.dtype,
    device: torch.device,
    n_kv_heads: Optional[int],
    use_flash: bool,
    kwargs: Dict[str, Any],
):
    if use_flash and dtype == torch.float32:
        pytest.skip("flash requires a low precision dtype")

    if dtype == torch.bfloat16 and device.type == "cpu":
        pytest.skip("bf16 requires GPU")

    torch.random.manual_seed(0)

    d_model = 128
    seq_len = 32

    attention = Attention(
        d_model=d_model,
        n_heads=4,
        n_kv_heads=n_kv_heads,
        use_flash=use_flash,
        init_device=device.type,
        **kwargs,
    )

    x1 = torch.randn(1, seq_len, d_model, dtype=dtype, device=device)
    x2 = torch.randn(1, seq_len, d_model, dtype=dtype, device=device)
    x = torch.cat([x1, x2])

    # Make sure batch outputs match individual outputs.
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
        y1 = attention(x1)
        y2 = attention(x2)
        y = attention(x)

    torch.testing.assert_close(y[0:1, :, :], y1)
    torch.testing.assert_close(y[1:, :, :], y2)


@requires_gpu
@requires_flash_attn
@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="bf16")])
@pytest.mark.parametrize(
    "use_flash", [pytest.param(True, id="flash"), pytest.param(False, id="torch-SDPA")]
)
def test_fused_attention_against_non_fused(dtype: torch.dtype, use_flash: bool):
    torch.random.manual_seed(0)

    d_model = 128
    seq_len = 32
    batch_size = 2
    kwargs: Dict[str, Any] = dict(
        d_model=d_model,
        n_heads=8,
        init_device="cuda",
    )

    attention = Attention(use_flash=use_flash, **kwargs)
    fused_att = FusedAttention(**kwargs)

    # Make sure weights match.
    with torch.no_grad():
        fused_att.w_out.load_state_dict(attention.w_out.state_dict())
        fused_att.w_qkv.weight.copy_(
            torch.cat([attention.w_q.weight, attention.w_k.weight, attention.w_v.weight])
        )
        fused_att.w_qkv.bias.copy_(
            torch.cat([attention.w_q.bias, attention.w_k.bias, attention.w_v.bias])
        )

    x1 = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device="cuda")
    x2 = x1.clone()

    with torch.autocast("cuda", dtype=dtype, enabled=True):
        y1 = attention(x1)
        y2 = fused_att(x2)

    torch.testing.assert_close(y1, y2)


@requires_gpu
@requires_flash_attn
def test_fused_attention_with_rope():
    torch.random.manual_seed(0)

    d_model = 128
    seq_len = 32

    fused_att = FusedAttention(
        d_model=d_model, n_heads=8, rope=RoPEConfig(name=RoPEType.fused), init_device="cuda"
    )

    x1 = torch.randn(1, seq_len, d_model, dtype=torch.bfloat16, device="cuda")
    x2 = torch.randn(1, seq_len, d_model, dtype=torch.bfloat16, device="cuda")
    x = torch.cat([x1, x2])

    # Make sure batch outputs match individual outputs.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y1 = fused_att(x1)
        y2 = fused_att(x2)
        y = fused_att(x)

    torch.testing.assert_close(y[0:1, :, :], y1)
    torch.testing.assert_close(y[1:, :, :], y2)


@requires_gpu
@requires_flash_attn
def test_attention_with_intra_document_masking():
    torch.random.manual_seed(0)

    d_model = 128
    seq_len = 32

    attention = Attention(d_model=d_model, n_heads=8, init_device="cuda", use_flash=True)
    fused_att = FusedAttention(d_model=d_model, n_heads=8, init_device="cuda")

    # Make sure weights match.
    with torch.no_grad():
        fused_att.w_out.load_state_dict(attention.w_out.state_dict())
        fused_att.w_qkv.weight.copy_(
            torch.cat([attention.w_q.weight, attention.w_k.weight, attention.w_v.weight])
        )
        fused_att.w_qkv.bias.copy_(
            torch.cat([attention.w_q.bias, attention.w_k.bias, attention.w_v.bias])
        )

    x = torch.randn(2, seq_len, d_model, dtype=torch.bfloat16, device="cuda")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y1 = attention(x.clone())
        y2 = attention(
            x.clone(),
            max_doc_len=seq_len,
            cu_doc_lens=torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32, device="cuda"),
        )

        y1_fused = fused_att(x.clone())
        y2_fused = fused_att(
            x.clone(),
            max_doc_len=seq_len,
            cu_doc_lens=torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32, device="cuda"),
        )

    torch.testing.assert_close(y1, y2)
    torch.testing.assert_close(y1_fused, y2_fused)
    torch.testing.assert_close(y1, y1_fused)


@pytest.mark.parametrize(
    "attn_config",
    [
        AttentionConfig(name=AttentionType.default, n_heads=8, n_kv_heads=1, bias=True),
        AttentionConfig(name=AttentionType.default, n_heads=8, n_kv_heads=1, bias=False),
        AttentionConfig(
            name=AttentionType.default, n_heads=8, bias=False, qk_norm=LayerNormConfig()
        ),
    ],
)
def test_attention_builder_config(attn_config: AttentionConfig):
    d_model = 64

    attn = attn_config.build(d_model, layer_idx=0, n_layers=1)

    # Make sure the estimated number of params matches the actual number of params.
    n_params = sum(p.numel() for p in attn.parameters())
    assert attn_config.num_params(d_model) == n_params


def _get_lb(rank: int, world_size: int) -> RingAttentionZigZagLoadBalancer:
    return RingAttentionZigZagLoadBalancer(cp_rank=rank, cp_world_size=world_size)


def test_zig_zag_load_balancer_padding():
    x, padding_added = _get_lb(0, 4).pad(torch.tensor([0, 1, 2, 3, 4, 5]).unsqueeze(0), 1, -1)
    assert x.tolist() == [[0, 1, 2, 3, 4, 5, -1, -1]]
    assert padding_added == 2


def test_zig_zag_load_balancer_shard():
    x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).unsqueeze(0)
    assert _get_lb(0, 4).batch_shard(inputs=[x], seq_dims=[1])[0].tolist() == [
        [
            0,
            7,
        ]
    ]
    assert _get_lb(3, 4).batch_shard(inputs=[x], seq_dims=[1])[0].tolist() == [
        [
            3,
            4,
        ]
    ]


def test_zig_zag_load_balancer_shard_with_padding():
    x = torch.tensor([0, 1, 2, 3, 4, 5]).unsqueeze(0)
    assert _get_lb(0, 4).batch_shard(inputs=[x], seq_dims=[1], pad_values=[-1])[0].tolist() == [
        [
            0,
            -1,
        ]
    ]
    assert _get_lb(3, 4).batch_shard(inputs=[x], seq_dims=[1], pad_values=[-1])[0].tolist() == [
        [
            3,
            4,
        ]
    ]


def test_zig_zag_load_balancer_shard_by_document():
    x = torch.tensor(list(range(12))).unsqueeze(0)
    cu_doc_lens = torch.tensor([0, 8, 12])

    assert _get_lb(0, 2).batch_shard_by_document(inputs=[x], seq_dims=[1], cu_doc_lens=cu_doc_lens)[
        0
    ][0].tolist() == [
        [
            0,
            1,
            6,
            7,
            8,
            11,
        ]
    ]

    assert _get_lb(1, 2).batch_shard_by_document(inputs=[x], seq_dims=[1], cu_doc_lens=cu_doc_lens)[
        0
    ][0].tolist() == [
        [
            2,
            3,
            4,
            5,
            9,
            10,
        ]
    ]


def test_zig_zag_load_balancer_shard_by_document_with_padding():
    x = torch.tensor(list(range(12))).unsqueeze(0)
    cu_doc_lens = torch.tensor([0, 7, 10])

    res, opts = _get_lb(0, 2).batch_shard_by_document(
        inputs=[x],
        seq_dims=[1],
        cu_doc_lens=cu_doc_lens,
        pad_values=[-1],
    )
    new_doc_lens = opts["cu_doc_lens"]
    assert new_doc_lens.tolist() == [0, 4, 6]
    assert res[0].tolist() == [
        [
            0,
            1,
            6,
            -1,
            7,
            -1,
        ]
    ]


@pytest.mark.parametrize(
    "force_first, force_last, layer_idx, expected_window_size, expected_should_use_swa",
    [
        # Test with forcing full attention on neither first nor last layer.
        (False, False, 0, 1024, True),  # Pattern start
        (False, False, 1, 2048, True),  # Pattern middle
        (False, False, 2, -1, False),  # Pattern end
        (False, False, 11, -1, False),  # Last layer, pattern end
        (True, False, 1, 1024, True),  # Effective layer=0
        (True, False, 11, 2048, True),  # Effective layer=10
        # Test with forcing full attention only on the last layer.
        (False, True, 0, 1024, True),  # First layer, not forced
        (False, True, 11, -1, False),  # Forced last
        # Test with forcing full attention on both first and last layers.
        (True, True, 0, -1, False),  # Forced first
        (True, True, 1, 1024, True),  # Effective layer=0
        (True, True, 11, -1, False),  # Forced last
    ],
)
def test_sliding_window_attention_config_window_size(
    force_first: bool,
    force_last: bool,
    layer_idx: int,
    expected_window_size: int,
    expected_should_use_swa: bool,
):
    n_layers = 12
    pattern = [1024, 2048, -1]

    config = SlidingWindowAttentionConfig(
        pattern=pattern,
        force_full_attention_on_first_layer=force_first,
        force_full_attention_on_last_layer=force_last,
    )

    assert config._get_window_size(layer_idx, n_layers) == expected_window_size
    assert config.should_use_swa(layer_idx, n_layers) == expected_should_use_swa


def test_sliding_window_attention_config_get_window_size_error():
    n_layers = 12
    pattern = [1024, 2048, -1]
    config = SlidingWindowAttentionConfig(
        pattern=pattern,
        force_full_attention_on_first_layer=True,
        force_full_attention_on_last_layer=True,
    )

    assert config.get_window_size(1, n_layers) == 1024  # This layer uses SWA
    with pytest.raises(ValueError):
        config.get_window_size(0, n_layers)  # This layer uses full attention


def test_sliding_window_attention_config_invalid_pattern_error():
    with pytest.raises(OLMoConfigurationError):
        bad_config = SlidingWindowAttentionConfig(
            pattern=[0], force_full_attention_on_first_layer=False
        )
        bad_config._get_window_size(0, n_layers=12)


def _run_tensor_parallel_attention(
    checkpoint_dir: str, inputs_path: str, outputs_path: str, attn_kwargs: Dict[str, Any]
):
    device = get_default_device()
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("tp",))

    attn = Attention(init_device=device.type, **attn_kwargs)

    # Shard sequence dim in/out like the transformer block does.
    attn.apply_tp(mesh["tp"], input_layout=Shard(1), output_layout=Shard(1), use_local_output=False)
    load_model_and_optim_state(checkpoint_dir, attn)

    x = torch.load(inputs_path, map_location=device)
    rank, world_size = get_rank(), get_world_size()
    chunk = x.size(1) // world_size
    x_local = x[:, rank * chunk : (rank + 1) * chunk, :]
    y_local = attn(x_local)

    # Backward to exercise graph in TP mode.
    y_local.sum().backward()

    y_ref = torch.load(outputs_path, map_location=device)
    y_ref_local = y_ref[:, rank * chunk : (rank + 1) * chunk, :]
    torch.testing.assert_close(y_ref_local, y_local.to_local())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "attn_kwargs",
    [
        pytest.param({}, id="default"),
        pytest.param({"rope": RoPEConfig()}, id="rope"),
        pytest.param({"qk_norm": LayerNormConfig()}, id="qk-layernorm"),
        pytest.param({"qk_norm": LayerNormConfig(), "rope": RoPEConfig()}, id="qk-layernorm-rope"),
        pytest.param(
            {"qk_norm": LayerNormConfig(), "use_head_qk_norm": True},
            id="headwise-qk-layernorm",
        ),
        pytest.param(
            {"qk_norm": LayerNormConfig(), "use_head_qk_norm": True, "rope": RoPEConfig()},
            id="headwise-qk-layernorm-rope",
        ),
    ],
)
def test_tensor_parallel_attention(backend: str, attn_kwargs: Dict[str, Any], tmp_path):
    device = torch.device("cuda") if "nccl" in backend else torch.device("cpu")

    seed_all(0)
    attn_kwargs.update({"d_model": 128, "n_heads": 8, "use_flash": False})
    attn = Attention(init_device=device.type, **attn_kwargs)

    bs, seq_len = 2, 64
    x = torch.randn(bs, seq_len, attn_kwargs["d_model"], device=device)
    y = attn(x)

    outputs_path = tmp_path / "attn_y.pt"
    torch.save(y, outputs_path)
    inputs_path = tmp_path / "attn_x.pt"
    torch.save(x, inputs_path)
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, attn)

    run_distributed_test(
        _run_tensor_parallel_attention,
        backend=backend,
        start_method="spawn",
        func_args=(checkpoint_dir, inputs_path, outputs_path, attn_kwargs),
    )


def _run_context_parallel_attention(
    checkpoint_dir: str,
    inputs_path: str,
    outputs_path: str,
    attn_kwargs: Dict[str, Any],
    load_balancer_type,
    head_stride: int,
):
    device = get_default_device()
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("cp",))

    attn = Attention(init_device=device.type, **attn_kwargs)
    attn.apply_cp(mesh["cp"], load_balancer_type, head_stride=head_stride)
    load_model_and_optim_state(checkpoint_dir, attn)

    # Load the input and split it across ranks on the sequence dimension.
    x = torch.load(inputs_path, map_location=device)
    rank, world_size = get_rank(), get_world_size()
    chunk_size = x.size(1) // world_size
    x_local = x[:, rank * chunk_size : (rank + 1) * chunk_size, :]

    with torch.autocast(device.type, dtype=x_local.dtype):
        y_local = attn(x_local)

    # Backward to exercise graph in CP mode.
    y_local.sum().backward()

    # Load the reference output and split it across ranks on the sequence dimension.
    y_ref = torch.load(outputs_path, map_location=device)
    y_ref_local = y_ref[:, rank * chunk_size : (rank + 1) * chunk_size, :]

    # Compare the local output with the reference output.
    torch.testing.assert_close(y_ref_local, y_local, rtol=BF16_RTOL, atol=BF16_ATOL)


@requires_multi_gpu
@requires_flash_attn
@pytest.mark.parametrize(
    "load_balancer_type",
    [
        pytest.param(RingAttentionLoadBalancerType.zig_zag, id="zig_zag"),
        pytest.param(RingAttentionLoadBalancerType.llama3, id="llama3"),
    ],
)
@pytest.mark.parametrize("head_stride", [pytest.param(1), pytest.param(8)])
@pytest.mark.parametrize(
    "attn_kwargs", [pytest.param({}, id="default"), pytest.param({"window_size": 8}, id="swa")]
)
def test_context_parallel_attention(
    attn_kwargs: Dict[str, Any], load_balancer_type, head_stride: int, tmp_path
):
    seed_all(0)
    device = torch.device("cuda")

    # CP requires flash-attn and low precision dtypes.
    attn_kwargs.update({"d_model": 128, "n_heads": 8, "use_flash": True})
    attn = Attention(init_device=device.type, **attn_kwargs)

    bs, seq_len = 2, 64
    x = torch.randn(bs, seq_len, attn_kwargs["d_model"], device=device, dtype=torch.bfloat16)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = attn(x)

    outputs_path = tmp_path / "attn_y.pt"
    torch.save(y, outputs_path)
    inputs_path = tmp_path / "attn_x.pt"
    torch.save(x, inputs_path)
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, attn)

    run_distributed_test(
        _run_context_parallel_attention,
        backend="nccl",
        start_method="spawn",
        func_args=(
            checkpoint_dir,
            inputs_path,
            outputs_path,
            attn_kwargs,
            load_balancer_type,
            head_stride,
        ),
    )
