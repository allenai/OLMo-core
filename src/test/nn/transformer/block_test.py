from typing import Any, Dict, Optional, Type

import pytest
import torch
from torch.distributed.tensor import Shard, init_device_mesh

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.fla import FLAConfig
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.transformer.block import (
    FLABlock,
    PeriNormTransformerBlock,
    ReorderedNormTransformerBlock,
    TransformerBlock,
    TransformerBlockBase,
)
from olmo_core.testing import BACKENDS, GPU_MARKS, requires_multi_gpu, run_distributed_test
from olmo_core.utils import get_default_device, seed_all


def _build_block(
    block_cls: Type[TransformerBlockBase],
    *,
    d_model: int,
    init_device: str,
    kwargs: Optional[Dict[str, Any]] = None,
) -> TransformerBlockBase:
    ln_cfg = LayerNormConfig()
    ff_cfg = FeedForwardConfig(hidden_size=4 * d_model)
    kwargs = kwargs or {}

    if block_cls == FLABlock:
        fla_cfg = FLAConfig(name=kwargs.get("name", "GatedDeltaNet"))
        return FLABlock(
            d_model=d_model,
            n_heads=kwargs.get("n_heads", 4),
            block_idx=0,
            n_layers=1,
            fla=fla_cfg,
            feed_forward=ff_cfg,
            layer_norm=ln_cfg,
            init_device=init_device,
        )
    else:
        attn_cfg = AttentionConfig(**kwargs)
        # block_cls is a TransformerBlock subclass here
        return block_cls(  # type: ignore[call-arg]
            d_model=d_model,
            block_idx=0,
            n_layers=1,
            attention=attn_cfg,
            feed_forward=ff_cfg,
            layer_norm=ln_cfg,
            init_device=init_device,
        )


def _run_tensor_parallel_block(
    checkpoint_dir: str,
    inputs_path: str,
    outputs_path: str,
    block_cls: Type[TransformerBlockBase],
    d_model: int,
    kwargs: Optional[Dict[str, Any]] = None,
    compile_model: bool = False,
):
    device = get_default_device()
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("tp",))

    block = _build_block(block_cls, d_model=d_model, init_device=device.type, kwargs=kwargs)

    # Shard sequence dim in/out like the transformer model does.
    block.apply_tp(mesh["tp"], input_layout=Shard(1))
    load_model_and_optim_state(checkpoint_dir, block)

    if compile_model:
        block.apply_compile()

    x = torch.load(inputs_path, map_location=device)
    rank, world_size = get_rank(), get_world_size()
    chunk = x.size(1) // world_size
    x_local = x[:, rank * chunk : (rank + 1) * chunk, :]
    y_local = block(x_local)

    # Backward to exercise graph in TP mode.
    y_local.sum().backward()

    y_ref = torch.load(outputs_path, map_location=device)
    y_ref_local = y_ref[:, rank * chunk : (rank + 1) * chunk, :]
    torch.testing.assert_close(y_ref_local, y_local.to_local())


def _fla_available():
    try:
        import fla  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "compile_model",
    [pytest.param(False, id="compile_model=False"), pytest.param(True, id="compile_model=True")],
)
@pytest.mark.parametrize(
    "block_cls,kwargs",
    [
        pytest.param(TransformerBlock, dict(n_heads=8), id="transformer-default"),
        pytest.param(
            TransformerBlock, dict(n_heads=8, rope=None, bias=False), id="transformer-no-bias"
        ),
        pytest.param(ReorderedNormTransformerBlock, dict(n_heads=8), id="reordered-norm"),
        pytest.param(PeriNormTransformerBlock, dict(n_heads=8), id="peri-norm"),
        pytest.param(
            FLABlock,
            dict(name="GatedDeltaNet", n_heads=4),
            id="fla-gated-deltanet",
            marks=pytest.mark.skipif(not _fla_available(), reason="fla library not installed"),
        ),
    ],
)
def test_tensor_parallel_block(
    backend: str,
    compile_model: bool,
    block_cls: Type[TransformerBlockBase],
    kwargs: Dict[str, Any],
    tmp_path,
):
    device = torch.device("cuda") if "nccl" in backend else torch.device("cpu")

    seed_all(0)
    d_model = 64

    # Add attention-specific defaults for non-FLA blocks
    if block_cls != FLABlock:
        kwargs = {**kwargs, "name": AttentionType.default, "use_flash": False}

    block = _build_block(block_cls, d_model=d_model, init_device=device.type, kwargs=kwargs)

    if compile_model:
        block.apply_compile()

    # FLA GatedDeltaNet requires seq_len > 64 for training (chunk mode)
    # because fused_recurrent mode (used when seq_len <= 64) is inference-only
    bs, seq_len = 2, 128
    x = torch.randn(bs, seq_len, d_model, device=device)
    y = block(x)

    outputs_path = tmp_path / "block_y.pt"
    torch.save(y, outputs_path)
    inputs_path = tmp_path / "block_x.pt"
    torch.save(x, inputs_path)
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, block)

    run_distributed_test(
        _run_tensor_parallel_block,
        backend=backend,
        start_method="spawn",
        func_args=(
            checkpoint_dir,
            inputs_path,
            outputs_path,
            block_cls,
            d_model,
            kwargs,
            compile_model,
        ),
    )


def _run_context_parallel_block(checkpoint_dir, inputs_path, outputs_path, d_model, kwargs):
    device = get_default_device()
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("cp",))

    block = _build_block(TransformerBlock, d_model=d_model, init_device=device.type, kwargs=kwargs)
    block.apply_cp(mesh["cp"], load_balancer=None)
    load_model_and_optim_state(checkpoint_dir, block)

    x = torch.load(inputs_path, map_location=device)
    logits = block(x)
    logits.sum().backward()

    ref = torch.load(outputs_path, map_location=device)
    torch.testing.assert_close(ref, logits)


@requires_multi_gpu
@pytest.mark.parametrize(
    "backend", ["nccl"], ids=["ulysses-cp"]
)  # ring CP still tested elsewhere
def test_context_parallel_block(backend: str, tmp_path):
    device = torch.device("cuda")
    seed_all(0)
    d_model = 64
    kwargs = dict(name=AttentionType.default, n_heads=8, use_flash=True)

    block = _build_block(TransformerBlock, d_model=d_model, init_device=device.type, kwargs=kwargs)
    block.init_weights(device=device)

    bs, seq_len = 2, 128
    x = torch.randn(bs, seq_len, d_model, device=device)
    y = block(x)

    outputs_path = tmp_path / "block_y.pt"
    torch.save(y, outputs_path)
    inputs_path = tmp_path / "block_x.pt"
    torch.save(x, inputs_path)
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, block)

    run_distributed_test(
        _run_context_parallel_block,
        backend=backend,
        start_method="spawn",
        func_args=(
            checkpoint_dir,
            inputs_path,
            outputs_path,
            d_model,
            kwargs,
        ),
    )
