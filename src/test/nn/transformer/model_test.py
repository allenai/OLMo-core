import logging

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, init_device_mesh

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_world_size
from olmo_core.nn.layer_norm import LayerNorm
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelConfig
from olmo_core.utils import get_default_device

from ...distributed.utils import BACKENDS, requires_multi_gpu, run_distributed_test
from ...utils import GPU_MARKS

log = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "init_device, device",
    [
        pytest.param("cpu", "cuda", id="cpu->cuda", marks=GPU_MARKS),
        pytest.param("cpu", "cpu", id="cpu->cpu"),
    ],
)
def test_small_llama2_builder_config(init_device, device):
    config = TransformerConfig.llama2_271M(vocab_size=50257)
    log.info(config)
    model = config.build(init_device=init_device, device=torch.device(device))

    # Make sure num params estimate is correct.
    num_actual_params = 0
    for _, p in model.named_parameters():
        num_actual_params += p.numel()
    assert config.num_params == num_actual_params
    assert model.num_params == num_actual_params

    for module in model.modules():
        # Make sure there are no biases anywhere and layer norm weights are all 1.
        if isinstance(module, (nn.Linear, LayerNorm)):
            assert module.bias is None
        if isinstance(module, LayerNorm):
            assert module.weight is not None
            assert (module.weight == 1).all()

    # Make sure block_idx is set correctly.
    assert model.blocks[0].block_idx == 0
    assert model.blocks[-1].block_idx == len(model.blocks) - 1


def check_ngpt_matrices(model: nn.Module, d_model: int):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            assert module.bias is None

            w = module.weight
            if isinstance(w, DTensor):
                w = w.full_tensor()

            if w.shape[1] == d_model and "attention.w_out" not in name:
                pass
            elif w.shape[0] == d_model:
                w = w.transpose(0, 1)
            else:
                continue

            log.info(f"Checking norm for '{name}'")
            norm = torch.linalg.vector_norm(w, dim=1)
            torch.testing.assert_close(norm, torch.ones_like(norm))


@pytest.mark.parametrize(
    "init_device, device",
    [
        pytest.param("cpu", "cuda", id="cpu->cuda", marks=GPU_MARKS),
        pytest.param("cpu", "cpu", id="cpu->cpu"),
    ],
)
def test_small_ngpt_builder_config(init_device, device):
    config = TransformerConfig.ngpt_271M(vocab_size=50257)
    model = config.build(init_device=init_device, device=torch.device(device))

    # Make sure num params estimate is correct.
    num_actual_params = 0
    for _, p in model.named_parameters():
        num_actual_params += p.numel()
    assert config.num_params == num_actual_params
    assert model.num_params == num_actual_params

    # Make sure block_idx is set correctly.
    assert model.blocks[0].block_idx == 0
    assert model.blocks[-1].block_idx == len(model.blocks) - 1

    # Make sure all weights are normalized in the embedding dimension.
    check_ngpt_matrices(model, config.d_model)


def run_ngpt_with_fsdp2():
    config = TransformerConfig.ngpt_271M(
        vocab_size=50257,
        use_flash=False,
        dp_config=TransformerDataParallelConfig(name=DataParallelType.fsdp),
    )
    model = config.build(init_device="meta", max_seq_len=1024)
    optim = torch.optim.Adam(model.parameters())

    # Take an optimizer step.
    model(input_ids=torch.randint(0, 50257, (2, 128))).sum().backward()
    optim.step()

    # Re-normalize weights.
    model.normalize_matrices()

    # Check that the re-normalization was successful.
    check_ngpt_matrices(model, config.d_model)


@requires_multi_gpu
def test_ngpt_with_fsdp2():
    run_distributed_test(run_ngpt_with_fsdp2, backend="nccl", start_method="spawn")


def get_transformer_config(architecture: str) -> TransformerConfig:
    config: TransformerConfig
    if architecture == "olmo2":
        config = TransformerConfig.olmo2_190M(
            vocab_size=16_000,
            n_layers=2,
            fused_ops=False,
            use_flash=False,
        )
    elif architecture == "llama":
        config = TransformerConfig.llama2_271M(
            vocab_size=16_000,
            n_layers=2,
            fused_ops=False,
            use_flash=False,
        )
    else:
        raise NotImplementedError(architecture)

    return config


def get_transformer_inputs() -> torch.Tensor:
    return torch.arange(0, 128).unsqueeze(0)


def run_tensor_parallel_transformer(checkpoint_dir, outputs_path, architecture: str):
    device = get_default_device()
    config = get_transformer_config(architecture)
    input_ids = get_transformer_inputs().to(device)

    mesh = init_device_mesh(
        device.type,
        (get_world_size(),),
        mesh_dim_names=("tp",),
    )

    model = config.build(device=device, max_seq_len=512, tp_mesh=mesh["tp"])
    load_model_and_optim_state(checkpoint_dir, model)

    logits = model(input_ids=input_ids)

    loss = logits.sum()
    loss.backward()

    og_logits = torch.load(outputs_path, map_location=device)
    torch.testing.assert_close(og_logits, logits)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("architecture", ["olmo2", "llama"])
def test_tensor_parallel_transformer(backend: str, architecture: str, tmp_path):
    device = torch.device("cuda") if "nccl" in backend else torch.device("cpu")
    config = get_transformer_config(architecture)
    model = config.build(device=device, max_seq_len=512)
    input_ids = get_transformer_inputs().to(device)
    logits = model(input_ids=input_ids)

    outputs_path = tmp_path / "logits.pt"
    torch.save(logits, outputs_path)

    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, model)

    run_distributed_test(
        run_tensor_parallel_transformer,
        backend=backend,
        start_method="spawn",
        func_args=(
            checkpoint_dir,
            outputs_path,
            architecture,
        ),
    )
