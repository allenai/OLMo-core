from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.distributed as dist

from olmo_core.config import DType
from olmo_core.distributed.checkpoint import save_model_and_optim_state
from olmo_core.distributed.parallel.data_parallel import DataParallelType
from olmo_core.distributed.utils import get_world_size
from olmo_core.generate.generation import (
    GenerationConfig,
    TransformerGenerationModule,
    TransformerGenerationModuleConfig,
)
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.testing import requires_multi_gpu, run_distributed_test
from olmo_core.train.train_module.transformer.config import (
    TransformerDataParallelConfig,
)
from olmo_core.utils import seed_all


@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.0])
@pytest.mark.parametrize("compile_model", [False, True])
@pytest.mark.parametrize("dtype", [DType.float32, DType.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_generation_module_basic(
    temperature: float, compile_model: bool, dtype: DType, device: str
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    seed_all(42)

    # Create a small transformer config
    transformer_config = TransformerConfig.llama_like(
        d_model=128, n_heads=4, n_layers=2, vocab_size=1000
    )

    # Create generation config
    generation_config = GenerationConfig(
        max_length=32, temperature=temperature, eos_token_id=2, pad_token_id=0
    )

    # Build generation module
    model = transformer_config.build()
    generation_module = TransformerGenerationModule(
        model=model,
        generation_config=generation_config,
        compile_model=compile_model,
        autocast_precision=dtype.as_pt() if dtype != DType.float32 else None,
        device=torch.device(device),
    )

    # Create input
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(1, 100, (batch_size, seq_len), device=device)

    output1 = generation_module.generate_batch(input_ids)

    # TODO: test to make sure the model is compiled exactly once if compile_model is True
    # We do not want to accidentally compile the model each time the sequence length changes.

    # Verify output shape and properties
    output = output1 if compile_model else output1
    assert output.shape[0] == batch_size
    assert output.shape[1] <= generation_config.max_length
    assert output.shape[1] >= seq_len + 1
    assert torch.all(output[:, :seq_len] == input_ids)

    # Check that generation stopped at EOS or max_length
    for i in range(batch_size):
        seq = output[i]
        eos_positions = (seq == generation_config.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            # If EOS found, check padding after it
            first_eos = eos_positions[0].item()
            if first_eos < output.shape[1] - 1:
                assert torch.all(seq[first_eos + 1 :] == generation_config.pad_token_id)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_generation_module_state_dict(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    seed_all(42)

    transformer_config = TransformerConfig.llama_like(
        d_model=128, n_heads=4, n_layers=2, vocab_size=1000
    )

    generation_config = GenerationConfig(max_length=16)

    # Create first module
    model1 = transformer_config.build()
    module1 = TransformerGenerationModule(
        model=model1,
        generation_config=generation_config,
        device=torch.device(device),
    )

    # Get state dict
    state_dict = module1.state_dict()
    assert "model" in state_dict

    # Create second module and load state dict
    model2 = transformer_config.build()
    module2 = TransformerGenerationModule(
        model=model2,
        generation_config=generation_config,
        device=torch.device(device),
    )
    module2.load_state_dict(state_dict)

    # Verify they produce same output
    input_ids = torch.randint(1, 100, (1, 4), device=device)
    output1 = module1.generate_batch(input_ids)
    output2 = module2.generate_batch(input_ids)
    torch.testing.assert_close(output1, output2)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_generation_module_from_checkpoint(tmp_path: Path, device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    seed_all(42)

    # Create and save a model
    transformer_config = TransformerConfig.llama_like(
        d_model=128, n_heads=4, n_layers=2, vocab_size=1000
    )

    model = transformer_config.build()
    generation_module = TransformerGenerationModule(
        model=model,
        generation_config=GenerationConfig(),
        device=torch.device(device),
    )

    # Save checkpoint
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, generation_module.model)

    # Load from checkpoint
    generation_module2 = TransformerGenerationModule.from_checkpoint(
        transformer_config=transformer_config,
        checkpoint_dir=checkpoint_dir,
        generation_config=GenerationConfig(max_length=20, temperature=0.5),
    )

    # Verify generation works
    input_ids = torch.randint(1, 100, (2, 4), device=device)
    output = generation_module2.generate_batch(input_ids)
    assert output.shape[0] == 2
    assert output.shape[1] >= 4
    assert output.shape[1] <= 20


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("max_length", [16, 32, 64])
@pytest.mark.parametrize("eos_token_id", [None, 2])
def test_generation_config_overrides(device: str, max_length: int, eos_token_id: Optional[int]):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    seed_all(42)

    transformer_config = TransformerConfig.llama_like(
        d_model=128, n_heads=4, n_layers=2, vocab_size=1000
    )

    # Create module with default config
    generation_config = GenerationConfig(
        max_length=128,
        eos_token_id=1,
        pad_token_id=0,
        temperature=0.0,
    )

    model = transformer_config.build()
    generation_module = TransformerGenerationModule(
        model=model,
        generation_config=generation_config,
        device=torch.device(device),
    )

    # Generate with overrides
    input_ids = torch.randint(1, 50, (1, 4), device=device)
    output = generation_module.generate_batch(
        input_ids,
        max_length=max_length,
        eos_token_id=eos_token_id,
        temperature=0.8,
    )

    assert output.shape[1] <= max_length

    # Check EOS behavior
    if eos_token_id is not None:
        eos_positions = (output == eos_token_id).nonzero(as_tuple=True)
        if len(eos_positions[0]) > 0:
            # Should have padding after EOS
            first_eos_idx = eos_positions[1][0].item()
            if first_eos_idx < output.shape[1] - 1:
                assert torch.all(output[0, first_eos_idx + 1 :] == generation_config.pad_token_id)


def run_distributed_generation(
    checkpoint_dir: Path,
    transformer_config: TransformerConfig,
    generation_config: GenerationConfig,
    dp_config: Optional[TransformerDataParallelConfig],
    input_ids: torch.Tensor,
    expected_shape: tuple,
):
    seed_all(42)

    # Create generation module with parallelism configs
    generation_module = TransformerGenerationModule.from_checkpoint(
        transformer_config=transformer_config,
        checkpoint_dir=checkpoint_dir,
        generation_config=generation_config,
        dp_config=dp_config,
    )

    # Move input to correct device
    device = torch.device("cuda", dist.get_rank())
    input_ids = input_ids.to(device)

    # Generate
    output = generation_module.generate_batch(input_ids)

    # Basic checks
    assert output.shape == expected_shape
    assert output.device == device

    # Verify all ranks got same result using FSDP
    if dp_config is not None and get_world_size() > 1:
        output_list = [torch.zeros_like(output) for _ in range(get_world_size())]
        dist.all_gather(output_list, output)
        for i in range(1, len(output_list)):
            torch.testing.assert_close(output_list[0], output_list[i])


@requires_multi_gpu
def test_generation_module_distributed_fsdp(tmp_path: Path):
    seed_all(42)

    # Create and save a model on single device first
    transformer_config = TransformerConfig.llama_like(
        d_model=128, n_heads=4, n_layers=2, vocab_size=1000
    )

    model = transformer_config.build()
    generation_module = TransformerGenerationModule(
        model=model,
        generation_config=GenerationConfig(),
        device=torch.device("cuda"),
    )

    # Save checkpoint
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, generation_module.model)

    # Prepare configs for distributed test
    dp_config = TransformerDataParallelConfig(name=DataParallelType.fsdp)

    generation_config = GenerationConfig(
        max_length=16,
        temperature=0.0,
        eos_token_id=2,
        pad_token_id=0,
    )

    # Create test input
    input_ids = torch.randint(1, 100, (2, 8))
    expected_shape = (2, 16)  # max_length

    # Run distributed test with 2 GPUs
    run_distributed_test(
        run_distributed_generation,
        world_size=2,
        backend="nccl",
        start_method="spawn",
        func_args=(
            checkpoint_dir,
            transformer_config,
            generation_config,
            dp_config,
            input_ids,
            expected_shape,
        ),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_generation_module_config_build(tmp_path: Path, device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    seed_all(42)

    # Create and save a model
    transformer_config = TransformerConfig.llama_like(
        d_model=128, n_heads=4, n_layers=2, vocab_size=1000
    )

    model = transformer_config.build()
    generation_module = TransformerGenerationModule(
        model=model,
        generation_config=GenerationConfig(),
        device=torch.device(device),
    )

    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, generation_module.model)

    # Build from config with checkpoint
    config = TransformerGenerationModuleConfig(
        generation_config=GenerationConfig(max_length=24),
        float8_config=None,
    )

    generation_module2 = config.build(
        checkpoint_dir=checkpoint_dir,
        transformer_config=transformer_config,
        work_dir=tmp_path / "work",
        device=torch.device(device),
    )

    # Verify it works
    input_ids = torch.randint(1, 100, (2, 4), device=device)
    output = generation_module2.generate_batch(input_ids)
    assert output.shape[0] == 2
    assert output.shape[1] <= 24
