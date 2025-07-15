from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.distributed as dist

from olmo_core.distributed.checkpoint import save_model_and_optim_state
from olmo_core.distributed.parallel.data_parallel import DataParallelType
from olmo_core.distributed.utils import get_world_size
from olmo_core.generate.config import (
    GenerationConfig,
    TransformerGenerationModuleConfig,
)
from olmo_core.generate.generation import TransformerGenerationModule
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.testing import requires_multi_gpu, run_distributed_test
from olmo_core.testing.utils import DEVICES, GPU_MARKS, requires_gpu
from olmo_core.train.train_module.transformer.config import (
    TransformerDataParallelConfig,
)
from olmo_core.utils import seed_all


@pytest.fixture
def transformer_config():
    return TransformerConfig.llama_like(d_model=128, n_heads=4, n_layers=2, vocab_size=512)


@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.0])
@pytest.mark.parametrize("compile_model", [False, True])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, id="dtype=float32"),
        pytest.param(torch.bfloat16, id="dtype=bfloat16", marks=GPU_MARKS),
    ],
)
@pytest.mark.parametrize("device", DEVICES)
def test_generation_module_basic(
    transformer_config: TransformerConfig,
    temperature: float,
    compile_model: bool,
    dtype: torch.dtype,
    device: torch.device,
):
    seed_all(42)

    generation_config = GenerationConfig(
        max_length=32, temperature=temperature, eos_token_id=2, pad_token_id=0
    )

    # Build generation module
    model = transformer_config.build()
    generation_module = TransformerGenerationModule(
        model=model,
        generation_config=generation_config,
        compile_model=compile_model,
        autocast_precision=dtype if dtype != torch.float32 else None,
        device=device,
    )

    # Create test input
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(1, 100, (batch_size, seq_len), device=device)

    output_ids, output_logits = generation_module.generate_batch(
        input_ids, return_logits=True, completions_only=False
    )

    # TODO: test to make sure the model is compiled exactly once if compile_model is True
    # We do not want to accidentally compile the model each time the sequence length changes.

    # Verify output shape and properties
    assert output_ids.shape[0] == batch_size
    assert output_ids.shape[1] <= generation_config.max_length
    assert output_ids.shape[1] >= seq_len + 1
    assert torch.all(output_ids[:, :seq_len] == input_ids)
    assert isinstance(output_logits, torch.Tensor)
    assert output_logits.shape == (batch_size, output_ids.shape[1], transformer_config.vocab_size)

    # Check that generation stopped at EOS or max_length
    for i in range(batch_size):
        seq = output_ids[i]
        eos_positions = (seq == generation_config.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            # If EOS found, check padding after it
            first_eos = eos_positions[0].item()
            if first_eos < output_ids.shape[1] - 1:
                assert torch.all(seq[first_eos + 1 :] == generation_config.pad_token_id)


@pytest.mark.parametrize("device", DEVICES)
def test_generation_module_state_dict(transformer_config: TransformerConfig, device: torch.device):
    seed_all(42)

    generation_config = GenerationConfig(max_length=16)

    # Create first generation module
    model1 = transformer_config.build()
    module1 = TransformerGenerationModule(
        model=model1,
        generation_config=generation_config,
        device=device,
    )

    # Get state dict
    state_dict = module1.state_dict()
    assert "model" in state_dict

    # Create second generation module and load state dict
    model2 = transformer_config.build()
    module2 = TransformerGenerationModule(
        model=model2,
        generation_config=generation_config,
        device=device,
    )
    module2.load_state_dict(state_dict)

    # Verify they produce same output
    input_ids = torch.randint(1, 100, (1, 4), device=device)
    output_ids1, _ = module1.generate_batch(input_ids, return_logits=False)
    output_ids2, _ = module2.generate_batch(input_ids, return_logits=False)
    torch.testing.assert_close(output_ids1, output_ids2)


@requires_gpu
@pytest.mark.parametrize("max_length", [16, 64])
@pytest.mark.parametrize("eos_token_id", [1, 2])
def test_generation_config_overrides(
    transformer_config: TransformerConfig,
    max_length: int,
    eos_token_id: Optional[int],
    device: torch.device = torch.device("cuda"),
):
    seed_all(42)

    # Create generation module with default config
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
        device=device,
    )

    # Generate with overrides
    input_ids = torch.randint(1, 50, (1, 4), device=device)
    output_ids, _ = generation_module.generate_batch(
        input_ids,
        max_length=max_length,
        eos_token_id=eos_token_id,
        temperature=0.8,
    )

    assert output_ids.shape[1] <= max_length

    # Check EOS behavior
    eos_positions = (output_ids == eos_token_id).nonzero(as_tuple=True)
    if len(eos_positions[0]) > 0:
        # Should have padding after EOS
        first_eos_idx = eos_positions[1][0].item()
        if first_eos_idx < output_ids.shape[1] - 1:
            assert torch.all(output_ids[0, first_eos_idx + 1 :] == generation_config.pad_token_id)


@pytest.mark.parametrize("device", DEVICES)
def test_generation_module_config_build(
    transformer_config: TransformerConfig, tmp_path: Path, device: torch.device
):
    seed_all(42)

    # Create and save a generation module
    generation_config = GenerationConfig(max_length=24, temperature=0.0)
    model = transformer_config.build()
    generation_module = TransformerGenerationModule(
        model=model, generation_config=generation_config, device=device
    )

    # Generate output before saving
    input_ids = torch.randint(1, 100, (2, 4), device=device)
    output_before, _ = generation_module.generate_batch(input_ids)

    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, generation_module.model)

    # Build from config with checkpoint
    config = TransformerGenerationModuleConfig(generation_config=generation_config)
    generation_module2 = config.build(
        checkpoint_dir=checkpoint_dir,
        transformer_config=transformer_config,
        work_dir=tmp_path / "work",
        device=device,
    )

    # Generate output after loading from checkpoint
    output_after, _ = generation_module2.generate_batch(input_ids)

    # Verify predictions are the same before and after saving
    torch.testing.assert_close(output_before, output_after)
    assert output_after.shape[0] == 2
    assert output_after.shape[1] <= 24


def run_distributed_generation(
    checkpoint_dir: Path,
    transformer_config: TransformerConfig,
    generation_config: GenerationConfig,
    dp_config: Optional[TransformerDataParallelConfig],
    input_ids: torch.Tensor,
    expected_shape: tuple,
):
    seed_all(42)

    # Create generation module with parallelism config
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
    output_ids, _ = generation_module.generate_batch(input_ids, completions_only=False)

    # Basic checks
    assert (
        output_ids.shape == expected_shape
    ), f"output_ids.shape: {output_ids.shape}, expected_shape: {expected_shape}"
    assert output_ids.device == device

    # Verify all ranks got same result using FSDP
    if dp_config is not None and get_world_size() > 1:
        output_list = [torch.zeros_like(output_ids) for _ in range(get_world_size())]
        dist.all_gather(output_list, output_ids)
        for i in range(1, len(output_list)):
            torch.testing.assert_close(output_list[0], output_list[i])


@requires_multi_gpu
def test_generation_module_distributed_fsdp(transformer_config: TransformerConfig, tmp_path: Path):
    seed_all(42)

    # Create and save a generation module on single device first
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
        max_length=16, temperature=0.0, eos_token_id=2, pad_token_id=0
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
