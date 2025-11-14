from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.distributed as dist

from olmo_core.config import DType
from olmo_core.distributed.checkpoint import save_model_and_optim_state
from olmo_core.distributed.parallel.data_parallel import DataParallelType
from olmo_core.distributed.utils import get_world_size
from olmo_core.generate.generation_module import TransformerGenerationModule
from olmo_core.generate.generation_module.config import GenerationConfig
from olmo_core.generate.generation_module.transformer.config import (
    TransformerGenerationModuleConfig,
)
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.testing import requires_multi_gpu, run_distributed_test
from olmo_core.testing.utils import (
    has_flash_attn_2,
    requires_flash_attn_2,
    requires_gpu,
)
from olmo_core.train.train_module.transformer.config import (
    TransformerDataParallelConfig,
)
from olmo_core.utils import seed_all

BF16_RTOL = 1e-5
BF16_ATOL = 5e-3


def small_transformer_config(n_layers: int = 2, use_rope: bool = True, **kwargs):
    config = TransformerConfig.llama_like(
        d_model=128, n_heads=4, n_layers=n_layers, vocab_size=512, **kwargs
    )
    if not use_rope:
        config.block.attention.rope = None
    return config


@requires_gpu
@pytest.mark.parametrize(
    "compile_model",
    [pytest.param(False, id="compile_model=False"), pytest.param(True, id="compile_model=True")],
)
@pytest.mark.parametrize(
    "use_cache",
    [pytest.param(False, id="use_cache=False"), pytest.param(True, id="use_cache=True")],
)
def test_generation_module_basic(compile_model: bool, use_cache: bool):
    device = torch.device("cuda")
    dtype = DType.bfloat16
    seed_all(0)

    flash_attn_available = dtype == DType.bfloat16 and has_flash_attn_2
    if not flash_attn_available and use_cache:
        pytest.skip("flash-attn is required for use_cache")

    generation_config = GenerationConfig(
        use_cache=use_cache,
        max_length=20,
        eos_token_id=1,
        pad_token_id=0,
        do_sample=False,
    )

    # Build generation module
    transformer_config = small_transformer_config(use_flash=flash_attn_available, dtype=dtype)
    model = transformer_config.build()
    generation_module = TransformerGenerationModule(
        model=model,
        generation_config=generation_config,
        compile_model=compile_model,
        device=device,
    )

    # Create test input
    batch_size, context_len = 2, 16
    input_ids = torch.randint(2, 100, (batch_size, context_len), device=device)
    attention_mask = torch.ones(batch_size, context_len, device=device, dtype=torch.bool)

    output_ids, output_logits, output_logprobs = generation_module.generate_batch(  # type: ignore
        input_ids,
        attention_mask=attention_mask,
        return_logits=True,
        return_logprobs=True,
        completions_only=False,
    )

    # TODO: test to make sure the model is compiled exactly once if compile_model is True
    # We do not want to accidentally compile the model each time the sequence length changes.

    # Verify output shape and properties
    assert output_ids.shape[0] == batch_size, "output batch size does not match input batch size"
    if generation_config.max_length is not None:
        assert output_ids.shape[1] <= generation_config.max_length, "output_ids too long"
    assert output_ids.shape[1] >= context_len + 1, "no new tokens generated"
    assert torch.all(output_ids[:, :context_len] == input_ids), "input_ids not preserved"

    assert isinstance(output_logits, torch.Tensor), "output_logits is not a tensor"
    assert output_logits.shape == (
        batch_size,
        output_ids.shape[1] - context_len,
        transformer_config.vocab_size,
    ), "output_logits shape does not match expected shape"

    assert isinstance(output_logprobs, torch.Tensor), "output_logprobs is not a tensor"
    assert output_logprobs.shape == (
        batch_size,
        output_ids.shape[1] - context_len,
    ), "output_logprobs shape does not match expected shape: " + str(output_logprobs.shape)

    # Check that generation stopped at EOS or max_length
    for i in range(batch_size):
        seq = output_ids[i]
        eos_positions = (seq == generation_config.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            # If EOS found, check padding after it
            first_eos = eos_positions[0].item()
            if first_eos < output_ids.shape[1] - 1:
                assert torch.all(seq[first_eos + 1 :] == generation_config.pad_token_id), (
                    f"padding not added after EOS. Sequence {i}: {seq.tolist()}, "
                    f"EOS at position {first_eos}, expected padding after position {first_eos}, "
                    f"but found tokens: {seq[first_eos + 1 :].tolist()}"
                )


@requires_gpu
def test_generation_module_state_dict():
    seed_all(0)
    device = torch.device("cuda")
    generation_config = GenerationConfig(
        max_length=16, pad_token_id=0, eos_token_id=2, use_cache=False
    )
    transformer_config = small_transformer_config()

    # Init two modules with same config
    module1 = TransformerGenerationModule(
        transformer_config.build(), generation_config, device=device
    )
    module2 = TransformerGenerationModule(
        transformer_config.build(), generation_config, device=device
    )

    # Load state dict from first to second
    module2.load_state_dict(module1.state_dict())

    # Verify same output
    input_ids = torch.randint(1, 100, (1, 4), device=device)
    output1, _, _ = module1.generate_batch(input_ids)
    output2, _, _ = module2.generate_batch(input_ids)
    torch.testing.assert_close(output1, output2)


@requires_gpu
@pytest.mark.parametrize("max_length", [16, 64])
@pytest.mark.parametrize("eos_token_id", [1, 2])
def test_generation_config_overrides(max_length: int, eos_token_id: int):
    seed_all(0)
    device = torch.device("cuda")

    generation_config = GenerationConfig(
        max_length=128, eos_token_id=1, pad_token_id=0, use_cache=False
    )
    generation_module = TransformerGenerationModule(
        model=small_transformer_config().build(), generation_config=generation_config, device=device
    )

    input_ids = torch.randint(1, 50, (1, 4), device=device)
    output_ids, _, _ = generation_module.generate_batch(
        input_ids, max_length=max_length, eos_token_id=eos_token_id
    )

    assert output_ids.shape[1] <= max_length

    # Check EOS padding
    eos_positions = (output_ids == eos_token_id).nonzero(as_tuple=True)
    if len(eos_positions[0]) > 0:
        first_eos_idx = eos_positions[1][0].item()
        if first_eos_idx < output_ids.shape[1] - 1:
            assert torch.all(output_ids[0, first_eos_idx + 1 :] == generation_config.pad_token_id)


@requires_gpu
def test_generation_module_config_build(tmp_path: Path):
    seed_all(0)
    device = torch.device("cuda")

    generation_config = GenerationConfig(
        max_length=24, do_sample=False, pad_token_id=0, eos_token_id=2, use_cache=False
    )
    transformer_config = small_transformer_config()

    # Create and save generation module
    generation_module = TransformerGenerationModule(
        model=transformer_config.build(), generation_config=generation_config, device=device
    )

    input_ids = torch.randint(1, 100, (2, 4), device=device)
    output_before, _, _ = generation_module.generate_batch(input_ids)

    # Save and rebuild from checkpoint
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, generation_module.model)

    config = TransformerGenerationModuleConfig(generation_config=generation_config)
    generation_module2 = config.build(
        checkpoint_dir=checkpoint_dir,
        transformer_config=transformer_config,
        work_dir=tmp_path / "work",
        device=device,
    )

    output_after, _, _ = generation_module2.generate_batch(input_ids)

    # Verify same outputs
    torch.testing.assert_close(output_before, output_after)
    assert output_after.shape == (2, output_after.shape[1])
    assert output_after.shape[1] <= 24


@requires_gpu
def test_generation_module_stop_sequences():
    seed_all(0)
    device = torch.device("cuda")

    # Create generation config with stop tokens
    generation_config = GenerationConfig(
        max_length=20,
        pad_token_id=0,
        eos_token_id=1,
        stop_token_ids=[10, 20],
        use_cache=False,
    )

    # Create model
    transformer_config = small_transformer_config()
    model = transformer_config.build()
    generation_module = TransformerGenerationModule(
        model=model, generation_config=generation_config, device=device
    )

    def create_mock_forward(tokens_to_generate):
        def mock_forward(input_ids: torch.Tensor, **_kwargs):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1] - 3  # Subtract initial input length
            token = tokens_to_generate[seq_len] if seq_len < len(tokens_to_generate) else 99
            logits = torch.zeros(batch_size, 1, transformer_config.vocab_size, device=device)
            logits[:, 0, token] = 100.0  # High logit for desired token
            return logits

        return mock_forward

    # Test stop at stop token
    input_ids = torch.tensor([[3, 5, 7]], dtype=torch.long, device=device)
    generation_module.model.forward = create_mock_forward([8, 10, 99])  # type: ignore[method-assign]
    output, _, _ = generation_module.generate_batch(input_ids, completions_only=False)
    assert torch.equal(output, torch.tensor([[3, 5, 7, 8, 10]], device=device))

    # Test stop at EOS token
    generation_module.model.forward = create_mock_forward([8, 1, 99])  # type: ignore[method-assign]
    output, _, _ = generation_module.generate_batch(input_ids, completions_only=False)
    assert torch.equal(output, torch.tensor([[3, 5, 7, 8, 1]], device=device))


@requires_gpu
@requires_flash_attn_2
def test_generation_with_attention_mask():
    device = torch.device("cuda")
    pad_token_id = 0

    generation_module = TransformerGenerationModule(
        model=small_transformer_config(use_flash=True, dtype=DType.bfloat16).build(),
        generation_config=GenerationConfig(
            max_length=20, temperature=0.0, pad_token_id=pad_token_id, eos_token_id=1
        ),
        device=device,
    )

    input_ids = torch.tensor(
        [[pad_token_id, pad_token_id, 3, 5, 7]], dtype=torch.long, device=device
    )

    # Test with different attention masks
    mask1 = (input_ids != 0).to(torch.bool)
    mask2 = mask1.clone()
    mask2[0, 2] = False  # Mask first non-pad token

    output1, _, _ = generation_module.generate_batch(
        input_ids,
        attention_mask=mask1,
        log_timing=True,  # just to check that log_timing works too
    )
    output2, _, _ = generation_module.generate_batch(input_ids, attention_mask=mask2)

    assert not torch.equal(output1, output2), "Using different attention masks should affect output"


@requires_gpu
@requires_flash_attn_2
@pytest.mark.parametrize("use_rope", [True, False], ids=["rope", "no-rope"])
def test_left_padded_attention_mask_equivalence(use_rope):
    device = torch.device("cuda")
    pad_token_id = 0

    generation_config = GenerationConfig(
        max_new_tokens=8, temperature=0.0, pad_token_id=pad_token_id, eos_token_id=1, use_cache=True
    )

    transformer_config = small_transformer_config(
        use_flash=True, n_layers=2, dtype=DType.bfloat16, use_rope=use_rope
    )
    generation_module = TransformerGenerationModule(
        model=transformer_config.build(), generation_config=generation_config, device=device
    )

    # Test inputs: unpadded vs left-padded sequences
    unpadded = torch.tensor([[3, 5, 7]], dtype=torch.long, device=device)
    left_padded = torch.tensor(
        [[pad_token_id, pad_token_id, 3, 5, 7]], dtype=torch.long, device=device
    )

    attn_unpadded = (unpadded != pad_token_id).to(torch.bool)
    attn_left_padded = (left_padded != pad_token_id).to(torch.bool)

    # Generate with both inputs using same seed
    seed_all(0)
    out_ids_unpadded, logits_unpadded, _ = generation_module.generate_batch(
        unpadded, attention_mask=attn_unpadded, completions_only=True, return_logits=True
    )

    seed_all(0)
    out_ids_left, logits_left, _ = generation_module.generate_batch(
        left_padded, attention_mask=attn_left_padded, completions_only=True, return_logits=True
    )

    # Verify equivalence
    assert torch.equal(out_ids_unpadded, out_ids_left)
    torch.testing.assert_close(logits_unpadded, logits_left, rtol=BF16_RTOL, atol=BF16_ATOL)


@requires_gpu
@requires_flash_attn_2
@pytest.mark.parametrize("batch_size", [1, 8])
def test_generation_cache_consistency(batch_size: int):
    if not has_flash_attn_2:
        pytest.skip("flash-attn is required for KV cache usage")

    device = torch.device("cuda")
    model = small_transformer_config(dtype=DType.bfloat16, n_layers=1, use_flash=True).build()
    gen_config = GenerationConfig(max_length=128, pad_token_id=0, eos_token_id=1, use_cache=False)
    generation_module = TransformerGenerationModule(
        model=model, generation_config=gen_config, device=device
    )

    seq_len = 124
    input_ids = torch.randint(2, 100, (batch_size, seq_len), device=device)

    seed_all(0)
    output_ids_no_cache, output_logits_no_cache, _ = generation_module.generate_batch(
        input_ids, completions_only=True, return_logits=True, use_cache=False
    )
    seed_all(0)
    output_ids_with_cache, output_logits_with_cache, _ = generation_module.generate_batch(
        input_ids, completions_only=True, return_logits=True, use_cache=True
    )

    assert torch.equal(output_ids_no_cache, output_ids_with_cache)
    torch.testing.assert_close(output_logits_no_cache, output_logits_with_cache)


def run_distributed_generation(
    checkpoint_dir: Path,
    transformer_config: TransformerConfig,
    generation_config: GenerationConfig,
    dp_config: Optional[TransformerDataParallelConfig],
    input_ids: torch.Tensor,
    expected_shape: tuple,
    attention_mask: Optional[torch.Tensor] = None,
):
    seed_all(0)

    generation_module = TransformerGenerationModule.from_checkpoint(
        transformer_config=transformer_config,
        checkpoint_dir=checkpoint_dir,
        generation_config=generation_config,
        dp_config=dp_config,
    )

    device = torch.device("cuda", dist.get_rank())
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    output_ids, _, _ = generation_module.generate_batch(
        input_ids, attention_mask=attention_mask, completions_only=False
    )

    assert output_ids.shape == expected_shape
    assert output_ids.device == device

    # Verify all ranks got same result using FSDP
    if dp_config is not None and get_world_size() > 1:
        output_list = [torch.zeros_like(output_ids) for _ in range(get_world_size())]
        dist.all_gather(output_list, output_ids)
        for i in range(1, len(output_list)):
            torch.testing.assert_close(output_list[0], output_list[i])


@requires_multi_gpu
@pytest.mark.parametrize("use_cache", [True, False], ids=["with_cache", "without_cache"])
@pytest.mark.parametrize(
    "use_attention_mask", [True, False], ids=["with_attn_mask", "without_attn_mask"]
)
def test_generation_module_distributed_fsdp(
    tmp_path: Path, use_cache: bool, use_attention_mask: bool
):
    seed_all(0)

    if not has_flash_attn_2 and use_cache:
        pytest.skip("flash-attn is required for use_cache")
    if use_attention_mask and not use_cache:
        pytest.skip("attention mask test is only valid with use_cache=True")

    # Create and save a generation module on single device first
    generation_config = GenerationConfig(
        max_length=16, do_sample=False, pad_token_id=0, eos_token_id=1, use_cache=use_cache
    )
    transformer_config = small_transformer_config(dtype=DType.bfloat16, use_flash=has_flash_attn_2)
    model = transformer_config.build()
    generation_module = TransformerGenerationModule(
        model=model, generation_config=generation_config, device=torch.device("cuda")
    )

    # Save checkpoint
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, generation_module.model)

    # Prepare configs for distributed test
    dp_config = TransformerDataParallelConfig(name=DataParallelType.fsdp)

    # Create test input
    input_ids = torch.randint(2, 100, (2, 8))
    expected_shape = (2, 16)  # max_length

    attention_mask = None
    if use_attention_mask:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

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
            attention_mask,
        ),
    )


def create_test_checkpoint_for_merging(
    checkpoint_dir: Path, seed: int, transformer_config: TransformerConfig
) -> None:
    """Helper function to create a test checkpoint with a specific seed."""
    seed_all(seed)
    model = transformer_config.build()
    generation_config = GenerationConfig(max_new_tokens=10, pad_token_id=0, eos_token_id=1)
    generation_module = TransformerGenerationModule(
        model=model, generation_config=generation_config, device="cpu"
    )
    save_model_and_optim_state(checkpoint_dir, generation_module.model)

    # Save config.json so from_checkpoint can load the config
    import json

    config_path = checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        # from_checkpoint expects the config to be nested under "model" key
        # and also needs a "dataset" with "tokenizer" config
        config_dict = {
            "model": transformer_config.as_dict(),
            "dataset": {
                "tokenizer": {
                    "identifier": "dummy",
                    "vocab_size": transformer_config.vocab_size,
                    "eos_token_id": 1,
                    "pad_token_id": 0,
                }
            },
        }
        json.dump(config_dict, f)


def test_from_checkpoints_single_checkpoint(tmp_path: Path):
    """Test that from_checkpoints with a single checkpoint behaves like from_checkpoint."""
    config = small_transformer_config(n_layers=2)
    checkpoint_dir = tmp_path / "checkpoint"
    create_test_checkpoint_for_merging(checkpoint_dir, seed=42, transformer_config=config)

    # Load using from_checkpoints with single checkpoint
    model_merged = TransformerGenerationModule.from_checkpoints([checkpoint_dir], device="cpu")

    # Load using from_checkpoint directly
    model_direct = TransformerGenerationModule.from_checkpoint(checkpoint_dir, device="cpu")

    # Verify state dicts are identical
    merged_state = model_merged.model.state_dict()
    direct_state = model_direct.model.state_dict()

    assert set(merged_state.keys()) == set(direct_state.keys())
    for key in merged_state.keys():
        torch.testing.assert_close(merged_state[key], direct_state[key])


def test_from_checkpoints_weight_averaging(tmp_path: Path):
    """Test that weights are correctly averaged across all tensors with proper shape and dtype preservation."""
    config = small_transformer_config(n_layers=2)

    # Create three checkpoints with different seeds to test averaging
    checkpoint_dirs = []
    for i, seed in enumerate([42, 43, 44]):
        checkpoint_dir = tmp_path / f"checkpoint{i}"
        create_test_checkpoint_for_merging(checkpoint_dir, seed=seed, transformer_config=config)
        checkpoint_dirs.append(checkpoint_dir)

    # Load individual models
    models = [
        TransformerGenerationModule.from_checkpoint(checkpoint_dir, device="cpu")
        for checkpoint_dir in checkpoint_dirs
    ]

    # Merge the checkpoints (without specifying dtype, should default to float32)
    merged_model = TransformerGenerationModule.from_checkpoints(checkpoint_dirs, device="cpu")

    # Get state dicts
    state_dicts = [model.model.state_dict() for model in models]
    merged_state = merged_model.model.state_dict()

    # Verify all keys are present and shapes match
    assert set(merged_state.keys()) == set(state_dicts[0].keys())

    # Check that all tensors are correctly averaged with proper shapes and dtypes
    for key in merged_state.keys():
        # Verify shapes match across all checkpoints and merged model
        for state_dict in state_dicts:
            assert state_dict[key].shape == merged_state[key].shape, f"Shape mismatch for {key}"

        # Verify dtype is float32 (default) for floating point tensors
        if merged_state[key].is_floating_point():
            assert (
                merged_state[key].dtype == torch.float32
            ), f"Expected float32 for {key}, got {merged_state[key].dtype}"

        # Calculate expected average
        tensors = [state_dict[key].float() for state_dict in state_dicts]
        expected_avg = torch.stack(tensors).mean(dim=0)

        # Verify the merged weights match the expected average
        torch.testing.assert_close(
            merged_state[key].float(),
            expected_avg,
            rtol=1e-5,
            atol=1e-5,
            msg=f"Weight averaging failed for tensor {key}",
        )


def test_from_checkpoints_with_bfloat16(tmp_path: Path):
    """Test merging with bfloat16 dtype conversion."""
    config = small_transformer_config(n_layers=2)

    # Create two checkpoints
    checkpoint_dir1 = tmp_path / "checkpoint1"
    checkpoint_dir2 = tmp_path / "checkpoint2"
    create_test_checkpoint_for_merging(checkpoint_dir1, seed=42, transformer_config=config)
    create_test_checkpoint_for_merging(checkpoint_dir2, seed=43, transformer_config=config)

    # Merge with bfloat16 dtype
    merged_model = TransformerGenerationModule.from_checkpoints(
        [checkpoint_dir1, checkpoint_dir2], dtype=DType.bfloat16, device="cpu"
    )

    # Check that parameters are bfloat16
    for param in merged_model.model.parameters():
        if param.is_floating_point():
            assert param.dtype == torch.bfloat16


def test_from_checkpoints_produces_valid_model(tmp_path: Path):
    """Test that the merged model produces valid output."""
    config = small_transformer_config(n_layers=2)

    # Create two checkpoints with different seeds
    checkpoint_dir1 = tmp_path / "checkpoint1"
    checkpoint_dir2 = tmp_path / "checkpoint2"
    create_test_checkpoint_for_merging(checkpoint_dir1, seed=42, transformer_config=config)
    create_test_checkpoint_for_merging(checkpoint_dir2, seed=43, transformer_config=config)

    # Merge both checkpoints
    merged_model = TransformerGenerationModule.from_checkpoints(
        [checkpoint_dir1, checkpoint_dir2], device="cpu"
    )

    # Verify the merged model can do a forward pass without errors
    input_ids = torch.randint(0, config.vocab_size, (2, 10), device="cpu")
    with torch.no_grad():
        output = merged_model.model(input_ids)

    # Verify output shape and validity
    assert output.shape == (2, 10, config.vocab_size)
    assert torch.isfinite(output).all()  # No NaN or Inf values


@requires_gpu
def test_from_checkpoints_can_generate(tmp_path: Path):
    """Test that the merged model can generate text."""
    config = small_transformer_config(n_layers=2)

    # Create two checkpoints with different seeds
    checkpoint_dir1 = tmp_path / "checkpoint1"
    checkpoint_dir2 = tmp_path / "checkpoint2"
    create_test_checkpoint_for_merging(checkpoint_dir1, seed=42, transformer_config=config)
    create_test_checkpoint_for_merging(checkpoint_dir2, seed=43, transformer_config=config)

    # Merge both checkpoints
    merged_model = TransformerGenerationModule.from_checkpoints(
        [checkpoint_dir1, checkpoint_dir2], device=torch.device("cuda")
    )

    # Verify the merged model can generate text
    input_ids = torch.randint(0, config.vocab_size, (2, 10), device="cuda")
    gen_output, _, _ = merged_model.generate_batch(input_ids, max_length=15, use_cache=False)

    # Verify generation output
    assert gen_output.shape[0] == 2  # batch size
    assert gen_output.shape[1] >= 10  # at least as long as input
    assert gen_output.shape[1] <= 15  # input length (10) + 5 new tokens
    assert torch.isfinite(gen_output.float()).all()  # No NaN or Inf


def test_from_checkpoints_empty_list_error(tmp_path: Path):
    """Test that passing an empty checkpoint list raises an error."""
    with pytest.raises((ValueError, AssertionError, IndexError)):
        TransformerGenerationModule.from_checkpoints([], device="cpu")
