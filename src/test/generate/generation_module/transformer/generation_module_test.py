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
from olmo_core.generate.generation_module.config import (
    GenerationConfig,
    TransformerGenerationModuleConfig,
)
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.testing import requires_multi_gpu, run_distributed_test
from olmo_core.testing.utils import has_flash_attn, requires_flash_attn, requires_gpu
from olmo_core.train.train_module.transformer.config import TransformerDataParallelConfig
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

    flash_attn_available = dtype == DType.bfloat16 and has_flash_attn
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

    # Create first generation module
    transformer_config = small_transformer_config()
    model1 = transformer_config.build()
    module1 = TransformerGenerationModule(
        model=model1, generation_config=generation_config, device=device
    )

    # Get state dict
    state_dict = module1.state_dict()
    assert "model" in state_dict

    # Create second generation module and load state dict
    model2 = transformer_config.build()
    module2 = TransformerGenerationModule(
        model=model2, generation_config=generation_config, device=device
    )
    module2.load_state_dict(state_dict)

    # Verify they produce same output
    input_ids = torch.randint(1, 100, (1, 4), device=device)
    output_ids1, _, _ = module1.generate_batch(input_ids)
    output_ids2, _, _ = module2.generate_batch(input_ids)
    torch.testing.assert_close(output_ids1, output_ids2)


@requires_gpu
@pytest.mark.parametrize("max_length", [16, 64])
@pytest.mark.parametrize("eos_token_id", [1, 2])
def test_generation_config_overrides(
    max_length: int, eos_token_id: Optional[int], device: torch.device = torch.device("cuda")
):
    seed_all(0)

    # Create generation module with default config
    generation_config = GenerationConfig(
        max_length=128, eos_token_id=1, pad_token_id=0, temperature=0.0, use_cache=False
    )

    transformer_config = small_transformer_config()
    model = transformer_config.build()
    generation_module = TransformerGenerationModule(
        model=model, generation_config=generation_config, device=device
    )

    # Generate with overrides
    input_ids = torch.randint(1, 50, (1, 4), device=device)
    output_ids, _, _ = generation_module.generate_batch(
        input_ids, max_length=max_length, eos_token_id=eos_token_id, temperature=0.8
    )

    assert output_ids.shape[1] <= max_length

    # Check EOS behavior
    eos_positions = (output_ids == eos_token_id).nonzero(as_tuple=True)
    if len(eos_positions[0]) > 0:
        # Should have padding after EOS
        first_eos_idx = eos_positions[1][0].item()
        if first_eos_idx < output_ids.shape[1] - 1:
            assert torch.all(output_ids[0, first_eos_idx + 1 :] == generation_config.pad_token_id)


@requires_gpu
def test_generation_module_config_build(tmp_path: Path):
    seed_all(0)
    device = torch.device("cuda")

    # Create and save a generation module
    generation_config = GenerationConfig(
        max_length=24, do_sample=False, pad_token_id=0, eos_token_id=2, use_cache=False
    )
    transformer_config = small_transformer_config()
    model = transformer_config.build()
    generation_module = TransformerGenerationModule(
        model=model, generation_config=generation_config, device=device
    )

    # Generate output before saving
    input_ids = torch.randint(1, 100, (2, 4), device=device)
    output_before, _, _ = generation_module.generate_batch(input_ids)

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
    output_after, _, _ = generation_module2.generate_batch(input_ids)

    # Verify predictions are the same before and after saving
    torch.testing.assert_close(output_before, output_after)
    assert output_after.shape[0] == 2
    assert output_after.shape[1] <= 24


@requires_gpu
def test_generation_module_stop_sequences():
    seed_all(0)
    device = torch.device("cuda")

    # Create generation config with individual stop tokens
    generation_config = GenerationConfig(
        max_length=50,
        do_sample=False,
        pad_token_id=0,
        eos_token_id=1,
        stop_token_ids=[10, 20, 30, 40, 50],
        use_cache=False,
    )

    # Create model
    transformer_config = small_transformer_config()
    model = transformer_config.build()
    generation_module = TransformerGenerationModule(
        model=model, generation_config=generation_config, device=device
    )

    def create_mock_forward(tokens_to_generate):
        def mock_forward(input_ids: torch.Tensor, **kwargs):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1] - 3  # Subtract initial input length
            token = tokens_to_generate[seq_len] if seq_len < len(tokens_to_generate) else 99
            print(f"Generating token {token} for input_ids {input_ids.tolist()}")
            logits = torch.zeros(batch_size, 1, transformer_config.vocab_size, device=device)
            logits[:, 0, token] = 100.0  # High logit for desired token
            return logits

        return mock_forward

    # Stop at first stop token (10)
    input_ids = torch.tensor([[3, 5, 7]], dtype=torch.long, device=device)
    generation_module.model.forward = create_mock_forward([8, 9, 10, 99])
    output, _, _ = generation_module.generate_batch(input_ids, completions_only=False)
    assert torch.equal(output, torch.tensor([[3, 5, 7, 8, 9, 10]], device=device))

    # Stop at second stop token (20)
    input_ids = torch.tensor([[2, 4, 6]], dtype=torch.long, device=device)
    generation_module.model.forward = create_mock_forward([25, 15, 20, 99])
    output, _, _ = generation_module.generate_batch(input_ids, completions_only=False)
    assert torch.equal(output, torch.tensor([[2, 4, 6, 25, 15, 20]], device=device))

    # Stop at third stop token (30)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)
    generation_module.model.forward = create_mock_forward([5, 30, 99])
    output, _, _ = generation_module.generate_batch(input_ids, completions_only=False)
    assert torch.equal(output, torch.tensor([[1, 2, 3, 5, 30]], device=device))

    # Stop at EOS token (not stop token)
    input_ids = torch.tensor([[3, 5, 7]], dtype=torch.long, device=device)
    generation_module.model.forward = create_mock_forward([60, 70, 1, 99])
    output, _, _ = generation_module.generate_batch(input_ids, completions_only=False)
    assert torch.equal(output, torch.tensor([[3, 5, 7, 60, 70, 1]], device=device))

    # No stop tokens - only stops at EOS
    generation_module._generation_config = GenerationConfig(
        max_length=20,
        do_sample=False,
        pad_token_id=0,
        eos_token_id=1,
        stop_token_ids=None,
        use_cache=False,
    )
    input_ids = torch.tensor([[3, 5, 7]], dtype=torch.long, device=device)
    generation_module.model.forward = create_mock_forward([10, 20, 30, 40, 50, 1])
    output, *_ = generation_module.generate_batch(input_ids, completions_only=False)
    assert torch.equal(output, torch.tensor([[3, 5, 7, 10, 20, 30, 40, 50, 1]], device=device))


@requires_gpu
@requires_flash_attn
def test_generation_with_attention_mask():
    device = torch.device("cuda")
    pad_token_id = 0

    generation_config = GenerationConfig(
        max_length=20, temperature=0.0, pad_token_id=pad_token_id, eos_token_id=1
    )

    transformer_config = small_transformer_config(use_flash=True, dtype=DType.bfloat16)
    model = transformer_config.build()
    generation_module = TransformerGenerationModule(
        model=model, generation_config=generation_config, device=device
    )

    # Create left-padded input
    input_ids = torch.tensor(
        [[pad_token_id, pad_token_id, 3, 5, 7]], dtype=torch.long, device=device
    )

    # Create two different attention masks
    attention_mask1 = (input_ids != pad_token_id).to(torch.bool)
    attention_mask2 = attention_mask1.clone()
    attention_mask2[0, 2] = False  # Mask the first non-pad token (as if it were padding)

    # Generate with different attention masks
    output_with_mask1, _, _ = generation_module.generate_batch(
        input_ids, attention_mask=attention_mask1, completions_only=False
    )
    output_with_mask2, _, _ = generation_module.generate_batch(
        input_ids, attention_mask=attention_mask2, completions_only=False
    )

    # Different attention masks should produce different outputs
    assert not torch.equal(output_with_mask1, output_with_mask2), (
        "Different attention masks should affect generation output"
    )


@requires_gpu
@requires_flash_attn
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
    model = transformer_config.build()
    generation_module = TransformerGenerationModule(
        model=model, generation_config=generation_config, device=device
    )

    unpadded = torch.tensor([[3, 5, 7]], dtype=torch.long, device=device)
    left_padded = torch.tensor(
        [[pad_token_id, pad_token_id, 3, 5, 7]], dtype=torch.long, device=device
    )
    pseudo_left_padded = torch.tensor([[2, 2, 3, 5, 7]], dtype=torch.long, device=device)

    attn_unpadded = (unpadded != pad_token_id).to(torch.bool)
    attn_left_padded = (left_padded != pad_token_id).to(torch.bool)

    seed_all(0)
    out_ids_unpadded, logits_unpadded, _ = generation_module.generate_batch(
        unpadded, attention_mask=attn_unpadded, completions_only=True, return_logits=True
    )

    seed_all(0)
    out_ids_left, logits_left, _ = generation_module.generate_batch(
        left_padded, attention_mask=attn_left_padded, completions_only=True, return_logits=True
    )

    seed_all(0)
    out_ids_pseudo_left, logits_pseudo_left, _ = generation_module.generate_batch(
        pseudo_left_padded,
        attention_mask=attn_left_padded,  # reuse the left-padded attention mask
        completions_only=True,
        return_logits=True,
    )

    assert torch.equal(out_ids_unpadded, out_ids_left)
    assert torch.equal(out_ids_left, out_ids_pseudo_left)

    torch.testing.assert_close(logits_unpadded, logits_left, rtol=BF16_RTOL, atol=BF16_ATOL)
    torch.testing.assert_close(logits_left, logits_pseudo_left, rtol=BF16_RTOL, atol=BF16_ATOL)


@requires_gpu
@requires_flash_attn
@pytest.mark.parametrize("batch_size", [1, 8])
def test_generation_cache_consistency(batch_size: int):
    if not has_flash_attn:
        pytest.skip("flash-attn is required for KV cache usage")

    device = torch.device("cuda")

    transformer_config = small_transformer_config(dtype=DType.bfloat16, n_layers=1, use_flash=True)
    model = transformer_config.build()

    gen_config = GenerationConfig(
        max_length=128, do_sample=False, pad_token_id=0, eos_token_id=1, use_cache=False
    )
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

    output_ids, _, _ = generation_module.generate_batch(input_ids, completions_only=False)

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
def test_generation_module_distributed_fsdp(tmp_path: Path, use_cache: bool):
    seed_all(0)

    if not has_flash_attn and use_cache:
        pytest.skip("flash-attn is required for use_cache")

    # Create and save a generation module on single device first
    generation_config = GenerationConfig(
        max_length=16, do_sample=False, pad_token_id=0, eos_token_id=1, use_cache=use_cache
    )
    transformer_config = small_transformer_config(dtype=DType.bfloat16, use_flash=has_flash_attn)
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
