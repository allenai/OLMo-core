import logging
from dataclasses import replace
from typing import Optional, cast

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor, Shard, init_device_mesh

from olmo_core.config import DType
from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.parallel import (
    DataParallelConfig,
    DataParallelType,
    build_world_mesh,
)
from olmo_core.distributed.utils import get_full_tensor, get_world_size
from olmo_core.nn.attention import (
    AttentionConfig,
    RingAttentionLoadBalancerType,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNorm, LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe import MoEConfig, MoERouterConfig, MoEType
from olmo_core.nn.rope import RoPEConfig
from olmo_core.nn.transformer import (
    MoEHybridTransformerBlockBase,
    MoEReorderedNormTransformerBlock,
    MoETransformer,
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.testing import (
    BACKENDS,
    GPU_MARKS,
    requires_flash_attn_2,
    requires_multi_gpu,
    run_distributed_test,
)
from olmo_core.utils import get_default_device

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
    model = config.build(init_device=init_device)
    model.init_weights(device=torch.device(device))

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
    assert model.blocks["0"].block_idx == 0
    assert model.blocks[str(len(model.blocks) - 1)].block_idx == len(model.blocks) - 1


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
    model = config.build(init_device=init_device)
    model.init_weights(device=torch.device(device))

    # Make sure num params estimate is correct.
    num_actual_params = 0
    for _, p in model.named_parameters():
        num_actual_params += p.numel()
    assert config.num_params == num_actual_params
    assert model.num_params == num_actual_params

    # Make sure block_idx is set correctly.
    assert model.blocks["0"].block_idx == 0
    assert model.blocks[str(len(model.blocks) - 1)].block_idx == len(model.blocks) - 1

    # Make sure all weights are normalized in the embedding dimension.
    check_ngpt_matrices(model, config.d_model)


def run_ngpt_with_fsdp2():
    config = TransformerConfig.ngpt_271M(vocab_size=50257)
    model = config.build(
        init_device="meta",
    )
    model.apply_fsdp()
    model.init_weights(max_seq_len=1024, device=get_default_device())
    optim = torch.optim.Adam(model.parameters())

    # Take an optimizer step.
    model(input_ids=torch.randint(0, 50257, (2, 128))).sum().backward()
    optim.step()

    # Re-normalize weights.
    model.normalize_matrices()  # type: ignore

    # Check that the re-normalization was successful.
    check_ngpt_matrices(model, config.d_model)


@requires_multi_gpu
def test_ngpt_with_fsdp2():
    run_distributed_test(run_ngpt_with_fsdp2, backend="nccl", start_method="spawn")


def get_transformer_config(
    architecture: str,
    dtype: torch.dtype = torch.float32,
    swa: Optional[SlidingWindowAttentionConfig] = None,
) -> TransformerConfig:
    config: TransformerConfig
    if architecture == "olmo2":
        config = TransformerConfig.olmo2_190M(
            vocab_size=16_000,
            n_layers=2,
            fused_ops=False,
            use_flash=False,
            dtype=DType.from_pt(dtype),
        )
    elif architecture == "llama":
        config = TransformerConfig.llama2_271M(
            vocab_size=16_000,
            n_layers=2,
            fused_ops=False,
            use_flash=False,
            dtype=DType.from_pt(dtype),
        )
    else:
        raise NotImplementedError(architecture)

    if swa is not None:
        config.block.attention.sliding_window = swa
        config.block.attention.use_flash = True

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

    model = config.build()
    model.apply_tp(mesh["tp"])
    model.init_weights(device=device, max_seq_len=512)
    load_model_and_optim_state(checkpoint_dir, model)

    logits = model(input_ids=input_ids)

    loss = logits.sum()
    loss.backward()

    og_logits = torch.load(outputs_path, map_location=device)
    torch.testing.assert_close(og_logits, get_full_tensor(logits))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("architecture", ["olmo2", "llama"])
def test_tensor_parallel_transformer(backend: str, architecture: str, tmp_path):
    device = torch.device("cuda") if "nccl" in backend else torch.device("cpu")
    config = get_transformer_config(architecture)
    model = config.build()
    model.init_weights(device=device, max_seq_len=512)
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


def run_context_parallel_transformer(checkpoint_dir, outputs_path, architecture: str):
    device = get_default_device()
    config = get_transformer_config(architecture, dtype=torch.bfloat16)
    config.block.attention.use_flash = True

    mesh = init_device_mesh(
        device.type,
        (get_world_size(),),
        mesh_dim_names=("cp",),
    )

    model = config.build()
    model.apply_cp(mesh["cp"], RingAttentionLoadBalancerType.zig_zag)
    model.init_weights(device=device, max_seq_len=512)
    load_model_and_optim_state(checkpoint_dir, model)

    input_ids = get_transformer_inputs().to(device)
    local_logits = model(input_ids=input_ids)
    logits = DTensor.from_local(local_logits, mesh, (Shard(1),))

    og_logits = torch.load(outputs_path, map_location=device)
    torch.testing.assert_close(og_logits, get_full_tensor(logits))


@requires_multi_gpu
@requires_flash_attn_2
@pytest.mark.parametrize("architecture", ["olmo2"])
@pytest.mark.skip("known precision issues with ring-flash-attn")
def test_context_parallel_transformer(architecture: str, tmp_path):
    device = torch.device("cuda")
    config = get_transformer_config(architecture, dtype=torch.bfloat16)
    config.block.attention.use_flash = True

    model = config.build()
    model.init_weights(device=device, max_seq_len=512)
    input_ids = get_transformer_inputs().to(device)
    logits = model(input_ids=input_ids)

    outputs_path = tmp_path / "logits.pt"
    torch.save(logits, outputs_path)

    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, model)

    run_distributed_test(
        run_context_parallel_transformer,
        backend="nccl",
        start_method="spawn",
        func_args=(
            checkpoint_dir,
            outputs_path,
            architecture,
        ),
    )


def run_init_with_hsdp():
    assert dist.get_world_size() == 4
    mesh = build_world_mesh(
        dp=DataParallelConfig(name=DataParallelType.hsdp, shard_degree=2, num_replicas=2)
    )
    config = get_transformer_config("olmo2")
    model = config.build(init_device="meta")
    model.apply_fsdp(mesh)
    model.init_weights(max_seq_len=512, device=get_default_device())

    # Check that params across all replica groups are exactly the same.
    for name, param in model.named_parameters():
        full_param = get_full_tensor(param).detach()
        full_param_avg = full_param / 4
        dist.all_reduce(full_param_avg)
        torch.testing.assert_close(
            full_param_avg,
            full_param,
            msg=f"parameter '{name}' is inconsistent across the process group",
        )


@requires_multi_gpu
def test_init_with_hsdp():
    if torch.cuda.device_count() < 4:
        pytest.skip("Requires 4 GPUs")

    run_distributed_test(
        run_init_with_hsdp,
        backend="nccl",
        start_method="spawn",
        world_size=4,
    )


def run_moe_hybrid_combined_forward(
    dropless: bool, shared_experts: bool, reordered_norm: bool, tp: bool
):
    layer_norm = LayerNormConfig(name=LayerNormType.rms, bias=False)
    config = TransformerConfig(
        name=TransformerType.moe,
        d_model=512,
        vocab_size=16_000,
        n_layers=2,
        block=TransformerBlockConfig(
            name=(
                TransformerBlockType.moe_hybrid_reordered_norm
                if reordered_norm
                else TransformerBlockType.moe_hybrid
            ),
            attention=AttentionConfig(n_heads=8, rope=RoPEConfig(), qk_norm=layer_norm),
            layer_norm=layer_norm,
            feed_forward=FeedForwardConfig(hidden_size=1024, bias=False),
            feed_forward_moe=MoEConfig(
                name=MoEType.dropless if dropless else MoEType.default,
                num_experts=4,
                hidden_size=256,
                shared_mlp=(
                    FeedForwardConfig(hidden_size=512, bias=False) if shared_experts else None
                ),
                router=MoERouterConfig(uniform_expert_assignment=True),
            ),
        ),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False),
    )

    device = get_default_device()
    model = config.build(init_device=device.type)
    assert isinstance(model, MoETransformer)
    mesh = init_device_mesh(
        device.type,
        (get_world_size(),),
        mesh_dim_names=("tp" if tp else "ep",),
    )

    if tp:
        model.apply_tp(mesh["tp"])
    else:
        model.apply_ep(mesh["ep"])

    input_ids = get_transformer_inputs().to(device)
    model.init_weights(device=device, max_seq_len=512, max_local_microbatch_size=input_ids.numel())

    for block in model.blocks.values():
        cast(MoEHybridTransformerBlockBase, block).use_combined_forward = False
    output1 = model(input_ids)

    for block in model.blocks.values():
        cast(MoEHybridTransformerBlockBase, block).use_combined_forward = True
    output2 = model(input_ids)

    torch.testing.assert_close(output1, output2)


@requires_multi_gpu
@pytest.mark.parametrize(
    "dropless", [pytest.param(True, id="dropless"), pytest.param(False, id="default-router")]
)
@pytest.mark.parametrize(
    "shared_experts", [pytest.param(True, id="shared-experts"), pytest.param(False, id="no-shared")]
)
@pytest.mark.parametrize(
    "reordered_norm",
    [pytest.param(True, id="reordered-norm"), pytest.param(False, id="default-block")],
)
@pytest.mark.parametrize("tp", [pytest.param(True, id="TP"), pytest.param(False, id="EP")])
def test_moe_hybrid_combined_forward(
    dropless: bool, shared_experts: bool, reordered_norm: bool, tp: bool
):
    run_distributed_test(
        run_moe_hybrid_combined_forward,
        backend="nccl",
        start_method="spawn",
        func_args=(
            dropless,
            shared_experts,
            reordered_norm,
            tp,
        ),
    )


def test_build_with_block_overrides():
    d_model = 512
    config = TransformerConfig.llama_like_moe(
        vocab_size=16_000,
        d_model=d_model,
        n_layers=8,
        n_heads=8,
        num_experts=32,
        top_k=4,
        expert_hidden_size=int(0.5 * d_model),
        capacity_factor=1.2,
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
        reordered_norm=True,
        hybrid=True,
        qk_norm=True,
        rope_theta=10_000,
        layer_norm_eps=1e-6,
        feed_forward=FeedForwardConfig(hidden_size=d_model * 2, bias=False),
    )
    assert config.block.feed_forward_moe is not None
    moe_config = replace(config.block.feed_forward_moe, shared_mlp=config.block.feed_forward)
    config.block_overrides = {
        0: replace(
            config.block,
            name=TransformerBlockType(str(config.block.name).replace("_hybrid_", "_")),
            feed_forward=None,
            feed_forward_moe=moe_config,
        )
    }

    model = config.build(init_device="cpu")
    assert isinstance(model.blocks["0"], MoEReorderedNormTransformerBlock)
    assert isinstance(model.blocks["1"], MoEHybridTransformerBlockBase)

    assert config.num_params == model.num_params


def test_num_flops_per_token_with_gqa():
    """Test that num_flops_per_token handles GQA (n_kv_heads < n_heads) without errors."""
    seq_len = 2048
    d_model = 512
    n_heads = 8
    n_kv_heads = 2  # GQA with 4:1 ratio

    # Config without GQA
    config_no_gqa = TransformerConfig.llama_like(
        vocab_size=16_000,
        d_model=d_model,
        n_layers=4,
        n_heads=n_heads,
        n_kv_heads=None,  # Same as n_heads
    )

    # Config with GQA
    config_gqa = TransformerConfig.llama_like(
        vocab_size=16_000,
        d_model=d_model,
        n_layers=4,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
    )

    flops_no_gqa = config_no_gqa.num_flops_per_token(seq_len)
    flops_gqa = config_gqa.num_flops_per_token(seq_len)

    # Both variants should produce valid, positive FLOPS estimates.
    assert flops_no_gqa > 0
    assert flops_gqa > 0

    # Also test on built model
    model_no_gqa = config_no_gqa.build(init_device="cpu")
    model_gqa = config_gqa.build(init_device="cpu")

    assert model_no_gqa.num_flops_per_token(seq_len) == flops_no_gqa
    assert model_gqa.num_flops_per_token(seq_len) == flops_gqa


def test_num_flops_per_token_with_swa():
    """Test that num_flops_per_token accounts for SWA (sliding window attention)."""
    seq_len = 2048
    window_size = 1024
    d_model = 512
    n_heads = 8

    # Config without SWA
    config_no_swa = TransformerConfig.llama_like(
        vocab_size=16_000,
        d_model=d_model,
        n_layers=4,
        n_heads=n_heads,
    )

    # Config with SWA (all layers use window)
    sliding_window = SlidingWindowAttentionConfig(
        pattern=[window_size],
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=False,
    )
    config_swa = TransformerConfig.llama_like(
        vocab_size=16_000,
        d_model=d_model,
        n_layers=4,
        n_heads=n_heads,
        sliding_window=sliding_window,
    )

    flops_no_swa = config_no_swa.num_flops_per_token(seq_len)
    flops_swa = config_swa.num_flops_per_token(seq_len)

    # With SWA, FLOPS should be lower because attention window is smaller
    # The reduction should be approximately proportional to window_size / seq_len
    expected_ratio = window_size / seq_len
    # Base FLOPS (non-attention) should be the same, only attention FLOPS change
    base_flops = 6 * config_no_swa.num_non_embedding_params
    attention_flops_no_swa = flops_no_swa - base_flops
    attention_flops_swa = flops_swa - base_flops

    # The attention FLOPS should scale approximately with window_size
    actual_ratio = attention_flops_swa / attention_flops_no_swa
    # Allow some tolerance due to rounding
    assert actual_ratio < 1.0, "SWA should reduce FLOPS"
    assert (
        abs(actual_ratio - expected_ratio) < 0.1
    ), f"Expected ratio ~{expected_ratio}, got {actual_ratio}"

    # Also test on built model
    model_no_swa = config_no_swa.build(init_device="cpu")
    model_swa = config_swa.build(init_device="cpu")

    assert model_no_swa.num_flops_per_token(seq_len) == flops_no_swa
    assert model_swa.num_flops_per_token(seq_len) == flops_swa


def test_num_flops_per_token_with_swa_and_gqa():
    """Test that num_flops_per_token accounts for both SWA and GQA together."""
    seq_len = 2048
    window_size = 1024
    d_model = 512
    n_heads = 8
    n_kv_heads = 2

    # Config with both SWA and GQA
    sliding_window = SlidingWindowAttentionConfig(
        pattern=[window_size],
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=False,
    )
    config_combined = TransformerConfig.llama_like(
        vocab_size=16_000,
        d_model=d_model,
        n_layers=4,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        sliding_window=sliding_window,
    )

    # Config without SWA or GQA
    config_baseline = TransformerConfig.llama_like(
        vocab_size=16_000,
        d_model=d_model,
        n_layers=4,
        n_heads=n_heads,
    )

    flops_baseline = config_baseline.num_flops_per_token(seq_len)
    flops_combined = config_combined.num_flops_per_token(seq_len)

    # Combined should have even lower FLOPS
    assert flops_combined < flops_baseline, "SWA + GQA should reduce FLOPS"

    # Also test on built model
    model_combined = config_combined.build(init_device="cpu")
    assert model_combined.num_flops_per_token(seq_len) == flops_combined


def test_num_flops_per_token_with_swa_pattern():
    """Test that num_flops_per_token handles SWA patterns correctly (different windows per layer)."""
    seq_len = 2048
    window_size_1 = 1024
    window_size_2 = 512
    d_model = 512
    n_heads = 8

    # Config with SWA pattern: alternating window sizes
    sliding_window = SlidingWindowAttentionConfig(
        pattern=[window_size_1, window_size_2],
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=False,
    )
    config_swa_pattern = TransformerConfig.llama_like(
        vocab_size=16_000,
        d_model=d_model,
        n_layers=4,
        n_heads=n_heads,
        sliding_window=sliding_window,
    )

    # Config without SWA
    config_no_swa = TransformerConfig.llama_like(
        vocab_size=16_000,
        d_model=d_model,
        n_layers=4,
        n_heads=n_heads,
    )

    flops_no_swa = config_no_swa.num_flops_per_token(seq_len)
    flops_swa_pattern = config_swa_pattern.num_flops_per_token(seq_len)

    # With SWA pattern, FLOPS should be lower
    assert flops_swa_pattern < flops_no_swa, "SWA pattern should reduce FLOPS"

    # Also test on built model
    model_swa_pattern = config_swa_pattern.build(init_device="cpu")
    assert model_swa_pattern.num_flops_per_token(seq_len) == flops_swa_pattern


def test_num_flops_per_token_with_block_overrides_different_n_heads():
    """Test that num_flops_per_token works when block overrides change n_heads."""
    seq_len = 2048
    d_model = 512
    base_n_heads = 8
    override_n_heads = 4  # Different n_heads in override (fewer heads = larger head_dim)
    override_n_kv_heads = 2  # Also use GQA to make the difference more pronounced

    # Config with block override that changes n_heads and uses GQA
    config_with_override = TransformerConfig.llama_like(
        vocab_size=16_000,
        d_model=d_model,
        n_layers=4,
        n_heads=base_n_heads,
    )

    # Create override block with different n_heads and n_kv_heads
    override_attention = replace(
        config_with_override.block.attention,
        n_heads=override_n_heads,
        n_kv_heads=override_n_kv_heads,
    )
    override_block = replace(config_with_override.block, attention=override_attention)
    config_with_override.block_overrides = {0: override_block}  # First layer has different n_heads

    # Config without override (but with same GQA for comparison)
    config_no_override = TransformerConfig.llama_like(
        vocab_size=16_000,
        d_model=d_model,
        n_layers=4,
        n_heads=base_n_heads,
        n_kv_heads=base_n_heads // 2,  # Use GQA to make it comparable
    )

    flops_no_override = config_no_override.num_flops_per_token(seq_len)
    flops_with_override = config_with_override.num_flops_per_token(seq_len)

    # The function should be stable and produce positive FLOPS estimates even when
    # block overrides change n_heads and n_kv_heads.
    assert flops_no_override > 0
    assert flops_with_override > 0

    # Also test on built model
    model_with_override = config_with_override.build(init_device="cpu")
    assert model_with_override.num_flops_per_token(seq_len) == flops_with_override


def test_num_flops_per_token_with_swa_window_larger_than_seq():
    """Test that num_flops_per_token caps window size at sequence length."""
    seq_len = 512  # Short sequence
    window_size = 2048  # Window larger than sequence
    d_model = 512
    n_heads = 8

    # Config with SWA window larger than sequence length
    sliding_window = SlidingWindowAttentionConfig(
        pattern=[window_size],
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=False,
    )
    config_swa_large_window = TransformerConfig.llama_like(
        vocab_size=16_000,
        d_model=d_model,
        n_layers=4,
        n_heads=n_heads,
        sliding_window=sliding_window,
    )

    # Config without SWA (baseline)
    config_no_swa = TransformerConfig.llama_like(
        vocab_size=16_000,
        d_model=d_model,
        n_layers=4,
        n_heads=n_heads,
    )

    flops_no_swa = config_no_swa.num_flops_per_token(seq_len)
    flops_swa_large_window = config_swa_large_window.num_flops_per_token(seq_len)

    # When window size > sequence length, effective window should be capped at seq_len
    # So FLOPS should be the same as without SWA (since we're using full attention anyway)
    assert flops_swa_large_window == flops_no_swa, (
        "When window size exceeds sequence length, FLOPS should equal full attention "
        "(window should be capped at sequence length)"
    )

    # Also test on built model
    model_swa_large_window = config_swa_large_window.build(init_device="cpu")
    assert model_swa_large_window.num_flops_per_token(seq_len) == flops_swa_large_window
