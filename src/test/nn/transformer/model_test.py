import logging
from dataclasses import replace
from typing import cast

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor, init_device_mesh

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
from olmo_core.nn.attention import AttentionConfig
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
            name=TransformerBlockType.moe_hybrid_reordered_norm
            if reordered_norm
            else TransformerBlockType.moe_hybrid,
            attention=AttentionConfig(n_heads=8, rope=RoPEConfig(), qk_norm=layer_norm),
            layer_norm=layer_norm,
            feed_forward=FeedForwardConfig(hidden_size=1024, bias=False),
            feed_forward_moe=MoEConfig(
                name=MoEType.dropless if dropless else MoEType.default,
                num_experts=4,
                hidden_size=256,
                shared_mlp=FeedForwardConfig(hidden_size=512, bias=False)
                if shared_experts
                else None,
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


def test_attention_mask_conversion():
    """Test that transformer properly converts attention masks to cu_doc_lens."""
    config = TransformerConfig.llama_like(
        d_model=128,
        n_layers=2,
        n_heads=4,
        vocab_size=1000,
    )
    
    model = config.build(init_device="cpu")
    model.init_weights()
    
    # Create input with attention mask
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Create attention mask with prefix padding
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    attention_mask[0, 4:] = True  # First sequence: 4 padding, 12 valid
    attention_mask[1, 8:] = True  # Second sequence: 8 padding, 8 valid
    
    # Run forward pass - should convert attention mask to cu_doc_lens internally
    output = model(input_ids, attention_mask=attention_mask)
    
    # Check output shape is correct
    assert output.shape == (batch_size, seq_len, 1000)
    
    # Test with flash attention if available
    try:
        config_flash = TransformerConfig.llama_like(
            d_model=128,
            n_layers=2,
            n_heads=4,
            vocab_size=1000,
            attention=AttentionConfig(n_heads=4, use_flash=True),
        )
        model_flash = config_flash.build(init_device="cpu")
        model_flash.init_weights()
        
        # This should also work with flash attention
        output_flash = model_flash(input_ids, attention_mask=attention_mask)
        assert output_flash.shape == (batch_size, seq_len, 1000)
    except ImportError:
        # Flash attention not available, skip this part
        pass
