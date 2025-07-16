"""
Train a medium OLMoE model. Run this script without any arguments to see usage info.
"""

import logging
from dataclasses import replace

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import AOFloat8LinearConfig, Float8Config
from olmo_core.internal.experiment import CommonComponents, ExperimentConfig, main
from olmo_core.launch.beaker import OLMoCoreBeakerImage
from olmo_core.nn.transformer import TransformerBlockType, TransformerConfig, TransformerBlockConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import TrainerConfig, Duration
from olmo_core.train.callbacks import CheckpointerCallback, CometCallback, WandBCallback, ProfilerCallback, NvidiaProfilerCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
import math

log = logging.getLogger(__name__)

NUM_EXPERTS = 32
TOP_K = 8
NUM_LAYERS=6

MAIN_LR = 2e-3
# reduce the learning rate for expert parallel training
# topk, N experts, each expert gets k/N of the full batch tokens
EXPERT_LR = MAIN_LR * math.sqrt(TOP_K / NUM_EXPERTS)
# EXPERT_LR = MAIN_LR

USE_MOE = True
USE_MLA = False

def build_model_config(common: CommonComponents) -> TransformerConfig:
    d_model = 4096

    from olmo_core.nn.lm_head import LMHeadConfig
    from olmo_core.nn.attention import AttentionConfig, AttentionType, MultiheadLatentAttentionConfig
    from olmo_core.nn.feed_forward import FeedForwardConfig, FeedForwardType
    from olmo_core.nn.rope import RoPEConfig, RoPEType
    from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
    from olmo_core.nn.buffer_cache import BufferCache
    from olmo_core.nn.moe import MoEConfig, MoERouterConfig, MoEType

    
    dtype = 'float32'
    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=dtype,
    )
    
    if USE_MLA:
        attn_config = MultiheadLatentAttentionConfig(
            n_heads=24,
            bias=None,
            dropout=0.0,
            dtype=dtype,
            q_lora_rank=1024,
            kv_lora_rank=512, 
            qk_nope_head_dim=192,
            # qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=256,
            # v_head_dim=192,
            use_flash=False,
            qkv_norm=layer_norm,
            rope=RoPEConfig(name=RoPEType.default, theta=500_000, scaling=None),
        )
    else:
        attn_config = AttentionConfig(
            name=AttentionType.default,
            n_heads=16,
            n_kv_heads=None,
            bias=False,
            rope=RoPEConfig(name=RoPEType.default, theta=500_000, scaling=None),
            qk_norm=layer_norm,
            use_flash=False,
            dtype=dtype,
        )
    
    
    if USE_MOE:
        block_name = TransformerBlockType.moe_reordered_norm
        
        config = TransformerConfig(
            name='moe',
            d_model=d_model,
            vocab_size=common.tokenizer.padded_vocab_size(),
            n_layers=NUM_LAYERS,
            block=TransformerBlockConfig(
                name=block_name,
                attention=attn_config,
                # dense
                # feed_forward=FeedForwardConfig(hidden_size=(d_model*2), bias=False, dtype=dtype),
                # moe
                feed_forward=None,
                feed_forward_moe=MoEConfig(
                    name=MoEType.default,
                    num_experts=NUM_EXPERTS,
                    hidden_size=int(0.5 * d_model),
                    capacity_factor=1.25,
                    router=MoERouterConfig(top_k=TOP_K),
                    shared_mlp= FeedForwardConfig(hidden_size=int(d_model*2), bias=False),
                    lb_loss_weight=0.01,
                    z_loss_weight=0.001,
                ),
                layer_norm=layer_norm,
            ),
            lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
            dtype=DType.float32,
        )
    else:
        block_name = TransformerBlockType.default
        config = TransformerConfig(
            name='default',
            d_model=d_model,
            vocab_size=common.tokenizer.padded_vocab_size(),
            n_layers=NUM_LAYERS,
            block=TransformerBlockConfig(
                name=block_name,
                attention=attn_config,
                # dense
                feed_forward=FeedForwardConfig(hidden_size=(d_model*2), bias=False, dtype=dtype),
                # moe
                layer_norm=layer_norm,
            ),
            lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
            dtype=DType.float32,
        )

    return config


def finalize_config(config: ExperimentConfig):
    # add active & total params to the wandb name
    total_params_in_B = config.model.num_params/1000/1000/1000
    active_params_in_B = config.model.num_active_params/1000/1000/1000
    config.trainer.callbacks['wandb'].name += f"_{active_params_in_B:.2f}@{total_params_in_B:.2f}B"  # print to 2 decimal places
    config.trainer.callbacks['wandb'].name += f"_{TOP_K}K{NUM_EXPERTS}N"  # print to 2 decimal places
    


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:

    config = TransformerTrainModuleConfig(
        rank_microbatch_size=4 * 4096,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=AdamWConfig(
            lr=MAIN_LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0)),
                OptimGroupOverride(params=["blocks.*.feed_forward_moe.experts.mlp.*"], opts=dict(lr=EXPERT_LR))
            ],
            fused=True,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.bfloat16,
            shard_degree=8,
            prefetch_factor=1,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        #  ep_config=TransformerExpertParallelConfig(degree=-1),
        float8_config=Float8Config(
            ao=AOFloat8LinearConfig(
                enable_fsdp_float8_all_gather=True,
                force_recompute_fp8_weight_in_bwd=True,
                round_scales_to_power_of_2=True,
            ),
            enabled=False,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=1000),
    )
    return config


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    config = (
        TrainerConfig(
            save_folder=f'/workspace/tmp/{common.run_name}',
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=1,
            max_duration=Duration.tokens(100_000_000_000) # 100 Billion tokens
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=10_000,
                ephemeral_save_interval=250,
                save_async=True,
                pre_train_checkpoint=False,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=common.run_name,
                workspace="ai2",
                project="OLMoE",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="tianhua-moe",
                enabled=True,
                cancel_check_interval=10,
            ),
        ).
        with_callback(
            "profiler", 
            NvidiaProfilerCallback(enabled=False,
                                   profile_ranks=[0],
            )
        )
    )
    return config

if __name__ == "__main__":
    main(
        global_batch_size=1024 * 4096,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_default_evals=False,
        finalize_config=finalize_config,
    )
