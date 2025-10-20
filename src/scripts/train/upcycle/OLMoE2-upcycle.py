"""
Train a medium OLMoE model. Run this script without any arguments to see usage info.

Virtual-group upcycling: MLP weights are sharded and duplicated across virtual groups of experts, and router weights are duplicated per virtual group.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from olmo_core.config import Config, DType
from olmo_core.internal.experiment import CommonComponents
from olmo_core.nn.transformer import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.train.checkpoint import UpcycleCheckpointer
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


global_args = dict()


NUM_LAYERS = 16
USE_MOE = True
USE_MLA = False


def build_model_config(
    routed_expert_norm=False,
    shared_expert_norm=False,
    feed_forward_norm=True,
    common: Optional[CommonComponents] = None,
) -> TransformerConfig:
    d_model = 2048
    NUM_EXPERTS = global_args["NUM_EXPERTS"]
    TOP_K = global_args["TOP_K"]
    MOE_EXPANSION_FACTOR = global_args["MOE_EXPANSION_FACTOR"]
    SHARED_EXPERT_EXPANSION_FACTOR = global_args["SHARED_EXPERT_EXPANSION_FACTOR"]

    from olmo_core.data import TokenizerConfig
    from olmo_core.nn.attention import (
        AttentionConfig,
        AttentionType,
        MultiheadLatentAttentionConfig,
    )
    from olmo_core.nn.feed_forward import FeedForwardConfig
    from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
    from olmo_core.nn.lm_head import LMHeadConfig
    from olmo_core.nn.moe import MoEConfig, MoERouterConfig, MoEType
    from olmo_core.nn.rope import RoPEConfig, RoPEType
    from olmo_core.nn.transformer.config import TransformerType

    dtype = DType.float32
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
            name=TransformerType.moe,
            d_model=d_model,
            vocab_size=TokenizerConfig.dolma2().padded_vocab_size(),
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
                    hidden_size=int(MOE_EXPANSION_FACTOR * d_model),
                    capacity_factor=1.25,
                    router=MoERouterConfig(top_k=TOP_K),
                    shared_mlp=FeedForwardConfig(
                        hidden_size=int(d_model * SHARED_EXPERT_EXPANSION_FACTOR), bias=False
                    ),
                    lb_loss_weight=0.01,
                    z_loss_weight=0.001,
                    routed_expert_norm=layer_norm if routed_expert_norm else None,
                    shared_expert_norm=layer_norm if shared_expert_norm else None,
                ),
                attention_norm=layer_norm,
                feed_forward_norm=layer_norm if feed_forward_norm else None,
            ),
            lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
            dtype=DType.float32,
        )
    else:
        raise
        # block_name = TransformerBlockType.default
        # config = TransformerConfig(
        #     name=TransformerType.default,
        #     d_model=d_model,
        #     vocab_size=TokenizerConfig.dolma2().padded_vocab_size(),
        #     n_layers=NUM_LAYERS,
        #     block=TransformerBlockConfig(
        #         name=block_name,
        #         attention=attn_config,
        #         # dense
        #         feed_forward=FeedForwardConfig(hidden_size=(d_model*2), bias=False, dtype=dtype),
        #         # moe
        #         layer_norm=layer_norm,
        #     ),
        #     lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
        #     dtype=DType.float32,
        # )

    return config


@dataclass
class UpcycleConfig(Config):
    source_model_checkpoint: str
    target_model_output_path: str
    method: str
    target_model: TransformerConfig
    init_seed: int = 2025


from transformers import AutoModelForCausalLM


def upcycle_copy_mlp(source_model, target_model):
    NUM_EXPERTS = global_args["NUM_EXPERTS"]

    def map_up_proj_to_w1(weight):
        """
        source weight: (intermediate_size, d_model)
        target weight: (d_model * NUM_EXPERTS, intermediate_size)

        """
        # 1) transpose so that rows = d_model, cols = intermediate_size
        target_weight = weight.transpose(0, 1).repeat(NUM_EXPERTS, 1)
        return target_weight

    def map_down_proj_to_w2(weight):
        """
        source weight: (d_model, intermediate_size)
        target weight: (intermediate_size * NUM_EXPERTS, d_model)
        """

        target_model = weight.transpose(0, 1).repeat(NUM_EXPERTS, 1)
        return target_model

    def map_gate_proj_to_w3(weight):
        """
        source weight: (intermediate_size, d_model)
        target weight: (d_model * NUM_EXPERTS, intermediate_size)

        """
        target_model = weight.transpose(0, 1).repeat(NUM_EXPERTS, 1)
        return target_model

    src_tgt_mapping = {
        "model.embed_tokens.weight": "embeddings.weight",
        "model.layers.$LAYER.self_attn.q_proj.weight": "blocks.$LAYER.attention.w_q.weight",
        "model.layers.$LAYER.self_attn.k_proj.weight": "blocks.$LAYER.attention.w_k.weight",
        "model.layers.$LAYER.self_attn.v_proj.weight": "blocks.$LAYER.attention.w_v.weight",
        "model.layers.$LAYER.self_attn.o_proj.weight": "blocks.$LAYER.attention.w_out.weight",
        "model.layers.$LAYER.self_attn.q_norm.weight": "blocks.$LAYER.attention.q_norm.weight",
        "model.layers.$LAYER.self_attn.k_norm.weight": "blocks.$LAYER.attention.k_norm.weight",
        "model.layers.$LAYER.mlp.gate_proj.weight": (
            "blocks.$LAYER.feed_forward_moe.experts.mlp.w3",
            map_gate_proj_to_w3,
        ),  # means special handling
        "model.layers.$LAYER.mlp.up_proj.weight": (
            "blocks.$LAYER.feed_forward_moe.experts.mlp.w1",
            map_up_proj_to_w1,
        ),  # means special handling
        "model.layers.$LAYER.mlp.down_proj.weight": (
            "blocks.$LAYER.feed_forward_moe.experts.mlp.w2",
            map_down_proj_to_w2,
        ),  # means special handling
        "model.layers.$LAYER.post_attention_layernorm.weight": "blocks.$LAYER.attention_norm.weight",
        "model.layers.$LAYER.post_feedforward_layernorm.weight": "blocks.$LAYER.feed_forward_norm.weight",
        "model.norm.weight": "lm_head.norm.weight",
        "lm_head.weight": "lm_head.w_out.weight",
    }

    untouched_target_keys = [
        "blocks.*.feed_forward_moe.router.weight",
        "blocks.*.feed_forward_moe.shared_mlp.w1.weight",
        "blocks.*.feed_forward_moe.shared_mlp.w2.weight",
        "blocks.*.feed_forward_moe.shared_mlp.w3.weight",
    ]

    # expand the $LAYER in src_tgt_mapping
    src_tgt_mapping_expanded = {}
    for src_key, tgt_key in src_tgt_mapping.items():
        if "$LAYER" in src_key:
            for i in range(NUM_LAYERS):
                src_key_expanded = src_key.replace("$LAYER", str(i))
                # check if tgt_key is typle
                if isinstance(tgt_key, tuple):
                    tgt_key_expanded = (tgt_key[0].replace("$LAYER", str(i)), tgt_key[1])
                else:  # just string
                    tgt_key_expanded = tgt_key.replace("$LAYER", str(i))
                src_tgt_mapping_expanded[src_key_expanded] = tgt_key_expanded
        else:
            src_tgt_mapping_expanded[src_key] = tgt_key

    # make a copy of the target model keys
    all_target_keys = set(target_model.state_dict().keys())

    print("\n\nStart copying weights from source to target model...\n")
    # copy weights from source to target
    for src_key, tgt_key in src_tgt_mapping_expanded.items():
        if isinstance(tgt_key, tuple):
            print(f"{src_key} -> {tgt_key[0]} ({tgt_key[1].__name__})")
            # if tgt_key is tuple, it means special handling
            tgt_key, map_func = tgt_key
        else:
            print(f"{src_key} -> {tgt_key}")
            map_func = lambda x: x  # identity function

        # check if the key is in source model
        if src_key not in source_model.state_dict():
            raise ValueError(f"Key {src_key} not found in source model")
        # check if the key is in target model
        if tgt_key not in all_target_keys:
            raise ValueError(f"Key {tgt_key} not found in target model")
        # get the weight from source model
        weight = source_model.state_dict()[src_key]
        # map the weight to target model
        target_model.state_dict()[tgt_key].copy_(map_func(weight))

        # remove the key from all_target_keys, mark it as copied
        all_target_keys.remove(tgt_key)

    # check weights not handled by src_tgt_mapping
    print("\n\nUntouched weights in target model:\n")
    for key in all_target_keys:
        print(key)
        # check if the key matches the untouched_target_keys
        for untouched_key in untouched_target_keys:
            if re.match(untouched_key, key):
                break
        else:
            # if the key is not in untouched_target_keys, raise error
            raise ValueError(f"Key {key} not found in src_tgt_mapping or untouched_target_keys")

    return target_model


def upcycle_virtual_group_init(
    source_model,
    target_model,
):
    # ------------------------------------------------------------------
    # 0. Hyper‑params that control the sharding/scaling logic
    # ------------------------------------------------------------------
    NUM_EXPERTS = global_args["NUM_EXPERTS"]
    TOP_K = global_args["TOP_K"]

    # ------------------------------------------------------------------
    # 1. non‑MLP weights are copied verbatim via src→tgt mapping
    # ------------------------------------------------------------------
    src_tgt_mapping = {
        "model.embed_tokens.weight": "embeddings.weight",
        "model.layers.$LAYER.self_attn.q_proj.weight": "blocks.$LAYER.attention.w_q.weight",
        "model.layers.$LAYER.self_attn.k_proj.weight": "blocks.$LAYER.attention.w_k.weight",
        "model.layers.$LAYER.self_attn.v_proj.weight": "blocks.$LAYER.attention.w_v.weight",
        "model.layers.$LAYER.self_attn.o_proj.weight": "blocks.$LAYER.attention.w_out.weight",
        "model.layers.$LAYER.self_attn.q_norm.weight": "blocks.$LAYER.attention.q_norm.weight",
        "model.layers.$LAYER.self_attn.k_norm.weight": "blocks.$LAYER.attention.k_norm.weight",
        "model.layers.$LAYER.post_attention_layernorm.weight": "blocks.$LAYER.attention_norm.weight",
        "model.layers.$LAYER.post_feedforward_layernorm.weight": "blocks.$LAYER.feed_forward_norm.weight",
        "model.norm.weight": "lm_head.norm.weight",
        "lm_head.weight": "lm_head.w_out.weight",
    }

    untouched_target_keys = [
        "blocks.*.feed_forward_moe.shared_mlp.w1.weight",
        "blocks.*.feed_forward_moe.shared_mlp.w2.weight",
        "blocks.*.feed_forward_moe.shared_mlp.w3.weight",
    ]

    # expand the $LAYER in src_tgt_mapping
    src_tgt_mapping_expanded = {}
    for src_key, tgt_key in src_tgt_mapping.items():
        if "$LAYER" in src_key:
            for i in range(NUM_LAYERS):
                src_key_expanded = src_key.replace("$LAYER", str(i))
                # check if tgt_key is typle
                tgt_key_expanded = tgt_key.replace("$LAYER", str(i))
                src_tgt_mapping_expanded[src_key_expanded] = tgt_key_expanded
        else:
            src_tgt_mapping_expanded[src_key] = tgt_key

    # create a mutable set of *all* parameter keys for the sanity check
    remaining_target_keys = set(target_model.state_dict().keys())

    print("\n[virtual-group-init] Copying NON-MLP weights")
    for src_key, tgt_key in src_tgt_mapping_expanded.items():
        print(f"{src_key}  ->  {tgt_key}")
        target_model.state_dict()[tgt_key].copy_(source_model.state_dict()[src_key])
        remaining_target_keys.discard(tgt_key)

    # ------------------------------------------------------------------
    # 3. Prepare constants for sharding the dense MLP into virtual groups
    # ------------------------------------------------------------------
    # dense dims
    inter_dense, d_model = source_model.state_dict()["model.layers.0.mlp.up_proj.weight"].shape
    # expert hidden size in the MoE
    _, expert_hidden = target_model.state_dict()["blocks.0.feed_forward_moe.experts.mlp.w1"].shape

    G = inter_dense // expert_hidden  # #shards per layer
    assert G * expert_hidden == inter_dense, "expert_hidden does not divide intermediate_size"
    assert NUM_EXPERTS % G == 0, "NUM_EXPERTS must be an integer multiple of shard count G"

    dup_per_shard = NUM_EXPERTS // G  # this is “E” in the paper
    scale = ((dup_per_shard * (G**2)) / TOP_K) ** (1 / 3)

    print(
        f"\n[virtual-group-init] Sharding MLP: inter_dense={inter_dense}, "
        f"expert_hidden={expert_hidden}, G={G}, scale={scale:.4f}"
    )

    # ------------------------------------------------------------------
    # 4. Shard & scatter W1/W2/W3 into experts for every layer
    # ------------------------------------------------------------------
    for layer in range(NUM_LAYERS):
        # dense weights
        up = source_model.state_dict()[
            f"model.layers.{layer}.mlp.up_proj.weight"
        ].T  # (d_model, inter)
        gate = source_model.state_dict()[f"model.layers.{layer}.mlp.gate_proj.weight"].T
        down = source_model.state_dict()[
            f"model.layers.{layer}.mlp.down_proj.weight"
        ].T  # (inter, d_model)

        # target aggregated expert weight tensors
        w1_key = f"blocks.{layer}.feed_forward_moe.experts.mlp.w1"
        w2_key = f"blocks.{layer}.feed_forward_moe.experts.mlp.w2"
        w3_key = f"blocks.{layer}.feed_forward_moe.experts.mlp.w3"
        w1 = target_model.state_dict()[w1_key]
        w2 = target_model.state_dict()[w2_key]
        w3 = target_model.state_dict()[w3_key]

        # w1 is up, (d_model * NUM_EXPERTS, intermediate_size)
        # w2 is down, (intermediate_size * NUM_EXPERTS, d_model)
        # w3 is gate, (d_model * NUM_EXPERTS, intermediate_size)
        for expert_idx in range(NUM_EXPERTS):
            shard_idx = expert_idx % G
            col_start = shard_idx * expert_hidden
            col_end = (shard_idx + 1) * expert_hidden

            # row/col ranges inside the flattened expert‑matrix view
            row_start = expert_idx * d_model
            row_end = (expert_idx + 1) * d_model
            exp_col_start = expert_idx * expert_hidden
            exp_col_end = (expert_idx + 1) * expert_hidden

            assert w1[row_start:row_end, :].shape == up[:, col_start:col_end].shape
            assert w3[row_start:row_end, :].shape == gate[:, col_start:col_end].shape
            assert w2[exp_col_start:exp_col_end, :].shape == down[col_start:col_end, :].shape

            # W1 / gate : (d_model, expert_hidden)
            w1[row_start:row_end, :].copy_(up[:, col_start:col_end] * scale)
            w3[row_start:row_end, :].copy_(gate[:, col_start:col_end] * scale)

            # W2 : (expert_hidden, d_model)
            w2[exp_col_start:exp_col_end, :].copy_(down[col_start:col_end, :] / scale)

        # ---------- router (virtual-group duplication, flattened param) ----------
        router_key = f"blocks.{layer}.feed_forward_moe.router.weight"
        router_w_flat = target_model.state_dict()[router_key]  # (N*d_model,)
        router_2d = router_w_flat.view(NUM_EXPERTS, d_model)  # (N, d_model)

        # first G rows are the prototypes for the G shard-groups
        prototypes = router_2d[:G].clone()  # (G, d_model)
        for expert_idx in range(NUM_EXPERTS):
            group_id = expert_idx % G
            router_2d[expert_idx].copy_(prototypes[group_id])

        # router_w_flat already updated (router_2d shares storage)
        remaining_target_keys.discard(router_key)

        # mark these big tensors as handled
        remaining_target_keys.discard(w1_key)
        remaining_target_keys.discard(w2_key)
        remaining_target_keys.discard(w3_key)

        print(
            f"\n[virtual-group-init] Layer {layer}: {w1_key}, {w2_key}, {w3_key}, {router_key} | shape = {w1.shape} {w2.shape}, {w3.shape}, {router_2d.shape}"
        )

    # ------------------------------------------------------------------
    # 5. Final sanity check — every remaining key must match an untouched pattern
    # ------------------------------------------------------------------

    print("\n\nUntouched weights in target model:\n")
    for key in remaining_target_keys:
        print(key)
        # check if the key matches the untouched_target_keys
        for untouched_key in untouched_target_keys:
            if re.match(untouched_key, key):
                break
        else:
            # if the key is not in untouched_target_keys, raise error
            raise ValueError(f"Key {key} not found in src_tgt_mapping or untouched_target_keys")

    print("\n✔  All parameters (including router) accounted for.")
    return target_model


def upcycle_copy_mlp_as_shared_expert(source_model, target_model, norm_type):
    if norm_type == "1":
        src_tgt_mapping = {
            "model.embed_tokens.weight": "embeddings.weight",
            "model.layers.$LAYER.self_attn.q_proj.weight": "blocks.$LAYER.attention.w_q.weight",
            "model.layers.$LAYER.self_attn.k_proj.weight": "blocks.$LAYER.attention.w_k.weight",
            "model.layers.$LAYER.self_attn.v_proj.weight": "blocks.$LAYER.attention.w_v.weight",
            "model.layers.$LAYER.self_attn.o_proj.weight": "blocks.$LAYER.attention.w_out.weight",
            "model.layers.$LAYER.self_attn.q_norm.weight": "blocks.$LAYER.attention.q_norm.weight",
            "model.layers.$LAYER.self_attn.k_norm.weight": "blocks.$LAYER.attention.k_norm.weight",
            "model.layers.$LAYER.mlp.gate_proj.weight": "blocks.$LAYER.feed_forward_moe.shared_mlp.w3.weight",
            "model.layers.$LAYER.mlp.up_proj.weight": "blocks.$LAYER.feed_forward_moe.shared_mlp.w1.weight",
            "model.layers.$LAYER.mlp.down_proj.weight": "blocks.$LAYER.feed_forward_moe.shared_mlp.w2.weight",
            "model.layers.$LAYER.post_attention_layernorm.weight": "blocks.$LAYER.attention_norm.weight",
            "model.layers.$LAYER.post_feedforward_layernorm.weight": "blocks.$LAYER.feed_forward_norm.weight",
            "model.norm.weight": "lm_head.norm.weight",
            "lm_head.weight": "lm_head.w_out.weight",
        }

        untouched_target_keys = [
            "blocks.*.feed_forward_moe.router.weight",
            "blocks.*.feed_forward_moe.experts.mlp.w1",
            "blocks.*.feed_forward_moe.experts.mlp.w2",
            "blocks.*.feed_forward_moe.experts.mlp.w3",
        ]
    elif norm_type == "2":
        src_tgt_mapping = {
            "model.embed_tokens.weight": "embeddings.weight",
            "model.layers.$LAYER.self_attn.q_proj.weight": "blocks.$LAYER.attention.w_q.weight",
            "model.layers.$LAYER.self_attn.k_proj.weight": "blocks.$LAYER.attention.w_k.weight",
            "model.layers.$LAYER.self_attn.v_proj.weight": "blocks.$LAYER.attention.w_v.weight",
            "model.layers.$LAYER.self_attn.o_proj.weight": "blocks.$LAYER.attention.w_out.weight",
            "model.layers.$LAYER.self_attn.q_norm.weight": "blocks.$LAYER.attention.q_norm.weight",
            "model.layers.$LAYER.self_attn.k_norm.weight": "blocks.$LAYER.attention.k_norm.weight",
            "model.layers.$LAYER.mlp.gate_proj.weight": "blocks.$LAYER.feed_forward_moe.shared_mlp.w3.weight",
            "model.layers.$LAYER.mlp.up_proj.weight": "blocks.$LAYER.feed_forward_moe.shared_mlp.w1.weight",
            "model.layers.$LAYER.mlp.down_proj.weight": "blocks.$LAYER.feed_forward_moe.shared_mlp.w2.weight",
            "model.layers.$LAYER.post_attention_layernorm.weight": "blocks.$LAYER.attention_norm.weight",
            "model.layers.$LAYER.post_feedforward_layernorm.weight": "blocks.$LAYER.feed_forward_norm.weight",
            "model.norm.weight": "lm_head.norm.weight",
            "lm_head.weight": "lm_head.w_out.weight",
        }

        untouched_target_keys = [
            "blocks.*.feed_forward_moe.router.weight",
            "blocks.*.feed_forward_moe.experts.mlp.w1",
            "blocks.*.feed_forward_moe.experts.mlp.w2",
            "blocks.*.feed_forward_moe.experts.mlp.w3",
            "blocks.*.feed_forward_moe.routed_expert_norm.weight",
            "blocks.*.feed_forward_moe.shared_expert_norm.weight",
        ]
    elif norm_type == "2B":
        src_tgt_mapping = {
            "model.embed_tokens.weight": "embeddings.weight",
            "model.layers.$LAYER.self_attn.q_proj.weight": "blocks.$LAYER.attention.w_q.weight",
            "model.layers.$LAYER.self_attn.k_proj.weight": "blocks.$LAYER.attention.w_k.weight",
            "model.layers.$LAYER.self_attn.v_proj.weight": "blocks.$LAYER.attention.w_v.weight",
            "model.layers.$LAYER.self_attn.o_proj.weight": "blocks.$LAYER.attention.w_out.weight",
            "model.layers.$LAYER.self_attn.q_norm.weight": "blocks.$LAYER.attention.q_norm.weight",
            "model.layers.$LAYER.self_attn.k_norm.weight": "blocks.$LAYER.attention.k_norm.weight",
            "model.layers.$LAYER.mlp.gate_proj.weight": "blocks.$LAYER.feed_forward_moe.shared_mlp.w3.weight",
            "model.layers.$LAYER.mlp.up_proj.weight": "blocks.$LAYER.feed_forward_moe.shared_mlp.w1.weight",
            "model.layers.$LAYER.mlp.down_proj.weight": "blocks.$LAYER.feed_forward_moe.shared_mlp.w2.weight",
            "model.layers.$LAYER.post_attention_layernorm.weight": "blocks.$LAYER.attention_norm.weight",
            "model.layers.$LAYER.post_feedforward_layernorm.weight": "blocks.$LAYER.feed_forward_moe.shared_expert_norm.weight",
            "model.norm.weight": "lm_head.norm.weight",
            "lm_head.weight": "lm_head.w_out.weight",
        }

        untouched_target_keys = [
            "blocks.*.feed_forward_moe.router.weight",
            "blocks.*.feed_forward_moe.experts.mlp.w1",
            "blocks.*.feed_forward_moe.experts.mlp.w2",
            "blocks.*.feed_forward_moe.experts.mlp.w3",
            "blocks.*.feed_forward_norm.weight",
            "blocks.*.feed_forward_moe.routed_expert_norm.weight",
        ]
    elif norm_type == "3":
        src_tgt_mapping = {
            "model.embed_tokens.weight": "embeddings.weight",
            "model.layers.$LAYER.self_attn.q_proj.weight": "blocks.$LAYER.attention.w_q.weight",
            "model.layers.$LAYER.self_attn.k_proj.weight": "blocks.$LAYER.attention.w_k.weight",
            "model.layers.$LAYER.self_attn.v_proj.weight": "blocks.$LAYER.attention.w_v.weight",
            "model.layers.$LAYER.self_attn.o_proj.weight": "blocks.$LAYER.attention.w_out.weight",
            "model.layers.$LAYER.self_attn.q_norm.weight": "blocks.$LAYER.attention.q_norm.weight",
            "model.layers.$LAYER.self_attn.k_norm.weight": "blocks.$LAYER.attention.k_norm.weight",
            "model.layers.$LAYER.mlp.gate_proj.weight": "blocks.$LAYER.feed_forward_moe.shared_mlp.w3.weight",
            "model.layers.$LAYER.mlp.up_proj.weight": "blocks.$LAYER.feed_forward_moe.shared_mlp.w1.weight",
            "model.layers.$LAYER.mlp.down_proj.weight": "blocks.$LAYER.feed_forward_moe.shared_mlp.w2.weight",
            "model.layers.$LAYER.post_attention_layernorm.weight": "blocks.$LAYER.attention_norm.weight",
            "model.layers.$LAYER.post_feedforward_layernorm.weight": "blocks.$LAYER.feed_forward_moe.shared_expert_norm.weight",
            "model.norm.weight": "lm_head.norm.weight",
            "lm_head.weight": "lm_head.w_out.weight",
        }

        untouched_target_keys = [
            "blocks.*.feed_forward_moe.router.weight",
            "blocks.*.feed_forward_moe.experts.mlp.w1",
            "blocks.*.feed_forward_moe.experts.mlp.w2",
            "blocks.*.feed_forward_moe.experts.mlp.w3",
            # 'blocks.*.feed_forward_norm.weight',
            "blocks.*.feed_forward_moe.routed_expert_norm.weight",
        ]
    elif norm_type == "4":
        src_tgt_mapping = {
            "model.embed_tokens.weight": "embeddings.weight",
            "model.layers.$LAYER.self_attn.q_proj.weight": "blocks.$LAYER.attention.w_q.weight",
            "model.layers.$LAYER.self_attn.k_proj.weight": "blocks.$LAYER.attention.w_k.weight",
            "model.layers.$LAYER.self_attn.v_proj.weight": "blocks.$LAYER.attention.w_v.weight",
            "model.layers.$LAYER.self_attn.o_proj.weight": "blocks.$LAYER.attention.w_out.weight",
            "model.layers.$LAYER.self_attn.q_norm.weight": "blocks.$LAYER.attention.q_norm.weight",
            "model.layers.$LAYER.self_attn.k_norm.weight": "blocks.$LAYER.attention.k_norm.weight",
            "model.layers.$LAYER.mlp.gate_proj.weight": "blocks.$LAYER.feed_forward_moe.shared_mlp.w3.weight",
            "model.layers.$LAYER.mlp.up_proj.weight": "blocks.$LAYER.feed_forward_moe.shared_mlp.w1.weight",
            "model.layers.$LAYER.mlp.down_proj.weight": "blocks.$LAYER.feed_forward_moe.shared_mlp.w2.weight",
            "model.layers.$LAYER.post_attention_layernorm.weight": "blocks.$LAYER.attention_norm.weight",
            "model.layers.$LAYER.post_feedforward_layernorm.weight": "blocks.$LAYER.feed_forward_norm.weight",  # the mlp norm
            "model.norm.weight": "lm_head.norm.weight",
            "lm_head.weight": "lm_head.w_out.weight",
        }

        untouched_target_keys = [
            "blocks.*.feed_forward_moe.router.weight",
            "blocks.*.feed_forward_moe.experts.mlp.w1",
            "blocks.*.feed_forward_moe.experts.mlp.w2",
            "blocks.*.feed_forward_moe.experts.mlp.w3",
            "blocks.*.feed_forward_moe.routed_expert_norm.weight",
        ]
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

    # expand the $LAYER in src_tgt_mapping
    src_tgt_mapping_expanded = {}
    for src_key, tgt_key in src_tgt_mapping.items():
        if "$LAYER" in src_key:
            for i in range(NUM_LAYERS):
                src_key_expanded = src_key.replace("$LAYER", str(i))
                # check if tgt_key is typle
                if isinstance(tgt_key, tuple):
                    tgt_key_expanded = (tgt_key[0].replace("$LAYER", str(i)), tgt_key[1])
                else:  # just string
                    tgt_key_expanded = tgt_key.replace("$LAYER", str(i))
                src_tgt_mapping_expanded[src_key_expanded] = tgt_key_expanded
        else:
            src_tgt_mapping_expanded[src_key] = tgt_key

    # make a copy of the target model keys
    all_target_keys = set(target_model.state_dict().keys())

    print("\n\nStart copying weights from source to target model...\n")
    # copy weights from source to target
    for src_key, tgt_key in src_tgt_mapping_expanded.items():
        if isinstance(tgt_key, tuple):
            print(f"{src_key} -> {tgt_key[0]} ({tgt_key[1].__name__})")
            # if tgt_key is tuple, it means special handling
            tgt_key, map_func = tgt_key
        else:
            print(f"{src_key} -> {tgt_key}")
            map_func = lambda x: x  # identity function

        # check if the key is in source model
        if src_key not in source_model.state_dict():
            raise ValueError(f"Key {src_key} not found in source model")
        # check if the key is in target model
        if tgt_key not in all_target_keys:
            raise ValueError(f"Key {tgt_key} not found in target model")
        # get the weight from source model
        weight = source_model.state_dict()[src_key]
        # map the weight to target model
        target_model.state_dict()[tgt_key].copy_(map_func(weight))

        # remove the key from all_target_keys, mark it as copied
        all_target_keys.remove(tgt_key)

    # check weights not handled by src_tgt_mapping
    print("\n\nUntouched weights in target model:\n")
    for key in all_target_keys:
        print(key)
        # check if the key matches the untouched_target_keys
        for untouched_key in untouched_target_keys:
            if re.match(untouched_key, key):
                break
        else:
            # if the key is not in untouched_target_keys, raise error
            raise ValueError(f"Key {key} not found in src_tgt_mapping or untouched_target_keys")

    return target_model


def upcycle(config: UpcycleConfig):
    target_model = config.target_model.build()

    source_model = AutoModelForCausalLM.from_pretrained(config.source_model_checkpoint)

    source_model_state_dict = source_model.state_dict()

    # print key + size
    print("\n--- target_model ---")
    for key, value in target_model.state_dict().items():
        print(key, tuple(value.size()))

    print("\n--- source_model ---")
    for key, value in source_model_state_dict.items():
        print(key, tuple(value.size()))

    # method 1: "copy-mlp"
    if config.method == "copy-mlp":
        target_model = upcycle_copy_mlp(source_model, target_model)

    # method 2: "copy-mlp-noise"
    elif config.method == "copy-mlp-noise":
        pass
    elif config.method == "copy-mlp-partial-reinit":
        pass
    elif config.method == "copy-mlp-as-shared-expert-type1":
        target_model = upcycle_copy_mlp_as_shared_expert(source_model, target_model, norm_type="1")
    elif config.method == "copy-mlp-as-shared-expert-type2":
        target_model = upcycle_copy_mlp_as_shared_expert(source_model, target_model, norm_type="2")
    elif config.method == "copy-mlp-as-shared-expert-type2B":
        target_model = upcycle_copy_mlp_as_shared_expert(source_model, target_model, norm_type="2B")
    elif config.method == "copy-mlp-as-shared-expert-type3":
        target_model = upcycle_copy_mlp_as_shared_expert(source_model, target_model, norm_type="3")
    elif config.method == "copy-mlp-as-shared-expert-type4":
        target_model = upcycle_copy_mlp_as_shared_expert(source_model, target_model, norm_type="4")
    # method 3: virtual group initialization
    elif config.method == "virtual-group":
        target_model = upcycle_virtual_group_init(source_model, target_model)

    else:
        raise ValueError(f"Unknown method: {config.method}")

    UpcycleCheckpointer(work_dir=Path(".")).save_upcycled_model(
        dir=config.target_model_output_path,
        model_state_dict=target_model.state_dict(),
    )

    print("\n\n--- target_model saved ---")


if __name__ == "__main__":
    prepare_cli_environment()
    SRC_CKPT = "/weka/oe-training-default/tianhua/ws-megatron/OLMo-2-0425-1B/stage1-step1907359-tokens4001B"

    # # copy-mlp
    # global_args['NUM_EXPERTS'] = 8
    # global_args['TOP_K'] = 2
    # global_args['MOE_EXPANSION_FACTOR'] = 4
    # global_args['SHARED_EXPERT_EXPANSION_FACTOR'] = 4
    # upcycle_config = UpcycleConfig(
    #     source_model_checkpoint=SRC_CKPT,
    #     target_model=build_model_config(),
    #     target_model_output_path="/workspace/tmp/upcycled-OLMo-2-0425-1B/copy-mlp",
    #     method="copy-mlp",
    # )

    # upcycle(upcycle_config)

    # # copy-mlp-as-shared-expert-type1
    # global_args['NUM_EXPERTS'] = 32
    # global_args['TOP_K'] = 8
    # global_args['MOE_EXPANSION_FACTOR'] = 1
    # global_args['SHARED_EXPERT_EXPANSION_FACTOR'] = 4
    # upcycle_config = UpcycleConfig(
    #     source_model_checkpoint=SRC_CKPT,
    #     target_model=build_model_config(routed_expert_norm=False, shared_expert_norm=False),
    #     target_model_output_path="/workspace/tmp/upcycled-OLMo-2-0425-1B/copy-mlp-as-shared-expert-type1",
    #     method="copy-mlp-as-shared-expert-type1",
    # )
    # upcycle(upcycle_config)

    # # copy-mlp-as-shared-expert-type2
    # global_args['NUM_EXPERTS'] = 32
    # global_args['TOP_K'] = 8
    # global_args['MOE_EXPANSION_FACTOR'] = 1
    # global_args['SHARED_EXPERT_EXPANSION_FACTOR'] = 4
    # upcycle_config = UpcycleConfig(
    #     source_model_checkpoint=SRC_CKPT,
    #     target_model=build_model_config(routed_expert_norm=True, shared_expert_norm=True),
    #     target_model_output_path="/workspace/tmp/upcycled-OLMo-2-0425-1B/copy-mlp-as-shared-expert-type2",
    #     method="copy-mlp-as-shared-expert-type2",
    # )
    # upcycle(upcycle_config)

    # # copy-mlp-as-shared-expert-type2B
    # global_args['NUM_EXPERTS'] = 32
    # global_args['TOP_K'] = 8
    # global_args['MOE_EXPANSION_FACTOR'] = 1
    # global_args['SHARED_EXPERT_EXPANSION_FACTOR'] = 4
    # upcycle_config = UpcycleConfig(
    #     source_model_checkpoint=SRC_CKPT,
    #     target_model=build_model_config(routed_expert_norm=True, shared_expert_norm=True),
    #     target_model_output_path="/workspace/tmp/upcycled-OLMo-2-0425-1B/copy-mlp-as-shared-expert-type2B",
    #     method="copy-mlp-as-shared-expert-type2B",
    # )
    # upcycle(upcycle_config)

    # # copy-mlp-as-shared-expert-type3
    # global_args['NUM_EXPERTS'] = 32
    # global_args['TOP_K'] = 8
    # global_args['MOE_EXPANSION_FACTOR'] = 1
    # global_args['SHARED_EXPERT_EXPANSION_FACTOR'] = 4
    # upcycle_config = UpcycleConfig(
    #     source_model_checkpoint=SRC_CKPT,
    #     target_model=build_model_config(routed_expert_norm=True, shared_expert_norm=True, feed_forward_norm=False), # remove the final layer norm
    #     target_model_output_path="/workspace/tmp/upcycled-OLMo-2-0425-1B/copy-mlp-as-shared-expert-type3",
    #     method="copy-mlp-as-shared-expert-type3",
    # )
    # upcycle(upcycle_config)

    #  # copy-mlp-as-shared-expert-type4
    # global_args['NUM_EXPERTS'] = 32
    # global_args['TOP_K'] = 8
    # global_args['MOE_EXPANSION_FACTOR'] = 1
    # global_args['SHARED_EXPERT_EXPANSION_FACTOR'] = 4
    # upcycle_config = UpcycleConfig(
    #     source_model_checkpoint=SRC_CKPT,
    #     target_model=build_model_config(routed_expert_norm=True, shared_expert_norm=False, feed_forward_norm=True), # remove the final layer norm
    #     target_model_output_path="/workspace/tmp/upcycled-OLMo-2-0425-1B/copy-mlp-as-shared-expert-type4",
    #     method="copy-mlp-as-shared-expert-type4",
    # )
    # upcycle(upcycle_config)

    # virtual group init
    global_args["NUM_EXPERTS"] = 32
    global_args["TOP_K"] = 8
    global_args["MOE_EXPANSION_FACTOR"] = 1
    global_args["SHARED_EXPERT_EXPANSION_FACTOR"] = 4
    upcycle_config = UpcycleConfig(
        source_model_checkpoint=SRC_CKPT,
        target_model=build_model_config(routed_expert_norm=False, shared_expert_norm=False),
        target_model_output_path="/workspace/tmp/upcycled-OLMo-2-0425-1B/virtual-group-init",
        method="virtual-group",
    )
    upcycle(upcycle_config)
