"""
Train a medium OLMoE model. Run this script without any arguments to see usage info.

Virtual-group upcycling: MLP weights are sharded and duplicated across virtual groups of experts, and router weights are duplicated per virtual group.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

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
    use_shared_expert=False,
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
                    router=MoERouterConfig(top_k=TOP_K, normalize_expert_weights=1.0),  # L1 norm
                    shared_mlp=FeedForwardConfig(
                        hidden_size=int(d_model * SHARED_EXPERT_EXPANSION_FACTOR), bias=False
                    )
                    if use_shared_expert
                    else None,
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

    return config


@dataclass
class UpcycleConfig(Config):
    source_model_checkpoint: str
    target_model_output_path: str
    method: str
    target_model: TransformerConfig
    init_seed: int = 2025


from transformers import AutoModelForCausalLM

''' [Routed Expert MLP]
class MoEMLP(MoEMLPBase):
    """
    A basic expert MLP module with SwiGLU activation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        num_experts: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__(d_model=d_model, hidden_size=hidden_size, num_experts=num_experts)
        # NOTE: these parameters need to have a large enough first dimension (which would be num experts)
        # in order to be sharded over big world sizes with FSDP, so we flatten the first 2 dimensions.
        self.w1 = nn.Parameter(
            torch.empty(
                num_experts * d_model,
                hidden_size,
                device=init_device,
                dtype=dtype,
            ),
        )
        self.w2 = nn.Parameter(
            torch.empty(
                num_experts * hidden_size,
                d_model,
                device=init_device,
                dtype=dtype,
            ),
        )
        self.w3 = nn.Parameter(
            torch.empty(
                num_experts * d_model,
                hidden_size,
                device=init_device,
                dtype=dtype,
            ),
        )
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the expert outputs.

        :param x: The input of shape ``(num_local_experts, N, d_model)``.
        """
        og_dtype = x.dtype

        # Scale gradients and get local tensors (in case of expert parallelism).
        # shape (all): (num_local_experts, hidden_size, d_model)
        w1, w2, w3 = (
            get_local_tensor(self.w1.view(self.num_experts, self.d_model, self.hidden_size)),
            get_local_tensor(self.w2.view(self.num_experts, self.hidden_size, self.d_model)),
            get_local_tensor(self.w3.view(self.num_experts, self.d_model, self.hidden_size)),
        )

        x = x.type_as(w1)

        # Compute the MLP.
        return torch.bmm(F.silu(torch.bmm(x, w1)) * torch.bmm(x, w3), w2).to(dtype=og_dtype)
'''

''' [Shared Expert MLP]
class FeedForward(nn.Module):
    """
    Basic feed-forward module with SwiGLU activation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.w1 = nn.Linear(d_model, hidden_size, bias=bias, dtype=dtype, device=init_device)
        self.w2 = nn.Linear(hidden_size, d_model, bias=bias, dtype=dtype, device=init_device)
        self.w3 = nn.Linear(d_model, hidden_size, bias=bias, dtype=dtype, device=init_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the feed-forward on the input ``x``.

        :param x: The input of shape ``(*, d_model)``.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
'''

""" [Dense MLP]
class Olmo2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
"""

""" [Source model example shape]
model.embed_tokens.weight (100352, 2048)
model.layers.0.self_attn.q_proj.weight (2048, 2048)
model.layers.0.self_attn.k_proj.weight (2048, 2048)
model.layers.0.self_attn.v_proj.weight (2048, 2048)
model.layers.0.self_attn.o_proj.weight (2048, 2048)
model.layers.0.self_attn.q_norm.weight (2048,)
model.layers.0.self_attn.k_norm.weight (2048,)
model.layers.0.mlp.gate_proj.weight (8192, 2048)    # (hidden_size, d_model)
model.layers.0.mlp.up_proj.weight (8192, 2048)      # (hidden_size, d_model)
model.layers.0.mlp.down_proj.weight (2048, 8192)    # (d_model, hidden_size)
model.layers.0.post_attention_layernorm.weight (2048,)
model.layers.0.post_feedforward_layernorm.weight (2048,)
model.norm.weight (2048,)
lm_head.weight (100352, 2048)
"""

""" [Target model example shape]
embeddings.weight (100352, 2048)
blocks.0.attention.w_q.weight (2048, 2048)
blocks.0.attention.w_k.weight (2048, 2048)
blocks.0.attention.w_v.weight (2048, 2048)
blocks.0.attention.w_out.weight (2048, 2048)
blocks.0.attention.q_norm.weight (2048,)
blocks.0.attention.k_norm.weight (2048,)
blocks.0.attention_norm.weight (2048,)
blocks.0.feed_forward_moe.router.weight (65536,)
blocks.0.feed_forward_moe.experts.mlp.w1 (65536, 2048)  # (num_experts * d_model, hidden_size)
blocks.0.feed_forward_moe.experts.mlp.w2 (65536, 2048)  # (num_experts * hidden_size, d_model)
blocks.0.feed_forward_moe.experts.mlp.w3 (65536, 2048)  # (num_experts * d_model, hidden_size)
blocks.0.feed_forward_norm.weight (2048,)
lm_head.norm.weight (2048,)
lm_head.w_out.weight (100352, 2048)
"""


def upcycle_virtual_group_init(
    source_model,
    target_model,
):
    # check if target model has shared_mlp
    HAS_SHARED_MLP = False
    for key in target_model.state_dict().keys():
        if "shared_mlp" in key:
            HAS_SHARED_MLP = True
            break

    print(f"Target model has shared_mlp: {HAS_SHARED_MLP}")

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

    if HAS_SHARED_MLP:
        # shared expert is not handled by virtual group init
        untouched_target_keys = [
            "blocks.*.feed_forward_moe.shared_mlp.w1.weight",
            "blocks.*.feed_forward_moe.shared_mlp.w2.weight",
            "blocks.*.feed_forward_moe.shared_mlp.w3.weight",
        ]
    else:
        untouched_target_keys = []

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

    G = inter_dense // expert_hidden  # shards per layer
    assert G * expert_hidden == inter_dense, "expert_hidden does not divide intermediate_size"
    assert NUM_EXPERTS % G == 0, "NUM_EXPERTS must be an integer multiple of shard count G"

    dup_per_shard = NUM_EXPERTS // G  # this is “E” in the paper
    scale = ((dup_per_shard * (G**2)) / TOP_K) ** (1 / 3)  #  reference from paper

    print(
        f"\n[virtual-group-init] Sharding MLP: inter_dense={inter_dense}, "
        f"expert_hidden={expert_hidden}, G={G}, scale={scale:.4f}"
    )

    # ------------------------------------------------------------------
    # 4. Shard & scatter W1/W2/W3 into experts for every layer
    # W1 -> gate
    # W2 -> down
    # W3 -> up
    #
    # dense: down(act_fn(gate(x)) * up(x))
    # moe routed experts: torch.bmm(F.silu(torch.bmm(x, w1)) * torch.bmm(x, w3), w2)
    # moe shared expert: w2(F.silu(w1(x)) * w3(x))
    # ------------------------------------------------------------------
    for layer in range(NUM_LAYERS):
        # dense weights
        gate = source_model.state_dict()[
            f"model.layers.{layer}.mlp.gate_proj.weight"
        ].T  # (d_model, inter)
        down = source_model.state_dict()[
            f"model.layers.{layer}.mlp.down_proj.weight"
        ].T  # (inter, d_model)
        up = source_model.state_dict()[
            f"model.layers.{layer}.mlp.up_proj.weight"
        ].T  # (d_model, inter)

        # target aggregated expert weight tensors
        w1_key = f"blocks.{layer}.feed_forward_moe.experts.mlp.w1"
        w2_key = f"blocks.{layer}.feed_forward_moe.experts.mlp.w2"
        w3_key = f"blocks.{layer}.feed_forward_moe.experts.mlp.w3"
        w1 = target_model.state_dict()[w1_key]  # (num_experts * d_model, hidden_size)
        w2 = target_model.state_dict()[w2_key]  # (num_experts * hidden_size, d_model)
        w3 = target_model.state_dict()[w3_key]  # (num_experts * d_model, hidden_size)

        # w1 is gate, (d_model * NUM_EXPERTS, intermediate_size)
        # w2 is down, (intermediate_size * NUM_EXPERTS, d_model)
        # w3 is up, (d_model * NUM_EXPERTS, intermediate_size)

        w1 = w1.view(NUM_EXPERTS, d_model, expert_hidden)
        w2 = w2.view(NUM_EXPERTS, expert_hidden, d_model)
        w3 = w3.view(NUM_EXPERTS, d_model, expert_hidden)

        for expert_idx in range(NUM_EXPERTS):
            # eg,
            # NUM_EXPERTS=32,
            # G = dense_hidden/expert_hidden=8192/2048=4
            # E = NUM_EXPERTS/G = 32/4=8
            # T = TOP K = 4, same as G to get exact same output as dense model
            # experts:  [0-3] -> one complete dense mlp
            #           [4-7] -> one complete dense mlp, same as [0-3]
            #           ...
            #           [28-31] -> one complete dense mlp, same as [0-3]
            shard_idx = expert_idx % G

            # process W1 & W3
            source_start = shard_idx * expert_hidden
            source_end = (shard_idx + 1) * expert_hidden
            w1[expert_idx, :].copy_(gate[:, source_start:source_end])
            w3[expert_idx, :].copy_(
                up[:, source_start:source_end] * (G**0.5)
            )  # scale up by sqrt(scale)

            # process W2
            w2[expert_idx, :].copy_(down[source_start:source_end, :] * (G**0.5))

        # ---------- router (virtual-group duplication, flattened param) ----------
        router_key = f"blocks.{layer}.feed_forward_moe.router.weight"
        router_w_flat = target_model.state_dict()[router_key]  # (N*d_model,)
        router_2d = router_w_flat.view(NUM_EXPERTS, d_model)  # (N, d_model)

        # prototypes shape: (E, d_model)
        prototypes = router_2d[::G].clone()  # 0,G,2G,…
        for expert_idx in range(NUM_EXPERTS):
            group_id = expert_idx // G  # ← integer-div
            router_2d[expert_idx].copy_(prototypes[group_id])

        # prototypes = router_2d[:G].clone()   # one prototype per shard 0…G-1
        # for expert_idx in range(NUM_EXPERTS):
        #     shard_id   = expert_idx % G          # group by shard
        #     router_2d[expert_idx].copy_(prototypes[shard_id])

        # router_2d[0] == router_2d[1] == ... == router_2d[G-1], and repeat for every virtual group

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
    # compare output
    with torch.no_grad():
        x = torch.randn(2, 4, d_model)  # (bsz, seqlen, dmodel)\

        # get output for dense mlp
        dense_out = source_model.model.layers[0].mlp(x)  # from the source model
        w_gate = source_model.model.layers[0].mlp.gate_proj.weight
        w_up = source_model.model.layers[0].mlp.up_proj.weight
        w_down = source_model.model.layers[0].mlp.down_proj.weight

        dense_up = torch.matmul(x, w_up.T)
        dense_gate = torch.matmul(x, w_gate.T)
        dense_hidden = F.silu(dense_gate) * dense_up
        dense_out2 = torch.matmul(dense_hidden, w_down.T)  # maunal compute
        assert torch.allclose(dense_out, dense_out2)

        '''
        Reference from MoEMLP
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Compute the expert outputs.

            :param x: The input of shape ``(num_local_experts, N, d_model)``.
            """
            og_dtype = x.dtype

            # Scale gradients and get local tensors (in case of expert parallelism).
            # shape (all): (num_local_experts, hidden_size, d_model)
            w1, w2, w3 = (
                get_local_tensor(self.w1.view(self.num_experts, self.d_model, self.hidden_size)),
                get_local_tensor(self.w2.view(self.num_experts, self.hidden_size, self.d_model)),
                get_local_tensor(self.w3.view(self.num_experts, self.d_model, self.hidden_size)),
            )

            x = x.type_as(w1)

            # Compute the MLP.
            return torch.bmm(F.silu(torch.bmm(x, w1)) * torch.bmm(x, w3), w2).to(dtype=og_dtype)
        '''

        # we cannot run the MoEMLP.foward() because it requires GPU kernel and we are on cpu
        # moe_out  = target_model.blocks['0'].feed_forward_moe(x)

        # but we can compute manually
        expert_weights, expert_indices, batch_size_per_expert, aux_loss = target_model.blocks[
            "0"
        ].feed_forward_moe.router(x)
        # expert_weights    # (2, 4, topK)
        # expert_indices    # (2, 4, topK)
        moe_out = torch.zeros_like(dense_out)  # (2, 4, d_model)
        for batch_idx in range(x.shape[0]):
            for token_idx in range(x.shape[1]):
                for local_expert_idx, expert_idx in enumerate(expert_indices[batch_idx][token_idx]):
                    w1 = target_model.blocks["0"].feed_forward_moe.experts.mlp.w1
                    w3 = target_model.blocks["0"].feed_forward_moe.experts.mlp.w3
                    w2 = target_model.blocks["0"].feed_forward_moe.experts.mlp.w2

                    w1, w2, w3 = (
                        w1.view(NUM_EXPERTS, d_model, expert_hidden),
                        w2.view(NUM_EXPERTS, expert_hidden, d_model),
                        w3.view(NUM_EXPERTS, d_model, expert_hidden),
                    )
                    w1 = w1[expert_idx]  # torch.allclose(w1[expert_idx], w_gate.T * scale)
                    w2 = w2[expert_idx]  # torch.allclose(w2[expert_idx], w_down.T * scale)
                    w3 = w3[expert_idx]  # torch.allclose(w3[expert_idx], w_up.T)
                    x_local = x[batch_idx][token_idx]
                    h_gate = x_local @ w1  # h_gate vs dense_gate
                    h_up = x_local @ w3  # h_up vs dense_up
                    hidden = F.silu(h_gate) * h_up  # hidden vs dense_hidden
                    out = hidden @ w2

                    moe_out[batch_idx][token_idx] += (
                        out * expert_weights[batch_idx][token_idx][local_expert_idx]
                    )

        print("‖dense‖:", dense_out.std().item(), "‖moe‖:", moe_out.std().item())

        scaling_factor_array = dense_out / moe_out
        print(scaling_factor_array)
        scaling_factor = scaling_factor_array[0][0][0].item()

        # we should scale up the output of the moe model by scaling_factor
        # so we should scale up and down by sqrt(scaling_factor)

        print(f"E={dup_per_shard}, G={G}, T={TOP_K}, scale={scaling_factor:.4f}")

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
        pass
    # method 2: "copy-mlp-noise"
    elif config.method == "copy-mlp-noise":
        pass
    elif config.method == "copy-mlp-partial-reinit":
        pass
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

    # virtual group init
    global_args["NUM_EXPERTS"] = 32
    global_args["TOP_K"] = 8
    global_args["MOE_EXPANSION_FACTOR"] = 1
    global_args["SHARED_EXPERT_EXPANSION_FACTOR"] = None
    upcycle_config = UpcycleConfig(
        source_model_checkpoint=SRC_CKPT,
        target_model=build_model_config(
            routed_expert_norm=False, shared_expert_norm=False, use_shared_expert=False
        ),
        target_model_output_path="/workspace/tmp/upcycled-OLMo-2-0425-1B/virtual-group-init-E8G4T8",
        method="virtual-group",
    )
    upcycle(upcycle_config)
