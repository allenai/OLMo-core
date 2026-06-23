
import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from olmo_core.distributed.checkpoint import RemoteFileSystemReader
from olmo_core.aliases import PathOrStr
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM, Olmo3MoeModel
from functools import partial
import sys
import importlib.util
from pathlib import Path
from transformers import AutoTokenizer


def import_module_from_path(path: str, module_name: str):
    path = str(Path(path).resolve())
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import from path: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod  # so imports inside it can reference it
    spec.loader.exec_module(mod)
    return mod

def load_state_dict_direct(
    dir: PathOrStr,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    pre_download: bool = False,
    work_dir: Optional[PathOrStr] = None,
    thread_count: Optional[int] = None,
):
    from olmo_core.io import normalize_path

    # sd_to_load = self.optim.state_dict()

    dir = normalize_path(dir)
    reader = RemoteFileSystemReader(
        dir, 
        thread_count=thread_count, 
        pre_download=pre_download, work_dir=work_dir
    )

    metadata = reader.read_metadata()
    # example: 'module.blocks.0.attention.w_q.weight.main'
    model_sd_meta = {k: v for k, v in metadata.state_dict_metadata.items() if k.endswith('main')}
    sd_to_load = {}
    for k in model_sd_meta.keys():
        props = getattr(model_sd_meta[k], "properties", None)
        dtype = getattr(props, "dtype", torch.float32)
        sd_to_load[k] = torch.empty(model_sd_meta[k].size, dtype=dtype)

    dist_cp.state_dict_loader.load(
        sd_to_load,
        checkpoint_id=dir,
        storage_reader=reader,
        process_group=process_group,
        # planner=FlatLoadPlanner(),
    )

    return sd_to_load


def _get_attention_cfg(block_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the attention config for old MoE configs and new OLMo DDP configs."""
    if "attention" in block_cfg:
        return block_cfg["attention"]
    if "sequence_mixer" in block_cfg:
        sequence_mixer = block_cfg["sequence_mixer"]
        if sequence_mixer.get("type", "attention") != "attention":
            raise ValueError(f"Unsupported sequence_mixer type: {sequence_mixer.get('type')}")
        return sequence_mixer
    raise KeyError("Expected block config to contain either 'attention' or 'sequence_mixer'")


def _block_cfg_for_layer(olmo_model_cfg: Dict[str, Any], layer_idx: int) -> Dict[str, Any]:
    overrides = olmo_model_cfg.get("block_overrides") or {}
    return overrides.get(str(layer_idx), overrides.get(layer_idx, olmo_model_cfg["block"]))


def _is_dense_block(block_cfg: Dict[str, Any]) -> bool:
    if "feed_forward" in block_cfg:
        return True
    return block_cfg.get("routed_experts") is None


def _dense_hidden_size(block_cfg: Dict[str, Any]) -> int:
    if "feed_forward" in block_cfg:
        return block_cfg["feed_forward"]["hidden_size"]
    shared_experts = block_cfg.get("shared_experts")
    if shared_experts is None:
        raise ValueError("Dense OLMo DDP blocks are expected to use shared_experts")
    if shared_experts.get("num_experts", 1) != 1:
        raise ValueError("HF dense MLP conversion only supports dense blocks with one shared expert")
    return shared_experts["hidden_size"]


def _build_layer_types(attention_cfg: Dict[str, Any], num_layers: int) -> list[str]:
    sliding_window = attention_cfg.get("sliding_window")
    if not sliding_window:
        return ["full_attention"] * num_layers

    pattern = sliding_window.get("pattern")
    if not pattern:
        return ["full_attention"] * num_layers

    layer_types = []
    force_first = sliding_window.get("force_full_attention_on_first_layer", False)
    force_last = sliding_window.get("force_full_attention_on_last_layer", False)
    for layer_idx in range(num_layers):
        if (layer_idx == 0 and force_first) or (layer_idx == num_layers - 1 and force_last):
            layer_types.append("full_attention")
            continue

        window = pattern[layer_idx % len(pattern)]
        layer_types.append("full_attention" if window == -1 else "sliding_attention")
    return layer_types


def build_hf_config_from_olmo_config(olmo_cfg: Dict[str, Any]) -> Olmo3MoeConfig:
    olmo_model_cfg = olmo_cfg["model"]
    num_layers = olmo_model_cfg["n_layers"]

    dense_layers_indices = []
    dense_mlp_intermediate_size = None
    first_moe_block_cfg = None
    for layer_idx in range(num_layers):
        block_cfg = _block_cfg_for_layer(olmo_model_cfg, layer_idx)
        if _is_dense_block(block_cfg):
            dense_layers_indices.append(layer_idx)
            if dense_mlp_intermediate_size is None:
                dense_mlp_intermediate_size = _dense_hidden_size(block_cfg)
        elif first_moe_block_cfg is None:
            first_moe_block_cfg = block_cfg

    if dense_mlp_intermediate_size is None:
        dense_mlp_intermediate_size = olmo_model_cfg["block"].get("feed_forward", {}).get("hidden_size")
    if dense_mlp_intermediate_size is None:
        dense_mlp_intermediate_size = olmo_model_cfg["block"]["shared_experts"]["hidden_size"]
    if first_moe_block_cfg is None:
        first_moe_block_cfg = olmo_model_cfg["block"]

    attention_cfg = _get_attention_cfg(olmo_model_cfg["block"])
    d_attn = attention_cfg.get("d_attn", olmo_model_cfg["d_model"])
    head_dim = attention_cfg.get("head_dim", d_attn // attention_cfg["n_heads"])
    sliding_window_cfg = attention_cfg.get("sliding_window") or {}
    sliding_pattern = sliding_window_cfg.get("pattern") or [-1]
    positive_windows = [window for window in sliding_pattern if window and window > 0]

    routed_router_cfg = first_moe_block_cfg["routed_experts_router"]
    routed_experts_cfg = first_moe_block_cfg["routed_experts"]
    shared_experts_cfg = first_moe_block_cfg.get("shared_experts")
    rope_cfg = attention_cfg.get("rope") or {}

    return Olmo3MoeConfig(
        vocab_size=olmo_model_cfg["vocab_size"],
        hidden_size=olmo_model_cfg["d_model"],
        attention_hidden_size=d_attn,
        head_dim=head_dim,
        dense_mlp_intermediate_size=dense_mlp_intermediate_size,
        moe_intermediate_size=routed_experts_cfg["hidden_size"],
        shared_expert_intermediate_size=(
            shared_experts_cfg["hidden_size"] if shared_experts_cfg is not None else None
        ),
        n_routed_experts=routed_experts_cfg["num_experts"],
        num_experts_per_tok=routed_router_cfg["top_k"],
        num_hidden_layers=num_layers,
        num_attention_heads=attention_cfg["n_heads"],
        num_key_value_heads=attention_cfg.get("n_kv_heads"),
        hidden_act="silu",
        gating_function=routed_router_cfg["gating_function"],
        normalize_expert_weights=routed_router_cfg.get("normalize_expert_weights"),
        restore_weight_scale=routed_router_cfg.get("restore_weight_scale", True),
        max_position_embeddings=olmo_cfg["dataset"]["sequence_length"],
        initializer_range=olmo_model_cfg["init_std"],
        use_cache=True,
        pad_token_id=olmo_cfg["dataset"]["tokenizer"]["pad_token_id"],
        bos_token_id=None,  # no BOS token
        eos_token_id=olmo_cfg["dataset"]["tokenizer"]["eos_token_id"],
        tie_word_embeddings=False,
        rope_theta=rope_cfg.get("theta", 10000.0),
        rope_scaling=rope_cfg.get("scaling"),
        attention_bias=attention_cfg.get("bias", False),
        attention_dropout=attention_cfg.get("dropout", 0.0),
        rms_norm_eps=olmo_model_cfg["block"]["attention_norm"]["eps"],
        sliding_window=max(positive_windows) if positive_windows else None,
        use_head_qk_norm=attention_cfg.get("use_head_qk_norm", False),
        layer_types=_build_layer_types(attention_cfg, num_layers),
        dense_layers_indices=dense_layers_indices,
        original_num_experts_per_tok=routed_router_cfg.get("original_top_k"),
        embed_scale=olmo_model_cfg.get("embed_scale", 1.0),
        embed_norm=olmo_model_cfg.get("embedding_norm") is not None,
        use_peri_ln=olmo_model_cfg["block"].get("use_peri_norm", False),
    )


def load_hf_model_from_olmo_checkpoint(hf_model, olmo_state_dict):

    hf_state_dict = hf_model.state_dict()
    olmo_keys = set(olmo_state_dict.keys())
    remaining_hf_keys = set(hf_state_dict.keys())

    def _has_olmo_key(name: str) -> bool:
        return name in olmo_keys

    def _update_param(hf_name, olmo_name_or_fn):
        hf_param = hf_state_dict[hf_name]

        if isinstance(olmo_name_or_fn, str): # direct name mapping
            if olmo_name_or_fn not in olmo_state_dict:
                raise KeyError(f"Missing OLMo checkpoint tensor for {hf_name}: {olmo_name_or_fn}")
            olmo_param = olmo_state_dict[olmo_name_or_fn]
        else:
            # function to extract from olmo param
            olmo_name, extract_fn = olmo_name_or_fn
            if olmo_name not in olmo_state_dict:
                raise KeyError(f"Missing OLMo checkpoint tensor for {hf_name}: {olmo_name}")
            olmo_param = olmo_state_dict[olmo_name]
            olmo_param = extract_fn(olmo_param)

        if hf_param.numel() != olmo_param.numel(): # olmo checkpoint saved as 1D tensors
            raise ValueError(f"numel mismatch for {hf_name}: HF shape {hf_param.shape}, OLMo shape {olmo_param.shape}")
        hf_param.copy_(olmo_param.reshape(hf_param.shape))
        remaining_hf_keys.remove(hf_name)


    # update rules ##############

    mapping_hf_to_olmo: Dict[str, Union[str, Tuple[str, Any]]] = {
        # embed
        'model.embed_tokens.weight': 'module.embeddings.weight.main',

        # lm head
        'lm_head.weight': 'module.lm_head.w_out.weight.main',
        'model.norm.weight':'module.lm_head.norm.weight.main',
    }

    if hf_model.config.embed_norm:
        mapping_hf_to_olmo['model.embed_norm.weight'] = 'module.embedding_norm.weight.main'

    def _extract_qkv_proj(olmo_w_qkv, q_dim, kv_dim, d_model, weight_name):
        W = olmo_w_qkv.reshape(q_dim + 2 * kv_dim, d_model)
        W_q, W_k, W_v = W.split((q_dim, kv_dim, kv_dim), dim=0)
        if weight_name == "q":
            return W_q.contiguous()
        if weight_name == "k":
            return W_k.contiguous()
        if weight_name == "v":
            return W_v.contiguous()
        raise ValueError(f"Unknown weight_name: {weight_name}")

    def _extract_experts_up_gate_proj(olmo_w_up_gate, num_experts, expert_hidden_size, d_model, weight_name, expert_idx):
        W = olmo_w_up_gate.reshape(num_experts, 2 * expert_hidden_size, d_model)  # (E, 2H, D)
        W_up   = W[:, :expert_hidden_size, :]                                     # (E, H, D)
        W_gate = W[:, expert_hidden_size:, :]                                     # (E, H, D)

        if weight_name == "up":
            return W_up[expert_idx].contiguous()      # (H, D) matches HF up_proj.weight
        elif weight_name == "gate":
            return W_gate[expert_idx].contiguous()    # (H, D) matches HF gate_proj.weight
        else:
            raise ValueError(f"Unknown weight_name: {weight_name}")

        
    def _extract_expert_down_proj(olmo_w_down, num_experts, expert_hidden_size, d_model, expert_idx):
        # checkpoint stores it flattened; restore to (E, H, D)
        olmo_w_down = olmo_w_down.reshape(num_experts, expert_hidden_size, d_model)  # (E, H, D)
        # HF down_proj.weight expects (D, H)
        return olmo_w_down[expert_idx].T.contiguous()  # (D, H)


    def _extract_shared_expert_up_or_gate_for_hf(olmo_w_up_gate, expert_hidden_size, d_model, weight_name):
        # Accept 1D / 2D; normalize to (D, 2H)
        W = olmo_w_up_gate.reshape(d_model, 2 * expert_hidden_size)  # (D, 2H)
        W_up   = W[:, :expert_hidden_size]                            # (D, H)
        W_gate = W[:, expert_hidden_size:2 * expert_hidden_size]      # (D, H)

        if weight_name == "up":
            return W_up.T.contiguous()      # (H, D) for nn.Linear.weight
        elif weight_name == "gate":
            return W_gate.T.contiguous()    # (H, D)
        else:
            raise ValueError(f"Unknown weight_name: {weight_name}")

    def _extract_shared_expert_down_for_hf(olmo_w_down, expert_hidden_size, d_model):
        # Accept 1D / 2D / 3D; normalize to (1, H, D)
        Wd = olmo_w_down.reshape(1, expert_hidden_size, d_model)[0]  # (H, D)
        return Wd.T.contiguous()                                     # (D, H)



    # layers
    num_layers = hf_model.config.num_hidden_layers
    q_dim = hf_model.config.num_attention_heads * hf_model.config.head_dim
    kv_dim = hf_model.config.num_key_value_heads * hf_model.config.head_dim
    for layer_idx in range(num_layers):
        # model.layers.0.self_attn.q_proj.weight -- module.blocks.0.attention.w_q.weight.main
        qkv_key = f'module.blocks.{layer_idx}.attention.w_qkv.weight.main'
        if _has_olmo_key(qkv_key):
            mapping_hf_to_olmo[f'model.layers.{layer_idx}.self_attn.q_proj.weight'] = (
                qkv_key,
                partial(
                    _extract_qkv_proj,
                    q_dim=q_dim,
                    kv_dim=kv_dim,
                    d_model=hf_model.config.hidden_size,
                    weight_name='q',
                ),
            )
            mapping_hf_to_olmo[f'model.layers.{layer_idx}.self_attn.k_proj.weight'] = (
                qkv_key,
                partial(
                    _extract_qkv_proj,
                    q_dim=q_dim,
                    kv_dim=kv_dim,
                    d_model=hf_model.config.hidden_size,
                    weight_name='k',
                ),
            )
            mapping_hf_to_olmo[f'model.layers.{layer_idx}.self_attn.v_proj.weight'] = (
                qkv_key,
                partial(
                    _extract_qkv_proj,
                    q_dim=q_dim,
                    kv_dim=kv_dim,
                    d_model=hf_model.config.hidden_size,
                    weight_name='v',
                ),
            )
        else:
            mapping_hf_to_olmo[f'model.layers.{layer_idx}.self_attn.q_proj.weight'] = f'module.blocks.{layer_idx}.attention.w_q.weight.main'
            mapping_hf_to_olmo[f'model.layers.{layer_idx}.self_attn.k_proj.weight'] = f'module.blocks.{layer_idx}.attention.w_k.weight.main'
            mapping_hf_to_olmo[f'model.layers.{layer_idx}.self_attn.v_proj.weight'] = f'module.blocks.{layer_idx}.attention.w_v.weight.main'
        mapping_hf_to_olmo[f'model.layers.{layer_idx}.self_attn.o_proj.weight'] = f'module.blocks.{layer_idx}.attention.w_out.weight.main'

        # model.layers.0.self_attn.q_norm.weight -- module.blocks.0.attention.q_norm.weight.main
        mapping_hf_to_olmo[f'model.layers.{layer_idx}.self_attn.q_norm.weight'] = f'module.blocks.{layer_idx}.attention.q_norm.weight.main'
        mapping_hf_to_olmo[f'model.layers.{layer_idx}.self_attn.k_norm.weight'] = f'module.blocks.{layer_idx}.attention.k_norm.weight.main'

        if layer_idx in hf_model.config.dense_layers_indices:
            # Older checkpoints used feed_forward.{w1,w2,w3}. OLMo DDP dense
            # blocks are represented as one shared expert with no routed expert.
            dense_shared_up_gate_key = f'module.blocks.{layer_idx}.shared_experts.w_up_gate.main'
            dense_shared_down_key = f'module.blocks.{layer_idx}.shared_experts.w_down.main'
            if _has_olmo_key(dense_shared_up_gate_key):
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.mlp.gate_proj.weight'] = (
                    dense_shared_up_gate_key,
                    partial(_extract_shared_expert_up_or_gate_for_hf,
                            expert_hidden_size=hf_model.config.dense_mlp_intermediate_size,
                            d_model=hf_model.config.hidden_size,
                            weight_name='gate')
                )
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.mlp.up_proj.weight'] = (
                    dense_shared_up_gate_key,
                    partial(_extract_shared_expert_up_or_gate_for_hf,
                            expert_hidden_size=hf_model.config.dense_mlp_intermediate_size,
                            d_model=hf_model.config.hidden_size,
                            weight_name='up')
                )
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.mlp.down_proj.weight'] = (
                    dense_shared_down_key,
                    partial(_extract_shared_expert_down_for_hf,
                            expert_hidden_size=hf_model.config.dense_mlp_intermediate_size,
                            d_model=hf_model.config.hidden_size)
                )
            else:
                # case: dense layer
                # model.layers.0.mlp.gate_proj.weight -- module.blocks.0.feed_forward.w1.weight.main
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.mlp.gate_proj.weight'] = f'module.blocks.{layer_idx}.feed_forward.w1.weight.main'
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.mlp.up_proj.weight'] = f'module.blocks.{layer_idx}.feed_forward.w3.weight.main'
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.mlp.down_proj.weight'] = f'module.blocks.{layer_idx}.feed_forward.w2.weight.main'
        else:
            # case: moe layer

            # router
            # model.layers.1.mlp.gate.weight.weight -- module.blocks.1.routed_experts_router.weight.main
            mapping_hf_to_olmo[f'model.layers.{layer_idx}.mlp.router.gate.weight'] = f'module.blocks.{layer_idx}.routed_experts_router.weight.main'

            if hf_model.config.shared_expert_intermediate_size is not None:
                # shared expert (FIXED)
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight'] = (
                    f'module.blocks.{layer_idx}.shared_experts.w_up_gate.main',
                    partial(_extract_shared_expert_up_or_gate_for_hf,
                            expert_hidden_size=hf_model.config.shared_expert_intermediate_size,
                            d_model=hf_model.config.hidden_size,
                            weight_name='gate')
                )

                mapping_hf_to_olmo[f'model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight'] = (
                    f'module.blocks.{layer_idx}.shared_experts.w_up_gate.main',
                    partial(_extract_shared_expert_up_or_gate_for_hf,
                            expert_hidden_size=hf_model.config.shared_expert_intermediate_size,
                            d_model=hf_model.config.hidden_size,
                            weight_name='up')
                )

                mapping_hf_to_olmo[f'model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight'] = (
                    f'module.blocks.{layer_idx}.shared_experts.w_down.main',
                    partial(_extract_shared_expert_down_for_hf,
                            expert_hidden_size=hf_model.config.shared_expert_intermediate_size,
                            d_model=hf_model.config.hidden_size)
                )

            # routed experts
            for expert_idx in range(hf_model.config.n_routed_experts):
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight'] = (
                    f'module.blocks.{layer_idx}.routed_experts.w_up_gate.main',
                    partial(_extract_experts_up_gate_proj,
                            num_experts=hf_model.config.n_routed_experts,
                            expert_hidden_size=hf_model.config.moe_intermediate_size,
                            d_model=hf_model.config.hidden_size,
                            weight_name='gate',
                            expert_idx=expert_idx)
                )
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight'] = (
                    f'module.blocks.{layer_idx}.routed_experts.w_up_gate.main',
                    partial(_extract_experts_up_gate_proj,
                            num_experts=hf_model.config.n_routed_experts,
                            expert_hidden_size=hf_model.config.moe_intermediate_size,
                            d_model=hf_model.config.hidden_size,
                            weight_name='up',
                            expert_idx=expert_idx)
                )

                mapping_hf_to_olmo[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight'] = (
                    f'module.blocks.{layer_idx}.routed_experts.w_down.main',
                    partial(_extract_expert_down_proj,
                            num_experts=hf_model.config.n_routed_experts,
                            expert_hidden_size=hf_model.config.moe_intermediate_size,    
                            d_model=hf_model.config.hidden_size,
                            expert_idx=expert_idx)
                )


        # peri ln
        if hf_model.config.use_peri_ln:
            # OLMo DDP peri-norm names are consistent for dense and MoE blocks:
            # input norms are attention_input_norm/feed_forward_input_norm and
            # output norms are attention_norm/feed_forward_norm. Older dense
            # checkpoints used attention_norm/feed_forward_norm as input norms
            # and post_* names as output norms.
            if _has_olmo_key(f'module.blocks.{layer_idx}.attention_input_norm.weight.main'):
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.pre_attention_layernorm.weight'] = f'module.blocks.{layer_idx}.attention_input_norm.weight.main'
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.pre_feedforward_layernorm.weight'] = f'module.blocks.{layer_idx}.feed_forward_input_norm.weight.main'
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.post_attention_layernorm.weight'] = f'module.blocks.{layer_idx}.attention_norm.weight.main'
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.post_feedforward_layernorm.weight'] = f'module.blocks.{layer_idx}.feed_forward_norm.weight.main'
            elif layer_idx in hf_model.config.dense_layers_indices:
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.pre_attention_layernorm.weight'] = f'module.blocks.{layer_idx}.attention_norm.weight.main'
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.pre_feedforward_layernorm.weight'] = f'module.blocks.{layer_idx}.feed_forward_norm.weight.main'
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.post_attention_layernorm.weight'] = f'module.blocks.{layer_idx}.post_attention_norm.weight.main'
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.post_feedforward_layernorm.weight'] = f'module.blocks.{layer_idx}.post_feed_forward_norm.weight.main'
            else:
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.pre_attention_layernorm.weight'] = f'module.blocks.{layer_idx}.attention_input_norm.weight.main'
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.pre_feedforward_layernorm.weight'] = f'module.blocks.{layer_idx}.feed_forward_input_norm.weight.main'
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.post_attention_layernorm.weight'] = f'module.blocks.{layer_idx}.attention_norm.weight.main'
                mapping_hf_to_olmo[f'model.layers.{layer_idx}.post_feedforward_layernorm.weight'] = f'module.blocks.{layer_idx}.feed_forward_norm.weight.main'
        else:
            # reordered norm, consistent between dense and moe layers
            # model.layers.0.post_attention_layernorm.weight -- module.blocks.0.attention_norm.weight
            mapping_hf_to_olmo[f'model.layers.{layer_idx}.post_attention_layernorm.weight'] = f'module.blocks.{layer_idx}.attention_norm.weight.main'
            mapping_hf_to_olmo[f'model.layers.{layer_idx}.post_feedforward_layernorm.weight'] = f'module.blocks.{layer_idx}.feed_forward_norm.weight.main'

    # apply rules ##############

    # map
    for hf_name, olmo_name in mapping_hf_to_olmo.items():
        print(f"{olmo_name} --> {hf_name}")
        _update_param(hf_name, olmo_name)


    if len(remaining_hf_keys) > 0:
        raise ValueError(f"Some HF model parameters were not loaded from OLMo checkpoint: {remaining_hf_keys}")


    return hf_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OLMo checkpoint to HuggingFace format")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to the OLMo checkpoint directory"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the converted HuggingFace model"
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="/workspace/tmp",
        help="Working directory for pre-downloads (default: /workspace/tmp)"
    )
    args = parser.parse_args()

    CKPT_PATH = args.ckpt_path
    output_path = args.output_path
    work_dir = args.work_dir

    olmo_config_path = os.path.join(CKPT_PATH, 'config.json')
    with open(olmo_config_path, 'r') as f:
        olmo_cfg = json.load(f)

    print("Loaded OLMo config:")
    print(olmo_cfg)
    print()

    hf_config = build_hf_config_from_olmo_config(olmo_cfg)

    print("Constructed HF config:")
    print(hf_config)
    print()
    # create HF model
    hf_model = Olmo3MoeForCausalLM(hf_config)
    print("Created HF model - Done.")

    print(f"Loading OLMo checkpoint from {CKPT_PATH}...")
    main_sd = load_state_dict_direct(
        dir=os.path.join(CKPT_PATH, 'model_and_optim'),
        process_group=None, pre_download=True, work_dir=work_dir
    )
    print("Loaded OLMo checkpoint state dict.")

    load_hf_model_from_olmo_checkpoint(hf_model, main_sd)
    print("Loaded HF model from OLMo checkpoint.")

    Olmo3MoeConfig.register_for_auto_class()  # AutoConfig
    Olmo3MoeForCausalLM.register_for_auto_class("AutoModelForCausalLM")


    # Olmo3MoeModel.register_for_auto_class("AutoModel") # this does not change config.json
    # workaround: manually set auto_map in config to include AutoModel
    auto_map = getattr(hf_model.config, "auto_map", None)
    if auto_map is None:
        hf_model.config.auto_map = {}
    hf_model.config.auto_map["AutoModel"] = "modeling_olmo3moe.Olmo3MoeModel"

    hf_model.save_pretrained(output_path)

    # save tokenizer
    tokenizer_id = olmo_cfg["dataset"]["tokenizer"].get("identifier", "allenai/dolma2-tokenizer")
    tok = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    tok.save_pretrained(output_path)

    print(f"Saved HF model to {output_path}.")
