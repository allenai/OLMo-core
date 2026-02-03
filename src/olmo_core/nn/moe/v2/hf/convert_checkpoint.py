
import json
import os
from dataclasses import replace
from typing import Any, Dict, Generator, Optional, Tuple, Union, Iterable, Sequence
import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
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
        sd_to_load[k] = torch.empty(model_sd_meta[k].size, dtype=torch.float32)

    dist_cp.state_dict_loader.load(
        sd_to_load,
        checkpoint_id=dir,
        storage_reader=reader,
        process_group=process_group,
        # planner=FlatLoadPlanner(),
    )

    return sd_to_load


def load_hf_model_from_olmo_checkpoint(hf_model, olmo_state_dict):

    remaining_hf_keys = set(hf_model.state_dict().keys())

    def _update_param(hf_name, olmo_name_or_fn):
        hf_param = hf_model.state_dict()[hf_name]

        if isinstance(olmo_name_or_fn, str): # direct name mapping
            olmo_param = olmo_state_dict[olmo_name_or_fn]
        else:
            # function to extract from olmo param
            olmo_name, extract_fn = olmo_name_or_fn
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
    for layer_idx in range(num_layers):
        # model.layers.0.self_attn.q_proj.weight -- module.blocks.0.attention.w_q.weight.main
        mapping_hf_to_olmo[f'model.layers.{layer_idx}.self_attn.q_proj.weight'] = f'module.blocks.{layer_idx}.attention.w_q.weight.main'
        mapping_hf_to_olmo[f'model.layers.{layer_idx}.self_attn.k_proj.weight'] = f'module.blocks.{layer_idx}.attention.w_k.weight.main'
        mapping_hf_to_olmo[f'model.layers.{layer_idx}.self_attn.v_proj.weight'] = f'module.blocks.{layer_idx}.attention.w_v.weight.main'
        mapping_hf_to_olmo[f'model.layers.{layer_idx}.self_attn.o_proj.weight'] = f'module.blocks.{layer_idx}.attention.w_out.weight.main'

        # model.layers.0.self_attn.q_norm.weight -- module.blocks.0.attention.q_norm.weight.main
        mapping_hf_to_olmo[f'model.layers.{layer_idx}.self_attn.q_norm.weight'] = f'module.blocks.{layer_idx}.attention.q_norm.weight.main'
        mapping_hf_to_olmo[f'model.layers.{layer_idx}.self_attn.k_norm.weight'] = f'module.blocks.{layer_idx}.attention.k_norm.weight.main'

        if layer_idx in hf_model.config.dense_layers_indices:
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

        # model.layers.0.post_attention_layernorm.weight -- module.blocks.0.attention_norm.weight
        mapping_hf_to_olmo[f'model.layers.{layer_idx}.post_attention_layernorm.weight'] = f'module.blocks.{layer_idx}.attention_norm.weight.main'
        mapping_hf_to_olmo[f'model.layers.{layer_idx}.post_feedforward_layernorm.weight'] = f'module.blocks.{layer_idx}.feed_forward_norm.weight.main'

    # apply rules ##############

    # direct mapping
    for hf_name, olmo_name in mapping_hf_to_olmo.items():
        _update_param(hf_name, olmo_name)


    if len(remaining_hf_keys) > 0:
        raise ValueError(f"Some HF model parameters were not loaded from OLMo checkpoint: {remaining_hf_keys}")


    return hf_model


if __name__ == "__main__":
    CKPT_PATH = '/workspace/checkpoint/OLMoE3-abl-260102-018a_1024d1024a_12L768M768S_32E4K1S_abl/step10000'
    output_path = '/workspace/tmp/step10000_hf_model2'
    main_sd = load_state_dict_direct(
        dir=os.path.join(CKPT_PATH, 'model_and_optim'),
        process_group=None, pre_download=True, work_dir='/workspace/tmp'
    )

    olmo_config_path = os.path.join(CKPT_PATH, 'config.json')
    with open(olmo_config_path, 'r') as f:
        olmo_cfg = json.load(f)

    olmo_model_cfg = olmo_cfg['model']

    # config dense_mlp_intermediate_size
    dense_mlp_intermediate_size = None
    overrides = olmo_model_cfg['block_overrides']
    dense_layers_indices = []
    # find out which layers are dense layers
    for layer_idx_str, layer_cfg in overrides.items():
        if 'feed_forward' in layer_cfg:
            dense_layers_indices.append(int(layer_idx_str))
    
    # pick the first dense layer's feed_forward hidden_size as dense_mlp_intermediate_size
    dense_config = next(iter(overrides.items()))
    dense_mlp_intermediate_size = dense_config[1]['feed_forward']['hidden_size']

    hf_config = Olmo3MoeConfig(
        vocab_size=olmo_model_cfg['vocab_size'],
        hidden_size=olmo_model_cfg['d_model'],
        dense_mlp_intermediate_size=dense_mlp_intermediate_size,
        moe_intermediate_size=olmo_model_cfg['block']['routed_experts']['hidden_size'],
        shared_expert_intermediate_size=olmo_model_cfg['block']['shared_experts']['hidden_size'],
        n_routed_experts=olmo_model_cfg['block']['routed_experts']['num_experts'],
        num_experts_per_tok=olmo_model_cfg['block']['routed_experts_router']['top_k'],
        num_hidden_layers=olmo_model_cfg['n_layers'],
        num_attention_heads=olmo_model_cfg['block']['attention']['n_heads'],
        num_key_value_heads=olmo_model_cfg['block']['attention']['n_kv_heads'],
        hidden_act="silu",
        gating_function=olmo_model_cfg['block']['routed_experts_router']['gating_function'],
        normalize_expert_weights=olmo_model_cfg['block']['routed_experts_router']['normalize_expert_weights'],
        restore_weight_scale=olmo_model_cfg['block']['routed_experts_router']['restore_weight_scale'],
        max_position_embeddings=olmo_cfg['dataset']['sequence_length'],
        initializer_range=olmo_model_cfg['init_std'],
        use_cache=True,
        pad_token_id=olmo_cfg['dataset']['tokenizer']['pad_token_id'],
        bos_token_id=None, # no BOS token
        eos_token_id=olmo_cfg['dataset']['tokenizer']['eos_token_id'],
        tie_word_embeddings=False,
        rope_theta=olmo_model_cfg['block']['attention']['rope']['theta'],
        rope_scaling=None,
        attention_bias=olmo_model_cfg['block']['attention']['bias'],
        attention_dropout=olmo_model_cfg['block']['attention'].get('dropout', 0.0),
        rms_norm_eps=olmo_model_cfg['block']['attention_norm']['eps'], # WARNING: assume all norm layers (attention Q,K, feedforward and lmhead) use the same eps
        sliding_window=max(olmo_model_cfg['block']['attention']['sliding_window']['pattern']),
        use_head_qk_norm=olmo_model_cfg['block']['attention']['use_head_qk_norm'],
        layer_types = None,
        dense_layers_indices=dense_layers_indices
    )

    # create HF model
    hf_model = Olmo3MoeForCausalLM(hf_config)

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
    tok = AutoTokenizer.from_pretrained('allenai/dolma2-tokenizer', use_fast=True)
    tok.save_pretrained(output_path)

    print(f"Saved HF model to {output_path}.")


