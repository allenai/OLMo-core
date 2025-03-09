import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import torch
import torch.distributed as dist
from huggingface_hub import repo_exists
from torch.distributed.tensor import DTensor, distribute_tensor
from transformers import AutoModelForCausalLM, Olmo2Config

from olmo_core.aliases import PathOrStr
from olmo_core.distributed.utils import barrier, get_fs_local_rank, get_full_tensor
from olmo_core.io import copy_dir, file_exists, is_url, upload
from olmo_core.nn.transformer.model import Transformer

try:
    from accelerate import init_empty_weights
except ImportError:

    @contextmanager
    def init_empty_weights(include_buffers: bool = False) -> Generator[None, None, None]:
        log.warning("accelerate not installed, will initialize weights.")
        yield None


log = logging.getLogger(__name__)


BLOCK_STR = "[block]"

# Map of Hugging Face keys to olmo_core keys. Different HF models may use different
# names for a given olmo_core state, but we assume for now that the same HF name does not
# refer to different olmo_core states in different models. That is, we assume that
# the mapping from HF state names to olmo_core state names is many to one.
# TODO: Update this comment
HF_TO_OLMO_CORE_MAPPING: Dict[str, str] = {
    "model.embed_tokens.weight": "embeddings.weight",
    "model.norm.weight": "lm_head.norm.weight",
    "lm_head.weight": "lm_head.w_out.weight",
    # Attention.
    f"model.layers.{BLOCK_STR}.self_attn.q_proj.weight": f"blocks.{BLOCK_STR}.attention.w_q.weight",
    f"model.layers.{BLOCK_STR}.self_attn.k_proj.weight": f"blocks.{BLOCK_STR}.attention.w_k.weight",
    f"model.layers.{BLOCK_STR}.self_attn.v_proj.weight": f"blocks.{BLOCK_STR}.attention.w_v.weight",
    f"model.layers.{BLOCK_STR}.self_attn.o_proj.weight": f"blocks.{BLOCK_STR}.attention.w_out.weight",
    # MLP.
    f"model.layers.{BLOCK_STR}.mlp.gate_proj.weight": f"blocks.{BLOCK_STR}.feed_forward.w1.weight",
    f"model.layers.{BLOCK_STR}.mlp.down_proj.weight": f"blocks.{BLOCK_STR}.feed_forward.w2.weight",
    f"model.layers.{BLOCK_STR}.mlp.up_proj.weight": f"blocks.{BLOCK_STR}.feed_forward.w3.weight",
    # Layer norms.
    f"model.layers.{BLOCK_STR}.input_layernorm.weight": f"blocks.{BLOCK_STR}.attention_norm.weight",
    f"model.layers.{BLOCK_STR}.post_attention_layernorm.weight": f"blocks.{BLOCK_STR}.attention_norm.weight",
    f"model.layers.{BLOCK_STR}.post_feedforward_layernorm.weight": f"blocks.{BLOCK_STR}.feed_forward_norm.weight",
    f"model.layers.{BLOCK_STR}.self_attn.q_norm.weight": f"blocks.{BLOCK_STR}.attention.q_norm.weight",
    f"model.layers.{BLOCK_STR}.self_attn.k_norm.weight": f"blocks.{BLOCK_STR}.attention.k_norm.weight",
}

MODEL_SPECIFIC_HF_TO_OLMO_CORE_MAPPING: Dict[str, Dict[str, str]] = {
    "meta-llama/Llama-3.2-1B": {
        f"model.layers.{BLOCK_STR}.post_attention_layernorm.weight": f"blocks.{BLOCK_STR}.feed_forward_norm.weight"
    }
}

OLMO_CORE_TO_HF_MAPPING: Dict[str, str] = {
    "embeddings.weight": "model.embed_tokens.weight",
    "lm_head.norm.weight": "model.norm.weight",
    "lm_head.w_out.weight": "lm_head.weight",
    # Attention.
    f"blocks.{BLOCK_STR}.attention.w_q.weight": f"model.layers.{BLOCK_STR}.self_attn.q_proj.weight",
    f"blocks.{BLOCK_STR}.attention.w_k.weight": f"model.layers.{BLOCK_STR}.self_attn.k_proj.weight",
    f"blocks.{BLOCK_STR}.attention.w_v.weight": f"model.layers.{BLOCK_STR}.self_attn.v_proj.weight",
    f"blocks.{BLOCK_STR}.attention.w_out.weight": f"model.layers.{BLOCK_STR}.self_attn.o_proj.weight",
    # MLP.
    f"blocks.{BLOCK_STR}.feed_forward.w1.weight": f"model.layers.{BLOCK_STR}.mlp.gate_proj.weight",
    f"blocks.{BLOCK_STR}.feed_forward.w2.weight": f"model.layers.{BLOCK_STR}.mlp.down_proj.weight",
    f"blocks.{BLOCK_STR}.feed_forward.w3.weight": f"model.layers.{BLOCK_STR}.mlp.up_proj.weight",
    # Layer norms.
    f"blocks.{BLOCK_STR}.attention_norm.weight": f"model.layers.{BLOCK_STR}.post_attention_layernorm.weight",
    f"blocks.{BLOCK_STR}.feed_forward_norm.weight": f"model.layers.{BLOCK_STR}.post_feedforward_layernorm.weight",
    f"blocks.{BLOCK_STR}.attention.q_norm.weight": f"model.layers.{BLOCK_STR}.self_attn.q_norm.weight",
    f"blocks.{BLOCK_STR}.attention.k_norm.weight": f"model.layers.{BLOCK_STR}.self_attn.k_norm.weight",
}


def _get_hf_to_olmo_core_mapping(model_id: str | None, n_layers: int) -> Dict[str, str]:
    mapping = {
        k.replace(BLOCK_STR, str(i)): v.replace(BLOCK_STR, str(i))
        for i in range(n_layers)
        for k, v in HF_TO_OLMO_CORE_MAPPING.items()
    }

    if model_id is not None:
        model_specific_mapping = {
            k.replace(BLOCK_STR, str(i)): v.replace(BLOCK_STR, str(i))
            for i in range(n_layers)
            for k, v in MODEL_SPECIFIC_HF_TO_OLMO_CORE_MAPPING.get(model_id, {}).items()
        }
        mapping.update(model_specific_mapping)

    return mapping


def _get_olmo_core_to_hf_mapping(n_layers: int) -> Dict[str, str]:
    return {
        k.replace(BLOCK_STR, str(i)): v.replace(BLOCK_STR, str(i))
        for i in range(n_layers)
        for k, v in OLMO_CORE_TO_HF_MAPPING.items()
    }


def load_hf_checkpoint(
    dir: PathOrStr,
    model_state_dict: Dict[str, Any],
    n_layers: int,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    work_dir: Optional[PathOrStr] = None,
):
    work_dir = f"{work_dir}/hf-tmp" if work_dir is not None else None

    if is_url(dir):
        log.warning(
            "Load path provided is a remote Hugging Face directory. This may not be suitable for unshared file systems."
        )
        assert work_dir is not None
        assert file_exists(f"{dir}/model.safetensors.index.json") or file_exists(
            f"{dir}/pytorch_model.bin"
        )
        model_name_or_path = dir
        model_id = None

        # Download model to local FS
        if get_fs_local_rank() == 0:
            copy_dir(dir, work_dir)
        barrier(group=process_group)
    elif Path(dir).is_dir():
        assert file_exists(f"{dir}/model.safetensors.index.json") or file_exists(
            f"{dir}/pytorch_model.bin"
        )
        model_name_or_path = dir
        model_id = None
    elif repo_exists(str(dir)):
        log.warning(
            "Load path provided is a Hugging Face model id. This may not be suitable for unshared file systems."
        )
        model_name_or_path = dir
        model_id = str(model_name_or_path)
    else:
        raise NotImplementedError

    # Warm up the HF local cache by downloading the model on just local rank 0
    if get_fs_local_rank() == 0:
        hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        del hf_model
    barrier(group=process_group)

    hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    log.info(f"Loaded hf model: {hf_model}")

    if n_layers != len(hf_model.model.layers):
        raise RuntimeError(
            f"Trying to load a HF model with {len(hf_model.model.layers)} layers into a model with {n_layers} layers."
        )

    hf_state_dict: Dict[str, Any] = hf_model.state_dict()

    hf_to_olmo_core_state_mapping = _get_hf_to_olmo_core_mapping(model_id, n_layers)

    for hf_key, hf_state in hf_state_dict.items():
        olmo_core_key = hf_to_olmo_core_state_mapping[hf_key]
        olmo_core_state = model_state_dict[olmo_core_key]

        # Initialize DTensor state from the global HF state tensors
        if isinstance(olmo_core_state, DTensor):
            olmo_core_state = distribute_tensor(
                hf_state, olmo_core_state.device_mesh, olmo_core_state.placements
            )
        else:
            olmo_core_state = hf_state

        model_state_dict[olmo_core_key] = olmo_core_state


def save_as_hf_checkpoint(
    dir: PathOrStr,
    model_state_dict: Dict[str, Any],
    model: Transformer,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    work_dir: Optional[PathOrStr] = None,
    save_overwrite: bool = False,
):
    n_layers = model.n_layers
    olmo_core_to_hf_mapping = _get_olmo_core_to_hf_mapping(n_layers)

    hf_state_dict = {}
    for key, value in model_state_dict.items():
        if isinstance(value, torch.Tensor):
            value = get_full_tensor(value)

        hf_state_dict[olmo_core_to_hf_mapping[key]] = value

    # Create HF model instance and load state dict
    huggingface_config = Olmo2Config(
        vocab_size=model.vocab_size,
        hidden_size=model.d_model,
        intermediate_size=model.blocks["0"].feed_forward.hidden_size,
        num_hidden_layers=model.n_layers,
        num_attention_heads=model.blocks["0"].attention.n_heads,
        num_key_value_heads=model.blocks["0"].attention.n_kv_heads,
        hidden_act="silu",
        max_position_embeddings=-1,
        rope_theta=model.blocks["0"].attention.rope.theta,
        attention_bias=model.blocks["0"].attention.w_out.bias,
        pad_token_id=None,  # type: ignore
        bos_token_id=None,
        eos_token_id=None,  # type: ignore
        rms_norm_eps=model.blocks["0"].feed_forward_norm.eps,
        tie_word_embeddings=False,
    )

    with init_empty_weights():
        log.info("Initializing HF model with empty weights...")
        model = AutoModelForCausalLM.from_config(huggingface_config)

    model.load_state_dict(hf_state_dict, assign=True)

    if get_fs_local_rank(process_group) == 0:
        if is_url(dir):
            assert work_dir is not None
            model.save_pretrained(work_dir)

            target = f"{dir}"
            upload(work_dir, target, save_overwrite=save_overwrite)
        else:
            target = Path(dir)
            if target.is_dir() and not save_overwrite:
                raise FileExistsError(target)
            target.parent.mkdir(exist_ok=True, parents=True)
            model.save_pretrained(target)
