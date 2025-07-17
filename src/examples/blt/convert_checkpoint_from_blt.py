from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Optional
import tempfile
import json
import logging
import traceback


import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from torch.distributed.tensor import DTensor, distribute_tensor

from olmo_core.aliases import PathOrStr
from olmo_core.distributed.checkpoint import save_model_and_optim_state
from olmo_core.io import copy_file, file_exists, join_path
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.hf.convert import TemplatePlaceholder, StateConverter, StateMappingTemplate, TemplatePlaceholder
from olmo_core.utils import get_default_device, prepare_cli_environment

log = logging.getLogger(__name__)

EXPERT = TemplatePlaceholder.EXPERT
LAYER = TemplatePlaceholder.LAYER
LOCAL_ENCODER_LAYER = TemplatePlaceholder.LOCAL_ENCODER_LAYER
LOCAL_DECODER_LAYER = TemplatePlaceholder.LOCAL_DECODER_LAYER


BLT_TO_OLMO_CORE_MAPPINGS: Dict[str, str] = {
    ## LOCAL ENCODER
    # Embeddings.
    "local_encoder.tok_embeddings.weight": "local_encoder.embedding.weight",
    f"encoder_hash_tok_embedding.{EXPERT}.weight": f"local_encoder.hash_embeddings.{EXPERT}.weight",
    # Patch projection.
    "local_encoder.patch_embedding_projection.weight": "local_encoder.patch_embedding_projection.weight",
    # Patch -> Byte cross attention.
    "local_encoder.cross_attn_layers.0.cross_attn_norm_q.weight": f"local_encoder.cross_attention.q_norm.weight",
    "local_encoder.cross_attn_layers.0.cross_attn_norm_kv.weight": f"local_encoder.cross_attention.kv_norm.weight",
    "local_encoder.cross_attn_layers.0.wq.weight": f"local_encoder.cross_attention.w_q.weight",
    "local_encoder.cross_attn_layers.0.wk.weight": f"local_encoder.cross_attention.w_k.weight",
    "local_encoder.cross_attn_layers.0.wv.weight": f"local_encoder.cross_attention.w_v.weight",
    "local_encoder.cross_attn_layers.0.wo.weight": f"local_encoder.cross_attention.w_out.weight",
    # Self-Attention.
    f"local_encoder.layers.{LOCAL_ENCODER_LAYER}.attention.wq.weight": f"local_encoder.blocks.{LOCAL_ENCODER_LAYER}.attention.w_q.weight",
    f"local_encoder.layers.{LOCAL_ENCODER_LAYER}.attention.wk.weight": f"local_encoder.blocks.{LOCAL_ENCODER_LAYER}.attention.w_k.weight",
    f"local_encoder.layers.{LOCAL_ENCODER_LAYER}.attention.wv.weight": f"local_encoder.blocks.{LOCAL_ENCODER_LAYER}.attention.w_v.weight",
    f"local_encoder.layers.{LOCAL_ENCODER_LAYER}.attention.wo.weight": f"local_encoder.blocks.{LOCAL_ENCODER_LAYER}.attention.w_out.weight",
    # MLP.
    f"local_encoder.layers.{LOCAL_ENCODER_LAYER}.feed_forward.w1.weight": f"local_encoder.blocks.{LOCAL_ENCODER_LAYER}.feed_forward.w1.weight",
    f"local_encoder.layers.{LOCAL_ENCODER_LAYER}.feed_forward.w2.weight": f"local_encoder.blocks.{LOCAL_ENCODER_LAYER}.feed_forward.w2.weight",
    f"local_encoder.layers.{LOCAL_ENCODER_LAYER}.feed_forward.w3.weight": f"local_encoder.blocks.{LOCAL_ENCODER_LAYER}.feed_forward.w3.weight",
    # Layer Norms.
    f"local_encoder.layers.{LOCAL_ENCODER_LAYER}.attention_norm.weight": f"local_encoder.blocks.{LOCAL_ENCODER_LAYER}.attention_norm.weight",
    f"local_encoder.layers.{LOCAL_ENCODER_LAYER}.ffn_norm.weight": f"local_encoder.blocks.{LOCAL_ENCODER_LAYER}.feed_forward_norm.weight",
    ## GLOBAL (LATENT) TRANSFORMER
    # Self-Attention.
    f"global_transformer.layers.{LAYER}.attention.wq.weight": f"blocks.{LAYER}.attention.w_q.weight",
    f"global_transformer.layers.{LAYER}.attention.wk.weight": f"blocks.{LAYER}.attention.w_k.weight",
    f"global_transformer.layers.{LAYER}.attention.wv.weight": f"blocks.{LAYER}.attention.w_v.weight",
    f"global_transformer.layers.{LAYER}.attention.wo.weight": f"blocks.{LAYER}.attention.w_out.weight",
    # MLP.
    f"global_transformer.layers.{LAYER}.feed_forward.w1.weight": f"blocks.{LAYER}.feed_forward.w1.weight",
    f"global_transformer.layers.{LAYER}.feed_forward.w2.weight": f"blocks.{LAYER}.feed_forward.w2.weight",
    f"global_transformer.layers.{LAYER}.feed_forward.w3.weight": f"blocks.{LAYER}.feed_forward.w3.weight",
    # Layer Norms.
    f"global_transformer.layers.{LAYER}.attention_norm.weight": f"blocks.{LAYER}.attention_norm.weight",
    f"global_transformer.layers.{LAYER}.ffn_norm.weight": f"blocks.{LAYER}.feed_forward_norm.weight",
    ## LOCAL DECODER
    # Patch projection.
    "local_decoder.patch_embedding_projection.weight": "local_decoder.patch_embedding_projection.weight",
    # Cross Attention.
    f"local_decoder.cross_attn_layers.{LOCAL_DECODER_LAYER}.cross_attn_norm_q.weight": f"local_decoder.cross_attentions.{LOCAL_DECODER_LAYER}.q_norm.weight",
    f"local_decoder.cross_attn_layers.{LOCAL_DECODER_LAYER}.cross_attn_norm_kv.weight": f"local_decoder.cross_attentions.{LOCAL_DECODER_LAYER}.kv_norm.weight",
    f"local_decoder.cross_attn_layers.{LOCAL_DECODER_LAYER}.wq.weight": f"local_decoder.cross_attentions.{LOCAL_DECODER_LAYER}.w_q.weight",
    f"local_decoder.cross_attn_layers.{LOCAL_DECODER_LAYER}.wk.weight": f"local_decoder.cross_attentions.{LOCAL_DECODER_LAYER}.w_k.weight",
    f"local_decoder.cross_attn_layers.{LOCAL_DECODER_LAYER}.wv.weight": f"local_decoder.cross_attentions.{LOCAL_DECODER_LAYER}.w_v.weight",
    f"local_decoder.cross_attn_layers.{LOCAL_DECODER_LAYER}.wo.weight": f"local_decoder.cross_attentions.{LOCAL_DECODER_LAYER}.w_out.weight",
    # Self-Attention.
    f"local_decoder.layers.{LOCAL_DECODER_LAYER}.attention.wq.weight": f"local_decoder.blocks.{LOCAL_DECODER_LAYER}.attention.w_q.weight",
    f"local_decoder.layers.{LOCAL_DECODER_LAYER}.attention.wk.weight": f"local_decoder.blocks.{LOCAL_DECODER_LAYER}.attention.w_k.weight",
    f"local_decoder.layers.{LOCAL_DECODER_LAYER}.attention.wv.weight": f"local_decoder.blocks.{LOCAL_DECODER_LAYER}.attention.w_v.weight",
    f"local_decoder.layers.{LOCAL_DECODER_LAYER}.attention.wo.weight": f"local_decoder.blocks.{LOCAL_DECODER_LAYER}.attention.w_out.weight",
    # MLP.
    f"local_decoder.layers.{LOCAL_DECODER_LAYER}.feed_forward.w1.weight": f"local_decoder.blocks.{LOCAL_DECODER_LAYER}.feed_forward.w1.weight",
    f"local_decoder.layers.{LOCAL_DECODER_LAYER}.feed_forward.w2.weight": f"local_decoder.blocks.{LOCAL_DECODER_LAYER}.feed_forward.w2.weight",
    f"local_decoder.layers.{LOCAL_DECODER_LAYER}.feed_forward.w3.weight": f"local_decoder.blocks.{LOCAL_DECODER_LAYER}.feed_forward.w3.weight",
    # Layer Norms.
    f"local_decoder.layers.{LOCAL_DECODER_LAYER}.attention_norm.weight": f"local_decoder.blocks.{LOCAL_DECODER_LAYER}.attention_norm.weight",
    f"local_decoder.layers.{LOCAL_DECODER_LAYER}.ffn_norm.weight": f"local_decoder.blocks.{LOCAL_DECODER_LAYER}.feed_forward_norm.weight",
    # Final layer norm and lm head.
    "local_decoder.norm.weight": "lm_head.norm.weight",
    "local_decoder.output.weight": "lm_head.w_out.weight",
}


def _get_transformer_config(model_arch: str) -> TransformerConfig:
    transformer_configs = {
        "blt_1b": TransformerConfig.blt_1b,
    }

    return transformer_configs[model_arch.lower()]()


def _load_blt_model(
    checkpoint_path: PathOrStr,
    model_state_dict: Dict[str, Any],
    transformer_config_dict: Dict[str, Any],
):
    blt_state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

    mapping_templates = [
        StateMappingTemplate(hf_key, olmo_core_key)
        for hf_key, olmo_core_key in BLT_TO_OLMO_CORE_MAPPINGS.items()
    ]
    converter = StateConverter(mapping_templates)

    placeholder_bounds = {
        TemplatePlaceholder.LAYER: transformer_config_dict["n_layers"],
        TemplatePlaceholder.EXPERT: (
            len(transformer_config_dict["local_encoder"]["hash_byte_group_size"])
            * transformer_config_dict["local_encoder"]["hash_byte_group_nb_functions"]
        ),
        TemplatePlaceholder.LOCAL_ENCODER_LAYER: transformer_config_dict["local_encoder"]["n_layers"],
        TemplatePlaceholder.LOCAL_DECODER_LAYER: transformer_config_dict["local_decoder"]["n_layers"],
    }

    converted_state_dict = converter.convert(blt_state_dict, placeholder_bounds)

    for key in sorted(converted_state_dict.keys()):
        state = converted_state_dict[key]
        olmo_core_state = model_state_dict[key]
        if isinstance(olmo_core_state, DTensor):
            olmo_core_state = distribute_tensor(
                state, olmo_core_state.device_mesh, olmo_core_state.placements
            )
        else:
            olmo_core_state = state

        model_state_dict[key] = olmo_core_state

def convert_checkpoint_from_blt(
    hf_checkpoint_path: str | Path,
    output_path: str | Path,
    transformer_config_dict: Dict[str, Any],
    *,
    max_sequence_length: int = -1,
    device: torch.device | None = None,
) -> None:
    if max_sequence_length <= 0:
        raise ValueError(f"Missing or invalid sequence length: {max_sequence_length}")

    # Remove deprecated transformer config options
    if "compile" in transformer_config_dict:
        del transformer_config_dict["compile"]
    if "dp_config" in transformer_config_dict:
        del transformer_config_dict["dp_config"]
    if "tp_config" in transformer_config_dict:
        del transformer_config_dict["tp_config"]
    if "float8_config" in transformer_config_dict:
        del transformer_config_dict["float8_config"]

    model = TransformerConfig.from_dict(transformer_config_dict).build()
    device = device or get_default_device()
    model.to_empty(device=device)

    state_dict_options = dist_cp_sd.StateDictOptions(
        flatten_optimizer_state_dict=True, cpu_offload=True
    )
    model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)
    _load_blt_model(hf_checkpoint_path, model_state_dict, transformer_config_dict)

    model.load_state_dict(model_state_dict)

    model_and_optim_dir = join_path(output_path, "model_and_optim")
    log.info(f"Saving OLMo core checkpoint to '{model_and_optim_dir}'")
    save_model_and_optim_state(model_and_optim_dir, model, save_overwrite=True)
    log.info(f"Successfully saved converted model to '{output_path}'")

    config_path = join_path(output_path, "config.json")
    log.info(f"Writing partial experiment config to '{config_path}'")
    experiment_config_dict = {
        "model": transformer_config_dict,
    }

    with tempfile.NamedTemporaryFile(mode="w") as temp_file:
        json.dump(experiment_config_dict, temp_file)
        copy_file(temp_file.name, config_path, save_overwrite=True)
        log.info(f"Successfully wrote partial experiment config to '{config_path}'")

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--checkpoint-input-path",
        type=str,
        required=True,
        help="Local or remote directory containing the HF checkpoint, or the model id of a HF Hub repo.",
    )
    parser.add_argument(
        "-m",
        "--model-arch",
        help="OLMo Core model architecture corresponding to the HF model. New architectures should be added to ``_get_transformer_config``. This is required when an OLMo Core experiment config is not provided.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Local or remote directory where the converted checkpoint should be saved.",
    )
    parser.add_argument(
        "-s",
        "--max-sequence-length",
        type=int,
        required=True,
        help="Max sequence length supported by the model.",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        help="The device on which conversion and validation occurs. Defaults to CUDA or MPS if available and initialized.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    transformer_config_dict = None
    transformer_config_dict = _get_transformer_config(args.model_arch).as_config_dict()

    assert transformer_config_dict is not None

    convert_checkpoint_from_blt(
        hf_checkpoint_path=args.checkpoint_input_path,
        output_path=args.output_dir,
        transformer_config_dict=transformer_config_dict,
        max_sequence_length=args.max_sequence_length,
        device=args.device,
    )


if __name__ == "__main__":
    prepare_cli_environment()
    try:
        main()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        traceback.print_exc()
        import ipdb; ipdb.post_mortem()