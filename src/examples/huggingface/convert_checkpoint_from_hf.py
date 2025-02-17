"""
Example script showing how you could convert model weights on HuggingFace for an OLMo2 or Llama-3.*
model into a format that can be loaded by OLMo-core for fine-tuning.

Note that this script is architecture-dependent, meaning it may only work for OLMo2/Llama models on
HuggingFace.
"""

import json
import logging
import os

import torch
from transformers import AutoModelForCausalLM

from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_state_dict
from olmo_core.io import clear_directory, dir_is_empty
from olmo_core.nn.rope import RoPELlamaScalingConfig, RoPELinearScalingConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import get_default_device, prepare_cli_environment

log = logging.getLogger(__name__)

# HF_MODEL = f"{os.environ['SHARE_RES_DIR']}/models/deepseek/deepseek-coder-1.3b-base"
HF_MODEL = f"/home1/09636/zyliu/scratch/base_models/deepseek/hf/deepseek-coder-1.3b-base"
# HF_MODEL = "/home/zliu/shared_resources/models/llama3/hf/Llama-3.2-1B"
# HF_MODEL = ""

SAVE_PATH = f"/home1/09636/zyliu/scratch/base_models/deepseek/olmo/deepseek-coder-1.3b-base"
# SAVE_PATH = "/home/zliu/shared_resources/models/llama3/olmo/Llama-3.2-1B"
SAVE_OVERWRITE = True

# TOKENIZER_CONFIG = TokenizerConfig.from_hf(HF_MODEL)
TOKENIZER_CONFIG = TokenizerConfig.from_hf("deepseek-ai/deepseek-coder-1.3b-base")
# TOKENIZER_CONFIG = TokenizerConfig.from_hf("meta-llama/Llama-3.2-1B")
MODEL_CONFIG: TransformerConfig
if "Llama-3.2-1B" in HF_MODEL:
    MODEL_CONFIG = TransformerConfig.llama3_1B(
        TOKENIZER_CONFIG.vocab_size,
        fused_ops=False,
        use_flash=False,
        rope_scaling=RoPELlamaScalingConfig(),
    )

elif "deepseek-coder-1.3b-base" in HF_MODEL:
    MODEL_CONFIG = TransformerConfig.deepseek_1B(
        TOKENIZER_CONFIG.vocab_size,
        fused_ops=False,
        use_flash=False,
        rope_scaling=RoPELinearScalingConfig(factor=4.0),
    )

elif HF_MODEL.startswith("allenai/OLMo-2-1124-7B"):
    MODEL_CONFIG = TransformerConfig.olmo2_7B(
        TOKENIZER_CONFIG.vocab_size,
        fused_ops=False,
        use_flash=False,
    )
elif HF_MODEL.startswith("allenai/OLMo-2-1124-13B"):
    MODEL_CONFIG = TransformerConfig.olmo2_13B(
        TOKENIZER_CONFIG.vocab_size,
        fused_ops=False,
        use_flash=False,
    )
else:
    raise NotImplementedError(HF_MODEL)


def convert_checkpoint() -> AutoModelForCausalLM:
    log.info(f"Loading HF checkpoint '{HF_MODEL}'")
    hf_model = AutoModelForCausalLM.from_pretrained(HF_MODEL)
    print(hf_model)

    if not dir_is_empty(SAVE_PATH):
        if SAVE_OVERWRITE:
            log.warning(f"Clearing existing checkpoint at '{SAVE_PATH}'")
            clear_directory(SAVE_PATH)
        else:
            log.warning(f"Using existing checkpoint at '{SAVE_PATH}'")
            return hf_model

    n_layers = len(hf_model.model.layers)
    state_dict = hf_model.state_dict()

    # Map old keys to OLMo-core keys.
    new_state_dict = {
        "embeddings.weight": state_dict.pop("model.embed_tokens.weight"),
        "lm_head.norm.weight": state_dict.pop("model.norm.weight"),
        "lm_head.w_out.weight": state_dict.pop("lm_head.weight"),
    }
    for block in range(n_layers):
        # Attention.
        new_state_dict[f"blocks.{block}.attention.w_q.weight"] = state_dict.pop(
            f"model.layers.{block}.self_attn.q_proj.weight"
        )
        new_state_dict[f"blocks.{block}.attention.w_k.weight"] = state_dict.pop(
            f"model.layers.{block}.self_attn.k_proj.weight"
        )
        new_state_dict[f"blocks.{block}.attention.w_v.weight"] = state_dict.pop(
            f"model.layers.{block}.self_attn.v_proj.weight"
        )
        new_state_dict[f"blocks.{block}.attention.w_out.weight"] = state_dict.pop(
            f"model.layers.{block}.self_attn.o_proj.weight"
        )

        # MLP.
        new_state_dict[f"blocks.{block}.feed_forward.w1.weight"] = state_dict.pop(
            f"model.layers.{block}.mlp.gate_proj.weight"
        )
        new_state_dict[f"blocks.{block}.feed_forward.w2.weight"] = state_dict.pop(
            f"model.layers.{block}.mlp.down_proj.weight"
        )
        new_state_dict[f"blocks.{block}.feed_forward.w3.weight"] = state_dict.pop(
            f"model.layers.{block}.mlp.up_proj.weight"
        )

        # Layer norms.
        if "Llama" or "deepseek" in HF_MODEL:
            new_state_dict[f"blocks.{block}.feed_forward_norm.weight"] = state_dict.pop(
                f"model.layers.{block}.post_attention_layernorm.weight"
            )
            new_state_dict[f"blocks.{block}.attention_norm.weight"] = state_dict.pop(
                f"model.layers.{block}.input_layernorm.weight"
            )
        else:
            new_state_dict[f"blocks.{block}.attention_norm.weight"] = state_dict.pop(
                f"model.layers.{block}.post_attention_layernorm.weight"
            )
            new_state_dict[f"blocks.{block}.feed_forward_norm.weight"] = state_dict.pop(
                f"model.layers.{block}.post_feedforward_layernorm.weight"
            )
            new_state_dict[f"blocks.{block}.attention.q_norm.weight"] = state_dict.pop(
                f"model.layers.{block}.self_attn.q_norm.weight"
            )
            new_state_dict[f"blocks.{block}.attention.k_norm.weight"] = state_dict.pop(
                f"model.layers.{block}.self_attn.k_norm.weight"
            )

    assert len(state_dict) == 0

    log.info(f"Saving converted model checkpoint '{SAVE_PATH}'...")
    save_state_dict(os.path.join(SAVE_PATH, "model_and_optim"), {"model": new_state_dict})

    with open(os.path.join(SAVE_PATH, "config.json"), "w") as f:
        json.dump({"model": MODEL_CONFIG.as_dict()}, f)

    return hf_model


def validate_conversion(hf_model):
    device = get_default_device()

    B, T = 1, 120
    input_ids = torch.randint(0, TOKENIZER_CONFIG.vocab_size, (B, T)).to(device)

    hf_model = hf_model.to(device).eval()
    with torch.no_grad():
        hf_logits, *_ = hf_model(input_ids=input_ids, return_dict=False)

    del hf_model

    model = MODEL_CONFIG.build(device=device, max_seq_len=131072).eval()

    log.info("Loading converted checkpoint for validation...")
    load_model_and_optim_state(os.path.join(SAVE_PATH, "model_and_optim"), model)

    with torch.no_grad():
        logits = model(input_ids=input_ids)

    torch.testing.assert_close(hf_logits, logits)

    log.info("Conversion successful")


if __name__ == "__main__":
    prepare_cli_environment()

    config = MODEL_CONFIG.as_dict()
    hf_model = convert_checkpoint()
    validate_conversion(hf_model)
