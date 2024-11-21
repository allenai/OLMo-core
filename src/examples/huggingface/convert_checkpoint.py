"""
Example script showing how you could convert model weights on HuggingFace for a Llama model
into a format that can be loaded by OLMo-core for fine-tuning.

Note that this script is architecture-dependent, meaning it will only work for Llama models on
HuggingFace.
"""

import logging

import torch
from transformers import AutoModelForCausalLM

from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_state_dict
from olmo_core.nn.rope import RoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import get_default_device, prepare_cli_environment

log = logging.getLogger(__name__)

HF_MODEL = "meta-llama/Llama-3.2-1B"
SAVE_PATH = f"/tmp/checkpoints/{HF_MODEL}"

TOKENIZER_CONFIG = TokenizerConfig.from_hf(HF_MODEL)
MODEL_CONFIG = TransformerConfig.llama3_1B(
    TOKENIZER_CONFIG.vocab_size, fused_ops=False, use_flash=False, rope_scaling=RoPEScalingConfig()
)


def convert_checkpoint() -> AutoModelForCausalLM:
    log.info(f"Loading HF checkpoint '{HF_MODEL}'")
    hf_model = AutoModelForCausalLM.from_pretrained(HF_MODEL)
    print(hf_model)

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

        # Attention layer norm.
        new_state_dict[f"blocks.{block}.attention_norm.weight"] = state_dict.pop(
            f"model.layers.{block}.input_layernorm.weight"
        )

        # MLP layer norm.
        new_state_dict[f"blocks.{block}.feed_forward_norm.weight"] = state_dict.pop(
            f"model.layers.{block}.post_attention_layernorm.weight"
        )

    assert len(state_dict) == 0

    log.info(f"Saving converted model checkpoint '{SAVE_PATH}'...")
    save_state_dict(SAVE_PATH, {"model": new_state_dict})

    return hf_model


def validate_conversion(hf_model):
    log.info("Loading converted checkpoint for validation...")

    device = get_default_device()

    model = MODEL_CONFIG.build(device=device, max_seq_len=131072).eval()
    load_model_and_optim_state(SAVE_PATH, model)

    hf_model = hf_model.to(device).eval()

    input_ids = torch.randint(0, TOKENIZER_CONFIG.vocab_size, (1, 64)).to(device)
    cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
    position_ids = cache_position.unsqueeze(0)

    # Check models layer-by-layer.
    with torch.no_grad():
        # Token embeddings.
        log.info("Checking token embeddings...")
        h = model.embeddings(input_ids)
        hf_h = hf_model.model.embed_tokens(input_ids)
        torch.testing.assert_close(h, hf_h)

        position_embeddings = hf_model.model.rotary_emb(h, position_ids)

        for idx, (block, hf_block) in enumerate(zip(model.blocks, hf_model.model.layers)):
            log.info(f"Checking block {idx}...")
            h = block(h)

            r = h
            hf_r = hf_h

            log.info(f"Checking block {idx} input/attention norm...")
            h = block.attention_norm(h)
            hf_h = hf_block.input_layernorm(hf_h)
            torch.testing.assert_close(h, hf_h)

            log.info(f"Checking block {idx} attention...")
            h = block.attention(h)
            hf_h = hf_block.self_attn(
                hf_h, position_ids=position_ids, position_embeddings=position_embeddings
            )
            torch.testing.assert_close(h, hf_h)

            h = r + h
            hf_h = hf_r + hf_h

            r = h
            hf_r = hf_h

            log.info(f"Checking block {idx} MLP norm...")
            h = block.feed_forward_norm(h)
            hf_h = hf_block.post_attention_layernorm(hf_h)
            torch.testing.assert_close(h, hf_h)

            log.info(f"Checking block {idx} MLP...")
            h = block.feed_forward(h)
            hf_h = hf_block.mlp(hf_h)
            torch.testing.assert_close(h, hf_h)

            log.info(f"Checking block {idx} output...")
            h = r + h
            hf_h = hf_r + hf_h
            torch.testing.assert_close(h, hf_h)

        logits = model.lm_head(h)
        hf_logits = hf_model.lm_head(hf_model.model.norm(hf_h))
        torch.testing.assert_close(hf_logits, logits)

        #  logits = model(input_ids=input_ids)
        #  hf_logits, *_ = hf_model(input_ids=input_ids, return_dict=False)
        #  torch.testing.assert_close(hf_logits, logits)

    log.info("Conversion successful")


if __name__ == "__main__":
    prepare_cli_environment()
    hf_model = convert_checkpoint()
    validate_conversion(hf_model)
