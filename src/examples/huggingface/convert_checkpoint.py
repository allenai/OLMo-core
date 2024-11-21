"""
Example script showing how you could convert model weights on HuggingFace for a Llama model
into a format that can be loaded by OLMo-core for fine-tuning.

Note that this script is architecture-dependent, meaning it will only work for Llama models on
HuggingFace.
"""

import logging

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_state_dict
from olmo_core.io import clear_directory, dir_is_empty
from olmo_core.nn.rope import RoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import get_default_device, prepare_cli_environment

log = logging.getLogger(__name__)

HF_MODEL = "meta-llama/Llama-3.2-1B"
SAVE_PATH = f"/tmp/checkpoints/{HF_MODEL}"
SAVE_OVERWRITE = False

TOKENIZER_CONFIG = TokenizerConfig.from_hf(HF_MODEL)
MODEL_CONFIG = TransformerConfig.llama3_1B(
    TOKENIZER_CONFIG.vocab_size, fused_ops=False, use_flash=False, rope_scaling=RoPEScalingConfig()
)


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
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    log.info("Loading converted checkpoint for validation...")

    device = get_default_device()

    model = MODEL_CONFIG.build(device=device, max_seq_len=131072).eval()
    load_model_and_optim_state(SAVE_PATH, model)

    hf_model = hf_model.to(device).eval()

    B, T, D = 1, 64, 2048
    n_heads, n_kv_heads = 32, 8
    head_dim = D // n_heads

    input_ids = torch.randint(0, TOKENIZER_CONFIG.vocab_size, (B, T)).to(device)
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

            r = h

            hf_r = hf_h

            log.info(f"Checking block {idx} input/attention norm...")

            # OLMo-core
            h = block.attention_norm(h)

            # HuggingFace
            hf_h = hf_block.input_layernorm(hf_h)

            torch.testing.assert_close(h, hf_h)

            log.info(f"Checking block {idx} attention projections...")

            # OLMo-core
            q, k, v = block.attention.w_q(h), block.attention.w_k(h), block.attention.w_v(h)

            # HuggingFace
            hf_q, hf_k, hf_v = (
                hf_block.self_attn.q_proj(hf_h),
                hf_block.self_attn.k_proj(hf_h),
                hf_block.self_attn.v_proj(hf_h),
            )

            torch.testing.assert_close(q, hf_q)
            torch.testing.assert_close(k, hf_k)
            torch.testing.assert_close(v, hf_v)

            log.info(f"Checking block {idx} rotary embeddings...")

            # OLMo-core
            q, k, v = (
                q.view(B, T, n_heads, head_dim),
                k.view(B, T, n_kv_heads, head_dim),
                v.view(B, T, n_kv_heads, head_dim),
            )
            q, k = block.attention.rope(q, k, head_first=False)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

            # HuggingFace
            hf_q, hf_k, hf_v = (
                hf_q.view(B, T, n_heads, head_dim).transpose(1, 2),
                hf_k.view(B, T, n_kv_heads, head_dim).transpose(1, 2),
                hf_v.view(B, T, n_kv_heads, head_dim).transpose(1, 2),
            )
            cos, sin = position_embeddings
            hf_q, hf_k = apply_rotary_pos_emb(hf_q, hf_k, cos, sin)

            torch.testing.assert_close(q, hf_q)
            torch.testing.assert_close(k, hf_k)

            log.info(f"Checking block {idx} SDPA...")

            # OLMo-core
            k = k.repeat_interleave(n_heads // n_kv_heads, dim=1, output_size=n_heads)
            v = v.repeat_interleave(n_heads // n_kv_heads, dim=1, output_size=n_heads)

            # HuggingFace
            hf_k = repeat_kv(hf_k, n_heads // n_kv_heads)
            hf_v = repeat_kv(hf_v, n_heads // n_kv_heads)

            torch.testing.assert_close(k, hf_k)
            torch.testing.assert_close(v, hf_v)

            # OLMo-core
            attn = (
                F.scaled_dot_product_attention(q, k, v, is_causal=True)
                .transpose(1, 2)
                .contiguous()
                .view(B, T, -1)
            )

            # HuggingFace
            hf_attn = (
                F.scaled_dot_product_attention(hf_q, hf_k, hf_v, is_causal=True)
                .transpose(1, 2)
                .contiguous()
                .view(B, T, -1)
            )

            torch.testing.assert_close(attn, hf_attn)

            log.info(f"Checking block {idx} attention projection...")

            # OLMo-core
            h = block.attention.w_out(attn)

            # HuggingFace
            hf_h = hf_block.attention.o_proj(attn)

            torch.testing.assert_close(h, hf_h)

            #  h = block.attention(h)
            #  hf_h, *_ = hf_block.self_attn(
            #      hf_h, position_ids=position_ids, position_embeddings=position_embeddings
            #  )
            #  torch.testing.assert_close(h, hf_h)

            # OLMo-core
            h = r + h

            # HuggingFace
            hf_h = hf_r + hf_h

            # OLMo-core
            r = h

            # HuggingFace
            hf_r = hf_h

            log.info(f"Checking block {idx} MLP norm...")

            # OLMo-core
            h = block.feed_forward_norm(h)

            # HuggingFace
            hf_h = hf_block.post_attention_layernorm(hf_h)

            torch.testing.assert_close(h, hf_h)

            log.info(f"Checking block {idx} MLP...")

            # OLMo-core
            h = block.feed_forward(h)

            # HuggingFace
            hf_h = hf_block.mlp(hf_h)

            torch.testing.assert_close(h, hf_h)

            log.info(f"Checking block {idx} output...")

            # OLMo-core
            h = r + h

            # HuggingFace
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
