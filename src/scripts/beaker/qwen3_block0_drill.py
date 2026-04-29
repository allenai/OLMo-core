"""Block-0 submodule drill: HF Qwen3 vs OLMo-core single-doc parity.

Locates the first submodule inside layer/block 0 where OLMo-core diverges from
HuggingFace Transformers on Qwen3-4B-Base. Reports max|Δ|, mean|Δ|, and dtype
at every tap point so the regression can be narrowed to one operation
(QwenRMSNorm cast point, RoPE dtype, q/k_norm, SDPA, or MLP).

Run inside a single-GPU Beaker job (see qwen3_block0_drill_launch.py).
"""

import logging
import os
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
import transformers
from transformers.models.qwen3 import modeling_qwen3

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-4B-Base"
PROMPT = "The quick brown fox jumps over the lazy dog."


def make_hook(store: Dict[str, torch.Tensor], name: str) -> Callable:
    def hook(_module, _inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        if isinstance(t, torch.Tensor):
            store[name] = t.detach()

    return hook


def make_pre_hook(store: Dict[str, torch.Tensor], name: str) -> Callable:
    def hook(_module, inputs):
        t = inputs[0] if isinstance(inputs, tuple) else inputs
        if isinstance(t, torch.Tensor):
            store[name] = t.detach()

    return hook


def diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, str]:
    if a.shape != b.shape:
        return float("nan"), float("nan"), f"SHAPE {tuple(a.shape)} vs {tuple(b.shape)}"
    d = (a.float() - b.float()).abs()
    return d.max().item(), d.mean().item(), f"{a.dtype}/{b.dtype}"


def report(name: str, hf: Optional[torch.Tensor], oc: Optional[torch.Tensor]) -> None:
    if hf is None or oc is None:
        log.info("  %-30s  MISSING (hf=%s, olmo=%s)", name, hf is not None, oc is not None)
        return
    mx, mn, info = diff(hf, oc)
    log.info(
        "  %-30s  shape=%-22s  max=%.3e  mean=%.3e  dtype=%s",
        name,
        str(tuple(hf.shape)),
        mx,
        mn,
        info,
    )


def main() -> None:
    prepare_cli_environment()
    hf_token = os.environ.get("HF_TOKEN")
    device = torch.device("cuda")
    dtype = torch.bfloat16

    log.info("Loading HF %s (sdpa, %s)", MODEL_ID, dtype)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=hf_token, torch_dtype=dtype, attn_implementation="sdpa"
    )
    hf_model.to(device).eval()
    hf_config = hf_model.config

    log.info("Building OLMo qwen3_4B (TorchAttentionBackend / sdpa, vocab_size=%d)", hf_config.vocab_size)
    olmo_config = TransformerConfig.qwen3_4B(
        vocab_size=hf_config.vocab_size,
        attn_backend=AttentionBackendName.torch,
    )
    olmo_model = olmo_config.build(init_device="cpu")

    log.info("Converting and loading HF weights into OLMo-core model")
    converted_state = convert_state_from_hf(hf_config, hf_model.state_dict(), model_type="qwen3")
    olmo_model.load_state_dict(converted_state)
    olmo_model.to(device=device, dtype=dtype).eval()

    hf_acts: Dict[str, torch.Tensor] = {}
    oc_acts: Dict[str, torch.Tensor] = {}

    # Monkey-patch HF's apply_rotary_pos_emb to capture post-RoPE q,k for layer 0 only.
    _rope_seen = {"count": 0}
    _orig_apply_rope = modeling_qwen3.apply_rotary_pos_emb

    def _patched_apply_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        q_out, k_out = _orig_apply_rope(q, k, cos, sin, position_ids, unsqueeze_dim)
        if _rope_seen["count"] == 0:
            # HF shape: (B, n_heads, T, hd). Transpose to (B, T, n_heads, hd) to match OLMo.
            hf_acts["q_post_rope"] = q_out.detach().transpose(1, 2).contiguous()
            hf_acts["k_post_rope"] = k_out.detach().transpose(1, 2).contiguous()
            hf_acts["rope_cos"] = cos.detach()
            hf_acts["rope_sin"] = sin.detach()
        _rope_seen["count"] += 1
        return q_out, k_out

    modeling_qwen3.apply_rotary_pos_emb = _patched_apply_rope

    hf_layer = hf_model.model.layers[0]
    hf_attn = hf_layer.self_attn
    hf_mlp = hf_layer.mlp

    hf_model.model.embed_tokens.register_forward_hook(make_hook(hf_acts, "embed"))
    hf_layer.register_forward_pre_hook(make_pre_hook(hf_acts, "block_in"))
    hf_layer.input_layernorm.register_forward_hook(make_hook(hf_acts, "pre_attn_norm"))
    hf_attn.q_proj.register_forward_hook(make_hook(hf_acts, "q_proj"))
    hf_attn.k_proj.register_forward_hook(make_hook(hf_acts, "k_proj"))
    hf_attn.v_proj.register_forward_hook(make_hook(hf_acts, "v_proj"))
    hf_attn.q_norm.register_forward_hook(make_hook(hf_acts, "q_norm"))
    hf_attn.k_norm.register_forward_hook(make_hook(hf_acts, "k_norm"))
    hf_attn.o_proj.register_forward_pre_hook(make_pre_hook(hf_acts, "attn_out_pre_o"))
    hf_attn.o_proj.register_forward_hook(make_hook(hf_acts, "o_proj"))
    hf_attn.register_forward_hook(make_hook(hf_acts, "self_attn"))
    hf_layer.post_attention_layernorm.register_forward_hook(make_hook(hf_acts, "pre_mlp_norm"))
    hf_mlp.gate_proj.register_forward_hook(make_hook(hf_acts, "mlp_gate"))
    hf_mlp.up_proj.register_forward_hook(make_hook(hf_acts, "mlp_up"))
    hf_mlp.down_proj.register_forward_hook(make_hook(hf_acts, "mlp_down"))
    hf_layer.register_forward_hook(make_hook(hf_acts, "block_out"))

    oc_block = olmo_model.blocks["0"]
    oc_attn = oc_block.attention
    oc_ff = oc_block.feed_forward

    olmo_model.embeddings.register_forward_hook(make_hook(oc_acts, "embed"))
    oc_block.register_forward_pre_hook(make_pre_hook(oc_acts, "block_in"))
    oc_block.attention_norm.register_forward_hook(make_hook(oc_acts, "pre_attn_norm"))
    oc_attn.w_q.register_forward_hook(make_hook(oc_acts, "q_proj"))
    oc_attn.w_k.register_forward_hook(make_hook(oc_acts, "k_proj"))
    oc_attn.w_v.register_forward_hook(make_hook(oc_acts, "v_proj"))
    if oc_attn.q_norm is not None:
        oc_attn.q_norm.register_forward_hook(make_hook(oc_acts, "q_norm"))
    if oc_attn.k_norm is not None:
        oc_attn.k_norm.register_forward_hook(make_hook(oc_acts, "k_norm"))
    def oc_sdpa_pre_hook(_module, inputs):
        qkv = inputs[0]
        q, k, v = qkv[0], qkv[1], qkv[2]
        oc_acts["q_post_rope"] = q.detach()
        oc_acts["k_post_rope"] = k.detach()
        oc_acts["v_pre_sdpa"] = v.detach()

    oc_attn.backend.register_forward_pre_hook(oc_sdpa_pre_hook)
    oc_attn.backend.register_forward_hook(make_hook(oc_acts, "sdpa_out"))
    oc_attn.w_out.register_forward_pre_hook(make_pre_hook(oc_acts, "attn_out_pre_o"))
    oc_attn.w_out.register_forward_hook(make_hook(oc_acts, "o_proj"))
    oc_attn.register_forward_hook(make_hook(oc_acts, "self_attn"))
    oc_block.feed_forward_norm.register_forward_hook(make_hook(oc_acts, "pre_mlp_norm"))
    oc_ff.w1.register_forward_hook(make_hook(oc_acts, "mlp_gate"))
    oc_ff.w3.register_forward_hook(make_hook(oc_acts, "mlp_up"))
    oc_ff.w2.register_forward_hook(make_hook(oc_acts, "mlp_down"))
    oc_block.register_forward_hook(make_hook(oc_acts, "block_out"))

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

    log.info("Running forward passes (seq_len=%d, sdpa, bf16)", input_ids.shape[1])
    with torch.no_grad():
        hf_logits = hf_model(
            input_ids=input_ids, attention_mask=None, position_ids=position_ids
        ).logits
        olmo_logits = olmo_model(input_ids)

    log.info("=" * 88)
    log.info("Block-0 submodule drill (max/mean |hf - olmo|)")
    log.info("=" * 88)

    keys_in_order = [
        "embed",
        "block_in",
        "pre_attn_norm",
        "q_proj",
        "k_proj",
        "v_proj",
        "q_norm",
        "k_norm",
        "q_post_rope",
        "k_post_rope",
        "sdpa_out",
        "attn_out_pre_o",
        "o_proj",
        "self_attn",
        "pre_mlp_norm",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
        "block_out",
    ]
    for k in keys_in_order:
        report(k, hf_acts.get(k), oc_acts.get(k))

    log.info("=" * 88)
    olmo_logits_t = olmo_logits if isinstance(olmo_logits, torch.Tensor) else olmo_logits.logits
    hf_lp = F.log_softmax(hf_logits.float(), dim=-1)
    oc_lp = F.log_softmax(olmo_logits_t.float(), dim=-1)
    log.info(
        "logits  |Δ| max=%.3e  mean=%.3e",
        (hf_logits.float() - olmo_logits_t.float()).abs().max().item(),
        (hf_logits.float() - olmo_logits_t.float()).abs().mean().item(),
    )
    log.info(
        "logprob |Δ| max=%.3e  mean=%.3e",
        (hf_lp - oc_lp).abs().max().item(),
        (hf_lp - oc_lp).abs().mean().item(),
    )
    log.info(
        "argmax-agree=%.1f%%",
        (hf_logits.argmax(-1) == olmo_logits_t.argmax(-1)).float().mean().item() * 100,
    )


if __name__ == "__main__":
    main()
