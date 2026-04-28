"""
Reproduce the reported forward-pass parity divergence between OLMo-core's Qwen3
implementation and HuggingFace's Qwen3ForCausalLM on Qwen/Qwen3-4B-Base.

Reports per-block max/mean |hf - olmo| of hidden states, plus final-logits
argmax agreement and mean |Δ logprob|.

Run inside a single-GPU Beaker job (see qwen3_parity_launch.py).
"""

import logging
import os

import torch
import torch.nn.functional as F
import transformers

from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-4B-Base"
SEQ_LEN = 10
SEED = 0


def main() -> None:
    prepare_cli_environment()
    hf_token = os.environ.get("HF_TOKEN")
    device = torch.device("cuda")
    dtype = torch.bfloat16

    log.info("Loading HF model %s", MODEL_ID)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=hf_token, torch_dtype=dtype, attn_implementation="eager"
    )
    hf_model.to(device).eval()
    hf_config = hf_model.config

    log.info("Building OLMo-core qwen3_4B (vocab_size=%d)", hf_config.vocab_size)
    olmo_config = TransformerConfig.qwen3_4B(vocab_size=hf_config.vocab_size)
    olmo_model = olmo_config.build(init_device="cpu")

    log.info("Converting and loading HF weights into OLMo-core model")
    converted_state = convert_state_from_hf(hf_config, hf_model.state_dict(), model_type="qwen3")
    olmo_model.load_state_dict(converted_state)
    olmo_model.to(device=device, dtype=dtype).eval()

    hf_outs: dict[str, torch.Tensor] = {}
    olmo_outs: dict[str, torch.Tensor] = {}

    def hf_block_hook(idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hf_outs[f"block_{idx:02d}"] = hidden.detach()

        return hook

    def hf_norm_hook(_module, _inputs, output):
        hf_outs["final_norm"] = output.detach()

    def olmo_block_hook(idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            olmo_outs[f"block_{idx:02d}"] = hidden.detach()

        return hook

    def olmo_norm_hook(_module, _inputs, output):
        olmo_outs["final_norm"] = output.detach()

    for i, layer in enumerate(hf_model.model.layers):
        layer.register_forward_hook(hf_block_hook(i))
    hf_model.model.norm.register_forward_hook(hf_norm_hook)

    olmo_block_keys = sorted(olmo_model.blocks.keys(), key=int)
    for i, key in enumerate(olmo_block_keys):
        olmo_model.blocks[key].register_forward_hook(olmo_block_hook(i))
    olmo_model.lm_head.norm.register_forward_hook(olmo_norm_hook)

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, hf_config.vocab_size, (1, SEQ_LEN), device=device)
    log.info("Input token ids: %s", input_ids.tolist())

    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids).logits
        olmo_logits = olmo_model(input_ids)

    log.info("=" * 60)
    log.info("Per-layer hidden-state divergence (|hf - olmo|)")
    log.info("%-14s %12s %12s", "name", "max", "mean")
    log.info("-" * 60)

    n_layers = len(hf_model.model.layers)
    for i in range(n_layers):
        name = f"block_{i:02d}"
        diff = (hf_outs[name].float() - olmo_outs[name].float()).abs()
        log.info("%-14s %12.3e %12.3e", name, diff.max().item(), diff.mean().item())

    diff = (hf_outs["final_norm"].float() - olmo_outs["final_norm"].float()).abs()
    log.info("%-14s %12.3e %12.3e", "final_norm", diff.max().item(), diff.mean().item())

    log.info("=" * 60)
    hf_argmax = hf_logits.argmax(-1)
    olmo_argmax = olmo_logits.argmax(-1)
    agree = (hf_argmax == olmo_argmax).float().mean().item()
    hf_lp = F.log_softmax(hf_logits.float(), dim=-1)
    olmo_lp = F.log_softmax(olmo_logits.float(), dim=-1)
    lp_diff = (hf_lp - olmo_lp).abs()
    log.info("argmax agreement: %.4f", agree)
    log.info("logprob mean |Δ|: %.3e   max |Δ|: %.3e", lp_diff.mean().item(), lp_diff.max().item())


if __name__ == "__main__":
    main()
