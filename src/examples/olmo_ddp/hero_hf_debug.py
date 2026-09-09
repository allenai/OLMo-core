"""Read-only layerwise diagnostic for an unpublished hero conversion."""

import argparse
import gc
import logging
import os

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

from hero_hf_convert import reference_config
from hero_hf_download import SCRATCH, TARGETS
from hero_hf_reference_ops import install, self_test
from olmo_core.nn.hf.config import _register_olmo3moe_auto_classes
from olmo_core.nn.hf.convert_checkpoint import (
    _load_ddp_optimizer_model_state,
    load_config,
)
from olmo_core.nn.transformer.config import TransformerConfig


def capture(model, modules):
    results = {}

    def hook(key, module, args, output):
        del module
        for suffix, value in (("input", args), ("output", output)):
            if not isinstance(value, tuple):
                value = (value,)
            for index, tensor in enumerate(value):
                if isinstance(tensor, torch.Tensor):
                    results[f"{key}/{suffix}{index}"] = tensor.detach().cpu().clone()

    from functools import partial

    for key, path in modules.items():
        model.get_submodule(path).register_forward_hook(partial(hook, key))
    return results


def mappings():
    core, hf = {}, {}
    aliases = {
        "attention_input_norm": "pre_attention_layernorm",
        "attention.w_q": "self_attn.q_proj",
        "attention.w_k": "self_attn.k_proj",
        "attention.w_v": "self_attn.v_proj",
        "attention.q_conv1d": "self_attn.q_conv1d",
        "attention.k_conv1d": "self_attn.k_conv1d",
        "attention.v_conv1d": "self_attn.v_conv1d",
        "attention.f_proj_1": "self_attn.f_proj_1",
        "attention.f_proj_2": "self_attn.f_proj_2",
        "attention.w_b": "self_attn.beta_proj",
        "attention.g_proj_1": "self_attn.g_proj_1",
        "attention.g_proj_2": "self_attn.g_proj_2",
        "attention.o_norm": "self_attn.o_norm",
        "attention.w_out": "self_attn.o_proj",
        "attention_norm": "post_attention_layernorm",
        "feed_forward_input_norm": "pre_feedforward_layernorm",
        "latent_down_proj": "mlp.latent_down_proj",
        "routed_experts_router": "mlp.router",
        "shared_experts": "mlp.shared_expert",
        "latent_up_proj": "mlp.latent_up_proj",
        "feed_forward_norm": "post_feedforward_layernorm",
    }
    for i in range(3):
        for left, right in aliases.items():
            key = f"{i}.{left}"
            core[key] = f"blocks.{i}.{left}"
            hf[key] = f"model.layers.{i}.{right}"
    return core, hf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("emo", "non-emo"), required=True)
    parser.add_argument("--step", type=int, choices=tuple(TARGETS.values()), required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    root = SCRATCH / args.arm / f"step{args.step}"
    if (root / "conversion-success.json").exists():
        raise RuntimeError("Diagnostic is restricted to unpublished conversions")
    _register_olmo3moe_auto_classes()
    self_test()
    install()
    os.environ["OLMO_HF_MOE_CORE_REFERENCE"] = "1"
    config = load_config(root / "olmo-core")
    vocab = config["dataset"]["tokenizer"]["vocab_size"]
    ids = torch.randint(vocab, (1, 63), generator=torch.Generator().manual_seed(20260909)).cuda()
    core_map, hf_map = mappings()
    model = TransformerConfig.from_dict(reference_config(config["model"])).build(init_device="meta")
    model.to_empty(device="cpu")
    _load_ddp_optimizer_model_state(
        root / "olmo-core/model_and_optim",
        model,
        work_dir=str(root / "load-work"),
        return_state_dict=False,
    )
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    core = capture(model, core_map)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        core_logits = model(input_ids=ids)[..., :vocab].cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = (
        AutoModelForCausalLM.from_pretrained(
            root / "hf.partial", dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    hf = capture(model, hf_map)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        hf_logits = model(ids, use_cache=False).logits.cpu()
    for key, a in core.items():
        if key not in hf:
            continue
        b = hf[key]
        if a.numel() != b.numel():
            print("LAYER_DEBUG_SHAPE", key, tuple(a.shape), tuple(b.shape), flush=True)
            continue
        a, b = a.flatten(), b.flatten()
        difference = a.float() - b.float()
        print(
            "LAYER_DEBUG",
            key,
            "mean",
            difference.abs().mean().item(),
            "max",
            difference.abs().max().item(),
            "mismatches",
            (a != b).sum().item(),
            "total",
            a.numel(),
            "dtype",
            a.dtype,
            b.dtype,
            flush=True,
        )
    print("LOGITS_MAX", (core_logits.float() - hf_logits.float()).abs().max().item(), flush=True)


if __name__ == "__main__":
    main()
