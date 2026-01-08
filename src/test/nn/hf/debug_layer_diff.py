import os
from typing import Dict, List, Tuple

import torch
import transformers

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.convert import (
    HF_TO_OLMO_CORE_MODULE_MAPPINGS,
    MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_MODULE_MAPPINGS,
    convert_state_from_hf,
)
from olmo_core.nn.transformer import TransformerConfig

HF_TOKEN = os.environ.get("HF_TOKEN")


def _get_gemma3_config(hf_config) -> TransformerConfig:
    padded_vocab_size = (hf_config.vocab_size + 255) // 256 * 256
    return TransformerConfig.gemma3_like(
        d_model=hf_config.hidden_size,
        vocab_size=padded_vocab_size,
        n_layers=hf_config.num_hidden_layers,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=hf_config.num_key_value_heads,
        hidden_size=hf_config.intermediate_size,
        head_dim=hf_config.head_dim,
        local_window_size=hf_config.sliding_window or 1024,
        local_rope_theta=getattr(hf_config, "rope_local_base_freq", None) or 10_000,
        global_rope_theta=hf_config.rope_theta or 1_000_000,
        layer_norm_eps=hf_config.rms_norm_eps,
        attn_backend=AttentionBackendName.torch,
    )


def get_layer_mapping(model_type: str, n_layers: int) -> List[Tuple[str, str]]:
    base_mapping = dict(HF_TO_OLMO_CORE_MODULE_MAPPINGS)
    if model_type in MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_MODULE_MAPPINGS:
        base_mapping.update(MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_MODULE_MAPPINGS[model_type])

    mapping = []
    for hf_template, olmo_template in base_mapping.items():
        if "{layer}" in hf_template:
            for i in range(n_layers):
                hf_key = hf_template.replace("{layer}", str(i))
                olmo_key = olmo_template.replace("{layer}", str(i))
                mapping.append((hf_key, olmo_key))
        else:
            mapping.append((hf_template, olmo_template))

    return mapping


def capture_activations(
    model, layer_patterns: List[str]
) -> Tuple[Dict[str, torch.Tensor], List]:
    activations = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach().clone()

        return hook

    for name, module in model.named_modules():
        for pattern in layer_patterns:
            if pattern == name:
                hooks.append(module.register_forward_hook(make_hook(name)))
                break

    return activations, hooks


def remove_hooks(hooks: List):
    for h in hooks:
        h.remove()


def compare_activations(
    hf_acts: Dict[str, torch.Tensor],
    olmo_acts: Dict[str, torch.Tensor],
    mapping: List[Tuple[str, str]],
):
    print("\n" + "=" * 80)
    print("LAYER-BY-LAYER ACTIVATION COMPARISON")
    print("=" * 80)

    for hf_name, olmo_name in mapping:
        if hf_name not in hf_acts:
            continue
        if olmo_name not in olmo_acts:
            print(f"  [SKIP] OLMo layer '{olmo_name}' not found (HF: {hf_name})")
            continue

        hf_val = hf_acts[hf_name]
        olmo_val = olmo_acts[olmo_name]

        if hf_val.shape != olmo_val.shape:
            print(f"  [SHAPE] {hf_name} -> {olmo_name}")
            print(f"          HF: {hf_val.shape}, OLMo: {olmo_val.shape}")
            continue

        diff = (hf_val - olmo_val).abs()
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()

        status = "OK" if max_diff < 1e-5 else "DIFF"
        print(f"  [{status}] {hf_name}")
        print(f"       -> {olmo_name}")
        print(f"       mean={mean_diff:.2e}, max={max_diff:.2e}")


def debug_gemma3_layer_diff():
    if not HF_TOKEN:
        print("HF_TOKEN not set, skipping")
        return

    model_id = "google/gemma-3-270m"
    model_type = "gemma3_text"

    print(f"Loading HF model: {model_id}")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, token=HF_TOKEN, torch_dtype=torch.float32, attn_implementation="eager"
    )
    hf_model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

    print("Building OLMo model")
    olmo_config = _get_gemma3_config(hf_model.config)
    olmo_model = olmo_config.build(init_device="cpu")
    converted_state = convert_state_from_hf(
        hf_model.config, hf_model.state_dict(), model_type=model_type
    )
    olmo_model.load_state_dict(converted_state)
    olmo_model.eval()

    n_layers = hf_model.config.num_hidden_layers
    mapping = get_layer_mapping(model_type, n_layers)

    hf_layers = list(set(hf for hf, _ in mapping))
    olmo_layers = list(set(olmo for _, olmo in mapping))

    prompt = "Hello!"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Input: '{prompt}' -> {input_ids.shape}")

    hf_acts, hf_hooks = capture_activations(hf_model, hf_layers)
    olmo_acts, olmo_hooks = capture_activations(olmo_model, olmo_layers)

    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        olmo_logits = olmo_model(input_ids)

    remove_hooks(hf_hooks)
    remove_hooks(olmo_hooks)

    compare_activations(hf_acts, olmo_acts, mapping)

    print("\n" + "=" * 80)
    print("FINAL LOGITS")
    print("=" * 80)
    diff = (hf_logits - olmo_logits).abs()
    print(f"  mean={diff.mean().item():.2e}, max={diff.max().item():.2e}")


if __name__ == "__main__":
    debug_gemma3_layer_diff()
