"""
HydraTransformer inference demo.

Loads OLMo2 1B instruct weights into a HydraTransformer and runs
greedy generation with averaged head logits.

Usage:
    pixi run python experiments/hydra_demo.py
"""

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.transformer import HydraTransformer, HydraTransformerConfig

WEIGHTS_DIR = "/home/owain/olmo2-1b-instruct-weights"
VOCAB_SIZE = 100352
MAX_NEW_TOKENS = 20


def main():
    # Build model on meta device (zero memory until weights are loaded).
    config = HydraTransformerConfig.from_olmo2_1B(n_heads=5, split_layer=13)
    model = config.build(init_device="meta")

    # Load and convert HF weights to OLMo format.
    hf_config = AutoConfig.from_pretrained(WEIGHTS_DIR)
    hf_state = load_file(f"{WEIGHTS_DIR}/model.safetensors")
    olmo_state = convert_state_from_hf(hf_config, hf_state)
    del hf_state, hf_config

    # Distribute weights across trunk/heads/lm_head.
    HydraTransformer.load_olmo_state(
        model, olmo_state, trunk_layers=config.trunk_layers, vocab_size=VOCAB_SIZE
    )
    del olmo_state

    model.to(device="cuda", dtype=torch.bfloat16)
    model.eval()

    print(
        f"\nLoaded: trunk={config.trunk_layers} layers, "
        f"heads={config.head_layers} layers x{config.n_heads}, "
        f"{model.num_params:,} params"
    )

    # Tokenize with chat template.
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    prompt = "What is the capital of France?"
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    input_ids = torch.tensor([tokenizer.encode(chat_prompt)], device="cuda")
    max_seq_len = input_ids.shape[1] + MAX_NEW_TOKENS

    # Initialize KV caches.
    model.init_kv_cache(batch_size=1, max_seq_len=max_seq_len)

    # Generate.
    with torch.no_grad():
        # Prefill: process full prompt, populate KV cache.
        all_logits = model(input_ids, return_logits=True)
        merged_logits = all_logits[:, 0, -1, :].mean(dim=0)
        next_token = merged_logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
        generated = [next_token.item()]

        # Decode: one token at a time using cached KVs.
        for _ in range(MAX_NEW_TOKENS - 1):
            all_logits = model(next_token, return_logits=True)
            merged_logits = all_logits[:, 0, -1, :].mean(dim=0)
            next_token = merged_logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
            generated.append(next_token.item())

    full_ids = input_ids[0].tolist() + generated
    print(f"\nPrompt: {prompt!r}")
    print(f"Output: {tokenizer.decode(full_ids)}")


if __name__ == "__main__":
    main()
