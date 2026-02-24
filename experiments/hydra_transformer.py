"""
HydraTransformer: multi-head branched transformer.

Shares a common trunk (early layers) and branches into N independent heads
(late layers + lm_head). Each head produces its own logits.

This script loads weights and runs generation. The model implementation
lives in olmo_core.nn.transformer.hydra.
"""

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.transformer import HydraTransformer, HydraTransformerConfig


WEIGHTS_DIR = "/home/owain/olmo2-1b-instruct-weights"
VOCAB_SIZE = 100352  # this is LARGER than the tokeniser vocab size, needs padding later

config = HydraTransformerConfig.from_olmo2_1B(n_heads=5, split_layer=13)
model = config.build(init_device="meta")

hf_config = AutoConfig.from_pretrained(WEIGHTS_DIR)
hf_state: dict[str, torch.Tensor] = load_file(f"{WEIGHTS_DIR}/model.safetensors")

olmo_state = convert_state_from_hf(hf_config, hf_state)
del hf_state, hf_config  # or we get RAM issues

HydraTransformer.load_olmo_state(
    model, olmo_state, trunk_layers=config.trunk_layers, vocab_size=VOCAB_SIZE
)
del olmo_state  # free up ram

model.to(device="cuda", dtype=torch.bfloat16)
model.eval()

print(
    f"\nLoaded: trunk={config.trunk_layers} layers, heads={config.head_layers} layers x{config.n_heads}"
)


tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
prompt = "What is the capital of France?"
messages = [{"role": "user", "content": prompt}]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
MAX_NEW_TOKENS = 20

input_ids = torch.tensor([tokenizer.encode(chat_prompt)], device="cuda")
max_seq_len = input_ids.shape[1] + MAX_NEW_TOKENS  # pre-allocate KV cache dimensions

model.init_kv_cache(batch_size=1, max_seq_len=max_seq_len)

with torch.no_grad():
    # prefill
    all_logits = model(input_ids, return_logits=True)
    merged_logits = all_logits[:, 0, -1, :].mean(dim=0)  # average across heads
    next_token = merged_logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
    generated = [next_token.item()]

    # decode
    for _ in range(MAX_NEW_TOKENS - 1):
        all_logits = model(next_token, return_logits=True)
        merged_logits = all_logits[:, 0, -1, :].mean(dim=0)
        next_token = merged_logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
        generated.append(next_token.item())

full_ids = input_ids[0].tolist() + generated
print(f"\nPrompt: {prompt!r}")
print(f"Output: {tokenizer.decode(full_ids)}")
