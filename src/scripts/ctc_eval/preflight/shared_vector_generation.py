"""Load a real shared-vector checkpoint and compare cached one-shot/chunked generation."""

import argparse
import torch
from transformers import AutoTokenizer
from olmo_core.config import DType
from olmo_core.generate.generation_module.config import GenerationConfig
from olmo_core.generate.generation_module.transformer import TransformerGenerationModuleConfig
from olmo_core.nn.attention.landmark_shared_vector import SharedVectorLandmarkAttention

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--tokenizer", required=True)
a = p.parse_args()
tok = AutoTokenizer.from_pretrained(a.tokenizer)
gc = GenerationConfig(
    eos_token_id=tok.eos_token_id,
    pad_token_id=tok.pad_token_id,
    max_length=4096,
    use_cache=True,
    landmark_top_k_fraction=0.1,
    landmark_decode_mode="extend_last_block",
)
gm = TransformerGenerationModuleConfig(
    gc, float8_config=None, dtype=DType.bfloat16, compile_model=False
).build(checkpoint_dir=a.checkpoint, device=torch.device("cuda"))
layers = gm._landmark_attention_layers()
assert layers and all(isinstance(layer, SharedVectorLandmarkAttention) for layer in layers)
assert not gm.supports_landmark_ragged_batch()
text = tok.apply_chat_template(
    [
        {
            "role": "user",
            "content": "Remember this fact: the secret word is violet. " * 40
            + "What is the secret word? Answer with one word.",
        }
    ],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
x = tok(text, return_tensors="pt")["input_ids"].cuda()
outputs = []
for chunk in (None, 128):
    gc.prefill_chunk_size = chunk
    output, _, _ = gm.generate_batch(
        input_ids=x, max_new_tokens=12, completions_only=True, log_timing=True
    )
    assert torch.isfinite(output).all()
    outputs.append(output)
    print("SHAREDVEC_GENERATION", chunk, output.tolist(), tok.batch_decode(output), flush=True)
assert torch.equal(outputs[0], outputs[1]), "Chunked and single-shot greedy generations differ"
print("SHAREDVEC_CHECKPOINT_PREFLIGHT_PASSED", flush=True)
