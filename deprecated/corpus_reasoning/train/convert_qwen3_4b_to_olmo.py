"""Convert HF Qwen/Qwen3-4B-Base -> olmo-core checkpoint.

olmo-core's example CLI (convert_checkpoint_from_hf.py) only registers olmo2/olmo3/llama
in its arch dict, so we call its convert function directly with the qwen3_4B config.
"""

import sys

import torch
from huggingface_hub import snapshot_download
 # sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
from convert_checkpoint_from_hf import convert_checkpoint_from_hf  # noqa: E402

from olmo_core.data.tokenizer import TokenizerConfig  # noqa: E402
from olmo_core.nn.transformer.config import TransformerConfig  # noqa: E402

raise NotImplementedError(
    "convert_qwen3_4b_to_olmo.py is DEPRECATED — superseded by "
    "scripts/train/convert_hf_to_olmo.py (any Qwen3-dense size; derives eos/pad/bos from the "
    "real tokenizer instead of hardcoding eos=151645). The olmo-core dispatcher invokes it."
)

HF_ID = "Qwen/Qwen3-4B-Base"
OUT = "/scratch/users/prasann/olmo_ckpts/qwen3_4b_base_olmo"
VOCAB = 151936  # Qwen3-4B, already a multiple of 128

local = snapshot_download(HF_ID)
print(f"[convert] HF snapshot: {local}", flush=True)

model_cfg = TransformerConfig.qwen3_4B(vocab_size=VOCAB)
# Qwen3 special-token ids from the HF config (bos/pad = <|endoftext|> 151643, eos 151645).
tok_cfg = TokenizerConfig(
    vocab_size=VOCAB, eos_token_id=151645, pad_token_id=151643, bos_token_id=151643, identifier=HF_ID
)

print("[convert] starting conversion (validate=True on GPU)...", flush=True)
convert_checkpoint_from_hf(
    hf_checkpoint_path=local,
    output_path=OUT,
    transformer_config_dict=model_cfg.as_config_dict(),
    tokenizer_config_dict=tok_cfg.as_config_dict(),
    validate=True,
    device=torch.device("cuda"),
    validation_device=torch.device("cuda"),
)
print(f"[convert] DONE -> {OUT}", flush=True)
