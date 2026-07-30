"""Convert an HF Qwen3-dense base model -> olmo-core checkpoint.

olmo-core's example CLI (convert_checkpoint_from_hf.py) only registers olmo2/olmo3/llama
in its arch dict, so we call its convert function directly with the qwen3 config resolved
from --base-model via scripts.lib.olmo_models (same resolver the trainer/exporter use).

Default out dir: /scratch/users/prasann/olmo_ckpts/<builder>_olmo  (e.g. qwen3_0_6B_olmo).
"""

import argparse
import sys

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig
 # sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
from convert_checkpoint_from_hf import convert_checkpoint_from_hf  # noqa: E402

from olmo_core.data.tokenizer import TokenizerConfig  # noqa: E402

from corpus_reasoning.lib.olmo_models import build_transformer_config, resolve_olmo_model


def main():
    raise NotImplementedError(
        "convert_qwen3_to_olmo.py is DEPRECATED — superseded by "
        "scripts/train/convert_hf_to_olmo.py (the version the olmo-core dispatcher invokes)."
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--out", default=None, help="default /scratch/.../olmo_ckpts/<builder>_olmo")
    ap.add_argument("--cpu", action="store_true", help="convert/validate on CPU (small models)")
    args = ap.parse_args()

    spec = resolve_olmo_model(args.base_model)
    out = args.out or f"/scratch/users/prasann/olmo_ckpts/{spec.builder}_olmo"
    local = snapshot_download(args.base_model)
    print(f"[convert] {args.base_model} ({spec.builder}, vocab={spec.vocab_size}) snapshot: {local}\n[convert] out: {out}", flush=True)

    hf = AutoConfig.from_pretrained(args.base_model)
    eos = getattr(hf, "eos_token_id", None)
    bos = getattr(hf, "bos_token_id", None)
    pad = getattr(hf, "pad_token_id", None) or eos
    model_cfg = build_transformer_config(spec)
    tok_cfg = TokenizerConfig(
        vocab_size=spec.vocab_size, eos_token_id=eos, pad_token_id=pad,
        bos_token_id=bos, identifier=args.base_model,
    )

    dev = torch.device("cpu" if args.cpu else "cuda")
    print(f"[convert] starting conversion (validate=True on {dev})...", flush=True)
    convert_checkpoint_from_hf(
        hf_checkpoint_path=local,
        output_path=out,
        transformer_config_dict=model_cfg.as_config_dict(),
        tokenizer_config_dict=tok_cfg.as_config_dict(),
        validate=True,
        device=dev,
        validation_device=dev,
    )
    print(f"[convert] DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
