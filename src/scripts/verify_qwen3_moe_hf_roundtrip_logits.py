"""Compare an original Qwen MoE HF checkpoint with a round-trip HF export."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from olmo_core.config import DType
from olmo_core.utils import prepare_cli_environment


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def _get_logits(
    model_name_or_path: str | Path,
    input_ids: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=False,
    ).to(device)
    model.eval()
    logits = model(input_ids.to(device), use_cache=False).logits.float().cpu()
    del model
    _release_cuda_memory()
    return logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-model", default="Qwen/Qwen3.5-35B-A3B-Base")
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument(
        "--prompt",
        default="Explain why careful checkpoint conversion matters in one sentence.",
    )
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument("--dtype", type=DType, default=DType.bfloat16)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    prepare_cli_environment()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=False)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    reference = _get_logits(
        args.reference_model,
        input_ids,
        device=args.device,
        dtype=args.dtype.as_pt(),
    )
    candidate = _get_logits(
        args.candidate_model,
        input_ids,
        device=args.device,
        dtype=args.dtype.as_pt(),
    )

    diff = (reference - candidate).abs()
    metrics = {
        "reference_model": args.reference_model,
        "candidate_model": str(args.candidate_model),
        "tokenizer": args.tokenizer_name,
        "num_input_tokens": input_ids.numel(),
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            reference.reshape(1, -1), candidate.reshape(1, -1)
        ).item(),
        "top1_agreement": (reference.argmax(dim=-1) == candidate.argmax(dim=-1))
        .float()
        .mean()
        .item(),
        "rtol": args.rtol,
        "atol": args.atol,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    torch.testing.assert_close(candidate, reference, rtol=args.rtol, atol=args.atol)


if __name__ == "__main__":
    main()
