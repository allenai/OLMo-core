"""Verify logits from a converted Qwen3 MoE checkpoint against Hugging Face."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from olmo_core.config import DType
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.moe.v2.qwen import build_qwen3_moe_config_from_hf_config
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def verify_logits(
    *,
    hf_model: str,
    checkpoint_path: Path,
    revision: str,
    prompt: str,
    device: torch.device,
    dtype: DType,
    rtol: float,
    atol: float,
) -> dict[str, float | int | str]:
    tokenizer = AutoTokenizer.from_pretrained(hf_model, revision=revision, trust_remote_code=False)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    log.info("Loading Hugging Face reference model %s", hf_model)
    hf = AutoModelForCausalLM.from_pretrained(
        hf_model,
        revision=revision,
        dtype=dtype.as_pt(),
        attn_implementation="eager",
        trust_remote_code=False,
    ).to(device)
    hf.eval()
    hf_logits = hf(input_ids, use_cache=False).logits.float().cpu()
    del hf
    _release_cuda_memory()

    log.info("Loading converted OLMo-core checkpoint %s", checkpoint_path)
    hf_config = AutoConfig.from_pretrained(
        hf_model,
        revision=revision,
        trust_remote_code=False,
    ).to_dict()
    model_config = build_qwen3_moe_config_from_hf_config(
        hf_config,
        dtype=dtype,
        attention_backend=AttentionBackendName.torch,
        compile_friendly_recompute=False,
    )
    olmo = model_config.build(init_device="meta")
    olmo.to_empty(device=device)
    load_model_and_optim_state(
        checkpoint_path / "model_and_optim",
        olmo,
        thread_count=32,
    )
    olmo.eval()
    olmo_logits = olmo(input_ids).float().cpu()

    if hf_logits.shape != olmo_logits.shape:
        raise RuntimeError(
            f"Logit shape mismatch: HF {tuple(hf_logits.shape)} != OLMo {tuple(olmo_logits.shape)}"
        )

    diff = (hf_logits - olmo_logits).abs()
    top1_agreement = (hf_logits.argmax(-1) == olmo_logits.argmax(-1)).float().mean().item()
    cosine = torch.nn.functional.cosine_similarity(
        hf_logits.reshape(1, -1),
        olmo_logits.reshape(1, -1),
    ).item()
    metrics: dict[str, float | int | str] = {
        "hf_model": hf_model,
        "revision": revision,
        "num_input_tokens": input_ids.numel(),
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "cosine_similarity": cosine,
        "top1_agreement": top1_agreement,
        "rtol": rtol,
        "atol": atol,
    }
    log.info("Logit comparison:\n%s", json.dumps(metrics, indent=2))
    torch.testing.assert_close(olmo_logits, hf_logits, rtol=rtol, atol=atol)
    del olmo
    _release_cuda_memory()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", default="Qwen/Qwen3-30B-A3B-Base")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="Explain why careful checkpoint conversion matters in one sentence.",
    )
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument("--dtype", type=DType, default=DType.bfloat16)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    prepare_cli_environment()
    metrics = verify_logits(
        hf_model=args.hf_model,
        checkpoint_path=args.checkpoint_path,
        revision=args.revision,
        prompt=args.prompt,
        device=args.device,
        dtype=args.dtype,
        rtol=args.rtol,
        atol=args.atol,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
