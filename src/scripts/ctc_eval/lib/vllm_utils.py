"""Shared vLLM model loading and inference utilities.

All standard-attention evaluation scripts use vLLM for fast batched inference.
This module provides common argument parsing, model loading (with optional LoRA),
and a simple inference wrapper.

Note: Chunked attention evaluation uses HuggingFace Transformers directly
(see evaluate_chunked.py) because vLLM doesn't support custom 4D attention masks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

from ctc_eval.lib.io import ALPACA_TEMPLATE, format_alpaca_prompt  # noqa: F401 — re-exported
from ctc_eval.lib.adapter_save import prepare_adapter_for_backend


def add_vllm_args(parser: argparse.ArgumentParser) -> None:
    """Add standard vLLM eval arguments to a parser."""
    parser.add_argument("--base-model", type=str, default="NousResearch/Llama-3.2-1B")
    parser.add_argument("--lora-path", type=str, default="", help="Path to LoRA adapter (empty=base only)")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--output-file", type=str, default="outputs/eval_results.json")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable torch.compile and CUDA graphs for compatibility")
    parser.add_argument("--language-model-only", action="store_true",
                        help="Force text-only mode for multimodal models (e.g. Qwen3.5)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Override tokenizer (e.g. use HF model tokenizer for merged checkpoints)")


def load_model(args) -> tuple[LLM, LoRARequest | None]:
    """Load vLLM model and optional LoRA adapter from parsed args.

    Returns (llm, lora_request) where lora_request is None for base-only eval.
    """
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    enable_lora = bool(args.lora_path)

    # Qwen3.5 is a hybrid multimodal model that also needs eager mode to avoid
    # a LoRA packed-layer crash during CUDA graph profiling. Auto-enable both
    # workarounds whenever the base model is Qwen3.5.
    is_qwen35 = "Qwen3.5" in args.base_model or "qwen3_5" in args.base_model.lower()
    language_model_only = getattr(args, "language_model_only", False) or is_qwen35
    enforce_eager = getattr(args, "enforce_eager", False) or is_qwen35
    print(f"Loading model: {args.base_model} (enable_lora={enable_lora}, language_model_only={language_model_only}, enforce_eager={enforce_eager})")

    hf_overrides = {"architectures": ["Qwen3_5ForCausalLM"]} if language_model_only else None

    tokenizer = getattr(args, "tokenizer", None)
    # The chunked-vllm backend (in evaluate.py) wires the chunked mask through
    # vLLM's FlexAttention backend; force-select it via attention_config so
    # vLLM doesn't auto-pick FlashAttention.
    attention_config = None
    if getattr(args, "force_flex_attention", False):
        attention_config = {"backend": "FLEX_ATTENTION"}
    llm = LLM(
        model=args.base_model,
        tokenizer=tokenizer,
        enable_lora=enable_lora,
        max_lora_rank=64 if enable_lora else None,  # support up to r=64 LoRA
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.5,  # conservative — leaves room for LoRA overhead
        enforce_eager=enforce_eager,
        hf_overrides=hf_overrides,
        attention_config=attention_config,
    )

    # Create a LoRA request if a LoRA adapter path was provided.
    # vLLM loads LoRA weights on-the-fly per request, so the base model
    # is loaded once and the adapter is applied at inference time.
    lora_request = None
    if args.lora_path:
        normalized = prepare_adapter_for_backend(args.lora_path, args.base_model, backend="vllm")
        lora_request = LoRARequest("lora", 1, str(Path(normalized).resolve()))
    return llm, lora_request


def run_inference(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    lora_request: LoRARequest | None = None,
) -> list[str]:
    """Run vLLM inference and return response texts."""
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    return [o.outputs[0].text for o in outputs]
