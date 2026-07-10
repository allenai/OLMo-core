"""Verify logits from a converted Qwen3 MoE checkpoint against Hugging Face."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint import FileSystemReader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from olmo_core.config import DType
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.moe.v2.hf.convert_checkpoint import load_state_dict_direct
from olmo_core.nn.moe.v2.qwen import build_qwen3_moe_config_from_hf_config
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                pass
    raise TypeError(f"Expected a tensor output, got {type(value).__name__}")


def _tensor_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    if reference.shape != actual.shape:
        raise RuntimeError(
            f"Tensor shape mismatch: reference {tuple(reference.shape)} != actual {tuple(actual.shape)}"
        )
    reference = reference.float()
    actual = actual.float()
    diff = (reference - actual).abs()
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            reference.reshape(1, -1),
            actual.reshape(1, -1),
        ).item(),
    }


def _write_metrics(path: Path | None, metrics: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n")


def _capture_router_output(
    weights: list[torch.Tensor],
    indices: list[torch.Tensor],
    output: tuple[Any, ...],
) -> None:
    weights.append(output[0 if len(output) == 4 else 1].detach().float().cpu())
    indices.append(output[1 if len(output) == 4 else 2].detach().cpu())


@torch.no_grad()
def _load_olmo_checkpoint(checkpoint_dir: Path, model: torch.nn.Module) -> None:
    metadata = FileSystemReader(checkpoint_dir).read_metadata().state_dict_metadata
    if "module.embeddings.weight.main" not in metadata:
        load_model_and_optim_state(checkpoint_dir, model, thread_count=32)
        return

    log.info("Detected OLMo DDP checkpoint layout")
    checkpoint_state = load_state_dict_direct(
        checkpoint_dir,
        process_group=None,
        pre_download=False,
        thread_count=32,
    )
    remaining = set(checkpoint_state)
    for model_name, target in model.state_dict().items():
        checkpoint_name = f"module.{model_name}.main"
        try:
            source = checkpoint_state[checkpoint_name]
        except KeyError as exc:
            raise KeyError(f"Checkpoint is missing expected tensor {checkpoint_name!r}") from exc
        if target.numel() != source.numel():
            raise ValueError(
                f"{checkpoint_name}: model shape {tuple(target.shape)} does not match "
                f"checkpoint shape {tuple(source.shape)}"
            )
        target.copy_(source.reshape(target.shape).to(device=target.device, dtype=target.dtype))
        remaining.remove(checkpoint_name)

    if remaining:
        raise RuntimeError(f"Unconsumed OLMo DDP checkpoint tensors: {sorted(remaining)[:20]}")
    log.info("Loaded all %d OLMo DDP model tensors", len(checkpoint_state))


@torch.no_grad()
def verify_logits(
    *,
    hf_model: str,
    checkpoint_path: Path,
    revision: str,
    prompt: str,
    hf_device: torch.device,
    device: torch.device,
    dtype: DType,
    rtol: float,
    atol: float,
    skip_assert_close: bool = False,
    max_mean_abs_diff: float | None = None,
    min_cosine_similarity: float | None = None,
    min_top1_agreement: float | None = None,
    layerwise: bool = False,
    output_json: Path | None = None,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(hf_model, revision=revision, trust_remote_code=False)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    log.info("Loading Hugging Face reference model %s", hf_model)
    hf = AutoModelForCausalLM.from_pretrained(
        hf_model,
        revision=revision,
        dtype=dtype.as_pt(),
        attn_implementation="eager",
        trust_remote_code=False,
    ).to(hf_device)
    hf.eval()

    hf_hidden_states: list[torch.Tensor] = []
    hf_attention_outputs: list[torch.Tensor] = []
    hf_router_weights: list[torch.Tensor] = []
    hf_router_indices: list[torch.Tensor] = []
    handles: list[Any] = []
    if layerwise:
        for layer in hf.model.layers:
            handles.append(
                layer.register_forward_hook(
                    lambda _module, _args, output: hf_hidden_states.append(
                        _first_tensor(output).detach().float().cpu()
                    )
                )
            )
            handles.append(
                layer.self_attn.register_forward_hook(
                    lambda _module, _args, output: hf_attention_outputs.append(
                        _first_tensor(output).detach().float().cpu()
                    )
                )
            )
            handles.append(
                layer.mlp.gate.register_forward_hook(
                    lambda _module, _args, output: _capture_router_output(
                        hf_router_weights,
                        hf_router_indices,
                        output,
                    )
                )
            )
    try:
        hf_logits = hf(input_ids.to(hf_device), use_cache=False).logits.float().cpu()
    finally:
        for handle in handles:
            handle.remove()
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
    _load_olmo_checkpoint(checkpoint_path / "model_and_optim", olmo)
    olmo.eval()

    olmo_hidden_states: list[torch.Tensor] = []
    olmo_attention_outputs: list[torch.Tensor] = []
    olmo_router_weights: list[torch.Tensor] = []
    olmo_router_indices: list[torch.Tensor] = []
    handles = []
    if layerwise:
        for block in olmo.blocks.values():
            handles.append(
                block.register_forward_hook(
                    lambda _module, _args, output: olmo_hidden_states.append(
                        _first_tensor(output).detach().float().cpu()
                    )
                )
            )
            handles.append(
                block.attention.register_forward_hook(
                    lambda _module, _args, output: olmo_attention_outputs.append(
                        _first_tensor(output).detach().float().cpu()
                    )
                )
            )
            assert block.routed_experts_router is not None
            handles.append(
                block.routed_experts_router.register_forward_hook(
                    lambda _module, _args, output: _capture_router_output(
                        olmo_router_weights,
                        olmo_router_indices,
                        output,
                    )
                )
            )
    try:
        olmo_logits = olmo(input_ids.to(device)).float().cpu()
    finally:
        for handle in handles:
            handle.remove()

    if hf_logits.shape != olmo_logits.shape:
        raise RuntimeError(
            f"Logit shape mismatch: HF {tuple(hf_logits.shape)} != OLMo {tuple(olmo_logits.shape)}"
        )

    logit_metrics = _tensor_metrics(hf_logits, olmo_logits)
    top1_agreement = (hf_logits.argmax(-1) == olmo_logits.argmax(-1)).float().mean().item()
    metrics: dict[str, Any] = {
        "hf_model": hf_model,
        "revision": revision,
        "hf_device": str(hf_device),
        "olmo_device": str(device),
        "num_input_tokens": input_ids.numel(),
        **logit_metrics,
        "top1_agreement": top1_agreement,
        "rtol": rtol,
        "atol": atol,
        "acceptance": {
            "skip_assert_close": skip_assert_close,
            "max_mean_abs_diff": max_mean_abs_diff,
            "min_cosine_similarity": min_cosine_similarity,
            "min_top1_agreement": min_top1_agreement,
        },
    }
    if layerwise:
        capture_lengths = {
            "hf_hidden_states": len(hf_hidden_states),
            "olmo_hidden_states": len(olmo_hidden_states),
            "hf_attention_outputs": len(hf_attention_outputs),
            "olmo_attention_outputs": len(olmo_attention_outputs),
            "hf_router_indices": len(hf_router_indices),
            "olmo_router_indices": len(olmo_router_indices),
        }
        expected_layers = model_config.n_layers
        if any(length != expected_layers for length in capture_lengths.values()):
            raise RuntimeError(
                f"Expected {expected_layers} layer captures, got {capture_lengths}"
            )

        layers: list[dict[str, Any]] = []
        for layer_idx in range(expected_layers):
            router_top_k = hf_router_indices[layer_idx].shape[-1]
            hf_indices = hf_router_indices[layer_idx].reshape(-1, router_top_k)
            olmo_indices = olmo_router_indices[layer_idx].reshape(
                -1, router_top_k
            )
            hf_weights = hf_router_weights[layer_idx].reshape(-1, router_top_k)
            olmo_weights = olmo_router_weights[layer_idx].reshape(-1, router_top_k)
            hf_sorted = hf_indices.sort(dim=-1).values
            olmo_sorted = olmo_indices.sort(dim=-1).values
            layers.append(
                {
                    "layer": layer_idx,
                    "hidden_state": _tensor_metrics(
                        hf_hidden_states[layer_idx], olmo_hidden_states[layer_idx]
                    ),
                    "attention_output": _tensor_metrics(
                        hf_attention_outputs[layer_idx], olmo_attention_outputs[layer_idx]
                    ),
                    "router_weight": _tensor_metrics(
                        hf_weights, olmo_weights
                    ),
                    "router_index_position_agreement": (
                        hf_indices == olmo_indices
                    ).float().mean().item(),
                    "router_expert_set_agreement": (
                        hf_sorted == olmo_sorted
                    ).all(dim=-1).float().mean().item(),
                }
            )
        metrics["layers"] = layers

    log.info("Logit comparison:\n%s", json.dumps(metrics, indent=2))
    _write_metrics(output_json, metrics)
    if max_mean_abs_diff is not None and logit_metrics["mean_abs_diff"] > max_mean_abs_diff:
        raise AssertionError(
            f"Mean absolute logit difference {logit_metrics['mean_abs_diff']} exceeds "
            f"{max_mean_abs_diff}"
        )
    if min_cosine_similarity is not None and logit_metrics["cosine_similarity"] < min_cosine_similarity:
        raise AssertionError(
            f"Logit cosine similarity {logit_metrics['cosine_similarity']} is below "
            f"{min_cosine_similarity}"
        )
    if min_top1_agreement is not None and top1_agreement < min_top1_agreement:
        raise AssertionError(f"Top-1 agreement {top1_agreement} is below {min_top1_agreement}")
    if not skip_assert_close:
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
    parser.add_argument(
        "--hf-device",
        type=torch.device,
        help="Device for the Hugging Face reference; defaults to --device.",
    )
    parser.add_argument("--dtype", type=DType, default=DType.bfloat16)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--skip-assert-close", action="store_true")
    parser.add_argument("--max-mean-abs-diff", type=float)
    parser.add_argument("--min-cosine-similarity", type=float)
    parser.add_argument("--min-top1-agreement", type=float)
    parser.add_argument("--layerwise", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    prepare_cli_environment()
    verify_logits(
        hf_model=args.hf_model,
        checkpoint_path=args.checkpoint_path,
        revision=args.revision,
        prompt=args.prompt,
        hf_device=args.hf_device or args.device,
        device=args.device,
        dtype=args.dtype,
        rtol=args.rtol,
        atol=args.atol,
        skip_assert_close=args.skip_assert_close,
        max_mean_abs_diff=args.max_mean_abs_diff,
        min_cosine_similarity=args.min_cosine_similarity,
        min_top1_agreement=args.min_top1_agreement,
        layerwise=args.layerwise,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
