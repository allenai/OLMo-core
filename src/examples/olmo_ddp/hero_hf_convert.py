"""Convert one scratch checkpoint and qualify tensor/logit/cached-decoding fidelity."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

from hero_hf_download import BATCH, SCRATCH, TARGETS, prepare_scratch, write_json

log = logging.getLogger(__name__)


def reference_config(config: dict) -> dict:
    """Use semantic-reference kernels only; all model parameters/architecture stay unchanged."""
    result = copy.deepcopy(config)

    def visit(value):
        if isinstance(value, dict):
            if "use_cute_kernel" in value:
                value["use_cute_kernel"] = False
            if value.get("qk_norm_per_head_gains"):
                value["backend"] = "torch"
                value["use_flash"] = False
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(result)
    return result


def cases(tokenizer, vocab: int, full: bool):
    """Deterministic natural-language/code/math and kernel-boundary input shapes."""
    import torch

    texts = [
        "The history of scientific discovery is full of careful experiments. A useful explanation is",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\n# Example:\n",
        "Question: A train travels 120 kilometers in two hours. What is its average speed?\nAnswer:",
    ]
    for i, text in enumerate(texts):
        yield f"text{i}", tokenizer(text, return_tensors="pt").input_ids
    lengths = (63, 65, 257, 1024, 8192) if full else (65, 257)
    generator = torch.Generator().manual_seed(20260909)
    for n in lengths:
        yield f"random{n}", torch.randint(0, vocab, (1, n), generator=generator)
    yield "batch2x127", torch.randint(0, vocab, (2, 127), generator=generator)


def statistics(actual, expected) -> dict:
    """Compare full logits without masking rare/large discrepancies."""
    import torch

    a, b = actual.float(), expected.float()
    diff = a - b
    return {
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "relative_l2": (diff.norm() / b.norm().clamp_min(1e-12)).item(),
        "top1_agreement": (a.argmax(-1) == b.argmax(-1)).float().mean().item(),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    }


def qualify(root: Path, *, full: bool, precise: bool = False) -> dict:
    """Run an exact semantic-reference comparison plus independent HF cache checks."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from olmo_core.nn.hf.config import _register_olmo3moe_auto_classes
    from olmo_core.nn.hf.convert_checkpoint import _load_ddp_optimizer_model_state, load_config
    from olmo_core.nn.transformer.config import TransformerConfig

    _register_olmo3moe_auto_classes()
    raw, hf = root / "olmo-core", root / "hf.partial"
    # Qualification-only retries may reuse serialized weights, but never publish
    # stale remote-code files from a prior exporter revision.
    from olmo_core.nn.moe.v2.hf import configuration_olmo3moe, modeling_olmo3moe

    for module in (configuration_olmo3moe, modeling_olmo3moe):
        source_file = Path(module.__file__)
        shutil.copy2(source_file, hf / source_file.name)
    experiment = load_config(raw)
    tokenizer = AutoTokenizer.from_pretrained(hf)
    exported_vocab = json.loads((hf / "config.json").read_text())["vocab_size"]
    if exported_vocab != experiment["dataset"]["tokenizer"]["vocab_size"]:
        raise RuntimeError("HF vocabulary does not match the training tokenizer")
    cfg = reference_config(experiment["model"])
    model = TransformerConfig.from_dict(cfg).build(init_device="meta")
    model.to_empty(device="cpu")
    _load_ddp_optimizer_model_state(
        raw / "model_and_optim", model, work_dir=str(root / "load-work"), return_state_dict=False
    )
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    inputs = list(cases(tokenizer, tokenizer.vocab_size, full))
    references = {}
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for name, ids in inputs:
            log.info("HERO_CORE_REFERENCE case=%s shape=%s", name, tuple(ids.shape))
            # The trainer pads the embedding/LM-head matrices to a multiple of 128.
            # HF intentionally removes those non-token rows, as the stock verifier does.
            references[name] = model(input_ids=ids.cuda())[..., :exported_vocab].cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    hf_model = (
        AutoModelForCausalLM.from_pretrained(
            hf, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    if not hf_model.config.qk_norm_per_head_gains or hf_model.config.latent_moe_dim != 512:
        raise RuntimeError("Export lost the hero's per-head gains or latent dimension")
    rows = []
    # This first gate isolates the mapping from different BF16 GEMM/kernel arithmetic.
    os.environ["OLMO_HF_MOE_CORE_REFERENCE"] = "1"
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for name, ids in inputs:
            out = hf_model(input_ids=ids.cuda(), use_cache=False).logits.cpu()
            ref = references.pop(name)
            stats = statistics(out, ref)
            rows.append({"case": name, "gate": "core_hf_reference", **stats})
            log.info("HERO_PARITY %s", json.dumps(rows[-1]))
            torch.testing.assert_close(out.float(), ref.float(), rtol=1e-4, atol=1e-4)
            del out, ref
    # Independent standalone HF route: never leave the core-layout oracle enabled
    # for the cached/uncached check or the later production vLLM evaluation.
    os.environ.pop("OLMO_HF_MOE_CORE_REFERENCE", None)
    os.environ["OLMO_HF_MOE_REFERENCE_LOOP"] = "1"
    if precise:
        # Keep the exact BF16 mapping oracle above independent of the numerical
        # precision recipe. Reload only AFTER that gate has passed.
        del hf_model
        gc.collect()
        torch.cuda.empty_cache()
        from olmo_core.nn.moe.v2.hf import inference_settings

        inference_settings.install(linear="float64", sdpa="float64", recurrent=True)
        shutil.copy2(inference_settings.__file__, hf / "inference_settings.py")
        write_json(
            hf / "inference-settings.json",
            dict(
                profile=inference_settings.PRECISE_PROFILE,
                model_load_dtype="float32",
                linear_compute="float64_then_float32",
                hf_attention="math_sdpa_float64_then_float32",
                kda="existing_fla_recurrent_prefill_and_decode",
                vllm_attention="existing_flex_attention_float32",
                vllm_env={"OLMO_HERO_PRECISE_INFERENCE": "1", "OLMO_VLLM_FLA_KDA": "1"},
                vllm_pool_blocks=128,
                note="Opt-in offline recipe, not a claim that default BF16 cache gates pass.",
            ),
        )
        # Transformers 5.16 deliberately prefers an explicit local AutoModel
        # registration, even with trust_remote_code=True. The mapping oracle
        # registered that class above, so load the bundled class directly here.
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        exported_config = json.loads((hf / "config.json").read_text())
        class_ref = exported_config.get("auto_map", {}).get("AutoModelForCausalLM")
        if class_ref != "modeling_olmo3moe.Olmo3MoeForCausalLM":
            raise RuntimeError("Export does not register the expected standalone HF model")
        exported_class = get_class_from_dynamic_module(class_ref, hf, local_files_only=True)
        hf_model = (
            exported_class.from_pretrained(hf, dtype=torch.float32, attn_implementation="sdpa")
            .cuda()
            .eval()
        )
        if not hf_model.__class__.__module__.startswith("transformers_modules."):
            raise RuntimeError("Portable HF gate must load the exported remote-code files")
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for name, ids in inputs:
            if ids.shape[1] > 1024:
                continue
            ids = ids.cuda()
            split = ids.shape[1] - 4
            complete = hf_model(ids, use_cache=False).logits[:, split:].cpu()
            first = hf_model(ids[:, :split], use_cache=True)
            cache = first.past_key_values
            pieces = []
            for index in range(split, ids.shape[1]):
                part = hf_model(ids[:, index : index + 1], past_key_values=cache, use_cache=True)
                cache = part.past_key_values
                pieces.append(part.logits.cpu())
            decoded = torch.cat(pieces, dim=1)
            stats = statistics(decoded, complete)
            rows.append({"case": name, "gate": "hf_cached_vs_uncached", **stats})
            log.info("HERO_PARITY %s", json.dumps(rows[-1]))
            # Keep the same numerical gates for either explicitly recorded profile.
            if not stats["finite"] or stats["relative_l2"] > 0.005:
                raise AssertionError(f"Cached HF relative-L2 gate failed: {stats}")
            logprob_error = (
                decoded.float().log_softmax(-1) - complete.float().log_softmax(-1)
            ).abs()
            if logprob_error.mean().item() > 0.01 or logprob_error.max().item() > 0.25:
                raise AssertionError("Cached HF full-vocabulary logprob gate failed")
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "passed": True,
        "full": full,
        "checks": rows,
        "compared_vocabulary": exported_vocab,
        "reference_logit_rtol": 1e-4,
        "reference_logit_atol": 1e-4,
        "cache_relative_l2_limit": 0.005,
        "cache_logprob_mean_abs_limit": 0.01,
        "cache_logprob_max_abs_limit": 0.25,
        "inference_profile": (inference_settings.PRECISE_PROFILE if precise else "bf16"),
        "exported_remote_code_tested": precise,
    }


def main() -> None:
    """Convert/qualify one input. Raw checkpoint deletion is a separate gated action."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("emo", "non-emo"), required=True)
    parser.add_argument("--step", type=int, choices=tuple(TARGETS.values()), required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--qualify-only", action="store_true")
    parser.add_argument("--portable-reference", action="store_true")
    parser.add_argument("--precise", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prepare_scratch()
    root = SCRATCH / args.arm / f"step{args.step}"
    if root.resolve() != root or not (root / "download-success.json").is_file():
        raise RuntimeError("Verified scratch download is required")
    raw, output = root / "olmo-core", root / "hf.partial"
    if output.is_symlink() or (root / "hf").exists():
        raise RuntimeError("Refusing to overwrite a published HF conversion")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Strict conversion requires a GPU")
    from olmo_core.config import DType
    from olmo_core.nn.hf.convert_checkpoint import convert_checkpoint_to_hf, load_config

    if args.portable_reference:
        from hero_hf_reference_ops import install, self_test

        self_test()
        install()
        log.info(
            "Using offline-only portable expert reference; independent HF/vLLM gates remain separate"
        )

    if not args.qualify_only:
        if output.exists():
            raise RuntimeError(
                "Partial output exists; inspect it and use --qualify-only if complete"
            )
        config = load_config(raw)
        os.environ["OLMO_USE_TORCH_GROUPED_MM"] = "0"
        os.environ["OLMO_HF_MOE_CORE_REFERENCE"] = "1"
        # Offline reference kernel choice is explicitly recorded, not a change to
        # training or an attempt to hide missing parameters in the conversion.
        convert_checkpoint_to_hf(
            raw,
            output,
            reference_config(config["model"]),
            config["dataset"]["tokenizer"],
            dtype=DType.bfloat16,
            max_sequence_length=8192,
            device=torch.device("cpu"),
            validation_device=torch.device("cuda"),
            validate=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
    result = qualify(root, full=args.full, precise=args.precise)
    result.update(
        {
            "arm": args.arm,
            "step": args.step,
            "tokens": args.step * BATCH,
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "reference_experts": (
                "portable_pytorch" if args.portable_reference else "training_extension"
            ),
            "completed_at_unix": time.time(),
            "output": str(root / "hf"),
        }
    )
    # Preserve exact serialized-output checksums for subsequent reload/eval/cleanup gates.
    hashes = {}
    for path in sorted(output.iterdir()):
        if path.is_file():
            with path.open("rb") as handle:
                hashes[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    result["output_sha256"] = hashes
    write_json(output / "_HERO_CONVERSION_SUCCESS.json", result)
    os.rename(output, root / "hf")
    write_json(root / "conversion-success.json", result)
    print("HERO_HF_CONVERSION_SUCCESS " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
