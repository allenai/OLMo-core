"""Evaluate released Hugging Face Molmo2 checkpoints on MMMU-Pro.

This runner uses the same pinned lmms-eval task definitions as ``s002_mmmu_pro.py``.
It supports normal cached greedy generation and the single-forward option-letter
scoring used to compare instruction-tuned releases with a pre-SFT Stage-1 model.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

DEFAULT_OUTPUT_ROOT = "/weka/oe-training-default/rustin/experiments/vision-moe/evals"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="allenai/Molmo2-4B")
    parser.add_argument("--revision", help="Hugging Face commit or ref (defaults to main).")
    parser.add_argument("--tasks", nargs="+", default=["mmmu_pro"])
    parser.add_argument(
        "--response-mode",
        choices=("generate", "letter_logits"),
        default="letter_logits",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override the task generation limit (generate-mode smoke tests only).",
    )
    parser.add_argument("--limit", type=float, help="lmms-eval sample limit (smoke only).")
    parser.add_argument("--hf-cache", help="Hugging Face cache root.")
    parser.add_argument("--output", help="Result JSON path under Rustin's eval root by default.")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    return parser.parse_args()


def _lmms_mmmu_dir() -> Path:
    import lmms_eval

    return Path(lmms_eval.__file__).parent / "tasks" / "mmmu_pro"


def _check_lmms_mmmu_assets() -> None:
    task_dir = _lmms_mmmu_dir()
    missing = [
        path
        for path in (
            task_dir / "_default_template_yaml",
            task_dir / "reasoning" / "_default_template_yaml",
        )
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            f"lmms-eval is missing {missing}. Apply requirements/lmms-eval-overrides "
            "as documented in src/scripts/eval/README.md."
        )


def _git_revision() -> Dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def _build_adapter_class():
    from accelerate import Accelerator, DistributedType
    from lmms_eval.api.instance import GenerationResult, TokenCounts
    from lmms_eval.api.model import lmms
    from lmms_eval.protocol import ChatMessages
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    class _Adapter(lmms):
        is_simple = False

        def __init__(
            self,
            checkpoint: str,
            *,
            response_mode: str,
            attn_implementation: str,
        ) -> None:
            super().__init__()
            self.accelerator = Accelerator()
            if self.accelerator.num_processes > 1:
                self._device = torch.device(f"cuda:{self.accelerator.local_process_index}")
            else:
                self._device = torch.device("cuda:0")

            self._model = AutoModelForImageTextToText.from_pretrained(
                checkpoint,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map=str(self._device),
                attn_implementation=attn_implementation,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(
                checkpoint,
                trust_remote_code=True,
                use_fast=False,
            )
            self._tokenizer = self.processor.tokenizer
            self.response_mode = response_mode

            option_encodings = [
                self.tokenizer.encode(letter, add_special_tokens=False) for letter in "ABCDEFGHIJ"
            ]
            if any(len(encoding) != 1 for encoding in option_encodings):
                raise ValueError("Each MMMU-Pro option letter must encode to one token")
            self.option_token_ids = [encoding[0] for encoding in option_encodings]

            if self.accelerator.num_processes > 1:
                if self.accelerator.distributed_type not in {
                    DistributedType.FSDP,
                    DistributedType.MULTI_GPU,
                }:
                    raise RuntimeError(
                        "Released Molmo2 evaluation supports multi-GPU DDP or FSDP only"
                    )
                self._model = self.accelerator.prepare_model(self._model, evaluation_mode=True)
            self._rank = self.accelerator.process_index
            self._world_size = self.accelerator.num_processes

        @property
        def model(self):
            return self.accelerator.unwrap_model(self._model)

        @property
        def tokenizer(self):
            return self._tokenizer

        @property
        def tokenizer_name(self) -> str:
            return str(getattr(self.tokenizer, "name_or_path", ""))

        @property
        def device(self) -> torch.device:
            return self._device

        @property
        def rank(self) -> int:
            return self._rank

        @property
        def world_size(self) -> int:
            return self._world_size

        def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
            raise NotImplementedError("MMMU-Pro uses generate_until")

        def generate_until_multi_round(self, requests) -> List[str]:
            raise NotImplementedError("MMMU-Pro is single-round")

        def _prepare_request(self, request):
            context, doc_to_messages, generation_kwargs, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            messages = ChatMessages(messages=doc_to_messages(doc))
            text = self.processor.apply_chat_template(
                messages.to_hf_messages(), tokenize=False, add_generation_prompt=True
            )
            images, videos, audios = messages.extract_media()
            media: Dict[str, Any] = {}
            if images:
                media["images"] = images
            if videos:
                media["videos"] = videos
            if audios:
                media["audios"] = audios
            inputs = self.processor(text=[text], padding=True, return_tensors="pt", **media).to(
                self.device
            )
            return context, generation_kwargs, inputs

        def _score_letter(self, inputs) -> Tuple[str, TokenCounts]:
            with torch.inference_mode():
                output = self.model(**inputs, use_cache=False, return_dict=True)
            option_ids = torch.tensor(self.option_token_ids, dtype=torch.long, device=self.device)
            option_index = int(output.logits[0, -1, option_ids].argmax(dim=-1).item())
            return "ABCDEFGHIJ"[option_index], TokenCounts(
                input_tokens=int(inputs["input_ids"].shape[1]), output_tokens=1
            )

        def _generate(self, inputs, generation_kwargs: Dict[str, Any]) -> Tuple[str, TokenCounts]:
            max_new_tokens = int(generation_kwargs.get("max_new_tokens", 256))
            temperature = float(generation_kwargs.get("temperature", 0.0))
            do_sample = temperature > 0
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=generation_kwargs.get("top_p") if do_sample else None,
                    num_beams=int(generation_kwargs.get("num_beams", 1)),
                    use_cache=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated = output_ids[0, inputs["input_ids"].shape[1] :]
            text = self.processor.decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            return text, TokenCounts(
                input_tokens=int(inputs["input_ids"].shape[1]),
                output_tokens=int(generated.numel()),
            )

        def generate_until(self, requests) -> List[GenerationResult]:
            responses: List[GenerationResult] = []
            for request in tqdm(requests, desc="MMMU-Pro", disable=self.rank != 0):
                context, generation_kwargs, inputs = self._prepare_request(request)
                if self.response_mode == "letter_logits":
                    text, token_counts = self._score_letter(inputs)
                else:
                    text, token_counts = self._generate(inputs, generation_kwargs)
                responses.append(GenerationResult(text=text, token_counts=token_counts))
                self.cache_hook.add_partial("generate_until", (context, generation_kwargs), text)
            return responses

    return _Adapter


def _default_output(model: str, response_mode: str, partial: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "partial" if partial else "complete"
    model_name = model.rstrip("/").split("/")[-1]
    return (
        Path(DEFAULT_OUTPUT_ROOT)
        / model_name
        / f"lmms-mmmu-pro-{response_mode}-{suffix}-{stamp}.json"
    )


def _resolve_snapshot(
    model: str, revision: str | None, hf_cache: str | None
) -> tuple[Path, str | None]:
    model_path = Path(model).expanduser()
    if model_path.exists():
        if revision is not None:
            raise ValueError("--revision cannot be used with a local model path")
        if not model_path.is_dir():
            raise ValueError(f"Local model path is not a directory: {model_path}")
        return model_path.resolve(), None

    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            cache_dir=hf_cache,
        )
    ).resolve()
    return snapshot, snapshot.name


def main() -> None:
    args = _parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.response_mode == "letter_logits" and args.max_new_tokens is not None:
        raise ValueError("--max-new-tokens is incompatible with --response-mode=letter_logits")
    if args.hf_cache:
        os.environ["HF_HOME"] = str(Path(args.hf_cache).resolve())

    from lmms_eval import evaluator as lmms_evaluator
    from lmms_eval.tasks import TaskManager
    from lmms_eval.utils import handle_non_serializable

    _check_lmms_mmmu_assets()
    snapshot, resolved_revision = _resolve_snapshot(args.model, args.revision, args.hf_cache)
    adapter_type = _build_adapter_class()
    adapter = adapter_type(
        str(snapshot),
        response_mode=args.response_mode,
        attn_implementation=args.attn_implementation,
    )
    task_manager = TaskManager(
        verbosity="INFO",
        include_path=str(_lmms_mmmu_dir()),
        include_defaults=False,
        model_name="huggingface",
    )
    results = lmms_evaluator.simple_evaluate(
        model=adapter,
        tasks=args.tasks,
        task_manager=task_manager,
        limit=args.limit,
        bootstrap_iters=0,
        log_samples=True,
        gen_kwargs=(
            f"max_new_tokens={args.max_new_tokens}" if args.max_new_tokens is not None else None
        ),
        distributed_executor_backend="accelerate",
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )
    if adapter.rank == 0:
        if results is None:
            raise RuntimeError("lmms-eval returned no results on rank 0")

        partial = args.limit is not None or args.max_new_tokens is not None
        output = (
            Path(args.output)
            if args.output
            else _default_output(args.model, args.response_mode, partial)
        )
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "snapshot": str(snapshot),
            "git": _git_revision(),
            "protocol": {
                "harness": "lmms-eval",
                "tasks": args.tasks,
                "partial": partial,
                "limit": args.limit,
                "world_size": adapter.world_size,
                "response_mode": args.response_mode,
                "generation": (
                    "single_forward_option_letter_logits"
                    if args.response_mode == "letter_logits"
                    else "greedy_hf_generate_with_kv_cache"
                ),
                "max_new_tokens_override": args.max_new_tokens,
                "attn_implementation": args.attn_implementation,
                "dtype": "bfloat16",
            },
            "lmms_eval": results,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                default=handle_non_serializable,
            )
            + "\n"
        )
        temporary.replace(output)
        print(f"Wrote results to {output}")

    adapter.accelerator.wait_for_everyone()
    adapter.accelerator.end_training()


if __name__ == "__main__":
    main()
