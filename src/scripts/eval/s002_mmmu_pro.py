"""Run MMMU-Pro against a native multimodal s002 Stage-1 checkpoint.

The checkpoint stays in its native OLMo-core distributed-checkpoint format and uses
expert parallelism. Every EP rank therefore executes the same lmms-eval request; the
harness is exposed as a single logical replica even when ``torchrun`` has many ranks.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import torch
import torch.distributed as dist
from s002_downstream import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TOKENIZER,
    _build_model_and_module_config,
    _checkpoint_state_dir,
    _config_path,
    _git_revision,
)

from olmo_core.data.multimodal.document_layout import document_prompt_ids, response_ids
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
from olmo_core.nn.vision.molmo2_image_processor import preprocess_image_molmo2
from olmo_core.nn.vision.molmo2_tokens import (
    Molmo2TokenIds,
    build_image_token_ids,
    prepare_molmo2_tokenizer,
)
from olmo_core.train import prepare_training_environment, teardown_training_environment

log = logging.getLogger(__name__)

_IMAGE_MARKER = re.compile(r"<image(?:\s+\d+)?>", flags=re.IGNORECASE)
_PROMPT_LAYOUTS = ("document", "text_sft", "answer_cue", "bare_chat")
_RESPONSE_MODES = ("generate", "letter_logits", "option_text_mean", "option_text_sum")


def _set_model_parts_eval(train_module) -> None:
    """Put every sharded model part into deterministic evaluation mode."""
    for model_part in train_module.model_parts:
        model_part.eval()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", help="Config JSON (defaults to CHECKPOINT/config.json).")
    parser.add_argument("--tasks", nargs="+", default=["mmmu_pro"])
    parser.add_argument("--ep-degree", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument(
        "--max-crops-total",
        type=int,
        default=8,
        help="High-resolution crop budget divided across all images in one question.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override the task's generation limit (use only for smoke tests).",
    )
    parser.add_argument(
        "--response-mode",
        choices=_RESPONSE_MODES,
        default="generate",
        help=(
            "Use normal greedy generation, score the ten option letters with one forward, "
            "or compare full option-text log likelihoods with mean/sum normalization."
        ),
    )
    parser.add_argument(
        "--prompt-layout",
        choices=_PROMPT_LAYOUTS,
        default="document",
        help=(
            "Diagnostic prompt interface. 'document' is the unchanged Stage-1 default; "
            "the other layouts test text-SFT, completion-cue, and bare role-header calibration."
        ),
    )
    parser.add_argument(
        "--sequence-bucket-size",
        type=int,
        default=128,
        help="Pad prompt+generation to this multiple to reuse FlexAttention kernels.",
    )
    parser.add_argument("--limit", type=float, help="lmms-eval sample limit (smoke only).")
    parser.add_argument("--tokenizer", help="Override the tokenizer ID from checkpoint config.")
    parser.add_argument("--hf-cache", help="Hugging Face cache root.")
    parser.add_argument("--output", help="Result JSON path under Rustin's eval root by default.")
    parser.add_argument("--checkpoint-load-threads", type=int, default=8)
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
            f"lmms-eval is missing {missing}. Its wheel omits these extensionless upstream "
            "assets; apply requirements/lmms-eval-overrides as documented in "
            "src/scripts/eval/README.md."
        )


@contextmanager
def _lmms_single_logical_replica() -> Iterator[None]:
    """Make lmms-eval replicate requests instead of sharding an EP model across ranks."""
    original = {name: os.environ.get(name) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")}
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    try:
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _clean_context(context: str) -> str:
    return _IMAGE_MARKER.sub("", context).strip()


def _prompt_ids_for_layout(
    tokenizer,
    context: str,
    image_ids: Sequence[int],
    *,
    layout: str,
) -> List[int]:
    """Build one of the explicitly diagnostic s002 MMMU prompt layouts."""
    if layout == "document":
        return document_prompt_ids(tokenizer, context, image_ids=image_ids)
    if layout == "text_sft":
        return document_prompt_ids(tokenizer, f"text_sft: {context}", image_ids=image_ids)
    if layout == "answer_cue":
        return document_prompt_ids(tokenizer, f"{context}\nAnswer:", image_ids=image_ids)
    if layout == "bare_chat":
        user_header = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
        assistant_suffix = tokenizer.encode(
            f"{context}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        return [*user_header, *image_ids, *assistant_suffix]
    raise ValueError(f"Unknown prompt layout {layout!r}")


def _candidate_ids_for_layout(tokenizer, text: str, *, layout: str) -> List[int]:
    """Tokenize a candidate exactly where the selected prompt expects a response."""
    if layout == "bare_chat":
        return tokenizer.encode(text, add_special_tokens=False)
    if layout in _PROMPT_LAYOUTS:
        return response_ids(tokenizer, text)
    raise ValueError(f"Unknown prompt layout {layout!r}")


def _parse_option_texts(doc: Dict[str, Any]) -> List[str]:
    """Parse MMMU-Pro's string-serialized option list without accepting other literals."""
    raw_options = doc.get("options")
    if isinstance(raw_options, str):
        raw_options = ast.literal_eval(raw_options)
    if not isinstance(raw_options, list) or not 2 <= len(raw_options) <= 10:
        raise ValueError("MMMU-Pro options must be a list containing between 2 and 10 entries")
    if not all(isinstance(option, str) and option for option in raw_options):
        raise ValueError("Every MMMU-Pro option must be a non-empty string")
    return raw_options


def _build_adapter_class():
    from lmms_eval.api.instance import GenerationResult, TokenCounts
    from lmms_eval.api.model import lmms

    class _Adapter(lmms):
        is_simple = True

        def __init__(
            self,
            train_module,
            tokenizer,
            token_ids: Molmo2TokenIds,
            *,
            max_sequence_length: int,
            max_crops_total: int,
            max_new_tokens: int | None,
            sequence_bucket_size: int,
            response_mode: str,
            prompt_layout: str,
            text_vocab_size: int,
        ) -> None:
            super().__init__()
            self.train_module = train_module
            self.model = train_module.model_parts[0]
            self.tokenizer = tokenizer
            self.token_ids = token_ids
            self.max_sequence_length = max_sequence_length
            self.max_crops_total = max_crops_total
            self.max_new_tokens_override = max_new_tokens
            self.sequence_bucket_size = sequence_bucket_size
            self.response_mode = response_mode
            self.prompt_layout = prompt_layout
            self.text_vocab_size = text_vocab_size
            # Stage 1 uses Molmo's role-free message format, where every response is the second
            # message and therefore begins with one space. Score the tokens the model was trained
            # to predict, not the bare Qwen-style option letters.
            option_encodings = [
                _candidate_ids_for_layout(self.tokenizer, letter, layout=self.prompt_layout)
                for letter in "ABCDEFGHIJ"
            ]
            if any(len(encoding) != 1 for encoding in option_encodings):
                raise ValueError("Each MMMU-Pro option letter must encode to one token")
            self.option_token_ids = [encoding[0] for encoding in option_encodings]
            self._rank = 0
            self._world_size = 1

        @property
        def device(self) -> torch.device:
            return self.train_module.device

        @property
        def tokenizer_name(self) -> str:
            return str(getattr(self.tokenizer, "name_or_path", ""))

        def clean(self) -> None:
            # The runner owns the distributed module and tears it down after result writing.
            return None

        def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
            raise NotImplementedError("The MMMU-Pro protocol uses generate_until")

        def generate_until_multi_round(self, requests) -> List[str]:
            raise NotImplementedError("The MMMU-Pro protocol is single-round")

        def _prepare_images(
            self, visuals: Sequence[Any]
        ) -> Tuple[torch.Tensor | None, torch.Tensor | None, List[int]]:
            if not visuals:
                return None, None, []

            if len(visuals) > self.max_crops_total:
                raise ValueError(
                    f"Request has {len(visuals)} images but --max-crops-total="
                    f"{self.max_crops_total}; at least one high-resolution crop is required "
                    "per image"
                )
            base_crops, extra_crops = divmod(self.max_crops_total, len(visuals))
            crop_budgets = [base_crops + int(index < extra_crops) for index in range(len(visuals))]
            image_parts: List[torch.Tensor] = []
            pooling_parts: List[torch.Tensor] = []
            image_token_ids: List[int] = []
            patch_offset = 0

            for visual, crop_budget in zip(visuals, crop_budgets):
                images, pooling, grid = preprocess_image_molmo2(
                    visual,
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                    max_crops=crop_budget,
                    is_training=False,
                )
                adjusted_pooling = pooling.clone()
                valid = adjusted_pooling >= 0
                adjusted_pooling[valid] += patch_offset
                patch_offset += int(images.shape[1] * images.shape[2])

                image_parts.append(images)
                pooling_parts.append(adjusted_pooling)
                resized_h, resized_w, height, width = (int(grid[i]) for i in range(4))
                image_token_ids.extend(
                    build_image_token_ids(
                        resized_h,
                        resized_w,
                        height,
                        width,
                        token_ids=self.token_ids,
                    )
                )

            return (
                torch.cat(image_parts, dim=1),
                torch.cat(pooling_parts, dim=1),
                image_token_ids,
            )

        def _prompt_ids(self, context: str, image_ids: Sequence[int]) -> List[int]:
            context = _clean_context(context)
            return _prompt_ids_for_layout(
                self.tokenizer,
                context,
                image_ids,
                layout=self.prompt_layout,
            )

        def _generation_length(self, generation_kwargs: Dict[str, Any]) -> int:
            if self.response_mode != "generate":
                return 1
            value = self.max_new_tokens_override
            if value is None:
                value = int(generation_kwargs.get("max_new_tokens", 256))
            if value <= 0:
                raise ValueError("max_new_tokens must be positive")
            return value

        def _buffer_length(self, required: int) -> int:
            rounded = (
                (required + self.sequence_bucket_size - 1) // self.sequence_bucket_size
            ) * self.sequence_bucket_size
            return min(rounded, self.max_sequence_length)

        def _build_inputs(
            self,
            prompt_ids: Sequence[int],
            image_ids: Sequence[int],
            *,
            suffix_ids: Sequence[int] = (),
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            required = len(prompt_ids) + len(suffix_ids)
            if required > self.max_sequence_length:
                raise ValueError(
                    f"MMMU-Pro request needs {required} tokens, exceeding "
                    f"--max-sequence-length={self.max_sequence_length}. Reduce "
                    "--max-crops-total; prompt and candidate truncation are disabled."
                )
            buffer_length = self._buffer_length(required)
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            if pad_token_id is None:
                raise ValueError("Tokenizer has neither a pad token nor an EOS token")

            input_ids = torch.full(
                (1, buffer_length),
                int(pad_token_id),
                dtype=torch.long,
                device=self.device,
            )
            full_ids = [*prompt_ids, *suffix_ids]
            input_ids[0, :required] = torch.tensor(full_ids, device=self.device)
            token_type_ids = torch.zeros_like(input_ids)
            prompt_tensor = input_ids[0, : len(prompt_ids)]
            for image_token_id in self.token_ids.image_token_ids:
                token_type_ids[0, : len(prompt_ids)] |= prompt_tensor.eq(image_token_id)
            position_ids = torch.arange(buffer_length, device=self.device).unsqueeze(0)
            return input_ids, token_type_ids, position_ids

        def _score_option_texts(
            self,
            prompt_ids: Sequence[int],
            image_ids: Sequence[int],
            option_texts: Sequence[str],
            encoded_features: torch.Tensor | None,
        ) -> int:
            scores: List[torch.Tensor] = []
            for option_text in option_texts:
                candidate_ids = _candidate_ids_for_layout(
                    self.tokenizer,
                    option_text,
                    layout=self.prompt_layout,
                )
                if not candidate_ids:
                    raise ValueError("An MMMU-Pro option tokenized to an empty sequence")
                input_ids, token_type_ids, position_ids = self._build_inputs(
                    prompt_ids,
                    image_ids,
                    suffix_ids=candidate_ids,
                )
                logits_positions = torch.arange(
                    len(prompt_ids) - 1,
                    len(prompt_ids) + len(candidate_ids) - 1,
                    device=self.device,
                ).unsqueeze(0)
                logits = self.train_module.model_forward_no_pipeline(
                    input_ids,
                    encoded_image_features=encoded_features,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    logits_to_keep=logits_positions,
                )
                if not isinstance(logits, torch.Tensor):
                    raise TypeError(f"Expected logits tensor, got {type(logits).__name__}")
                targets = torch.tensor(candidate_ids, device=self.device).unsqueeze(0)
                token_log_probs = (
                    logits.float().log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                )
                if self.response_mode == "option_text_mean":
                    scores.append(token_log_probs.mean())
                elif self.response_mode == "option_text_sum":
                    scores.append(token_log_probs.sum())
                else:
                    raise RuntimeError(f"Unexpected option-text mode {self.response_mode!r}")
            return int(torch.stack(scores).argmax().item())

        def _generate_one(
            self,
            context: str,
            generation_kwargs: Dict[str, Any],
            visuals: Sequence[Any],
            option_texts: Sequence[str] | None = None,
        ) -> GenerationResult:
            images, pooling, image_ids = self._prepare_images(visuals)
            prompt_ids = self._prompt_ids(context, image_ids)
            max_new_tokens = self._generation_length(generation_kwargs)
            required = len(prompt_ids) + max_new_tokens
            if required > self.max_sequence_length:
                raise ValueError(
                    f"MMMU-Pro request needs {required} tokens ({len(prompt_ids)} prompt + "
                    f"{max_new_tokens} generation), exceeding --max-sequence-length="
                    f"{self.max_sequence_length}. Reduce --max-crops-total or the smoke-only "
                    "--max-new-tokens override; prompt truncation is intentionally disabled."
                )

            input_ids, token_type_ids, position_ids = self._build_inputs(
                prompt_ids,
                image_ids,
            )
            buffer_length = int(input_ids.shape[1])

            encoded_features = None
            if images is not None:
                if pooling is None:
                    raise AssertionError("Image pooling indices are missing")
                with torch.inference_mode():
                    encoded_features = self.model.encode_images(images, pooling)

            if self.response_mode.startswith("option_text_"):
                if option_texts is None:
                    raise ValueError("Option-text scoring requires MMMU-Pro option strings")
                option_index = self._score_option_texts(
                    prompt_ids,
                    image_ids,
                    option_texts,
                    encoded_features,
                )
                answer = chr(ord("A") + option_index)
                return GenerationResult(
                    text=answer,
                    token_counts=TokenCounts(
                        input_tokens=len(prompt_ids),
                        output_tokens=1,
                    ),
                )

            until = generation_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            elif until is None:
                until = []
            elif not isinstance(until, list):
                raise TypeError("generation_kwargs['until'] must be a string or list")

            generated: List[int] = []
            with torch.inference_mode():
                for _ in range(max_new_tokens):
                    current_length = len(prompt_ids) + len(generated)
                    logits_position = torch.tensor(
                        [[current_length - 1]], dtype=torch.long, device=self.device
                    )
                    logits = self.train_module.model_forward_no_pipeline(
                        input_ids,
                        encoded_image_features=encoded_features,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        logits_to_keep=logits_position,
                    )
                    if not isinstance(logits, torch.Tensor):
                        raise TypeError(f"Expected logits tensor, got {type(logits).__name__}")
                    if self.response_mode == "letter_logits":
                        option_ids = torch.tensor(
                            self.option_token_ids, dtype=torch.long, device=self.device
                        )
                        option_index = int(logits[0, 0, option_ids].argmax(dim=-1).item())
                        next_token = self.option_token_ids[option_index]
                    else:
                        # Native s002 trains against a model-padded output vocabulary, but its HF
                        # text interface exports only tokenizer rows. Prevent greedy decoding from
                        # emitting padded or image-only IDs while leaving training/log-likelihood
                        # normalization over the native full vocabulary unchanged.
                        next_token = int(logits[0, 0, : self.text_vocab_size].argmax(dim=-1).item())

                    # Fail loudly if nominally replicated EP/DP groups ever diverge.
                    if dist.is_initialized() and dist.get_world_size() > 1:
                        consensus = torch.tensor(
                            [next_token, next_token], dtype=torch.int64, device=self.device
                        )
                        dist.all_reduce(consensus[:1], op=dist.ReduceOp.MIN)
                        dist.all_reduce(consensus[1:], op=dist.ReduceOp.MAX)
                        if int(consensus[0]) != int(consensus[1]):
                            raise RuntimeError(
                                "Distributed generation diverged across logical replicas: "
                                f"min={int(consensus[0])}, max={int(consensus[1])}"
                            )

                    generated.append(next_token)
                    if self.response_mode == "letter_logits":
                        break
                    if next_token == self.tokenizer.eos_token_id:
                        break
                    if current_length >= buffer_length:
                        break
                    input_ids[0, current_length] = next_token
                    decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
                    if any(term and term in decoded for term in until):
                        break

            answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            stop_positions = [answer.find(term) for term in until if term and term in answer]
            if stop_positions:
                answer = answer[: min(stop_positions)].strip()
            return GenerationResult(
                text=answer,
                token_counts=TokenCounts(
                    input_tokens=len(prompt_ids),
                    output_tokens=len(generated),
                ),
            )

        def generate_until(self, requests) -> List[GenerationResult]:
            from tqdm import tqdm

            responses: List[GenerationResult] = []
            actual_rank = dist.get_rank() if dist.is_initialized() else 0
            iterator = tqdm(requests, desc="MMMU-Pro", disable=actual_rank != 0)
            for request in iterator:
                context, generation_kwargs, doc_to_visual, doc_id, task, split = request.args
                doc = self.task_dict[task][split][doc_id]
                visuals = doc_to_visual(doc)
                if visuals is None:
                    visuals = []
                elif not isinstance(visuals, list):
                    visuals = [visuals]
                option_texts = (
                    _parse_option_texts(doc)
                    if self.response_mode.startswith("option_text_")
                    else None
                )
                response = self._generate_one(
                    context,
                    generation_kwargs,
                    visuals,
                    option_texts=option_texts,
                )
                responses.append(response)
                self.cache_hook.add_partial(
                    "generate_until", (context, generation_kwargs), response.text
                )
            return responses

    return _Adapter


def _default_output(checkpoint: Path, partial: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "partial" if partial else "complete"
    name = checkpoint.parent.name if checkpoint.name == "model_and_optim" else checkpoint.name
    return Path(DEFAULT_OUTPUT_ROOT) / name / f"lmms-mmmu-pro-{suffix}-{stamp}.json"


def main() -> None:
    args = _parse_args()
    if args.ep_degree <= 0:
        raise ValueError("--ep-degree must be positive")
    if args.max_sequence_length <= 0:
        raise ValueError("--max-sequence-length must be positive")
    if args.max_crops_total <= 0:
        raise ValueError("--max-crops-total must be positive")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.response_mode != "generate" and args.max_new_tokens is not None:
        raise ValueError("--max-new-tokens is only compatible with --response-mode=generate")
    if args.sequence_bucket_size <= 0:
        raise ValueError("--sequence-bucket-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size % args.ep_degree != 0:
        raise ValueError(
            f"WORLD_SIZE ({world_size}) must be divisible by --ep-degree ({args.ep_degree})"
        )

    if args.hf_cache:
        os.environ["HF_HOME"] = str(Path(args.hf_cache).resolve())
    os.environ.setdefault("OLMO_USE_OWN_SYMM_MEM", "1")
    os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1")
    os.environ.setdefault("OLMO_OWN_SYMM_PREWARM", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    prepare_training_environment()

    try:
        from lmms_eval import evaluator as lmms_evaluator
        from lmms_eval.tasks import TaskManager
        from lmms_eval.utils import handle_non_serializable
        from transformers import GPT2Tokenizer

        _check_lmms_mmmu_assets()
        checkpoint = Path(args.checkpoint).resolve()
        config_path = _config_path(checkpoint, args.config)
        with config_path.open() as f:
            raw_config = json.load(f)

        model, module_config, checkpoint_kind = _build_model_and_module_config(
            raw_config,
            ep_degree=args.ep_degree,
            max_sequence_length=args.max_sequence_length,
            rank_batch_size=args.max_sequence_length,
            # The rowwise NVSHMEM path can leave its persistent kernels waiting
            # indefinitely for batch-one autoregressive inference. The synchronized
            # all-to-all path uses the same checkpoint sharding without that constraint.
            ep_path=ExpertParallelPath.sync_1d,
        )
        if checkpoint_kind != "multimodal_stage1":
            raise ValueError(
                "MMMU-Pro requires a multimodal Stage-1 checkpoint; the supplied checkpoint "
                f"was detected as {checkpoint_kind!r}."
            )

        train_module = module_config.build(model, eval_only=True)
        state_dir = _checkpoint_state_dir(checkpoint)
        log.info("Loading native multimodal checkpoint from %s", state_dir)
        train_module.load_state_dict_direct(
            state_dir,
            process_group=dist.group.WORLD,
            thread_count=args.checkpoint_load_threads,
            load_optim_state=False,
        )
        _set_model_parts_eval(train_module)

        experiment = raw_config.get("experiment", raw_config)
        tokenizer_id = args.tokenizer or experiment.get("tokenizer_id") or DEFAULT_TOKENIZER
        hf_cache = args.hf_cache or experiment.get("hf_cache_dir")
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_id, cache_dir=hf_cache)
        model_vocab_size = int(raw_config["model"]["lm"]["vocab_size"])
        token_ids = prepare_molmo2_tokenizer(tokenizer, model_vocab_size=model_vocab_size)
        text_vocab_size = min(token_ids.image_token_ids)
        serialized_token_ids = raw_config.get("dataset", {}).get("token_ids")
        if serialized_token_ids is not None:
            expected_token_ids = Molmo2TokenIds.from_dict(serialized_token_ids)
            if token_ids != expected_token_ids:
                raise ValueError(
                    "Tokenizer image-token IDs do not match checkpoint dataset config: "
                    f"{token_ids} != {expected_token_ids}"
                )
        expected_patch_id = int(raw_config["model"]["image_patch_token_id"])
        if token_ids.im_patch_id != expected_patch_id:
            raise ValueError(
                f"Tokenizer image patch ID {token_ids.im_patch_id} does not match checkpoint "
                f"model.image_patch_token_id {expected_patch_id}"
            )

        adapter_type = _build_adapter_class()
        adapter = adapter_type(
            train_module,
            tokenizer,
            token_ids,
            max_sequence_length=args.max_sequence_length,
            max_crops_total=args.max_crops_total,
            max_new_tokens=args.max_new_tokens,
            sequence_bucket_size=args.sequence_bucket_size,
            response_mode=args.response_mode,
            prompt_layout=args.prompt_layout,
            text_vocab_size=text_vocab_size,
        )

        # The OLMo module is sharded with EP, not replicated data parallelism. lmms-eval's
        # normal torchrun behavior would give every EP rank a different request and deadlock
        # the MoE collectives, so all ranks intentionally build and execute the same requests.
        with _lmms_single_logical_replica():
            # Scope discovery to MMMU-Pro. The pinned upstream wheel includes task YAMLs
            # but omits extensionless templates for many unrelated task families, so a
            # repository-wide TaskManager scan would fail before reaching this benchmark.
            task_manager = TaskManager(
                verbosity="INFO",
                include_path=str(_lmms_mmmu_dir()),
                include_defaults=False,
                model_name="default",
            )
            results = lmms_evaluator.simple_evaluate(
                model=adapter,
                tasks=args.tasks,
                task_manager=task_manager,
                limit=args.limit,
                bootstrap_iters=0,
                log_samples=True,
                distributed_executor_backend="torchrun",
                random_seed=0,
                numpy_random_seed=1234,
                torch_random_seed=1234,
                fewshot_random_seed=1234,
            )

        if results is None:
            raise RuntimeError("lmms-eval returned no results")
        partial = args.limit is not None or args.max_new_tokens is not None
        output = Path(args.output) if args.output else _default_output(checkpoint, partial)
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint),
            "checkpoint_state_dir": str(state_dir),
            "config": str(config_path),
            "git": _git_revision(),
            "protocol": {
                "harness": "lmms-eval",
                "tasks": args.tasks,
                "partial": partial,
                "limit": args.limit,
                "world_size": world_size,
                "ep_degree": args.ep_degree,
                "expert_parallel_path": ExpertParallelPath.sync_1d.value,
                "logical_eval_replicas": 1,
                "max_sequence_length": args.max_sequence_length,
                "max_crops_total": args.max_crops_total,
                "max_new_tokens_override": args.max_new_tokens,
                "sequence_bucket_size": args.sequence_bucket_size,
                "attention_backend": "flex",
                "prompt_layout": args.prompt_layout,
                "response_separator": (
                    "none_after_assistant_header"
                    if args.prompt_layout == "bare_chat"
                    else "single_leading_space"
                ),
                "response_mode": args.response_mode,
                "text_vocab_size": text_vocab_size,
                "generation": (
                    "single_forward_option_letter_logits"
                    if args.response_mode == "letter_logits"
                    else (
                        "full_option_text_log_likelihood"
                        if args.response_mode.startswith("option_text_")
                        else "greedy_full_sequence_no_kv_cache"
                    )
                ),
            },
            "lmms_eval": results,
        }
        if get_rank() == 0:
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
            log.info("Wrote results to %s", output)
        dist.barrier()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
