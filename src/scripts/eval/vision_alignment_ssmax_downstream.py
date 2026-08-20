"""Evaluate dense Vision-Alignment SSMax checkpoints on the fast image-MC suite.

The exact suite is BLINK Jigsaw plus the MathVista testmini geometry/problem-solving
multiple-choice slice.  A checkpoint is loaded directly from OLMo-core DCP on one GPU; no
Hugging Face model conversion is involved.  Each request is scored only over its valid answer
letters, avoiding both open-ended judge noise and probability mass on nonexistent choices.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from olmo_core.config import DType
from olmo_core.data.multimodal.vision_alignment_sources import (
    VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
    VISION_ALIGNMENT_TOKENIZER_ID,
    VISION_ALIGNMENT_TOKENIZER_REVISION,
    load_pinned_vision_alignment_tokenizer,
)
from olmo_core.eval.vision_alignment_ssmax_downstream import (
    BLINK_DATASET_REVISION,
    BLINK_JIGSAW_EXAMPLES,
    LMMS_EVAL_REVISION,
    MATHVISTA_DATASET_REVISION,
    MATHVISTA_GEOMETRY_MC_EXAMPLES,
    SSMAX_DOWNSTREAM_TASKS,
    SSMAX_VARIANTS,
    task_definition_inventory,
    validate_ssmax_model_config,
    verify_checkpoint_identity,
)
from olmo_core.nn.vision import MultimodalLMConfig, prepare_molmo2_tokenizer
from olmo_core.optim import SkipStepAdamWConfig
from olmo_core.train import (
    Checkpointer,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.train_module import MultimodalTransformerTrainModuleConfig

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = (
    "/weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/"
    "vision-alignment/evals/downstream-fast-v1"
)
DEFAULT_HF_CACHE = "/weka/oe-training-default/rustin/hf-cache/hub"
DEFAULT_WORK_DIR = "/weka/oe-training-default/rustin/.cache/ssmax-downstream-dcp"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-model-variant", required=True, choices=SSMAX_VARIANTS)
    parser.add_argument(
        "--expected-phase", required=True, choices=("bridge", "perception", "joint")
    )
    parser.add_argument("--expected-global-step", required=True, type=int)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-marker-sha256", required=True)
    parser.add_argument("--expected-dcp-metadata-sha256", required=True)
    parser.add_argument("--expected-checkpoint-identity-sha256", required=True)
    parser.add_argument("--max-sequence-length", type=int, default=8192)
    parser.add_argument("--max-crops-total", type=int, default=8)
    parser.add_argument("--sequence-bucket-size", type=int, default=128)
    parser.add_argument("--checkpoint-load-threads", type=int, default=8)
    parser.add_argument("--hf-cache", default=DEFAULT_HF_CACHE)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--output")
    parser.add_argument("--limit", type=float, help="Per-task smoke limit; makes output partial.")
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _task_root() -> Path:
    return _repo_root() / "requirements/lmms-eval-overrides/vision_ssmax_downstream"


def _default_output(checkpoint: Path, variant: str, *, partial: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "partial" if partial else "complete"
    return (
        Path(DEFAULT_OUTPUT_ROOT)
        / variant
        / checkpoint.name
        / f"blink-jigsaw-mathvista-geometry-mc-{suffix}-{stamp}.json"
    )


def _git_revision() -> dict[str, Any]:
    import subprocess

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


def _verify_lmms_eval_revision() -> dict[str, Any]:
    distribution = importlib.metadata.distribution("lmms-eval")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError("lmms-eval installation has no direct_url.json revision receipt")
    direct_url = json.loads(direct_url_text)
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if commit != LMMS_EVAL_REVISION:
        raise RuntimeError(
            f"lmms-eval revision differs: expected {LMMS_EVAL_REVISION}, got {commit!r}"
        )
    return {
        "distribution_version": distribution.version,
        "revision": commit,
        "url": direct_url.get("url"),
    }


class _AutocastModelFacade:
    """Expose the two model operations used by the existing native lmms adapter."""

    def __init__(self, train_module) -> None:
        self._train_module = train_module
        self._model = train_module.multimodal_model

    def eval(self) -> _AutocastModelFacade:
        self._model.eval()
        return self

    def encode_images(self, images: torch.Tensor, pooled_patches_idx: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(), self._train_module._model_forward_context():
            return self._model.encode_images(images, pooled_patches_idx)


class _InferenceFacade:
    """Adapt a generic dense multimodal train module to the native lmms adapter protocol."""

    def __init__(self, train_module) -> None:
        self._train_module = train_module
        self.device = train_module.device
        self.model_parts = [_AutocastModelFacade(train_module)]

    def model_forward_no_pipeline(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        with torch.inference_mode(), self._train_module._model_forward_context():
            return self._train_module.multimodal_model(input_ids, labels=labels, **kwargs)


def _choices(document: Mapping[str, Any]) -> Sequence[Any]:
    choices = document.get("choices")
    if not isinstance(choices, list) or not 2 <= len(choices) <= 10:
        raise ValueError("Every fast-suite document must contain 2-10 valid choices")
    return choices


def _build_adapter_class():
    # Reuse the already-tested image preprocessing, document prompt, and no-KV-cache forward
    # implementation.  This subclass changes only the task loop so candidate logits are
    # restricted to the number of choices in each BLINK/MathVista row.
    from s002_mmmu_pro import _build_adapter_class as _build_native_adapter_class

    base = _build_native_adapter_class()

    class _SSMaxAdapter(base):
        def generate_until(self, requests):
            from tqdm import tqdm

            responses = []
            for request in tqdm(requests, desc="SSMax image-MC"):
                context, generation_kwargs, doc_to_visual, doc_id, task, split = request.args
                document = self.task_dict[task][split][doc_id]
                visuals = doc_to_visual(document)
                if visuals is None:
                    visuals = []
                elif not isinstance(visuals, list):
                    visuals = [visuals]

                valid_choices = _choices(document)
                all_option_ids = self.option_token_ids
                self.option_token_ids = all_option_ids[: len(valid_choices)]
                try:
                    response = self._generate_one(context, generation_kwargs, visuals)
                finally:
                    self.option_token_ids = all_option_ids
                responses.append(response)
                self.cache_hook.add_partial(
                    "generate_until", (context, generation_kwargs), response.text
                )
            return responses

    return _SSMaxAdapter


def _build_train_module(raw_config: Mapping[str, Any], *, max_sequence_length: int):
    model_dict = raw_config.get("model")
    if not isinstance(model_dict, dict):
        raise TypeError("checkpoint config lacks model")
    model_config = MultimodalLMConfig.from_dict(model_dict)
    validate_ssmax_model_config(
        model_config,
        expected_model_variant=str(raw_config["model_variant"]),
    )
    model_config.lm.recompute_each_block = False
    model_config.lm.recompute_all_blocks_by_chunk = False
    model = model_config.build(init_device="meta")
    module_config = MultimodalTransformerTrainModuleConfig(
        rank_microbatch_size=max_sequence_length,
        max_sequence_length=max_sequence_length,
        optim=SkipStepAdamWConfig(),
        compile_model=False,
        vision_activation_checkpointing=False,
        connector_activation_checkpointing=False,
        autocast_precision=DType.bfloat16,
        new_component_init_seed=int(raw_config.get("init_seed", 6198)),
    )
    return model_config, module_config.build(model, eval_only=True)


def _load_tokenizer(raw_config: Mapping[str, Any], *, cache_dir: str):
    artifacts = raw_config.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("checkpoint config lacks artifacts")
    expected = {
        "tokenizer_id": VISION_ALIGNMENT_TOKENIZER_ID,
        "tokenizer_revision": VISION_ALIGNMENT_TOKENIZER_REVISION,
        "tokenizer_fingerprint": VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
    }
    for key, value in expected.items():
        if artifacts.get(key) != value:
            raise ValueError(
                f"checkpoint {key} differs: expected {value!r}, got {artifacts.get(key)!r}"
            )
    model_dict = raw_config["model"]
    model_vocab_size = int(model_dict["lm"]["vocab_size"])
    tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
        identifier=VISION_ALIGNMENT_TOKENIZER_ID,
        revision=VISION_ALIGNMENT_TOKENIZER_REVISION,
        expected_fingerprint=VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
        cache_dir=cache_dir,
        model_vocab_size=model_vocab_size,
    )
    # ``load_pinned_vision_alignment_tokenizer`` already prepares these tokens.  Calling the
    # public helper a second time must be idempotent; this catches an incompatible tokenizer
    # implementation before model execution.
    if prepare_molmo2_tokenizer(tokenizer, model_vocab_size=model_vocab_size) != token_ids:
        raise RuntimeError("prepared Molmo2 token IDs are not idempotent")
    if token_ids.im_patch_id != int(model_dict["image_patch_token_id"]):
        raise ValueError("tokenizer image-patch ID differs from the checkpoint model config")
    return tokenizer, token_ids


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_sequence_length <= 0:
        raise ValueError("--max-sequence-length must be positive")
    if args.max_crops_total <= 0:
        raise ValueError("--max-crops-total must be positive")
    if args.sequence_bucket_size <= 0:
        raise ValueError("--sequence-bucket-size must be positive")
    if args.checkpoint_load_threads <= 0:
        raise ValueError("--checkpoint-load-threads must be positive")
    if args.expected_global_step < 0:
        raise ValueError("--expected-global-step must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("the dense fast-suite runner is deliberately single-process")
    if not torch.cuda.is_available():
        raise RuntimeError("SSMax downstream evaluation requires a CUDA GPU")


def _validate_result_coverage(results: Mapping[str, Any], *, partial: bool) -> None:
    samples = results.get("samples")
    if not isinstance(samples, Mapping):
        raise TypeError("lmms-eval did not return per-sample results")
    expected = {
        "ssmax_blink_jigsaw": BLINK_JIGSAW_EXAMPLES,
        "ssmax_mathvista_geometry_mc": MATHVISTA_GEOMETRY_MC_EXAMPLES,
    }
    for task, full_count in expected.items():
        rows = samples.get(task)
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"lmms-eval returned no samples for {task}")
        if not partial and len(rows) != full_count:
            raise RuntimeError(
                f"complete {task} coverage differs: expected {full_count}, got {len(rows)}"
            )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    identity, raw_config = verify_checkpoint_identity(
        args.checkpoint,
        expected_model_variant=args.expected_model_variant,
        expected_phase=args.expected_phase,
        expected_global_step=args.expected_global_step,
        expected_config_sha256=args.expected_config_sha256,
        expected_marker_sha256=args.expected_marker_sha256,
        expected_dcp_metadata_sha256=args.expected_dcp_metadata_sha256,
        expected_checkpoint_identity_sha256=args.expected_checkpoint_identity_sha256,
        hash_workers=args.checkpoint_load_threads,
    )
    task_inventory = task_definition_inventory(_task_root())
    lmms_install = _verify_lmms_eval_revision()

    # A single B300 easily holds this dense 1.4B model.  Avoiding torchrun keeps the model
    # replicated nowhere, eliminates collective ordering constraints during variable-length
    # multi-image requests, and lets DCP restore directly into one strict full state dict.
    prepare_training_environment(seed=6198, backend=None)
    try:
        from lmms_eval import evaluator as lmms_evaluator
        from lmms_eval.tasks import TaskManager
        from lmms_eval.utils import handle_non_serializable

        model_config, train_module = _build_train_module(
            raw_config, max_sequence_length=args.max_sequence_length
        )
        if train_module.state_dict_load_opts.strict is not True:
            raise RuntimeError("downstream evaluation requires a strict model-state load")
        checkpointer = Checkpointer(
            work_dir=Path(args.work_dir),
            load_thread_count=args.checkpoint_load_threads,
        )
        checkpointer.load(
            identity.checkpoint,
            train_module,
            load_trainer_state=False,
            load_optim_state=False,
        )
        train_module.multimodal_model.eval()

        tokenizer, token_ids = _load_tokenizer(raw_config, cache_dir=args.hf_cache)
        adapter_type = _build_adapter_class()
        adapter = adapter_type(
            _InferenceFacade(train_module),
            tokenizer,
            token_ids,
            max_sequence_length=args.max_sequence_length,
            max_crops_total=args.max_crops_total,
            max_crops_per_image=None,
            max_new_tokens=None,
            sequence_bucket_size=args.sequence_bucket_size,
            response_mode="letter_logits",
            prompt_layout="document",
            text_vocab_size=(
                model_config.output_vocab_size
                if model_config.output_vocab_size is not None
                else min(token_ids.image_token_ids)
            ),
        )
        task_manager = TaskManager(
            verbosity="INFO",
            include_path=str(_task_root()),
            include_defaults=False,
            model_name="default",
        )
        results = lmms_evaluator.simple_evaluate(
            model=adapter,
            tasks=list(SSMAX_DOWNSTREAM_TASKS),
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
            raise RuntimeError("lmms-eval returned no result")
        partial = args.limit is not None
        _validate_result_coverage(results, partial=partial)
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_identity": identity.as_dict(),
            "git": _git_revision(),
            "protocol": {
                "name": "vision_alignment_ssmax_downstream_fast_v1",
                "tasks": list(SSMAX_DOWNSTREAM_TASKS),
                "partial": partial,
                "limit": args.limit,
                "lmms_eval_revision": LMMS_EVAL_REVISION,
                "lmms_eval_install": lmms_install,
                "blink_dataset_revision": BLINK_DATASET_REVISION,
                "mathvista_dataset_revision": MATHVISTA_DATASET_REVISION,
                "dataset_auth": None,
                "task_definition_sha256": task_inventory["sha256"],
                "task_definitions": task_inventory["files"],
                "response_mode": "valid_choice_letter_logits",
                "prompt_layout": "document",
                "max_sequence_length": args.max_sequence_length,
                "max_crops_total": args.max_crops_total,
                "max_crops_per_image": None,
                "crop_budget_mode": "shared_total",
                "sequence_bucket_size": args.sequence_bucket_size,
                "world_size": 1,
                "checkpoint_format": "native_olmo_core_dcp",
                "checkpoint_conversion": None,
                "checkpoint_identity_semantics": (
                    "vision_alignment_ssmax_bridge.checkpoint_identity"
                ),
                "checkpoint_load": {
                    "strict_model_state": True,
                    "load_optimizer_state": False,
                    "load_trainer_state": False,
                    "state_file_count": identity.state_file_count,
                    "state_file_inventory_sha256": identity.state_file_inventory_sha256,
                    "trainer_state_count": identity.trainer_state_count,
                    "trainer_state_inventory_sha256": identity.trainer_state_inventory_sha256,
                },
                "mathvista_scoring": "local_valid_letter_choice_string_equal",
                "external_judge": None,
                "generation": "single_forward_valid_option_letter_logits",
            },
            "lmms_eval": results,
        }
        checkpoint = Path(identity.checkpoint)
        output = (
            Path(args.output)
            if args.output is not None
            else _default_output(checkpoint, args.expected_model_variant, partial=partial)
        )
        if output.exists():
            raise FileExistsError(f"refusing to overwrite downstream result {output}")
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
        log.info("Wrote complete SSMax downstream result to %s", output)
    finally:
        teardown_training_environment()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
