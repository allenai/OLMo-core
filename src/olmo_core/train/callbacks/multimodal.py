"""Multimodal model initialization and held-out response-loss evaluation."""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import torch.distributed as dist
from torch.utils.data import Subset

from olmo_core.data.multimodal.alignment import MultimodalMixtureConfig
from olmo_core.data.multimodal.collator import MultimodalCollator
from olmo_core.data.multimodal.data_loader import MultimodalDataLoader
from olmo_core.distributed.utils import broadcast_object, get_rank, get_world_size
from olmo_core.eval import Evaluator
from olmo_core.eval.multimodal_image_pairing import (
    MultimodalImagePairDataset,
    build_bounded_image_pairs,
)
from olmo_core.eval.multimodal_lm_evaluator import (
    MultimodalBlankImageEvaluator,
    MultimodalLMEvaluator,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.train.common import Duration

from .callback import Callback, CallbackConfig
from .evaluator_callback import EvaluatorCallback

if TYPE_CHECKING:
    from olmo_core.train import Trainer

log = logging.getLogger(__name__)


@dataclass
class InitializeMultimodalModelCallback(Callback):
    """Initialize a fresh alignment run after the trainer has attempted a resume.

    A restored checkpoint, including a step-zero checkpoint, always takes precedence.
    Language weights are loaded without optimizer state; the vision loader and image-token
    initializer synchronize their weights with any optimizer master parameters.

    This callback currently supports :class:`~olmo_core.train.train_module.transformer.multimodal_train_module.MultimodalOLMoDDPTrainModule`.
    """

    priority: ClassVar[int] = 4
    """Initialize weights before evaluators and pre-training checkpoint saves."""

    language_checkpoint: str
    """Language-model checkpoint directory, or its ``model_and_optim`` subdirectory."""

    vision_model_id: str
    """Hugging Face repository containing a SigLIP vision encoder."""

    image_token_ids: list[int]
    """Input-embedding rows to initialize for image tokens."""

    vision_revision: str | None = None
    cache_dir: str | None = None
    seed: int = 12536
    load_threads: int | None = None

    def pre_train(self):
        """Load pretrained components only when no full checkpoint was restored."""
        if self.trainer.checkpoint_loaded:
            return

        from olmo_core.nn.vision import load_siglip_hf_vision_state_dict
        from olmo_core.train.train_module.transformer.multimodal_train_module import (
            MultimodalOLMoDDPTrainModule,
        )

        train_module = self.trainer.train_module
        if not isinstance(train_module, MultimodalOLMoDDPTrainModule):
            raise OLMoConfigurationError(
                "Separate language/vision initialization currently requires "
                "MultimodalOLMoDDPTrainModule"
            )

        state_dir = self.language_checkpoint.rstrip("/")
        if state_dir.rsplit("/", 1)[-1] != "model_and_optim":
            state_dir += "/model_and_optim"
        log.info("Loading pretrained language weights from %s", state_dir)
        train_module.load_state_dict_direct(
            state_dir,
            process_group=dist.group.WORLD,
            thread_count=self.load_threads,
            load_optim_state=False,
        )

        log.info("Loading pretrained vision weights from %s", self.vision_model_id)
        vision_state = load_siglip_hf_vision_state_dict(
            self.vision_model_id, revision=self.vision_revision, cache_dir=self.cache_dir
        )
        train_module.load_siglip_vision_state_dict(vision_state)
        train_module.assert_vision_optimizer_state_synced()
        if self.image_token_ids:
            train_module.reset_image_token_rows(
                self.image_token_ids, seed=self.seed, reset_output_rows=False
            )


@dataclass
class MultimodalEvaluatorCallbackConfig(CallbackConfig):
    """Build deterministic response-loss evaluators from explicit held-out sources.

    Each source evaluates the first ``examples_per_source`` rows in its configured order.
    Provide a prepared selection when a subset or stratified holdout is required. Mixture
    loss weights and calibration means are not used for evaluation.
    """

    eval_dataset: MultimodalMixtureConfig
    """Held-out datasets; these must be disjoint from the training sources."""

    sequence_length: int
    rank_batch_size: int = 1
    """Number of evaluation sequences per data-parallel rank."""

    examples_per_source: int = 512
    eval_interval: int | None = 500
    blank_image_sources: list[str] = field(default_factory=list)
    matched_image_sources: list[str] = field(default_factory=list)
    """Sources with paired correct/wrong-image diagnostics, separate from the full validation."""

    matched_image_examples: int = 64
    matched_image_candidates: int = 512
    """Maximum held-out rows to preprocess once when selecting exact-geometry image pairs."""

    early_response_tokens: int | None = 8
    """Also compare the first N supervised positions per paired sequence; None disables this."""

    eval_on_startup: bool = True
    eval_on_finish: bool = True
    cancel_after_first_eval: bool = False
    """Cancel after the first evaluation; combine with startup evaluation for an eval-only run."""

    seed: int = 0

    def build(self, trainer: "Trainer") -> EvaluatorCallback:
        """Build source evaluators and bounded paired image-content diagnostics."""
        if self.sequence_length <= 0 or self.rank_batch_size <= 0:
            raise OLMoConfigurationError("Evaluation sequence and batch sizes must be positive")
        if self.examples_per_source <= 0:
            raise OLMoConfigurationError("examples_per_source must be positive")
        if self.eval_interval is not None and self.eval_interval <= 0:
            raise OLMoConfigurationError("eval_interval must be positive or None")
        unknown_sources = set(self.blank_image_sources) - set(self.eval_dataset.sources)
        if unknown_sources:
            raise OLMoConfigurationError(
                f"Blank-image controls refer to unknown sources: {sorted(unknown_sources)}"
            )
        unknown_sources = set(self.matched_image_sources) - set(self.eval_dataset.sources)
        if unknown_sources:
            raise OLMoConfigurationError(
                f"Matched-image controls refer to unknown sources: {sorted(unknown_sources)}"
            )
        if self.early_response_tokens is not None and self.early_response_tokens <= 0:
            raise OLMoConfigurationError("early_response_tokens must be positive or None")
        if self.matched_image_sources and (
            self.matched_image_examples <= 0
            or self.matched_image_candidates < self.matched_image_examples
            or self.seed < 0
        ):
            raise OLMoConfigurationError("Invalid matched-image example/candidate count or seed")
        dp_world_size = get_world_size(trainer.dp_process_group)
        dp_rank = get_rank(trainer.dp_process_group)
        global_instances = self.rank_batch_size * dp_world_size
        if self.examples_per_source % global_instances:
            raise OLMoConfigurationError(
                "examples_per_source must be divisible by the global evaluation batch size"
            )
        if self.matched_image_sources and self.matched_image_examples % global_instances:
            raise OLMoConfigurationError(
                "matched_image_examples must be divisible by the global evaluation batch size"
            )

        tokenizer, token_ids = self.eval_dataset.build_tokenizer()
        sources = self.eval_dataset.build_sources(tokenizer, token_ids)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            raise OLMoConfigurationError("The tokenizer must define a pad or EOS token")
        collator = MultimodalCollator(
            pad_token_id=pad_token_id, pad_sequence_length=self.sequence_length
        )

        def build_loader(dataset, name):
            return MultimodalDataLoader(
                dataset,
                collator,
                work_dir=trainer.work_dir / name,
                global_batch_size=global_instances * self.sequence_length,
                seed=self.seed,
                shuffle=False,
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
            )

        evaluators: list[Evaluator] = []
        for name, dataset in sources.items():
            if len(dataset) < self.examples_per_source:
                raise OLMoConfigurationError(
                    f"Evaluation source {name!r} has {len(dataset)} examples, "
                    f"but {self.examples_per_source} were requested"
                )
            loader = build_loader(
                Subset(dataset, range(self.examples_per_source)), f"{name}_validation"
            )
            evaluators.append(
                MultimodalLMEvaluator(
                    name=f"{name}-validation",
                    batches=loader,
                    device=trainer.device,
                    process_group=trainer.dp_process_group,
                    deterministic=True,
                )
            )
            if name in self.blank_image_sources:
                evaluators.append(
                    MultimodalBlankImageEvaluator(
                        name=f"{name}-blank-image",
                        batches=loader,
                        device=trainer.device,
                        process_group=trainer.dp_process_group,
                        deterministic=True,
                    )
                )
            if name in self.matched_image_sources:
                # Prepare only on the DP leader, then share small index pairs (or the error).
                # This avoids preprocessing the same candidate images on every GPU rank.
                result: dict[str, Any] | None = None
                if dp_rank == 0 or not dist.is_initialized():
                    try:
                        result = {
                            "pairs": build_bounded_image_pairs(
                                dataset,
                                examples=self.matched_image_examples,
                                max_candidates=self.matched_image_candidates,
                                seed=self.seed,
                            )
                        }
                    except Exception as error:  # noqa: BLE001 - propagate failures to every rank
                        result = {"error": f"{type(error).__name__}: {error}"}
                group = trainer.dp_process_group
                leader = dist.get_global_rank(group, 0) if group is not None else 0
                result = broadcast_object(result, src=leader, group=group)
                assert result is not None
                if "error" in result:
                    raise OLMoConfigurationError(f"Image pairing for {name!r}: {result['error']}")
                pairs = result["pairs"]
                log.info(
                    "Matched-image evaluation %s: %d pairs from at most %d held-out candidates",
                    name,
                    len(pairs),
                    min(self.matched_image_candidates, len(dataset)),
                )
                correct_loader = build_loader(
                    MultimodalImagePairDataset(dataset, pairs, wrong_images=False),
                    f"{name}_matched_correct",
                )
                wrong_loader = build_loader(
                    MultimodalImagePairDataset(dataset, pairs, wrong_images=True),
                    f"{name}_matched_wrong",
                )
                prefixes: list[int | None] = [None]
                if self.early_response_tokens is not None:
                    prefixes.append(self.early_response_tokens)
                for prefix in prefixes:
                    suffix = "" if prefix is None else f"-first{prefix}"
                    correct = MultimodalLMEvaluator(
                        name=f"{name}-matched-correct{suffix}",
                        batches=correct_loader,
                        device=trainer.device,
                        process_group=group,
                        response_prefix_tokens=prefix,
                    )
                    wrong = MultimodalLMEvaluator(
                        name=f"{name}-matched-wrong{suffix}",
                        batches=wrong_loader,
                        device=trainer.device,
                        process_group=group,
                        response_prefix_tokens=prefix,
                        reference_evaluator=correct,
                    )
                    evaluators.extend([correct, wrong])
        max_examples = max(
            self.examples_per_source,
            self.matched_image_examples if self.matched_image_sources else 0,
        )
        return EvaluatorCallback(
            evaluators=evaluators,
            eval_interval=self.eval_interval,
            eval_duration=Duration.steps(max_examples // global_instances),
            eval_on_startup=self.eval_on_startup,
            eval_on_finish=self.eval_on_finish,
            cancel_after_first_eval=self.cancel_after_first_eval,
        )
