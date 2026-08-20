"""Train module for :class:`~olmo_core.nn.vision.MultimodalLM` training.

:class:`MultimodalTransformerTrainModule` extends :class:`TransformerTrainModule` for a
:class:`~olmo_core.nn.vision.MultimodalLM` (a plain ``nn.Module``, *not* a
:class:`~olmo_core.nn.transformer.Transformer`). It differs from the base module in
three ways:

1. The model is **not** routed through ``parallelize_model`` (which requires a
   ``Transformer``). DDP (``replicate``) and FSDP/HSDP (``fully_shard`` of the LM,
   vision encoder, and connector) are applied here directly; TP/CP/PP/EP are out of scope.
2. The loss uses **float per-token** ``loss_masks`` (response-only, ``root_subsegments``
   weighted by the data pipeline) via
   :func:`~olmo_core.nn.functional.weighted_cross_entropy_loss`, reproducing mm_olmo.
3. The loss divisor is the **global** sum of ``loss_masks`` (all-reduced, divided by the
   DP world size) so that, after DDP gradient averaging, the effective normalization is
   the global loss-weight — matching mm_olmo's ``BatchDivisor.global_batch``.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
from dataclasses import dataclass, replace
from fnmatch import fnmatch
from functools import lru_cache
from typing import (
    Any,
    Collection,
    Dict,
    Generator,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    cast,
)

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn.functional as F
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.optim import Optimizer

from olmo_core.aliases import PathOrStr
from olmo_core.config import DType
from olmo_core.data.utils import split_batch
from olmo_core.distributed.checkpoint import get_checkpoint_metadata
from olmo_core.distributed.checkpoint import (
    load_state_dict as load_distributed_state_dict,
)
from olmo_core.distributed.checkpoint import (
    merge_state_dicts,
    prune_state_dict,
    swap_param_keys,
)
from olmo_core.distributed.parallel import (
    DataParallelType,
    build_world_mesh,
    get_dp_model_mesh,
)
from olmo_core.distributed.utils import (
    get_local_tensor,
    get_rank,
    get_world_size,
    is_distributed,
    reduce_distributed_failure_flag,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.functional import weighted_cross_entropy_loss
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.optim import OptimConfig, SkipStepOptimizer
from olmo_core.optim.scheduler import Scheduler
from olmo_core.utils import get_default_device, move_to_device, warn_once

from ...common import ReduceType
from ..config import TrainModuleConfig
from ..train_module import EvalBatchSpec, TrainModule
from .config import (
    OLMoDDPTrainModuleConfig,
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
)
from .ddp_train_module import OLMoDDPTrainModule
from .train_module import TransformerTrainModule

log = logging.getLogger(__name__)

__all__ = [
    "MultimodalTransformerTrainModule",
    "MultimodalTransformerTrainModuleConfig",
    "MultimodalOLMoDDPTrainModule",
    "MultimodalOLMoDDPTrainModuleConfig",
]


def _mm_train_verbose_logs() -> bool:
    """Per-step batch/optim diagnostics (forces CUDA sync via ``.item()``)."""
    return os.environ.get("MM_TRAIN_VERBOSE_LOGS", "0").lower() in ("1", "true", "yes")


def _retain_embedding_gradient_rows(grad: torch.Tensor, row_ids: Tuple[int, ...]) -> torch.Tensor:
    """Return an embedding gradient with only ``row_ids`` retained."""
    row_mask = torch.zeros((grad.shape[0], 1), dtype=grad.dtype, device=grad.device)
    row_mask[list(row_ids)] = 1
    if isinstance(grad, DTensor):
        # FSDP/HSDP commonly shard embedding rows while replicating the other mesh
        # dimension. Preserve that row sharding for the small ``vocab_size x 1`` mask;
        # replicate the mask across any mesh dimension that shards embedding columns.
        # Broadcasting then keeps the gradient's original placements.
        mask_placements = tuple(
            placement if isinstance(placement, Shard) and placement.dim == 0 else Replicate()
            for placement in grad.placements
        )
        row_mask = distribute_tensor(
            row_mask,
            device_mesh=grad.device_mesh,
            placements=mask_placements,
        )
    return grad * row_mask


def _matched_component_grad_norm_patterns(
    component_patterns: Mapping[str, Tuple[str, ...]], trainable_names: set[str]
) -> Dict[str, Tuple[str, ...]]:
    """Keep diagnostic components that match at least one trainable optimizer parameter."""
    return {
        component: patterns
        for component, patterns in component_patterns.items()
        if any(fnmatch(name, pattern) for name in trainable_names for pattern in patterns)
    }


_GENERIC_COMPONENT_GRAD_NORM_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "vision": ("vision.*", "*vision.*"),
    "connector": ("connector.*", "*connector.*"),
    "input embeddings": ("lm.embeddings.weight", "*lm.embeddings.weight"),
    "LM output head": ("lm.lm_head.w_out.*", "*lm.lm_head.w_out.*"),
    # TransformerBlock intentionally retains the historical ``attention`` attribute for
    # every sequence mixer, including GatedDeltaNet. Use the architecture-neutral label.
    "LM sequence mixers": ("lm.blocks.*.attention.*", "*lm.blocks.*.attention.*"),
    "LM feed forward": ("lm.blocks.*.feed_forward.*", "*lm.blocks.*.feed_forward.*"),
    "LM normalization": ("lm.*norm*", "*lm.*norm*"),
}


def _validate_source_loss_mass_targets(
    source_loss_mass_targets: Optional[Mapping[str, float]],
) -> Dict[str, float]:
    """Validate and normalize the source-delivery telemetry contract."""
    targets = dict(source_loss_mass_targets or {})
    if any(not isinstance(name, str) or not name for name in targets):
        raise OLMoConfigurationError("source_loss_mass_targets keys must be non-empty strings")
    try:
        invalid_value = any(not math.isfinite(value) or value <= 0 for value in targets.values())
        total = sum(targets.values())
    except TypeError as error:
        raise OLMoConfigurationError(
            "source_loss_mass_targets values must be finite positive numbers"
        ) from error
    if targets and (invalid_value or abs(total - 1.0) > 1e-6):
        raise OLMoConfigurationError("source_loss_mass_targets must be positive and sum to one")
    return targets


class MultimodalTransformerTrainModule(TransformerTrainModule):
    """A :class:`TrainModule` for :class:`~olmo_core.nn.vision.MultimodalLM` stage-1 training."""

    optim: Optional[Optimizer]

    def __init__(
        self,
        model: torch.nn.Module,
        optim: OptimConfig,
        rank_microbatch_size: int,
        max_sequence_length: int,
        *,
        freeze_params: Optional[List[str]] = None,
        z_loss_multiplier: Optional[float] = None,
        autocast_precision: Optional[torch.dtype] = None,
        max_grad_norm: Optional[float] = None,
        scheduler: Optional[Scheduler] = None,
        device: Optional[torch.device] = None,
        compile_model: bool = False,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        ac_config: Optional[TransformerActivationCheckpointingConfig] = None,
        vision_activation_checkpointing: bool = True,
        connector_activation_checkpointing: bool = True,
        label_ignore_index: int = -100,
        response_logits_only: bool = False,
        diagnostics_interval: Optional[int] = None,
        train_embedding_rows: Optional[List[int]] = None,
        source_loss_mass_targets: Optional[Dict[str, float]] = None,
        new_component_init_seed: Optional[int] = None,
        eval_only: bool = False,
        state_dict_save_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        state_dict_load_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        load_key_mapping: Optional[Dict[str, str]] = None,
    ):
        # NOTE: deliberately bypass ``TransformerTrainModule.__init__`` (which calls
        # ``parallelize_model``, requiring a ``Transformer``); call the grandparent.
        TrainModule.__init__(self)

        from olmo_core.nn.vision import MultimodalLM

        if not isinstance(model, MultimodalLM) or getattr(model, "_olmo_ddp_compatible", False):
            raise TypeError(
                f"{type(self).__name__} requires a generic MultimodalLM, "
                f"got {type(model).__name__}"
            )
        if rank_microbatch_size <= 0 or max_sequence_length <= 0:
            raise OLMoConfigurationError(
                "rank_microbatch_size and max_sequence_length must both be positive"
            )
        if rank_microbatch_size % max_sequence_length != 0:
            raise OLMoConfigurationError(
                f"'rank_microbatch_size' ({rank_microbatch_size:,d} tokens) must be divisible by "
                f"'max_sequence_length' ({max_sequence_length:,d} tokens)"
            )
        if dp_config is not None and dp_config.name not in (
            DataParallelType.ddp,
            DataParallelType.fsdp,
            DataParallelType.hsdp,
        ):
            raise OLMoConfigurationError(
                "MultimodalTransformerTrainModule only supports DDP / FSDP / HSDP data "
                f"parallelism (got dp_config.name={dp_config.name!r}); TP/CP/PP/EP of the "
                "multimodal model are not yet supported."
            )
        if diagnostics_interval is not None and diagnostics_interval <= 0:
            raise OLMoConfigurationError("diagnostics_interval must be positive or None")
        if new_component_init_seed is not None and (
            not isinstance(new_component_init_seed, int)
            or isinstance(new_component_init_seed, bool)
            or new_component_init_seed < 0
        ):
            raise OLMoConfigurationError("new_component_init_seed must be a non-negative integer")

        freeze_params = list(freeze_params or [])
        if any(not isinstance(pattern, str) or not pattern for pattern in freeze_params):
            raise OLMoConfigurationError("freeze_params patterns must be non-empty strings")
        named_params = dict(model.named_parameters())
        original_trainability = {name: param.requires_grad for name, param in named_params.items()}
        for pattern in freeze_params:
            if not any(fnmatch(name, pattern) for name in named_params):
                raise OLMoConfigurationError(
                    f"freeze_params pattern {pattern!r} does not match any model parameter"
                )

        raw_embedding_rows = list(train_embedding_rows or [])
        if any(
            not isinstance(row_id, int) or isinstance(row_id, bool) for row_id in raw_embedding_rows
        ):
            raise OLMoConfigurationError("train_embedding_rows must contain integer token IDs")
        if len(raw_embedding_rows) != len(set(raw_embedding_rows)):
            raise OLMoConfigurationError("train_embedding_rows must contain unique IDs")
        self.train_embedding_rows = tuple(sorted(raw_embedding_rows))
        self.diagnostics_interval = diagnostics_interval
        self.source_loss_mass_targets = _validate_source_loss_mass_targets(source_loss_mass_targets)
        self.new_component_init_seed = new_component_init_seed

        self.device = device or get_default_device()
        self.world_mesh = None
        if is_distributed():
            self.world_mesh = build_world_mesh(dp=dp_config, device_type=self.device.type)
        elif dp_config is not None:
            raise OLMoConfigurationError(
                "Training parallelism configs are only valid for distributed training"
            )

        # Freeze before parallelization so FSDP sees the intended trainability topology.
        # Meta materialization can replace Parameter objects, so the exact same operation is
        # repeated after initialization below before the optimizer is built.
        self.freeze_params = freeze_params
        for name, param in model.named_parameters():
            if any(fnmatch(name, pattern) for pattern in self.freeze_params):
                param.requires_grad_(False)

        meta_flags = {param.is_meta for param in model.parameters()}
        if len(meta_flags) > 1:
            raise OLMoConfigurationError(
                "MultimodalLM parameters must be either entirely materialized or entirely meta"
            )
        materialize_from_meta = meta_flags == {True}
        if materialize_from_meta and self.new_component_init_seed is None:
            raise OLMoConfigurationError(
                "new_component_init_seed is required for a meta-initialized MultimodalLM so "
                "new vision/connector initialization is explicit and auditable"
            )
        if not materialize_from_meta:
            model.to(self.device)
        if vision_activation_checkpointing and hasattr(
            model.vision, "apply_activation_checkpointing"
        ):
            model.vision.apply_activation_checkpointing()
            log.info("Applied per-block activation checkpointing to model.vision")
        if connector_activation_checkpointing and hasattr(
            model.connector, "apply_activation_checkpointing"
        ):
            model.connector.apply_activation_checkpointing()
            log.info("Applied activation checkpointing to model.connector")
        if ac_config is not None:
            model.lm.apply_activation_checkpointing(
                ac_config.mode,
                block_interval=ac_config.block_interval,
                modules=ac_config.modules,
                activation_memory_budget=ac_config.activation_memory_budget,
                determinism_check=ac_config.determinism_check,
            )
            log.info("Applied '%s' activation checkpointing to model.lm", ac_config.mode)
        if compile_model:
            log.info("Compiling model.lm blocks ...")
            model.lm.apply_compile()
        self.model = model
        self._model_mode = None
        self._forbid_packed_multimodal_metadata = bool(
            getattr(model, "_has_gated_delta_net", False)
        )

        self._dp_config = dp_config
        self._cp_config = None
        self._tp_config = None
        self._ep_config = None
        self.label_ignore_index = label_ignore_index
        self.response_logits_only = response_logits_only
        self.z_loss_multiplier = z_loss_multiplier
        self.rank_microbatch_size = rank_microbatch_size
        self.max_sequence_length = max_sequence_length
        self.autocast_precision = autocast_precision
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.state_dict_save_opts = state_dict_save_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        self.state_dict_load_opts = state_dict_load_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, strict=True
        )
        self.load_key_mapping = load_key_mapping
        self.eval_only = eval_only

        # Apply data parallelism IN-PLACE *before* building the optimizer: composable
        # DDP/FSDP keep the model's type, attributes, and (prefix-free) parameter names,
        # and FSDP additionally needs the optimizer built on the sharded DTensor params.
        if self.world_mesh is not None:
            assert dp_config is not None
            self._parallelize(dp_config)

        if materialize_from_meta:
            log.info("Materializing meta-initialized multimodal model on %s", self.device)
            model.lm.init_weights(
                max_seq_len=max_sequence_length,
                max_local_microbatch_size=rank_microbatch_size,
                device=self.device,
                world_mesh=self.world_mesh,
            )
            # The generic MultimodalLM intentionally has no monolithic init method because
            # pretrained LM loading and independent vision initialization are common. Keep new
            # component initialization independent of the parent's LM init seed: bridge recipes
            # provide one explicit phase seed so paired arms get bit-identical connector starts.
            assert self.new_component_init_seed is not None
            fork_devices = [self.device] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(self.new_component_init_seed)
                model.vision.to_empty(device=self.device)
                model.vision.reset_parameters()
                torch.manual_seed(self.new_component_init_seed + 1)
                model.connector.to_empty(device=self.device)
                model.connector.reset_parameters()

        frozen_names = []
        for name, param in model.named_parameters():
            should_freeze = any(fnmatch(name, pattern) for pattern in self.freeze_params)
            param.requires_grad_(original_trainability[name] and not should_freeze)
            if should_freeze:
                frozen_names.append(name)
        if self.freeze_params:
            log.info(
                "Froze %d parameter tensors matching %s",
                len(frozen_names),
                self.freeze_params,
            )
        if not any(param.requires_grad for param in model.parameters()):
            raise OLMoConfigurationError("freeze_params leave no trainable model parameters")

        self._embedding_grad_hook = None
        if self.train_embedding_rows:
            embeddings = model.lm.embeddings
            if embeddings is None:
                raise OLMoConfigurationError("train_embedding_rows requires LM embeddings")
            if self.train_embedding_rows[0] < 0 or self.train_embedding_rows[-1] >= int(
                embeddings.weight.shape[0]
            ):
                raise OLMoConfigurationError(
                    "train_embedding_rows contains an ID outside the LM embedding table"
                )
            if not embeddings.weight.requires_grad:
                raise OLMoConfigurationError(
                    "The LM embedding parameter must remain trainable when row masking is enabled"
                )
            if model.lm.lm_head is not None and model.lm.lm_head.w_out.weight is embeddings.weight:
                raise OLMoConfigurationError(
                    "Row-masked image embeddings require untied LM input and output weights"
                )
            self._embedding_grad_hook = embeddings.weight.register_hook(
                lambda grad: _retain_embedding_gradient_rows(grad, self.train_embedding_rows)
            )
            log.info(
                "Restricted LM input-embedding gradients to rows %s",
                self.train_embedding_rows,
            )

        self.optim = None
        if not self.eval_only:
            log.info("Building optimizer...")
            self.optim = optim.build(self.model, strict=True)
        else:
            log.info("Skipping optimizer build because eval_only=True")

    def _parallelize(self, dp_config: TransformerDataParallelConfig) -> None:
        """Apply DDP (``replicate``) or FSDP (``fully_shard``) to the multimodal model
        in-place. FSDP shards the LM (the bulk of the parameters), the vision encoder,
        and the connector across the DP mesh so the 4B model + optimizer fit per GPU."""
        assert self.world_mesh is not None
        dp_mesh = get_dp_model_mesh(self.world_mesh)
        if dp_config.name == DataParallelType.ddp:
            from torch.distributed._composable.replicate import replicate

            replicate(self.model, device_mesh=dp_mesh, bucket_cap_mb=100)
        else:  # fsdp / hsdp
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

            param_dtype = (
                dp_config.param_dtype.as_pt() if dp_config.param_dtype is not None else None
            )
            reduce_dtype = dp_config.reduce_dtype.as_pt()
            # Shard the language model with its own (per-block) FSDP wrapping.
            self.model.lm.apply_fsdp(
                dp_mesh=dp_mesh,
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                wrapping_strategy=dp_config.wrapping_strategy,
                prefetch_factor=dp_config.prefetch_factor,
            )
            # Shard the vision encoder + connector, then the root so ``self.model`` is an
            # FSDPModule (the inherited micro-batch / gradient-sync handling keys off this).
            mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
            fully_shard(self.model.vision, mesh=dp_mesh, mp_policy=mp)
            fully_shard(self.model.connector, mesh=dp_mesh, mp_policy=mp)
            fully_shard(self.model, mesh=dp_mesh, mp_policy=mp)

    # -- helpers to reach the underlying MultimodalLM / its Transformer ----------

    @property
    def _multimodal(self) -> torch.nn.Module:
        # ``replicate`` is applied in-place, so ``self.model`` is the MultimodalLM itself.
        return self.model

    @property
    def multimodal_model(self) -> torch.nn.Module:
        """Return the underlying :class:`~olmo_core.nn.vision.MultimodalLM`.

        This mirrors :attr:`MultimodalOLMoDDPTrainModule.multimodal_model` so recipe code can
        remain independent of the optimizer/parallelism implementation.
        """
        return self._multimodal

    @property
    def _lm(self) -> torch.nn.Module:
        return self._multimodal.lm

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        return EvalBatchSpec(
            self.rank_microbatch_size, max_sequence_length=self.max_sequence_length
        )

    # -- batch preparation -------------------------------------------------------

    def _prepare_batch(  # type: ignore[override]
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Dict[str, Any]]:
        """Split off ``input_ids`` / ``labels`` / float ``loss_masks``; the rest
        (``images``, ``pooled_patches_idx``, ``token_type_ids``, ``subsegment_ids``,
        ``position_ids``) flows to :meth:`MultimodalLM.forward` as kwargs."""
        input_ids = batch.pop("input_ids")
        labels = labels if labels is not None else batch.pop("labels", None)
        loss_masks = batch.pop("loss_masks")
        batch.pop("pack_source_names", None)
        batch.pop("source_names", None)
        batch.pop("router_token_mask", None)
        batch.pop("image_crop_counts", None)
        batch.pop("pooled_token_counts", None)
        if self._forbid_packed_multimodal_metadata:
            forbidden = {"example_ids", "subsegment_ids"} & batch.keys()
            if forbidden:
                raise OLMoConfigurationError(
                    "GatedDeltaNet multimodal training forbids packed/branched metadata; "
                    f"unexpected fields: {sorted(forbidden)}"
                )
        return input_ids, labels, loss_masks, batch

    def _iter_maximal_fully_frozen_modules(
        self,
    ) -> Generator[Tuple[str, torch.nn.Module], None, None]:
        """Yield maximal submodules whose complete parameter subtree is frozen."""

        def visit(prefix: str, module: torch.nn.Module):
            params = tuple(module.parameters())
            if params and not any(param.requires_grad for param in params):
                yield prefix, module
                return
            for child_name, child in module.named_children():
                child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                yield from visit(child_prefix, child)

        for name, child in self._multimodal.named_children():
            yield from visit(name, child)

    def _set_model_mode(self, mode: Literal["train", "eval"]):
        super()._set_model_mode(mode)
        if mode == "train":
            # ``model.train()`` recursively enables dropout in every child. Reassert eval mode
            # for completely frozen components and blocks so the forward is invariant across
            # repeated bridge steps. Partially trainable parents remain in train mode.
            for _, module in self._iter_maximal_fully_frozen_modules():
                module.eval()

    def _log_batch_sources(self, batch: Dict[str, Any], local_weight: torch.Tensor) -> None:
        """Log per-rank packed source names when verbose diagnostics are enabled."""
        if not _mm_train_verbose_logs():
            return
        sources = batch.get("pack_source_names", batch.get("source_names"))
        if sources is None:
            return
        images = batch.get("images")
        n_crops = int(images.shape[1]) if images is not None else 0
        n_im_patch = int(
            (batch["input_ids"] == self._multimodal.cfg.image_patch_token_id).sum().item()
        )
        log.info(
            "batch sources rank=%d sources=%s local_weight=%.1f im_patch=%d crops=%d shape=%s",
            get_rank(),
            sources,
            float(local_weight.item()),
            n_im_patch,
            n_crops,
            tuple(batch["input_ids"].shape),
        )

    def _diagnostics_enabled_for_step(self) -> bool:
        return bool(
            self.diagnostics_interval is not None
            and self._trainer is not None
            and self.trainer.global_step % self.diagnostics_interval == 0
        )

    def _batch_token_mask(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Return and validate the collator's real-token mask for diagnostics."""
        token_mask = batch.get("router_token_mask")
        if token_mask is None:
            # Generic dense models do not use this field for routing, but the multimodal
            # collator emits it as the authoritative non-padding mask. An example-id sentinel
            # is an equivalent fallback for older packed batches.
            example_ids = batch.get("example_ids")
            if example_ids is not None:
                token_mask = example_ids >= 0
            else:
                raise OLMoConfigurationError(
                    "Multimodal diagnostics require router_token_mask (or example_ids with "
                    "-1 padding) so padding cannot be reported as delivered data"
                )
        if token_mask.shape != batch["input_ids"].shape:
            raise OLMoConfigurationError(
                "The diagnostic token mask must match input_ids: "
                f"got {tuple(token_mask.shape)} and {tuple(batch['input_ids'].shape)}"
            )
        return token_mask.to(torch.bool)

    def _record_data_metrics(self, batch: Dict[str, Any]) -> None:
        """Record supervision, packing, crop, and source-delivery metrics."""
        token_mask = self._batch_token_mask(batch)
        if self.source_loss_mass_targets and (
            self._trainer is None or self._diagnostics_enabled_for_step()
        ):
            self._record_source_data_metrics(batch, token_mask)

        self.record_metric(
            "packing fill",
            token_mask.float().mean(),
            ReduceType.mean,
            namespace="data",
        )
        if (loss_masks := batch.get("loss_masks")) is not None:
            self.record_metric(
                "response token density",
                ((loss_masks > 0) & token_mask).float().mean(),
                ReduceType.mean,
                namespace="data",
            )
        if (token_type_ids := batch.get("token_type_ids")) is not None:
            self.record_metric(
                "image token density",
                ((token_type_ids != 0) & token_mask).float().mean(),
                ReduceType.mean,
                namespace="data",
            )
        if (example_ids := batch.get("example_ids")) is not None:
            # Count distinct non-negative IDs rather than assuming dense IDs; this remains
            # correct for explicitly serialized, un-packed bridge examples.
            counts = torch.tensor(
                [torch.unique(row[row >= 0]).numel() for row in example_ids],
                device=example_ids.device,
                dtype=torch.float32,
            )
            self.record_metric(
                "examples per sequence",
                counts.mean(),
                ReduceType.mean,
                namespace="data",
            )
        else:
            self.record_metric(
                "examples per sequence",
                torch.ones((), device=token_mask.device),
                ReduceType.mean,
                namespace="data",
            )

        if (crop_counts := batch.get("image_crop_counts")) is not None:
            self.record_metric(
                "real crops per sequence",
                crop_counts.float().mean(),
                ReduceType.mean,
                namespace="data",
            )
            images = batch.get("images")
            if images is not None:
                padded_crops = int(images.shape[1])
                utilization = crop_counts.float().sum() / max(
                    int(crop_counts.numel()) * padded_crops, 1
                )
                self.record_metric(
                    "padded crops per sequence",
                    torch.tensor(float(padded_crops), device=crop_counts.device),
                    ReduceType.mean,
                    namespace="data",
                )
                self.record_metric(
                    "crop utilization",
                    utilization,
                    ReduceType.mean,
                    namespace="data",
                )
        if (pooled_counts := batch.get("pooled_token_counts")) is not None:
            self.record_metric(
                "pooled image tokens per sequence",
                pooled_counts.float().mean(),
                ReduceType.mean,
                namespace="data",
            )

    def _record_source_data_metrics(self, batch: Dict[str, Any], token_mask: torch.Tensor) -> None:
        """Record globally summed examples, tokens, and weighted loss mass by source."""
        packed_sources = batch.get("pack_source_names")
        unpacked_sources = batch.get("source_names")
        example_ids = batch.get("example_ids")
        loss_masks = batch.get("loss_masks")
        labels = batch.get("labels")
        if loss_masks is None or labels is None:
            raise OLMoConfigurationError("Per-source telemetry requires labels and loss masks")
        if (packed_sources is None) == (unpacked_sources is None):
            raise OLMoConfigurationError(
                "Per-source telemetry requires exactly one of pack_source_names (packed) or "
                "source_names (unpacked)"
            )
        if packed_sources is not None:
            if example_ids is None:
                raise OLMoConfigurationError("Packed per-source telemetry requires example_ids")
            if len(packed_sources) != int(example_ids.shape[0]):
                raise OLMoConfigurationError("Packed source metadata does not match the rank batch")
        else:
            assert unpacked_sources is not None
            if example_ids is not None:
                raise OLMoConfigurationError(
                    "Unpacked per-source telemetry must not carry example_ids"
                )
            if len(unpacked_sources) != int(token_mask.shape[0]):
                raise OLMoConfigurationError(
                    "Unpacked source metadata does not match the rank batch"
                )
        metric_names = (
            "examples",
            "tokens",
            "positive_tokens",
            "loss_weight",
            "active_loss_weight",
        )
        stats: Dict[str, Dict[str, torch.Tensor]] = {
            source_name: {name: loss_masks.new_zeros(()) for name in metric_names}
            for source_name in self.source_loss_mass_targets
        }
        observations: List[Tuple[int, str, torch.Tensor]] = []
        if packed_sources is not None:
            assert example_ids is not None
            for row, source_names in enumerate(packed_sources):
                observed_ids = torch.unique(example_ids[row][example_ids[row] >= 0]).tolist()
                if len(source_names) != len(observed_ids):
                    raise OLMoConfigurationError(
                        "Packed source metadata count does not match observed example IDs in row "
                        f"{row}: {len(source_names)} names for {len(observed_ids)} IDs"
                    )
                observations.extend(
                    (row, source_name, (example_ids[row] == example_id) & token_mask[row])
                    for example_id, source_name in zip(observed_ids, source_names)
                )
        else:
            assert unpacked_sources is not None
            observations.extend(
                (row, source_name, token_mask[row])
                for row, source_name in enumerate(unpacked_sources)
            )

        for row, source_name, positions in observations:
            if not isinstance(source_name, str) or not source_name:
                raise OLMoConfigurationError(
                    f"Observed invalid source name {source_name!r} in source telemetry"
                )
            if source_name not in self.source_loss_mass_targets:
                raise OLMoConfigurationError(
                    f"Observed unconfigured source {source_name!r} in source telemetry"
                )
            active_positions = positions & (labels[row] != self.label_ignore_index)
            observed_stats = stats[source_name]
            observed_stats["examples"] += 1
            observed_stats["tokens"] += positions.sum()
            observed_stats["positive_tokens"] += ((loss_masks[row] > 0) & active_positions).sum()
            observed_stats["loss_weight"] += (loss_masks[row] * positions).sum()
            observed_stats["active_loss_weight"] += (loss_masks[row] * active_positions).sum()

        source_names = tuple(self.source_loss_mass_targets)
        global_stats = torch.stack(
            [stats[source_name][name] for source_name in source_names for name in metric_names]
        )
        if is_distributed():
            global_stats = move_to_device(global_stats, self.device)
            dist.all_reduce(global_stats, group=self.dp_process_group)
        global_stats = global_stats.reshape(len(source_names), len(metric_names))
        total_loss_weight = global_stats[:, metric_names.index("loss_weight")].sum().clamp_min(1.0)
        for source_index, (source_name, target) in enumerate(self.source_loss_mass_targets.items()):
            for metric_index, metric_name in enumerate(metric_names):
                self.record_metric(
                    f"source/{source_name}/{metric_name}",
                    global_stats[source_index, metric_index],
                    # Every rank holds the same already-reduced value.
                    ReduceType.mean,
                    namespace="data",
                )
            realized_share = (
                global_stats[source_index, metric_names.index("loss_weight")] / total_loss_weight
            )
            self.record_metric(
                f"source/{source_name}/loss_mass_share",
                realized_share,
                ReduceType.mean,
                namespace="data",
            )
            self.record_metric(
                f"source/{source_name}/loss_mass_target_abs_error",
                (realized_share - target).abs(),
                ReduceType.mean,
                namespace="data",
            )

    # -- training step -----------------------------------------------------------

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        if self.optim is None:
            raise RuntimeError("train_batch() is unavailable when eval_only=True")
        if not dry_run:
            self._record_data_metrics(batch)
        collect_diagnostics = not dry_run and self._diagnostics_enabled_for_step()
        if collect_diagnostics:
            self._multimodal.set_input_diagnostics(True)
        try:
            result = self._train_batch_impl(batch, dry_run=dry_run)
        except BaseException:
            if collect_diagnostics:
                self._multimodal.set_input_diagnostics(False)
            raise
        if collect_diagnostics:
            diagnostics = self._multimodal.pop_input_diagnostics(
                reduce_across_process_group=is_distributed(),
                process_group=self.dp_process_group,
            )
            for name, value in diagnostics.items():
                self.record_metric(name, value, reduce_type=None, namespace="multimodal")
        return result

    def _train_batch_impl(self, batch: Dict[str, Any], dry_run: bool = False):
        self._set_model_mode("train")

        # Global loss-weight divisor (mm_olmo BatchDivisor.global_batch): the sum of
        # positive loss weights over the whole global batch, divided by DP world size.
        # After DDP averages gradients across ranks, the effective divisor is the global
        # weight. For a single rank this is just the local weight sum.
        loss_masks = batch["loss_masks"].to(self.device).float()
        if (batch_labels := batch.get("labels")) is None:
            raise OLMoConfigurationError("Multimodal training batches require explicit labels")
        active_loss_mask = (loss_masks > 0) & (
            batch_labels.to(self.device) != self.label_ignore_index
        )
        local_weight = loss_masks.masked_select(active_loss_mask).sum()
        if is_distributed():
            div_factor = local_weight.clone()
            dist.all_reduce(div_factor, group=self.dp_process_group)
            div_factor = div_factor / get_world_size(self.dp_process_group)
        else:
            div_factor = local_weight
        div_factor = torch.clamp(div_factor, min=1.0)

        pack_sources = batch.get("pack_source_names")
        if not dry_run:
            self._log_batch_sources(batch, local_weight)

        ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        z_batch_loss: Optional[torch.Tensor] = (
            move_to_device(torch.tensor(0.0), self.device)
            if self.z_loss_multiplier is not None
            else None
        )
        weight_total = move_to_device(torch.tensor(0.0), self.device)

        if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative to "
                f"sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
        num_micro_batches = len(micro_batches)

        if get_rank() == 0 and not dry_run and _mm_train_verbose_logs():
            images = batch.get("images")
            bsz, seq_len = batch["input_ids"].shape[:2]
            n_crops = int(images.shape[1]) if images is not None else 0
            vit_bt = bsz * n_crops
            gpu_mem_gb = (
                torch.cuda.max_memory_allocated(self.device) / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            )
            log.info(
                "batch shapes: input_ids=%s crops/seq=%d vit_B*T=%d gpu_max_alloc_gb=%.2f",
                tuple(batch["input_ids"].shape),
                n_crops,
                vit_bt,
                gpu_mem_gb,
            )

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                input_ids, labels, mb_loss_masks, model_kwargs = self._prepare_batch(micro_batch)
                assert labels is not None
                mb_loss_masks = mb_loss_masks.to(self.device).float()

                # ``labels`` / ``loss_masks`` are already next-token-aligned (shifted) by
                # the data pipeline, so no additional shift here.
                with self._model_forward_context():
                    if self.response_logits_only:
                        logits = self.model(
                            input_ids,
                            labels=None,
                            response_logits_only=True,
                            loss_masks=mb_loss_masks,
                            **model_kwargs,
                        )
                        response_mask = mb_loss_masks > 0
                        flat_logits = logits
                        flat_labels = labels.to(self.device).reshape(-1)[response_mask.reshape(-1)]
                        flat_weights = mb_loss_masks.reshape(-1)[response_mask.reshape(-1)]
                    else:
                        logits = self.model(input_ids, labels=None, **model_kwargs)
                        vocab_size = logits.shape[-1]
                        flat_logits = logits.reshape(-1, vocab_size)
                        flat_labels = labels.to(self.device).reshape(-1)
                        flat_weights = mb_loss_masks.reshape(-1)
                        # Mask out non-loss positions from the CE target for safety.
                        flat_labels = torch.where(
                            flat_weights > 0,
                            flat_labels,
                            flat_labels.new_full((), self.label_ignore_index),
                        )

                ce_loss, z_loss = weighted_cross_entropy_loss(
                    flat_logits,
                    flat_labels,
                    flat_weights,
                    ignore_index=self.label_ignore_index,
                    compute_z_loss=self.z_loss_multiplier is not None and not dry_run,
                    z_loss_multiplier=self.z_loss_multiplier or 1e-4,
                )

                # Every rank must enter the distributed failure reduction on every
                # microbatch. Otherwise a failing rank could issue this collective while
                # healthy ranks continue into backward collectives and hang NCCL.
                local_failed = not bool(torch.isfinite(ce_loss).item())
                if reduce_distributed_failure_flag(
                    local_failed, self.device, group=self.dp_process_group
                ):
                    if local_failed:
                        n_im_patch = int(
                            (input_ids == self._multimodal.cfg.image_patch_token_id).sum()
                        )
                        raise RuntimeError(
                            f"Non-finite CE loss on rank {get_rank()}: ce={ce_loss.item()}, "
                            f"local_weight={local_weight.item():.4f}, "
                            f"logits_nan={bool(torch.isnan(logits).any())}, "
                            f"logits_inf={bool(torch.isinf(logits).any())}, "
                            f"im_patch_tokens={n_im_patch}, seq_len={input_ids.shape[1]}, "
                            f"sources={pack_sources}"
                        )
                    raise RuntimeError(
                        f"Training failed on another rank (rank {get_rank()} had finite CE)"
                    )

                if dry_run:
                    continue

                loss = ce_loss / div_factor
                if z_loss is not None:
                    loss = loss + z_loss / div_factor

                ce_batch_loss += get_local_tensor(ce_loss.detach())
                active_weights = flat_weights.masked_select(flat_labels != self.label_ignore_index)
                weight_total += get_local_tensor(active_weights.sum().detach()).float()
                if z_batch_loss is not None and z_loss is not None:
                    z_batch_loss += get_local_tensor(z_loss.detach())

                loss.backward()

        del batch

        # Delegate auxiliary-metric bookkeeping to the underlying Transformer.
        if hasattr(self._lm, "post_batch"):
            self._lm.post_batch(dry_run=dry_run)
        if dry_run:
            if hasattr(self._lm, "reset_auxiliary_metrics"):
                self._lm.reset_auxiliary_metrics()
            return

        # Reduce numerator and denominator together so every rank (and SkipStep's rolling
        # history) sees the same globally weighted mean, even when source delivery is uneven.
        global_totals = torch.stack(
            [
                ce_batch_loss.float(),
                weight_total.float(),
                (
                    z_batch_loss.float()
                    if z_batch_loss is not None
                    else torch.zeros_like(ce_batch_loss).float()
                ),
            ]
        )
        if is_distributed():
            dist.all_reduce(global_totals, group=self.dp_process_group)
        mean_ce = global_totals[0] / global_totals[1].clamp_min(1.0)
        self.record_ce_loss(mean_ce, reduce_type=None)
        if isinstance(self.optim, SkipStepOptimizer):
            self.optim.latest_loss = mean_ce.detach()
        if z_batch_loss is not None:
            assert self.z_loss_multiplier is not None
            mean_z = global_totals[2] / global_totals[1].clamp_min(1.0)
            self.record_metric("Z loss", mean_z, reduce_type=None, namespace="train")

        if hasattr(self._lm, "compute_auxiliary_metrics"):
            for metric_name, (
                metric_val,
                reduction,
            ) in self._lm.compute_auxiliary_metrics(reset=True).items():
                self.record_metric(metric_name, metric_val, reduction, namespace="train")

        if not dry_run and _mm_train_verbose_logs():
            log.info(
                "train_batch rank=%d complete local_weight=%.1f",
                get_rank(),
                float(local_weight.item()),
            )

    def optim_step(self):
        if self.optim is None:
            raise RuntimeError("optim_step() is unavailable when eval_only=True")
        if self._diagnostics_enabled_for_step():
            trainable = {
                name: param for name, param in self.model.named_parameters() if param.requires_grad
            }
            patterns = _matched_component_grad_norm_patterns(
                _GENERIC_COMPONENT_GRAD_NORM_PATTERNS, set(trainable)
            )
            for component, component_patterns in patterns.items():
                params = [
                    param
                    for name, param in trainable.items()
                    if any(fnmatch(name, pattern) for pattern in component_patterns)
                ]
                grads = [param.grad for param in params if param.grad is not None]
                if grads:
                    grad_norm = torch.nn.utils.get_total_norm(
                        grads,
                        norm_type=2.0,
                        error_if_nonfinite=False,
                    )
                    if isinstance(grad_norm, DTensor):
                        grad_norm = grad_norm.full_tensor()
                else:
                    grad_norm = torch.zeros((), device=self.device)
                self.record_metric(
                    f"{component} grad norm",
                    grad_norm,
                    reduce_type=None,
                    namespace="optim",
                )

        if self.max_grad_norm is not None or isinstance(self.optim, SkipStepOptimizer):
            grad_norm = self._clip_grad_norm(
                self.max_grad_norm if self.max_grad_norm is not None else math.inf
            )
            if isinstance(self.optim, SkipStepOptimizer):
                self.optim.latest_grad_norm = grad_norm.detach()
        if self.max_grad_norm is not None:
            self.trainer.record_metric(
                "total grad norm", grad_norm, reduce_type=None, namespace="optim"
            )

        if self.scheduler is not None:
            for group_idx, group in enumerate(self.optim.param_groups):
                new_lr = self.scheduler.set_lr(group, self.trainer)
                self.trainer.record_metric(f"LR (group {group_idx})", new_lr, namespace="optim")

        self.optim.step()
        if isinstance(self.optim, SkipStepOptimizer):
            self.record_metric("step skipped", self.optim.step_skipped, namespace="optim")

        if hasattr(self._lm, "post_optim_step"):
            self._lm.post_optim_step()

    def eval_batch(
        self,
        batch: Dict[str, Any],
        labels: Optional[torch.Tensor] = None,
        *,
        return_response_logits: bool = False,
    ) -> LMOutputWithLoss:
        """Evaluate a multimodal weighted-loss batch without mutating evaluator input.

        Batches carrying ``loss_masks`` return a scalar summed weighted CE, which is the contract
        expected by :class:`~olmo_core.eval.MultimodalLMEvaluator`. Ordinary text-only batches
        retain the standard per-token CE shape used by :class:`~olmo_core.eval.LMEvaluator`.

        :param return_response_logits: Retain logits at supervised response positions. This is
            allowed only when ``response_logits_only=True`` so callers cannot accidentally retain
            a full ``batch x sequence x vocab`` tensor during alignment evaluation.
        """
        if return_response_logits and "loss_masks" not in batch:
            raise ValueError("return_response_logits requires a multimodal loss-mask batch")
        if return_response_logits and not self.response_logits_only:
            raise RuntimeError(
                "return_response_logits requires response_logits_only=True to avoid "
                "materializing full-sequence vocabulary logits"
            )

        model_batch = dict(batch)
        if (batch_labels := model_batch.pop("labels", None)) is not None:
            labels = batch_labels
        if labels is None:
            raise OLMoConfigurationError("Evaluation batches require labels")

        self._set_model_mode("eval")
        try:
            if "loss_masks" not in model_batch:
                input_ids = model_batch.pop("input_ids")
                for metadata_name in (
                    "pack_source_names",
                    "source_names",
                    "image_crop_counts",
                    "pooled_token_counts",
                    "router_token_mask",
                ):
                    model_batch.pop(metadata_name, None)
                with self._eval_batch_context(), self._model_forward_context():
                    if self._multimodal.cfg.output_vocab_size is None:
                        output = self.model(
                            input_ids,
                            labels=labels,
                            ignore_index=self.label_ignore_index,
                            loss_reduction="none",
                            **model_batch,
                        )
                        if not isinstance(output, LMOutputWithLoss):
                            raise RuntimeError(
                                "Text evaluation with labels must return loss output"
                            )
                    else:
                        logits = self.model(input_ids, labels=None, **model_batch)
                        if not isinstance(logits, torch.Tensor):
                            raise RuntimeError("Text evaluation without labels must return logits")
                        flat_loss = F.cross_entropy(
                            logits.float().reshape(-1, logits.shape[-1]),
                            labels.to(self.device).reshape(-1),
                            ignore_index=self.label_ignore_index,
                            reduction="none",
                        )
                        ce_loss = flat_loss.reshape_as(labels)
                        output = LMOutputWithLoss(logits, ce_loss, ce_loss, None)
                return output._replace(
                    logits=output.logits.detach() if output.logits is not None else None,
                    loss=output.loss.detach(),
                    ce_loss=output.ce_loss.detach(),
                    z_loss=output.z_loss.detach() if output.z_loss is not None else None,
                )

            input_ids, labels, loss_masks, model_kwargs = self._prepare_batch(model_batch, labels)
            assert labels is not None
            loss_masks = loss_masks.to(self.device).float()
            with self._eval_batch_context(), self._model_forward_context():
                if self.response_logits_only:
                    logits = self.model(
                        input_ids,
                        labels=None,
                        response_logits_only=True,
                        loss_masks=loss_masks,
                        **model_kwargs,
                    )
                    response_mask = loss_masks > 0
                    flat_logits = logits
                    flat_labels = labels.to(self.device).reshape(-1)[response_mask.reshape(-1)]
                    flat_weights = loss_masks.reshape(-1)[response_mask.reshape(-1)]
                else:
                    logits = self.model(input_ids, labels=None, **model_kwargs)
                    flat_logits = logits.reshape(-1, logits.shape[-1])
                    flat_labels = labels.to(self.device).reshape(-1)
                    flat_weights = loss_masks.reshape(-1)
                    flat_labels = torch.where(
                        flat_weights > 0,
                        flat_labels,
                        flat_labels.new_full((), self.label_ignore_index),
                    )
                ce_loss, _ = weighted_cross_entropy_loss(
                    flat_logits,
                    flat_labels,
                    flat_weights,
                    ignore_index=self.label_ignore_index,
                    compute_z_loss=False,
                )
            detached_loss = ce_loss.detach()
            return LMOutputWithLoss(
                logits=logits.detach() if return_response_logits else None,
                loss=detached_loss,
                ce_loss=detached_loss,
                z_loss=None,
            )
        finally:
            if hasattr(self._lm, "reset_auxiliary_metrics"):
                self._lm.reset_auxiliary_metrics()

    def load_molmo2_vision_state_dict(self, hf_state_dict: Dict[str, torch.Tensor]) -> None:
        """Convert and strictly load a Molmo2 Hugging Face vision state dictionary."""
        from olmo_core.nn.vision import molmo2_hf_state_dict_to_vision

        vision_state = molmo2_hf_state_dict_to_vision(
            hf_state_dict, self.multimodal_model.cfg.vision
        )
        self.load_vision_state_dict(vision_state)

    def load_siglip_vision_state_dict(self, hf_state_dict: Dict[str, torch.Tensor]) -> None:
        """Convert and strictly load a SigLIP Hugging Face vision state dictionary."""
        from olmo_core.nn.vision import siglip_hf_state_dict_to_vision

        vision_state = siglip_hf_state_dict_to_vision(
            hf_state_dict, self.multimodal_model.cfg.vision
        )
        self.load_vision_state_dict(vision_state)

    @torch.no_grad()
    def load_vision_state_dict(self, vision_state: Dict[str, torch.Tensor]) -> None:
        """Strictly load native vision weights without replacing optimizer parameters.

        The distributed state-dict API scatters full input tensors into FSDP/HSDP DTensors and
        copies into the existing parameters. Unlike OLMoDDP, the generic optimizer directly owns
        those model parameters and therefore has no separate FP32 master weights to synchronize.

        :param vision_state: Complete native vision-encoder state dictionary.
        """
        # Use the native sharded state representation. ``full_state_dict=True`` gathers every
        # FSDP2 tensor and depends on an NCCL all-gather-coalesced operation that is unavailable
        # in some supported torch builds. The source vision dictionary is already complete on
        # every rank, so distributing each tensor directly is both cheaper and backend-safe.
        options = dist_cp_sd.StateDictOptions(strict=True)
        current_state = dist_cp_sd.get_model_state_dict(
            self.multimodal_model.vision, options=options
        )
        current_keys = set(current_state)
        incoming_keys = set(vision_state)
        if current_keys != incoming_keys:
            raise OLMoConfigurationError(
                "Vision state dictionary inventory is not exact; "
                f"missing={sorted(current_keys - incoming_keys)}, "
                f"unexpected={sorted(incoming_keys - current_keys)}"
            )
        state_to_load: Dict[str, torch.Tensor] = {}
        for key, current_value in current_state.items():
            incoming_value = vision_state[key]
            if not isinstance(incoming_value, torch.Tensor):
                raise OLMoConfigurationError(f"Vision state entry {key!r} is not a tensor")
            if tuple(current_value.shape) != tuple(incoming_value.shape):
                raise OLMoConfigurationError(
                    f"Vision tensor shape mismatch for {key!r}: "
                    f"current={tuple(current_value.shape)}, incoming={tuple(incoming_value.shape)}"
                )
            if current_value.dtype != incoming_value.dtype:
                raise OLMoConfigurationError(
                    f"Vision tensor dtype mismatch for {key!r}: "
                    f"current={current_value.dtype}, incoming={incoming_value.dtype}"
                )
            if current_value.layout != incoming_value.layout:
                raise OLMoConfigurationError(
                    f"Vision tensor layout mismatch for {key!r}: "
                    f"current={current_value.layout}, incoming={incoming_value.layout}"
                )
            local_incoming = incoming_value.to(current_value.device)
            if isinstance(current_value, DTensor):
                local_incoming = distribute_tensor(
                    local_incoming,
                    device_mesh=current_value.device_mesh,
                    placements=current_value.placements,
                )
            state_to_load[key] = local_incoming
        dist_cp_sd.set_model_state_dict(
            self.multimodal_model.vision,
            state_to_load,
            options=options,
        )
        if self.optim is not None:
            self.assert_vision_optimizer_state_synced()

    def _assert_optimizer_parameter_ownership(
        self, module: torch.nn.Module, *, component_name: str
    ) -> None:
        """Verify that every trainable parameter in ``module`` is optimizer-owned once."""
        if self.optim is None:
            raise RuntimeError("Cannot check optimizer state when eval_only=True")
        trainable_parameters = {
            id(param): name for name, param in module.named_parameters() if param.requires_grad
        }
        optimizer_counts = {param_id: 0 for param_id in trainable_parameters}
        for group in self.optim.param_groups:
            for param in group["params"]:
                param_id = id(param)
                if param_id in optimizer_counts:
                    optimizer_counts[param_id] += 1
        missing = sorted(
            trainable_parameters[param_id]
            for param_id, count in optimizer_counts.items()
            if count == 0
        )
        duplicated = sorted(
            trainable_parameters[param_id]
            for param_id, count in optimizer_counts.items()
            if count > 1
        )
        if missing or duplicated:
            raise RuntimeError(
                f"{component_name} optimizer parameter ownership is not exact; "
                f"missing={missing}, duplicated={duplicated}"
            )

    def assert_vision_optimizer_state_synced(self) -> None:
        """Verify that each trainable vision parameter is optimizer-owned exactly once.

        Generic AdamW/SkipStepAdamW optimize the model Parameters directly, including FSDP2
        DTensors. Identity and coverage are therefore the relevant synchronization invariants.
        """
        self._assert_optimizer_parameter_ownership(
            self.multimodal_model.vision, component_name="Vision"
        )

    @staticmethod
    @torch.no_grad()
    def _index_copy_parameter_rows(
        parameter: torch.Tensor, row_index: torch.Tensor, rows: torch.Tensor
    ) -> None:
        """Copy full-vocabulary rows into either an ordinary Tensor or an FSDP2 DTensor."""
        if isinstance(parameter, DTensor):
            full_parameter = parameter.full_tensor()
            full_parameter.index_copy_(0, row_index.to(full_parameter.device), rows)
            distributed_parameter = distribute_tensor(
                full_parameter,
                device_mesh=parameter.device_mesh,
                placements=parameter.placements,
            )
            parameter.copy_(distributed_parameter)
        else:
            parameter.index_copy_(0, row_index.to(parameter.device), rows.to(parameter.device))

    @classmethod
    @torch.no_grad()
    def _zero_optimizer_rows(
        cls, state_value: torch.Tensor, row_index: torch.Tensor, parameter_shape: torch.Size
    ) -> None:
        """Zero optimizer moments corresponding to reset rows, if moments already exist."""
        if state_value.ndim == 0 or tuple(state_value.shape) != tuple(parameter_shape):
            return
        zero_rows = torch.zeros(
            (row_index.numel(), *parameter_shape[1:]),
            dtype=state_value.dtype,
            device=(state_value.device if not isinstance(state_value, DTensor) else "cpu"),
        )
        if isinstance(state_value, DTensor):
            zero_rows = zero_rows.to(state_value.device)
        cls._index_copy_parameter_rows(state_value, row_index, zero_rows)

    @torch.no_grad()
    def reset_image_token_rows(
        self, token_ids: List[int], *, seed: int, reset_output_rows: bool = True
    ) -> None:
        """Deterministically initialize newly assigned image-token rows.

        Existing optimizer moments for these rows are cleared. This normally runs before the
        first step, when Adam has no state yet, but clearing makes the API safe after a resumed
        initialization workflow as well.
        """
        if self.optim is None:
            raise RuntimeError("reset_image_token_rows() is unavailable when eval_only=True")
        if not token_ids or len(set(token_ids)) != len(token_ids):
            raise ValueError("token_ids must be a non-empty list of unique IDs")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        lm = self.multimodal_model.lm
        if lm.embeddings is None or lm.lm_head is None:
            raise RuntimeError("Image-token initialization requires LM embeddings and an LM head")
        if min(token_ids) < 0 or max(token_ids) >= lm.vocab_size:
            raise ValueError(
                f"Image token IDs must be within [0, {lm.vocab_size}), got {token_ids}"
            )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        row_count = len(token_ids)
        embedding_rows = torch.nn.Embedding(
            row_count,
            lm.d_model,
            device=self.device,
            dtype=lm.embeddings.weight.dtype,
        )
        lm.init_method.init_embeddings(
            embedding_rows,
            d_model=lm.d_model,
            embed_scale=lm.embed_scale,
            std=lm.embedding_init_std if lm.embedding_init_std is not None else lm.init_std,
            generator=generator,
        )
        row_index = torch.tensor(token_ids, device=self.device, dtype=torch.long)
        reset_parameters = [lm.embeddings.weight]
        self._index_copy_parameter_rows(lm.embeddings.weight, row_index, embedding_rows.weight)

        if reset_output_rows and lm.lm_head.w_out.weight is not lm.embeddings.weight:
            output_rows = torch.nn.Linear(
                lm.d_model,
                row_count,
                bias=False,
                device=self.device,
                dtype=lm.lm_head.w_out.weight.dtype,
            )
            lm.init_method.init_final_w_out(
                output_rows,
                d_model=lm.d_model,
                std=lm.init_std,
                generator=generator,
            )
            self._index_copy_parameter_rows(lm.lm_head.w_out.weight, row_index, output_rows.weight)
            reset_parameters.append(lm.lm_head.w_out.weight)

        for parameter in reset_parameters:
            for state_value in self.optim.state.get(parameter, {}).values():
                if isinstance(state_value, torch.Tensor):
                    self._zero_optimizer_rows(state_value, row_index, parameter.shape)
        self.assert_vision_optimizer_state_synced()

    @torch.no_grad()
    def load_parent_model_state_dict(
        self,
        checkpoint_dir: PathOrStr,
        *,
        current_to_checkpoint_key_mapping: Mapping[str, str],
        expected_loaded_model_keys: Collection[str],
        expected_missing_model_keys: Collection[str],
        expected_loaded_parameter_keys: Collection[str],
        process_group: Optional[dist.ProcessGroup] = None,
        pre_download: bool = False,
        work_dir: Optional[PathOrStr] = None,
        thread_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Strictly initialize this multimodal model from a model-only parent checkpoint.

        This method is deliberately separate from :meth:`state_dict_to_load`: bridge
        initialization has expected new vision/connector tensors, while every same-phase resume
        must retain the ordinary strict checkpoint contract. No optimizer state is read or reset.

        Keys in the three inventory arguments are *current* model-state keys without the outer
        ``"model."`` checkpoint prefix. A native text parent commonly maps
        ``"lm.blocks.0..."`` to ``"blocks.0..."``. The loaded and missing tensor inventories
        must form an exact partition of the current model state, every parent model tensor must be
        consumed exactly once, and the loaded parameter inventory must exactly identify which
        loaded tensors are parameters rather than buffers.

        :param checkpoint_dir: Exact DCP directory containing ``.metadata`` (normally the
            parent's ``model_and_optim`` directory).
        :param current_to_checkpoint_key_mapping: Exact current-to-parent key aliases, excluding
            the outer ``model.`` prefix on both sides.
        :param expected_loaded_model_keys: Current tensor/buffer keys expected from the parent.
        :param expected_missing_model_keys: Current keys intentionally initialized by this model.
        :param expected_loaded_parameter_keys: Exact parameter subset of the loaded inventory.
        :returns: An auditable inventory receipt. The full sorted key tuples are included.

        :raises OLMoConfigurationError: If inventories, aliases, shapes, or checkpoint coverage do
            not match exactly.
        """
        metadata = get_checkpoint_metadata(checkpoint_dir)
        checkpoint_model_keys = {
            key.removeprefix("model.")
            for key in metadata.state_dict_metadata
            if key.startswith("model.")
        }
        if not checkpoint_model_keys:
            raise OLMoConfigurationError(
                f"Parent checkpoint {checkpoint_dir!s} contains no model.* tensors"
            )

        state_opts = replace(self.state_dict_load_opts, strict=True)
        complete_state = {"model": dist_cp_sd.get_model_state_dict(self.model, options=state_opts)}
        current_model_keys = set(complete_state["model"])
        current_parameter_keys = set(dict(self.model.named_parameters()))
        expected_loaded = set(expected_loaded_model_keys)
        expected_missing = set(expected_missing_model_keys)
        expected_loaded_params = set(expected_loaded_parameter_keys)

        overlap = expected_loaded & expected_missing
        if overlap:
            raise OLMoConfigurationError(
                f"Parent load inventories overlap on current keys: {sorted(overlap)}"
            )
        if expected_loaded | expected_missing != current_model_keys:
            absent = current_model_keys - expected_loaded - expected_missing
            unknown = (expected_loaded | expected_missing) - current_model_keys
            raise OLMoConfigurationError(
                "Parent load inventories do not exactly partition current model state; "
                f"unclassified={sorted(absent)}, unknown={sorted(unknown)}"
            )
        actual_loaded_params = expected_loaded & current_parameter_keys
        if expected_loaded_params != actual_loaded_params:
            raise OLMoConfigurationError(
                "Loaded parameter inventory is not exact; "
                f"missing={sorted(actual_loaded_params - expected_loaded_params)}, "
                f"unexpected={sorted(expected_loaded_params - actual_loaded_params)}"
            )

        key_mapping = dict(current_to_checkpoint_key_mapping)
        unknown_mapping_keys = set(key_mapping) - expected_loaded
        missing_mapping_keys = expected_loaded - set(key_mapping)
        if unknown_mapping_keys or missing_mapping_keys:
            raise OLMoConfigurationError(
                "Parent key mapping must cover the loaded inventory exactly; "
                f"missing={sorted(missing_mapping_keys)}, "
                f"unexpected={sorted(unknown_mapping_keys)}"
            )
        if any(not current or not parent for current, parent in key_mapping.items()):
            raise OLMoConfigurationError("Parent key mapping cannot contain empty keys")
        resolved_checkpoint_keys = dict(key_mapping)
        if len(set(resolved_checkpoint_keys.values())) != len(resolved_checkpoint_keys):
            raise OLMoConfigurationError(
                "Parent key mapping maps multiple current keys to one tensor"
            )
        resolved_key_set = set(resolved_checkpoint_keys.values())
        if resolved_key_set != checkpoint_model_keys:
            raise OLMoConfigurationError(
                "Parent model tensor inventory does not exactly match the expected loaded "
                f"inventory; missing_in_checkpoint={sorted(resolved_key_set - checkpoint_model_keys)}, "
                f"unexpected_in_checkpoint={sorted(checkpoint_model_keys - resolved_key_set)}"
            )

        loaded_dtype_counts: Dict[str, int] = {}
        loaded_layout_counts: Dict[str, int] = {}
        for current_key, checkpoint_key in resolved_checkpoint_keys.items():
            tensor_meta = metadata.state_dict_metadata[f"model.{checkpoint_key}"]
            if not isinstance(tensor_meta, TensorStorageMetadata):
                raise OLMoConfigurationError(
                    f"Parent model entry model.{checkpoint_key} is not a tensor"
                )
            current_value = complete_state["model"][current_key]
            if tuple(tensor_meta.size) != tuple(current_value.shape):
                raise OLMoConfigurationError(
                    f"Parent tensor shape mismatch for {current_key!r} <- {checkpoint_key!r}: "
                    f"current={tuple(current_value.shape)}, checkpoint={tuple(tensor_meta.size)}"
                )
            if tensor_meta.properties.dtype != current_value.dtype:
                raise OLMoConfigurationError(
                    f"Parent tensor dtype mismatch for {current_key!r} <- {checkpoint_key!r}: "
                    f"current={current_value.dtype}, checkpoint={tensor_meta.properties.dtype}"
                )
            if tensor_meta.properties.layout != current_value.layout:
                raise OLMoConfigurationError(
                    f"Parent tensor layout mismatch for {current_key!r} <- {checkpoint_key!r}: "
                    f"current={current_value.layout}, checkpoint={tensor_meta.properties.layout}"
                )
            dtype_name = str(tensor_meta.properties.dtype)
            layout_name = str(tensor_meta.properties.layout)
            loaded_dtype_counts[dtype_name] = loaded_dtype_counts.get(dtype_name, 0) + 1
            loaded_layout_counts[layout_name] = loaded_layout_counts.get(layout_name, 0) + 1

        # Rename only for this one load, prune the intentionally new components, and then merge
        # their untouched initialized values back before a strict set_model_state_dict call.
        state_to_load = complete_state
        swap_param_keys(state_to_load, key_mapping)
        pruned = prune_state_dict(
            state_to_load,
            {f"model.{key}" for key in checkpoint_model_keys},
        )
        expected_pruned = {f"model.{key}" for key in expected_missing}
        if pruned != expected_pruned:
            raise OLMoConfigurationError(
                "Internal parent-load pruning did not match the declared missing inventory; "
                f"actual={sorted(pruned)}, expected={sorted(expected_pruned)}"
            )
        load_distributed_state_dict(
            checkpoint_dir,
            state_to_load,
            process_group=process_group,
            pre_download=pre_download,
            work_dir=work_dir,
            thread_count=thread_count,
        )
        swap_param_keys(state_to_load, key_mapping, reverse=True, quiet=True)
        initialized_state = {
            "model": dist_cp_sd.get_model_state_dict(self.model, options=state_opts)
        }
        merge_state_dicts(state_to_load, initialized_state)
        dist_cp_sd.set_model_state_dict(
            self.model,
            state_to_load["model"],
            options=state_opts,
        )
        if self.optim is not None:
            self._assert_optimizer_parameter_ownership(self.model, component_name="Model")

        return {
            "checkpoint_dir": str(checkpoint_dir),
            "loaded_model_tensor_count": len(expected_loaded),
            "loaded_parameter_count": len(expected_loaded_params),
            "missing_initialized_tensor_count": len(expected_missing),
            "missing_initialized_parameter_count": len(expected_missing & current_parameter_keys),
            "loaded_tensor_dtype_counts": dict(sorted(loaded_dtype_counts.items())),
            "loaded_tensor_layout_counts": dict(sorted(loaded_layout_counts.items())),
            "loaded_model_keys": tuple(sorted(expected_loaded)),
            "loaded_parameter_keys": tuple(sorted(expected_loaded_params)),
            "missing_initialized_model_keys": tuple(sorted(expected_missing)),
        }

    @lru_cache
    def num_flops_per_token(self, seq_len: int) -> Optional[int]:
        try:
            if hasattr(self._lm, "num_flops_per_token"):
                return self._lm.num_flops_per_token(seq_len)
        except NotImplementedError as ex:
            warn_once(f"Unable to estimate num flops per token: {ex}")
        return None

    def extra_flops_per_batch(self, batch: Dict[str, Any]) -> int:
        """Vision-encoder + connector FLOPs for ``batch`` (read by the speed monitor and
        added to the per-token LM FLOPs). The ViT processes every crop in ``batch["images"]``
        — including padded / dummy crops — so we size it off that tensor's shape."""
        images = batch.get("images")
        if images is None:
            return 0
        b, t, n_patches = (
            int(images.shape[0]),
            int(images.shape[1]),
            int(images.shape[2]),
        )
        n_pooled = int((batch["input_ids"] == self._multimodal.cfg.image_patch_token_id).sum())
        return self._multimodal.image_encoder_flops(b * t, n_patches, n_pooled)


@dataclass
class MultimodalTransformerTrainModuleConfig(TrainModuleConfig):
    """Configuration for :class:`MultimodalTransformerTrainModule`."""

    rank_microbatch_size: int
    max_sequence_length: int
    optim: OptimConfig
    freeze_params: Optional[List[str]] = None
    max_grad_norm: Optional[float] = None
    scheduler: Optional[Scheduler] = None
    compile_model: bool = False
    dp_config: Optional[TransformerDataParallelConfig] = None
    ac_config: Optional[TransformerActivationCheckpointingConfig] = None
    vision_activation_checkpointing: bool = True
    connector_activation_checkpointing: bool = True
    z_loss_multiplier: Optional[float] = None
    autocast_precision: Optional[DType] = None
    label_ignore_index: int = -100
    response_logits_only: bool = False
    diagnostics_interval: Optional[int] = None
    train_embedding_rows: Optional[List[int]] = None
    """Embedding rows allowed to receive gradients; every other input row is held fixed."""
    source_loss_mass_targets: Optional[Dict[str, float]] = None
    """Expected source loss-mass shares used for online delivery telemetry."""
    new_component_init_seed: Optional[int] = None
    """Explicit seed for meta-initialized vision/connector components."""
    state_dict_save_opts: Optional[Dict[str, Any]] = None
    state_dict_load_opts: Optional[Dict[str, Any]] = None
    load_key_mapping: Optional[Dict[str, str]] = None

    def build(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        eval_only: bool = False,
    ) -> "MultimodalTransformerTrainModule":
        """Build the generic multimodal train module.

        :param model: Generic multimodal model to parallelize and materialize.
        :param device: Runtime device.
        :param eval_only: Skip optimizer construction for strict checkpoint evaluation.
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**save_opts)
        if (load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**load_opts)
        return MultimodalTransformerTrainModule(
            model=model, device=device, eval_only=eval_only, **kwargs
        )


class MultimodalOLMoDDPTrainModule(OLMoDDPTrainModule):
    """OLMoDDP EP/DP training for a multimodal model with weighted token loss."""

    def __init__(
        self,
        model: torch.nn.Module,
        *args,
        freeze_params: Optional[List[str]] = None,
        vision_activation_checkpointing: bool = False,
        connector_activation_checkpointing: bool = False,
        response_logits_only: bool = False,
        diagnostics_interval: Optional[int] = None,
        train_embedding_rows: Optional[List[int]] = None,
        source_loss_mass_targets: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        from olmo_core.nn.vision import MultimodalOLMoDDPModel

        if not isinstance(model, MultimodalOLMoDDPModel):
            raise TypeError(
                f"{type(self).__name__} requires MultimodalOLMoDDPModel, got {type(model).__name__}"
            )
        unsupported = [
            name for name in ("tp_config", "cp_config", "pp_config") if kwargs.get(name) is not None
        ]
        if unsupported:
            raise OLMoConfigurationError(
                "Multimodal OLMoDDP currently supports data and expert parallelism only; "
                f"unset {', '.join(unsupported)}"
            )
        if model.tbo:
            raise OLMoConfigurationError(
                "Two-batch overlap is not supported for multimodal OLMoDDP"
            )
        if diagnostics_interval is not None and diagnostics_interval <= 0:
            raise OLMoConfigurationError("diagnostics_interval must be positive or None")

        self.freeze_params = freeze_params or []
        frozen = []
        for name, param in model.named_parameters():
            if any(fnmatch(name, pattern) for pattern in self.freeze_params):
                param.requires_grad_(False)
                frozen.append(name)
        if self.freeze_params:
            log.info(
                "Froze %d parameter tensors matching %s",
                len(frozen),
                self.freeze_params,
            )
        self.train_embedding_rows = tuple(sorted(train_embedding_rows or []))
        if len(self.train_embedding_rows) != len(set(self.train_embedding_rows)):
            raise OLMoConfigurationError("train_embedding_rows must contain unique IDs")
        if vision_activation_checkpointing:
            model.vision.apply_activation_checkpointing()
            log.info("Applied activation checkpointing to the vision encoder")
        if connector_activation_checkpointing:
            model.connector.apply_activation_checkpointing()
            log.info("Applied activation checkpointing to the vision connector")
        self.response_logits_only = response_logits_only
        self.diagnostics_interval = diagnostics_interval
        self.source_loss_mass_targets = dict(source_loss_mass_targets or {})
        if self.source_loss_mass_targets and (
            any(
                not math.isfinite(value) or value <= 0
                for value in self.source_loss_mass_targets.values()
            )
            or abs(sum(self.source_loss_mass_targets.values()) - 1.0) > 1e-6
        ):
            raise OLMoConfigurationError("source_loss_mass_targets must be positive and sum to one")
        super().__init__(model, *args, **kwargs)

        # OLMoDDP materializes meta-device weights inside ``super().__init__`` with ``to_empty``,
        # which replaces Parameter objects. Install gradient hooks only on the final materialized
        # parameters so they run before MultiGroupDDP's post-accumulate FP32 reduction hooks.
        materialized_lm = self.multimodal_model.lm
        self._embedding_grad_hook = None
        if self.train_embedding_rows:
            embeddings = materialized_lm.embeddings
            if embeddings is None:
                raise OLMoConfigurationError("train_embedding_rows requires LM embeddings")
            if self.train_embedding_rows[0] < 0 or self.train_embedding_rows[-1] >= int(
                embeddings.weight.shape[0]
            ):
                raise OLMoConfigurationError(
                    "train_embedding_rows contains an ID outside the LM embedding table"
                )
            if not embeddings.weight.requires_grad:
                raise OLMoConfigurationError(
                    "The LM embedding parameter must remain trainable when row masking is enabled"
                )
            if (
                materialized_lm.lm_head is not None
                and materialized_lm.lm_head.w_out.weight is embeddings.weight
            ):
                raise OLMoConfigurationError(
                    "Row-masked image embeddings require untied LM input and output weights"
                )
            self._embedding_grad_hook = embeddings.weight.register_hook(
                lambda grad: _retain_embedding_gradient_rows(grad, self.train_embedding_rows)
            )
            log.info(
                "Restricted LM input-embedding gradients to rows %s",
                self.train_embedding_rows,
            )

    @property
    def multimodal_model(self):
        model = self.model_parts[0]
        return getattr(model, "module", model)

    def _prepare_batch(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        input_ids, labels, model_kwargs = super()._prepare_batch(batch, labels)
        model_kwargs.pop("image_crop_counts", None)
        model_kwargs.pop("pooled_token_counts", None)
        # A full microbatch needs no routing override and remains compatible with the ordinary
        # sync/no-EP paths used by small tests and evaluation. A mask containing padding is kept
        # and the routed block enforces the production rowwise path.
        router_token_mask = model_kwargs.get("router_token_mask")
        if router_token_mask is not None and bool(router_token_mask.all()):
            model_kwargs.pop("router_token_mask")
        # Response-only logits are specific to multimodal batches carrying loss weights.
        # Ordinary downstream LM evaluators do not provide ``loss_masks`` and require full
        # sequence logits, even when the training module uses response-only logits for Stage 1.
        if self.response_logits_only and "loss_masks" in batch:
            model_kwargs["response_logits_only"] = True
        return input_ids, labels, model_kwargs

    @contextlib.contextmanager
    def _eval_batch_context(self):
        # Downstream text batches use synchronized EP because rank-local sequence shapes may
        # differ. Torch 2.11 Inductor cannot lower several valid OLMES shapes, while the eager EP
        # path is correct. Force the complete text forward eager, including attention and the LM
        # head, without changing the compiled training model. Keep grad mode enabled because the
        # torch 2.11 no-grad specialization can produce incorrect attention outputs on B300.
        with torch.enable_grad(), torch.compiler.set_stance("force_eager"):
            yield

    @contextlib.contextmanager
    def _multimodal_eval_batch_context(self):
        # Multimodal eval batches have the same fixed, padded shape and router token mask as
        # training batches. Keep the routed blocks on their proven no-sync rowwise path and in the
        # same grad-enabled compile regime. On B300 with torch 2.11, compiled no-grad attention can
        # produce wrong logits and compiled EP can access memory illegally. Setting only the
        # block's dispatch flag does not recursively enable training behavior in its children, and
        # OLMoDDP blocks do not support dropout. Returned losses are detached below so the
        # unconsumed graph is released before control returns to the evaluator.
        block_modes = []
        for block in self.multimodal_model.lm.routed_blocks():
            block_modes.append(
                (
                    block,
                    block.training,
                    block._ep_no_sync_force_scratch_lifetime_buffers,
                )
            )
            block.training = True
            # This is a forward-only graph, so no backward pass exists to release the rowwise
            # lifetime leases used by training. Use the prewarmed static scratch buffers instead.
            block._ep_no_sync_force_scratch_lifetime_buffers = True
        try:
            with torch.enable_grad():
                yield
        finally:
            for block, training, force_scratch in block_modes:
                block.training = training
                block._ep_no_sync_force_scratch_lifetime_buffers = force_scratch

    def _batch_auxiliary_loss_kwargs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        token_mask = batch.get("router_token_mask")
        if token_mask is None:
            raise OLMoConfigurationError(
                "Multimodal OLMoDDP batches require router_token_mask so padding is excluded "
                "from MoE routing and auxiliary losses"
            )
        if token_mask.shape != batch["input_ids"].shape:
            raise OLMoConfigurationError(
                "router_token_mask must match input_ids: "
                f"got {tuple(token_mask.shape)} and {tuple(batch['input_ids'].shape)}"
            )

        # Match OLMo-core's DDP loss-scaling convention: normalize by the global valid-token
        # count divided by the DP world size, so subsequent averaged gradients have the same
        # scale as a single global batch. This population is deliberately independent of the
        # response-only CE loss weights.
        router_loss_div_factor = move_to_device(token_mask.sum(dtype=torch.long), self.device)
        if is_distributed():
            dist.all_reduce(router_loss_div_factor, group=self.dp_process_group)
            router_loss_div_factor = router_loss_div_factor.clamp_min(1)
            router_loss_div_factor = router_loss_div_factor / get_world_size(self.dp_process_group)
        else:
            router_loss_div_factor = router_loss_div_factor.clamp_min(1)
        return {"router_loss_div_factor": router_loss_div_factor}

    def _record_data_metrics(self, batch: Dict[str, Any]) -> None:
        """Record packing and supervision density without synchronizing CUDA."""
        token_mask = batch.get("router_token_mask")
        if token_mask is None:
            return
        if getattr(self, "source_loss_mass_targets", None) and (
            getattr(self, "_trainer", None) is None or self._diagnostics_enabled_for_step()
        ):
            self._record_source_data_metrics(batch, token_mask)
        self.record_metric(
            "packing fill",
            token_mask.float().mean(),
            ReduceType.mean,
            namespace="data",
        )

        if (loss_masks := batch.get("loss_masks")) is not None:
            self.record_metric(
                "response token density",
                ((loss_masks > 0) & token_mask).float().mean(),
                ReduceType.mean,
                namespace="data",
            )
        if (token_type_ids := batch.get("token_type_ids")) is not None:
            self.record_metric(
                "image token density",
                ((token_type_ids != 0) & token_mask).float().mean(),
                ReduceType.mean,
                namespace="data",
            )
        if (example_ids := batch.get("example_ids")) is not None:
            self.record_metric(
                "examples per sequence",
                (example_ids.amax(dim=1) + 1).float().mean(),
                ReduceType.mean,
                namespace="data",
            )

        if (crop_counts := batch.get("image_crop_counts")) is not None:
            self.record_metric(
                "real crops per sequence",
                crop_counts.float().mean(),
                ReduceType.mean,
                namespace="data",
            )
            images = batch.get("images")
            if images is not None:
                padded_crops = int(images.shape[1])
                utilization = crop_counts.float().sum() / max(
                    int(crop_counts.numel()) * padded_crops, 1
                )
                self.record_metric(
                    "padded crops per sequence",
                    torch.tensor(float(padded_crops), device=crop_counts.device),
                    ReduceType.mean,
                    namespace="data",
                )
                self.record_metric(
                    "crop utilization",
                    utilization,
                    ReduceType.mean,
                    namespace="data",
                )
        if (pooled_counts := batch.get("pooled_token_counts")) is not None:
            self.record_metric(
                "pooled image tokens per sequence",
                pooled_counts.float().mean(),
                ReduceType.mean,
                namespace="data",
            )

    def _record_source_data_metrics(self, batch: Dict[str, Any], token_mask: torch.Tensor) -> None:
        """Record globally summed examples, tokens, and weighted loss mass by source.

        The configured mixture targets are expressed in *global supervised-loss mass*.
        Computing a ratio on each DP rank and averaging those ratios is not equivalent to
        dividing globally summed source weights by the globally summed total, especially when
        dense native-text rows and short response-only visual rows land on different ranks.
        This method therefore performs one stacked DP reduction at the diagnostics cadence and
        derives every reported share from those global sums.
        """
        packed_sources = batch.get("pack_source_names")
        example_ids = batch.get("example_ids")
        loss_masks = batch.get("loss_masks")
        labels = batch.get("labels")
        if packed_sources is None or example_ids is None or loss_masks is None or labels is None:
            raise OLMoConfigurationError(
                "Per-source telemetry requires packed source names, example IDs, labels, "
                "and loss masks"
            )
        if len(packed_sources) != int(example_ids.shape[0]):
            raise OLMoConfigurationError("Packed source metadata does not match the rank batch")
        metric_names = (
            "examples",
            "tokens",
            "positive_tokens",
            "loss_weight",
            "active_loss_weight",
        )
        stats: Dict[str, Dict[str, torch.Tensor]] = {
            source_name: {name: loss_masks.new_zeros(()) for name in metric_names}
            for source_name in self.source_loss_mass_targets
        }
        label_ignore_index = getattr(self, "label_ignore_index", -100)
        for row, source_names in enumerate(packed_sources):
            for example_id, source_name in enumerate(source_names):
                if source_name not in self.source_loss_mass_targets:
                    raise OLMoConfigurationError(
                        f"Observed unconfigured source {source_name!r} in packed telemetry"
                    )
                positions = (example_ids[row] == example_id) & token_mask[row]
                observed_stats = stats[source_name]
                active_positions = positions & (labels[row] != label_ignore_index)
                observed_stats["examples"] += 1
                observed_stats["tokens"] += positions.sum()
                observed_stats["positive_tokens"] += (
                    (loss_masks[row] > 0) & active_positions
                ).sum()
                observed_stats["loss_weight"] += (loss_masks[row] * positions).sum()
                observed_stats["active_loss_weight"] += (loss_masks[row] * active_positions).sum()

        source_names = tuple(self.source_loss_mass_targets)
        global_stats = torch.stack(
            [stats[source_name][name] for source_name in source_names for name in metric_names]
        )
        if is_distributed():
            global_stats = move_to_device(global_stats, self.device)
            dist.all_reduce(global_stats, group=self.dp_process_group)
        global_stats = global_stats.reshape(len(source_names), len(metric_names))
        stats = {
            source_name: {
                name: global_stats[source_index, metric_index]
                for metric_index, name in enumerate(metric_names)
            }
            for source_index, source_name in enumerate(source_names)
        }
        total_loss_weight = sum(
            (source_stats["loss_weight"] for source_stats in stats.values()),
            start=global_stats.new_zeros(()),
        ).clamp_min(1.0)
        for source_name, target in self.source_loss_mass_targets.items():
            metric_stats = stats[source_name]
            for metric_name, value in metric_stats.items():
                self.record_metric(
                    f"source/{source_name}/{metric_name}",
                    value,
                    # Every DP rank holds the identical already-summed tensor.
                    ReduceType.mean,
                    namespace="data",
                )
            realized_share = metric_stats["loss_weight"] / total_loss_weight
            self.record_metric(
                f"source/{source_name}/loss_mass_share",
                realized_share,
                ReduceType.mean,
                namespace="data",
            )
            self.record_metric(
                f"source/{source_name}/loss_mass_target_abs_error",
                (realized_share - target).abs(),
                ReduceType.mean,
                namespace="data",
            )

    def _diagnostics_enabled_for_step(self) -> bool:
        return bool(
            self.diagnostics_interval is not None
            and self._trainer is not None
            and self.trainer.global_step % self.diagnostics_interval == 0
        )

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        if not dry_run:
            self._record_data_metrics(batch)
        collect_diagnostics = not dry_run and self._diagnostics_enabled_for_step()
        if collect_diagnostics:
            self.multimodal_model.set_input_diagnostics(True)
        try:
            result = super().train_batch(batch, dry_run=dry_run)
        except BaseException:
            if collect_diagnostics:
                self.multimodal_model.set_input_diagnostics(False)
            raise
        if collect_diagnostics:
            diagnostics = self.multimodal_model.pop_input_diagnostics(
                reduce_across_process_group=is_distributed(),
                process_group=self.dp_process_group,
            )
            for name, value in diagnostics.items():
                self.record_metric(name, value, reduce_type=None, namespace="multimodal")
        return result

    def optim_step(self):
        optim = self._require_optimizer()
        collect_diagnostics = self._diagnostics_enabled_for_step()
        if collect_diagnostics:
            component_patterns = {
                "vision": ("vision.*", "*vision.*"),
                "connector": ("connector.*", "*connector.*"),
                "input embeddings": ("lm.embeddings.weight", "*lm.embeddings.weight"),
                "LM output head": (
                    "lm.lm_head.w_out.*",
                    "*lm.lm_head.w_out.*",
                ),
                "LM attention": ("lm.blocks.*.attention.*", "*lm.blocks.*.attention.*"),
                "LM routed experts": (
                    "lm.blocks.*.routed_experts.*",
                    "*lm.blocks.*.routed_experts.*",
                ),
                "LM shared experts": (
                    "lm.blocks.*.shared_experts.*",
                    "*lm.blocks.*.shared_experts.*",
                ),
                "LM routers": (
                    "lm.blocks.*.routed_experts_router.*",
                    "*lm.blocks.*.routed_experts_router.*",
                ),
                "LM normalization": ("lm.*norm*", "*lm.*norm*"),
            }
            trainable_names = {
                name
                for group in optim.param_groups
                for name, param in group["named_params"].items()
                if param.requires_grad
            }
            optim.set_component_grad_norm_patterns(
                _matched_component_grad_norm_patterns(component_patterns, trainable_names)
            )
        try:
            super().optim_step()
            if collect_diagnostics:
                for component, value in optim.latest_component_grad_norms.items():
                    self.record_metric(
                        f"{component} grad norm",
                        value,
                        reduce_type=None,
                        namespace="optim",
                    )
                for group_name, value in optim.latest_clip_group_grad_norms.items():
                    component = (
                        "language model"
                        if group_name == optim.DEFAULT_CLIP_GROUP_NAME
                        else group_name
                    )
                    self.record_metric(
                        f"{component} clip group grad norm",
                        value,
                        reduce_type=None,
                        namespace="optim",
                    )
                for group_name, value in optim.latest_clip_group_coefficients.items():
                    component = (
                        "language model"
                        if group_name == optim.DEFAULT_CLIP_GROUP_NAME
                        else group_name
                    )
                    self.record_metric(
                        f"{component} clip coefficient",
                        value,
                        reduce_type=None,
                        namespace="optim",
                    )
        finally:
            optim.set_component_grad_norm_patterns(None)

    def extra_flops_per_batch(self, batch: Dict[str, Any]) -> int:
        """Return vision and connector FLOPs for speed-monitor MFU accounting."""
        images = batch.get("images")
        if images is None:
            return 0
        batch_size, crops, patches = (int(value) for value in images.shape[:3])
        pooled = int((batch["input_ids"] == self.multimodal_model.cfg.image_patch_token_id).sum())
        return self.multimodal_model.image_encoder_flops(batch_size * crops, patches, pooled)

    def eval_batch(
        self,
        batch: Dict[str, Any],
        labels: Optional[torch.Tensor] = None,
        *,
        return_response_logits: bool = False,
    ) -> LMOutputWithLoss:
        """Evaluate multimodal response loss or delegate ordinary LM evaluation.

        Multimodal Stage 1 batches carry ``loss_masks`` and use a scalar summed response-token
        loss without materializing full-sequence logits. Text-only downstream batches do not
        carry ``loss_masks`` and need the standard OLMoDDP path so evaluators receive logits.

        :param return_response_logits: Retain logits at supervised response positions for a
            multimodal batch. This requires ``response_logits_only=True`` on the train module so
            an evaluator cannot accidentally materialize full-sequence vocabulary logits.
        """
        if "loss_masks" not in batch:
            if return_response_logits:
                raise ValueError(
                    "return_response_logits is only valid for multimodal loss-mask batches"
                )
            output = super().eval_batch(batch, labels=labels)
            assert isinstance(output, LMOutputWithLoss), "Expected LMOutputWithLoss"
            return output._replace(
                logits=output.logits.detach() if output.logits is not None else None,
                loss=output.loss.detach(),
                ce_loss=output.ce_loss.detach(),
                z_loss=output.z_loss.detach() if output.z_loss is not None else None,
            )

        if self.cp_enabled or self.tp_enabled or self.pp_enabled:
            raise RuntimeError(
                f"{self.__class__.__name__}.eval_batch() only supports the Stage 1 EP/DP topology"
            )
        if return_response_logits and not self.response_logits_only:
            raise RuntimeError(
                "return_response_logits requires response_logits_only=True to avoid "
                "materializing full-sequence vocabulary logits"
            )

        # EvaluatorCallback derives ordinary LM labels from input_ids, but multimodal batches
        # carry branch-aware, already-shifted labels. Prefer those and leave the original batch
        # intact so the evaluator can use its loss weights after the forward pass.
        model_batch = dict(batch)
        if (batch_labels := model_batch.pop("labels", None)) is not None:
            labels = batch_labels
        input_ids, labels, model_kwargs = self._prepare_batch(model_batch, labels)
        if labels is None:
            raise OLMoConfigurationError("Multimodal evaluation batches require labels")

        for model_part in self.model_parts:
            model_part.eval()

        try:
            with self._multimodal_eval_batch_context():
                output = self.model_forward_no_pipeline(
                    input_ids,
                    labels=labels,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="sum",
                    return_logits=return_response_logits,
                    **model_kwargs,
                )
                assert isinstance(output, LMOutputWithLoss), "Expected LMOutputWithLoss"
                return output._replace(
                    logits=(
                        output.logits.detach()
                        if return_response_logits and output.logits is not None
                        else None
                    ),
                    loss=output.loss.detach(),
                    ce_loss=output.ce_loss.detach(),
                    z_loss=output.z_loss.detach() if output.z_loss is not None else None,
                )
        finally:
            # Router metrics from held-out data must not leak into the next training window.
            for model_part in self.model_parts:
                model_part.reset_auxiliary_metrics()

    def load_molmo2_vision_state_dict(self, hf_state_dict: Dict[str, torch.Tensor]) -> None:
        """Strictly load the Molmo2 vision tower, leaving the connector untouched."""
        from olmo_core.nn.vision import molmo2_hf_state_dict_to_vision

        model = self.multimodal_model
        vision_state = molmo2_hf_state_dict_to_vision(hf_state_dict, model.cfg.vision)
        self.load_vision_state_dict(vision_state)

    def load_siglip_vision_state_dict(self, hf_state_dict: Dict[str, torch.Tensor]) -> None:
        """Strictly load a SigLIP vision tower and synchronize optimizer masters."""
        from olmo_core.nn.vision import siglip_hf_state_dict_to_vision

        model = self.multimodal_model
        vision_state = siglip_hf_state_dict_to_vision(hf_state_dict, model.cfg.vision)
        self.load_vision_state_dict(vision_state)

    @torch.no_grad()
    def load_vision_state_dict(self, vision_state: Dict[str, torch.Tensor]) -> None:
        """Strictly load vision weights and synchronize trainable optimizer masters.

        OLMoDDP creates FP32 optimizer master parameters when the train module is built. Any
        model-only load performed afterwards must update those masters before the first optimizer
        step, otherwise that step copies the stale initialization back into the vision tower.

        :param vision_state: State dictionary in the native vision encoder format.
        """
        model = self.multimodal_model
        model.vision.load_state_dict(vision_state, strict=True)

        if self.optim is None:
            return

        trainable_vision_params = {
            id(param) for param in model.vision.parameters() if param.requires_grad
        }
        if not trainable_vision_params:
            return

        optim = self._require_optimizer()
        vision_param_names = {
            name
            for param_group in optim.param_groups
            for name, param in param_group["named_params"].items()
            if id(param) in trainable_vision_params
        }
        if len(vision_param_names) != len(trainable_vision_params):
            raise RuntimeError(
                "Could not map every trainable vision parameter to its optimizer master: "
                f"found {len(vision_param_names)} of {len(trainable_vision_params)}"
            )
        optim._copy_model_params_to_main_params(vision_param_names)
        optim._check_model_param_main_param_the_same(vision_param_names)

    def assert_vision_optimizer_state_synced(self) -> None:
        """Check every trainable vision tensor against its optimizer-owned FP32 master."""
        if self.optim is None:
            raise RuntimeError("Cannot check optimizer state on an eval-only train module")

        model = self.multimodal_model
        trainable_vision_params = {
            id(param) for param in model.vision.parameters() if param.requires_grad
        }
        vision_param_names = {
            name
            for param_group in self.optim.param_groups
            for name, param in param_group["named_params"].items()
            if id(param) in trainable_vision_params
        }
        if len(vision_param_names) != len(trainable_vision_params):
            raise RuntimeError(
                "Could not map every trainable vision parameter to its optimizer master: "
                f"found {len(vision_param_names)} of {len(trainable_vision_params)}"
            )
        self.optim._check_model_param_main_param_the_same(vision_param_names)

    @torch.no_grad()
    def reset_image_token_rows(
        self, token_ids: List[int], *, seed: int, reset_output_rows: bool = True
    ) -> None:
        """Initialize newly assigned image-token rows and update optimizer main state.

        :param token_ids: Input-embedding row IDs to initialize.
        :param seed: Initialization seed.
        :param reset_output_rows: Also initialize the same untied LM-head rows. This must remain
            false when adapting s002 because its padded output rows already participated in the
            native pretraining softmax.
        """
        if not token_ids or len(set(token_ids)) != len(token_ids):
            raise ValueError("token_ids must be a non-empty list of unique IDs")

        model = self.multimodal_model
        lm = model.lm
        if lm.embeddings is None or lm.lm_head is None:
            raise RuntimeError("Image-token initialization requires LM embeddings and an LM head")
        if min(token_ids) < 0 or max(token_ids) >= lm.vocab_size:
            raise ValueError(
                f"Image token IDs must be within [0, {lm.vocab_size}), got {token_ids}"
            )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        row_count = len(token_ids)
        embedding_rows = torch.nn.Embedding(
            row_count,
            lm.d_model,
            device=self.device,
            dtype=lm.embeddings.weight.dtype,
        )
        lm.init_method.init_embeddings(
            embedding_rows,
            d_model=lm.d_model,
            embed_scale=lm.embed_scale,
            std=lm.embedding_init_std if lm.embedding_init_std is not None else lm.init_std,
            generator=generator,
        )
        row_index = torch.tensor(token_ids, device=self.device, dtype=torch.long)
        lm.embeddings.weight.index_copy_(0, row_index, embedding_rows.weight)

        if reset_output_rows and lm.lm_head.w_out.weight is not lm.embeddings.weight:
            output_rows = torch.nn.Linear(
                lm.d_model,
                row_count,
                bias=False,
                device=self.device,
                dtype=lm.lm_head.w_out.weight.dtype,
            )
            lm.init_method.init_final_w_out(
                output_rows,
                d_model=lm.d_model,
                std=lm.init_std,
                generator=generator,
            )
            lm.lm_head.w_out.weight.index_copy_(0, row_index, output_rows.weight)

        optim = self._require_optimizer()
        reset_params = {
            name
            for group in optim.param_groups
            for name, param in group["named_params"].items()
            if param is lm.embeddings.weight
            or (reset_output_rows and param is lm.lm_head.w_out.weight)
        }
        optim._copy_model_param_rows_to_main_params(reset_params, token_ids)

    def _resolve_model_checkpoint_key(self, param_name: str, checkpoint_keys) -> Optional[str]:
        checkpoint_key = super()._resolve_model_checkpoint_key(param_name, checkpoint_keys)
        if checkpoint_key is not None:
            return checkpoint_key

        stripped = self._strip_wrapper_prefixes(param_name)
        if stripped.startswith("lm."):
            return super()._resolve_model_checkpoint_key(
                stripped.removeprefix("lm."), checkpoint_keys
            )
        return None

    def _frozen_checkpoint_model_param_state_dict_for_load(self, checkpoint_keys):
        """Load frozen multimodal parameters from stable or native optimizer-main keys.

        Native s002 checkpoints predate ``frozen_model.*`` entries and store every model tensor
        only as a flattened FP32 optimizer main parameter. A parameter frozen by the multimodal
        recipe is absent from the current optimizer state, so explicitly map that native tensor
        onto the frozen model parameter without routing trainable masters through BF16.
        """
        state = super()._frozen_checkpoint_model_param_state_dict_for_load(checkpoint_keys)
        stable_prefix = self._FROZEN_MODEL_PARAM_KEY_PREFIX
        optimizer_owned_anchor_ids = self._optimizer_owned_anchor_param_ids()
        for model_part in self.model_parts:
            for name, param in model_part.named_parameters():
                if param.requires_grad or id(param) in optimizer_owned_anchor_ids:
                    continue
                stable_key = stable_prefix + self._strip_wrapper_prefixes(name)
                if stable_key in state:
                    continue
                checkpoint_key = self._resolve_model_checkpoint_key(name, checkpoint_keys)
                if checkpoint_key is not None and checkpoint_key.endswith(".main"):
                    if checkpoint_key in state:
                        raise RuntimeError(
                            f"Multiple frozen parameters map to checkpoint key '{checkpoint_key}'"
                        )
                    state[checkpoint_key] = param
        return state

    def _resolve_optimizer_checkpoint_key(self, state_key: str, checkpoint_keys) -> Optional[str]:
        checkpoint_key = super()._resolve_optimizer_checkpoint_key(state_key, checkpoint_keys)
        if checkpoint_key is not None:
            return checkpoint_key

        candidates = []
        for prefix in ("model.module.lm.", "model.lm.", "module.lm.", "lm."):
            if state_key.startswith(prefix):
                candidates.append(state_key.replace(prefix, prefix.replace("lm.", ""), 1))
        for candidate in candidates:
            if candidate in checkpoint_keys:
                return candidate
        return None

    def _allow_missing_optimizer_checkpoint_key(self, state_key: str) -> bool:
        return any(
            f".{component}." in state_key or state_key.startswith(f"{component}.")
            for component in ("connector", "vision")
        )


@dataclass
class MultimodalOLMoDDPTrainModuleConfig(OLMoDDPTrainModuleConfig):
    """Configuration for :class:`MultimodalOLMoDDPTrainModule`."""

    freeze_params: Optional[List[str]] = None
    vision_activation_checkpointing: bool = False
    connector_activation_checkpointing: bool = False
    response_logits_only: bool = False
    diagnostics_interval: Optional[int] = None
    train_embedding_rows: Optional[List[int]] = None
    """Embedding rows allowed to receive gradients; all other rows are held fixed."""
    source_loss_mass_targets: Optional[Dict[str, float]] = None
    """Optional expected source loss-mass shares that enable online delivery telemetry."""

    def _build_train_module(self, **kwargs) -> MultimodalOLMoDDPTrainModule:
        return MultimodalOLMoDDPTrainModule(**kwargs)
