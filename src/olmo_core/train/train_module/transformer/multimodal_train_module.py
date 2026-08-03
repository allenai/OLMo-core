"""Train module for Molmo2 :class:`~olmo_core.nn.vision.MultimodalLM` stage-1 training.

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

import logging
import os
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd

from olmo_core.config import DType
from olmo_core.data.utils import split_batch
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
from olmo_core.optim import OptimConfig
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


class MultimodalTransformerTrainModule(TransformerTrainModule):
    """A :class:`TrainModule` for :class:`~olmo_core.nn.vision.MultimodalLM` stage-1 training."""

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
        state_dict_save_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        state_dict_load_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        load_key_mapping: Optional[Dict[str, str]] = None,
    ):
        # NOTE: deliberately bypass ``TransformerTrainModule.__init__`` (which calls
        # ``parallelize_model``, requiring a ``Transformer``); call the grandparent.
        TrainModule.__init__(self)

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

        self.device = device or get_default_device()
        self.world_mesh = None
        if is_distributed():
            self.world_mesh = build_world_mesh(dp=dp_config, device_type=self.device.type)
        elif dp_config is not None:
            raise OLMoConfigurationError(
                "Training parallelism configs are only valid for distributed training"
            )

        # Freeze parameters (e.g. the vision encoder for stage 1) before building the
        # optimizer so frozen params are excluded from optimizer groups.
        self.freeze_params = freeze_params or []
        n_frozen = 0
        for name, p in model.named_parameters():
            if any(fnmatch(name, pat) for pat in self.freeze_params):
                p.requires_grad_(False)
                n_frozen += 1
        if self.freeze_params:
            log.info(f"Froze {n_frozen} parameter tensors matching {self.freeze_params}")

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

        # Apply data parallelism IN-PLACE *before* building the optimizer: composable
        # DDP/FSDP keep the model's type, attributes, and (prefix-free) parameter names,
        # and FSDP additionally needs the optimizer built on the sharded DTensor params.
        if self.world_mesh is not None:
            self._parallelize(dp_config)

        log.info("Building optimizer...")
        self.optim = optim.build(self.model, strict=True)

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
        return input_ids, labels, loss_masks, batch

    def _set_model_mode(self, mode: Literal["train", "eval"]):
        super()._set_model_mode(mode)
        if mode == "train" and any(fnmatch(name, "vision.*") for name in self.freeze_params):
            self._multimodal.vision.eval()

    def _log_batch_sources(self, batch: Dict[str, Any], local_weight: torch.Tensor) -> None:
        """Log per-rank packed source names when verbose diagnostics are enabled."""
        if not _mm_train_verbose_logs():
            return
        sources = batch.get("pack_source_names")
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

    # -- training step -----------------------------------------------------------

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        self._set_model_mode("train")

        # Global loss-weight divisor (mm_olmo BatchDivisor.global_batch): the sum of
        # positive loss weights over the whole global batch, divided by DP world size.
        # After DDP averages gradients across ranks, the effective divisor is the global
        # weight. For a single rank this is just the local weight sum.
        loss_masks = batch["loss_masks"].to(self.device).float()
        local_weight = (loss_masks * (loss_masks > 0)).sum()
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
                weight_total += get_local_tensor((flat_weights > 0).sum().detach()).float()
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

        # Record a per-weighted-token CE loss (comparable across steps).
        mean_ce = ce_batch_loss / torch.clamp(local_weight, min=1.0)
        self.record_ce_loss(mean_ce, ReduceType.mean)
        if z_batch_loss is not None:
            assert self.z_loss_multiplier is not None
            mean_z = z_batch_loss / torch.clamp(local_weight, min=1.0)
            self.record_metric("Z loss", mean_z, ReduceType.mean, namespace="train")

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
        if self.max_grad_norm is not None:
            grad_norm = self._clip_grad_norm(self.max_grad_norm)
            self.trainer.record_metric(
                "total grad norm", grad_norm, reduce_type=None, namespace="optim"
            )

        if self.scheduler is not None:
            for group_idx, group in enumerate(self.optim.param_groups):
                new_lr = self.scheduler.set_lr(group, self.trainer)
                self.trainer.record_metric(f"LR (group {group_idx})", new_lr, namespace="optim")

        self.optim.step()

        if hasattr(self._lm, "post_optim_step"):
            self._lm.post_optim_step()

    def eval_batch(self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None):
        raise NotImplementedError(
            "In-loop evaluation is not implemented for MultimodalTransformerTrainModule "
            "(stage-1 training runs without in-loop eval)."
        )

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
    state_dict_save_opts: Optional[Dict[str, Any]] = None
    state_dict_load_opts: Optional[Dict[str, Any]] = None
    load_key_mapping: Optional[Dict[str, str]] = None

    def build(
        self, model: torch.nn.Module, device: Optional[torch.device] = None
    ) -> "MultimodalTransformerTrainModule":
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**save_opts)
        if (load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**load_opts)
        return MultimodalTransformerTrainModule(model=model, device=device, **kwargs)


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
        **kwargs,
    ):
        from olmo_core.nn.vision import MultimodalOLMoDDPModel

        if not isinstance(model, MultimodalOLMoDDPModel):
            raise TypeError(
                f"{type(self).__name__} requires MultimodalOLMoDDPModel, "
                f"got {type(model).__name__}"
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
        if vision_activation_checkpointing:
            model.vision.apply_activation_checkpointing()
            log.info("Applied activation checkpointing to the vision encoder")
        if connector_activation_checkpointing:
            model.connector.apply_activation_checkpointing()
            log.info("Applied activation checkpointing to the vision connector")
        self.response_logits_only = response_logits_only
        super().__init__(model, *args, **kwargs)

    @property
    def multimodal_model(self):
        model = self.model_parts[0]
        return getattr(model, "module", model)

    def _prepare_batch(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        input_ids, labels, model_kwargs = super()._prepare_batch(batch, labels)
        if self.response_logits_only:
            model_kwargs["response_logits_only"] = True
        return input_ids, labels, model_kwargs

    def load_molmo2_vision_state_dict(self, hf_state_dict: Dict[str, torch.Tensor]) -> None:
        """Strictly load the Molmo2 vision tower, leaving the connector untouched."""
        from olmo_core.nn.vision import molmo2_hf_state_dict_to_vision

        model = self.multimodal_model
        vision_state = molmo2_hf_state_dict_to_vision(hf_state_dict, model.cfg.vision)
        model.vision.load_state_dict(vision_state, strict=True)

    @torch.no_grad()
    def reset_image_token_rows(self, token_ids: List[int], *, seed: int) -> None:
        """Initialize newly assigned image-token rows and update optimizer main state."""
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

        if lm.lm_head.w_out.weight is not lm.embeddings.weight:
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
            if param is lm.embeddings.weight or param is lm.lm_head.w_out.weight
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
        return ".connector." in state_key or state_key.startswith("connector.")


@dataclass
class MultimodalOLMoDDPTrainModuleConfig(OLMoDDPTrainModuleConfig):
    """Configuration for :class:`MultimodalOLMoDDPTrainModule`."""

    freeze_params: Optional[List[str]] = None
    vision_activation_checkpointing: bool = False
    connector_activation_checkpointing: bool = False
    response_logits_only: bool = False

    def _build_train_module(self, **kwargs) -> MultimodalOLMoDDPTrainModule:
        return MultimodalOLMoDDPTrainModule(**kwargs)
