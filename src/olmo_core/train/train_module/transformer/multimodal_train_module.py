"""Train module for Molmo2 :class:`~olmo_core.nn.vision.MultimodalLM` stage-1 training.

:class:`MultimodalTransformerTrainModule` extends :class:`TransformerTrainModule` for a
:class:`~olmo_core.nn.vision.MultimodalLM` (a plain ``nn.Module``, *not* a
:class:`~olmo_core.nn.transformer.Transformer`). It differs from the base module in
three ways:

1. The model is **not** routed through ``parallelize_model`` (which requires a
   ``Transformer``); only single-GPU or DDP is supported here. FSDP/TP/CP/PP/EP of the
   multimodal model are out of scope (a later effort can shard ``model.lm``).
2. The loss uses **float per-token** ``loss_masks`` (response-only, ``root_subsegments``
   weighted by the data pipeline) via
   :func:`~olmo_core.nn.functional.weighted_cross_entropy_loss`, reproducing mm_olmo.
3. The loss divisor is the **global** sum of ``loss_masks`` (all-reduced, divided by the
   DP world size) so that, after DDP gradient averaging, the effective normalization is
   the global loss-weight — matching mm_olmo's ``BatchDivisor.global_batch``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from torch.nn.parallel import DistributedDataParallel as DDP

from olmo_core.config import DType
from olmo_core.data.utils import split_batch
from olmo_core.distributed.parallel import build_world_mesh, get_dp_process_group
from olmo_core.distributed.utils import get_local_tensor, get_world_size, is_distributed
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.functional import weighted_cross_entropy_loss
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.utils import get_default_device, move_to_device, warn_once

from ...common import ReduceType
from ..config import TrainModuleConfig
from ..train_module import EvalBatchSpec, TrainModule
from .config import TransformerDataParallelConfig
from .train_module import TransformerTrainModule

log = logging.getLogger(__name__)

__all__ = ["MultimodalTransformerTrainModule", "MultimodalTransformerTrainModuleConfig"]


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
        label_ignore_index: int = -100,
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
        if dp_config is not None and dp_config.name not in ("ddp",):
            raise OLMoConfigurationError(
                "MultimodalTransformerTrainModule only supports DDP data parallelism "
                f"(got dp_config.name={dp_config.name!r}); FSDP/TP/CP/PP/EP of the "
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
        if compile_model:
            log.info("Compiling model.lm ...")
            model.lm = torch.compile(model.lm)  # type: ignore[assignment]
        # NOTE: keep ``model`` unwrapped for now. The optimizer is built below on the
        # unwrapped module so group-override patterns (e.g. ``connector.*``) match
        # parameter names without DDP's ``module.`` prefix; DDP is applied afterwards.
        self.model = model
        self._model_mode = None

        self._dp_config = dp_config
        self._cp_config = None
        self._tp_config = None
        self._ep_config = None
        self.label_ignore_index = label_ignore_index
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

        # Build the optimizer on the *unwrapped* model so group-override patterns like
        # "connector.*" match parameter names without DDP's "module." prefix.
        log.info("Building optimizer...")
        self.optim = optim.build(self.model, strict=True)

        # Now wrap in DDP. The optimizer holds references to the same Parameter objects,
        # so gradient all-reduce (DDP) and the optimizer step stay consistent.
        if self.world_mesh is not None:
            self.model = DDP(self.model, process_group=get_dp_process_group(self.world_mesh))

    # -- helpers to reach the underlying MultimodalLM / its Transformer ----------

    @property
    def _multimodal(self) -> torch.nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

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
        return input_ids, labels, loss_masks, batch

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

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                input_ids, labels, mb_loss_masks, model_kwargs = self._prepare_batch(micro_batch)
                assert labels is not None
                mb_loss_masks = mb_loss_masks.to(self.device).float()

                # MultimodalLM with ``labels=None`` returns raw logits ``(B, S, V)``.
                # ``labels`` / ``loss_masks`` are already next-token-aligned (shifted) by
                # the data pipeline, so no additional shift here.
                with self._model_forward_context():
                    logits = self.model(input_ids, labels=None, **model_kwargs)
                vocab_size = logits.shape[-1]
                flat_logits = logits.reshape(-1, vocab_size)
                flat_labels = labels.to(self.device).reshape(-1)
                flat_weights = mb_loss_masks.reshape(-1)
                # Mask out non-loss positions from the CE target for safety.
                flat_labels = torch.where(
                    flat_weights > 0, flat_labels, flat_labels.new_full((), self.label_ignore_index)
                )

                ce_loss, z_loss = weighted_cross_entropy_loss(
                    flat_logits,
                    flat_labels,
                    flat_weights,
                    ignore_index=self.label_ignore_index,
                    compute_z_loss=self.z_loss_multiplier is not None,
                    z_loss_multiplier=self.z_loss_multiplier or 1e-4,
                )

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
            for metric_name, (metric_val, reduction) in self._lm.compute_auxiliary_metrics(
                reset=True
            ).items():
                self.record_metric(metric_name, metric_val, reduction, namespace="train")

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
    z_loss_multiplier: Optional[float] = None
    autocast_precision: Optional[DType] = None
    label_ignore_index: int = -100
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
