"""
Train module for :class:`~olmo_core.nn.vision.MultimodalTransformer`.

Reuses :class:`TransformerTrainModule`'s machinery (microbatching, optimizer
step, scheduler, autocast, state dict, loss accounting) and adds:

- FSDP / DDP support tailored to the composite model — each vision block,
  the connector, and the LM (via its own ``apply_fsdp``) become FSDP units,
  with a final outer wrap on the whole :class:`MultimodalTransformer`.
- The batch carries ``loss_masks: (B, S) float32``. Before computing
  autoregressive labels we convert it to the boolean ``label_mask`` the base
  class understands — that's what implements response-only loss.
- The batch also carries ``images`` and ``pooled_patches_idx`` which flow
  through to ``MultimodalTransformer.forward`` as ``**model_kwargs``.

TP / CP / PP / EP are still out of scope and silently ignored with a warning.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, cast

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from torch.optim import Optimizer

from olmo_core.config import DType
from olmo_core.distributed.parallel import build_world_mesh, get_dp_model_mesh
from olmo_core.distributed.utils import is_distributed
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.vision import MultimodalTransformer
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.train_module.train_module import TrainModule
from olmo_core.train.train_module.transformer.config import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.train_module import TransformerTrainModule
from olmo_core.utils import get_default_device

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

__all__ = [
    "MultimodalTransformerTrainModule",
    "MultimodalTransformerTrainModuleConfig",
]


class MultimodalTransformerTrainModule(TransformerTrainModule):
    """A :class:`TrainModule` for :class:`MultimodalTransformer`.

    Reuses the full :class:`TransformerTrainModule` pipeline (microbatching,
    state dict, autocast, scheduler) and runs the multimodal-specific
    parallelization in ``__init__``. The
    :func:`~olmo_core.train.train_module.transformer.common.parallelize_model`
    helper isn't called because it's :class:`Transformer`-specific; we call
    :meth:`MultimodalTransformer.apply_fsdp` / :meth:`apply_ddp` directly.
    """

    def __init__(
        self,
        model: MultimodalTransformer,
        optim: OptimConfig,
        rank_microbatch_size: int,
        max_sequence_length: int,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        z_loss_multiplier: Optional[float] = None,
        autocast_precision: Optional[torch.dtype] = None,
        max_grad_norm: Optional[float] = None,
        scheduler: Optional[Scheduler] = None,
        device: Optional[torch.device] = None,
        state_dict_save_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        state_dict_load_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        load_key_mapping: Optional[Dict[str, str]] = None,
        label_ignore_index: int = -100,
    ):
        # Skip TransformerTrainModule.__init__ (which calls parallelize_model
        # on a Transformer) and run the base TrainModule init manually.
        TrainModule.__init__(self)

        if rank_microbatch_size % max_sequence_length != 0:
            raise OLMoConfigurationError(
                f"'rank_microbatch_size' ({rank_microbatch_size:,d} tokens) must be divisible by "
                f"'max_sequence_length' ({max_sequence_length:,d} tokens)"
            )

        self.device = device or get_default_device()

        # Build the world mesh if distributed and a DP config was given.
        self.world_mesh = None
        if dp_config is not None:
            if not is_distributed():
                raise OLMoConfigurationError("dp_config is only valid in a distributed setting")
            self.world_mesh = build_world_mesh(dp=dp_config, device_type=self.device.type)

        # Parallelize, then materialize and initialize weights.
        if dp_config is None:
            self.model: MultimodalTransformer = model.to(self.device)  # type: ignore[assignment]
        else:
            assert self.world_mesh is not None
            dp_mesh = get_dp_model_mesh(self.world_mesh)
            param_dtype = (
                dp_config.param_dtype.as_pt() if dp_config.param_dtype is not None else None
            )
            reduce_dtype = dp_config.reduce_dtype.as_pt()
            from olmo_core.distributed.parallel import DataParallelType

            if dp_config.name in (DataParallelType.fsdp, DataParallelType.hsdp):
                model.apply_fsdp(
                    dp_mesh=dp_mesh,
                    param_dtype=param_dtype,
                    reduce_dtype=reduce_dtype,
                    wrapping_strategy=dp_config.wrapping_strategy,
                    prefetch_factor=dp_config.prefetch_factor,
                )
            elif dp_config.name == DataParallelType.ddp:
                model.apply_ddp(dp_mesh=dp_mesh, param_dtype=param_dtype)
            else:
                raise NotImplementedError(dp_config.name)
            self.model = model
            self.model.init_weights(
                max_seq_len=max_sequence_length,
                max_local_microbatch_size=rank_microbatch_size,
                device=self.device,
                world_mesh=self.world_mesh,
            )

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

        log.info("Building optimizer for multimodal model...")
        self.optim: Optimizer = optim.build(self.model, strict=True)

    # ------------------------------------------------------------------
    # Batch handling: loss_masks → label_mask + image kwargs
    # ------------------------------------------------------------------

    def _convert_loss_masks(self, batch: Dict[str, Any]) -> None:
        """In-place: convert ``loss_masks`` to ``label_mask`` so the base
        class's :func:`get_labels` masks non-response positions."""
        if "loss_masks" in batch and "label_mask" not in batch:
            loss_masks = batch.pop("loss_masks")
            if not isinstance(loss_masks, torch.Tensor):
                loss_masks = torch.as_tensor(loss_masks)
            batch["label_mask"] = loss_masks.bool()

    def _move_image_kwargs_to_device(self, batch: Dict[str, Any]) -> None:
        """Move multimodal-specific tensors to ``self.device``.

        The base ``Transformer._prepare_inputs`` moves ``input_ids`` and
        ``labels`` but doesn't know about our ``images`` /
        ``pooled_patches_idx``. Moving them here keeps the train module
        agnostic to whatever device the data loader produced batches on."""
        from olmo_core.utils import move_to_device

        for key in ("images", "pooled_patches_idx"):
            if key in batch and isinstance(batch[key], torch.Tensor):
                batch[key] = move_to_device(batch[key], self.device)

    def pre_train(self):
        """Validate sizing in examples, not tokens.

        ``TransformerTrainModule.pre_train`` checks
        ``global_batch_size % (rank_microbatch_size * dp_ws) == 0``, assuming
        both are in tokens (text loaders' convention). Our multimodal data
        loader counts ``global_batch_size`` in **examples**, while
        ``rank_microbatch_size`` is still in tokens. We translate to a
        per-rank-instance check instead.
        """
        from olmo_core.distributed.utils import get_world_size

        dp_ws = get_world_size(self.trainer.dp_process_group)
        rank_examples = self.trainer.global_batch_size // dp_ws
        microbatch_instances = max(1, self.rank_microbatch_size // self.max_sequence_length)
        if rank_examples % microbatch_instances != 0:
            raise OLMoConfigurationError(
                f"global batch size ({self.trainer.global_batch_size} examples) divided by "
                f"DP world size ({dp_ws}) gives {rank_examples} examples per rank, which is "
                f"not divisible by microbatch instances ({microbatch_instances} = "
                f"{self.rank_microbatch_size} tokens / {self.max_sequence_length} max_seq_len)"
            )

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        self._convert_loss_masks(batch)
        self._move_image_kwargs_to_device(batch)
        super().train_batch(batch, dry_run=dry_run)

    def eval_batch(self, batch, labels=None):
        self._convert_loss_masks(batch)
        self._move_image_kwargs_to_device(batch)
        return super().eval_batch(batch, labels)

    def _prepare_batch(
        self,
        batch: Dict[str, Any],
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Pop ``label_mask`` from the kwargs that flow through to the model.

        ``MultimodalTransformer.forward`` doesn't accept ``label_mask`` and
        we've already used it to compute ``labels`` upstream.
        """
        input_ids, labels, model_kwargs = super()._prepare_batch(batch, labels)
        model_kwargs.pop("label_mask", None)
        return input_ids, labels, model_kwargs


@dataclass
class MultimodalTransformerTrainModuleConfig(TransformerTrainModuleConfig):
    """Configuration for :class:`MultimodalTransformerTrainModule`.

    Inherits every field from :class:`TransformerTrainModuleConfig` for API
    symmetry. :attr:`dp_config` is honored (FSDP / HSDP / DDP); other
    parallelism configs (``tp_config``, ``cp_config``, ``pp_config``,
    ``ep_config``) are silently ignored with a warning. ``compile_model``
    is a no-op for now.
    """

    def build(  # type: ignore[override]
        self,
        model: MultimodalTransformer,
        device: Optional[torch.device] = None,
    ) -> MultimodalTransformerTrainModule:
        """Instantiate the train module."""
        if self.pp_config is not None:
            raise NotImplementedError(
                "Pipeline parallelism is not yet supported for MultimodalTransformer"
            )
        if any(cfg is not None for cfg in (self.tp_config, self.cp_config, self.ep_config)):
            log.warning(
                "TP/CP/EP configs are not yet honored for MultimodalTransformer; "
                "proceeding with DP only."
            )
        if self.compile_model:
            log.warning(
                "compile_model is not yet supported for MultimodalTransformer; "
                "proceeding without torch.compile()."
            )
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # Strip fields that don't apply to the multimodal train module.
        for unsupported in (
            "compile_model",
            "float8_config",
            "tp_config",
            "cp_config",
            "ep_config",
            "ac_config",
            "pp_config",
        ):
            kwargs.pop(unsupported, None)
        # dp_config goes through as a nested object, not flattened by as_dict.
        kwargs["dp_config"] = self.dp_config
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)
        if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)
        return MultimodalTransformerTrainModule(model=model, device=device, **kwargs)
