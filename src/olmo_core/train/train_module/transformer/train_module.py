import contextlib
import logging
import math
import os
import time
from dataclasses import replace
from functools import cached_property, lru_cache
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.checkpoint import (
    merge_state_dicts,
    prune_state_dict,
    swap_param_keys,
)
from olmo_core.distributed.parallel import (
    DataParallelType,
    build_world_mesh,
    get_dp_process_group,
)
from olmo_core.distributed.utils import (
    get_local_tensor,
    get_reduce_divide_factor,
    get_world_size,
    is_distributed,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.transformer import Transformer
from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode
from olmo_core.optim import OptimConfig, OptimGroupOverride, SkipStepOptimizer
from olmo_core.optim.scheduler import Scheduler
from olmo_core.utils import (
    gc_cuda,
    get_default_device,
    log_once,
    move_to_device,
    warn_once,
)

from ...common import ReduceType
from ..train_module import EvalBatchSpec, TrainModule
from .common import parallelize_model
from .config import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
    TransformerTensorParallelConfig,
)

log = logging.getLogger(__name__)


class TransformerTrainModule(TrainModule):
    """
    A :class:`TrainModule` for any :class:`~olmo_core.nn.transformer.Transformer` model
    implementation provided by this library.

    .. tip::
        Use the :class:`TransformerTrainModuleConfig` to easily configure and build
        :class:`TransformerTrainModule` instances.

    :param model: The :class:`~olmo_core.nn.transformer.Transformer` model to train.
    :param optim: The corresponding optimizer config.
    :param rank_microbatch_size: The microbatch size *in tokens* per rank,
        i.e. the number of tokens to process at a time from each rank.

        .. note:: This must evenly divide into the global batch size by a factor of the data
            parallel world size. If this is less than the global batch divided by the data
            parallel world size then gradient accumulation is used.
    :param max_sequence_length: The maximum expected sequence length during training and evaluation.
    :param compile_model: Whether to compile to the model.
    :param float8_config: Float8 configuration for the model.
    :param dp_config: Data parallel configuration for the model.
    :param tp_config: Tensor parallel configuration for the model.
    :param cp_config: Context parallel configuration for the model.
    :param ac_config: Activation checkpointing configuration for the model.
    :param z_loss_multiplier: Use Z-loss with this multiplier.
    :param autocast_precision: Enable AMP with this data type.
    :param max_grad_norm: Clip gradient norms to this value.
    :param scheduler: Optional learning rate scheduler for the optimizer.
    :param device: The device to train on.
    :param state_dict_save_opts: Can be used to override the state dict options used
        when saving a checkpoint.
    :param state_dict_load_opts: Can be used to override the state dict options used
        when loading a checkpoint.
    :param load_key_mapping: Can be used to load a checkpoint where certain parameter have different names.
        This dictionary should map current keys to keys in the checkpoint to be loaded.
    """

    def __init__(
        self,
        model: Transformer,
        optim: OptimConfig,
        rank_microbatch_size: int,
        max_sequence_length: int,
        eval_rank_microbatch_size: Optional[int] = None,
        compile_model: bool = False,
        float8_config: Optional[Float8Config] = None,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        tp_config: Optional[TransformerTensorParallelConfig] = None,
        cp_config: Optional[TransformerContextParallelConfig] = None,
        ep_config: Optional[TransformerExpertParallelConfig] = None,
        ac_config: Optional[TransformerActivationCheckpointingConfig] = None,
        z_loss_multiplier: Optional[float] = None,
        soft_ce_alpha_start: Optional[float] = None,
        soft_ce_alpha_ramp_fraction: float = 0.5,
        soft_ce_truncation: str = "renormalize",
        poe_lambda: Optional[float] = None,
        poe_lambda_learnable: bool = False,
        poe_lambda_lr: Optional[float] = None,
        poe_lambda_decay_to_zero_windows: Optional[List[Tuple[int, int]]] = None,
        poe_ngram_table_dir: Optional[str] = None,
        poe_ngram_K: int = 16,
        poe_ngram_N_max: int = 5,
        poe_sb_table_dir: Optional[str] = None,
        poe_sb_alpha: float = 0.4,
        poe_sb_N_max: int = 5,
        poe_sb_dolma2_vocab_size: int = 100352,
        poe_sb_max_order2_continuations: Optional[int] = None,
        poe_sb_max_order_continuations: Optional[Dict[int, int]] = None,
        poe_sb_min_order_counts: Optional[Dict[int, int]] = None,
        poe_sb_index_access: str = "mmap",
        poe_sb_mirror_to_shm: bool = True,
        poe_sb_lookup_threads: int = 1,
        poe_sb_eval_lookup_threads: Optional[int] = None,
        poe_sb_topk_uniform_residual_k: Optional[int] = None,
        poe_sb_recursive_topk_uniform_residual_k: Optional[int] = None,
        autocast_precision: Optional[torch.dtype] = None,
        max_grad_norm: Optional[float] = None,
        scheduler: Optional[Scheduler] = None,
        device: Optional[torch.device] = None,
        state_dict_save_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        state_dict_load_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        load_key_mapping: Optional[Dict[str, str]] = None,
        label_ignore_index: int = -100,
    ):
        super().__init__()

        # Validate some options.
        if rank_microbatch_size % max_sequence_length != 0:
            raise OLMoConfigurationError(
                f"'rank_microbatch_size' ({rank_microbatch_size:,d} tokens) must be divisible by "
                f"'max_sequence_length' ({max_sequence_length:,d} tokens)"
            )
        if (
            eval_rank_microbatch_size is not None
            and eval_rank_microbatch_size % max_sequence_length != 0
        ):
            raise OLMoConfigurationError(
                f"'eval_rank_microbatch_size' ({eval_rank_microbatch_size:,d} tokens) must be "
                f"divisible by 'max_sequence_length' ({max_sequence_length:,d} tokens)"
            )

        # Build world mesh.
        self.device = device or get_default_device()
        self.world_mesh: Optional[DeviceMesh] = None
        if is_distributed():
            self.world_mesh = build_world_mesh(
                dp=dp_config, tp=tp_config, cp=cp_config, ep=ep_config, device_type=self.device.type
            )
            log.info(f"Data parallel world size = {get_world_size(self.dp_process_group):,d}")
        elif (
            dp_config is not None
            or tp_config is not None
            or ep_config is not None
            or cp_config is not None
        ):
            raise OLMoConfigurationError(
                "Training parallelism configs are only valid for distributed training"
            )

        if (
            ac_config is not None
            and ac_config.mode == TransformerActivationCheckpointingMode.budget
            and not compile_model
        ):
            raise OLMoConfigurationError(
                "Activation checkpointing with 'budget' mode requires compilation to be enabled"
            )

        # Register learned PoE lambda before parallelization. Put it on the
        # LM head, which is used in the model forward, so FSDP manages it as
        # an ordinary trainable model parameter.
        self._poe_lambda_log_name = "poe_lambda_log"
        self._poe_lambda_log_param_name = f"lm_head.{self._poe_lambda_log_name}"
        if poe_lambda_learnable and poe_lambda is not None:
            if poe_lambda <= 0:
                raise OLMoConfigurationError(f"poe_lambda must be positive, got {poe_lambda}")
            if model.lm_head is None:
                raise OLMoConfigurationError("poe_lambda_learnable requires a model LM head")
            log_lambda = torch.log(
                torch.tensor([float(poe_lambda)], dtype=torch.float32, device=self.device)
            )
            model.lm_head.register_parameter(self._poe_lambda_log_name, nn.Parameter(log_lambda))

        # Parallelize model.
        self.model = parallelize_model(
            model,
            world_mesh=self.world_mesh,
            device=self.device,
            max_sequence_length=max_sequence_length,
            rank_microbatch_size=rank_microbatch_size,
            compile_model=compile_model,
            float8_config=float8_config,
            dp_config=dp_config,
            tp_config=tp_config,
            cp_config=cp_config,
            ep_config=ep_config,
            ac_config=ac_config,
        )
        # `parallelize_model()` materializes and initializes parameters. Since
        # the learned PoE scalar is registered before parallelization so FSDP
        # can own it, restore its user-requested initialization after that
        # model-wide init pass and before the optimizer captures parameters.
        if poe_lambda_learnable and poe_lambda is not None:
            with torch.no_grad():
                self._poe_lambda_log_param().fill_(math.log(float(poe_lambda)))
        self._model_mode: Optional[Literal["train", "eval"]] = None

        self._dp_config = dp_config
        self._cp_config = cp_config
        self._tp_config = tp_config
        self._ep_config = ep_config
        self.label_ignore_index = label_ignore_index
        self.z_loss_multiplier = z_loss_multiplier
        self.soft_ce_alpha_start = soft_ce_alpha_start
        self.soft_ce_alpha_ramp_fraction = soft_ce_alpha_ramp_fraction
        if soft_ce_truncation not in ("renormalize", "uniform_residual"):
            raise OLMoConfigurationError(
                f"soft_ce_truncation must be 'renormalize' or 'uniform_residual', "
                f"got {soft_ce_truncation!r}"
            )
        self.soft_ce_truncation = soft_ce_truncation
        self.poe_lambda = poe_lambda
        self.poe_lambda_learnable = bool(poe_lambda_learnable)
        self.poe_lambda_lr = poe_lambda_lr
        self.poe_lambda_decay_to_zero_windows = poe_lambda_decay_to_zero_windows
        if self.soft_ce_alpha_start is not None and self.poe_lambda is not None:
            raise OLMoConfigurationError(
                "soft_ce_alpha_start and poe_lambda are mutually exclusive — "
                "they correspond to two different ways of injecting the ngram "
                "signal (soft-cross-entropy auxiliary loss vs product-of-experts "
                "logit bias). Pick one."
            )
        if self.soft_ce_alpha_start is not None:
            if not (0.0 <= self.soft_ce_alpha_start <= 1.0):
                raise OLMoConfigurationError(
                    f"soft_ce_alpha_start must be in [0, 1], got {self.soft_ce_alpha_start}"
                )
            if not (0.0 < self.soft_ce_alpha_ramp_fraction <= 1.0):
                raise OLMoConfigurationError(
                    f"soft_ce_alpha_ramp_fraction must be in (0, 1], got {self.soft_ce_alpha_ramp_fraction}"
                )
            if tp_config is not None or cp_config is not None:
                raise OLMoConfigurationError(
                    "soft-CE auxiliary loss is not yet supported with TP or CP"
                )
        if self.poe_lambda is not None:
            if self.poe_lambda <= 0:
                raise OLMoConfigurationError(
                    f"poe_lambda must be positive, got {self.poe_lambda}"
                )
            if tp_config is not None or cp_config is not None:
                raise OLMoConfigurationError(
                    "PoE training is not yet supported with TP or CP"
                )
            if poe_ngram_table_dir is None and poe_sb_table_dir is None:
                raise OLMoConfigurationError(
                    "poe_lambda requires either poe_ngram_table_dir (KN-smoothed) "
                    "or poe_sb_table_dir (stupid-backoff) to be set so the eval "
                    "path can apply the same ngram bias as the train path"
                )
            if poe_ngram_table_dir is not None and poe_sb_table_dir is not None:
                raise OLMoConfigurationError(
                    "poe_ngram_table_dir and poe_sb_table_dir are mutually exclusive "
                    "— pick one of the two PoE ngram backends"
                )
        elif self.poe_lambda_learnable:
            raise OLMoConfigurationError("poe_lambda_learnable requires poe_lambda to be set")
        if self.poe_lambda_lr is not None and not self.poe_lambda_learnable:
            raise OLMoConfigurationError("poe_lambda_lr is only valid when poe_lambda_learnable=True")
        if self.poe_lambda_lr is not None and self.poe_lambda_lr <= 0:
            raise OLMoConfigurationError(f"poe_lambda_lr must be positive, got {self.poe_lambda_lr}")
        if (
            poe_sb_topk_uniform_residual_k is not None
            and poe_sb_recursive_topk_uniform_residual_k is not None
        ):
            raise OLMoConfigurationError(
                "poe_sb_topk_uniform_residual_k and "
                "poe_sb_recursive_topk_uniform_residual_k are mutually exclusive"
            )
        if self.poe_lambda_decay_to_zero_windows:
            prev_end = -1
            for start, end in self.poe_lambda_decay_to_zero_windows:
                if start <= 0 or end < start:
                    raise OLMoConfigurationError(
                        "poe_lambda_decay_to_zero_windows entries must have positive "
                        f"start <= end, got {(start, end)}"
                    )
                if start <= prev_end:
                    raise OLMoConfigurationError(
                        "poe_lambda_decay_to_zero_windows must be sorted and non-overlapping"
                    )
                prev_end = end
        self.poe_ngram_table_dir = poe_ngram_table_dir
        self.poe_ngram_K = int(poe_ngram_K)
        self.poe_ngram_N_max = int(poe_ngram_N_max)
        self.poe_sb_table_dir = poe_sb_table_dir
        self.poe_sb_alpha = float(poe_sb_alpha)
        self.poe_sb_N_max = int(poe_sb_N_max)
        self.poe_sb_dolma2_vocab_size = int(poe_sb_dolma2_vocab_size)
        self.poe_sb_max_order2_continuations = poe_sb_max_order2_continuations
        self.poe_sb_max_order_continuations = poe_sb_max_order_continuations
        self.poe_sb_min_order_counts = poe_sb_min_order_counts
        self.poe_sb_index_access = poe_sb_index_access
        self.poe_sb_mirror_to_shm = bool(poe_sb_mirror_to_shm)
        self.poe_sb_lookup_threads = int(poe_sb_lookup_threads)
        self.poe_sb_eval_lookup_threads = (
            int(poe_sb_eval_lookup_threads)
            if poe_sb_eval_lookup_threads is not None
            else self.poe_sb_lookup_threads
        )
        self.poe_sb_topk_uniform_residual_k = poe_sb_topk_uniform_residual_k
        self.poe_sb_recursive_topk_uniform_residual_k = (
            poe_sb_recursive_topk_uniform_residual_k
        )
        # Lazy: instantiated on first eval_batch call (per process), so we
        # don't open the mmap on the main coordinator rank that may never
        # actually run an eval.
        self._poe_eval_ngram_source = None
        # SB-side lazy state: reader (CPU) + unigram_floor on the model device.
        self._poe_sb_reader = None
        self._poe_sb_unigram_floor_dev: Optional[torch.Tensor] = None
        self._poe_sb_eval_bias_calls = 0
        self.rank_microbatch_size = rank_microbatch_size
        self.eval_rank_microbatch_size = eval_rank_microbatch_size or rank_microbatch_size
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

        if self.poe_lambda_learnable:
            assert self.poe_lambda is not None
            poe_lambda_opts: Dict[str, Any] = {"weight_decay": 0.0}
            if self.poe_lambda_lr is not None:
                poe_lambda_opts["lr"] = self.poe_lambda_lr
                poe_lambda_opts["initial_lr"] = self.poe_lambda_lr
            group_overrides = list(optim.group_overrides or [])
            group_overrides.append(
                OptimGroupOverride(params=[self._poe_lambda_log_param_name], opts=poe_lambda_opts)
            )
            optim = replace(optim, group_overrides=group_overrides)

        # Build optimizer(s).
        log.info("Building optimizer...")
        self.optim: Optimizer = optim.build(self.model, strict=True)

    def _poe_lambda_decay_to_zero_multiplier(self) -> float:
        """Multiplicative schedule for anneal-time PoE ablations.

        During configured inclusive step windows, the effective PoE weight
        decays linearly from the learned/static lambda to zero. The learned
        lambda parameter is not mutated or optimized during these windows, so
        the next training period can resume from the value learned before the
        decay window instead of fighting the scheduled multiplier.
        """
        windows = self.poe_lambda_decay_to_zero_windows
        if not windows:
            return 1.0
        step = self.trainer.global_step
        for start, end in windows:
            if step < start:
                return 1.0
            if start <= step <= end:
                if end == start:
                    return 0.0
                return max(0.0, min(1.0, (end - step) / (end - start)))
        return 1.0

    def _poe_lambda_decay_to_zero_active(self) -> bool:
        """Whether the current trainer step is inside a lambda-decay window."""
        windows = self.poe_lambda_decay_to_zero_windows
        if not windows:
            return False
        step = self.trainer.global_step
        return any(start <= step <= end for start, end in windows)

    def _effective_poe_lambda(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        assert self.poe_lambda is not None
        if self.poe_lambda_learnable:
            lambda_log = self._poe_lambda_log_param()
            base = torch.exp(lambda_log).squeeze().to(dtype=dtype)
        else:
            base = torch.tensor(float(self.poe_lambda), device=self.device, dtype=dtype)
        if self.poe_lambda_decay_to_zero_windows:
            multiplier = torch.tensor(
                self._poe_lambda_decay_to_zero_multiplier(),
                device=base.device,
                dtype=dtype,
            )
            return base * multiplier
        return base

    def _effective_poe_lambda_for_logging(self) -> Union[float, torch.Tensor]:
        if self.poe_lambda_learnable or self.poe_lambda_decay_to_zero_windows:
            return self._effective_poe_lambda().detach().squeeze()
        assert self.poe_lambda is not None
        return float(self.poe_lambda)

    def _base_poe_lambda_for_logging(self) -> Union[float, torch.Tensor]:
        if self.poe_lambda_learnable:
            return torch.exp(self._poe_lambda_log_param()).detach().squeeze()
        assert self.poe_lambda is not None
        return float(self.poe_lambda)

    def _poe_lambda_tensor(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return PoE lambda as a tensor, with compatibility for bound test stubs."""
        if hasattr(self, "_effective_poe_lambda"):
            return self._effective_poe_lambda(dtype=dtype)
        assert self.poe_lambda is not None
        device = getattr(self, "device", torch.device("cpu"))
        return torch.tensor(float(self.poe_lambda), device=device, dtype=dtype)

    def _poe_lambda_log_for_logging(self) -> torch.Tensor:
        return self._poe_lambda_log_param().detach().squeeze()

    @staticmethod
    def _scalar_metric_tensor(value: torch.Tensor) -> torch.Tensor:
        local_value = get_local_tensor(value.detach())
        if local_value.numel() == 0:
            return torch.zeros((), device=local_value.device, dtype=local_value.dtype)
        return local_value.reshape(-1)[0]

    def _poe_lambda_log_param(self) -> nn.Parameter:
        lm_head = getattr(self.model, "lm_head", None)
        param = (
            getattr(lm_head, self._poe_lambda_log_name)
            if lm_head is not None and hasattr(lm_head, self._poe_lambda_log_name)
            else getattr(self.model, self._poe_lambda_log_name)
        )
        assert isinstance(param, nn.Parameter)
        return param

    @property
    def dp_process_group(self) -> Optional[dist.ProcessGroup]:
        return None if self.world_mesh is None else get_dp_process_group(self.world_mesh)

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        return EvalBatchSpec(
            self.eval_rank_microbatch_size,
            max_sequence_length=self.max_sequence_length,
            #  fixed_sequence_length=self.tp_enabled,
        )

    @property
    def dp_config(self) -> Optional[TransformerDataParallelConfig]:
        return self._dp_config

    @property
    def tp_enabled(self) -> bool:
        return self._tp_config is not None

    @property
    def cp_enabled(self) -> bool:
        return self._cp_config is not None

    @property
    def ep_enabled(self) -> bool:
        return self._ep_config is not None

    @cached_property
    def world_size(self) -> int:
        return get_world_size()

    @cached_property
    def _reduce_divide_factor(self) -> float:
        return get_reduce_divide_factor(self.world_size)

    def pre_train(self):
        # Validate batch size.
        # NOTE: we run this in `pre_train()` instead of, say, `on_attach()` because callbacks
        # like `BatchSizeScheduler` may change the global batch size after the module is attached.
        dp_ws = get_world_size(self.trainer.dp_process_group)
        if self.trainer.global_batch_size % (self.rank_microbatch_size * dp_ws) != 0:
            raise OLMoConfigurationError(
                f"global batch size ({self.trainer.global_batch_size:,d}) must be divisible by "
                f"micro-batch size ({self.rank_microbatch_size:,d}) x DP world size ({dp_ws})"
            )

    def state_dict(self, *, optim: Optional[bool] = None) -> Dict[str, Any]:
        if optim is None:
            optim = True
        return self._get_state_dict(self.state_dict_save_opts, optim=optim)

    def state_dict_to_load(
        self, metadata: Metadata, *, optim: Optional[bool] = None
    ) -> Dict[str, Any]:
        has_optim_state: bool = False
        for key in metadata.state_dict_metadata.keys():
            if key.startswith("optim."):
                has_optim_state = True
                break

        if optim is None:
            if not has_optim_state:
                log.warning("No optimizer state found in checkpoint")
                optim = False
            else:
                optim = True

        load_opts = self.state_dict_load_opts
        if optim:
            if not has_optim_state:
                raise RuntimeError(
                    "Checkpoint does not contain optimizer state, but 'optim=True' was requested"
                )

            if "optim.param_groups.0.params" in metadata.state_dict_metadata:
                # unflattened optimizer state
                if load_opts.flatten_optimizer_state_dict:
                    log.warning(
                        "Loading checkpoint with an unflattened optimizer state even though "
                        "'flatten_optimizer_state_dict=True' in train module's 'state_dict_load_opts', "
                        "automatically switching to 'flatten_optimizer_state_dict=False'."
                    )
                    load_opts = replace(load_opts, flatten_optimizer_state_dict=False)
            else:
                # flattened optimizer state
                if not load_opts.flatten_optimizer_state_dict:
                    log.warning(
                        "Loading checkpoint with a flattened optimizer state even though "
                        "'flatten_optimizer_state_dict=False' in train module's 'state_dict_load_opts', "
                        "automatically switching to 'flatten_optimizer_state_dict=True'."
                    )
                    load_opts = replace(load_opts, flatten_optimizer_state_dict=True)

        state_dict = self._get_state_dict(load_opts, optim=optim)
        if self.load_key_mapping is not None:
            swap_param_keys(state_dict, self.load_key_mapping, metadata=metadata)

        if not load_opts.strict:
            # Remove any keys in the 'state_dict' that are not present in the checkpoint.
            pruned_keys = prune_state_dict(state_dict, set(metadata.state_dict_metadata.keys()))
            if pruned_keys:
                log.warning(f"Checkpoint is missing the following keys: {pruned_keys}")

        return state_dict

    def state_dict_to_save(self, *, optim: Optional[bool] = None) -> Dict[str, Any]:
        if optim is None:
            optim = True
        return self._get_state_dict(self.state_dict_save_opts, optim=optim)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        load_optim = "optim" in state_dict

        if self.load_key_mapping is not None:
            swap_param_keys(state_dict, self.load_key_mapping, reverse=True, quiet=True)

        # NOTE: `dist_cp_sd.set_(model|optimizer)_state_dict()` doesn't respect `strict=False`
        # option with missing keys, so we have to handle that on our own.
        if not self.state_dict_load_opts.strict:
            flatten_optimizer_state_dict = (
                False if not load_optim else ("state" not in state_dict["optim"])
            )
            load_opts = replace(
                self.state_dict_load_opts, flatten_optimizer_state_dict=flatten_optimizer_state_dict
            )
            full_state_dict = self._get_state_dict(load_opts, optim=load_optim)
            merge_state_dicts(state_dict, full_state_dict)

        dist_cp_sd.set_model_state_dict(
            self.model,
            state_dict["model"],
            options=self.state_dict_load_opts,
        )
        gc_cuda()
        if load_optim:
            dist_cp_sd.set_optimizer_state_dict(
                self.model,
                self.optim,
                state_dict["optim"],
                options=self.state_dict_load_opts,
            )
            gc_cuda()

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        # Set model to train mode if it isn't already.
        self._set_model_mode("train")

        # Generate labels.
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)

        # Calculate how many tokens will be used in the loss.
        batch_num_tokens = batch["labels"].numel()
        batch_num_tokens_per_instance = batch["labels"].shape[1]
        batch_num_tokens_for_loss = move_to_device(
            (batch["labels"] != self.label_ignore_index).sum(), self.device
        )

        # Record percentage of masked labels.
        self.record_metric(
            "train/masked labels (%)",  # just a proportion, not a percentage
            (batch_num_tokens - batch_num_tokens_for_loss) / batch_num_tokens,
            ReduceType.mean,
        )

        # Record percentage of masked instances.
        if (instance_mask := batch.get("instance_mask")) is not None:
            self.record_metric(
                "train/masked instances (%)",  # just a proportion, not a percentage
                (~instance_mask).float().mean(),
                ReduceType.mean,
            )

            # WARN: When we mask out instances with the instance filter, we count those tokens
            # for the loss anyways. They will count as tokens with a zero loss. This means we
            # get an artificially *low* loss for these batches. But it is really hard (and slow)
            # to do this properly in a distributed setup. We add back in the full number of tokens
            # for the loss so that each rank contributes to the loss calculation fairly.
            batch_num_tokens_for_loss += (~instance_mask).sum() * batch_num_tokens_per_instance

        # Batch losses to record.
        ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        z_batch_loss: Optional[torch.Tensor] = None
        if self.z_loss_multiplier is not None:
            z_batch_loss = move_to_device(torch.tensor(0.0), self.device)

        # Soft-CE auxiliary loss (ngram soft targets). Active only when the
        # train module is configured for it AND the batch actually carries
        # soft-target tensors (emitted by NgramSoftTargetInstanceSource).
        # The required field depends on truncation mode:
        #   - "renormalize"      → soft_target_probs (linear, sums to 1 over top-K)
        #   - "uniform_residual" → soft_target_log_probs (raw KN log-probs;
        #     the loss computes residual mass = 1 - Σ topK p at runtime).
        soft_ce_target_field = (
            "soft_target_probs" if self.soft_ce_truncation == "renormalize"
            else "soft_target_log_probs"
        )
        soft_ce_active = (
            self.soft_ce_alpha_start is not None
            and soft_ce_target_field in batch
            and "soft_target_token_ids" in batch
        )
        alpha: float = self._current_soft_ce_alpha() if soft_ce_active else 0.0
        soft_ce_batch_loss: Optional[torch.Tensor] = None
        if soft_ce_active:
            soft_ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)

        # Product-of-experts logit bias (ngram log-probs scattered onto LM
        # log-probs). Active when the train module is configured with
        # ``poe_lambda`` AND the batch carries ``soft_target_log_probs``
        # (emitted by NgramSoftTargetInstanceSource with output_log_probs=True).
        poe_active = (
            self.poe_lambda is not None
            and "soft_target_log_probs" in batch
            and "soft_target_token_ids" in batch
        )
        # SB variant of the PoE bias. Activates when the batch carries the
        # ragged sb_override_* fields (emitted by NgramStupidBackoffInstanceSource).
        # Mutually exclusive with the KN-smoothed poe_active path — the
        # init-time validator guarantees only one of poe_ngram_table_dir /
        # poe_sb_table_dir can be set, but we still gate the train-step
        # branch on which fields the batch actually carries.
        poe_sb_active = (
            self.poe_lambda is not None
            and "sb_override_batch_idx" in batch
        )
        if (
            self.poe_lambda is not None
            and self.poe_sb_table_dir is not None
            and not poe_sb_active
        ):
            raise RuntimeError(
                "SB PoE training is configured but the batch has no sb_override_* fields. "
                "This would silently train the baseline LM instead of the intended PoE model."
            )

        # Split into micro-batches.
        if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative to sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
        num_micro_batches = len(micro_batches)
        if poe_sb_active:
            sb_split_logs = getattr(self, "_sb_split_debug_logs", 0)
            if sb_split_logs < 3:
                total_overrides = int(batch["sb_override_batch_idx"].numel())
                per_mb_overrides = [
                    int(mb["sb_override_batch_idx"].numel()) for mb in micro_batches
                ]
                print(
                    f"[SB train pid={os.getpid()} rank={dist.get_rank() if dist.is_initialized() else '?'}] "
                    f"split batch_size={batch['input_ids'].shape[0]} "
                    f"num_micro_batches={num_micro_batches} "
                    f"total_overrides={total_overrides:,} "
                    f"per_microbatch_overrides={per_mb_overrides[:8]}"
                    f"{'...' if len(per_mb_overrides) > 8 else ''}",
                    flush=True,
                )
                self._sb_split_debug_logs = sb_split_logs + 1

        # Train one micro-batch at a time.
        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                # Pop soft-target fields out of the micro-batch BEFORE
                # _prepare_batch, so they don't leak into model.forward(**kwargs).
                # _prepare_batch only moves standard fields to device; we move
                # the soft-target tensors here so the loss path's gather doesn't
                # mix CPU indices with CUDA logits.
                soft_target_token_ids = micro_batch.pop("soft_target_token_ids", None)
                soft_target_probs = micro_batch.pop("soft_target_probs", None)
                soft_target_log_probs = micro_batch.pop("soft_target_log_probs", None)
                # Pop the SB ragged-override tensors too; same reason — don't
                # leak them into model.forward(**kwargs).
                sb_override_batch_idx = micro_batch.pop("sb_override_batch_idx", None)
                sb_override_position = micro_batch.pop("sb_override_position", None)
                sb_override_token_id = micro_batch.pop("sb_override_token_id", None)
                sb_override_log_score = micro_batch.pop("sb_override_log_score", None)
                sb_train_log = (
                    poe_sb_active
                    and getattr(self, "_sb_train_debug_logs", 0) < 6
                )
                t_sb_transfer = time.perf_counter() if sb_train_log else None
                if sb_override_batch_idx is not None:
                    sb_override_batch_idx = sb_override_batch_idx.to(
                        self.device, non_blocking=True
                    )
                    sb_override_position = sb_override_position.to(
                        self.device, non_blocking=True
                    )
                    sb_override_token_id = sb_override_token_id.to(
                        self.device, non_blocking=True
                    )
                    sb_override_log_score = sb_override_log_score.to(
                        self.device, non_blocking=True
                    )
                if sb_train_log and t_sb_transfer is not None:
                    n_overrides = (
                        int(sb_override_batch_idx.numel())
                        if sb_override_batch_idx is not None
                        else 0
                    )
                    print(
                        f"[SB train pid={os.getpid()} rank={dist.get_rank() if dist.is_initialized() else '?'}] "
                        f"microbatch={micro_batch_idx + 1}/{num_micro_batches} "
                        f"override_transfer_dispatched={time.perf_counter() - t_sb_transfer:.4f}s "
                        f"overrides={n_overrides:,}",
                        flush=True,
                    )
                if soft_target_token_ids is not None:
                    soft_target_token_ids = soft_target_token_ids.to(
                        self.device, non_blocking=True
                    )
                if soft_target_probs is not None:
                    soft_target_probs = soft_target_probs.to(
                        self.device, non_blocking=True
                    )
                if soft_target_log_probs is not None:
                    soft_target_log_probs = soft_target_log_probs.to(
                        self.device, non_blocking=True
                    )

                input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)

                # NOTE on gradient flow: when ``return_logits=True`` is set,
                # the LMHead returns ``output.ce_loss`` and ``output.z_loss``
                # as detached tensors (they're meant for metric reporting,
                # not for backprop). Only ``output.loss`` (= ce_loss +
                # z_loss combined) carries gradient back to the model. So
                # we always combine through ``output.loss`` for the gradient
                # path, and only consult ``output.ce_loss`` / ``output.z_loss``
                # for recording metrics.
                if poe_active and soft_target_log_probs is not None:
                    # Product-of-experts path: at every position, compute the
                    # model's full-vocabulary log-probability distribution, add
                    # ``λ * log p_ngram`` at the K candidate positions, then
                    # take negative log-likelihood at the hard label. Because
                    # both terms are properly-normalized log-probabilities over
                    # the full vocabulary, the result is the cross-entropy of
                    # the joint distribution
                    #     p_final(w|h) ∝ p_lm(w|h) * p_ngram(w|h)^λ
                    # at the hard label.
                    assert soft_target_token_ids is not None
                    assert self.poe_lambda is not None
                    output = self.model_forward(
                        input_ids,
                        labels=labels,
                        ignore_index=self.label_ignore_index,
                        loss_reduction="sum",
                        z_loss_multiplier=self.z_loss_multiplier,
                        loss_div_factor=batch_num_tokens_for_loss,
                        return_logits=True,
                        **model_kwargs,
                    )
                    logits = output.logits
                    z_loss = output.z_loss
                    assert logits is not None, (
                        "PoE path requires LMHead loss_implementation='default' "
                        "so full logits are returned"
                    )

                    poe_loss = self._compute_poe_loss(
                        logits=logits,
                        soft_target_token_ids=soft_target_token_ids,
                        soft_target_log_probs=soft_target_log_probs,
                        labels=labels,
                        loss_div_factor=batch_num_tokens_for_loss,
                    )

                    # z_loss with gradient. ``output.z_loss`` is detached
                    # (metrics-only), so we recompute the regularizer here
                    # from the same logits we used for the PoE loss. The
                    # formula matches olmo-core's
                    # ``cross_entropy_loss``:
                    #   z_loss = z_multiplier · Σ (logsumexp(logits))² · mask
                    #            / div_factor
                    # over positions where labels != ignore_index.
                    if self.z_loss_multiplier is not None:
                        logits_local_f32 = get_local_tensor(logits).float()
                        z_log_sum_exp = torch.logsumexp(logits_local_f32, dim=-1)
                        z_labels_dev = labels.to(
                            z_log_sum_exp.device, non_blocking=True
                        )
                        z_mask = (z_labels_dev != self.label_ignore_index).to(
                            z_log_sum_exp.dtype
                        )
                        z_loss_with_grad = (
                            self.z_loss_multiplier
                            * (z_log_sum_exp.pow(2) * z_mask).sum()
                            / batch_num_tokens_for_loss
                        )
                        del logits_local_f32, z_log_sum_exp, z_mask
                    else:
                        z_loss_with_grad = None
                    del logits

                    combined_loss = poe_loss
                    if z_loss_with_grad is not None:
                        combined_loss = combined_loss + z_loss_with_grad

                    # We log "CE loss" as the PoE-joint cross-entropy at the
                    # hard label — i.e. the actual training objective. That's
                    # what the existing wandb panel calls "train/CE loss" so
                    # the metric is comparable to baseline runs.
                    ce_batch_loss += get_local_tensor(poe_loss.detach())
                    if z_batch_loss is not None:
                        # Use the detached value from the LMHead for the
                        # metric (it matches the reduction conventions); the
                        # gradient-bearing version above feeds the optimizer.
                        assert z_loss is not None
                        z_batch_loss += get_local_tensor(z_loss.detach())

                    combined_loss.backward()
                    del combined_loss, poe_loss
                    if z_loss_with_grad is not None:
                        del z_loss_with_grad
                    if z_loss is not None:
                        del z_loss
                elif poe_sb_active and sb_override_batch_idx is not None:
                    # Stupid-backoff PoE path. Same structure as the
                    # KN-smoothed branch above: materialize logits via
                    # model_forward(..., return_logits=True), apply the SB
                    # bias via _compute_poe_loss_sb (which uses a fresh
                    # clone of the logits so the original tensor stays
                    # intact for downstream readers like z_loss), then
                    # recompute z_loss from the un-biased logits for the
                    # regularizer term.
                    assert self.poe_lambda is not None
                    t_forward = time.perf_counter() if sb_train_log else None
                    output = self.model_forward(
                        input_ids,
                        labels=labels,
                        ignore_index=self.label_ignore_index,
                        loss_reduction="sum",
                        z_loss_multiplier=self.z_loss_multiplier,
                        loss_div_factor=batch_num_tokens_for_loss,
                        return_logits=True,
                        **model_kwargs,
                    )
                    logits = output.logits
                    z_loss = output.z_loss
                    if sb_train_log and t_forward is not None:
                        print(
                            f"[SB train pid={os.getpid()} rank={dist.get_rank() if dist.is_initialized() else '?'}] "
                            f"microbatch={micro_batch_idx + 1}/{num_micro_batches} "
                            f"model_forward_dispatched={time.perf_counter() - t_forward:.4f}s",
                            flush=True,
                        )
                    assert logits is not None, (
                        "SB-PoE path requires LMHead loss_implementation='default' "
                        "so full logits are returned"
                    )
                    t_loss = time.perf_counter() if sb_train_log else None
                    poe_loss = self._compute_poe_loss_sb(
                        logits=logits,
                        sb_override_batch_idx=sb_override_batch_idx,
                        sb_override_position=sb_override_position,
                        sb_override_token_id=sb_override_token_id,
                        sb_override_log_score=sb_override_log_score,
                        labels=labels,
                        loss_div_factor=batch_num_tokens_for_loss,
                    )
                    if sb_train_log and t_loss is not None:
                        print(
                            f"[SB train pid={os.getpid()} rank={dist.get_rank() if dist.is_initialized() else '?'}] "
                            f"microbatch={micro_batch_idx + 1}/{num_micro_batches} "
                            f"sb_loss_dispatched={time.perf_counter() - t_loss:.4f}s",
                            flush=True,
                        )
                    if self.z_loss_multiplier is not None:
                        logits_local_f32 = get_local_tensor(logits).float()
                        z_log_sum_exp = torch.logsumexp(logits_local_f32, dim=-1)
                        z_labels_dev = labels.to(
                            z_log_sum_exp.device, non_blocking=True
                        )
                        z_mask = (z_labels_dev != self.label_ignore_index).to(
                            z_log_sum_exp.dtype
                        )
                        z_loss_with_grad = (
                            self.z_loss_multiplier
                            * (z_log_sum_exp.pow(2) * z_mask).sum()
                            / batch_num_tokens_for_loss
                        )
                        del logits_local_f32, z_log_sum_exp, z_mask
                    else:
                        z_loss_with_grad = None
                    del logits
                    combined_loss = poe_loss
                    if z_loss_with_grad is not None:
                        combined_loss = combined_loss + z_loss_with_grad
                    ce_batch_loss += get_local_tensor(poe_loss.detach())
                    if z_batch_loss is not None:
                        assert z_loss is not None
                        z_batch_loss += get_local_tensor(z_loss.detach())
                    combined_loss.backward()
                    del combined_loss, poe_loss
                    if z_loss_with_grad is not None:
                        del z_loss_with_grad
                    if z_loss is not None:
                        del z_loss
                    if sb_train_log:
                        self._sb_train_debug_logs = (
                            getattr(self, "_sb_train_debug_logs", 0) + 1
                        )
                elif soft_ce_active and (
                    (self.soft_ce_truncation == "renormalize" and soft_target_probs is not None)
                    or (self.soft_ce_truncation == "uniform_residual" and soft_target_log_probs is not None)
                ):
                    # Soft-CE path: materialize logits so we can compute soft CE.
                    assert soft_target_token_ids is not None
                    output = self.model_forward(
                        input_ids,
                        labels=labels,
                        ignore_index=self.label_ignore_index,
                        loss_reduction="sum",
                        z_loss_multiplier=self.z_loss_multiplier,
                        loss_div_factor=batch_num_tokens_for_loss,
                        return_logits=True,
                        **model_kwargs,
                    )
                    logits = output.logits
                    # ``output.loss`` carries gradient and equals ce_loss +
                    # z_loss combined (when z_loss_multiplier is set).
                    # ``output.ce_loss`` and ``output.z_loss`` are detached
                    # — for metrics only.
                    hard_loss_with_grad = output.loss
                    hard_ce_loss_metric = output.ce_loss
                    z_loss_metric = output.z_loss
                    assert logits is not None, (
                        "soft-CE path requires LMHead loss_implementation='default' "
                        "so full logits are returned"
                    )

                    soft_ce_loss = self._compute_soft_ce_loss(
                        logits=logits,
                        soft_target_token_ids=soft_target_token_ids,
                        soft_target_probs=soft_target_probs,
                        soft_target_log_probs=soft_target_log_probs,
                        labels=labels,
                        loss_div_factor=batch_num_tokens_for_loss,
                    )
                    del logits

                    # Backprop:
                    #   (1 − α) · (hard_ce + z) + α · soft_ce
                    # Slight z-loss discounting: at α > 0 the regularizer's
                    # gradient is scaled by (1 − α) instead of being applied at
                    # full strength, but z_loss is small (≈ 1e-3) and acts as
                    # logit-norm regularization that's least useful when the
                    # model is being trained primarily by the soft target
                    # anyway. Cleaner alternative — recompute hard_ce + z_loss
                    # with gradient inline from ``logits`` — would let us
                    # weight z exactly at 1.0, but this minimal fix gets the
                    # critical thing right: at α = 0, full hard_ce + full
                    # z_loss gradient is applied.
                    combined_loss = (1.0 - alpha) * hard_loss_with_grad + alpha * soft_ce_loss

                    ce_batch_loss += get_local_tensor(hard_ce_loss_metric.detach())
                    assert soft_ce_batch_loss is not None
                    soft_ce_batch_loss += get_local_tensor(soft_ce_loss.detach())
                    if z_batch_loss is not None and z_loss_metric is not None:
                        z_batch_loss += get_local_tensor(z_loss_metric.detach())

                    combined_loss.backward()
                    del combined_loss, hard_loss_with_grad, soft_ce_loss
                    del hard_ce_loss_metric, z_loss_metric
                else:
                    # Hard-CE-only path — baseline, byte-identical to pre-soft-CE behavior.
                    _, loss, ce_loss, z_loss = self.model_forward(
                        input_ids,
                        labels=labels,
                        ignore_index=self.label_ignore_index,
                        loss_reduction="sum",
                        z_loss_multiplier=self.z_loss_multiplier,
                        loss_div_factor=batch_num_tokens_for_loss,
                        return_logits=False,
                        **model_kwargs,
                    )

                    ce_batch_loss += get_local_tensor(ce_loss.detach())
                    del ce_loss
                    if z_batch_loss is not None:
                        assert z_loss is not None
                        z_batch_loss += get_local_tensor(z_loss.detach())
                        del z_loss

                    loss.backward()

        del batch  # In case this helps with memory utilization.

        self.model.post_batch(dry_run=dry_run)

        if dry_run:
            self.model.reset_auxiliary_metrics()
            return

        # Record loss metrics.
        if isinstance(self.optim, SkipStepOptimizer):
            # Need to reduce the loss right away for the SkipStepOptimizer.
            if is_distributed():
                ce_batch_loss.div_(self._reduce_divide_factor)
                dist.all_reduce(ce_batch_loss)
                ce_batch_loss.div_(self.world_size)
                ce_batch_loss.mul_(self._reduce_divide_factor)
            self.record_ce_loss(ce_batch_loss)
            self.optim.latest_loss = ce_batch_loss
        else:
            self.record_ce_loss(ce_batch_loss, ReduceType.mean)
        if z_batch_loss is not None:
            assert self.z_loss_multiplier is not None
            self.record_metric(
                "Z loss",
                z_batch_loss,
                ReduceType.mean,
                namespace="train",
            )
            self.record_metric(
                "Z loss unscaled",
                z_batch_loss / self.z_loss_multiplier,
                ReduceType.mean,
                namespace="train",
            )
        if soft_ce_batch_loss is not None:
            self.record_metric(
                "soft CE loss",
                soft_ce_batch_loss,
                ReduceType.mean,
                namespace="train",
            )
            # Combined optimization objective:
            #   (1-α) · hard_CE + α · soft_CE + z_loss
            # This is the actual loss being backpropped each step; useful as
            # a single number to track training progress when α is changing.
            total_batch_loss = (1.0 - alpha) * ce_batch_loss + alpha * soft_ce_batch_loss
            if z_batch_loss is not None:
                total_batch_loss = total_batch_loss + z_batch_loss
            self.record_metric(
                "total loss",
                total_batch_loss,
                ReduceType.mean,
                namespace="train",
            )
        # Always log the soft-CE α schedule so the wandb panel shows the
        # full ramp — including the post-ramp tail at α=0 — instead of
        # going blank once soft-CE deactivates.
        if self.soft_ce_alpha_start is not None:
            self.record_metric(
                "soft CE alpha",
                alpha,
                namespace="train",
            )
        # Log the PoE λ so the wandb panel shows the (constant) mixing weight.
        if self.poe_lambda is not None:
            self.record_metric(
                "poe lambda",
                self._effective_poe_lambda_for_logging(),
                namespace="train",
            )
            if self.poe_lambda_decay_to_zero_windows:
                self.record_metric(
                    "poe lambda multiplier",
                    self._poe_lambda_decay_to_zero_multiplier(),
                    namespace="train",
                )
                self.record_metric(
                    "poe lambda base",
                    self._base_poe_lambda_for_logging(),
                    namespace="train",
                )
            if self.poe_lambda_learnable:
                self.record_metric(
                    "poe lambda log",
                    self._poe_lambda_log_for_logging(),
                    namespace="train",
                )

        # And additional metrics.
        for metric_name, (metric_val, reduction) in self.model.compute_auxiliary_metrics(
            reset=True
        ).items():
            self.record_metric(
                metric_name,
                metric_val,
                reduction,
                namespace="train",
            )

    def eval_batch(
        self,
        batch: Dict[str, Any],
        labels: Optional[torch.Tensor] = None,
        *,
        compute_ce_loss: bool = True,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        # CP and TP are supported for PPL evals (LMEvaluator) since they only need per-token
        # CE loss. Downstream evals that require full logits will fail naturally if attempted
        # with CP or TP.

        input_ids, labels, model_kwargs = self._prepare_batch(batch, labels)

        # When using CP/TP, shard the label_mask along the sequence dimension to match the
        # sharded ce_loss output shape. CP shards first (S -> S/CP), then TP shards further
        # (S/CP -> S/(CP*TP)).
        if self.cp_enabled and "label_mask" in model_kwargs:
            assert self.model._cp_load_balancer is not None
            (label_mask,) = self.model._cp_load_balancer.batch_shard(
                inputs=[model_kwargs["label_mask"]],
                seq_dims=[1],
                pad_values=[0],
            )
            model_kwargs["label_mask"] = label_mask.to(torch.bool)

        if self.tp_enabled and "label_mask" in model_kwargs:
            tp_mesh = self.model._tp_mesh
            assert tp_mesh is not None
            chunks = model_kwargs["label_mask"].chunk(tp_mesh.size(), dim=1)
            model_kwargs["label_mask"] = chunks[tp_mesh.get_local_rank()]

        self._set_model_mode("eval")

        # PoE eval path: we need the model's full logits so we can scatter-add
        # the ngram bias before computing CE. Force return_logits=True for PoE.
        # Bare-model logits would represent the wrong distribution at eval time
        # (the model trained as the complement of the ngram).
        if self.poe_lambda is not None:
            with self._eval_batch_context():
                output = self.model_forward(
                    input_ids,
                    labels=labels,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="none",
                    return_logits=True,
                    **model_kwargs,
                )
            assert isinstance(output, LMOutputWithLoss)
            assert output.logits is not None, (
                "PoE eval requires LMHead loss_implementation='default' so logits "
                "are returned"
            )
            if self.poe_sb_table_dir is not None:
                biased_logits, biased_ce_loss = self._apply_poe_eval_bias_sb(
                    logits=output.logits,
                    input_ids=input_ids,
                    labels=labels,
                    compute_ce_loss=compute_ce_loss,
                )
            else:
                biased_logits, biased_ce_loss = self._apply_poe_eval_bias(
                    logits=output.logits,
                    input_ids=input_ids,
                    labels=labels,
                    compute_ce_loss=compute_ce_loss,
                )
            return output._replace(logits=biased_logits, ce_loss=biased_ce_loss)

        with self._eval_batch_context():
            output = self.model_forward(
                input_ids,
                labels=labels,
                ignore_index=self.label_ignore_index,
                loss_reduction="none",
                return_logits=False if (self.cp_enabled or self.tp_enabled) else None,
                **model_kwargs,
            )

        if self.tp_enabled and isinstance(output, LMOutputWithLoss):
            output = output._replace(ce_loss=get_local_tensor(output.ce_loss))

        return output

    def _get_poe_eval_ngram_source(self):
        """Lazy-instantiate the ngram source for eval-time PoE bias lookup.

        Each process gets its own instance; the underlying mmap'd file is
        shared via the OS page cache (and the /dev/shm mirror put there by
        the training-time data-loader workers).
        """
        if self._poe_eval_ngram_source is None:
            assert self.poe_ngram_table_dir is not None
            from olmo_core.data.ngram_soft_target import NgramTableSoftTargetSource

            self._poe_eval_ngram_source = NgramTableSoftTargetSource(
                table_dir=self.poe_ngram_table_dir,
                K=self.poe_ngram_K,
                N_max=self.poe_ngram_N_max,
                output_log_probs=True,
            )
        return self._poe_eval_ngram_source

    def _get_poe_sb_reader(self):
        """Lazy-instantiate the StupidBackoffNgramLM reader.

        Parallels :meth:`_get_poe_eval_ngram_source` but for the SB index:
        each process opens its own mmap; OS page cache shares pages with
        the training-time dataloader workers built by
        :class:`olmo_core.data.composable.NgramStupidBackoffInstanceSource`.
        """
        if self._poe_sb_reader is None:
            assert self.poe_sb_table_dir is not None
            from olmo_core.data.stupid_backoff_ngram import StupidBackoffNgramLM

            self._poe_sb_reader = StupidBackoffNgramLM(
                table_dir=self.poe_sb_table_dir,
                dolma2_vocab_size=self.poe_sb_dolma2_vocab_size,
                N_max=self.poe_sb_N_max,
                alpha=self.poe_sb_alpha,
                max_order2_continuations=self.poe_sb_max_order2_continuations,
                max_order_continuations=self.poe_sb_max_order_continuations,
                min_order_counts=self.poe_sb_min_order_counts,
                mirror_to_shm=self.poe_sb_mirror_to_shm,
                index_access=self.poe_sb_index_access,
                lookup_threads=self.poe_sb_eval_lookup_threads,
                topk_uniform_residual_k=self.poe_sb_topk_uniform_residual_k,
                recursive_topk_uniform_residual_k=(
                    self.poe_sb_recursive_topk_uniform_residual_k
                ),
            )
        return self._poe_sb_reader

    def _get_poe_sb_unigram_floor_dev(self, *, dtype: torch.dtype) -> torch.Tensor:
        """Lazy: pull the SB unigram floor onto the model device once.

        The floor is constant across batches and is the only V-sized piece
        of the SB bias we need on-device. Cached as a non-parameter tensor.
        """
        if (
            self._poe_sb_unigram_floor_dev is None
            or self._poe_sb_unigram_floor_dev.dtype != dtype
            or self._poe_sb_unigram_floor_dev.device != self.device
        ):
            reader = self._get_poe_sb_reader()
            if (
                self.poe_sb_topk_uniform_residual_k is not None
                or self.poe_sb_recursive_topk_uniform_residual_k is not None
            ):
                floor_cpu = torch.zeros(
                    self.poe_sb_dolma2_vocab_size,
                    dtype=dtype,
                )
            else:
                floor_cpu = torch.from_numpy(reader.unigram_floor).to(dtype=dtype)
            self._poe_sb_unigram_floor_dev = floor_cpu.to(self.device, non_blocking=True)
        return self._poe_sb_unigram_floor_dev

    @staticmethod
    def _env_flag(name: str) -> bool:
        value = os.environ.get(name, "")
        return value.lower() not in ("", "0", "false", "no", "off")

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        value = os.environ.get(name)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    def _sb_eval_should_log_timing(self, *, call_idx: int, total_s: float) -> bool:
        if not self._env_flag("OLMO_SB_EVAL_TIMING"):
            return False
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return False

        first_n = self._env_int("OLMO_SB_EVAL_TIMING_FIRST_N", 3)
        every_n = self._env_int("OLMO_SB_EVAL_TIMING_EVERY_N", 100)
        slow_s = self._env_float("OLMO_SB_EVAL_TIMING_SLOW_S", 1.0)
        return (
            call_idx < first_n
            or (every_n > 0 and (call_idx + 1) % every_n == 0)
            or total_s >= slow_s
        )

    @staticmethod
    def _sync_for_timing(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _apply_poe_eval_bias(
        self,
        *,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        compute_ce_loss: bool = True,
    ) -> tuple:
        """Apply the PoE ngram bias to eval-time logits and recompute CE.

        Computes per-position contexts from input_ids (same convention as
        NgramSoftTargetInstanceSource.__getitem__), looks up per-position
        top-K (token_ids, log_probs) from the ngram source, scatter-adds
        ``λ * log_p_ngram`` onto the K candidate positions of the LM logits,
        then computes per-position CE loss on the modified logits at the hard
        labels. Returns the modified logits and the per-position CE loss.
        """
        import numpy as np

        assert self.poe_lambda is not None
        prefix_len = self.poe_ngram_N_max - 1
        K = self.poe_ngram_K

        local_logits = get_local_tensor(logits)
        B, S, V = local_logits.shape

        # Build per-position contexts (one per (b, s)) from input_ids.
        # Same indexing as NgramSoftTargetInstanceSource: at position s, the
        # context is the up-to-prefix_len tokens ending at s inclusive.
        cpu_input_ids = input_ids.detach().to("cpu").numpy()
        contexts = []
        for b in range(B):
            row = cpu_input_ids[b]
            for s in range(S):
                start = max(0, s + 1 - prefix_len)
                contexts.append(tuple(int(t) for t in row[start : s + 1]))

        ngram_source = self._get_poe_eval_ngram_source()
        ids_np, log_probs_np = ngram_source.lookup_batch(contexts)
        soft_target_token_ids = torch.as_tensor(
            ids_np.reshape(B, S, K), dtype=torch.long, device=local_logits.device
        )
        soft_target_log_probs = torch.as_tensor(
            log_probs_np.reshape(B, S, K), dtype=torch.float32, device=local_logits.device
        )

        # PoE bias under the uniform-residual approximation (see plan.md
        # "Approach 2: Product of experts"): the truncated top-K ngram
        # distribution is extended to a proper full-vocab distribution by
        # spreading the residual mass (1 - Σ topK p_ngram) uniformly over the
        # V−K non-top-K tokens. Per position the ngram log-prob is therefore
        #
        #     log p_ngram[w in topK] = stored value
        #     log p_ngram[w not in topK] = log_residual = log((1 − Σ topK p) / (V − K))
        #
        # Naive PoE would add λ * log_p_ngram[w] at every position, but by
        # softmax shift-invariance subtracting the per-position constant
        # (λ * log_residual) from every logit doesn't change the joint. So the
        # equivalent operation we actually run is:
        #
        #     bias[w in topK]    = λ * (log_p_ngram[w] − log_residual)
        #     bias[w not in topK] = 0    (unchanged logit)
        #
        # log_residual is typically very negative (≈ −log V ≈ −11.5 at V=100K),
        # so subtracting it adds ~+11 to top-K logits, which is the "boost
        # top-K relative to non-top-K" behavior PoE prescribes. Sentinel slots
        # (-inf log-prob, used when fewer than K candidates exist) scatter 0
        # so they don't corrupt logit[0] when their token id collides.
        finite_mask = torch.isfinite(soft_target_log_probs)
        # Σ exp(log p_ngram) over the finite top-K entries, per position.
        # Sentinel slots contribute 0 because exp(-inf) → 0.
        masked_log_p = torch.where(
            finite_mask, soft_target_log_probs,
            torch.full_like(soft_target_log_probs, float("-inf")),
        )
        sum_topK_ngram_p = torch.exp(masked_log_p).sum(dim=-1, keepdim=True)  # (B, S, 1)
        # Residual mass distributed uniformly over the V−K non-top-K tokens.
        # Cap sum_topK_ngram_p at (1 − 1/V) so residual_per_token ≥ 1/(V·(V−K)),
        # i.e. log_residual ≥ −log(V·(V−K)) ≈ −2·log V ≈ −23 (V=100K). Without
        # this cap, KN smoothing artifacts at build time can produce
        # sum_topK_ngram_p > 1 due to floating-point drift, causing the old
        # ``clamp_min(1e-12)`` to push log_residual to ≈ −28 and producing huge
        # gradient cliffs at affected positions (the optimizer sees the joint
        # distribution flip violently between "top-K covers everything" and
        # "uniform-residual" depending on numerical noise). Capping ensures the
        # residual always represents at least 1/V of the non-top-K mass per
        # token, which is a sensible floor — every token in vocab is
        # "at least uniform-prior likely".
        sum_topK_ngram_p_capped = sum_topK_ngram_p.clamp_max(1.0 - 1.0 / float(V))
        residual_per_token = (1.0 - sum_topK_ngram_p_capped) / float(V - K)
        log_residual = torch.log(residual_per_token)  # (B, S, 1)
        # Bias to scatter onto top-K logits. Sentinel slots get 0.
        scatter_bias = torch.where(
            finite_mask,
            soft_target_log_probs - log_residual,
            torch.zeros_like(soft_target_log_probs),
        )
        if hasattr(self, "_effective_poe_lambda"):
            poe_lambda = self._effective_poe_lambda(dtype=torch.float32).to(local_logits.device)
        else:
            poe_lambda = torch.tensor(float(self.poe_lambda), device=local_logits.device, dtype=torch.float32)
        biased_logits_f32 = local_logits.float().clone()
        biased_logits_f32.scatter_add_(
            -1, soft_target_token_ids, poe_lambda * scatter_bias
        )

        # Per-position CE on the biased logits at the hard label.
        if labels is not None and compute_ce_loss:
            log_sum_exp = torch.logsumexp(biased_logits_f32, dim=-1)
            # Replace ignored labels with 0 for safe gather; mask after.
            safe_labels = labels.clone()
            safe_labels[safe_labels == self.label_ignore_index] = 0
            label_logits = biased_logits_f32.gather(
                -1, safe_labels.unsqueeze(-1).to(biased_logits_f32.device)
            ).squeeze(-1)
            ce_loss = -(label_logits - log_sum_exp)
            ce_loss = torch.where(
                labels.to(ce_loss.device) == self.label_ignore_index,
                torch.zeros_like(ce_loss),
                ce_loss,
            )
        else:
            ce_loss = None

        return biased_logits_f32, ce_loss

    def _apply_poe_eval_bias_sb(
        self,
        *,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        compute_ce_loss: bool = True,
    ) -> tuple:
        """Stupid-backoff analog of :meth:`_apply_poe_eval_bias`.

        Eval-callback path. Computes per-instance SB overrides synchronously on
        CPU via :func:`olmo_core.data.sb_bias.compute_sb_overrides_for_batch`,
        mirroring what the dataloader does at train time. LM eval then computes
        exact CE from a dense unigram-floor normalization plus sparse
        higher-order corrections. Downstream evals still get a materialized
        biased-logits tensor because they may need full logits.
        """
        from olmo_core.data.sb_bias import (
            compute_sb_overrides_for_batch,
            compute_sparse_sb_ce_loss,
        )

        assert self.poe_lambda is not None
        assert self.poe_sb_table_dir is not None

        local_logits = get_local_tensor(logits)
        B, S, V = local_logits.shape
        timing = self._env_flag("OLMO_SB_EVAL_TIMING")
        call_idx = self._poe_sb_eval_bias_calls
        self._poe_sb_eval_bias_calls += 1
        if timing:
            self._sync_for_timing(local_logits.device)
        t0 = time.perf_counter()
        local_logits_f32 = local_logits.float()
        unigram_floor = self._get_poe_sb_unigram_floor_dev(
            dtype=local_logits_f32.dtype
        )
        if timing:
            self._sync_for_timing(local_logits_f32.device)
        t_clone = time.perf_counter()

        # Synchronous CPU lookup of per-instance overrides.
        reader = self._get_poe_sb_reader()
        overrides_cpu = compute_sb_overrides_for_batch(input_ids, reader)
        t_lookup = time.perf_counter()
        # Move to logits device (the helper expects on-device tensors).
        bidx = overrides_cpu["sb_override_batch_idx"].to(
            local_logits_f32.device, non_blocking=True
        )
        pos = overrides_cpu["sb_override_position"].to(
            local_logits_f32.device, non_blocking=True
        )
        tok = overrides_cpu["sb_override_token_id"].to(
            local_logits_f32.device, non_blocking=True
        )
        sc = overrides_cpu["sb_override_log_score"].to(
            local_logits_f32.device, non_blocking=True
        )
        if timing:
            self._sync_for_timing(local_logits_f32.device)
        t_copy = time.perf_counter()
        if labels is not None and compute_ce_loss:
            ce_loss = compute_sparse_sb_ce_loss(
                local_logits_f32,
                labels,
                unigram_floor,
                bidx,
                pos,
                tok,
                sc,
                float(self._effective_poe_lambda().detach().cpu()),
                self.label_ignore_index,
            )
            biased_logits_f32 = None
            t_apply = t_copy
        else:
            from olmo_core.data.sb_bias import apply_sb_bias_inplace

            biased_logits_f32 = local_logits_f32.clone()
            apply_sb_bias_inplace(
                biased_logits_f32,
                unigram_floor,
                bidx,
                pos,
                tok,
                sc,
                float(self._effective_poe_lambda().detach().cpu()),
            )
            if timing:
                self._sync_for_timing(biased_logits_f32.device)
            t_apply = time.perf_counter()
            ce_loss = None
        if timing:
            self._sync_for_timing(local_logits_f32.device)
        t_ce = time.perf_counter()

        total_s = t_ce - t0
        if self._sb_eval_should_log_timing(call_idx=call_idx, total_s=total_s):
            override_count = int(overrides_cpu["sb_override_token_id"].numel())
            tokens = B * S
            log.info(
                "[SB eval timing call=%d shape=(%d,%d,%d) overrides=%d "
                "overrides_per_token=%.2f compute_ce=%s total=%.3fs "
                "clone_floor=%.3fs lookup_cpu=%.3fs copy_to_gpu=%.3fs "
                "apply_bias=%.3fs ce=%.3fs]",
                call_idx + 1,
                B,
                S,
                V,
                override_count,
                override_count / max(tokens, 1),
                compute_ce_loss,
                total_s,
                t_clone - t0,
                t_lookup - t_clone,
                t_copy - t_lookup,
                t_apply - t_copy,
                t_ce - t_apply,
            )

        return biased_logits_f32, ce_loss

    def optim_step(self):
        if self.poe_lambda_learnable:
            lambda_log = self._poe_lambda_log_param()
            lambda_log_grad = lambda_log.grad
            self.trainer.record_metric(
                "poe lambda log grad",
                (
                    torch.zeros((), device=lambda_log.device, dtype=lambda_log.dtype)
                    if lambda_log_grad is None
                    else self._scalar_metric_tensor(lambda_log_grad)
                ),
                reduce_type=None,
                namespace="optim",
            )
            self.trainer.record_metric(
                "poe lambda grad is none",
                float(lambda_log_grad is None),
                reduce_type=None,
                namespace="optim",
            )

            lambda_frozen = self._poe_lambda_decay_to_zero_active()
            self.trainer.record_metric(
                "poe lambda frozen",
                float(lambda_frozen),
                reduce_type=None,
                namespace="optim",
            )
            if lambda_frozen:
                # Use grad=None, not a zero tensor. Adam-style optimizers can
                # still move a parameter with a zero grad via momentum state;
                # None makes the optimizer skip the learned lambda parameter.
                lambda_log.grad = None

        # Maybe clip gradients.
        if self.max_grad_norm is not None:
            grad_norm = self._clip_grad_norm(self.max_grad_norm)
            # NOTE: grad norm is already reduced over ranks, so we set `reduce_type` to `None`.
            self.trainer.record_metric(
                "total grad norm", grad_norm, reduce_type=None, namespace="optim"
            )
            if isinstance(self.optim, SkipStepOptimizer):
                self.optim.latest_grad_norm = grad_norm
            if self.poe_lambda_learnable:
                lambda_log = self._poe_lambda_log_param()
                lambda_log_grad = lambda_log.grad
                self.trainer.record_metric(
                    "poe lambda log grad clipped",
                    (
                        torch.zeros((), device=lambda_log.device, dtype=lambda_log.dtype)
                        if lambda_log_grad is None
                        else self._scalar_metric_tensor(lambda_log_grad)
                    ),
                    reduce_type=None,
                    namespace="optim",
                )

        # Maybe adjust learning rate.
        if self.scheduler is not None:
            for group_idx, group in enumerate(self.optim.param_groups):
                new_lr = self.scheduler.set_lr(group, self.trainer)
                self.trainer.record_metric(f"LR (group {group_idx})", new_lr, namespace="optim")

        # Step optimizer.
        self.optim.step()
        if isinstance(self.optim, SkipStepOptimizer):
            self.record_metric("step skipped", self.optim.step_skipped, namespace="optim")

        self.model.post_optim_step()

    def zero_grads(self):
        self.optim.zero_grad(set_to_none=True)

    def model_forward(
        self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """
        Run a forward pass on a micro-batch, returning the logits.
        """
        with self._model_forward_context():
            return self.model(input_ids, labels=labels, **kwargs)

    @lru_cache
    def num_flops_per_token(self, seq_len: int) -> Optional[int]:
        try:
            return self.model.num_flops_per_token(seq_len)
        except NotImplementedError as ex:
            warn_once(f"Unable to estimate num flops per token: {ex}")
            return None

    def global_num_flops_in_batch(self, batch: Dict[str, Any]) -> Optional[int]:
        global_num_tokens = self.trainer.data_loader.global_num_tokens_in_batch(batch)
        if global_num_tokens is None:
            return None
        flops_per_token = self.num_flops_per_token(seq_len=batch["input_ids"].shape[1])
        return flops_per_token * global_num_tokens if flops_per_token is not None else None

    @contextlib.contextmanager
    def _train_microbatch_context(
        self, micro_batch_idx: int, num_micro_batches: int
    ) -> Generator[None, None, None]:
        is_last_mb = micro_batch_idx == num_micro_batches - 1
        with contextlib.ExitStack() as stack:
            if isinstance(self.model, FSDPModule):
                assert self.dp_config is not None
                # On the last backward FSDP waits on pending gradient reduction and clears internal data
                # data structures for backward prefetching.
                self.model.set_is_last_backward(is_last_mb)
                # For HSDP we can delay the gradients all-reduce until the final micro-batch.
                if self.dp_config.name == DataParallelType.hsdp:
                    self.model.set_requires_all_reduce(is_last_mb)
            elif isinstance(self.model, DDP):
                # For DDP, only sync gradients on the final micro-batch.
                if not is_last_mb:
                    stack.enter_context(self.model.no_sync())

            yield

    @contextlib.contextmanager
    def _eval_batch_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            yield

    @contextlib.contextmanager
    def _model_forward_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if self.autocast_precision is not None:
                stack.enter_context(torch.autocast(self.device.type, dtype=self.autocast_precision))
            yield

    def _get_state_dict(
        self, sd_options: dist_cp_sd.StateDictOptions, optim: bool = True
    ) -> Dict[str, Any]:
        state_dict: Dict[str, Any] = {
            "model": dist_cp_sd.get_model_state_dict(self.model, options=sd_options),
        }
        if optim:
            state_dict["optim"] = dist_cp_sd.get_optimizer_state_dict(
                self.model, self.optim, options=sd_options
            )
        return state_dict

    def _clip_grad_norm(
        self, max_grad_norm: float, norm_type: float = 2.0, foreach: Optional[bool] = None
    ) -> torch.Tensor:
        if isinstance(self.model, FSDP):
            return self.model.clip_grad_norm_(max_grad_norm)

        # Adapted from https://github.com/pytorch/torchtitan/blob/2a4437014e66bcf88a3f0419b816266e6326d539/torchtitan/utils.py#L348

        parameters = [p for p in self.model.parameters()]
        grads = [p.grad for p in parameters if p.grad is not None]

        total_norm = nn.utils.get_total_norm(
            grads, norm_type=norm_type, error_if_nonfinite=False, foreach=foreach
        )

        # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
        # We can simply reduce the DTensor to get the total norm in this tensor's process group
        # and then convert it to a local tensor.
        # NOTE: It has two purposes:
        #       1. to make sure the total norm is computed correctly when PP is used (see below)
        #       2. to return a reduced total_norm tensor whose .item() would return the correct value
        if isinstance(total_norm, DTensor):
            # Will reach here if any non-PP parallelism is used.
            # If only using PP, total_norm will be a local tensor.
            total_norm = total_norm.full_tensor()

        torch.nn.utils.clip_grads_with_norm_(parameters, max_grad_norm, total_norm, foreach=foreach)
        return total_norm

    def _prepare_batch(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        input_ids = batch.pop("input_ids")
        labels = labels if labels is not None else batch.pop("labels", None)
        if "doc_lens" in batch and "max_doc_lens" in batch:
            log_once(log, "intra-document masking enabled")
        return input_ids, labels, batch

    def _current_soft_ce_alpha(self) -> float:
        """The ngram soft-CE mixing weight alpha(t) for the current step.

        Linear decay from ``soft_ce_alpha_start`` at step 0 to 0 at step
        ``soft_ce_alpha_ramp_fraction * max_steps``, then 0 thereafter.
        Returns 0 when soft-CE is disabled or ramp is complete.
        """
        if self.soft_ce_alpha_start is None:
            return 0.0
        max_steps = self.trainer.max_steps
        if max_steps is None or max_steps <= 0:
            # Token-based duration with no derivable step total, or single-step run.
            # Fall back to a constant alpha_start so the user at least sees the
            # intended behaviour; log once so they notice.
            log_once(
                log,
                "soft-CE alpha ramp needs a step-resolvable max_duration; "
                "falling back to constant alpha=soft_ce_alpha_start",
            )
            return float(self.soft_ce_alpha_start)
        ramp_steps = int(self.soft_ce_alpha_ramp_fraction * max_steps)
        step = self.trainer.global_step
        if ramp_steps <= 0 or step >= ramp_steps:
            return 0.0
        return float(self.soft_ce_alpha_start) * (1.0 - step / ramp_steps)

    def _compute_soft_ce_loss(
        self,
        *,
        logits: torch.Tensor,
        soft_target_token_ids: torch.Tensor,
        soft_target_probs: Optional[torch.Tensor] = None,
        soft_target_log_probs: Optional[torch.Tensor] = None,
        labels: torch.Tensor,
        loss_div_factor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sum-reduction soft cross-entropy divided by ``loss_div_factor``.

        Two truncation-handling modes (selected by ``self.soft_ce_truncation``):

        **renormalize** (default). The target is a top-K distribution
        renormalized to sum to 1 over the K candidate tokens; non-top-K
        targets are zero. The loss at position t is

            L = - Σ_{k=1..K} p_k · log q_k

        where ``p_k`` is the renormalized top-K target probability (from
        ``soft_target_probs``) and ``log q_k`` is the model's full-vocab
        log-probability at the corresponding token id.

        **uniform_residual**. The target is the raw KN top-K distribution
        extended into a proper full-vocab distribution by spreading the
        residual mass ``r = 1 − Σ topK p_ngram`` uniformly over the V−K
        non-top-K tokens. The loss is

            L = - Σ_topK p_k · log q_k
                - (r / (V−K)) · Σ_non-topK log q_w

        This avoids forcing the model to put zero mass on out-of-top-K
        tokens (including the gold whenever the gold is rare). We avoid
        materializing the (B, S, V) log-probs tensor by writing

            Σ_non-topK log q_w = Σ_all log q_w − Σ_topK log q_w
                              = (Σ_all logit_w) − V·logsumexp − Σ_topK log q_w

        — only the per-position scalar ``Σ_all logit_w`` is new; everything
        else we already need to compute for the top-K branch.

        Naming: forward KL up to a constant. The ngram-side target
        distribution p is fixed (not a function of model parameters), so
        KL(p ‖ q_θ) = soft_CE(p, q_θ) − H(p) and the H(p) term has zero
        gradient w.r.t. θ. We compute and log soft_CE because it's the
        actual quantity we're summing, but the optimization is equivalent
        to forward KL minimization. If we ever switch to reverse KL
        (KL(q_θ ‖ p) — mode-seeking), this function needs to be rewritten.
        """
        # Upcast logits to fp32 for numerical stability of the normalizer.
        logits_f32 = get_local_tensor(logits).float()
        V = logits_f32.size(-1)
        # Per-position log-of-sum-of-exponentials normalizer over the full
        # vocabulary. Shape: (sequences in microbatch, positions per sequence).
        log_sum_exp = torch.logsumexp(logits_f32, dim=-1)
        # Raw logits at the K target tokens for each position. (B, S, K).
        gathered_logits = logits_f32.gather(-1, soft_target_token_ids)
        # log q_k per top-K target. (B, S, K).
        gathered_log_probs = gathered_logits - log_sum_exp.unsqueeze(-1)

        if self.soft_ce_truncation == "renormalize":
            assert soft_target_probs is not None
            soft_ce_per_pos = -(soft_target_probs.float() * gathered_log_probs).sum(dim=-1)
        else:  # uniform_residual
            assert soft_target_log_probs is not None
            # Convert raw KN log-probs to linear probs at the K slots. Slots
            # whose stored log-prob is non-finite are sentinels — placeholder
            # entries used when a prefix has fewer than K real candidates,
            # padded with token_id=0 — and must be excluded from both the
            # top-K sum and the non-top-K count.
            log_p_ngram = soft_target_log_probs.float()
            finite_mask = torch.isfinite(log_p_ngram)
            finite_mask_f = finite_mask.to(log_p_ngram.dtype)
            log_p_ngram_safe = torch.where(
                finite_mask, log_p_ngram,
                torch.full_like(log_p_ngram, float("-inf")),
            )
            top_k_p = torch.exp(log_p_ngram_safe)               # (B, S, K), 0 at sentinels
            sum_top_k_p = top_k_p.sum(dim=-1)                   # (B, S)
            # Cap sum_top_k_p ≤ 1 − 1/V to avoid residual_total < 0 due to KN
            # smoothing drift; matches the cap in the PoE wrapper paths. With
            # the cap, residual_total ≥ 1/V, and the non-top-K CE term keeps a
            # well-defined small contribution at positions where the ngram
            # numerically claims top-K covers everything.
            sum_top_k_p = sum_top_k_p.clamp_max(1.0 - 1.0 / float(V))
            residual_total = (1.0 - sum_top_k_p)                # (B, S), guaranteed ≥ 1/V

            # Real-top-K count per position (K minus the number of sentinels).
            K_real = finite_mask_f.sum(dim=-1)                  # (B, S)
            V_minus_K_real = (float(V) - K_real).clamp_min(1.0) # (B, S)

            # Top-K contribution: -Σ p_k · log q_k. Sentinel slots have p_k = 0
            # so they contribute 0 to this sum automatically.
            ce_topK = -(top_k_p * gathered_log_probs).sum(dim=-1)  # (B, S)

            # Non-top-K contribution: -(r / (V − K_real)) · Σ_non-topK log q_w.
            # Σ_all log q_w = Σ_all (logit_w − logsumexp) = sum_logits_all − V·logsumexp.
            # Σ_non-topK log q_w = Σ_all log q_w − Σ_real-topK log q_w.
            sum_logits_all = logits_f32.sum(dim=-1)                       # (B, S)
            sum_log_q_all = sum_logits_all - float(V) * log_sum_exp        # (B, S)
            # Mask gathered_log_probs to only sum over real top-K slots
            # (sentinel slots' gathered log_q is whatever log_q[0] happens to
            # be; we don't want it in this sum).
            sum_log_q_real_topK = (gathered_log_probs * finite_mask_f).sum(dim=-1)
            sum_log_q_non_topK = sum_log_q_all - sum_log_q_real_topK
            ce_non_topK = -(residual_total / V_minus_K_real) * sum_log_q_non_topK

            soft_ce_per_pos = ce_topK + ce_non_topK

        # Mask out ignored label positions. Labels arrive on CPU (the standard
        # hard-CE path runs them through model_forward which handles the move
        # internally); move them to logits' device for the dtype-cast multiply.
        labels_dev = labels.to(soft_ce_per_pos.device, non_blocking=True)
        mask = (labels_dev != self.label_ignore_index).to(soft_ce_per_pos.dtype)
        soft_ce_per_pos = soft_ce_per_pos * mask
        # Sum → scalar; divide by the same denominator hard CE used so the two
        # terms are on a comparable per-valid-token scale.
        return soft_ce_per_pos.sum() / loss_div_factor

    def _compute_poe_loss(
        self,
        *,
        logits: torch.Tensor,
        soft_target_token_ids: torch.Tensor,
        soft_target_log_probs: torch.Tensor,
        labels: torch.Tensor,
        loss_div_factor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sum-reduction product-of-experts cross-entropy at the hard
        labels, divided by ``loss_div_factor``.

        The product-of-experts joint is

            log p_final(w | h) = log p_lm(w | h) + λ · log p_ngram(w | h) − log Z

        — the joint log-probability of token w given context h equals the LM's
        log-probability plus λ times the ngram's log-probability, minus a
        per-position normalizer Z.

        The ngram input arrives as a top-K truncation: we have raw KN
        log-probs for K specific tokens per position, nothing for the other
        V−K. To make this a proper full-vocab distribution (which PoE
        requires), we extend with a **uniform residual**: spread whatever
        mass isn't in top-K uniformly over the V−K non-top-K tokens. So per
        position

            sum_topK p_ngram = Σ exp(stored log-probs over finite slots)
            log_residual    = log((1 − sum_topK p_ngram) / (V − K))
            log p_ngram[w in topK]    = stored value
            log p_ngram[w not in topK] = log_residual

        With those values, Z decomposes cleanly:

            Z = Σ_topK exp(log p_lm + λ · log p_ngram[w])
              + Σ_non-topK exp(log p_lm[w] + λ · log_residual)
              = sum_joint_topk + exp(λ · log_residual) · (1 − sum_lm_topk)

        because the LM log-probs sum to 1 over the full vocab. We compute Z
        without materializing the (B, S, V) tensor.

        At the gold label position, the bias term is the stored log-prob if
        the gold ∈ top-K, else log_residual.
        """
        # Upcast logits to fp32 for numerical stability of the log_softmax.
        logits_f32 = get_local_tensor(logits).float()
        V = logits_f32.size(-1)
        K = soft_target_log_probs.size(-1)
        # Per-position log-of-sum-of-exponentials normalizer over the full
        # vocabulary. Shape: (sequences in microbatch, positions per sequence).
        log_sum_exp_lm = torch.logsumexp(logits_f32, dim=-1)
        # LM full-vocabulary log-probabilities at the K target tokens.
        # Shape: (sequences, positions, K).
        gathered_lm_log_probs = logits_f32.gather(-1, soft_target_token_ids) - log_sum_exp_lm.unsqueeze(-1)

        # Compute log_residual for the uniform-non-top-K extension.
        soft_target_log_probs_f32 = soft_target_log_probs.float()
        finite_mask = torch.isfinite(soft_target_log_probs_f32)
        # Σ p_ngram over finite top-K entries (sentinels contribute 0 via exp(-inf)).
        masked_log_p_ngram = torch.where(
            finite_mask, soft_target_log_probs_f32,
            torch.full_like(soft_target_log_probs_f32, float("-inf")),
        )
        sum_topK_ngram_p = torch.exp(masked_log_p_ngram).sum(dim=-1)  # (B, S)
        # Residual mass per non-top-K token (uniform spread). Cap
        # sum_topK_ngram_p ≤ 1 − 1/V so log_residual is bounded below by
        # −log(V·(V−K)) ≈ −2·log V (≈ −23 at V=100K). This protects against
        # KN smoothing drift at build time producing sum_topK_ngram_p > 1,
        # which would otherwise make log_residual ≈ log(eps) ≈ −28 at
        # affected positions and create gradient cliffs that the optimizer
        # sees as massive instability. See the matching cap in
        # ``_apply_poe_eval_bias`` for the same rationale.
        sum_topK_ngram_p_capped = sum_topK_ngram_p.clamp_max(1.0 - 1.0 / float(V))
        residual_per_token = (1.0 - sum_topK_ngram_p_capped) / float(V - K)
        log_residual = torch.log(residual_per_token)  # (B, S)

        # Joint log-probabilities at top-K target tokens (unnormalized).
        # For sentinel slots we use -inf so they contribute 0 to sum_joint_topk.
        ngram_log_p_safe = torch.where(
            finite_mask, soft_target_log_probs_f32,
            torch.full_like(soft_target_log_probs_f32, float("-inf")),
        )
        if hasattr(self, "_effective_poe_lambda"):
            poe_lambda = self._effective_poe_lambda(dtype=torch.float32).to(logits_f32.device)
        else:
            poe_lambda = torch.tensor(float(self.poe_lambda), device=logits_f32.device, dtype=torch.float32)
        gathered_joint = gathered_lm_log_probs + poe_lambda * ngram_log_p_safe

        # Compute the joint normalizer Z without materializing (B, S, V):
        #   Z = sum_joint_topk + exp(λ · log_residual) · (1 − sum_lm_topk)
        sum_lm_topk = torch.exp(gathered_lm_log_probs).sum(dim=-1)  # (B, S)
        sum_joint_topk = torch.exp(gathered_joint).sum(dim=-1)      # (B, S)
        non_topk_lm_mass = (1.0 - sum_lm_topk).clamp_min(0.0)
        non_topk_contrib = torch.exp(poe_lambda * log_residual) * non_topk_lm_mass
        Z = sum_joint_topk + non_topk_contrib
        log_Z = torch.log(Z.clamp_min(torch.finfo(Z.dtype).tiny))    # (B, S)

        # Cross-entropy at the hard label. Look up log p_lm(label) directly,
        # then add λ · (log p_ngram[label] if in top-K else log_residual).
        labels_dev = labels.to(logits_f32.device, non_blocking=True)
        safe_labels = labels_dev.clone()
        safe_labels[safe_labels == self.label_ignore_index] = 0
        label_lm_log_probs = logits_f32.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1) - log_sum_exp_lm

        match = soft_target_token_ids == safe_labels.unsqueeze(-1)  # (B, S, K)
        # match must also fall on a finite slot (sentinel tokens default to 0,
        # which can spuriously match label 0); only count finite-slot matches.
        match = match & finite_mask
        any_match = match.any(dim=-1)                                # (B, S)
        # Bias at the gold position: stored log-prob if gold ∈ top-K, else log_residual.
        topk_match_bias = (match.to(soft_target_log_probs_f32.dtype) * ngram_log_p_safe).sum(dim=-1)
        match_bias = torch.where(any_match, topk_match_bias, log_residual)
        label_joint_log_probs = label_lm_log_probs + poe_lambda * match_bias - log_Z

        # NLL per position, masked at ignored labels.
        per_pos_loss = -label_joint_log_probs
        mask = (labels_dev != self.label_ignore_index).to(per_pos_loss.dtype)
        per_pos_loss = per_pos_loss * mask
        # Sum → scalar; divide by the same denominator hard CE uses.
        return per_pos_loss.sum() / loss_div_factor

    def _compute_poe_loss_sb(
        self,
        *,
        logits: torch.Tensor,
        sb_override_batch_idx: torch.Tensor,
        sb_override_position: torch.Tensor,
        sb_override_token_id: torch.Tensor,
        sb_override_log_score: torch.Tensor,
        labels: torch.Tensor,
        loss_div_factor: torch.Tensor,
    ) -> torch.Tensor:
        """Stupid-backoff analog of :meth:`_compute_poe_loss`.

        Unlike the KN-smoothed-top-K path, the SB bias touches *every* w in
        the vocab (the unigram floor as a baseline + sparse overrides for
        observed higher-order (h_k, w) pairs), so we don't have a clean
        gather + logsumexp trick for the joint normalizer Z. Instead we
        materialize the biased logits explicitly via
        :func:`olmo_core.data.sb_bias.apply_sb_bias_inplace` on an fp32
        clone of the LM logits, then call standard cross-entropy.

        The clone is required because the original logits tensor is shared
        with the rest of the forward pass (e.g. ``output.z_loss`` was
        already computed from it inside the LMHead); in-place modification
        would corrupt those readers. Gradient flow goes from the CE through
        the clone (and its broadcast-add + scatter-add ops) back to the
        original logits tensor — autograd handles this correctly because
        every op we apply is differentiable.
        """
        from olmo_core.data.sb_bias import apply_sb_bias_inplace

        local_logits = get_local_tensor(logits)
        # Clone-then-modify so the original LM logits stay intact for any
        # downstream consumer (e.g. z_loss recomputed by the caller).
        biased_logits_f32 = local_logits.float().clone()
        unigram_floor = self._get_poe_sb_unigram_floor_dev(
            dtype=biased_logits_f32.dtype
        )
        assert self.poe_lambda is not None
        apply_sb_bias_inplace(
            biased_logits_f32,
            unigram_floor,
            sb_override_batch_idx,
            sb_override_position,
            sb_override_token_id,
            sb_override_log_score,
            self._effective_poe_lambda(dtype=torch.float32),
        )
        labels_dev = labels.to(biased_logits_f32.device, non_blocking=True)
        flat_logits = biased_logits_f32.flatten(0, 1)  # (B*S, V)
        flat_labels = labels_dev.flatten()             # (B*S,)
        ce_sum = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=self.label_ignore_index,
            reduction="sum",
        )
        return ce_sum / loss_div_factor

    def _set_model_mode(self, mode: Literal["train", "eval"]):
        if self._model_mode != mode:
            if mode == "train":
                self.model.train()
            elif mode == "eval":
                self.model.eval()
            else:
                raise ValueError(f"Invalid model mode: {mode}")
            self._model_mode = mode
