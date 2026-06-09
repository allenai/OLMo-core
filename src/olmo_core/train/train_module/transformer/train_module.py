import contextlib
import logging
import math
from dataclasses import replace
from functools import cached_property, lru_cache
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
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
        poe_lambda: Optional[float] = None,
        poe_lambda_learnable: bool = False,
        poe_lambda_lr: Optional[float] = None,
        poe_lambda_decay_to_zero_windows: Optional[List[Tuple[int, int]]] = None,
        poe_ngram_table_dir: Optional[str] = None,
        poe_ngram_K: int = 16,
        poe_ngram_N_max: int = 5,
        early_fusion_ngram: bool = False,
        early_fusion_alpha_init: float = 0.1,
        early_fusion_alpha_lr: Optional[float] = None,
        early_fusion_ngram_table_dir: Optional[str] = None,
        early_fusion_ngram_K: int = 16,
        early_fusion_ngram_N_max: int = 5,
        early_fusion_engram: bool = False,
        early_fusion_engram_alpha_init: float = 5.0,
        early_fusion_engram_alpha_lr: Optional[float] = None,
        early_fusion_engram_table_dir: Optional[str] = None,
        early_fusion_engram_N_max: int = 5,
        early_fusion_engram_code_dim: int = 32,
        early_fusion_engram_top_m: int = 32,
        early_fusion_engram_vocab_chunk_size: int = 4096,
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

        # Register learned early-fusion alpha before parallelization so FSDP
        # owns it like any other model parameter. It lives on the LM head
        # because the prior is built from LM-head unembedding rows.
        self._early_fusion_alpha_log_name = "early_fusion_alpha_log"
        self._early_fusion_alpha_log_param_name = (
            f"lm_head.{self._early_fusion_alpha_log_name}"
        )
        early_fusion_any = bool(early_fusion_ngram or early_fusion_engram)
        if early_fusion_ngram and early_fusion_engram:
            raise OLMoConfigurationError(
                "early_fusion_ngram and early_fusion_engram are mutually exclusive"
            )
        if early_fusion_engram_alpha_lr is not None and not early_fusion_engram:
            raise OLMoConfigurationError(
                "early_fusion_engram_alpha_lr is only valid when "
                "early_fusion_engram=True"
            )
        effective_early_fusion_alpha_init = (
            early_fusion_engram_alpha_init
            if early_fusion_engram
            else early_fusion_alpha_init
        )
        effective_early_fusion_alpha_lr = (
            early_fusion_engram_alpha_lr
            if early_fusion_engram
            else early_fusion_alpha_lr
        )
        if early_fusion_any:
            if effective_early_fusion_alpha_init <= 0:
                raise OLMoConfigurationError(
                    "early-fusion alpha init must be positive, "
                    f"got {effective_early_fusion_alpha_init}"
                )
            if model.lm_head is None:
                raise OLMoConfigurationError("early fusion requires a model LM head")
            alpha_log = torch.log(
                torch.tensor(
                    [float(effective_early_fusion_alpha_init)],
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            model.lm_head.register_parameter(
                self._early_fusion_alpha_log_name,
                nn.Parameter(alpha_log),
            )
        early_fusion_engram_num_contexts: Optional[int] = None
        if early_fusion_engram:
            if early_fusion_engram_table_dir is None:
                raise OLMoConfigurationError(
                    "early_fusion_engram requires early_fusion_engram_table_dir"
                )
            if early_fusion_engram_code_dim <= 0:
                raise OLMoConfigurationError(
                    "early_fusion_engram_code_dim must be positive, "
                    f"got {early_fusion_engram_code_dim}"
                )
            if early_fusion_engram_top_m <= 0:
                raise OLMoConfigurationError(
                    "early_fusion_engram_top_m must be positive, "
                    f"got {early_fusion_engram_top_m}"
                )
            if early_fusion_engram_top_m > model.vocab_size:
                raise OLMoConfigurationError(
                    "early_fusion_engram_top_m cannot exceed model vocab size "
                    f"({early_fusion_engram_top_m} > {model.vocab_size})"
                )
            if early_fusion_engram_vocab_chunk_size <= 0:
                raise OLMoConfigurationError(
                    "early_fusion_engram_vocab_chunk_size must be positive, "
                    f"got {early_fusion_engram_vocab_chunk_size}"
                )
            from olmo_core.data.ngram_topk import NgramContextSource

            context_source = NgramContextSource(
                table_dir=early_fusion_engram_table_dir,
                N_max=early_fusion_engram_N_max,
            )
            early_fusion_engram_num_contexts = context_source.num_contexts
            context_vocab_size = context_source.vocab_size
            context_source.close()
            if context_vocab_size > model.vocab_size:
                raise OLMoConfigurationError(
                    "Engram context table vocab size exceeds model vocab size "
                    f"({context_vocab_size} > {model.vocab_size})"
                )
            assert model.lm_head is not None
            init_device = model.lm_head.w_out.weight.device
            init_dtype = model.lm_head.w_out.weight.dtype
            model.lm_head.early_fusion_engram_context_codes = nn.Embedding(
                early_fusion_engram_num_contexts,
                early_fusion_engram_code_dim,
                dtype=init_dtype,
                device=init_device,
            )
            model.lm_head.early_fusion_engram_token_decoder = nn.Embedding(
                model.vocab_size,
                early_fusion_engram_code_dim,
                dtype=init_dtype,
                device=init_device,
            )
            model.lm_head.early_fusion_engram_top_m = int(early_fusion_engram_top_m)
            model.lm_head.early_fusion_engram_vocab_chunk_size = int(
                early_fusion_engram_vocab_chunk_size
            )
            model.__dict__.pop("num_params", None)
            model.__dict__.pop("num_non_embedding_params", None)

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
        if early_fusion_any:
            with torch.no_grad():
                self._early_fusion_alpha_log_param().fill_(
                    math.log(float(effective_early_fusion_alpha_init))
                )
        if early_fusion_engram:
            assert early_fusion_engram_num_contexts is not None
            with torch.no_grad():
                self._reset_early_fusion_engram_parameters(
                    code_dim=early_fusion_engram_code_dim
                )
        self._model_mode: Optional[Literal["train", "eval"]] = None

        self._dp_config = dp_config
        self._cp_config = cp_config
        self._tp_config = tp_config
        self._ep_config = ep_config
        self.label_ignore_index = label_ignore_index
        self.z_loss_multiplier = z_loss_multiplier
        self.poe_lambda = poe_lambda
        self.poe_lambda_learnable = bool(poe_lambda_learnable)
        self.poe_lambda_lr = poe_lambda_lr
        self.poe_lambda_decay_to_zero_windows = poe_lambda_decay_to_zero_windows
        if self.poe_lambda is not None:
            if self.poe_lambda <= 0:
                raise OLMoConfigurationError(
                    f"poe_lambda must be positive, got {self.poe_lambda}"
                )
            if tp_config is not None or cp_config is not None:
                raise OLMoConfigurationError(
                    "PoE training is not yet supported with TP or CP"
                )
            if poe_ngram_table_dir is None:
                raise OLMoConfigurationError(
                    "poe_lambda requires poe_ngram_table_dir so the eval path "
                    "can apply the same KN top-k ngram bias as the train path"
                )
        elif self.poe_lambda_learnable:
            raise OLMoConfigurationError("poe_lambda_learnable requires poe_lambda to be set")
        if self.poe_lambda_lr is not None and not self.poe_lambda_learnable:
            raise OLMoConfigurationError("poe_lambda_lr is only valid when poe_lambda_learnable=True")
        if self.poe_lambda_lr is not None and self.poe_lambda_lr <= 0:
            raise OLMoConfigurationError(f"poe_lambda_lr must be positive, got {self.poe_lambda_lr}")
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
        self.early_fusion_ngram = bool(early_fusion_ngram)
        self.early_fusion_alpha_init = early_fusion_alpha_init
        self.early_fusion_alpha_lr = early_fusion_alpha_lr
        self.early_fusion_ngram_table_dir = early_fusion_ngram_table_dir
        self.early_fusion_ngram_K = int(early_fusion_ngram_K)
        self.early_fusion_ngram_N_max = int(early_fusion_ngram_N_max)
        self.early_fusion_engram = bool(early_fusion_engram)
        self.early_fusion_engram_alpha_init = early_fusion_engram_alpha_init
        self.early_fusion_engram_alpha_lr = early_fusion_engram_alpha_lr
        self.early_fusion_engram_table_dir = early_fusion_engram_table_dir
        self.early_fusion_engram_N_max = int(early_fusion_engram_N_max)
        self.early_fusion_engram_code_dim = int(early_fusion_engram_code_dim)
        self.early_fusion_engram_top_m = int(early_fusion_engram_top_m)
        self.early_fusion_engram_vocab_chunk_size = int(
            early_fusion_engram_vocab_chunk_size
        )
        self.early_fusion_engram_num_contexts = (
            None
            if early_fusion_engram_num_contexts is None
            else int(early_fusion_engram_num_contexts)
        )
        if self.early_fusion_ngram:
            if self.poe_lambda is not None:
                raise OLMoConfigurationError(
                    "early_fusion_ngram is a replacement for PoE in this "
                    "experiment; do not set poe_lambda at the same time"
                )
            if tp_config is not None or cp_config is not None:
                raise OLMoConfigurationError(
                    "Early ngram fusion is not yet supported with TP or CP"
                )
            if early_fusion_ngram_table_dir is None:
                raise OLMoConfigurationError(
                    "early_fusion_ngram requires early_fusion_ngram_table_dir "
                    "so eval can apply the same embedding prior as train"
                )
        if self.early_fusion_engram:
            if self.poe_lambda is not None:
                raise OLMoConfigurationError(
                    "early_fusion_engram is a replacement for PoE in this "
                    "experiment; do not set poe_lambda at the same time"
                )
            if tp_config is not None or cp_config is not None:
                raise OLMoConfigurationError(
                    "Early Engram fusion is not yet supported with TP or CP"
                )
        if effective_early_fusion_alpha_lr is not None:
            if not early_fusion_any:
                raise OLMoConfigurationError(
                    "early-fusion alpha LR is only valid when early fusion is enabled"
                )
            if effective_early_fusion_alpha_lr <= 0:
                raise OLMoConfigurationError(
                    "early-fusion alpha LR must be positive, "
                    f"got {effective_early_fusion_alpha_lr}"
                )
        # Lazy: instantiated on first eval_batch call (per process), so we
        # don't open the mmap on the main coordinator rank that may never
        # actually run an eval.
        self._poe_eval_ngram_source = None
        self._early_fusion_eval_ngram_source = None
        self._early_fusion_eval_engram_context_source = None
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

        if early_fusion_any:
            early_fusion_alpha_opts: Dict[str, Any] = {"weight_decay": 0.0}
            if effective_early_fusion_alpha_lr is not None:
                early_fusion_alpha_opts["lr"] = effective_early_fusion_alpha_lr
                early_fusion_alpha_opts["initial_lr"] = effective_early_fusion_alpha_lr
            group_overrides = list(optim.group_overrides or [])
            group_overrides.append(
                OptimGroupOverride(
                    params=[self._early_fusion_alpha_log_param_name],
                    opts=early_fusion_alpha_opts,
                )
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

    def _early_fusion_alpha(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.exp(self._early_fusion_alpha_log_param()).squeeze().to(dtype=dtype)

    def _early_fusion_alpha_for_logging(self) -> torch.Tensor:
        return self._early_fusion_alpha().detach().squeeze()

    def _early_fusion_alpha_log_for_logging(self) -> torch.Tensor:
        return self._early_fusion_alpha_log_param().detach().squeeze()

    def _early_fusion_alpha_log_param(self) -> nn.Parameter:
        lm_head = getattr(self.model, "lm_head", None)
        param = (
            getattr(lm_head, self._early_fusion_alpha_log_name)
            if lm_head is not None and hasattr(lm_head, self._early_fusion_alpha_log_name)
            else getattr(self.model, self._early_fusion_alpha_log_name)
        )
        assert isinstance(param, nn.Parameter)
        return param

    def _early_fusion_enabled(self) -> bool:
        return bool(self.early_fusion_ngram or self.early_fusion_engram)

    def _early_fusion_engram_context_codes(self) -> nn.Embedding:
        lm_head = getattr(self.model, "lm_head", None)
        module = (
            getattr(lm_head, "early_fusion_engram_context_codes")
            if lm_head is not None
            and hasattr(lm_head, "early_fusion_engram_context_codes")
            else getattr(self.model, "early_fusion_engram_context_codes")
        )
        assert isinstance(module, nn.Embedding)
        return module

    def _early_fusion_engram_token_decoder(self) -> nn.Embedding:
        lm_head = getattr(self.model, "lm_head", None)
        module = (
            getattr(lm_head, "early_fusion_engram_token_decoder")
            if lm_head is not None
            and hasattr(lm_head, "early_fusion_engram_token_decoder")
            else getattr(self.model, "early_fusion_engram_token_decoder")
        )
        assert isinstance(module, nn.Embedding)
        return module

    def _reset_early_fusion_engram_parameters(self, *, code_dim: int) -> None:
        std = 1.0 / math.sqrt(float(code_dim))
        self._early_fusion_engram_context_codes().weight.normal_(mean=0.0, std=std)
        self._early_fusion_engram_token_decoder().weight.normal_(mean=0.0, std=std)

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

        # Product-of-experts logit bias. Active when the train module is
        # configured with ``poe_lambda`` and the batch carries KN top-k
        # token IDs plus their full-vocabulary ngram log-probabilities.
        poe_active = (
            self.poe_lambda is not None
            and "ngram_log_probs" in batch
            and "ngram_token_ids" in batch
        )
        if self.early_fusion_ngram and (
            "ngram_log_probs" not in batch or "ngram_token_ids" not in batch
        ):
            raise OLMoConfigurationError(
                "early_fusion_ngram requires batches with ngram_token_ids and "
                "ngram_log_probs; wrap the data source with NgramTopKInstanceSource"
            )
        if self.early_fusion_engram and "engram_context_ids" not in batch:
            raise OLMoConfigurationError(
                "early_fusion_engram requires batches with engram_context_ids; "
                "wrap the data source with NgramContextInstanceSource"
            )

        # Split into micro-batches.
        if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative to sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
        num_micro_batches = len(micro_batches)

        # Train one micro-batch at a time.
        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                # Pop KN top-k fields out of the micro-batch before
                # _prepare_batch, so they don't leak into model.forward(**kwargs).
                # _prepare_batch only moves standard fields to device; we move
                # these tensors here so the loss path's gather doesn't mix CPU
                # indices with CUDA logits.
                ngram_token_ids = micro_batch.pop("ngram_token_ids", None)
                ngram_log_probs = micro_batch.pop("ngram_log_probs", None)
                engram_context_ids = micro_batch.pop("engram_context_ids", None)
                if ngram_token_ids is not None:
                    ngram_token_ids = ngram_token_ids.to(
                        self.device, non_blocking=True
                    )
                if ngram_log_probs is not None:
                    ngram_log_probs = ngram_log_probs.to(
                        self.device, non_blocking=True
                    )
                if engram_context_ids is not None:
                    engram_context_ids = engram_context_ids.to(
                        self.device, non_blocking=True
                    )

                input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)
                if self.early_fusion_ngram:
                    if ngram_token_ids is None or ngram_log_probs is None:
                        raise OLMoConfigurationError(
                            "early_fusion_ngram requires ngram_token_ids and "
                            "ngram_log_probs in every micro-batch"
                        )
                    model_kwargs["early_fusion_ngram_token_ids"] = ngram_token_ids
                    model_kwargs["early_fusion_ngram_log_probs"] = ngram_log_probs
                if self.early_fusion_engram:
                    if engram_context_ids is None:
                        raise OLMoConfigurationError(
                            "early_fusion_engram requires engram_context_ids "
                            "in every micro-batch"
                        )
                    model_kwargs["early_fusion_engram_context_ids"] = engram_context_ids

                # NOTE on gradient flow: when ``return_logits=True`` is set,
                # the LMHead returns ``output.ce_loss`` and ``output.z_loss``
                # as detached tensors (they're meant for metric reporting,
                # not for backprop). Only ``output.loss`` (= ce_loss +
                # z_loss combined) carries gradient back to the model. So
                # we always combine through ``output.loss`` for the gradient
                # path, and only consult ``output.ce_loss`` / ``output.z_loss``
                # for recording metrics.
                if poe_active and ngram_log_probs is not None:
                    # Product-of-experts path: at every position, compute the
                    # model's full-vocabulary log-probability distribution, add
                    # ``λ * log p_ngram`` at the K candidate positions, then
                    # take negative log-likelihood at the hard label. Because
                    # both terms are properly-normalized log-probabilities over
                    # the full vocabulary, the result is the cross-entropy of
                    # the joint distribution
                    #     p_final(w|h) ∝ p_lm(w|h) * p_ngram(w|h)^λ
                    # at the hard label.
                    assert ngram_token_ids is not None
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
                        ngram_token_ids=ngram_token_ids,
                        ngram_log_probs=ngram_log_probs,
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
        if self._early_fusion_enabled():
            self.record_metric(
                "early fusion alpha",
                self._early_fusion_alpha_for_logging(),
                namespace="train",
            )
            self.record_metric(
                "early fusion alpha log",
                self._early_fusion_alpha_log_for_logging(),
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
            biased_logits, biased_ce_loss = self._apply_poe_eval_bias(
                logits=output.logits,
                input_ids=input_ids,
                labels=labels,
                compute_ce_loss=compute_ce_loss,
            )
            return output._replace(logits=biased_logits, ce_loss=biased_ce_loss)

        if self.early_fusion_ngram:
            ngram_token_ids, ngram_log_probs = self._lookup_early_fusion_eval_ngram(
                input_ids=input_ids,
            )
            model_kwargs["early_fusion_ngram_token_ids"] = ngram_token_ids
            model_kwargs["early_fusion_ngram_log_probs"] = ngram_log_probs
        if self.early_fusion_engram:
            engram_context_ids = self._lookup_early_fusion_eval_engram(
                input_ids=input_ids,
            )
            model_kwargs["early_fusion_engram_context_ids"] = engram_context_ids

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
            from olmo_core.data.ngram_topk import NgramTopKSource

            self._poe_eval_ngram_source = NgramTopKSource(
                table_dir=self.poe_ngram_table_dir,
                K=self.poe_ngram_K,
                N_max=self.poe_ngram_N_max,
                output_log_probs=True,
            )
        return self._poe_eval_ngram_source

    def _get_early_fusion_eval_ngram_source(self):
        """Lazy-instantiate the ngram source for eval-time early fusion."""
        if self._early_fusion_eval_ngram_source is None:
            assert self.early_fusion_ngram_table_dir is not None
            from olmo_core.data.ngram_topk import NgramTopKSource

            self._early_fusion_eval_ngram_source = NgramTopKSource(
                table_dir=self.early_fusion_ngram_table_dir,
                K=self.early_fusion_ngram_K,
                N_max=self.early_fusion_ngram_N_max,
                output_log_probs=True,
            )
        return self._early_fusion_eval_ngram_source

    def _lookup_early_fusion_eval_ngram(
        self,
        *,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up per-position top-k fields for eval-time early fusion."""
        prefix_len = self.early_fusion_ngram_N_max - 1
        K = self.early_fusion_ngram_K
        B, S = input_ids.shape
        cpu_input_ids = input_ids.detach().to("cpu").numpy()
        contexts = []
        for b in range(B):
            row = cpu_input_ids[b]
            for s in range(S):
                start = max(0, s + 1 - prefix_len)
                contexts.append(tuple(int(t) for t in row[start : s + 1]))

        ngram_source = self._get_early_fusion_eval_ngram_source()
        ids_np, log_probs_np = ngram_source.lookup_batch(contexts)
        ngram_token_ids = torch.as_tensor(
            ids_np.reshape(B, S, K), dtype=torch.long, device=input_ids.device
        )
        ngram_log_probs = torch.as_tensor(
            log_probs_np.reshape(B, S, K), dtype=torch.float32, device=input_ids.device
        )
        return ngram_token_ids, ngram_log_probs

    def _get_early_fusion_eval_engram_context_source(self):
        """Lazy-instantiate the context source for eval-time Engram fusion."""
        if self._early_fusion_eval_engram_context_source is None:
            assert self.early_fusion_engram_table_dir is not None
            from olmo_core.data.ngram_topk import NgramContextSource

            self._early_fusion_eval_engram_context_source = NgramContextSource(
                table_dir=self.early_fusion_engram_table_dir,
                N_max=self.early_fusion_engram_N_max,
            )
        return self._early_fusion_eval_engram_context_source

    def _lookup_early_fusion_eval_engram(
        self,
        *,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Look up per-position context row IDs for eval-time Engram fusion."""
        prefix_len = self.early_fusion_engram_N_max - 1
        B, S = input_ids.shape
        cpu_input_ids = input_ids.detach().to("cpu").numpy()
        contexts = []
        for b in range(B):
            row = cpu_input_ids[b]
            for s in range(S):
                start = max(0, s + 1 - prefix_len)
                contexts.append(tuple(int(t) for t in row[start : s + 1]))

        context_source = self._get_early_fusion_eval_engram_context_source()
        context_ids_np = context_source.lookup_batch(contexts)
        return torch.as_tensor(
            context_ids_np.reshape(B, S), dtype=torch.long, device=input_ids.device
        )

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
        NgramTopKInstanceSource.__getitem__), looks up per-position
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
        # Same indexing as NgramTopKInstanceSource: at position s, the
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
        ngram_token_ids = torch.as_tensor(
            ids_np.reshape(B, S, K), dtype=torch.long, device=local_logits.device
        )
        ngram_log_probs = torch.as_tensor(
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
        finite_mask = torch.isfinite(ngram_log_probs)
        # Σ exp(log p_ngram) over the finite top-K entries, per position.
        # Sentinel slots contribute 0 because exp(-inf) → 0.
        masked_log_p = torch.where(
            finite_mask, ngram_log_probs,
            torch.full_like(ngram_log_probs, float("-inf")),
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
            ngram_log_probs - log_residual,
            torch.zeros_like(ngram_log_probs),
        )
        if hasattr(self, "_effective_poe_lambda"):
            poe_lambda = self._effective_poe_lambda(dtype=torch.float32).to(local_logits.device)
        else:
            poe_lambda = torch.tensor(float(self.poe_lambda), device=local_logits.device, dtype=torch.float32)
        biased_logits_f32 = local_logits.float().clone()
        biased_logits_f32.scatter_add_(
            -1, ngram_token_ids, poe_lambda * scatter_bias
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
        if self._early_fusion_enabled():
            alpha_log = self._early_fusion_alpha_log_param()
            alpha_log_grad = alpha_log.grad
            self.trainer.record_metric(
                "early fusion alpha log grad",
                (
                    torch.zeros((), device=alpha_log.device, dtype=alpha_log.dtype)
                    if alpha_log_grad is None
                    else self._scalar_metric_tensor(alpha_log_grad)
                ),
                reduce_type=None,
                namespace="optim",
            )
            self.trainer.record_metric(
                "early fusion alpha grad is none",
                float(alpha_log_grad is None),
                reduce_type=None,
                namespace="optim",
            )

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
            if self._early_fusion_enabled():
                alpha_log = self._early_fusion_alpha_log_param()
                alpha_log_grad = alpha_log.grad
                self.trainer.record_metric(
                    "early fusion alpha log grad clipped",
                    (
                        torch.zeros((), device=alpha_log.device, dtype=alpha_log.dtype)
                        if alpha_log_grad is None
                        else self._scalar_metric_tensor(alpha_log_grad)
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

    def _compute_poe_loss(
        self,
        *,
        logits: torch.Tensor,
        ngram_token_ids: torch.Tensor,
        ngram_log_probs: torch.Tensor,
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
        K = ngram_log_probs.size(-1)
        # Per-position log-of-sum-of-exponentials normalizer over the full
        # vocabulary. Shape: (sequences in microbatch, positions per sequence).
        log_sum_exp_lm = torch.logsumexp(logits_f32, dim=-1)
        # LM full-vocabulary log-probabilities at the K target tokens.
        # Shape: (sequences, positions, K).
        gathered_lm_log_probs = logits_f32.gather(-1, ngram_token_ids) - log_sum_exp_lm.unsqueeze(-1)

        # Compute log_residual for the uniform-non-top-K extension.
        ngram_log_probs_f32 = ngram_log_probs.float()
        finite_mask = torch.isfinite(ngram_log_probs_f32)
        # Σ p_ngram over finite top-K entries (sentinels contribute 0 via exp(-inf)).
        masked_log_p_ngram = torch.where(
            finite_mask, ngram_log_probs_f32,
            torch.full_like(ngram_log_probs_f32, float("-inf")),
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
            finite_mask, ngram_log_probs_f32,
            torch.full_like(ngram_log_probs_f32, float("-inf")),
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

        match = ngram_token_ids == safe_labels.unsqueeze(-1)  # (B, S, K)
        # match must also fall on a finite slot (sentinel tokens default to 0,
        # which can spuriously match label 0); only count finite-slot matches.
        match = match & finite_mask
        any_match = match.any(dim=-1)                                # (B, S)
        # Bias at the gold position: stored log-prob if gold ∈ top-K, else log_residual.
        topk_match_bias = (match.to(ngram_log_probs_f32.dtype) * ngram_log_p_safe).sum(dim=-1)
        match_bias = torch.where(any_match, topk_match_bias, log_residual)
        label_joint_log_probs = label_lm_log_probs + poe_lambda * match_bias - log_Z

        # NLL per position, masked at ignored labels.
        per_pos_loss = -label_joint_log_probs
        mask = (labels_dev != self.label_ignore_index).to(per_pos_loss.dtype)
        per_pos_loss = per_pos_loss * mask
        # Sum → scalar; divide by the same denominator hard CE uses.
        return per_pos_loss.sum() / loss_div_factor

    def _set_model_mode(self, mode: Literal["train", "eval"]):
        if self._model_mode != mode:
            if mode == "train":
                self.model.train()
            elif mode == "eval":
                self.model.eval()
            else:
                raise ValueError(f"Invalid model mode: {mode}")
            self._model_mode = mode
