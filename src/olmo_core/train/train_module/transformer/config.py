import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from torch.distributed import DeviceMesh
from torch.distributed.pipelining import PipelineStage

from olmo_core.config import Config, DType
from olmo_core.distributed.parallel import (
    ContextParallelConfig,
    DataParallelConfig,
    ExpertParallelConfig,
    PipelineParallelConfig,
    TensorParallelConfig,
)
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention.ring import (
    RingAttentionLoadBalancerType,
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.transformer import (
    Transformer,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.train_module.config import TrainModuleConfig

if TYPE_CHECKING:
    from .pipeline_train_module import TransformerPipelineTrainModule
    from .train_module import TransformerTrainModule

log = logging.getLogger(__name__)


@beta_feature
@dataclass
class TransformerPipelineParallelConfig(PipelineParallelConfig):
    """
    Transformer-specific pipeline parallel config.
    """

    split_points: Optional[List[int]] = None
    """
    A list of unique, increasing block indices that define how to split the model into stages.

    For example, ``split_points = [0, 2]`` with a 4-layer model means the model will be split into
    3 stages, with the first containing just the embedding, the second containing blocks 0 and 1,
    and the third containing blocks 2 and 3 and the language modeling head.

    If not specified the split points are determined automatically based on the schedule type.
    """

    def get_split_points(self, n_layers: int) -> List[int]:
        if self.split_points is not None:
            return self.split_points

        # Multi-stage schedules support more than 2 stages per rank, but this is the default if
        # no pipeline split is specified.
        num_stages_per_rank = 1 if self.schedule.is_single_stage else 2
        total_stages = self.degree * num_stages_per_rank
        num_layers = n_layers
        if total_stages > num_layers:
            raise OLMoConfigurationError("Total stages cannot be greater than the number of layers")

        base_interval = num_layers // total_stages
        extra_layers = num_layers % total_stages

        splits: List[int] = []
        current_layer = 0
        for i in range(total_stages - 1):
            if i == 0:
                current_layer += base_interval
            else:
                # Middle stages get an extra layer if there are any remaining
                if extra_layers > 0:
                    current_layer += base_interval + 1
                    extra_layers -= 1
                else:
                    current_layer += base_interval
            splits.append(current_layer)
        log.info(f"Auto generated pipeline split points will be {splits}")
        return splits

    def split_model(
        self, model: Transformer, *, pp_mesh: DeviceMesh, device: torch.device
    ) -> Tuple[List[PipelineStage], List[Transformer]]:
        split_points = self.get_split_points(model.n_layers)
        pp_rank = pp_mesh.get_local_rank()

        def build_stage(
            stage_idx: int,
            start_layer: Optional[int],
            stop_layer: Optional[int],
            is_first: bool = False,
            is_last: bool = False,
        ) -> Tuple[PipelineStage, Transformer]:
            model_chunk = copy.deepcopy(model)
            if not is_first:
                model_chunk.embeddings = None  # type: ignore

            drop_layers = start_layer is not None
            for block_idx in range(model.n_layers):
                # we keep layers in a contiguous region between start (inclusive) and stop (exclusive)
                if block_idx == start_layer:
                    drop_layers = False
                if block_idx == stop_layer:
                    drop_layers = True
                if drop_layers:
                    del model_chunk.blocks[str(block_idx)]

            if not is_last:
                model_chunk.lm_head = None  # type: ignore

            stage = PipelineStage(
                model_chunk,
                stage_idx,
                num_stages,
                device,
                group=pp_mesh.get_group("pp"),
            )
            return stage, model_chunk

        num_stages = len(split_points) + 1
        stage_idx = pp_rank

        stages = []
        models = []
        for stage_idx in self.stage_ids_this_rank(pp_rank, num_stages):
            start_layer = split_points[stage_idx - 1] if stage_idx > 0 else None
            stop_layer = split_points[stage_idx] if stage_idx < num_stages - 1 else None
            stage, model_chunk = build_stage(
                stage_idx,
                start_layer,
                stop_layer,
                is_first=stage_idx == 0,
                is_last=stage_idx == num_stages - 1,
            )
            log.info(
                f"PP rank {pp_rank} is building stage {stage_idx} with start layer "
                f"{start_layer}, stop layer {stop_layer}: {model_chunk}"
            )
            stages.append(stage)
            models.append(model_chunk)

        return stages, models


@dataclass
class TransformerDataParallelConfig(DataParallelConfig):
    """
    Transformer-specific data parallel config.
    """

    wrapping_strategy: TransformerDataParallelWrappingStrategy = (
        TransformerDataParallelWrappingStrategy.full
    )
    """
    The wrapping strategy.
    """

    prefetch_factor: int = 0


@dataclass
class TransformerTensorParallelConfig(TensorParallelConfig):
    """
    Transformer-specific tensor parallel config.
    """


@dataclass
class TransformerContextParallelConfig(ContextParallelConfig):
    """
    Transformer-specific context parallel config.
    """

    ring: RingContextParallelStyle | None = None
    uly: UlyssesContextParallelStyle | None = None

    def __post_init__(self):
        if self.ring is not None and self.uly is not None:
            raise NotImplementedError(
                "Only one of ring or ulysses can be specified. While not technically "
                "mutually exclusive, a combined context parallel style is not yet supported."
            )
        elif self.ring is None and self.uly is None:
            raise OLMoConfigurationError("One of ring or uly must be specified")

    @classmethod
    def zig_zag(cls, degree: int, head_stride: int = 1) -> "TransformerContextParallelConfig":
        return cls(
            degree=degree,
            ring=RingContextParallelStyle(
                load_balancer=RingAttentionLoadBalancerType.zig_zag,
                head_stride=head_stride,
            ),
        )

    @classmethod
    def llama3(cls, degree: int, head_stride: int = 1) -> "TransformerContextParallelConfig":
        return cls(
            degree=degree,
            ring=RingContextParallelStyle(
                load_balancer=RingAttentionLoadBalancerType.llama3,
                head_stride=head_stride,
            ),
        )

    @classmethod
    def ulysses(cls, degree: int) -> "TransformerContextParallelConfig":
        return cls(
            degree=degree,
            uly=UlyssesContextParallelStyle(),
        )


@dataclass
class TransformerExpertParallelConfig(ExpertParallelConfig):
    """
    Transformer-specific expert parallel config.
    """


@beta_feature
@dataclass
class TransformerActivationCheckpointingConfig(Config):
    """
    Defines the activation checkpointing strategy for a transformer model.
    """

    mode: TransformerActivationCheckpointingMode = TransformerActivationCheckpointingMode.full
    """
    The activation checkpointing mode.
    """

    block_interval: Optional[int] = None
    """
    Required when :data:`mode` is "selected_blocks". Determines which blocks are wrapped.
    """

    modules: Optional[List[str]] = None
    """
    Required when :data:`mode` is "selected_modules". A list of modules names to wrap for
    activation checkpointing. Globs are supported.
    """

    activation_memory_budget: Optional[float] = None
    """
    Required when :data:`mode` is "budget". Memory budget for activation checkpointing in range [0, 1].
    0 = recompute all activations, 1 = recompute none (default). Requires compilation to be enabled.

    See https://pytorch.org/blog/activation-checkpointing-techniques/ for more details.
    """

    def __post_init__(self):
        if (
            self.mode == TransformerActivationCheckpointingMode.selected_blocks
            and self.block_interval is None
        ):
            raise OLMoConfigurationError(
                "'block_interval' is required for 'selected_blocks' activation checkpointing"
            )
        elif (
            self.mode == TransformerActivationCheckpointingMode.selected_modules
            and self.modules is None
        ):
            raise OLMoConfigurationError(
                "'modules' is required for 'selected_modules' activation checkpointing"
            )


@dataclass
class TransformerTrainModuleConfig(TrainModuleConfig):
    """
    A configuration class for building :class:`TransformerTrainModule` or
    :class:`TransformerPipelineTrainModule` instances.

    .. seealso::
        See the :class:`TransformerTrainModule` and :class:`TransformerPipelineTrainModule`
        documentation for a description of the fields.
    """

    rank_microbatch_size: int
    max_sequence_length: int

    # Optimizer settings.

    optim: OptimConfig
    eval_rank_microbatch_size: Optional[int] = None
    max_grad_norm: Optional[float] = None
    scheduler: Optional[Scheduler] = None

    # Model settings.

    compile_model: bool = False
    float8_config: Optional[Float8Config] = None
    pp_config: Optional[TransformerPipelineParallelConfig] = None
    dp_config: Optional[TransformerDataParallelConfig] = None
    tp_config: Optional[TransformerTensorParallelConfig] = None
    cp_config: Optional[TransformerContextParallelConfig] = None
    ep_config: Optional[TransformerExpertParallelConfig] = None
    ac_config: Optional[TransformerActivationCheckpointingConfig] = None

    # Loss function settings.

    z_loss_multiplier: Optional[float] = None

    # Ngram product-of-experts logit bias. When ``poe_lambda`` is set and the
    # batch carries ``ngram_token_ids`` + ``ngram_log_probs`` emitted by
    # NgramTopKInstanceSource,
    # the training loss becomes the cross-entropy of the joint distribution
    #
    #     log p_final(w|h) = log p_lm(w|h) + λ · log p_ngram(w|h) − log Z(h)
    #
    # at the hard label. The ngram contributes at the K candidate positions
    # for each context; non-top-K positions get no bias. λ is a constant
    # mixing weight. Requires LMHead loss_implementation='default' so logits
    # can be materialized; not yet compatible with TP / CP.
    #
    # ``poe_ngram_table_dir`` / ``poe_ngram_K`` / ``poe_ngram_N_max`` are used
    # at eval time to apply the same bias when the in-loop evaluators run
    # forward — eval batches don't go through the InstanceSource wrapper, so
    # the train module instantiates its own NgramTopKSource and
    # computes per-position contexts on the fly. Required when poe_lambda
    # is set; otherwise the bare-model eval would not match the deployed
    # PoE inference distribution.
    poe_lambda: Optional[float] = None
    # If true, treat ``poe_lambda`` as the positive initialization for a
    # learned scalar. We optimize log(lambda) so the effective mixing weight
    # cannot go negative.
    poe_lambda_learnable: bool = False
    # Optional optimizer LR override for the learned scalar. If unset, it uses
    # the model LR schedule through its own zero-weight-decay parameter group.
    poe_lambda_lr: Optional[float] = None
    # Optional inclusive step windows where the effective PoE lambda is
    # multiplied by a linear 1 -> 0 ramp. The learned/static base lambda is
    # left untouched, so later training resumes from the pre-ramp value.
    poe_lambda_decay_to_zero_windows: Optional[List[Tuple[int, int]]] = None
    poe_ngram_table_dir: Optional[str] = None
    poe_ngram_K: int = 16
    poe_ngram_N_max: int = 5

    # Early fusion with the same Kneser-Ney top-k ngram table. When enabled,
    # the model adds a learned-scale weighted sum of LM-head unembedding rows
    # at the embedding boundary:
    #
    #     h_0(t) = h_token(t) + alpha * Σ_v p_ngram(v | ctx_t) W_out[v]
    #
    # The top-k probabilities are used as raw full-vocabulary KN probability
    # mass, not renormalized over the K candidates. The train loss remains
    # ordinary hard-label CE.
    early_fusion_ngram: bool = False
    early_fusion_alpha_init: float = 0.1
    early_fusion_alpha_lr: Optional[float] = None
    early_fusion_ngram_table_dir: Optional[str] = None
    early_fusion_ngram_K: int = 16
    early_fusion_ngram_N_max: int = 5
    # Engram-style early fusion. This keeps the embedding-boundary injection
    # point but replaces KN continuation probabilities with learned static
    # memory keyed by observed ngram contexts.
    early_fusion_engram: bool = False
    early_fusion_engram_alpha_init: float = 5.0
    early_fusion_engram_alpha_lr: Optional[float] = None
    early_fusion_engram_mode: str = "low_rank_vocab"
    early_fusion_engram_table_dir: Optional[str] = None
    early_fusion_engram_N_max: int = 5
    early_fusion_engram_code_dim: int = 16
    early_fusion_engram_top_m: int = 32
    early_fusion_engram_vocab_chunk_size: int = 4096
    early_fusion_engram_ngram_orders: Tuple[int, ...] = (2, 3)
    early_fusion_engram_heads_per_order: int = 4
    early_fusion_engram_head_dim: int = 4
    early_fusion_engram_slots_per_table: Optional[int] = None
    early_fusion_engram_hash_seed: int = 17

    # Checkpoint settings.

    state_dict_save_opts: Optional[Dict[str, Any]] = None
    state_dict_load_opts: Optional[Dict[str, Any]] = None
    load_key_mapping: Optional[Dict[str, str]] = None

    # Other train settings.

    autocast_precision: Optional[DType] = None
    label_ignore_index: int = -100

    def build(
        self,
        model: Transformer,
        device: Optional[torch.device] = None,
    ) -> Union["TransformerTrainModule", "TransformerPipelineTrainModule"]:
        """
        Build the corresponding :class:`TransformerTrainModule` or :class:`TransformerPipelineTrainModule.

        :param model: The :class:`~olmo_core.nn.transformer.Transformer` model to train.
        :param device: The device to train on.
        """
        from .pipeline_train_module import TransformerPipelineTrainModule
        from .train_module import TransformerTrainModule

        kwargs = self.as_dict(exclude_none=True, recurse=False)
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)
        if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)

        if self.pp_config is not None:
            return TransformerPipelineTrainModule(
                model=model,
                device=device,
                **kwargs,
            )
        else:
            return TransformerTrainModule(
                model=model,
                device=device,
                **kwargs,
            )


@beta_feature
@dataclass
class TransformerPipelineTrainModuleConfig(TransformerTrainModuleConfig):
    """
    Kept for backwards compatibility, but please use :class:`TransformerTrainModuleConfig` instead.
    """

    def __post_init__(self):
        if self.pp_config is None:
            raise OLMoConfigurationError("'pp_config' is required")
