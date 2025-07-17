import math
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import Placement, Shard
from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module

from olmo_core.distributed.parallel.tensor_parallel import SequenceParallel
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.doc_utils import beta_feature
from olmo_core.ops import attach_auxiliary_loss
from olmo_core.utils import get_or_init_stream
from ..attention import AttentionConfig, RingAttentionLoadBalancerType
from ..buffer_cache import BufferCache
from ..feed_forward import FeedForward, FeedForwardConfig
from ..functional import l2_normalize
from ..layer_norm import LayerNormConfig
from ..moe import MoEConfig, MoERouter
from ..moe.parallel_mlp import ParallelMLPBase, ParallelDroplessMLP
from .config import TransformerDataParallelWrappingStrategy
from torch.utils.checkpoint import checkpoint, CheckpointFunction
from ..moe.utils import async_copy_to_cpu, wait_stream_no_compile

import nvtx
import torch.distributed as dist
from olmo_core.distributed.utils import get_local_rank, get_rank


from transformer_engine.pytorch.permutation import (
    moe_permute,
    moe_sort_chunks_by_index,
    moe_unpermute,
)

from olmo_core.ops import moe as ops


# disable compile for permute
@torch.compiler.disable()
def moe_permute_no_compile(*args, **kwargs):
    return moe_permute(*args, **kwargs)
    
@torch.compiler.disable()
def moe_unpermute_no_compile(*args, **kwargs):
    return moe_unpermute(*args, **kwargs)    

@torch.compiler.disable()
def moe_sort_chunks_by_index_no_compile(*args, **kwargs):
    return moe_sort_chunks_by_index(*args, **kwargs)


if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType


class TransformerBlockBase(nn.Module):
    """
    Base class for transformer block implementations.
    """

    def __init__(self, *, n_layers: int):
        super().__init__()
        self.n_layers = n_layers

    @property
    def is_moe(self) -> bool:
        return False

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run the block on the input ``x``.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        """
        raise NotImplementedError

    def apply_pp(self, pp_mesh: DeviceMesh):
        del pp_mesh

    @abstractmethod
    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        raise NotImplementedError

    @abstractmethod
    def apply_cp(self, cp_mesh: DeviceMesh, load_balancer: RingAttentionLoadBalancerType):
        raise NotImplementedError

    def apply_compile(self):
        self.compile(fullgraph=False)

    @abstractmethod
    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        raise NotImplementedError


class TransformerBlock(TransformerBlockBase):
    """
    A typical "Llama-style" transformer block implementation.

    :param d_model: The model dimensionality.
    :param block_idx: The index/position of the block within the model. Ranges from 0 to ``n_layers - 1``.
    :param attention: The attention module config.
    :param feed_forward: The feed forward module config.
    :param layer_norm: The layer norm config for both the attention LN and the feed forward LN.
    :param dropout: Dropout probability.
    :param init_device: The device used when initializing parameters.
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        attention: AttentionConfig,
        feed_forward: FeedForwardConfig,
        attention_norm: Optional[LayerNormConfig],
        feed_forward_norm: Optional[LayerNormConfig],
        dropout: float = 0.0,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(n_layers=n_layers)
        self.d_model = d_model
        self.block_idx = block_idx
        self.attention = attention.build(
            d_model, layer_idx=block_idx, n_layers=n_layers, init_device=init_device, cache=cache
        )
        if attention_norm is not None:
            self.attention_norm = attention_norm.build(d_model, init_device=init_device)
        else:
            self.attention_norm = lambda x: x # identity
        self.feed_forward = feed_forward.build(d_model=d_model, init_device=init_device)
        if feed_forward_norm is not None:
            self.feed_forward_norm = feed_forward_norm.build(d_model, init_device=init_device)
        else:
            self.feed_forward_norm = lambda x: x # identity
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del loss_div_factor
        h = x + self.dropout(self.attention(self.attention_norm(x), **kwargs))
        return h + self.dropout(self.feed_forward(self.feed_forward_norm(h)))

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=(input_layout,),
                desired_input_layouts=(Shard(1),),
            ),
        )

        parallelize_module(
            self.attention_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )

        self.attention.apply_tp(
            tp_mesh,
            input_layout=Shard(1),
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=float8_enabled,
        )

        parallelize_module(
            self.feed_forward_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )

        self.feed_forward.apply_tp(
            tp_mesh,
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=float8_enabled,
        )

        parallelize_module(self.dropout, device_mesh=tp_mesh, parallelize_plan=SequenceParallel())

    def apply_cp(self, cp_mesh: DeviceMesh, load_balancer: RingAttentionLoadBalancerType):
        self.attention.apply_cp(cp_mesh, load_balancer)

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fsdp_att = cast(FSDPModule, fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs))
            fsdp_mlp = cast(FSDPModule, fully_shard(self.feed_forward, mesh=dp_mesh, **fsdp_kwargs))
            fsdp_root = cast(FSDPModule, fully_shard(self, mesh=dp_mesh, **fsdp_kwargs))
            if prefetch_factor > 0:
                fsdp_root.set_modules_to_forward_prefetch([fsdp_att])
                fsdp_att.set_modules_to_forward_prefetch([fsdp_mlp])
        else:
            fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)


class ReorderedNormTransformerBlock(TransformerBlock):
    """
    Like :class:`TransformerBlock` except that the attention norm is applied on the output
    of attention instead of the input, and likewise the feed-forward norm is applied on the output
    of the feed-forward instead of the input.
    """

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del loss_div_factor
        h = x + self.dropout(self.attention_norm(self.attention(x, **kwargs)))
        return h + self.dropout(self.feed_forward_norm(self.feed_forward(h)))


@beta_feature
class NormalizedTransformerBlock(TransformerBlockBase):
    """
    An nGPT block implementation to be used with the :class:`~olmo_core.nn.attention.NormalizedAttention`
    attention type and :class:`~olmo_core.nn.feed_forward.NormalizedFeedForward` feed-forward type.
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        attention: AttentionConfig,
        feed_forward: FeedForwardConfig,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(n_layers=n_layers)
        self.d_model = d_model
        self.block_idx = block_idx
        self.attention = attention.build(
            d_model, layer_idx=block_idx, n_layers=n_layers, init_device=init_device, cache=cache
        )
        self.feed_forward = feed_forward.build(d_model=d_model, init_device=init_device)

        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = 1.0 / math.sqrt(d_model)
        self.attn_alpha = nn.Parameter(
            torch.empty(d_model, dtype=torch.float32, device=init_device)
        )

        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = 1.0 / math.sqrt(d_model)
        self.mlp_alpha = nn.Parameter(torch.empty(d_model, dtype=torch.float32, device=init_device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.attn_alpha)
        nn.init.ones_(self.mlp_alpha)
        with torch.no_grad():
            self.attn_alpha.mul_(self.attn_alpha_init_scaling)
            self.mlp_alpha.mul_(self.mlp_alpha_init_scaling)

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del loss_div_factor
        h = l2_normalize(
            torch.lerp(
                x,
                l2_normalize(self.attention(x, **kwargs)),
                (
                    self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
                ).abs(),
            )
        )

        return l2_normalize(
            torch.lerp(
                h,
                l2_normalize(self.feed_forward(h)),
                (self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)).abs(),
            )
        )

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        del tp_mesh, input_layout, float8_enabled

        raise NotImplementedError(
            "TP is not implemented yet for the normalized transformer block variant"
        )

    def apply_cp(self, cp_mesh: DeviceMesh, load_balancer: RingAttentionLoadBalancerType):
        self.attention.apply_cp(cp_mesh, load_balancer)

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs)
            fully_shard(self.feed_forward, mesh=dp_mesh, **fsdp_kwargs)

        fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)

        if (
            wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained
            and prefetch_factor > 0
        ):
            cast(FSDPModule, self).set_modules_to_forward_prefetch(
                [cast(FSDPModule, self.attention)]
            )
            cast(FSDPModule, self.attention).set_modules_to_forward_prefetch(
                [cast(FSDPModule, self.feed_forward)]
            )

    @torch.no_grad()
    def normalize_matrices(self):
        """
        Normalize the weights in all matrices. This should be called after each optimizer step, which
        the :class:`~olmo_core.train.train_module.TransformerTrainModule` will handle for you.
        """
        if hasattr(self.attention, "normalize_matrices"):
            self.attention.normalize_matrices()  # type: ignore

        if hasattr(self.feed_forward, "normalize_matrices"):
            self.feed_forward.normalize_matrices()  # type: ignore

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(l2_normalize(w, dim=dim))


@beta_feature
class MoETransformerBlock(TransformerBlockBase):
    """
    Like :class:`TransformerBlock` except that the dense :class:`~olmo_core.nn.feed_forward.FeedForward`
    module is replaced with a mixture-of-experts (MoE).
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        attention: AttentionConfig,
        feed_forward_moe: MoEConfig,
        attention_norm: Optional[LayerNormConfig] = None,
        feed_forward_norm: Optional[LayerNormConfig] = None,
        dropout: float = 0.0,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(n_layers=n_layers)
        self.d_model = d_model
        self.block_idx = block_idx
        self.attention = attention.build(
            d_model, layer_idx=block_idx, n_layers=n_layers, init_device=init_device, cache=cache
        )
        if attention_norm is not None:
            self.attention_norm = attention_norm.build(d_model, init_device=init_device)
        else:
            self.attention_norm = lambda x: x # identity
        self.feed_forward_moe = feed_forward_moe.build(
            d_model=d_model, n_layers=n_layers, init_device=init_device, cache=cache
        )
        if feed_forward_norm is not None:
            self.feed_forward_norm = feed_forward_norm.build(d_model, init_device=init_device)
        else:
            self.feed_forward_norm = lambda x: x # identity
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self._ep_enabled = False
        self._tp_enabled = False

    @property
    def is_moe(self) -> bool:
        return True

    @property
    def router(self) -> MoERouter:
        return self.feed_forward_moe.router

    @property
    def shared_mlp(self) -> Optional[FeedForward]:
        return self.feed_forward_moe.shared_mlp

    @property
    def experts(self) -> ParallelMLPBase:
        return self.feed_forward_moe.experts

    @property
    def top_k(self) -> int:
        return self.feed_forward_moe.top_k

    @property
    def ep_enabled(self) -> bool:
        return self._ep_enabled

    @property
    def tp_enabled(self) -> bool:
        return self._tp_enabled

    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        return self.feed_forward_moe.compute_metrics(reset=reset)

    def reset_metrics(self):
        self.feed_forward_moe.reset_metrics()

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        h = x + self.dropout(self.attention(self.attention_norm(x), **kwargs))
        return h + self.dropout(
            self.feed_forward_moe(self.feed_forward_norm(h), loss_div_factor=loss_div_factor)
        )

    def apply_pp(self, pp_mesh: DeviceMesh):
        self.feed_forward_moe.apply_pp(pp_mesh)

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        self.feed_forward_moe.apply_ep(ep_mesh, **kwargs)
        self._ep_enabled = True

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=(input_layout,),
                desired_input_layouts=(Shard(1),),
            ),
        )

        parallelize_module(
            self.attention_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )

        self.attention.apply_tp(
            tp_mesh,
            input_layout=Shard(1),
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=float8_enabled,
        )

        parallelize_module(
            self.feed_forward_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )

        self.feed_forward_moe.apply_tp(
            tp_mesh,
            input_layout=Shard(1),
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=float8_enabled,
        )

        parallelize_module(self.dropout, device_mesh=tp_mesh, parallelize_plan=SequenceParallel())

        self._tp_enabled = True

    def apply_cp(self, cp_mesh: DeviceMesh, load_balancer: RingAttentionLoadBalancerType):
        self.attention.apply_cp(cp_mesh, load_balancer)
        self.feed_forward_moe.apply_cp(cp_mesh)

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fsdp_att = cast(FSDPModule, fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs))
            fsdp_moe = cast(
                FSDPModule, fully_shard(self.feed_forward_moe, mesh=dp_mesh, **fsdp_kwargs)
            )
            fsdp_root = cast(FSDPModule, fully_shard(self, mesh=dp_mesh, **fsdp_kwargs))
            if prefetch_factor > 0:
                fsdp_root.set_modules_to_forward_prefetch([fsdp_att])
                fsdp_att.set_modules_to_forward_prefetch([fsdp_moe])
        else:
            fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)


@beta_feature
class MoEReorderedNormTransformerBlock(MoETransformerBlock):
    """
    Like :class:`MoETransformerBlock` except that the attention norm is applied on the output
    of attention instead of the input, and likewise the feed-forward norm is applied on the
    output of the feed-forward MoE instead of the input.
    """
    @torch.compile
    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        h = x + self.dropout(self.attention_norm(self.attention(x, **kwargs)))
        return h + self.dropout(
            self.feed_forward_norm(self.feed_forward_moe(h, loss_div_factor=loss_div_factor))
        )

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fsdp_att = cast(FSDPModule, fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs))
            fsdp_moe = cast(
                FSDPModule, fully_shard(self.feed_forward_moe, mesh=dp_mesh, **fsdp_kwargs)
            )
            fsdp_root = cast(FSDPModule, fully_shard(self, mesh=dp_mesh, **fsdp_kwargs))
            if prefetch_factor > 0:
                fsdp_root.set_modules_to_forward_prefetch([fsdp_att])
                fsdp_att.set_modules_to_forward_prefetch([fsdp_moe])
        else:
            fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)


@beta_feature
class MoEHybridTransformerBlockBase(MoETransformerBlock):
    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int,
        feed_forward_norm: LayerNormConfig,
        attention_norm: LayerNormConfig,
        feed_forward: FeedForwardConfig,
        init_device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            n_layers=n_layers,
            # layer_norm=layer_norm, # note: split into attention and feed_forward norms
            feed_forward_norm=feed_forward_norm,
            attention_norm=attention_norm,
            init_device=init_device,
            **kwargs,
        )
        self.feed_forward = feed_forward.build(d_model=d_model, init_device=init_device)
        self.feed_forward_moe_norm = feed_forward_norm.build(d_model, init_device=init_device)
        self._use_combined_forward: Optional[bool] = None

    @property
    def use_combined_forward(self) -> bool:
        if self._use_combined_forward is not None:
            return self._use_combined_forward
        elif not self.ep_enabled and not self.tp_enabled:
            return False
        else:
            return True

    @use_combined_forward.setter
    def use_combined_forward(self, should_use: bool):
        if should_use and not (self.tp_enabled or self.ep_enabled):
            raise RuntimeError(
                "combined forward can only be used when expert parallelism is enabled"
            )
        self._use_combined_forward = should_use

    @abstractmethod
    def dense_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def sparse_forward(
        self, x: torch.Tensor, *, loss_div_factor: Optional[Union[torch.Tensor, float]] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def combined_forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def overlap_forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError
    

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # HACK: note tested
        if not self.use_combined_forward:
        
            return self.sparse_forward(x, loss_div_factor=loss_div_factor) + self.dense_forward(
                x, **kwargs
            )
        else:
            # NOTE: alternatively could do something like this, but even with an extra stream it's
            # not as fast as the hand-crafted 'combined_forward()'.
            # stream = get_or_init_stream()
            # stream.wait_stream(torch.cuda.default_stream())
            # h_sparse = self._fwd_sparse(x)
            # with torch.cuda.stream(stream):
            #     h_dense = self._fwd_dense(x, **kwargs)
            # torch.cuda.default_stream().wait_stream(stream)
            # return h_sparse + h_dense
            
            
            return cast(MoEHybridReorderedNormTransformerBlock, self).combined_forward(x, loss_div_factor=loss_div_factor, **kwargs)


    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        super().apply_tp(tp_mesh, input_layout=input_layout, float8_enabled=float8_enabled)

        self.feed_forward.apply_tp(
            tp_mesh,
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=float8_enabled,
        )

        parallelize_module(
            self.feed_forward_moe_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        from torch.distributed.fsdp import MixedPrecisionPolicy

        # Force router to be full-precision.
        fsdp_router = cast(
            FSDPModule,
            fully_shard(
                self.feed_forward_moe.router,
                mesh=dp_mesh,
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32),
            ),
        )

        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            if not self.use_combined_forward:
                fsdp_att = cast(
                    FSDPModule, fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs)
                )
                fsdp_mlp = cast(
                    FSDPModule, fully_shard(self.feed_forward, mesh=dp_mesh, **fsdp_kwargs)
                )
                fsdp_moe = cast(
                    FSDPModule, fully_shard(self.feed_forward_moe, mesh=dp_mesh, **fsdp_kwargs)
                )
                fsdp_root = cast(FSDPModule, fully_shard(self, mesh=dp_mesh, **fsdp_kwargs))
                if prefetch_factor > 0:
                    fsdp_root.set_modules_to_forward_prefetch([fsdp_router, fsdp_moe, fsdp_att])
                    fsdp_att.set_modules_to_forward_prefetch([fsdp_mlp])
            else:
                fsdp_att = cast(
                    FSDPModule, fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs)
                )
                fsdp_mlp = cast(
                    FSDPModule, fully_shard(self.feed_forward, mesh=dp_mesh, **fsdp_kwargs)
                )
                #  fsdp_moe = cast(
                #      FSDPModule,
                #      fully_shard(self.feed_forward_moe.experts.mlp, mesh=dp_mesh, **fsdp_kwargs),
                #  )
                fsdp_shared_mlp = (
                    None
                    if self.feed_forward_moe.shared_mlp is None
                    else cast(
                        FSDPModule,
                        fully_shard(self.feed_forward_moe.shared_mlp, mesh=dp_mesh, **fsdp_kwargs),
                    )
                )
                fsdp_root = cast(FSDPModule, fully_shard(self, mesh=dp_mesh, **fsdp_kwargs))

                if prefetch_factor > 0:
                    #  fsdp_root.set_modules_to_forward_prefetch([fsdp_att, fsdp_moe])
                    fsdp_root.set_modules_to_forward_prefetch([fsdp_att, fsdp_router])
                    if fsdp_shared_mlp is not None:
                        fsdp_att.set_modules_to_forward_prefetch([fsdp_mlp, fsdp_shared_mlp])
                    else:
                        fsdp_att.set_modules_to_forward_prefetch([fsdp_mlp])
        else:
            fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)


@beta_feature
class MoEHybridTransformerBlock(MoEHybridTransformerBlockBase):
    def dense_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x + self.dropout(self.attention(self.attention_norm(x), **kwargs))
        return h + self.dropout(self.feed_forward(self.feed_forward_norm(h)))

    def sparse_forward(
        self, x: torch.Tensor, *, loss_div_factor: Optional[Union[torch.Tensor, float]] = None
    ) -> torch.Tensor:
        return self.dropout(
            self.feed_forward_moe(self.feed_forward_moe_norm(x), loss_div_factor=loss_div_factor)
        )

    def combined_forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # NOTE: this follows the same code path as the MoE's forward pass, except that we run
        # dense operations while we wait on expert parallel all-to-all comms.
        B, _, D = x.shape

        x_moe = get_local_tensor(self.feed_forward_moe_norm(x))

        expert_weights, expert_indices, batch_size_per_expert, router_aux_loss = self.router(
            x_moe, loss_div_factor=loss_div_factor
        )

        if router_aux_loss is not None:
            x_moe = attach_auxiliary_loss(x_moe, router_aux_loss)

        # shape: (batch_size * seq_len, d_model)
        x_moe = x_moe.view(-1, D)
        # shape: (batch_size * top_k,)
        expert_weights = expert_weights.flatten()
        # shape: (batch_size * top_k,)
        expert_indices = expert_indices.flatten()

        with torch.no_grad():
            indices, bin_ids, bins = self.experts.indices_and_bins(
                expert_indices, batch_size_per_expert
            )

        (
            parallel_x,
            parallel_indices,
            parallel_bin_ids,
            parallel_bins,
            parallel_batch_size_per_expert,
            recv_counts,
            send_counts,
            expert_capacity,
            handle,
        ) = self.experts.permute_and_all_to_all(
            x_moe,
            indices=indices,
            bin_ids=bin_ids,
            bins=bins,
            batch_size_per_expert=batch_size_per_expert,
        )

        # Compute attention while all-to-all is in progress.
        h = x + self.dropout(self.attention(self.attention_norm(x), **kwargs))

        # Maybe compute MoE shared out while all-to-all is in progress.
        moe_shared_out: Optional[torch.Tensor] = None
        if self.shared_mlp is not None:
            # NOTE: -1 on seq dim in case of TP
            moe_shared_out = self.shared_mlp(x_moe.view(B, -1, D))

        handle.wait()
        parallel_x = self.experts.compute_local_experts(
            parallel_x,
            parallel_indices=parallel_indices,
            parallel_bin_ids=parallel_bin_ids,
            parallel_bins=parallel_bins,
            parallel_batch_size_per_expert=parallel_batch_size_per_expert,
            expert_capacity=expert_capacity,
        )

        x_moe, handle = self.experts.reverse_all_to_all(
            parallel_x, send_counts=send_counts, recv_counts=recv_counts
        )

        # Compute feed-forward while all-to-all is in progress.
        h = h + self.dropout(self.feed_forward(self.feed_forward_norm(h)))

        handle.wait()
        x_moe = self.experts.unpermute(
            x_moe,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            indices=indices,
            bin_ids=bin_ids,
            bins=bins,
        ).view(B, -1, D)

        if moe_shared_out is not None:
            moe_shared_out = moe_shared_out / (self.top_k + 1)
            x_moe = moe_shared_out.add(x_moe, alpha=self.top_k / (self.top_k + 1))

        return h + self.dropout(x_moe)

@beta_feature
class MoEHybridReorderedNormTransformerBlock(MoEHybridTransformerBlockBase):

    def __init__(self,
        *,
        d_model: int,
        n_layers: int,
        feed_forward_norm: LayerNormConfig,
        attention_norm: LayerNormConfig,
        feed_forward: FeedForwardConfig,
        init_device: str = "cpu",
        **kwargs,
        ):
        super().__init__(
            d_model=d_model,
            n_layers=n_layers,
            feed_forward_norm=feed_forward_norm,
            attention_norm=attention_norm,
            feed_forward=feed_forward,
            init_device=init_device,
            **kwargs,
        )
        # self.dense_stream = get_or_init_stream(id=2, priority=20) # positive number -> low priority # does not work with pp split_model

    def get_dense_stream(self) -> torch.cuda.Stream:
        return get_or_init_stream(id=2, priority=20)

    def dense_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x + self.dropout(self.attention_norm(self.attention(x, **kwargs)))
        return h + self.dropout(self.feed_forward_norm(self.feed_forward(h)))

    def sparse_forward(
        self, x: torch.Tensor, *, loss_div_factor: Optional[Union[torch.Tensor, float]] = None
    ) -> torch.Tensor:
        return self.dropout(
            self.feed_forward_moe_norm(self.feed_forward_moe(x, loss_div_factor=loss_div_factor))
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if not self.use_combined_forward:
            return self.sparse_forward(x, loss_div_factor=loss_div_factor) + self.dense_forward(
                x, **kwargs
            )
        else:
            # NOTE: alternatively could do something like this, but even with an extra stream it's
            # not as fast as the hand-crafted 'combined_forward()'.
            # stream = get_or_init_stream()
            # stream.wait_stream(torch.cuda.default_stream())
            # h_sparse = self._fwd_sparse(x)
            # with torch.cuda.stream(stream):
            #     h_dense = self._fwd_dense(x, **kwargs)
            # torch.cuda.default_stream().wait_stream(stream)
            # return h_sparse + h_dense

            return cast(MoEHybridReorderedNormTransformerBlock, self).combined_forward(x, loss_div_factor=loss_div_factor, **kwargs)

    @torch.compile
    def dense_forward_rc(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        h = x + self.dropout(self.attention_norm(self.attention(x, **kwargs)))
        return h + self.dropout(self.feed_forward_norm(self.feed_forward(h)))   
    
    @torch.compile
    def router_forward(
        self,
        local_x: torch.Tensor,
        loss_div_factor: torch.Tensor,
    ):
        return self.router(
            local_x, 
            loss_div_factor=loss_div_factor # scalar
        )
    
    @torch.compile
    def combined_forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:

        B, S, D = x.shape
        
        
        
        # dense_out = self.overlap_callback(
        #     x, **kwargs
        # )
        
        local_x = get_local_tensor(x)
    
            
        (
            local_x_global_expert_weights, # (B, S, top_k)
            local_x_global_expert_indices, # (B, S, top_k)
            local_batch_size_per_global_expert, # (num_experts, )
            router_aux_loss # scalar
        ) = self.router_forward(
            local_x, 
            loss_div_factor=loss_div_factor # scalar
        )
        
        
        ##### DENSE: submit the dense forward pass to a separate stream #####
        # NOTE: launch dense after the router, so it's more likely to overlap permute and all-to-all,
        # but it only depends on the x (before router), so the wait_stream() can be called here
        # priority sparse forward > dense forward, because the sparse one is going to take longer
        
        wait_stream_no_compile(
            this_stream=self.get_dense_stream(),  # type: ignore
            other_stream=torch.cuda.current_stream() # type: ignore
        ) # ignore
        
        # only when grad enabled
        if torch.is_grad_enabled():
            with nvtx.annotate("attach_auxiliary_loss", color="blue"):
                if router_aux_loss is not None:
                    local_x = attach_auxiliary_loss(local_x, router_aux_loss)


        # shape: (batch_size * seq_len, d_model)
        local_x = local_x.view(-1, D)
        # shape: (batch_size * seq_len * top_k,)
        local_x_global_expert_weights = get_local_tensor(local_x_global_expert_weights).flatten()
        # shape: (batch_size * seq_len * top_k,)
        local_x_global_expert_indices = get_local_tensor(local_x_global_expert_indices).flatten()


        assert self.shared_mlp is None
        
        
        ######################################################################
        ################# Call global_permute_mlp_unpermute() ##############
        ############################### START ################################


        with nvtx.annotate("permute_mlp_unpermute", color="blue"):
            if self.ep_enabled:
                x_moe, dense_out = cast(ParallelDroplessMLP, self.experts).global_permute_mlp_unpermute(
                    local_x,
                    local_x_global_expert_weights,
                    local_x_global_expert_indices,
                    local_batch_size_per_global_expert=local_batch_size_per_global_expert,
                    overlap_callback=self.overlap_callback,
                    # overlap_callback=None,
                    overlap_callback_x=x,
                    **kwargs,
                ) # type: ignore
            else:
                x_moe, dense_out = cast(ParallelDroplessMLP, self.experts).global_permute_mlp_unpermute_no_ep(
                    local_x,
                    local_x_global_expert_weights,
                    local_x_global_expert_indices, 
                    local_batch_size_per_global_expert=local_batch_size_per_global_expert,
                    overlap_callback=self.overlap_callback,
                    # overlap_callback=None,
                    overlap_callback_x=x,
                    **kwargs,
                ) # type: ignore
        ######################################################################
        ################# Call global_permute_mlp_unpermute() ##############
        ############################### END ##################################

        
        x_moe = x_moe.view(B, -1, D) # (B, S * top_k, d_model)

        

        
        ####################### # NOTE: fuse the add, but it takes more memory. Why?
        # wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream()) # type: ignore
        # # need to use dense_out
        
        # SPARSE_DROP_NORM_RECOMPUTE = True
        # if SPARSE_DROP_NORM_RECOMPUTE:
        #     final_out = checkpoint(
        #         self.sparse_add_drop_norm_forward, x_moe, dense_out, use_reentrant=False
        #     )
        # else:
        #     final_out = self.sparse_add_drop_norm_forward(
        #         x_moe, dense_out
        #     )
            
        #######################
        SPARSE_DROP_NORM_RECOMPUTE = True
        if SPARSE_DROP_NORM_RECOMPUTE:
            final_out = checkpoint(
                self.sparse_drop_norm_forward, x_moe, use_reentrant=False
            )
        else:
            final_out = self.sparse_drop_norm_forward(
                x_moe
            )
        
        wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream()) # type: ignore
        
        final_out = dense_out + final_out

        #######################

        
        return final_out

    @torch.compile
    def sparse_add_drop_norm_forward(
        self,
        x_moe: torch.Tensor,
        dense_out: torch.Tensor,
    ) -> torch.Tensor:
        return self.dropout(self.feed_forward_moe_norm(x_moe)) + dense_out
    
    @torch.compile
    def sparse_drop_norm_forward(
        self,
        x_moe: torch.Tensor,
    ) -> torch.Tensor:
        return self.dropout(self.feed_forward_moe_norm(x_moe))
    
    
    def overlap_callback(self, x, **kwargs):
        # NOTE: this is called in the middle of the global_permute_mlp_unpermute() function
        # to overlap the dense forward pass with the all-to-all communication.
        # It is called after the local_x_global_expert_weights and local_x_global_expert_indices
        # are ready, but before the all-to-all communication starts.
        with torch.cuda.stream(self.get_dense_stream()):
            # if self.block_idx % 2 == 0:
            # x = self.attention(x, **kwargs)
            if True:
                xx = checkpoint(
                    self.dense_forward_rc, x, use_reentrant=False, **kwargs
                )
            else:
                xx = self.dense_forward_rc(
                    x, **kwargs
                )
            return xx