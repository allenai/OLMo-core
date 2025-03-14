import math
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module

from olmo_core.distributed.parallel.tensor_parallel import SequenceParallel
from olmo_core.doc_utils import beta_feature

from ..attention import AttentionConfig, RingAttentionLoadBalancerType
from ..buffer_cache import BufferCache
from ..feed_forward import FeedForward, FeedForwardConfig
from ..functional import l2_normalize
from ..layer_norm import LayerNormConfig
from ..moe import MoEConfig, MoERouter
from ..moe.parallel_mlp import ParallelMLPBase
from .config import TransformerDataParallelWrappingStrategy

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType


class TransformerBlockBase(nn.Module):
    """
    Base class for transformer block implementations.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Run the block on the input ``x``.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        raise NotImplementedError

    @abstractmethod
    def apply_cp(self, cp_mesh: DeviceMesh, load_balancer: RingAttentionLoadBalancerType):
        raise NotImplementedError

    def apply_compile(self):
        self.compile(fullgraph=False)

    @abstractmethod
    def apply_fsdp(
        self,
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
        attention: AttentionConfig,
        feed_forward: FeedForwardConfig,
        layer_norm: LayerNormConfig,
        dropout: float = 0.0,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.block_idx = block_idx
        self.attention = attention.build(d_model, init_device=init_device, cache=cache)
        self.attention_norm = layer_norm.build(d_model, init_device=init_device)
        self.feed_forward = feed_forward.build(d_model=d_model, init_device=init_device)
        self.feed_forward_norm = layer_norm.build(d_model, init_device=init_device)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x + self.dropout(self.attention(self.attention_norm(x), **kwargs))
        return h + self.dropout(self.feed_forward(self.feed_forward_norm(h)))

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
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
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fsdp_att = cast(FSDPModule, fully_shard(self.attention, **fsdp_kwargs))
            fsdp_mlp = cast(FSDPModule, fully_shard(self.feed_forward, **fsdp_kwargs))
            fsdp_root = cast(FSDPModule, fully_shard(self, **fsdp_kwargs))
            if prefetch_factor > 0:
                fsdp_root.set_modules_to_forward_prefetch([fsdp_att])
                fsdp_att.set_modules_to_forward_prefetch([fsdp_mlp])
        else:
            fully_shard(self, **fsdp_kwargs)


class ReorderedNormTransformerBlock(TransformerBlock):
    """
    Like :class:`TransformerBlock` except that the attention norm is applied on the output
    of attention instead of the input, and likewise the feed-forward norm is applied on the output
    of the feed-forward instead of the input.
    """

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x + self.dropout(self.attention_norm(self.attention(x, **kwargs)))
        return h + self.dropout(self.feed_forward_norm(self.feed_forward(h)))

    def apply_fsdp(
        self,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fsdp_mlp = cast(FSDPModule, fully_shard(self.feed_forward, **fsdp_kwargs))
            fsdp_root = cast(FSDPModule, fully_shard(self, **fsdp_kwargs))
            if prefetch_factor > 0:
                fsdp_root.set_modules_to_forward_prefetch([fsdp_mlp])
        else:
            fully_shard(self, **fsdp_kwargs)


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
        attention: AttentionConfig,
        feed_forward: FeedForwardConfig,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.block_idx = block_idx
        self.attention = attention.build(d_model, init_device=init_device, cache=cache)
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

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
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

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        del tp_mesh, float8_enabled

        raise NotImplementedError(
            "TP is not implemented yet for the normalized transformer block variant"
        )

    def apply_cp(self, cp_mesh: DeviceMesh, load_balancer: RingAttentionLoadBalancerType):
        self.attention.apply_cp(cp_mesh, load_balancer)

    def apply_fsdp(
        self,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fully_shard(self.feed_forward, **fsdp_kwargs)

        fully_shard(self, **fsdp_kwargs)

        if (
            wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained
            and prefetch_factor > 0
        ):
            cast(FSDPModule, self).set_modules_to_forward_prefetch(
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
        attention: AttentionConfig,
        feed_forward_moe: MoEConfig,
        layer_norm: LayerNormConfig,
        dropout: float = 0.0,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.block_idx = block_idx
        self.attention = attention.build(d_model, init_device=init_device, cache=cache)
        self.attention_norm = layer_norm.build(d_model, init_device=init_device)
        self.feed_forward_moe = feed_forward_moe.build(
            d_model=d_model, init_device=init_device, cache=cache
        )
        self.feed_forward_norm = layer_norm.build(d_model, init_device=init_device)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self._ep_enabled = False
        self._tp_enabled = False

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

    def compute_losses(
        self, total_bz: Union[int, float, torch.Tensor], reset: bool = True
    ) -> Dict[str, torch.Tensor]:
        return self.feed_forward_moe.compute_losses(total_bz, reset=reset)

    def reset_losses(self):
        self.feed_forward_moe.reset_losses()

    def compute_metrics(
        self, total_bz: Union[int, float, torch.Tensor], reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        return self.feed_forward_moe.compute_metrics(total_bz, reset=reset)

    def reset_metrics(self):
        self.feed_forward_moe.reset_metrics()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Run the block on the input ``x``.

        Parameters are the same as :meth:`TransformerBlock.forward()`.
        """
        h = x + self.dropout(self.attention(self.attention_norm(x), **kwargs))
        return h + self.dropout(self.feed_forward_moe(self.feed_forward_norm(h)))

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        self.feed_forward_moe.apply_ep(ep_mesh, **kwargs)
        self._ep_enabled = True

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
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

    def apply_fsdp(
        self,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fsdp_att = cast(FSDPModule, fully_shard(self.attention, **fsdp_kwargs))
            fsdp_moe = cast(FSDPModule, fully_shard(self.feed_forward_moe, **fsdp_kwargs))
            fsdp_root = cast(FSDPModule, fully_shard(self, **fsdp_kwargs))
            if prefetch_factor > 0:
                fsdp_root.set_modules_to_forward_prefetch([fsdp_att])
                fsdp_att.set_modules_to_forward_prefetch([fsdp_moe])
        else:
            fully_shard(self, **fsdp_kwargs)


@beta_feature
class MoEReorderedNormTransformerBlock(MoETransformerBlock):
    """
    Like :class:`MoETransformerBlock` except that the attention norm is applied on the output
    of attention instead of the input, and likewise the feed-forward norm is applied on the
    output of the feed-forward MoE instead of the input.
    """

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x + self.dropout(self.attention_norm(self.attention(x, **kwargs)))
        return h + self.dropout(self.feed_forward_norm(self.feed_forward_moe(h)))

    def apply_fsdp(
        self,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fsdp_moe = cast(FSDPModule, fully_shard(self.feed_forward_moe, **fsdp_kwargs))
            fsdp_root = cast(FSDPModule, fully_shard(self, **fsdp_kwargs))
            if prefetch_factor > 0:
                fsdp_root.set_modules_to_forward_prefetch([fsdp_moe])
        else:
            fully_shard(self, **fsdp_kwargs)


@beta_feature
class MoEParallelTransformerBlockBase(MoETransformerBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_fsdp(
        self,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fsdp_att = cast(FSDPModule, fully_shard(self.attention, **fsdp_kwargs))
            fsdp_shared_mlp = (
                None
                if self.feed_forward_moe.shared_mlp is None
                else cast(FSDPModule, fully_shard(self.feed_forward_moe.shared_mlp, **fsdp_kwargs))
            )
            fsdp_root = cast(FSDPModule, fully_shard(self, **fsdp_kwargs))
            if prefetch_factor > 0:
                if fsdp_shared_mlp is not None:
                    fsdp_root.set_modules_to_forward_prefetch([fsdp_shared_mlp])
                    fsdp_shared_mlp.set_modules_to_forward_prefetch([fsdp_att])
                else:
                    fsdp_root.set_modules_to_forward_prefetch([fsdp_att])
        else:
            fully_shard(self, **fsdp_kwargs)

    def parallel_forward(
        self, x_moe: torch.Tensor, x_att: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # NOTE: this follows the same code path as the MoE's forward pass, except that we run
        # the shared MLP and attention while we wait on expert parallel all-to-all comms.

        expert_logits, expert_scores, expert_weights, expert_indices = self.router(x_moe)

        # shape: (batch_size * top_k,)
        expert_weights = expert_weights.flatten()
        # shape: (batch_size * top_k,)
        expert_indices = expert_indices.flatten()

        with torch.no_grad():
            indices, bin_ids, bins, batch_size_per_expert = self.experts.indices_and_bins(
                expert_indices
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
            parallel_x_handle,
        ) = self.experts.permute_and_all_to_all(
            x_moe.view(-1, x_moe.shape[-1]),
            indices=indices,
            bin_ids=bin_ids,
            bins=bins,
            batch_size_per_expert=batch_size_per_expert,
        )

        # Compute shared out while first all-to-all is in progress.
        shared_out: Optional[torch.Tensor] = None
        if self.shared_mlp is not None:
            shared_out = self.shared_mlp(x_moe)

        parallel_x_handle.wait()
        parallel_x = self.experts.compute_local_experts(
            parallel_x,
            parallel_indices=parallel_indices,
            parallel_bin_ids=parallel_bin_ids,
            parallel_bins=parallel_bins,
            parallel_batch_size_per_expert=parallel_batch_size_per_expert,
            expert_capacity=expert_capacity,
        )

        x_moe, x_handle = self.experts.reverse_all_to_all(
            parallel_x, send_counts=send_counts, recv_counts=recv_counts
        )

        # Compute attention while second all-to-all is in progress.
        x_att = self.attention(x_att, **kwargs)

        x_handle.wait()
        x_moe = self.experts.unpermute(
            x_moe,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            indices=indices,
            bin_ids=bin_ids,
            bins=bins,
        )

        if shared_out is not None:
            shared_out = shared_out / (self.top_k + 1)
            x_moe = shared_out.add(x_moe, alpha=self.top_k / (self.top_k + 1))

        if self.training:
            self.feed_forward_moe.update_losses_and_metrics(
                expert_logits=expert_logits,
                expert_scores=expert_scores,
                expert_weights=expert_weights,
                expert_indices=expert_indices,
                batch_size_per_expert=batch_size_per_expert,
            )

        return x_moe.view(x_att.shape), x_att


@beta_feature
class MoEParallelTransformerBlock(MoEParallelTransformerBlockBase):
    """
    Like :class:`MoETransformerBlock` except that the attention and MLP are done in parallel
    like PaLM instead of in sequence.
    """

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if not self.ep_enabled and not self.tp_enabled:
            return (
                x
                + self.dropout(self.feed_forward_moe(self.feed_forward_norm(x)))
                + self.dropout(self.attention(self.attention_norm(x), **kwargs))
            )

        x_moe = self.feed_forward_norm(x)
        x_att = self.attention_norm(x)
        x_moe, x_att = self.parallel_forward(x_moe, x_att, **kwargs)
        return x + self.dropout(x_moe) + self.dropout(x_att)


@beta_feature
class MoEParallelReorderedNormTransformerBlock(MoEParallelTransformerBlockBase):
    """
    Like :class:`MoEReorderedNormTransformerBlock` except that the attention and MLP are done in parallel
    like PaLM instead of in sequence.
    """

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if not self.ep_enabled and not self.tp_enabled:
            return (
                x
                + self.dropout(self.feed_forward_norm(self.feed_forward_moe(x)))
                + self.dropout(self.attention_norm(self.attention(x, **kwargs)))
            )

        x_moe, x_att = self.parallel_forward(x, x, **kwargs)
        return (
            x
            + self.dropout(self.feed_forward_norm(x_moe))
            + self.dropout(self.attention_norm(x_att))
        )
