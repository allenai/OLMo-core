import logging
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Tuple, Union

import torch
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.config import StrEnum
from olmo_core.distributed.parallel import (
    MeshDimName,
    get_dp_model_mesh,
    get_dp_shard_mesh,
    get_world_mesh,
)
from olmo_core.nn.attention import Attention, FusedAttention
from olmo_core.nn.transformer import Transformer

from .config import MatrixAwareOptimConfig, OptimConfig, OptimGroupOverride

log = logging.getLogger(__name__)


def _import_dion():
    try:
        from dion import Muon, NorMuon  # type: ignore
    except ImportError as e:
        raise ImportError(
            "The 'dion' package is required for Muon/NorMuon optimizers. "
            "Install it with: pip install git+https://github.com/microsoft/dion.git"
        ) from e
    return Muon, NorMuon


class MuonAdjustLRStrategy(StrEnum):
    spectral_norm = "spectral_norm"
    """Adjust based on spectral norm, for learning rate transfer across model scale."""

    rms_norm = "rms_norm"
    """Adjust based on RMS norm, for learning rate compatibility with Adam/AdamW (Kimi/Moonlight style: https://arxiv.org/abs/2502.16982)"""


@OptimConfig.register("muon")
@dataclass
class MuonConfig(MatrixAwareOptimConfig):
    """
    Configuration class for building a :class:`Muon` optimizer.

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix.

    Muon is only used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases are optimized using AdamW.

    Muon supports FSDP and HSDP parallelism strategies. Flattened mesh dimensions (eg. "dp_ep"
    and "dp_cp") can be supported but are currently not implemented.
    """

    lr: float = 0.01
    """
    Base learning rate. For Muon, this will be scaled based on the matrix dimensions. For AdamW,
    this is the actual learning rate and no additional scaling is done.
    """

    mu: float = 0.95
    """Momentum for Muon"""

    betas: Tuple[float, float] = (0.9, 0.95)
    """Betas for AdamW"""

    weight_decay: float = 0.1
    """Weight decay factor for non-embedding parameters"""

    cautious_wd: bool = False
    """Whether to apply weight decay only where update and parameter signs align."""

    nesterov: bool = False
    """Whether to use Nesterov momentum."""

    adjust_lr: MuonAdjustLRStrategy | None = MuonAdjustLRStrategy.rms_norm
    """How to adjust the learning rate for Muon updates."""

    flatten: bool = False
    """Whether to flatten 3D+ tensors to 2D for Muon updates. Use this for convolutional layers."""

    use_triton: bool = False
    """
    Whether to use optimized Triton kernels for Newton-Schulz iteration. Becauser the result of X@X.t
    is symmetric, we can avoid computing the upper triangular part of the matrix output.
    See: https://www.lakernewhouse.com/assets/writing/faster-symmul-with-thunderkittens.pdf
    """

    @classmethod
    def optimizer(cls) -> type:
        Muon, _ = _import_dion()
        return Muon

    def default_group_overrides(self, model: torch.nn.Module) -> list[OptimGroupOverride]:
        """
        Split the model parameters into Adam and Muon groups.

        Only >=2d, internal parameters are meant to be optimized with Muon. Query, key, and
        value projection matrices are further grouped by their number of attention heads so that
        Dion applies Newton-Schulz independently to each head.
        """
        assert isinstance(model, Transformer)
        params = self.categorize_parameters(model)

        # Attention projection matrices are optimized independently per head. Dion splits a 2D
        # parameter evenly along dim 0 according to the group's `num_heads` option. A fused Wqkv
        # contains Q, K, and V heads consecutively, hence 3 * n_heads splits.
        attention_params_by_num_heads: dict[int, list[str]] = {}
        matrix_params = set(params["matrix"])

        def add_attention_param(module_name: str, param_name: str, num_heads: int) -> None:
            fqn = f"blocks.{module_name}.{param_name}"
            if num_heads > 1 and fqn in matrix_params:
                attention_params_by_num_heads.setdefault(num_heads, []).append(fqn)
                matrix_params.remove(fqn)

        for module_name, module in model.blocks.named_modules():
            if isinstance(module, Attention):
                add_attention_param(module_name, "w_q.weight", module.n_heads)
                add_attention_param(module_name, "w_k.weight", module.n_kv_heads)
                add_attention_param(module_name, "w_v.weight", module.n_kv_heads)
                add_attention_param(module_name, "w_q_b.weight", module.n_heads)
                add_attention_param(module_name, "w_k_b.weight", module.n_kv_heads)
                add_attention_param(module_name, "w_v_b.weight", module.n_kv_heads)
            elif isinstance(module, FusedAttention):
                add_attention_param(module_name, "w_qkv.weight", 3 * module.n_heads)

        # All other matrix parameters are optimized as whole matrices with Muon.
        matrix_override = OptimGroupOverride(
            params=[name for name in params["matrix"] if name in matrix_params], opts=dict()
        )
        attention_overrides = [
            OptimGroupOverride(params=head_params, opts=dict(num_heads=num_heads))
            for num_heads, head_params in attention_params_by_num_heads.items()
        ]

        # Vector, embedding, and lm_head parameters are optimized with AdamW.
        embed_override = OptimGroupOverride(
            params=params["embed"], opts=dict(algorithm="adamw", weight_decay=0.0)
        )
        vector_override = OptimGroupOverride(params=params["vector"], opts=dict(algorithm="adamw"))
        lm_head_override = OptimGroupOverride(
            params=params["lm_head"], opts=dict(algorithm="adamw")
        )

        return [
            matrix_override,
            *attention_overrides,
            vector_override,
            embed_override,
            lm_head_override,
        ]

    def build_groups(
        self, model: torch.nn.Module, strict: bool = True
    ) -> Union[Iterable[torch.Tensor], list[dict[str, Any]]]:
        """
        Build parameters groups.

        :param model: The model to optimize.
        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """
        all_params: dict[str, torch.Tensor] = OrderedDict()
        frozen_params: set = set()
        for n, p in model.named_parameters():
            if p.requires_grad:
                all_params[n] = p
            else:
                frozen_params.add(n)

        if self.group_overrides is not None:
            raise RuntimeError("group_overrides are not supported for Muon")

        self.group_overrides = self.default_group_overrides(model)

        group_overrides = [
            self._expand_param_globs(go, all_params, frozen_params, g_idx, strict=strict)
            for g_idx, go in enumerate(self.group_overrides or [])
        ]

        # Treat no overrides as its own override group
        overridden_param_names = {name for go in group_overrides for name in go.params}
        default_override = OptimGroupOverride(
            [name for name in all_params.keys() if name not in overridden_param_names], {}
        )
        group_overrides.append(default_override)

        return [
            {"params": [all_params[param_name] for param_name in go.params], **go.opts}
            for go in group_overrides
            if len(go.params) > 0
        ]

    def build_parallelism_config(self) -> dict[str, DeviceMesh | None]:
        """
        Prepare device mesh for Muon optimizer based on the parallelism configuration.

        Muon requires a single 1D DeviceMesh for distributed training:
        - Single-device: Returns None
        - FSDP: Returns the DP mesh (parameter sharding mesh)
        - HSDP: Returns the DP shard mesh (the 1D sharded sub-mesh)

        Note: TP is not directly supported by Muon. For TP configurations,
        you may need to handle tensor parallelism separately.

        :returns: 1D DeviceMesh for distributed Muon, or None for single-device.
        """
        world_mesh = get_world_mesh()
        if world_mesh is None:
            return {"distributed_mesh": None}

        dim_names = world_mesh.mesh_dim_names
        if dim_names is None:
            raise RuntimeError("world mesh has no dimension names")

        # Check for HSDP (has both dp_replicate and dp_shard)
        has_dp_replicate = MeshDimName.dp_replicate in dim_names
        has_dp_shard = MeshDimName.dp_shard in dim_names
        has_tp = MeshDimName.tp in dim_names
        if has_tp:
            raise NotImplementedError("Tensor parallelism is not supported for Muon")

        parallelism_config: dict[str, DeviceMesh | None] = {}

        if has_dp_replicate and has_dp_shard:
            # HSDP configuration: use the shard mesh (1D sharded sub-mesh)
            parallelism_config["distributed_mesh"] = get_dp_shard_mesh(world_mesh)
        elif MeshDimName.dp in dim_names or any(d.startswith("dp") for d in dim_names):
            # FSDP configuration: use the DP mesh
            parallelism_config["distributed_mesh"] = get_dp_model_mesh(world_mesh)

        log.info(f"Muon parallelism_config: {parallelism_config}")
        return parallelism_config

    def create_optimizer(self, model: torch.nn.Module, strict: bool = True, **kwargs):
        # When using Muon, we need to set the recompile limit to 16 to avoid triggering an error
        # due to too many recompile requests. Typically, on the second recompilation, torch attempts
        # to compile a dynamic version of the op, unless dynamic=False is marked. Too many different
        # shapes passed to a compiled op with dynamic=False will trigger this error. Since we have
        # grad matrices with many different shapes, we need to set the recompile limit higher than
        # the default of 8.
        # https://docs.pytorch.org/docs/stable/compile/programming_model.recompilation.html
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 16)

        parallelism_config = self.build_parallelism_config()
        optim = self.optimizer()(
            self.build_groups(model, strict=strict),
            **parallelism_config,
            **kwargs,
        )
        return optim


@OptimConfig.register("nor_muon")
class NorMuonConfig(MuonConfig):
    """
    Configuration class for building a :class:`NorMuon` optimizer.

    NorMuon is a variant of Muon that adds neuron-wise adaptive learning rates.
    https://arxiv.org/abs/2510.05491
    """

    muon_beta2: float = 0.95
    """Beta2 for Muon"""

    @classmethod
    def optimizer(cls) -> type:
        _, NorMuon = _import_dion()
        return NorMuon
