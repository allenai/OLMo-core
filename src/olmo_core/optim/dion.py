import logging
import math
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Tuple, Type, Union, cast

import torch
from dion import Dion
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.distributed.parallel import (
    MeshDimName,
    get_dp_model_mesh,
    get_dp_replicate_mesh,
    get_dp_shard_mesh,
    get_tp_mesh,
    get_world_mesh,
)
from olmo_core.nn.transformer import Transformer
from olmo_core.optim import INITIAL_LR_FIELD, LR_FIELD
from olmo_core.optim.config import OptimConfig, OptimGroupOverride
from olmo_core.utils import move_to_device

log = logging.getLogger(__name__)


@dataclass
class DionConfig(OptimConfig):
    lr: float = 1e-3
    mu: float = 0.95  # momentum for Dion
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    rank_fraction: float = 1.0

    @classmethod
    def optimizer(cls) -> Type[Dion]:
        return Dion

    def default_group_overrides(self, model: torch.nn.Module) -> list[OptimGroupOverride]:
        """
        Split the model parameters into Adam and Muon groups.
        Only >=2d, internal parameters are meant to be optimized with Muon.
        """
        assert isinstance(model, Transformer)
        embed_params = [f"embeddings.{n}" for n, p in model.embeddings.named_parameters()]
        matrix_params = [f"blocks.{n}" for n, p in model.blocks.named_parameters() if p.ndim >= 2]
        vector_params = [f"blocks.{n}" for n, p in model.blocks.named_parameters() if p.ndim < 2]
        vector_params += [f"lm_head.{n}" for n, p in model.lm_head.named_parameters() if p.ndim < 2]
        lm_head_params = [
            f"lm_head.{n}" for n, p in model.lm_head.named_parameters() if p.ndim >= 2
        ]

        lm_head_out: torch.nn.Linear = model.lm_head.w_out
        model_dim = lm_head_out.weight.shape[1]

        matrix_override = OptimGroupOverride(params=matrix_params, opts=dict(algorithm="dion"))
        vector_override = OptimGroupOverride(params=vector_params, opts=dict(algorithm="adamw"))
        embed_override = OptimGroupOverride(
            params=embed_params, opts=dict(algorithm="adamw", weight_decay=0)
        )
        lm_head_override = OptimGroupOverride(
            params=lm_head_params, opts=dict(algorithm="adamw", lr=self.lr / math.sqrt(model_dim))
        )

        return [matrix_override, vector_override, embed_override, lm_head_override]

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
            raise RuntimeError("group_overrides are not supported for Dion")

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
        Prepare device meshes for Dion optimizer based on the parallelism configuration.

        Supports:
        - Single-device: All meshes are None
        - FSDP: replicate_mesh = DP mesh, outer_shard_mesh = None
        - HSDP: replicate_mesh = DP replicate mesh, outer_shard_mesh = DP shard mesh
        - TP: inner_shard_mesh = TP mesh (can be combined with FSDP or HSDP)

        :returns: Dictionary with 'replicate_mesh', 'outer_shard_mesh', and 'inner_shard_mesh' keys.
        """
        world_mesh = get_world_mesh()
        meshes: dict[str, DeviceMesh | None] = {
            "replicate_mesh": None,  # mesh for replicated data parallelism.
            "outer_shard_mesh": None,  # parameter sharding mesh, replicated during orthogonalization
            "inner_shard_mesh": None,  # parameter sharding mesh, remains sharded during orthogonalization
        }

        if world_mesh is None:
            return meshes
        dim_names = world_mesh.mesh_dim_names
        if dim_names is None:
            raise RuntimeError("world mesh has no dimension names")

        # Check for HSDP (has both dp_replicate and dp_shard)
        has_dp_replicate = MeshDimName.dp_replicate in dim_names
        has_dp_shard = MeshDimName.dp_shard in dim_names

        if has_dp_replicate and has_dp_shard:  # HSDP configuration
            meshes["replicate_mesh"] = get_dp_replicate_mesh(world_mesh)
            meshes["outer_shard_mesh"] = get_dp_shard_mesh(world_mesh)
        elif MeshDimName.dp in dim_names or any(d.startswith("dp") for d in dim_names):
            # TODO: is this right?
            # FSDP configuration (has DP dimension but not HSDP split)
            meshes["replicate_mesh"] = get_dp_model_mesh(world_mesh)
            # outer_shard_mesh is None for FSDP (FSDP handles sharding internally)

        if MeshDimName.tp in dim_names:
            meshes["inner_shard_mesh"] = get_tp_mesh(world_mesh)

        return meshes

    def build(self, model: torch.nn.Module, strict: bool = True) -> Dion:
        """
        Build the optimizer.

        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """
        kwargs = self.as_dict(exclude_private_fields=True)
        kwargs.pop("group_overrides")
        kwargs.pop("compile")
        kwargs.pop("fixed_fields")

        torch._dynamo.config.recompile_limit = 16

        parallelism_config = self.build_parallelism_config()
        optim: Dion = self.optimizer()(
            self.build_groups(model, strict=strict),
            replicate_mesh_grad_sync=False,  # HSDP / FSDP / DDP will handle gradient sync internally
            **parallelism_config,
            **kwargs,
        )

        # Set 'lr' and 'initial_lr' in each group if needed.
        fixed_fields_per_group: list[dict[str, Any]] = [{} for _ in optim.param_groups]
        for fixed_fields, group in zip(fixed_fields_per_group, optim.param_groups):
            lr: float | None = None
            if LR_FIELD in group:
                lr = group[LR_FIELD]
            elif hasattr(self, LR_FIELD):
                lr = getattr(self, LR_FIELD)

            if lr is not None:
                if self.compile:
                    # 'lr' should be a tensor.
                    group[LR_FIELD] = move_to_device(torch.tensor(lr), self.device)
                else:
                    group[LR_FIELD] = lr
                group.setdefault(INITIAL_LR_FIELD, lr)

            for k in self.fixed_fields:
                if k in group:
                    fixed_fields[k] = group[k]

        log.info(
            f"Building {self.optimizer().__name__} optimizer with {len(optim.param_groups)} param group(s)..."
        )
        for g_idx, group in enumerate(optim.param_groups):
            group_fields_list = "\n - ".join(
                [f"{k}: {v}" for k, v in optim.param_groups[g_idx].items() if k != "params"]
            )
            if group_fields_list:
                log.info(
                    f"Group {g_idx}, {len(group['params'])} parameter(s):\n - {group_fields_list}"
                )
            else:
                log.info(f"Group {g_idx}, {len(group['params'])} parameter(s)")

        if self.compile:
            raise NotImplementedError("Compiling optimizer step is not supported for Dion")

        # Register hook to reset fixed fields after loading a checkpoint.
        def reset_fixed_fields(opt: torch.optim.Optimizer):
            for fixed_fields, group in zip(fixed_fields_per_group, opt.param_groups):
                group.update(fixed_fields)

        optim.register_load_state_dict_post_hook(reset_fixed_fields)

        return cast(Dion, optim)
