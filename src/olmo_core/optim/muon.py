import logging
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Tuple, Type, Union

import torch
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.distributed.parallel import (
    MeshDimName,
    get_dp_model_mesh,
    get_dp_shard_mesh,
    get_world_mesh,
)
from olmo_core.nn.transformer import Transformer
from olmo_core.optim import INITIAL_LR_FIELD, LR_FIELD
from olmo_core.optim.config import OptimConfig, OptimGroupOverride
from olmo_core.utils import move_to_device

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dion import Muon


@dataclass
class MuonConfig(OptimConfig):
    lr: float = 0.01  # Shared lr for Muon and AdamW
    mu: float = 0.95  # momentum for Muon
    betas: Tuple[float, float] = (0.9, 0.95)  # betas for AdamW
    weight_decay: float = 0.1
    nesterov: bool = False
    adjust_lr: Literal["spectral_norm", "rms_norm"] | None = "rms_norm"
    use_triton: bool = False

    @classmethod
    def optimizer(cls) -> Type["Muon"]:
        from dion import Muon

        return Muon

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

        matrix_override = OptimGroupOverride(params=matrix_params, opts=dict())
        vector_override = OptimGroupOverride(params=vector_params, opts=dict(algorithm="adamw"))
        embed_override = OptimGroupOverride(
            params=embed_params, opts=dict(algorithm="adamw", weight_decay=0)
        )
        # lm_head_override = OptimGroupOverride(
        #     params=lm_head_params, opts=dict(algorithm="adamw", lr=self.lr / math.sqrt(model_dim))
        # )
        lm_head_override = OptimGroupOverride(params=lm_head_params, opts=dict(algorithm="adamw"))

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
        log.info(f"World mesh dimensions: {dim_names}")
        log.info(f"World mesh shape: {world_mesh.shape}")
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

    def build(self, model: torch.nn.Module, strict: bool = True) -> "Opt":
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
        optim = self.optimizer()(
            self.build_groups(model, strict=strict),
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
            raise NotImplementedError("Compiling optimizer step is not supported for Muon")

        # Register hook to reset fixed fields after loading a checkpoint.
        def reset_fixed_fields(opt: torch.optim.Optimizer):
            for fixed_fields, group in zip(fixed_fields_per_group, opt.param_groups):
                group.update(fixed_fields)

        optim.register_load_state_dict_post_hook(reset_fixed_fields)

        return optim


class NorMuonConfig(MuonConfig):
    muon_beta2: float = 0.95

    @classmethod
    def optimizer(cls) -> Type["NorMuon"]:
        from dion import NorMuon

        return NorMuon
