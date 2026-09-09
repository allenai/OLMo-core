"""Vision encoder + connector under one module tree (mm_olmo ``vision_backbone`` layout)."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from .connector import VisionConnectorConfig
from .image_vit import VisionTransformer
from .config import VisionEncoderConfig


class VisionBackbone(nn.Module):
    """ViT + vision-to-LM connector, grouped for FSDP nesting like mm_olmo."""

    vision: VisionTransformer
    connector: nn.Module

    def __init__(
        self,
        vision_cfg: VisionEncoderConfig,
        connector_cfg: VisionConnectorConfig,
        *,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.vision = vision_cfg.build(init_device=init_device)
        self.connector = connector_cfg.build(init_device=init_device)

    def apply_fsdp(
        self,
        *,
        dp_mesh: Optional[DeviceMesh] = None,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        reshard_after_forward: bool = True,
    ) -> None:
        """Shard ViT blocks, connector pooling/projector, then the backbone root.

        Matches mm_olmo ``vision_backbone.apply_fsdp2``: one FSDP subtree for the
        full vision path instead of separate sibling units under the multimodal root.
        """
        self.vision.apply_fsdp(
            dp_mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
        if hasattr(self.connector, "apply_fsdp"):
            self.connector.apply_fsdp(
                dp_mesh=dp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
            )
        fsdp_kwargs = {"mesh": dp_mesh, "reshard_after_forward": reshard_after_forward}
        if mp_policy is not None:
            fsdp_kwargs["mp_policy"] = mp_policy
        fully_shard(self, **fsdp_kwargs)
