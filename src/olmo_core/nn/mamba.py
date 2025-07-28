from dataclasses import dataclass
from mamba_ssm.modules.mamba2 import Mamba2

from olmo_core.config import Config

@dataclass
class MambaConfig(Config):
    chunk_size: int
    d_conv: int
    d_state: int
    expand: int

    def build(self, d_model: int, init_device) -> Mamba2:
        return Mamba2(
            d_model,
            chunk_size=self.chunk_size,
            d_conv=self.d_conv,
            d_state=self.d_state,
            expand=self.expand,
            device=init_device,
        )