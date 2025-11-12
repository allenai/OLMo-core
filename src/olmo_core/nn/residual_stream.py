import torch
import torch.nn as nn


class ResidualStream(nn.Module):
    """
    A parameter-free module that just handles a residual stream connection, like those in a transformer
    block. The benefit of using this module instead of a direct add operation is that the flexible
    to configure hooks for logging or other purposes, like with the
    :class:`olmo_core.train.callbacks.GAPMonitorCallback`.
    """

    def __init__(self, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, residual: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.add(residual, self.dropout(x), alpha=self.alpha)
