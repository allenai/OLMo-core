from dataclasses import dataclass, field
from typing import Optional

from olmo_core.config import Config
from olmo_core.data import ByteTokenizerConfig

@dataclass
class BLTConfig(Config):
    """Config for distillation into BLT."""
    tokenizer: Optional[ByteTokenizerConfig] = None
    losses: list[str] = field(default_factory=lambda: ["ce"])
    loss_weights: list[float] = field(default_factory=lambda: [1.0])
    binarization_temp: float = 1.0
    div_fn: str = "kl"
    rep_compare_fn: str = "l2"
    epsilon: float = 1e-6
    skip_blocks: bool = False
    skip_teacher: bool = False