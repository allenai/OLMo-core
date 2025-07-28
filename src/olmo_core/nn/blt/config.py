from dataclasses import dataclass, field
from typing import Optional

from torch import nn

from olmo_core.config import Config
from olmo_core.data.tokenizer import ByteTokenizerConfig

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
    use_oracle_patch_reps: bool = False
    add_boundary_logp: bool = True
    eval_add_boundary_logp: bool = True
    debug_boundary_shift: int = 2
    decoder_backprop_through_encoder: bool = True
    boundary_predictor_backprop_through_encoder: bool = True
    decoder_backprop_through_add_boundary_logp: bool = True


@dataclass
class LocalEncoderConfig(Config):
    hash_byte_group_size: list[int]
    hash_byte_group_vocab: int
    hash_byte_group_nb_functions: int
    sliding_window_size: int
    d_model: int
    n_layers: int
    cross_attn_n_heads: int
    block_config: Config
    add_out_projection: bool = False
    apply_residual_twice: bool = False # for compat with BLT checkpoints

    def build(self, vocab_size: int) -> nn.Module:
        from .local_models import LocalEncoder

        return LocalEncoder(
            vocab_size=vocab_size,
            hash_byte_group_size=self.hash_byte_group_size,
            hash_byte_group_vocab=self.hash_byte_group_vocab,
            hash_byte_group_nb_functions=self.hash_byte_group_nb_functions,
            sliding_window_size=self.sliding_window_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            cross_attn_n_heads=self.cross_attn_n_heads,
            block_config=self.block_config,
            add_out_projection=self.add_out_projection,
            apply_residual_twice=self.apply_residual_twice,
        )


@dataclass
class LocalDecoderConfig(Config):
    sliding_window_size: int
    d_model: int
    n_layers: int
    cross_attn_n_heads: int
    block_config: Config
    add_in_projection: bool = False
    apply_residual_twice: bool = False # for compat with BLT checkpoints

    def build(self, vocab_size: int, d_global_model: int) -> nn.Module:
        from .local_models import LocalDecoder

        return LocalDecoder(
            vocab_size=vocab_size,
            sliding_window_size=self.sliding_window_size,
            d_model=self.d_model,
            d_global_model=d_global_model,
            n_layers=self.n_layers,
            cross_attn_n_heads=self.cross_attn_n_heads,
            block_config=self.block_config,
            add_in_projection=self.add_in_projection,
            apply_residual_twice=self.apply_residual_twice,
        )