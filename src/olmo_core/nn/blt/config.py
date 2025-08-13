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
    temperature: float = 1.0
    div_fn: str = "tvd_temp_limit"
    n_distill_offsets: int = 8
    distill_offset_weights: list[float] = field(default_factory=lambda: [1.0] * 8)
    use_exhaustive_decoder_loss: bool = False
    rep_compare_fn: str = "l2"
    encoder_loss_lookahead: int = 0
    encoder_loss_lookahead_weights: list[float] = field(default_factory=lambda: [])
    patching: str = "dolma2"
    epsilon: float = 1e-6
    skip_blocks: bool = False
    skip_teacher: bool = False
    use_oracle_patch_reps: bool = False
    add_boundary_logp: bool = True
    eval_add_boundary_logp: bool = False
    decoder_backprop_through_encoder: bool = True
    boundary_predictor_backprop_through_encoder: bool = True
    decoder_backprop_through_add_boundary_logp: bool = True
    decoder_use_mse_loss: bool = False
    use_predicted_boundaries: bool = False
    boundary_threshold: float = 0.5
    # stage1 smoothing options
    smooth_encoder_out: bool = False
    smooth_decoder_in: bool = False
    smooth_boundary_means: tuple[float, float] = (0.1, 0.9)
    smooth_boundary_stds: tuple[float, float] = (0.1, 0.1)


@dataclass
class LocalEncoderConfig(Config):
    sliding_window_size: int
    d_model: int
    n_layers: int
    cross_attn_n_heads: int
    block_config: Config
    pooling: str = "cross_attn"
    add_hash_embeddings: bool = True
    hash_byte_group_size: list[int] | None = None
    hash_byte_group_vocab: list[int] | None = None
    hash_byte_group_nb_functions: int | None = None
    add_norm_after_last_block: bool = False
    add_norm_after_pool: bool = False
    add_out_projection: bool = False
    blt_k: Optional[int] = None  # used in blt
    blt_compat: bool = False # for compat with BLT checkpoints

    def build(self, vocab_size: int, d_global_model: int) -> nn.Module:
        from .local_models import LocalEncoder

        return LocalEncoder(
            vocab_size=vocab_size,
            sliding_window_size=self.sliding_window_size,
            d_model=self.d_model,
            d_global_model=d_global_model,
            n_layers=self.n_layers,
            cross_attn_n_heads=self.cross_attn_n_heads,
            block_config=self.block_config,
            add_hash_embeddings=self.add_hash_embeddings,
            hash_byte_group_size=self.hash_byte_group_size,
            hash_byte_group_vocab=self.hash_byte_group_vocab,
            hash_byte_group_nb_functions=self.hash_byte_group_nb_functions,
            pooling=self.pooling,
            add_norm_after_last_block=self.add_norm_after_last_block,
            add_norm_after_pool=self.add_norm_after_pool,
            add_out_projection=self.add_out_projection,
            blt_k=self.blt_k,
            blt_compat=self.blt_compat,
        )


@dataclass
class LocalDecoderConfig(Config):
    sliding_window_size: int
    d_model: int
    n_layers: int
    cross_attn_n_heads: int
    block_config: Config
    depooling: str = "cross_attn"
    add_norm_before_first_block: bool = False
    add_norm_onto_residual: bool = False
    add_in_projection: bool = False
    blt_k: Optional[int] = None  # used in blt
    blt_compat: bool = False # for compat with BLT checkpoints

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
            depooling=self.depooling,
            add_norm_before_first_block=self.add_norm_before_first_block,
            add_norm_onto_residual=self.add_norm_onto_residual,
            add_in_projection=self.add_in_projection,
            blt_k=self.blt_k,
            blt_compat=self.blt_compat,
        )