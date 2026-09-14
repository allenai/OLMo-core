"""Shared, dependency-free geometries for the September 2026 partner family."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    """Dimensions and independently checked, embedding-inclusive parameter counts."""

    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    expert_hidden_size: int
    num_routed_experts: int
    latent_dim: int
    expected_total_params: int
    expected_active_params: int
    attention_period: int = 8

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        """Zero-based gated full-attention block indices."""
        return tuple(range(self.attention_period - 1, self.n_layers, self.attention_period))


FAMILY = {
    "tiny": Geometry(1024, 16, 8, 4, 128, 1024, 512, 512, 12_496_341_632, 794_233_472),
    # The previous 3.781B-active production Large, including the QK-gain update.
    "small": Geometry(1536, 40, 16, 8, 128, 1536, 512, 768, 72_237_847_936, 3_780_515_200),
    "medium": Geometry(2560, 64, 24, 12, 128, 2560, 512, 1280, 322_601_566_720, 15_421_227_520),
    "large": Geometry(4608, 80, 48, 24, 128, 4608, 512, 2304, 1_310_163_554_560, 62_133_719_296),
}

# A legacy, deliberately unaligned installation check, NOT a fifth family member.
# The extra scalar compared with the old smoke is its scalable-softmax head gain.
GEOMETRIES = {
    "30m": Geometry(128, 5, 1, 1, 128, 192, 32, 64, 32_323_589, 29_964_293, 5),
    **FAMILY,
}
