from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from olmo_core.config import Config

if TYPE_CHECKING:
    pass


@dataclass
class GenerationConfig(Config):
    """Configuration for text generation."""

    pad_token_id: int
    """Padding token ID."""

    eos_token_id: int
    """End of sequence token ID."""

    max_length: Optional[int] = None
    """Maximum length of input + newly generated tokens."""

    max_new_tokens: Optional[int] = None
    """Maximum number of new tokens to generate. If provided, this takes precedence over max_length."""

    do_sample: bool = True
    """Whether to use sampling for generation. If False, greedy decoding is used. This overrides temperature, top_k, and top_p."""

    temperature: float = 0.0
    """Temperature for sampling. If 0, this is equivalent to greedy selection."""

    top_k: int = -1
    """Top-k sampling. Only consider the top k tokens with the highest probabilities. -1 means no filtering."""

    top_p: float = 1.0
    """Top-p (nucleus) sampling. Only consider the smallest set of tokens whose cumulative probability exceeds this threshold. 1.0 means no filtering."""

    use_cache: bool = True
    """Whether to use a kv-cache for generation. If True, the model will cache past key-value pairs to speed up generation."""

    stop_token_ids: Optional[List[int]] = None
    """Tokens to stop generation at. If provided, the generation will stop when any of these tokens are generated."""

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate the generation configuration."""
        if self.pad_token_id < 0:
            raise ValueError(f"pad_token_id must be non-negative, got {self.pad_token_id}")
        if self.eos_token_id < 0:
            raise ValueError(f"eos_token_id must be non-negative, got {self.eos_token_id}")
        if self.pad_token_id == self.eos_token_id:
            raise ValueError(
                f"pad_token_id and eos_token_id must be different, got {self.pad_token_id} and {self.eos_token_id}"
            )
        if self.max_length is not None and self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        if self.top_k <= 0 and self.top_k != -1:
            raise ValueError(f"top_k must be positive or -1, got {self.top_k}")
        if self.top_p <= 0.0 or self.top_p > 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
