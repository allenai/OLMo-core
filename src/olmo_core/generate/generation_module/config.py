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
    """Whether to use an inference cache (e.g. a kv-cache) for generation."""

    stop_token_ids: Optional[List[int]] = None
    """Tokens to stop generation at. If provided, the generation will stop when any of these tokens are generated."""

    landmark_mem_id: Optional[int] = None
    """
    For landmark-attention models only: the token ID used as the landmark ("memory") token. When the
    model uses a landmark attention variant (``fast_landmark`` / ``sparse_landmark``), generation
    automatically inserts this token into the *prompt* every ``mem_freq`` content tokens (so the
    prefill sees the same block structure the model was trained on) while decoding plain content
    tokens (no landmarks). This must be set for landmark models; it is ignored otherwise.
    """

    landmark_decode_mode: str = "extend_last_block"
    """
    For landmark-attention models only: how the decode step treats the prompt's final (possibly
    partial) content block, i.e. what counts as the growing "one long local block":

    - ``"extend_last_block"``: the local block spans the last prompt block boundary through all
      generated tokens, so generated tokens attend directly to the tail content of the prompt's
      final block as well as to each other (plus past blocks via their landmarks).
    - ``"generation_only"``: only the generated tokens form the local block; the entire prompt is
      reachable only through the prompt's landmarks. To keep landmarks at the trained (periodic)
      positions, the prompt's final partial block is padded with :data:`landmark_pad_id` up to the
      next landmark position so the prompt always ends with a landmark token.
    """

    landmark_pad_id: Optional[int] = None
    """
    For landmark-attention models in ``"generation_only"`` decode mode only: the token ID used to
    pad the prompt's final partial block (up to the next landmark position) so the prompt ends with a
    landmark token. Should be a semantically neutral token such as the tokenizer's space token. If
    ``None``, :data:`pad_token_id` is used.
    """

    landmark_top_k_blocks: Optional[int] = None
    """
    For landmark-attention models only: enable hard top-k landmark block retrieval at decode time,
    following the inference procedure of the landmark attention paper (Mohtashami & Jaggi 2023,
    https://arxiv.org/abs/2305.16300, section 3.2). At each decode step, each attention head scores
    the query against the cached landmark keys and keeps only the ``landmark_top_k_blocks``
    highest-scoring blocks; all other past blocks receive exactly zero attention weight, and the
    attention renormalizes over the local block plus the retrieved blocks. ``None`` (the default)
    defers to :data:`landmark_top_k_fraction`; an explicit value here takes precedence. Prefill is
    unaffected (it remains single-shot dense over the full prompt).
    """

    landmark_top_k_fraction: Optional[float] = 0.1
    """
    For landmark-attention models only: when :data:`landmark_top_k_blocks` is not set, enable hard
    top-k decode retrieval with ``k = ceil(landmark_top_k_fraction * num_prompt_blocks)`` -- keep the
    top fraction of past landmark blocks at each decode step. **Defaults to 0.1 (top 10%), so
    landmark eval uses top-k retrieval by default** (the paper's inference; far fewer attended blocks
    at long context than dense soft-gating). Set to ``None`` to disable top-k and fall back to dense
    soft-gating over all past blocks.
    """

    landmark_nonselected_mass: Optional[float] = None
    """
    For :class:`~olmo_core.nn.attention.FastCompressiveLandmarkAttention` models with top-k decode
    (``landmark_top_k_blocks`` set) only: the fraction of attention mass reserved for the landmark
    tokens of the non-selected blocks (split among them by a softmax over their landmark scores).
    ``None`` uses the value baked into the attention module (default 0.1). Ignored by other models
    and when top-k retrieval is disabled.
    """

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
        if self.landmark_decode_mode not in ("extend_last_block", "generation_only"):
            raise ValueError(
                "landmark_decode_mode must be 'extend_last_block' or 'generation_only', "
                f"got {self.landmark_decode_mode!r}"
            )
        if self.landmark_top_k_blocks is not None and self.landmark_top_k_blocks < 1:
            raise ValueError(
                f"landmark_top_k_blocks must be >= 1 or None, got {self.landmark_top_k_blocks}"
            )
        if self.landmark_nonselected_mass is not None and not (
            0.0 <= self.landmark_nonselected_mass < 1.0
        ):
            raise ValueError(
                "landmark_nonselected_mass must be in [0, 1) or None, "
                f"got {self.landmark_nonselected_mass}"
            )
