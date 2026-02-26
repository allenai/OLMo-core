"""
HydraTransformer: multi-head branched transformer.

Shares a common trunk (early layers) and branches into N independent heads
(late layers). All heads share a single lm_head. Each head produces its own
logits, which can be averaged or otherwise combined downstream.
"""

import logging
from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from ..config import ModelConfig
from .config import TransformerConfig

log = logging.getLogger(__name__)
VOCAB_SIZE = 100352


@dataclass
class HydraTransformerConfig(ModelConfig):
    """
    Config for building a :class:`HydraTransformer`.

    :param base_config: Full TransformerConfig for the underlying model architecture.
    :param n_heads: Number of parallel heads.
    :param trunk_layers: Number of layers in the shared trunk.
    :param head_layers: Number of layers per head.
    """

    base_config: TransformerConfig
    n_heads: int
    trunk_layers: int
    head_layers: int

    def validate(self):
        total = self.trunk_layers + self.head_layers
        expected = self.base_config.n_layers
        if total != expected:
            raise ValueError(
                f"trunk_layers ({self.trunk_layers}) + head_layers ({self.head_layers}) = {total}, "
                f"but base_config.n_layers = {expected}"
            )
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")

    @classmethod
    def from_olmo2_1B(
        cls,
        n_heads: int = 5,
        heads_depth: int = 3,
        vocab_size: int = VOCAB_SIZE,
    ) -> "HydraTransformerConfig":
        """
        Factory for OLMo2 1B with configurable split point.

        Sets flash_2 attention backend for KV cache support.

        Constructor for HydraTransformerConfig with (0-indexed):
            - Layers [0, 1, ..., (n_layers - heads_depth - 1)] in trunk
            - Layers [(n_layers - heads_depth), ..., (n_layers - 1)] in head(s)
        (OLMo2 1B has 16 layers)
        """
        from ..attention import AttentionBackendName

        base = TransformerConfig.olmo2_1B_v2(vocab_size=vocab_size)  # inherit base transformer
        # need to use attention backend for KV caching
        base.block.sequence_mixer.backend = AttentionBackendName.flash_2  # type: ignore[union-attr]
        return cls(
            base_config=base,
            n_heads=n_heads,
            trunk_layers=base.n_layers - heads_depth,
            head_layers=heads_depth,
        )

    def build(self, *, init_device: str = "cpu") -> "HydraTransformer":
        """
        Build the HydraTransformer.

        :param init_device: Device for parameter initialization (use ``"meta"`` for zero-memory).
        """
        self.validate()

        # new configs for the trunk and heads.
        trunk_config = replace(self.base_config, n_layers=self.trunk_layers)
        head_config = replace(self.base_config, n_layers=self.head_layers)

        # build meta trunk, no need for lm_head
        trunk = trunk_config.build(init_device=init_device)
        trunk.lm_head = None  # type: ignore[assignment]

        # Build one head to extract the shared lm_head, then strip it
        donor = head_config.build(init_device=init_device)
        lm_head = donor.lm_head
        donor.lm_head = None  # type: ignore[assignment]
        donor.embeddings = None  # type: ignore[assignment]

        # build all the meta heads, all without lm_head.
        # NOTE: currently only use one lm_head shared for all hydra heads.
        # think LoRA fine tuning usually does not touch this.
        heads = nn.ModuleList()
        heads.append(donor)
        for _ in range(self.n_heads - 1):
            head = head_config.build(init_device=init_device)
            head.embeddings = None  # type: ignore[assignment]
            head.lm_head = None  # type: ignore[assignment]
            heads.append(head)

        return HydraTransformer(trunk=trunk, heads=heads, lm_head=lm_head)

    @property
    def num_params(self) -> int:
        d = self.base_config.d_model
        block_params = self.base_config.block.num_params(d)

        # Trunk: embeddings + trunk blocks (no lm_head).
        n = d * self.base_config.vocab_size + self.trunk_layers * block_params

        # Heads: head blocks only (no embeddings, no lm_head).
        n += self.n_heads * self.head_layers * block_params

        # Shared lm_head: one copy.
        n += self.base_config.lm_head.num_params(d, self.base_config.vocab_size)

        return n

    @property
    def num_non_embedding_params(self) -> int:
        return self.num_params - self.base_config.d_model * self.base_config.vocab_size


class HydraTransformer(nn.Module):
    """
    A multi-head branched transformer.

    Runs input through a shared trunk, then fans out to N independent heads.
    All heads share a single lm_head for the final projection to vocab logits.

    :param trunk: Shared transformer trunk (no lm_head).
    :param heads: ModuleList of head transformers (no embeddings, no lm_head).
    :param lm_head: Shared language modeling head.
    """

    def __init__(
        self,
        trunk: nn.Module,
        heads: nn.ModuleList,
        lm_head: nn.Module,
    ):
        super().__init__()
        self.trunk = trunk
        self.heads = heads
        self.lm_head = lm_head

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_kv_cache(self, batch_size: int, max_seq_len: int):
        """Initialize KV caches for all blocks in trunk and heads."""
        for block in self.trunk.blocks.values():
            block.attention.init_kv_cache_manager(batch_size, max_seq_len)
        for head in self.heads:
            for block in head.blocks.values():
                block.attention.init_kv_cache_manager(batch_size, max_seq_len)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Run the full model.

        :param input_ids: Token IDs ``(batch, seq)``.
        :returns: Logits tensor ``(n_heads, batch, seq, vocab)``.
        """
        h = self.trunk(input_ids, **kwargs)

        # NOTE: Streaming was tried, but honestly we are too GPU poor to make a difference
        # TODO: Try to parallelise forward passes through heads
        head_hidden = [head(h, **kwargs) for head in self.heads]

        # combine lm_heads into one big matmul, benchmarked a LOT faster
        stacked = torch.cat(head_hidden, dim=0)  # (N*batch, seq, d_model)
        all_logits = self.lm_head(stacked)  # (N*batch, seq, vocab)
        return all_logits.reshape(len(self.heads), -1, *all_logits.shape[1:])

    @staticmethod
    def load_olmo_state(
        model: "HydraTransformer",
        olmo_state: dict[str, torch.Tensor],
        trunk_layers: int,
        vocab_size: int,
    ) -> None:
        """
        Load a flat OLMo-format state dict into a HydraTransformer.

        Splits the state by layer index into trunk/head/lm_head components,
        pads vocab embeddings if needed, and clones head weights for each head.

        :param model: The HydraTransformer to load into.
        :param olmo_state: OLMo-format state dict (output of ``convert_state_from_hf``).
        :param trunk_layers: Number of layers in the trunk.
        :param vocab_size: Target vocab size (for padding).
        """
        trunk_state: dict[str, torch.Tensor] = {}
        head_state: dict[str, torch.Tensor] = {}
        lm_head_state: dict[str, torch.Tensor] = {}

        for key, value in olmo_state.items():
            if key.startswith("blocks."):
                block_idx = int(key.split(".", 2)[1])
                suffix = key.split(".", 2)[2]
                if block_idx < trunk_layers:
                    trunk_state[f"blocks.{block_idx}.{suffix}"] = value
                else:
                    new_idx = block_idx - trunk_layers
                    head_state[f"blocks.{new_idx}.{suffix}"] = value
            elif key.startswith("lm_head."):
                lm_head_state[key.split(".", 1)[1]] = value
            else:
                trunk_state[key] = value

        # pad vocab so that it is a nice size for matmuls
        emb = trunk_state["embeddings.weight"]
        if emb.shape[0] < vocab_size:
            padding = torch.zeros(vocab_size - emb.shape[0], emb.shape[1], dtype=emb.dtype)
            trunk_state["embeddings.weight"] = torch.cat([emb, padding], dim=0)

        w_out = lm_head_state["w_out.weight"]
        if w_out.shape[0] < vocab_size:
            padding = torch.zeros(vocab_size - w_out.shape[0], w_out.shape[1], dtype=w_out.dtype)
            lm_head_state["w_out.weight"] = torch.cat([w_out, padding], dim=0)

        model.trunk.load_state_dict(trunk_state, assign=True)
        model.lm_head.load_state_dict(lm_head_state, assign=True)

        for i, head in enumerate(model.heads):
            # NOTE: For testing, can inject noise into head params here
            state = (
                head_state if i == 0 else {k: v.clone() for k, v in head_state.items()}
            )  # NEED COPY
            head.load_state_dict(state, assign=True)
