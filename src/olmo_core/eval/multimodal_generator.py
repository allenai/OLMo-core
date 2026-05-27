"""
Greedy / nucleus-sampling text generator for :class:`MultimodalTransformer`.

Designed for benchmark evaluation, not high-throughput inference. The
implementation re-runs the full forward pass each step (no KV cache); image
features are re-computed each step too. For the small token counts typical
of caption / VQA benchmarks (≤256 new tokens) this is fast enough and keeps
the code readable.

The generator is input-format agnostic — it takes already-tokenized
``input_ids`` and already-preprocessed ``images``/``pooled_patches_idx``.
Callers using the OLMo-core multimodal data pipeline should run their
inputs through :class:`MultimodalPreprocessor` first; callers comparing
against HuggingFace Molmo2 should use HF's :class:`Molmo2Processor` and
convert via the helpers in this module.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from ..nn.vision import MultimodalTransformer

__all__ = [
    "GenerationOutput",
    "MultimodalGenerator",
]


@dataclass
class GenerationOutput:
    """Result of one :meth:`MultimodalGenerator.generate` call."""

    token_ids: List[int]
    """The newly generated token IDs (excluding the prompt)."""

    finished_reason: str
    """``"eos"`` if generation stopped on the EOS token, ``"max_tokens"``
    if the cap was hit, ``"stop_token"`` if a custom stop token fired."""


class MultimodalGenerator:
    """Greedy or top-p sampling for :class:`MultimodalTransformer`."""

    def __init__(self, model: MultimodalTransformer):
        self.model = model

    # ------------------------------------------------------------------
    # Single-example generation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        images: Optional[torch.Tensor] = None,
        pooled_patches_idx: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        eos_token_id: Optional[int] = None,
        stop_token_ids: Sequence[int] = (),
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> GenerationOutput:
        """Generate up to ``max_new_tokens`` tokens autoregressively.

        :param input_ids: ``(1, seq_len)`` prompt token IDs. Generation
            starts immediately after this.
        :param images: ``(1, n_crops, n_patches, patch_dim)`` pre-patchified
            images. Pass ``None`` for text-only prompts.
        :param pooled_patches_idx: ``(1, n_pooled, pool_size)`` index
            tensor for the connector. Required when *images* is not ``None``.
        :param max_new_tokens: Hard cap on the number of tokens to produce.
        :param eos_token_id: Token ID that, when produced, stops generation
            (and is not included in the output).
        :param stop_token_ids: Additional stop-token IDs. Same semantics as
            *eos_token_id*.
        :param temperature: ``0.0`` ⇒ greedy. Otherwise softmax-sample with
            this temperature.
        :param top_p: Nucleus-sampling cutoff applied after temperature.
            Ignored when ``temperature == 0.0``.
        :returns: A :class:`GenerationOutput` with the produced tokens.
        """
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"input_ids must have shape (1, seq_len); got {tuple(input_ids.shape)}"
            )
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        if images is not None:
            if pooled_patches_idx is None:
                raise ValueError("pooled_patches_idx is required when images is provided")
            images = images.to(device)
            pooled_patches_idx = pooled_patches_idx.to(device)

        self.model.eval()
        generated: List[int] = []
        stop_set = set(stop_token_ids)
        if eos_token_id is not None:
            stop_set = stop_set | {eos_token_id}
        finished_reason = "max_tokens"

        for _ in range(max_new_tokens):
            # logits_to_keep=1 ⇒ the LM head only computes the last position.
            logits = self.model(
                input_ids=input_ids,
                images=images,
                pooled_patches_idx=pooled_patches_idx,
                logits_to_keep=1,
            )
            # logits shape: (1, 1, vocab_size).
            next_token_logits = logits[0, -1].float()
            next_token = self._sample(next_token_logits, temperature, top_p)

            if next_token in stop_set:
                finished_reason = "eos" if next_token == eos_token_id else "stop_token"
                break

            generated.append(next_token)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]], device=device, dtype=input_ids.dtype)],
                dim=1,
            )

        return GenerationOutput(token_ids=generated, finished_reason=finished_reason)

    # ------------------------------------------------------------------
    # Sampling primitive
    # ------------------------------------------------------------------

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
        """Pick one token from a logit vector."""
        if temperature == 0.0:
            return int(logits.argmax().item())
        probs = F.softmax(logits / temperature, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            # Keep tokens whose cumulative prob is below top_p, plus the first
            # that crosses it.
            mask = cumsum <= top_p
            mask[0] = True
            sorted_probs = sorted_probs * mask
            sorted_probs = sorted_probs / sorted_probs.sum()
            idx = torch.multinomial(sorted_probs, num_samples=1).item()
            return int(sorted_idx[idx].item())
        return int(torch.multinomial(probs, num_samples=1).item())
