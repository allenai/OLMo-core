"""
The one cross-encoder scorer. Lazily imported, never at module scope.

The pre-migration tree carried three near-identical copies of this class -- ``lib/cross_encoder.py``
(one importer), ``generate_beir_ce_data.py`` and ``generate_msmarco_trainhn_data.py`` -- and the
msmarco copy said why in its own docstring: *"duplicated to keep this data-gen script's import
chain free of the BEIR/BM25 modules"*. It was also the only copy with the multi-GPU path, so the
other two silently ran a fifth of the speed on an 8-GPU node.

The duplication existed to keep a heavy import off a light path. Keeping ``torch`` inside
``__init__`` solves that properly, so there is one implementation and it is the fast one.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

__all__ = ["DEFAULT_CE_MODEL", "CrossEncoderScorer"]

#: Higher logit = more relevant. Also the model MS MARCO's shipped score pickle was produced with,
#: so scores from the pickle and scores computed here are on one scale and comparable.
DEFAULT_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderScorer:
    """Score ``(query, passage)`` pairs for relevance with a HF cross-encoder."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 128,
        max_length: int = 512,
    ) -> None:
        """
        :param model_name: HF model id; defaults to :data:`DEFAULT_CE_MODEL`.
        :param batch_size: Pairs per forward pass **per GPU** -- the effective batch is scaled by
            the visible device count so adding GPUs adds throughput rather than idle memory.
        :param max_length: Tokenizer truncation length.

        :raises ImportError: If ``torch``/``transformers`` are missing; install ``ctc[gen]``.
        """
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:  # pragma: no cover - depends on the install
            raise ImportError(
                "cross-encoder scoring needs torch + transformers: pip install './ctc[gen]'"
            ) from e

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name or DEFAULT_CE_MODEL)
        model = (
            AutoModelForSequenceClassification.from_pretrained(model_name or DEFAULT_CE_MODEL)
            .to(self.device)
            .eval()
        )
        # Scoring is embarrassingly parallel over pairs, so shard each batch across every visible
        # GPU and scale the batch so each still gets `batch_size` rows.
        self.n_gpus = torch.cuda.device_count() if self.device == "cuda" else 0
        self.model = torch.nn.DataParallel(model) if self.n_gpus > 1 else model
        self.batch_size = batch_size * max(self.n_gpus, 1)
        self.max_length = max_length

    def score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """
        :param pairs: ``(query, passage)`` pairs.

        :returns: One relevance logit per pair, in input order.
        """
        out: List[float] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            inputs = self.tokenizer(
                [p[0] for p in chunk],
                [p[1] for p in chunk],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                logits = self.model(**inputs).logits.squeeze(-1)
            # squeeze collapses a length-1 batch to 0-d; re-expand or `.tolist()` returns a float.
            if logits.ndim == 0:
                logits = logits.unsqueeze(0)
            out.extend(logits.float().cpu().tolist())
        return out
