from dataclasses import dataclass, field
from typing import Any, Dict, List

from .callback import Callback


@dataclass
class DataMixtureMonitorCallback(Callback):
    """
    Track tokens and sequences per data source in the current training run.

    Assumes batches include a ``"metadata"`` field (a list of dicts) and that each
    metadata dict contains either ``"source"`` or ``"label"`` naming the source.
    """

    # Public config fields (OmegaConf-friendly)
    log_interval: int = 10
    source_key: str = "source"

    # Internal state fields (also kept OmegaConf-friendly: simple types only)
    _step: int = field(default=0, repr=False)
    _tokens_per_source: Dict[str, int] = field(default_factory=dict, repr=False)
    _seqs_per_source: Dict[str, int] = field(default_factory=dict, repr=False)
    _total_tokens: int = field(default=0, repr=False)

    def pre_train(self) -> None:
        # Reset internal state at the start of training
        self._step = 0
        self._tokens_per_source = {}
        self._seqs_per_source = {}
        self._total_tokens = 0

    def pre_step(self, batch: Dict[str, Any]) -> None:
        metas: List[Dict[str, Any]] | None = batch.get("metadata")
        input_ids = batch.get("input_ids")

        if metas is None or input_ids is None:
            return

        # input_ids is [B, T]; treat T as token count per sequence
        seq_len = int(input_ids.shape[1])

        self._step += 1

        for meta in metas:
            src = str(
                meta.get(self.source_key)
                or meta.get("label")  # for examples that use "label"
                or "unknown"
            )

            self._tokens_per_source[src] = self._tokens_per_source.get(src, 0) + seq_len
            self._seqs_per_source[src] = self._seqs_per_source.get(src, 0) + 1
            self._total_tokens += seq_len

        if self._step % self.log_interval != 0:
            return

        if self._total_tokens <= 0:
            return

        # Log cumulative metrics via Trainer
        for src, tok in self._tokens_per_source.items():
            seqs = self._seqs_per_source.get(src, 0)
            share = float(tok) / float(self._total_tokens)

            self.trainer.record_metric(
                f"data_mixture/tokens/source={src}",
                float(tok),
            )
            self.trainer.record_metric(
                f"data_mixture/sequences/source={src}",
                float(seqs),
            )
            self.trainer.record_metric(
                f"data_mixture/token_share/source={src}",
                share,
            )
