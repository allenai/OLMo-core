import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from olmo_core.distributed.utils import get_rank

from .callback import Callback

log = logging.getLogger(__name__)

__all__ = ["FlopMeterCallback"]


@dataclass
class FlopMeterCallback(Callback):
    """
    Method-aware training-FLOP meter for the compute-scaling study
    (``records/flop-scaling-ffn-kv-plan.md`` §5).

    The trainer's own ``throughput/total petaflops`` charges every token the DENSE per-token cost.
    That is wrong for the two compute-saving arms, so this callback integrates, per step:

    - **dense**: ``tokens x flops_per_token(seq_len)`` (identical to the trainer's number).
    - **nested-FFN routing**: the FFN share of ``flops_per_token`` is scaled by that step's hard
      routing cost averaged over ALL layers (unrouted / not-yet-opened layers count 1.0); read
      from the model's ``_nested_ffn_moe`` holder.
    - **pooled soft tokens**: the model runs on the COMPACTED sequence, so the dense formula is
      applied to the compacted token count (attention on the compacted row length); read from
      the counters the model accumulates in ``_soft_token_compaction``.

    Records ``flop_meter/actual_pflops`` (cumulative), ``flop_meter/dense_pflops`` (what the same
    tokens would have cost dense) and their ratio every step, and writes ``flops.json`` into the
    save folder at the end of training for the results collector.
    """

    seq_len: int = 0
    """The padded sequence length the batch is shaped to (per-token FLOPs depend on it)."""

    pad_id: Optional[int] = None
    """Pad token id. When given, tokens are counted as NON-pad tokens of each step's batch (summed
    over ranks), so a padded single-example path (the soft-token arm) and a packed path (dense)
    are charged for the same real tokens; when ``None`` the padded global batch size is used."""

    _actual: float = 0.0
    _dense: float = 0.0
    _tokens: int = 0
    _real_tokens_this_step: Optional[int] = None

    def pre_step(self, batch):
        if self.pad_id is None:
            return
        import torch
        import torch.distributed as dist

        ids = batch["input_ids"]
        n = torch.tensor([int((ids != self.pad_id).sum().item())], device=ids.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(n, op=dist.ReduceOp.SUM)
        self._real_tokens_this_step = int(n.item())
    _ffn_per_tok: Optional[int] = None
    _dense_per_tok: Optional[int] = None
    _attn_score_per_tok: Optional[int] = None
    _kv_keep_weighted: float = 0.0
    _ffn_cost_weighted: float = 0.0

    def _model(self):
        return self.trainer.train_module.model  # type: ignore[union-attr]

    def _per_token(self):
        if self._dense_per_tok is None:
            m = self._model()
            self._dense_per_tok = int(m.num_flops_per_token(self.seq_len))
            self._ffn_per_tok = int(
                sum(b.feed_forward.num_flops_per_token(self.seq_len) for b in m.blocks.values())
            )
        return self._dense_per_tok, self._ffn_per_tok

    def post_step(self):
        m = self._model()
        dense_tok, ffn_tok = self._per_token()
        tokens = int(self.trainer.global_batch_size)  # tokens in the (padded) global batch
        if self._real_tokens_this_step is not None:
            tokens = self._real_tokens_this_step
        dense_flops = tokens * dense_tok
        actual = float(dense_flops)

        nffn = getattr(m, "_nested_ffn_moe", None)
        if nffn is not None:
            holder = nffn["holder"]
            per_layer = holder.per_layer_cost(last_forward=True)
            n_layers = len(m.blocks)
            # layers absent from the snapshot ran dense (below the curriculum's min layer, or
            # below start_layer): cost 1.0
            cost_all = (sum(per_layer.values()) + (n_layers - len(per_layer))) / n_layers
            actual = tokens * (dense_tok - ffn_tok * (1.0 - cost_all))
            self.trainer.record_metric("flop_meter/ffn_cost_all_layers", cost_all)
            self._ffn_cost_weighted += cost_all * tokens

        kvr = getattr(m, "_kv_route", None)
        if kvr is not None:
            # Routed attention layers score only KEPT keys: their length-dependent (QK^T, PV)
            # FLOPs scale with the hard keep fraction of this step's last forward.
            keep = float(kvr["holder"].mean_keep(last_forward=True))
            if self._attn_score_per_tok is None:
                self._attn_score_per_tok = int(
                    sum(
                        m.blocks[str(li)].attention.num_flops_per_token(self.seq_len)
                        - m.blocks[str(li)].attention.num_flops_per_token(0)
                        for li in kvr["routed"]
                    )
                )
            actual -= tokens * self._attn_score_per_tok * (1.0 - keep)
            self._kv_keep_weighted += keep * tokens
            self.trainer.record_metric("flop_meter/kv_keep", keep)

        comp = getattr(m, "_soft_token_compaction", None)
        if comp is not None and comp["tokens_out"] > 0:
            rows = max(1, comp["rows"])
            out_len = comp["tokens_out"] / rows
            # the counters are per rank and count PADDED input tokens; the compacted output is
            # what actually ran. Scale per-rank -> world by the padded global batch.
            scale = int(self.trainer.global_batch_size) / max(1, comp["tokens_in"])
            actual = comp["tokens_out"] * scale * int(m.num_flops_per_token(int(out_len)))
            self.trainer.record_metric(
                "flop_meter/compaction_ratio", comp["tokens_in"] / comp["tokens_out"]
            )
            comp["tokens_in"] = comp["tokens_out"] = comp["rows"] = 0

        self._actual += actual
        self._dense += dense_flops
        self._tokens += tokens
        self.trainer.record_metric("flop_meter/actual_pflops", self._actual / 1e15)
        self.trainer.record_metric("flop_meter/dense_pflops", self._dense / 1e15)
        self.trainer.record_metric("flop_meter/actual_over_dense", self._actual / max(1.0, self._dense))
        if self.step % 50 == 0:
            log.info(
                "[flop-meter] step %d: actual %.2f PF, dense-equivalent %.2f PF (ratio %.3f), tokens %.1fM",
                self.step,
                self._actual / 1e15,
                self._dense / 1e15,
                self._actual / max(1.0, self._dense),
                self._tokens / 1e6,
            )

    def summary(self) -> Dict[str, Any]:
        return {
            "actual_pflops": self._actual / 1e15,
            "dense_equivalent_pflops": self._dense / 1e15,
            "actual_over_dense": self._actual / max(1.0, self._dense),
            "tokens_processed": self._tokens,
            "tokens_are_real": self.pad_id is not None,
            "ffn_cost_frac": (
                self._ffn_cost_weighted / self._tokens if self._ffn_cost_weighted > 0 and self._tokens else None
            ),
            "kv_route_keep_frac": (
                self._kv_keep_weighted / self._tokens if self._kv_keep_weighted > 0 and self._tokens else None
            ),
            "steps": self.step,
            "seq_len": self.seq_len,
        }

    def post_train(self):
        if get_rank() == 0:
            path = os.path.join(str(self.trainer.save_folder), "flops.json")
            try:
                with open(path, "w") as f:
                    json.dump(self.summary(), f, indent=2)
                log.info("[flop-meter] wrote %s: %s", path, self.summary())
            except OSError as e:  # remote save folders
                log.warning("[flop-meter] could not write %s: %s", path, e)
