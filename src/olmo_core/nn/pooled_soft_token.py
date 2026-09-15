"""
Soft-token document pooling ("B1") -- **train-time sequence compaction with off-the-shelf
full-attention inference**.

The sibling of :class:`~olmo_core.nn.attention.pooled_doc_kv.PooledDocKVAttention` (which pools
per-layer K/V but still runs every token through the network). Here the compression happens *once,
at the input*: each pooled context document is removed from the sequence entirely and replaced by a
single **soft token** -- a learned projection of the document's mean input embedding -- placed at
the document's center position (original ``position_ids`` are preserved so RoPE geometry matches
full-attention test time). The main stack then runs **plain causal attention on the compacted
sequence**: no custom masks, no custom kernels, and every per-token cost (attention, QKV, MLPs,
activations, backward) shrinks by the compaction factor.

Enabled via :meth:`~olmo_core.nn.transformer.model.Transformer.enable_pooled_soft_tokens`. The
projector is train-time scaffolding: at inference/export it is dropped and the checkpoint is an
ordinary dense model evaluated with full attention over the real tokens.

Design notes (validated by the probes in ``records/pooled-doc-kv-attention.md``):

* The projector is **residual-initialized**: ``P(x) = x + MLP(x)`` with the MLP's last layer
  zero-initialized, so at step 0 the soft token is exactly the document's mean input embedding --
  the feature the KV-predictability probe showed already carries most of the recoverable signal.
* Labels are only ever attached to FREE tokens (the answer region), which are never pooled, so
  compaction preserves the counted-label set exactly: the trainer's ``loss_div_factor`` (computed
  from the pre-compaction labels) stays correct with no train-module changes.
* Original PAD (everything after the first EOS) is dropped outright -- it carries no loss and is
  never attended.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention.chunked_mask import PAD_CHUNK_ID

log = logging.getLogger(__name__)

__all__ = [
    "PooledDocProjector",
    "CompactedBatch",
    "compact_pooled_rows",
    "add_soft_len_bias",
    "build_position_causal_bias",
    "masked_sdpa",
    "aux_matching_loss",
    "SLOT_MODES",
    "build_slot_stop_ids",
    "apply_slot_mode",
    "KEEP_TOKEN_RULES",
    "KEEP_TOKEN_FEATURES",
    "DEFAULT_KEEP_TOKEN_WEIGHTS",
    "KeepTokenTables",
    "build_token_idf",
    "build_token_piece_tables",
    "parse_keep_token_weights",
    "keep_token_scores",
]


@dataclass
class CompactedBatch:
    """Everything the soft-token forward needs about one compacted batch.

    ``shadow_*`` entries are the AUX-MATCHING candidates: one extra soft token per KEPT context
    doc, appended after the row's content, at the doc's center ``position_id``. They are attended
    by NOTHING (see :func:`build_position_causal_bias`) and carry no labels -- they exist solely so
    the projector's output for a doc whose REAL tokens are present can be matched, per layer,
    against those tokens' actual attention contribution (:func:`aux_matching_loss`).
    """

    input_ids: torch.Tensor  # (B, T2)
    labels: Optional[torch.Tensor]  # (B, T2)
    position_ids: torch.Tensor  # (B, T2) original positions
    soft_rows: torch.Tensor  # pooled-slot injection indices
    soft_cols: torch.Tensor
    soft_docs: torch.Tensor
    row_lens: torch.Tensor  # (B,) content length incl. shadows (pad starts here)
    # log(token count) of each pooled doc, aligned with soft_rows/soft_cols (for the +log(L) slot
    # logit bias, ``add_soft_len_bias``)
    soft_log_len: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    shadow_rows: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    shadow_cols: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    shadow_docs: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    shadow_log_len: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    # (n_shadow, max_kept_doc_len) compacted column indices of each shadow's REAL doc tokens, -1 pad
    shadow_doc_cols: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, 0, dtype=torch.long)
    )
    is_shadow: Optional[torch.Tensor] = None  # (B, T2) bool


class PooledDocProjector(nn.Module):
    """
    The soft-token projector ``P(x) = x + MLP(x)`` over mean input embeddings.

    :param d_model: Embedding dimensionality.
    :param hidden: MLP hidden size. Defaults to ``d_model``.
    """

    def __init__(
        self,
        d_model: int,
        hidden: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        hidden = hidden or d_model
        self.w_in = nn.Linear(d_model, hidden, dtype=dtype, device=init_device)
        self.w_out = nn.Linear(hidden, d_model, dtype=dtype, device=init_device)
        self.act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        """Residual init: zero the output layer so ``P(x) == x`` at step 0. Call this again after
        loading a base checkpoint that has no projector keys (a fresh global init may have
        randomized it)."""
        if self.w_in.weight.device.type != "meta":
            nn.init.normal_(self.w_in.weight, std=0.02)
            nn.init.zeros_(self.w_in.bias)
            nn.init.zeros_(self.w_out.weight)
            nn.init.zeros_(self.w_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.w_out(self.act(self.w_in(x)))


def compact_pooled_rows(
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor],
    chunk_ids: torch.Tensor,
    keep_docs: torch.Tensor,
    *,
    placeholder_id: int,
    pad_token_id: int,
    ignore_index: int = -100,
    add_shadows: bool = False,
    max_shadows_per_row: int = 8,
) -> CompactedBatch:
    """
    Compact a padded batch by dropping pooled documents' tokens (and original PAD), inserting one
    ``placeholder_id`` token per pooled document at its center position.

    :param input_ids: ``(B, T)`` token ids.
    :param labels: Optional ``(B, T)`` already-shifted labels aligned with ``input_ids`` (the value
        at position ``t`` is position ``t``'s next-token target). Gathered, not re-shifted.
    :param chunk_ids: ``(B, T)`` roles from ``build_chunk_ids_from_tokens``.
    :param keep_docs: ``(B, n_docs)`` bool -- ``True`` = document keeps its real tokens.
    :param placeholder_id: Token id emitted at soft-token slots (its embedding is overwritten by
        the projector output, so only its very existence matters).
    :param pad_token_id: Id used to right-pad the ragged compacted rows.

    :returns: ``(new_ids, new_labels, position_ids, soft_rows, soft_cols, soft_doc_ids)`` where the
        compacted batch is ``(B, T')`` (``T' = max compacted length``), ``position_ids`` holds each
        kept token's ORIGINAL position (soft tokens: the doc's center, pad: ``T - 1``), and the
        ``soft_*`` index tensors say which ``(row, col)`` of the compacted batch is the soft token
        for which document id.
    """
    B, T = input_ids.shape
    device = input_ids.device
    n_docs = keep_docs.shape[1]
    cid = chunk_ids.to(torch.long)

    rows_ids: List[torch.Tensor] = []
    rows_pos: List[torch.Tensor] = []
    rows_lab: List[torch.Tensor] = []
    soft_rows: List[int] = []
    soft_cols: List[int] = []
    soft_docs: List[int] = []
    shadow_rows: List[int] = []
    shadow_cols: List[int] = []
    shadow_docs: List[int] = []
    shadow_log_len: List[float] = []
    soft_log_len: List[float] = []
    shadow_doc_cols_list: List[torch.Tensor] = []
    for b in range(B):
        c = cid[b]
        kept_tok = (c != PAD_CHUNK_ID) & ((c < 0) | keep_docs[b].gather(0, c.clamp(min=0)))
        kept_idx = kept_tok.nonzero(as_tuple=True)[0]
        # Pooled docs present in this row, with their center positions.
        present = torch.zeros(n_docs, dtype=torch.bool, device=device)
        first = torch.full((n_docs,), T, dtype=torch.long, device=device)
        last = torch.full((n_docs,), -1, dtype=torch.long, device=device)
        is_ctx = c >= 0
        if is_ctx.any():
            d_ix = c[is_ctx]
            pos_ix = is_ctx.nonzero(as_tuple=True)[0]
            present.scatter_(0, d_ix, True)
            first.scatter_reduce_(0, d_ix, pos_ix, reduce="amin", include_self=True)
            last.scatter_reduce_(0, d_ix, pos_ix, reduce="amax", include_self=True)
        pooled = present & ~keep_docs[b]
        pooled_docs = pooled.nonzero(as_tuple=True)[0]
        centers = torch.div(first[pooled_docs] + last[pooled_docs], 2, rounding_mode="floor")

        # Merge kept tokens and soft tokens in original-position order.
        merge_pos = torch.cat([kept_idx, centers])
        is_soft = torch.cat(
            [
                torch.zeros(len(kept_idx), dtype=torch.bool, device=device),
                torch.ones(len(pooled_docs), dtype=torch.bool, device=device),
            ]
        )
        payload = torch.cat([input_ids[b, kept_idx], torch.full_like(pooled_docs, placeholder_id)])
        # A soft token inherits its doc's LAST position's label: in compacted order it is the
        # element right before the same next real token, so the prediction target is unchanged.
        # (Live labels on non-last pooled-doc positions target tokens inside the doc and are
        # necessarily dropped -- they don't occur in the answer-only-loss SFT layout.)
        lab = (
            torch.cat([labels[b, kept_idx], labels[b, last[pooled_docs]]])
            if labels is not None
            else None
        )
        doc_of = torch.cat([torch.full_like(kept_idx, -1), pooled_docs])
        # Chunk id of every merged entry (kept real tokens carry their doc id; -1 for free/soft).
        kept_entry_cid = torch.where(c[kept_idx] >= 0, c[kept_idx], torch.full_like(kept_idx, -1))
        entry_cid = torch.cat([kept_entry_cid, torch.full_like(pooled_docs, -1)])
        order = torch.argsort(merge_pos, stable=True)
        row_ids = payload[order]
        row_pos = merge_pos[order]
        row_cid = entry_cid[order]
        row_lab = lab[order] if lab is not None else None
        soft_mask = is_soft[order]
        doc_len = torch.bincount(c[is_ctx], minlength=n_docs) if is_ctx.any() else torch.zeros(n_docs, dtype=torch.long, device=device)
        doc_of_ordered = doc_of[order]
        # One host sync per row (not two per pooled document): with FSDP on 8 GPUs every .item()
        # drains the CUDA queue and stalls the overlapped all-gathers -- the per-doc version cost
        # ~50 s/step on 128-row steps (2026-09-08 ds64 campaign) while the model work was ~5 s.
        soft_cols_b = soft_mask.nonzero(as_tuple=True)[0]
        if soft_cols_b.numel():
            docs_b = doc_of_ordered[soft_cols_b]
            lens_b = doc_len[docs_b].clamp(min=1).to(torch.float32).log()
            cols_l = soft_cols_b.tolist()
            soft_rows += [b] * len(cols_l)
            soft_cols += cols_l
            soft_docs += docs_b.tolist()
            soft_log_len += lens_b.tolist()

        # AUX shadows: one soft-token candidate per KEPT context doc, appended after the content.
        if add_shadows:
            kept_ctx = (present & keep_docs[b]).nonzero(as_tuple=True)[0]
            kept_ctx = kept_ctx[torch.randperm(len(kept_ctx))[:max_shadows_per_row]]
            base = len(row_ids)
            sh_ids, sh_pos = [], []
            for j, d in enumerate(kept_ctx.tolist()):
                sh_ids.append(placeholder_id)
                center = int(torch.div(first[d] + last[d], 2, rounding_mode="floor"))
                sh_pos.append(center)
                shadow_rows.append(b)
                shadow_cols.append(base + j)
                shadow_docs.append(d)
                cols_d = (row_cid == d).nonzero(as_tuple=True)[0]
                shadow_doc_cols_list.append(cols_d)
                shadow_log_len.append(float(torch.log(torch.tensor(float(len(cols_d))))))
            if sh_ids:
                row_ids = torch.cat(
                    [row_ids, torch.tensor(sh_ids, dtype=row_ids.dtype, device=device)]
                )
                row_pos = torch.cat(
                    [row_pos, torch.tensor(sh_pos, dtype=row_pos.dtype, device=device)]
                )
                if row_lab is not None:
                    row_lab = torch.cat(
                        [
                            row_lab,
                            torch.full(
                                (len(sh_ids),), ignore_index, dtype=row_lab.dtype, device=device
                            ),
                        ]
                    )
        rows_ids.append(row_ids)
        rows_pos.append(row_pos)
        if row_lab is not None:
            rows_lab.append(row_lab)

    T2 = max(len(r) for r in rows_ids)
    new_ids = torch.full((B, T2), pad_token_id, dtype=input_ids.dtype, device=device)
    new_pos = torch.full((B, T2), T - 1, dtype=torch.long, device=device)
    new_lab = (
        torch.full((B, T2), ignore_index, dtype=labels.dtype, device=device)
        if labels is not None
        else None
    )
    row_lens = torch.zeros(B, dtype=torch.long, device=device)
    for b in range(B):
        L = len(rows_ids[b])
        row_lens[b] = L
        new_ids[b, :L] = rows_ids[b]
        new_pos[b, :L] = rows_pos[b]
        if new_lab is not None:
            new_lab[b, :L] = rows_lab[b]
    is_shadow = torch.zeros(B, T2, dtype=torch.bool, device=device)
    if shadow_rows:
        is_shadow[torch.tensor(shadow_rows), torch.tensor(shadow_cols)] = True
    n_sh = len(shadow_rows)
    max_dl = max((len(cs) for cs in shadow_doc_cols_list), default=0)
    sh_doc_cols = torch.full((n_sh, max_dl), -1, dtype=torch.long, device=device)
    for i, cs in enumerate(shadow_doc_cols_list):
        sh_doc_cols[i, : len(cs)] = cs
    return CompactedBatch(
        input_ids=new_ids,
        labels=new_lab,
        position_ids=new_pos,
        soft_rows=torch.tensor(soft_rows, dtype=torch.long, device=device),
        soft_cols=torch.tensor(soft_cols, dtype=torch.long, device=device),
        soft_docs=torch.tensor(soft_docs, dtype=torch.long, device=device),
        row_lens=row_lens,
        soft_log_len=torch.tensor(soft_log_len, dtype=torch.float32, device=device),
        shadow_rows=torch.tensor(shadow_rows, dtype=torch.long, device=device),
        shadow_cols=torch.tensor(shadow_cols, dtype=torch.long, device=device),
        shadow_docs=torch.tensor(shadow_docs, dtype=torch.long, device=device),
        shadow_log_len=torch.tensor(shadow_log_len, dtype=torch.float32, device=device),
        shadow_doc_cols=sh_doc_cols,
        is_shadow=is_shadow,
    )


SLOT_MODES = ("mean", "cmean", "cent_cmean")


def _token_id_counts(token_ids, vocab: Optional[int] = None):
    """Dense per-id occurrence counts over a 1-D array of training token ids.

    The single frequency pass shared by :func:`build_slot_stop_ids` (which ranks these counts to
    pick the stop set) and :func:`build_token_idf` (which smooths them into ``-log p``), so the
    slot's content filter and the keep-token rule's IDF are always taken over the same tokens.
    """
    import numpy as np

    arr = np.asarray(token_ids).reshape(-1).astype(np.int64, copy=False)
    n = int(arr.max()) + 1 if arr.size else 1
    if vocab is not None:
        n = max(n, int(vocab))
    return np.bincount(arr, minlength=n)[:n]


def build_slot_stop_ids(
    token_ids,
    *,
    top_k: int = 100,
    extra_ids: Iterable[int] = (),
    decode: Optional[Callable[[int], str]] = None,
    sample: int = 24,
) -> Tuple[List[int], List[str]]:
    """
    Build the **stop set** of token ids that :func:`apply_slot_mode` drops from a pooled
    document's mean input embedding (``cmean`` / ``cent_cmean``).

    The mean input embedding of a document is dominated by common-word mass: every document in a
    corpus sits at cosine 0.93-0.94 from the corpus centroid, gold and non-gold alike, which is
    why the plain-mean slot reads as empty (``records/outlier-richer-slot-probe.md``). The stop set
    removes that mass: the ``top_k`` most frequent ids of the training corpus (which is what
    punctuation, whitespace and function words are), plus any ``extra_ids`` the caller knows are
    not content (markers, pad, the placeholder). When ``decode`` is given, every id whose decoded
    piece has no alphanumeric character is also dropped, so punctuation that is rare in *this*
    corpus still goes.

    :param token_ids: A 1-D array/sequence of training token ids to take frequencies over (e.g.
        the first N rows of the training shard). Frequencies are only used for the ranking; the
        same counts feed the keep-token rule's IDF (:func:`build_token_idf`).
    :param top_k: How many of the most frequent ids to drop.
    :param extra_ids: Ids to drop unconditionally (markers / pad / placeholder).
    :param decode: Optional ``Callable[[int], str]`` (a tokenizer's single-id decode) used for the
        punctuation/whitespace rule and for the returned sample. ``None`` = frequency only.
    :param sample: How many decoded dropped pieces to return for logging.

    :returns: ``(sorted stop ids, decoded sample)``. The sample is raw ``"<id>"`` strings when
        ``decode`` is ``None``.
    """
    import numpy as np

    cnt = _token_id_counts(token_ids)
    uniq = np.flatnonzero(cnt)
    counts = cnt[uniq]
    order = np.argsort(-counts, kind="stable")
    stop = {int(t) for t in uniq[order[: max(0, int(top_k))]].tolist()}
    stop |= {int(t) for t in extra_ids}
    if decode is not None:
        for t in uniq.tolist():
            piece = decode(int(t))
            if not any(ch.isalnum() for ch in piece):
                stop.add(int(t))
    ranked = [int(t) for t in uniq[order].tolist() if int(t) in stop]
    head = ranked[:sample]
    if decode is None:
        shown = [str(t) for t in head]
    else:
        shown = [repr(decode(t)) for t in head]
    return sorted(stop), shown


def apply_slot_mode(
    emb: torch.Tensor,
    input_ids: torch.Tensor,
    chunk_ids: torch.Tensor,
    n_docs: int,
    plain_means: torch.Tensor,
    *,
    mode: str,
    stop_mask: torch.Tensor,
    centroid: str = "row",
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, int]:
    """
    Rebuild the per-document slot feature from **content tokens only**, optionally centred.

    * ``cmean`` -- mean input embedding over the document's tokens whose id is NOT in the stop set.
    * ``cent_cmean`` -- the same, minus the **row centroid** (the identically-filtered mean over
      *all* of that row's document tokens) and rescaled to the row's mean real-token embedding
      norm. The row centroid is the online, zero-extra-pass stand-in for the corpus centroid used
      eval-side in ``debug/pooled_kv/outlier_probe/outlier_richer_slot_probe.py``: a training row
      holds 14-56 documents drawn from the same corpus, so its content mean estimates the corpus
      content mean, and it needs no precomputed vector and no second pass over the shard. (Pass a
      corpus centroid instead only if rows ever hold a single document.)

    A document with no surviving token falls back to its plain mean (and is still centred /
    rescaled under ``cent_cmean``, matching the eval-side construction).

    :param emb: ``(B, T, D)`` input embeddings of ``input_ids``.
    :param input_ids: ``(B, T)`` token ids.
    :param chunk_ids: ``(B, T)`` document ids, ``< 0`` for free/pad tokens.
    :param n_docs: Number of document slots per row.
    :param plain_means: ``(B, n_docs, D)`` plain means, used for the empty-document fallback.
    :param mode: ``"cmean"`` or ``"cent_cmean"`` (``"mean"`` should not reach here).
    :param stop_mask: ``(vocab,)`` bool; ``True`` = drop this id from the mean.
    :param centroid: ``"row"`` (the only implemented centre; see above).

    :returns: ``((B, n_docs, D) slot features, number of fallback documents)``.

    :raises ValueError: On an unknown ``mode`` or ``centroid``.
    """
    if mode not in ("cmean", "cent_cmean"):
        raise ValueError(
            f"apply_slot_mode: unknown slot mode {mode!r} (expected one of {SLOT_MODES})"
        )
    if centroid != "row":
        raise ValueError(
            f"apply_slot_mode: unknown centroid {centroid!r} (only 'row' is implemented)"
        )
    B, T, D = emb.shape
    cid = chunk_ids.to(torch.long)
    is_ctx = (cid >= 0).reshape(-1)
    flat_doc = (torch.arange(B, device=emb.device)[:, None] * n_docs + cid.clamp(min=0)).reshape(
        -1
    )[is_ctx]
    flat_row = torch.arange(B, device=emb.device)[:, None].expand(B, T).reshape(-1)[is_ctx]
    e = emb.reshape(B * T, D)[is_ctx].float()
    w = (~stop_mask.to(emb.device)[input_ids]).reshape(-1)[is_ctx].float()

    sums = torch.zeros(B * n_docs, D, device=emb.device).index_add(0, flat_doc, e * w[:, None])
    mass = torch.zeros(B * n_docs, device=emb.device).index_add(0, flat_doc, w)
    n_tok = torch.zeros(B * n_docs, device=emb.device).index_add(0, flat_doc, torch.ones_like(w))
    cmean = (sums / mass.clamp(min=eps)[:, None]).reshape(B, n_docs, D)
    empty = (mass <= 0).reshape(B, n_docs)
    # Documents that HAVE tokens but lost every one of them to the stop set: the fallback the
    # caller counts. Slots past a row's document count are empty too, but are never read.
    n_fallback = int(((mass <= 0) & (n_tok > 0)).sum().item())
    feats = torch.where(empty[..., None], plain_means.float(), cmean)

    if mode == "cent_cmean":
        r_sums = torch.zeros(B, D, device=emb.device).index_add(0, flat_row, e * w[:, None])
        r_mass = torch.zeros(B, device=emb.device).index_add(0, flat_row, w)
        r_norm = torch.zeros(B, device=emb.device).index_add(0, flat_row, e.norm(dim=-1) * w)
        centre = r_sums / r_mass.clamp(min=eps)[:, None]
        target = (r_norm / r_mass.clamp(min=eps)).clamp(min=eps)
        feats = feats - centre[:, None, :]
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=eps) * target[:, None, None]

    return feats.to(emb.dtype), n_fallback


# ---------------------------------------------------------------------------
# Keep-token rule: which body tokens of a pooled document stay REAL
# ---------------------------------------------------------------------------
#
# ``--st-header-extra-tokens K`` keeps the FIRST K body tokens of every pooled document real. The
# eval-side saliency probe (``debug/pooled_kv/outlier_probe/outlier_saliency_preview_probe.py``,
# ``records/outlier-saliency-preview-probe.md``) showed that *which* K is the whole effect --
# ``rand8`` is worse than keeping nothing (CE 0.558 vs 0.470) while ``first8`` reaches 0.263 and a
# ridge over six cheap token features (``rule8``) reaches 0.071, matching the gradient ORACLE
# (``grad8`` 0.072). This is the training-side port of that ``rule{k}`` selector: a per-token
# linear score from features computable from TOKEN IDS AND POSITIONS ALONE (no model forward), of
# which the top k per document are kept real.

KEEP_TOKEN_RULES = ("none", "first", "rule")

#: The probe's feature list, in order (``outlier_saliency_preview_probe.FEATURES``).
KEEP_TOKEN_FEATURES = (
    "idf",
    "relpos",
    "first_sent",
    "is_cap",
    "is_dig",
    "tok_len",
    "log_doclen",
)

#: **Placeholder** weights -- see :func:`parse_keep_token_weights`. The probe prints its fitted
#: ridge weights into its job log and stores them in its result JSON, but neither is reachable from
#: this repo (the runs live on weka / Beaker), so these are derived from what IS recorded: the
#: within-document gradient-saliency profile of ``records/outlier-saliency-preview-probe.md`` 5a(c),
#: where 1.0 = that document's average token. Gradient prefers the first sentence (1.36), early
#: positions (position quartiles 1.18 -> 0.92), the RAREST idf quartile (1.18 vs 0.94 for the most
#: frequent), capitalised tokens (1.27), and actively avoids digits (0.78). Each entry below is that
#: sign and rough magnitude divided by the feature's nominal spread, so they apply to RAW features.
#: ``log_doclen`` is constant inside a document and therefore cannot affect a per-document top-k at
#: all; it is listed only to keep the vector interchangeable with the probe's 7-feature fit.
DEFAULT_KEEP_TOKEN_WEIGHTS: Dict[str, float] = {
    "idf": 0.40,
    "relpos": -1.30,
    "first_sent": 1.20,
    "is_cap": 0.75,
    "is_dig": -1.00,
    "tok_len": 0.04,
    "log_doclen": 0.0,
}


@dataclass
class KeepTokenTables:
    """Vocab-sized feature tables for :func:`keep_token_scores` (all ``(vocab,)``)."""

    idf: torch.Tensor
    """``-log p(id)`` over the training shard (:func:`build_token_idf`)."""
    sent_end: torch.Tensor
    """bool: the decoded piece contains ``.`` or a newline."""
    is_cap: torch.Tensor
    """bool: the stripped piece starts with an upper-case character."""
    is_dig: torch.Tensor
    """bool: the piece contains a digit."""
    tok_len: torch.Tensor
    """float: length of the stripped decoded piece."""

    def to(self, device) -> "KeepTokenTables":
        """Move every table to ``device`` (a no-op when already there)."""
        if self.idf.device == torch.device(device):
            return self
        return KeepTokenTables(
            idf=self.idf.to(device),
            sent_end=self.sent_end.to(device),
            is_cap=self.is_cap.to(device),
            is_dig=self.is_dig.to(device),
            tok_len=self.tok_len.to(device),
        )


def build_token_idf(token_ids, vocab: int):
    """
    Per-id ``-log p(id)`` ("IDF") over the head of the training shard, add-one smoothed.

    The same array and the same source tokens the ``cmean``/``cent_cmean`` stop set is built from
    (:func:`build_slot_stop_ids` ranks the very same counts), so a run needs ONE read of the shard
    head for both. Matches ``outlier_saliency_preview_probe.build_idf`` exactly.

    :param token_ids: 1-D array/sequence of training token ids.
    :param vocab: Table size (the model's embedding rows).

    :returns: ``(vocab,)`` float32 numpy array.
    """
    import numpy as np

    cnt = _token_id_counts(token_ids, vocab=vocab)[:vocab].astype(np.float64)
    p = (cnt + 1.0) / (cnt.sum() + float(vocab))
    return (-np.log(p)).astype(np.float32)


def build_token_piece_tables(pieces: Sequence[Optional[str]]):
    """
    Per-id ``(sent_end, is_cap, is_dig, tok_len)`` tables from decoded tokenizer pieces.

    Matches ``outlier_saliency_preview_probe.build_piece_tables``: byte-BPE markers ``Ġ``/``Ċ`` are
    mapped back to space/newline first.

    :param pieces: ``tok.convert_ids_to_tokens(range(vocab))`` (``None`` entries are skipped).

    :returns: ``(sent_end, is_cap, is_dig, tok_len)`` numpy arrays of length ``len(pieces)``.
    """
    import numpy as np

    vocab = len(pieces)
    sent_end = np.zeros(vocab, dtype=bool)
    is_cap = np.zeros(vocab, dtype=bool)
    is_dig = np.zeros(vocab, dtype=bool)
    tok_len = np.zeros(vocab, dtype=np.float32)
    for i, s in enumerate(pieces):
        if s is None:
            continue
        t = s.replace("\u0120", " ").replace("\u010a", "\n")
        if "." in t or "\n" in t:
            sent_end[i] = True
        st = t.strip()
        if st[:1].isupper():
            is_cap[i] = True
        if any(c.isdigit() for c in t):
            is_dig[i] = True
        tok_len[i] = float(len(st))
    return sent_end, is_cap, is_dig, tok_len


def parse_keep_token_weights(spec) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Resolve a ``--st-keep-token-weights`` spec into ``(weights, sd)``.

    Accepted: ``None`` (the documented placeholder :data:`DEFAULT_KEEP_TOKEN_WEIGHTS`), a dict, an
    inline JSON string, or a path to a JSON file. Two shapes are understood:

    * flat -- ``{"idf": 0.4, "relpos": -1.3, ...}``: weights on RAW features.
    * ridge -- ``{"weights": {...}, "sd": {...}, "mu": {...}, "bias": 0.0}``: the shape
      ``outlier_saliency_preview_probe.fit_ridge`` produces, i.e. weights on STANDARDISED features.
      ``sd`` is honoured (it rescales each weight); ``mu`` and ``bias`` are accepted and IGNORED
      because they shift every token of every document by the same constant and so cannot change a
      per-document top-k.

    :raises ValueError: On an unknown feature name or an unparseable spec.
    """
    import json
    import os

    if spec is None:
        return dict(DEFAULT_KEEP_TOKEN_WEIGHTS), {}
    obj = spec
    if isinstance(spec, str):
        text = spec
        if not spec.lstrip().startswith("{"):
            if not os.path.exists(spec):
                raise ValueError(
                    f"--st-keep-token-weights {spec!r} is neither inline JSON nor an existing file"
                )
            with open(spec) as f:
                text = f.read()
        obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError(f"keep-token weights must be a JSON object, got {type(obj).__name__}")
    raw = obj["weights"] if "weights" in obj else obj
    weights = {str(f): float(w) for f, w in dict(raw).items()}
    sd = {str(f): float(v) for f, v in dict(obj.get("sd") or {}).items()}
    unknown = sorted(f for f in set(weights) | set(sd) if f not in KEEP_TOKEN_FEATURES)
    if unknown:
        raise ValueError(
            f"unknown keep-token feature(s) {unknown}; expected a subset of "
            f"{list(KEEP_TOKEN_FEATURES)}"
        )
    return weights, sd


def keep_token_scores(
    input_ids: torch.Tensor,
    chunk_ids: torch.Tensor,
    *,
    tables: KeepTokenTables,
    weights: Dict[str, float],
    doc_start_id: int,
    doc_end_id: int,
    sd: Optional[Dict[str, float]] = None,
    n_docs: Optional[int] = None,
) -> torch.Tensor:
    """
    Per-token linear feature score for the ``rule`` keep-token selector -- the training-side port of
    ``outlier_saliency_preview_probe``'s ``token_features`` + ``apply_ridge``.

    Every feature is computable from token ids and positions alone (no model forward, no gradient):

    * ``idf`` -- ``-log p(id)`` over the training shard (:func:`build_token_idf`).
    * ``relpos`` -- index inside the document's body / ``max(1, body_len - 1)``.
    * ``first_sent`` -- 1.0 up to and including the body's first sentence-ending token (1.0
      everywhere when the body has none), else 0.0.
    * ``is_cap`` / ``is_dig`` / ``tok_len`` -- piece tables (:func:`build_token_piece_tables`).
    * ``log_doclen`` -- ``log(max(2, body_len))``; constant inside a document, so it cannot change a
      per-document top-k, and is carried only for weight-vector compatibility.

    :param input_ids: ``(B, S)`` token ids.
    :param chunk_ids: ``(B, S)`` roles, normally already header-freed, so "body" is exactly the part
        a slot would stand in for (:func:`~olmo_core.nn.attention.chunked_mask.doc_body_groups`).
    :param tables: Vocab-sized feature tables.
    :param weights: Feature -> weight (raw features).
    :param sd: Optional per-feature standard deviations to divide the weights by (for a ridge fit
        on standardised features).
    :param n_docs: Document-id space (see
        :func:`~olmo_core.nn.attention.chunked_mask.doc_body_groups`).

    :returns: ``(B, S)`` float32 scores, zero at non-body positions (never read by
        :func:`~olmo_core.nn.attention.chunked_mask.mark_doc_topk_tokens_free`).
    """
    from .attention.chunked_mask import doc_body_groups

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if chunk_ids.dim() == 1:
        chunk_ids = chunk_ids.unsqueeze(0)
    device = input_ids.device
    out = torch.zeros(input_ids.shape, dtype=torch.float32, device=device).reshape(-1)
    flat_idx, gid, within, counts, _n = doc_body_groups(
        chunk_ids, input_ids, doc_start_id=doc_start_id, doc_end_id=doc_end_id, n_docs=n_docs
    )
    M = int(flat_idx.numel())
    if M == 0:
        return out.reshape(input_ids.shape)
    tab = tables.to(device)
    ids_m = input_ids.reshape(-1)[flat_idx]
    body_len = counts[gid]
    body_len_f = body_len.to(torch.float32)
    # first_sent: 1.0 through the body's FIRST sentence-ending token; a body with none is all 1.0
    # (the probe's `end = n - 1` fallback).
    sentinel = M + 1
    se_pos = torch.where(tab.sent_end[ids_m], within, torch.full_like(within, sentinel))
    first_end = torch.full(
        (counts.numel(),), sentinel, dtype=within.dtype, device=device
    ).scatter_reduce(0, gid, se_pos, reduce="amin", include_self=True)[gid]
    end = torch.where(first_end >= sentinel, (body_len - 1).clamp(min=0), first_end)
    feats = {
        "idf": tab.idf[ids_m].to(torch.float32),
        "relpos": within.to(torch.float32) / (body_len_f - 1.0).clamp(min=1.0),
        "first_sent": (within <= end).to(torch.float32),
        "is_cap": tab.is_cap[ids_m].to(torch.float32),
        "is_dig": tab.is_dig[ids_m].to(torch.float32),
        "tok_len": tab.tok_len[ids_m].to(torch.float32),
        "log_doclen": torch.log(body_len_f.clamp(min=2.0)),
    }
    sd = sd or {}
    score = torch.zeros(M, dtype=torch.float32, device=device)
    for name, value in feats.items():
        scale = float(sd.get(name, 1.0) or 1.0)
        w = float(weights.get(name, 0.0)) / scale
        if w != 0.0:
            score = score + w * value
    out[flat_idx] = score
    return out.reshape(input_ids.shape)


def add_soft_len_bias(
    attn_bias: torch.Tensor, cb: CompactedBatch, scale: float = 1.0, extra: float = 0.0
) -> torch.Tensor:
    """
    Add ``+log(doc_len)`` to every pooled slot's attention logit (the "log-mass" trick of
    :class:`~olmo_core.nn.attention.pooled_doc_kv.PooledDocKVAttention`, applied to soft tokens):
    a slot with key ``k`` and logit bias ``log L`` contributes ``L * exp(q.k)`` to the softmax
    denominator, i.e. the mass of ``L`` copies of itself -- what a diffuse document of ``L``
    tokens would contribute -- instead of one token's worth. The slot's content (its live,
    detached soft token) is unchanged; only the mass is corrected.

    :param attn_bias: ``(B, 1, T2, T2)`` additive bias from :func:`build_position_causal_bias`.
    :param cb: The compacted batch (``soft_rows`` / ``soft_cols`` / ``soft_log_len``).
    :param scale: Multiplier on ``log(doc_len)`` (1 = the log-mass trick, 0 = constant only).
    :param extra: Constant added to every slot's logit on top (a calibration offset ``c``).

    :returns: The bias with the per-column slot term added (broadcast over queries).
    """
    if cb.soft_rows.numel() == 0:
        return attn_bias
    B, _, _, T2 = attn_bias.shape
    col = torch.zeros((B, 1, 1, T2), dtype=attn_bias.dtype, device=attn_bias.device)
    col[cb.soft_rows.to(attn_bias.device), 0, 0, cb.soft_cols.to(attn_bias.device)] = (
        scale * cb.soft_log_len + extra
    ).to(attn_bias.device, attn_bias.dtype)
    return attn_bias + col


def build_position_causal_bias(
    cb: CompactedBatch, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    The ``(B, 1, T2, T2)`` additive attention bias for a compacted batch with shadows:

    * causality by ORIGINAL POSITION (for content this equals sequence causality, since content is
      sorted by position; for shadows it gives exactly the view a deployed slot at the doc center
      would have),
    * nothing attends a shadow column (they must not perturb the LM computation),
    * a shadow does not attend its OWN doc's real tokens (a deployed slot's doc tokens are absent),
    * self-attention always allowed (NaN guard).
    """
    pos = cb.position_ids.to(device)
    B, T2 = pos.shape
    allowed = pos[:, None, :] <= pos[:, :, None]  # (B, q, kv): kv position <= q position
    if cb.is_shadow is not None and cb.is_shadow.any():
        allowed &= ~cb.is_shadow.to(device)[:, None, :]
        # Block each shadow row's view of its own doc's real tokens.
        n_sh = cb.shadow_rows.shape[0]
        for i in range(n_sh):
            b = int(cb.shadow_rows[i])
            qcol = int(cb.shadow_cols[i])
            cols = cb.shadow_doc_cols[i]
            cols = cols[cols >= 0]
            allowed[b, qcol, cols] = False
    eye = torch.eye(T2, dtype=torch.bool, device=device)
    allowed |= eye[None]
    finfo_min = torch.finfo(dtype).min
    return torch.where(
        allowed.unsqueeze(1),
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), finfo_min, dtype=dtype, device=device),
    )


def masked_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, scale: float
) -> torch.Tensor:
    """Direct SDPA with an additive bias over ``(B, T, H, D)`` q / ``(B, T, H_kv, D)`` k,v."""
    n_rep = q.shape[2] // k.shape[2]
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=2)
        v = v.repeat_interleave(n_rep, dim=2)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        # SDPA requires the bias dtype to match the (possibly autocast/FSDP-bf16) query dtype.
        attn_mask=bias.to(q.dtype),
        is_causal=False,
        scale=scale,
    )
    return out.transpose(1, 2).contiguous()


def aux_matching_loss(
    layers: List[Tuple[torch.Tensor, ...]],
    *,
    q_rows: torch.Tensor,
    shadow_rows: torch.Tensor,
    shadow_log_len: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    The per-layer attention-contribution matching loss (the learnable objective that makes a soft
    token's KV behave like its doc's real KV):

    For each sampled query token q and each shadow (a kept doc's soft-token candidate) in the same
    row, match IN LOGIT SPACE the doc's total softmax mass,

        q . k_shadow * scale + log(L_doc)   ~=   logsumexp_i (q . k_i * scale),

    and match the shadow's value vector to the doc's per-query attention-weighted mean value.
    Averaged over layers, heads, and (query, shadow) pairs.

    ``layers`` entries are ``(q_s, k_doc, v_doc, doc_of_kv, k_sh, v_sh)`` where ``q_s`` is
    ``(Nq, H, hd)`` post-RoPE sampled queries, ``k_doc``/``v_doc`` are ``(Nkv, H_kv, hd)`` the
    kept docs' real tokens, ``doc_of_kv`` maps each kv token to its shadow index, and
    ``k_sh``/``v_sh`` are ``(Ns, H_kv, hd)``.
    """
    total = None
    same_row = q_rows[:, None] == shadow_rows[None, :]  # (Nq, Ns)
    for q_s, k_doc, v_doc, doc_of_kv, k_sh, v_sh in layers:
        Nq, H, hd = q_s.shape
        n_rep = H // k_doc.shape[1]
        kd = k_doc.repeat_interleave(n_rep, dim=1) if n_rep > 1 else k_doc
        vd = v_doc.repeat_interleave(n_rep, dim=1) if n_rep > 1 else v_doc
        ks = k_sh.repeat_interleave(n_rep, dim=1) if n_rep > 1 else k_sh
        vs = v_sh.repeat_interleave(n_rep, dim=1) if n_rep > 1 else v_sh
        qf, kdf, vdf, ksf, vsf = (t.float() for t in (q_s, kd, vd, ks, vs))
        # (Nq, H, Nkv) real-token logits; (Nq, H, Ns) shadow logits.
        lg = torch.einsum("qhd,thd->qht", qf, kdf) * scale
        lg_sh = torch.einsum("qhd,shd->qhs", qf, ksf) * scale + shadow_log_len[None, None, :]
        Ns = ks.shape[0]
        # Per-shadow logsumexp over its doc's tokens + per-shadow weighted value target.
        onehot = F.one_hot(doc_of_kv, Ns).float()  # (Nkv, Ns)
        lg_grp = lg[:, :, :, None] + torch.log(onehot[None, None, :, :] + 1e-45)
        mass_target = torch.logsumexp(lg_grp, dim=2).detach()  # (Nq, H, Ns)
        w = torch.softmax(lg_grp, dim=2)  # normalized within each doc group
        val_target = torch.einsum("qhts,thd->qhsd", w, vdf).detach()  # (Nq, H, Ns, hd)
        mask = same_row[:, None, :].expand(Nq, H, Ns)
        n_pairs = mask.sum().clamp(min=1)
        mass_l = ((lg_sh - mass_target) ** 2 * mask).sum() / n_pairs
        val_pred = vsf.permute(1, 0, 2)[None].expand(Nq, H, Ns, hd)
        val_l = ((val_pred - val_target) ** 2).mean(dim=-1).mul(mask).sum() / n_pairs
        layer_loss = mass_l + val_l
        total = layer_loss if total is None else total + layer_loss
    return total / len(layers)
