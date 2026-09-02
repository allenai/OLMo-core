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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention.chunked_mask import PAD_CHUNK_ID

log = logging.getLogger(__name__)

__all__ = [
    "PooledDocProjector",
    "CompactedBatch",
    "compact_pooled_rows",
    "build_position_causal_bias",
    "masked_sdpa",
    "aux_matching_loss",
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
        for col in soft_mask.nonzero(as_tuple=True)[0].tolist():
            soft_rows.append(b)
            soft_cols.append(col)
            soft_docs.append(int(doc_of[order][col]))

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
        shadow_rows=torch.tensor(shadow_rows, dtype=torch.long, device=device),
        shadow_cols=torch.tensor(shadow_cols, dtype=torch.long, device=device),
        shadow_docs=torch.tensor(shadow_docs, dtype=torch.long, device=device),
        shadow_log_len=torch.tensor(shadow_log_len, dtype=torch.float32, device=device),
        shadow_doc_cols=sh_doc_cols,
        is_shadow=is_shadow,
    )


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
