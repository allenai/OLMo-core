"""One compressive landmark at each document end.

The layer defaults to query-tiled exact attention with recomputed backward and
native GQA. The independent eager function remains a small-sequence oracle.
Both use the summary-token role schema with one SUMMARY per document.
"""

from typing import Optional

import torch

from olmo_core.exceptions import OLMoConfigurationError

from . import Attention
from .landmark_document_end_tiled import tiled_document_end_compressive_attention
from .summary_mask import ROLE_DOC_ID, ROLE_EXAMPLE_ID, ROLE_KIND, ROLE_SUMMARY_OFFSET, TokenKind


def document_end_compressive_attention(q, k, v, roles, softmax_scale=None, *, query_start=0):
    """Exact attention on (B, H, T, D) tensors and (B, 4, T) roles.

    Instructions and the causal local section compete with completed-document
    landmarks in the gate softmax. Each document gate is redistributed across
    that document's content AND landmark. Padding rows return zero.
    """
    B, H, Tq, D = q.shape
    T = k.shape[2]
    if k.shape != v.shape or k.shape[:2] != (B, H) or k.shape[-1] != D:
        raise ValueError("Reference attention requires matching batch/head dimensions")
    if query_start < 0 or query_start + Tq > T:
        raise ValueError("Query range is outside the key sequence")
    if roles.shape != (B, 4, T):
        raise ValueError("Expected roles with shape (B, 4, T)")
    scale = D**-0.5 if softmax_scale is None else softmax_scale
    # Preserve float64 for gradient checks; accumulate low-precision inputs in float32.
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    scores = (q.to(dtype) @ k.to(dtype).transpose(-1, -2)) * scale
    rows = []
    positions = torch.arange(T, device=q.device)
    for b in range(B):
        doc, kind, ex = roles[b, ROLE_DOC_ID], roles[b, ROLE_KIND], roles[b, ROLE_EXAMPLE_ID]
        lm = kind == int(TokenKind.SUMMARY)
        # Reject ambiguous layouts instead of treating several summaries as several gates.
        for p in positions[lm]:
            if int((lm & (ex == ex[p]) & (doc == doc[p])).sum()) != 1:
                raise ValueError("Expected exactly one landmark per document")
        outputs = []
        for row in range(Tq):
            i = query_start + row
            s = scores[b, :, row]
            if kind[i] == int(TokenKind.PAD):
                outputs.append(v[b, :, i].to(dtype) * 0)
                continue
            visible = (positions <= i) & (ex == ex[i]) & (kind != int(TokenKind.PAD))
            earlier = visible & lm & (doc < doc[i]) & (doc >= 0)
            local = (
                visible
                & ~lm
                & ((kind == int(TokenKind.INSTRUCTION)) | ((doc == doc[i]) & (doc >= 0)))
            )
            gate_pos = positions[local | earlier]
            if gate_pos.numel() == 0:
                raise ValueError("A non-padding query has no visible local tokens")
            gate = torch.softmax(s[:, gate_pos], dim=-1)
            values = []
            for p in gate_pos:
                if earlier[p]:
                    members = (
                        visible & (doc == doc[p]) & ((kind == int(TokenKind.DOC_CONTENT)) | lm)
                    )
                    inner = torch.softmax(s[:, members], dim=-1)
                    values.append((inner.unsqueeze(-1) * v[b, :, members].to(dtype)).sum(-2))
                else:
                    values.append(v[b, :, p].to(dtype))
            outputs.append((gate.unsqueeze(-1) * torch.stack(values, dim=-2)).sum(-2))
        rows.append(torch.stack(outputs, dim=-2))
    return torch.stack(rows).to(q.dtype)


class DocumentEndCompressiveLandmarkAttention(Attention):
    """Projection/RoPE-integrated document-end attention; requires explicit summary_roles.

    Supports unpadded, single-example cached generation into the trailing query
    region. Context parallelism, dropout, and sliding windows are not implemented.
    """

    def __init__(
        self,
        *,
        softmax_scale: Optional[float] = None,
        document_end_backend: str = "tiled",
        query_tile_size: int = 64,
        **kwargs,
    ):
        if document_end_backend not in ("tiled", "reference"):
            raise OLMoConfigurationError("document_end_backend must be 'tiled' or 'reference'")
        if query_tile_size < 1:
            raise OLMoConfigurationError("document-end query tile size must be positive")
        self.document_end_backend = document_end_backend
        self.query_tile_size = query_tile_size
        if kwargs.get("window_size") is not None or kwargs.get("dropout", 0):
            raise ValueError("Document-end attention does not support windows or dropout")
        super().__init__(softmax_scale=softmax_scale, **kwargs)
        self.softmax_scale = softmax_scale
        self._document_roles = None
        self._cached_roles = None

    def forward(self, x, summary_roles=None, causal_example=None, **kwargs):
        if summary_roles is None and (
            self.kv_cache_manager is None or int(self.kv_cache_manager.current_position()) == 0
        ):
            raise ValueError("Document-end compressive attention requires summary_roles")
        if causal_example is not None and bool(causal_example.any()):
            raise ValueError("Document-end compressive attention does not support mask mixing")
        self._document_roles = summary_roles
        try:
            return super().forward(x, **kwargs)
        finally:
            self._document_roles = None

    def sdpa(self, q, k, v, **kwargs):
        if self.cp_enabled:
            raise NotImplementedError("Document-end attention does not support CP")
        cache_leftpad = kwargs.pop("cache_leftpad", None)
        if any(value is not None for value in kwargs.values()):
            raise NotImplementedError(
                "Document-end attention uses roles, not backend packing metadata"
            )
        if self.kv_cache_manager is not None:
            return self._sdpa_cached(q, k, v, cache_leftpad=cache_leftpad)
        if cache_leftpad is not None and bool(cache_leftpad.any()):
            raise NotImplementedError("Document-end attention does not support left-padding")
        return self._attend(q, k, v, self._document_roles)

    def _attend(self, q, k, v, roles, query_start=0):
        if self.document_end_backend == "tiled":
            return (
                tiled_document_end_compressive_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    roles,
                    self.softmax_scale,
                    query_start=query_start,
                    query_tile_size=self.query_tile_size,
                )
                .transpose(1, 2)
                .contiguous()
            )
        n_rep = q.shape[2] // k.shape[2]
        k = k.repeat_interleave(n_rep, dim=2)
        v = v.repeat_interleave(n_rep, dim=2)
        return (
            document_end_compressive_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                roles,
                self.softmax_scale,
                query_start=query_start,
            )
            .transpose(1, 2)
            .contiguous()
        )

    def init_kv_cache_manager(self, batch_size: int, max_seq_len: int):
        from .kv_cache import KVCacheManager

        if batch_size != 1:
            raise NotImplementedError("Document-end cached generation requires batch_size=1")
        self._cached_roles = None
        self.kv_cache_manager = KVCacheManager(
            batch_size,
            max_seq_len,
            self.n_kv_heads,
            self.head_dim,
            self.w_k.weight.device,
            self.w_k.weight.dtype,
        )

    def _sdpa_cached(self, q, k, v, *, cache_leftpad=None):
        """Exact prefill followed by one or more trailing QUERY tokens.

        Incoming per-step roles cannot describe prior documents; retain prompt
        roles and append QUERY roles instead. Position zero always replaces old
        metadata, including after the ordinary KV cache manager is reset.
        """
        kvm = self.kv_cache_manager
        assert kvm is not None
        if q.shape[0] != 1 or (cache_leftpad is not None and bool(cache_leftpad.any())):
            raise NotImplementedError(
                "Document-end cached generation requires one unpadded example"
            )
        pos, count = int(kvm.current_position()), q.shape[1]
        total = pos + count
        if total > kvm.k_cache.shape[1]:
            raise ValueError("Document-end KV cache capacity exceeded")
        if pos == 0:
            roles = self._document_roles
            if roles is None or roles.shape != (1, 4, count):
                raise ValueError("Prefill requires roles covering the complete prompt")
            if bool((roles[:, ROLE_KIND] == int(TokenKind.PAD)).any()):
                raise ValueError("Cached prefill does not support padding")
            if bool((roles[:, ROLE_EXAMPLE_ID] != 0).any()):
                raise ValueError("Cached prefill must contain one example")
            last_kind = int(roles[0, ROLE_KIND, -1])
            if last_kind == int(TokenKind.DOC_CONTENT):
                raise ValueError("Cached prompt ends in an incomplete document")
        else:
            if self._cached_roles is None or self._cached_roles.shape[-1] != pos:
                raise RuntimeError("Document-end role cache is missing or out of sync")
            tail = self._cached_roles[:, :, -1:].expand(-1, -1, count).clone()
            last_kind = int(tail[0, ROLE_KIND, 0])
            if last_kind == int(TokenKind.SUMMARY):
                tail[:, ROLE_DOC_ID] += 1
            elif last_kind == int(TokenKind.INSTRUCTION):
                tail[:, ROLE_DOC_ID] = 0
            tail[:, ROLE_KIND] = int(TokenKind.QUERY)
            tail[:, ROLE_SUMMARY_OFFSET] = 0
            roles = torch.cat([self._cached_roles, tail], dim=-1)
        kvm.k_cache[:, pos:total].copy_(k)
        kvm.v_cache[:, pos:total].copy_(v)
        out = self._attend(
            q, kvm.k_cache[:, :total], kvm.v_cache[:, :total], roles, query_start=pos
        )
        self._cached_roles = roles
        kvm.record_leftpad(cache_leftpad)
        kvm.update_seqlen(count)
        return out
