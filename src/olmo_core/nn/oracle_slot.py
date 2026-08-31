"""
Oracle log-mass slots for pooled-doc soft-token training ("B1-oracle").

The maximal-fidelity static slot construction under the no-full-context-forward constraint: for
each context document, a per-layer, per-KV-head slot ``(k*, v*)`` is fit OFFLINE so that a single
KV entry reproduces the document's true attention behavior against generic queries:

* ``k*`` is a ridge least-squares fit (prior = the doc's mean key) such that
  ``scale * q . k*  ~=  logsumexp_t(scale * q . k_t)`` -- the slot's logit matches the doc's total
  softmax log-mass -- over a stash of real, derotated queries rotated to sampled relative offsets.
* ``v*`` is the doc's attention-weighted mean value under the same queries.

Keys are position-dependent (RoPE), so slots are stored in the **doc-center frame**: the captured
post-RoPE key at absolute position ``s`` is derotated by the doc center ``c``; at train time the
cached ``k*`` is re-rotated to the doc's actual center position with the attention module's own
:class:`~olmo_core.nn.rope.RotaryEmbedding` (see ``soft_kv_override`` in
:meth:`~olmo_core.nn.attention.Attention.forward`). Values are position-free.

The mechanical capacity study (``records/pooled-doc-kv-attention.md``) showed this closed form
reaches ~0.9 top-doc attention-output cosine where mean-pooling cannot; the KV-predictability
probe showed doc K/V are content-determined across contexts (R^2 .85-.98), which is what makes an
offline cache legitimate. Building the cache runs each doc with **within-doc (+ preamble)
attention only** -- cost O(sum L_i^2), never O(T^2) -- so no full-context forward is ever taken.

Cache builder: ``src/scripts/train/memexpress/pooledkv/build_oracle_slot_cache.py``.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch

log = logging.getLogger(__name__)

__all__ = [
    "doc_hash64",
    "rotate_half",
    "rotate_keys",
    "derotate_keys",
    "fit_oracle_slots_layer",
    "OracleSlotCacheWriter",
    "OracleSlotCache",
]


def doc_hash64(tokens: Union[np.ndarray, Sequence[int]]) -> int:
    """
    Stable 64-bit content hash of a document's token span (markers included), used to key the
    oracle slot cache. Builder and runtime must hash the same span: every position whose
    ``chunk_id`` equals the doc index, i.e. ``<|doc_start|> ... <|doc_end|>`` inclusive.

    :param tokens: The document's token ids in order.

    :returns: The first 8 bytes of the SHA-1 digest as an unsigned 64-bit int.
    """
    arr = np.ascontiguousarray(np.asarray(tokens, dtype=np.uint32))
    return int.from_bytes(hashlib.sha1(arr.tobytes()).digest()[:8], "little")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    The half-split rotation companion, identical to
    :meth:`~olmo_core.nn.rope.RotaryEmbedding._rotate_half` but for arbitrary leading dims:
    the last dim is viewed as ``(2, D/2)`` halves ``(x1, x2)`` and mapped to ``(-x2, x1)``.
    """
    d = x.shape[-1]
    x_ = x.view(*x.shape[:-1], 2, d // 2)
    x1, x2 = x_.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def rotate_keys(t: torch.Tensor, pos_sin: torch.Tensor, pos_cos: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE rotation with pre-selected buffers.

    :param t: ``(N, H, D)`` vectors.
    :param pos_sin: ``(N, D)`` sine buffer rows (already selected at the target positions).
    :param pos_cos: ``(N, D)`` cosine buffer rows.
    """
    sin = pos_sin[:, None, :].to(t.dtype)
    cos = pos_cos[:, None, :].to(t.dtype)
    return t * cos + rotate_half(t) * sin


def derotate_keys(t: torch.Tensor, pos_sin: torch.Tensor, pos_cos: torch.Tensor) -> torch.Tensor:
    """
    Exact inverse of :func:`rotate_keys` at the same positions (rotation by the negative angle),
    valid even when the buffers carry a YaRN-style ``attention_rescale_factor`` (the elementwise
    ``cos^2 + sin^2`` normalizer removes it).
    """
    sin = pos_sin[:, None, :].float()
    cos = pos_cos[:, None, :].float()
    out = (t.float() * cos - rotate_half(t.float()) * sin) / (cos * cos + sin * sin)
    return out.to(t.dtype)


def fit_oracle_slots_layer(
    keys_cf: torch.Tensor,
    values: torch.Tensor,
    doc_of: torch.Tensor,
    n_docs: int,
    q_stash: torch.Tensor,
    delta_sin: torch.Tensor,
    delta_cos: torch.Tensor,
    *,
    scale: float,
    ridge: float = 1.0,
    holdout_frac: float = 0.25,
    doc_block: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Fit one layer's oracle slots for a chunk of documents (all tensors on one device, fp32 math).

    :param keys_cf: ``(Nt, H_kv, D)`` center-frame (derotated) post-RoPE keys of all doc tokens.
    :param values: ``(Nt, H_kv, D)`` the matching values.
    :param doc_of: ``(Nt,)`` document index in ``[0, n_docs)`` of each token.
    :param q_stash: ``(M, H_q, D)`` canonical (derotated) real queries.
    :param delta_sin: ``(M, D)`` RoPE sine rows at each query's sampled relative offset.
    :param delta_cos: ``(M, D)`` cosine rows.
    :param scale: Attention logit scale (``head_dim ** -0.5``).
    :param ridge: Ridge strength toward the doc's mean key.
    :param holdout_frac: Fraction of query pairs excluded from the fit and used for R^2.

    :returns: ``(k_star (n_docs, H_kv, D), v_star (n_docs, H_kv, D), bias (n_docs,),
        diagnostics)`` where ``bias`` is a per-doc scalar logit offset SHARED across heads (a
        bias-free linear slot cannot express the constant part of the log-mass -- fitting it
        through the query-mean direction injects variance -- so the bias is fit jointly per head,
        averaged across heads, and the keys refit given it; at runtime it is added to the slot's
        attention-bias column). Diagnostics hold holdout ``r2_oracle`` and ``r2_meanpool``
        (mean key + log-len bias).
    """
    device = keys_cf.device
    Nt, H_kv, D = keys_cf.shape
    M, H_q, _ = q_stash.shape
    group = H_q // H_kv

    q_rot = rotate_keys(q_stash.float(), delta_sin, delta_cos)  # (M, H_q, D)

    n_hold = max(1, int(M * holdout_frac))
    perm = torch.randperm(M, device=device)
    fit_idx, hold_idx = perm[n_hold:], perm[:n_hold]

    counts = torch.zeros(n_docs, device=device).index_add(0, doc_of, torch.ones(Nt, device=device))
    log_len = counts.clamp(min=1.0).log()
    k_mean = torch.zeros(n_docs, H_kv, D, device=device)
    k_mean.index_add_(0, doc_of, keys_cf.float())
    k_mean = k_mean / counts.clamp(min=1.0)[:, None, None]

    k_star = torch.zeros(n_docs, H_kv, D, device=device)
    v_star = torch.zeros(n_docs, H_kv, D, device=device)
    bias = torch.zeros(n_docs, device=device)

    # Shared per-KV-head design matrices and ridge projectors (identical across docs, so the
    # normal equations are solved ONCE per head and every doc is a cheap projection). Two
    # projectors per head: P1 over the bias-augmented design ``[A | 1]`` (stage 1: joint
    # (k, c) fit -> per-doc bias), and P over ``A`` alone (stage 2: refit k given the shared c).
    eye = torch.eye(D, device=device)
    eye1 = torch.eye(D + 1, device=device)
    eye1[D, D] = 1e-4  # barely penalize the bias coordinate
    A_fit_g: List[torch.Tensor] = []
    A_hold_g: List[torch.Tensor] = []
    P_g: List[torch.Tensor] = []
    P1_g: List[torch.Tensor] = []
    for g in range(H_kv):
        qg = q_rot[:, g * group : (g + 1) * group, :]  # (M, group, D)
        A_fit = (qg[fit_idx].reshape(-1, D) * scale).float()
        A_hold = (qg[hold_idx].reshape(-1, D) * scale).float()
        A_fit_g.append(A_fit)
        A_hold_g.append(A_hold)
        P_g.append(torch.linalg.solve(A_fit.T @ A_fit + ridge * eye, A_fit.T))  # (D, rows)
        A1 = torch.cat([A_fit, torch.ones(A_fit.shape[0], 1, device=device)], dim=1)
        P1_g.append(torch.linalg.solve(A1.T @ A1 + ridge * eye1, A1.T))  # (D+1, rows)

    sse_o = torch.zeros((), device=device)
    sse_m = torch.zeros((), device=device)
    sst = torch.zeros((), device=device)
    neg_inf = torch.finfo(torch.float32).min

    for lo in range(0, n_docs, doc_block):
        hi = min(n_docs, lo + doc_block)
        sel = (doc_of >= lo) & (doc_of < hi)
        if not sel.any():
            continue
        kb = keys_cf[sel].float()  # (nt, H_kv, D)
        vb = values[sel].float()
        db = (doc_of[sel] - lo).long()  # (nt,)
        nd = hi - lo
        nt = kb.shape[0]
        # (M, H_q, nt) logits of every rotated stash query against every doc token.
        kb_rep = kb.repeat_interleave(group, dim=1)  # (nt, H_q, D)
        lg = torch.einsum("mhd,thd->mht", q_rot, kb_rep) * scale
        # Segment (per-doc) logsumexp via scatter: b[m, h, d] = lse over tokens of doc d.
        mx = torch.full((M, H_q, nd), neg_inf, device=device)
        mx.scatter_reduce_(2, db.view(1, 1, nt).expand(M, H_q, nt), lg, reduce="amax")
        ex = torch.exp(lg - mx.gather(2, db.view(1, 1, nt).expand(M, H_q, nt)))
        sm = torch.zeros(M, H_q, nd, device=device)
        sm.scatter_add_(2, db.view(1, 1, nt).expand(M, H_q, nt), ex)
        present = sm > 0
        b = torch.where(present, mx + sm.clamp(min=1e-45).log(), torch.zeros_like(sm))
        # Within-doc softmax weight of each token, summed over queries and the head group,
        # gives the doc's value target with one index_add per kv head.
        w = ex / sm.clamp(min=1e-45).gather(2, db.view(1, 1, nt).expand(M, H_q, nt))
        # Stage 1: per-kv-head joint (k, c) fit; the per-doc bias is the head-average of c.
        c_sum = torch.zeros(nd, device=device)
        b_fits: List[torch.Tensor] = []
        b_holds: List[torch.Tensor] = []
        for g in range(H_kv):
            bg = b[:, g * group : (g + 1) * group, :]  # (M, group, nd)
            # Row order must match A_fit/A_hold, which reshape (M_sel, group, D) query-major.
            b_fit = bg[fit_idx].reshape(-1, nd)  # (M_fit*group, nd)
            b_hold = bg[hold_idx].reshape(-1, nd)
            b_fits.append(b_fit)
            b_holds.append(b_hold)
            c_sum += (P1_g[g] @ b_fit)[D, :]  # bias coordinate of the augmented solution
        c = c_sum / H_kv  # (nd,)
        bias[lo:hi] = c

        # Stage 2: refit each head's key against the bias-corrected target, prior = mean key.
        for g in range(H_kv):
            ww = w[:, g * group : (g + 1) * group, :].sum(dim=(0, 1))  # (nt,)
            vt = torch.zeros(nd, D, device=device)
            vt.index_add_(0, db, vb[:, g, :] * ww[:, None])
            v_star[lo:hi, g, :] = vt / (M * group)
            km = k_mean[lo:hi, g, :]  # (nd, D)
            resid = b_fits[g] - c[None, :] - A_fit_g[g] @ km.T
            ks = km + (P_g[g] @ resid).T
            k_star[lo:hi, g, :] = ks
            # Holdout diagnostics vs the mean-key + log-len baseline (the exact-meanpool slot).
            pred_o = A_hold_g[g] @ ks.T + c[None, :]
            pred_m = A_hold_g[g] @ km.T + log_len[lo:hi][None, :]
            b_hold = b_holds[g]
            sse_o += ((pred_o - b_hold) ** 2).sum()
            sse_m += ((pred_m - b_hold) ** 2).sum()
            sst += ((b_hold - b_hold.mean()) ** 2).sum()

    diags = {
        "r2_oracle": float(1.0 - (sse_o / sst.clamp(min=1e-9)).item()),
        "r2_meanpool": float(1.0 - (sse_m / sst.clamp(min=1e-9)).item()),
    }
    return k_star, v_star, bias, diags


class OracleSlotCacheWriter:
    """
    Streaming writer for one shard of the oracle slot cache. Appends per-doc slots and hashes;
    multiple writers (one per builder rank) produce parts that :class:`OracleSlotCache` merges.

    :param path: Cache directory.
    :param part: Part name (e.g. ``"rank0"``).
    :param n_layers: Number of transformer layers cached.
    :param n_kv_heads: KV head count.
    :param head_dim: Head dimensionality.
    """

    def __init__(
        self, path: Union[str, Path], part: str, n_layers: int, n_kv_heads: int, head_dim: int
    ):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.part = part
        self.n_layers, self.n_kv_heads, self.head_dim = n_layers, n_kv_heads, head_dim
        self._slots_f = open(self.dir / f"slots_{part}.fp16.bin", "wb")
        self._bias_f = open(self.dir / f"bias_{part}.fp16.bin", "wb")
        self._hashes: List[int] = []

    def append(self, hashes: Sequence[int], slots: torch.Tensor, biases: torch.Tensor) -> None:
        """
        :param hashes: ``(n,)`` doc hashes (:func:`doc_hash64`).
        :param slots: ``(n, n_layers, 2, n_kv_heads, head_dim)`` slot tensor (k=0 center-frame, v=1).
        :param biases: ``(n, n_layers)`` per-doc, per-layer scalar logit biases.
        """
        assert slots.shape[1:] == (self.n_layers, 2, self.n_kv_heads, self.head_dim)
        assert slots.shape[0] == len(hashes)
        assert biases.shape == (slots.shape[0], self.n_layers)
        self._slots_f.write(slots.to(torch.float16).cpu().numpy().tobytes())
        self._bias_f.write(biases.to(torch.float16).cpu().numpy().tobytes())
        self._hashes.extend(int(h) for h in hashes)

    def close(self) -> None:
        self._slots_f.close()
        self._bias_f.close()
        np.save(self.dir / f"hashes_{self.part}.npy", np.array(self._hashes, dtype=np.uint64))
        meta_path = self.dir / "meta.json"
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        meta.update(
            n_layers=self.n_layers,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
        )
        parts = set(meta.get("parts", []))
        parts.add(self.part)
        meta["parts"] = sorted(parts)
        meta_path.write_text(json.dumps(meta, indent=2))


class OracleSlotCache:
    """
    Runtime reader for the oracle slot cache: memory-mapped slot parts + a sorted hash index.

    :param path: Cache directory written by :class:`OracleSlotCacheWriter`.
    """

    def __init__(self, path: Union[str, Path]):
        self.dir = Path(path)
        meta = json.loads((self.dir / "meta.json").read_text())
        self.n_layers = int(meta["n_layers"])
        self.n_kv_heads = int(meta["n_kv_heads"])
        self.head_dim = int(meta["head_dim"])
        self._maps: List[np.memmap] = []
        self._bias_maps: List[np.memmap] = []
        hash_arrs: List[np.ndarray] = []
        offsets: List[int] = []
        off = 0
        row = self.n_layers * 2 * self.n_kv_heads * self.head_dim
        for part in meta["parts"]:
            h = np.load(self.dir / f"hashes_{part}.npy")
            m = np.memmap(self.dir / f"slots_{part}.fp16.bin", dtype=np.float16, mode="r")
            assert m.size == h.size * row, f"part {part}: {m.size} != {h.size} * {row}"
            self._maps.append(m.reshape(h.size, self.n_layers, 2, self.n_kv_heads, self.head_dim))
            bm = np.memmap(self.dir / f"bias_{part}.fp16.bin", dtype=np.float16, mode="r")
            assert bm.size == h.size * self.n_layers
            self._bias_maps.append(bm.reshape(h.size, self.n_layers))
            hash_arrs.append(h)
            offsets.append(off)
            off += h.size
        all_hashes = np.concatenate(hash_arrs) if hash_arrs else np.zeros(0, dtype=np.uint64)
        self._order = np.argsort(all_hashes, kind="stable")
        self._sorted = all_hashes[self._order]
        self._part_starts = np.array(offsets + [off], dtype=np.int64)
        self.n_docs = int(off)
        self.misses = 0
        self.hits = 0

    def lookup(self, hashes: Sequence[int]) -> np.ndarray:
        """Map doc hashes to global cache indices (``-1`` for missing)."""
        q = np.asarray(hashes, dtype=np.uint64)
        pos = np.searchsorted(self._sorted, q)
        pos = np.clip(pos, 0, max(0, self._sorted.size - 1))
        found = self._sorted.size > 0
        ok = (self._sorted[pos] == q) if found else np.zeros(q.shape, dtype=bool)
        idx = np.where(ok, self._order[pos] if found else -1, -1).astype(np.int64)
        self.hits += int(ok.sum())
        self.misses += int((~ok).sum())
        return idx

    def gather(self, idx: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read slots for global indices ``idx`` (all ``>= 0``).

        :returns: ``(slots (n, n_layers, 2, n_kv_heads, head_dim), biases (n, n_layers))``
            fp16 CPU tensors.
        """
        out = np.empty(
            (idx.size, self.n_layers, 2, self.n_kv_heads, self.head_dim), dtype=np.float16
        )
        out_b = np.empty((idx.size, self.n_layers), dtype=np.float16)
        part_of = np.searchsorted(self._part_starts, idx, side="right") - 1
        for p in np.unique(part_of):
            sel = part_of == p
            local = idx[sel] - self._part_starts[p]
            out[sel] = self._maps[p][local]
            out_b[sel] = self._bias_maps[p][local]
        return torch.from_numpy(out), torch.from_numpy(out_b)
