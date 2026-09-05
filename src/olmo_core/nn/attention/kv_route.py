"""
Learned per-layer KV-cache allocation ("KV routing").

The attention analogue of the nested-width FFN router (:mod:`olmo_core.nn.nested_ffn_moe`): every
full-attention layer gets a tiny per-token router that decides whether the token's key/value is
**kept** (written to that layer's KV cache and attendable by every later query) or **dropped** (the
token still attends *from* its own position, but nothing after it can attend *to* it at that layer).

With ``R`` routed layers a token can end up in ``R + 1`` graded tiers -- kept everywhere (full
attention), kept in a few layers, or evicted from the cache entirely -- so "10 different KV caches
with different penalties" falls out of one binary decision per layer: the *tier* of a token is the
number of layers that keep it, and its cost is that number divided by ``R``. No explicit tier
vocabulary is needed, and unlike a shared tier the per-layer decision lets early layers keep a
token whose late layers do not need it (or vice versa).

Training recipe (identical in spirit to the FFN router):

1. **Hard routing, straight-through gradient.** The forward runs the hard mask (a dropped key is
   really excluded from softmax); the router gets ``d loss / d p`` through the coefficient
   ``1 + p_sel - p_sel.detach()`` on that token's K and V (value 1.0, gradient 1), where ``p_sel``
   is the probability of the decision taken (``p`` if kept, ``1 - p`` if dropped -- the dropped
   token's own diagonal still uses its scaled K/V).
2. **Base-preserving init.** The router bias starts at ``+10`` logits, so every token is kept and
   the model is bit-identical to its base; keys leave the cache only as the budget pulls them out.
3. **Budget on the batch mean.** ``loss += lambda * |mean_p_keep - target|`` (two-sided by default;
   ``mean_p_keep`` is averaged over tokens and routed layers), so the expected cache size lands on
   the target rather than every token being pushed toward eviction.

Compute accounting: the kept fraction is both the KV-cache size fraction and, for prefill, the
fraction of attention-score FLOPs on the routed layers (a query scores only kept keys). The trainer's
FLOP meter prices it that way; the FFN router prices its own share independently, so the two
compose additively into one "how much FFN and how much attention did this task need" picture.

At inference the same decision is applied at prefill: attention runs with the kept-key mask, then
each row's kept K/V are **compacted** into the KV cache (leftpad grows by the number of evicted
tokens), so decoding reads a genuinely smaller cache. Generated tokens are always kept (they are a
negligible share of the cache and the decode step cannot afford a second router pass).

Related work: Dynamic Memory Compression (Nawrot et al. 2024) learns per-head merge-or-append
decisions for the cache with a similar straight-through/budget recipe; the difference here is a
drop (not merge) decision shared across heads, applied to a frozen-architecture pretrained hybrid,
and priced on the same FLOP axis as the FFN router so the two allocations can be compared.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from . import Attention
    from .kv_cache import KVCacheManager

log = logging.getLogger(__name__)

__all__ = [
    "KVRouteHolder",
    "KVRouter",
    "install_kv_route",
    "kv_route_attention",
    "reset_kv_route_extras",
    "KEEP_INIT_BIAS",
]

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    _flex_attention = torch.compile(flex_attention, dynamic=True)
    import torch._dynamo

    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
    _HAS_FLEX = True
except Exception:  # pragma: no cover - flex unavailable on old torch
    _HAS_FLEX = False

#: Router bias at init, in logit units: ``sigmoid(10) = 0.99995`` keep probability, so an untrained
#: router keeps every key and the model reproduces its base exactly.
KEEP_INIT_BIAS = 10.0

#: FlexAttention block size for the kept-key mask (128 matches the flash tile on H100).
FLEX_BLOCK_SIZE = 128


class KVRouteHolder:
    """
    Shared per-forward state for all routed attention layers: the differentiable expected keep
    fraction (budget loss), hard keep counts (metrics), the annealed target/exploration schedules,
    and the routing-decision cache that lets an activation-checkpoint recompute replay the same
    keep/drop split.

    :param target: Mean keep fraction (over tokens and routed layers) the budget pulls toward.
    :param budget_weight: ``lambda`` on the budget term.
    :param two_sided: Penalize ``|mean - target|`` (default) instead of ``relu(mean - target)``.
    :param target_anneal_calls: Linearly anneal the target from 1.0 (keep all) to ``target`` over
        this many forwards, so the cache shrinks gradually instead of the router being yanked at
        step 0.
    :param explore_prob: Probability of flipping a token's keep decision at random (annealed to 0
        over ``explore_anneal_calls``). Off by default -- the budget term is differentiable in the
        router probabilities, so eviction needs no exploration to get started.
    :param start_layer: First (attention) layer index that is routed.
    """

    def __init__(
        self,
        *,
        target: float = 0.5,
        budget_weight: float = 1.0,
        two_sided: bool = True,
        target_anneal_calls: int = 0,
        explore_prob: float = 0.0,
        explore_anneal_calls: int = 0,
        seed: int = 0,
        start_layer: int = 0,
        n_layers: int = 0,
        layer_cost: Optional[Dict[int, float]] = None,
    ):
        self.target = float(target)
        self.budget_weight = float(budget_weight)
        self.two_sided = bool(two_sided)
        self.target_anneal_calls = int(target_anneal_calls)
        self.explore_prob = float(explore_prob)
        self.explore_anneal_calls = int(explore_anneal_calls)
        self.seed = int(seed)
        self.start_layer = int(start_layer)
        self.n_layers = int(n_layers)
        #: Optional per-layer weight on the cost (default 1.0 each): lets later layers be priced
        #: differently from earlier ones if a study wants asymmetric tiers.
        self.layer_cost: Dict[int, float] = dict(layer_cost or {})
        self.enabled = True
        self.calls = 0
        self.collect_loss = True
        self.routed_layers: List[int] = []
        self._choice_cache: Dict[tuple, torch.Tensor] = {}
        self._exp_keep: List[torch.Tensor] = []
        self._exp_weights: List[float] = []
        self._hard_kept: Dict[int, float] = {}
        self._n_tokens: Dict[int, int] = {}
        self._tier_count: Optional[torch.Tensor] = None
        self.last_per_layer_keep: Dict[int, float] = {}
        self.last_tier_hist: List[float] = []
        self.cum_kept: float = 0.0
        self.cum_tokens: int = 0

    # -- per-forward bookkeeping ------------------------------------------------------------------

    def _reset(self) -> None:
        self._exp_keep = []
        self._exp_weights = []
        self._hard_kept = {}
        self._n_tokens = {}
        self._tier_count = None

    def begin_forward(self, *, collect_loss: bool = True) -> None:
        """Snapshot the last forward's metrics, reset the accumulators, advance the schedules."""
        if self._n_tokens:
            self.last_per_layer_keep = {
                li: self._hard_kept[li] / max(1, self._n_tokens[li]) for li in self._n_tokens
            }
            if self._tier_count is not None:
                n_r = max(1, len(self.routed_layers))
                hist = torch.bincount(self._tier_count, minlength=n_r + 1).float()
                self.last_tier_hist = (hist / hist.sum().clamp(min=1)).tolist()
        self._reset()
        self._choice_cache = {}
        self.collect_loss = bool(collect_loss)
        self.calls += 1

    def set_calls(self, calls: int) -> None:
        """Pin the schedule clock (the trainer callback does this from the global step)."""
        self.calls = int(calls)

    def current_target(self) -> float:
        """Budget target for this call, linearly annealed from 1.0 (keep all) to ``target``."""
        if self.target_anneal_calls <= 0:
            return self.target
        frac = min(1.0, self.calls / self.target_anneal_calls)
        return 1.0 + (self.target - 1.0) * frac

    def current_explore(self) -> float:
        if self.explore_prob <= 0:
            return 0.0
        if self.explore_anneal_calls <= 0:
            return self.explore_prob
        return self.explore_prob * max(0.0, 1.0 - self.calls / self.explore_anneal_calls)

    def accumulate(self, *, exp_keep: torch.Tensor, keep: torch.Tensor, layer_idx: int) -> None:
        """
        Record one routed layer's decision.

        :param exp_keep: Scalar differentiable mean keep probability of this layer.
        :param keep: ``(B, T)`` bool hard decision.
        :param layer_idx: The layer.
        """
        w = float(self.layer_cost.get(layer_idx, 1.0))
        if self.collect_loss:
            self._exp_keep.append(exp_keep * w)
            self._exp_weights.append(w)
        kept = float(keep.sum().item())
        n = int(keep.numel())
        self._hard_kept[layer_idx] = kept
        self._n_tokens[layer_idx] = n
        self.cum_kept += kept
        self.cum_tokens += n
        flat = keep.reshape(-1).to(torch.int64)
        self._tier_count = (
            flat
            if self._tier_count is None or self._tier_count.numel() != n
            else self._tier_count + flat
        )

    def regularization_loss(self) -> Optional[torch.Tensor]:
        """The budget term for the current forward, or ``None`` when nothing was routed."""
        if not self._exp_keep or self.budget_weight <= 0:
            return None
        mean_keep = torch.stack(self._exp_keep).sum() / max(1e-9, sum(self._exp_weights))
        gap = mean_keep - self.current_target()
        dev = gap.abs() if self.two_sided else torch.clamp(gap, min=0.0)
        return self.budget_weight * dev

    def mean_keep(self, *, last_forward: bool = True) -> float:
        """Hard mean keep fraction over routed layers of the last completed forward."""
        d = (
            self.last_per_layer_keep
            if last_forward
            else {li: self._hard_kept[li] / max(1, self._n_tokens[li]) for li in self._n_tokens}
        )
        if not d:
            return 1.0
        return sum(d.values()) / len(d)

    def metrics(self) -> Dict[str, float]:
        """Hard-routing metrics of the last completed forward (for the trainer callback)."""
        out: Dict[str, float] = {
            "kv_route/mean_keep": self.mean_keep(),
            "kv_route/target": self.current_target(),
            "kv_route/explore": self.current_explore(),
        }
        for li, kf in sorted(self.last_per_layer_keep.items()):
            out[f"kv_route/keep_L{li}"] = kf
        for t, frac in enumerate(self.last_tier_hist):
            out[f"kv_route/tier{t}_frac"] = frac
        return out

    def cumulative_metrics(self) -> Dict[str, float]:
        """Keep fraction over every routed forward so far (what an evaluator reports)."""
        return {
            "kv_route/cum_keep": self.cum_kept / max(1, self.cum_tokens),
            "kv_route/cum_tokens": float(self.cum_tokens),
        }


class KVRouter(nn.Module):
    """Per-token keep/drop logit: ``Linear(d_model, 1)`` initialised to "keep everything"."""

    def __init__(
        self,
        d_model: int,
        *,
        init_bias: float = KEEP_INIT_BIAS,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.w = nn.Linear(d_model, 1, bias=True, device=device, dtype=dtype)
        self.init_bias = float(init_bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.w.weight)
        nn.init.constant_(self.w.bias, self.init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x).squeeze(-1).float()


def _forward_generator(
    holder: KVRouteHolder, layer_idx: int, device: torch.device
) -> torch.Generator:
    """Deterministic per-(forward, layer) generator so a recompute draws the same exploration."""
    gen = torch.Generator(device=device)
    gen.manual_seed(holder.seed * 1_000_003 + holder.calls * 4099 + layer_idx)
    return gen


def _doc_ids(cu_doc_lens: torch.Tensor, T: int) -> torch.Tensor:
    """``(T,)`` document index per position from cumulative document lengths."""
    pos = torch.arange(T, device=cu_doc_lens.device, dtype=cu_doc_lens.dtype)
    return torch.searchsorted(cu_doc_lens[1:].contiguous(), pos, right=True)


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


def _masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep: torch.Tensor,
    doc: Optional[torch.Tensor],
    leftpad: Optional[torch.Tensor],
    scale: float,
) -> torch.Tensor:
    """
    Causal attention where query ``i`` sees key ``j`` iff ``j <= i``, same document, ``j`` not
    left-padding, and (``j`` kept OR ``j == i``). Inputs/outputs are ``(B, T, H, D)``.

    CUDA path: the kept keys are **compacted** (gathered in position order, padded to a multiple of
    1024 per batch) and concatenated with the full key set, whose only allowed entries are the
    diagonal for dropped queries. FlexAttention's block mask then skips every key block outside the
    causal frontier of the compacted set (plus one diagonal block per query block), so the kernel's
    work scales with the kept fraction instead of masking a dense ``T x T`` (which the first
    version did: 16 ms vs 11 ms dense at 64k regardless of keep; compacted keep 0.25 -> 5.8 ms,
    keep 0.05 -> 3.2 ms, ``debug/flop_scaling/bench_kv_route_mask.py``).
    """
    B, T, Hq, D = q.shape
    Hk = k.shape[2]
    pos = torch.arange(T, device=q.device)
    if _HAS_FLEX and q.is_cuda:
        keep_valid = keep.to(torch.bool)
        if leftpad is not None:
            keep_valid = keep_valid & (pos[None, :] >= leftpad[:, None])
        counts = keep_valid.sum(1)  # (B,)
        Kp = min(T, _round_up(max(1, int(counts.max().item())), 1024))
        # kept positions first (stable sort keeps them in position order), then a padded tail
        order = torch.argsort((~keep_valid).to(torch.int8), dim=1, stable=True)[:, :Kp]
        valid_slot = torch.arange(Kp, device=q.device)[None, :] < counts[:, None]
        pos_k = torch.where(valid_slot, order, torch.full_like(order, T + 1))  # (B, Kp)
        gidx = order[:, :, None, None].expand(B, Kp, Hk, D)
        k_c = torch.gather(k, 1, gidx)
        v_c = torch.gather(v, 1, gidx)
        k_cat = torch.cat([k_c, k], dim=1).transpose(1, 2).contiguous()  # (B, Hk, Kp + T, D)
        v_cat = torch.cat([v_c, v], dim=1).transpose(1, 2).contiguous()
        q_ = q.transpose(1, 2).contiguous()
        dropped_q = ~keep_valid  # dropped (or pad) queries see their own key via the diagonal part

        def mask_mod(b, h, qi, kj):
            in_c = kj < Kp
            pk = pos_k[b, torch.clamp(kj, max=Kp - 1)]
            c_ok = in_c & (pk <= qi)
            if doc is not None:
                c_ok = c_ok & (doc[qi] == doc[torch.clamp(pk, max=T - 1)])
            d_ok = (~in_c) & ((kj - Kp) == qi) & dropped_q[b, qi]
            return c_ok | d_ok

        block_mask = create_block_mask(
            mask_mod,
            B,
            None,
            T,
            Kp + T,
            device=q.device,
            BLOCK_SIZE=(FLEX_BLOCK_SIZE, FLEX_BLOCK_SIZE),
            _compile=True,
        )
        out = _flex_attention(
            q_, k_cat, v_cat, block_mask=block_mask, scale=scale, enable_gqa=(Hq != Hk)
        )
        return out.transpose(1, 2).contiguous()

    # Reference path (CPU / no flex): materialised boolean mask + SDPA.
    from .backend import _repeat_kv

    q_ = q.transpose(1, 2)
    allowed = pos[None, :] <= pos[:, None]  # (T, T) causal
    if doc is not None:
        allowed = allowed & (doc[:, None] == doc[None, :])
    allowed = allowed[None] & (
        keep[:, None, :] | torch.eye(T, dtype=torch.bool, device=q.device)[None]
    )
    if leftpad is not None:
        allowed = allowed & (pos[None, None, :] >= leftpad[:, None, None])
    n_rep = Hq // Hk
    k_ = _repeat_kv(k, n_rep).transpose(1, 2)
    v_ = _repeat_kv(v, n_rep).transpose(1, 2)
    out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=allowed[:, None], scale=scale)
    return out.transpose(1, 2).contiguous()


def _write_compacted_cache(
    kvm: "KVCacheManager",
    k: torch.Tensor,
    v: torch.Tensor,
    keep: torch.Tensor,
    leftpad: Optional[torch.Tensor],
) -> None:
    """
    Prefill: write each row's KEPT keys/values right-aligned into the cache (``[T - n_kept, T)``)
    and grow that row's leftpad by the number of evicted tokens, so decode attends to a genuinely
    smaller cache while the generated tokens keep appending at position ``T``.
    """
    B, T = keep.shape
    pos = torch.arange(T, device=keep.device)
    new_leftpad = torch.zeros(B, dtype=torch.int32, device=keep.device)
    for b in range(B):
        valid = keep[b]
        if leftpad is not None:
            valid = valid & (pos >= leftpad[b])
        idx = valid.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        kvm.k_cache[b, T - n : T] = k[b, idx].to(kvm.k_cache.dtype)
        kvm.v_cache[b, T - n : T] = v[b, idx].to(kvm.v_cache.dtype)
        new_leftpad[b] = T - n
    kvm.cache_leftpad.copy_(new_leftpad)
    kvm.update_seqlen(T)


def kv_route_attention(
    attn: "Attention",
    x: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_doc_lens: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Routed attention for one layer: decide keep/drop per token, scale K/V by the straight-through
    coefficient, attend with the kept-key mask, and (at prefill) compact the KV cache.

    :param attn: The attention module (carries ``_kv_route`` and ``_kvr_router``).
    :param x: ``(B, T, d_model)`` block input the router reads.
    :param q: ``(B, T, n_heads, head_dim)`` (RoPE applied).
    :param k: ``(B, T, n_kv_heads, head_dim)``.
    :param v: ``(B, T, n_kv_heads, head_dim)``.
    :returns: ``(B, T, n_heads, head_dim)`` attention output.
    """
    cfg = attn._kv_route  # type: ignore[attr-defined]
    holder: KVRouteHolder = cfg["holder"]
    layer_idx: int = cfg["layer_idx"]
    kvm = attn.kv_cache_manager
    B, T, _, _ = q.shape

    if kvm is not None and T == 1:
        # Decode step: generated tokens are always kept; plain cached flash path.
        return attn.sdpa(q, k, v, cache_leftpad=cache_leftpad)
    if attn.cp_enabled:
        raise RuntimeError(
            "kv_route does not support context parallelism (single-rank attention only)"
        )

    router: KVRouter = attn._kvr_router  # type: ignore[attr-defined]
    logits = router(x)  # (B, T) float32
    p = torch.sigmoid(logits)

    cache_key = (layer_idx, B, T)
    cache = holder._choice_cache
    if attn.training and cache_key in cache:
        keep = cache[cache_key]
    else:
        keep = logits.detach() > 0
        explore = holder.current_explore()
        if attn.training and explore > 0:
            gen = _forward_generator(holder, layer_idx, x.device)
            flip = torch.rand(B, T, device=x.device, generator=gen) < explore
            keep = keep ^ flip
        if attn.training:
            cache[cache_key] = keep

    # Straight-through: value exactly 1 in the forward, gradient 1 w.r.t. p_sel (see module doc).
    p_sel = torch.where(keep, p, 1.0 - p)
    coef = (1.0 + p_sel - p_sel.detach()).to(k.dtype)[:, :, None, None]
    k = k * coef
    v = v * coef

    holder.accumulate(exp_keep=p.mean(), keep=keep, layer_idx=layer_idx)

    doc = _doc_ids(cu_doc_lens, T) if cu_doc_lens is not None else None
    if doc is not None and B != 1:
        raise RuntimeError("kv_route: cu_doc_lens implies a single packed row (B == 1)")
    leftpad = cache_leftpad.to(torch.int64) if cache_leftpad is not None else None
    att = _masked_attention(q, k, v, keep, doc, leftpad, attn.head_dim**-0.5)

    if kvm is not None:
        with torch.no_grad():
            _write_compacted_cache(kvm, k, v, keep, leftpad)
    return att


def install_kv_route(
    blocks: nn.ModuleDict, holder: KVRouteHolder, *, start_layer: int = 0
) -> List[int]:
    """
    Attach a :class:`KVRouter` to every plain full-attention layer at or after ``start_layer``
    (recurrent / sliding-window / landmark mixers are left alone) and register the layer with the
    holder. Adds NEW state-dict keys ``blocks.<i>.attention._kvr_router.w.{weight,bias}``.

    :returns: The routed layer indices.
    """
    from . import Attention

    routed: List[int] = []
    for key, block in blocks.items():
        li = int(key)
        if li < start_layer:
            continue
        attn = getattr(block, "attention", None)
        if type(attn) is not Attention:
            continue
        if attn.window_size not in (None, (-1, -1), -1):
            continue
        d_model = attn.w_q.in_features
        attn._kvr_router = KVRouter(  # type: ignore[attr-defined]
            d_model, device=attn.w_q.weight.device, dtype=attn.w_q.weight.dtype
        )
        attn._kv_route = {"holder": holder, "layer_idx": li}  # type: ignore[attr-defined]
        routed.append(li)
    holder.routed_layers = routed
    return routed


def reset_kv_route_extras(attn: nn.Module) -> None:
    """Re-run the deterministic init of a layer's router (after a strict=False base load)."""
    attn._kvr_router.reset_parameters()  # type: ignore[attr-defined]


def enable_from_config_block(model: Any, block: Dict[str, Any]) -> None:
    """Enable routing on a built model exactly as a trainer's ``config.json`` ``kv_route`` block says."""
    model.enable_kv_route(start_layer=int(block.get("start_layer", 0)))
    log.info(
        "[kv-route] routing enabled from config.json: start_layer=%s routed=%s",
        block.get("start_layer", 0),
        model._kv_route["routed"],
    )
