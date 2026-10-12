"""
Learned per-token block skipping (mixture-of-depths style) -- the third routing dimension next to
the nested-width FFN router (:mod:`olmo_core.nn.nested_ffn_moe`) and the KV-cache router
(:mod:`olmo_core.nn.attention.kv_route`).

Per routed transformer block, a ``Linear(d_model, 1)`` router reads the block input and decides
for every token whether it **runs** the block or **skips** it. A skipped token's hidden state
passes the residual stream unchanged, and it is neither a query nor a key in that block's
attention (nothing later attends to it there), so the whole block -- Q/K/V/O projections, FFN and
attention scores -- is saved for that token. This is what reaches the per-token FLOPs the other
two routers cannot touch (projections, the FFN of un-routed layers, and on Qwen3.5 the recurrent
layers).

**Recurrent (GatedDeltaNet) blocks are routed too.** There the "not a key" half becomes "does not
write the recurrent state": the mixer receives the same ``block_keep`` and, for skipped tokens,
zeroes ``k``/``v`` before the causal short conv and forces ``beta=0``, ``g=0`` so the delta-rule
update is the identity at that step (:meth:`olmo_core.nn.attention.recurrent.GatedDeltaNet.forward`).
On a hybrid this is what lets the budget reach the 3/4 of the FFN that sits in GDN blocks -- the
GDN mixer itself is only ~9% of per-token FLOPs on Qwen3.5-4B, so a router over the mixer alone
would not be worth its overhead; skipping the whole block is.

Same recipe as the other routers so the three compose under one budget:

- hard decision in the forward, straight-through gradient through the coefficient
  ``1 + p_sel - p_sel.detach()`` on the block's residual update (value 1, gradient 1);
- keep-all init (router bias +10) so an enabled-but-untrained model reproduces its base;
- two-sided budget on the mean keep probability over tokens and routed blocks, annealed target.

Two implementations, selected by ``compact``:

**masked** (default, exact): the block runs on every token and the skipped tokens' outputs are
discarded (``where(keep, block(h), h)``), while the attention layer receives ``block_keep`` and
excludes skipped tokens as keys (through the KV-route masked path, which also drops them from the
KV cache at prefill). The maths is exactly mixture-of-depths, but **no wall-clock is saved** -- the
GPU still does every token's work, and the masking is itself extra work. Measured on a 4B step at
8k with nothing skipped, the masked path costs 1.21x a dense step; at keep=0.25 it is still 1.06x
(``debug/flexcompute_40x/router_matrix_4B_8192.json``). FLOPs are priced analytically by the
trainer's meter and do not convert.

**compact** (``compact=True``): the kept tokens are GATHERED into a shorter sequence, the block
runs on that, and the outputs are scattered back. ``cu_doc_lens`` is re-derived from the kept
tokens and ``position_ids`` carries their ORIGINAL positions, so RoPE geometry and document
boundaries are unchanged. Attention layers then take the ordinary flash path (there is no key mask
left to apply), which is where most of the saving comes from. Restrictions: batch size 1 (the
packed micro-batch this repo trains with) and prefill only; anything else falls back to masked.

    **Compaction is exact for attention and for the delta-rule scan, but NOT for GatedDeltaNet's
    causal short convolution.** In the masked path a skipped token occupies a conv slot with a
    zeroed k/v; in the compacted path it is gone, so a kept token's conv window sees its previous
    KEPT neighbours instead of zeros. Everything else matches to kernel rounding: attention over
    the compacted set is the same set of (query, key) pairs at the same RoPE positions, and the
    delta rule is invariant to removing identity updates (skipped tokens have ``beta=0, g=0``).
    Treat ``compact=True`` as a deliberate, flagged change of the recurrent mixer's definition of
    "skipped", not as a drop-in optimisation of the masked path.

**Span skipping** (:func:`set_block_skip_span`, orthogonal to both implementations): one router
decision covers ``span`` consecutive routed blocks instead of one. Same hard decision, same
straight-through coefficient, same keep-all init, same budget averaged over BLOCKS -- but the
router GEMM, the decision, the ``keep.sum().item()`` host sync and (with ``lazy_span_mix``) the
residual mix are paid once per span. Measured on 4B/8k, the per-block prologue adds **67 kernel
launches to a GDN block's 180**; ``span=8`` with the lazy mix cuts that to 34.6 per block. It costs
expressiveness: a token that skips a span's head skips every block in it and cannot re-enter
mid-span. See ``debug/flexcompute_40x/README.md``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

__all__ = [
    "BlockSkipHolder",
    "BlockSkipRouter",
    "install_block_skip",
    "block_skip_forward",
    "reset_block_skip_extras",
    "set_block_skip_span",
    "KEEP_INIT_BIAS",
]

KEEP_INIT_BIAS = 10.0


class BlockSkipHolder:
    """Per-forward accumulators, schedules and the decision cache shared by all routed blocks."""

    def __init__(
        self,
        *,
        target: float = 0.5,
        budget_weight: float = 1.0,
        two_sided: bool = True,
        target_anneal_calls: int = 0,
        seed: int = 0,
        start_layer: int = 0,
        n_layers: int = 0,
        span: int = 1,
        compact: bool = False,
        capacity_frac: Optional[float] = None,
    ):
        self.target = float(target)
        self.budget_weight = float(budget_weight)
        self.two_sided = bool(two_sided)
        self.target_anneal_calls = int(target_anneal_calls)
        self.seed = int(seed)
        self.start_layer = int(start_layer)
        self.n_layers = int(n_layers)
        self.span = max(1, int(span))
        # Mix the straight-through residual ONCE at the end of a span instead of after each block.
        # Exact, not an approximation -- see :func:`block_skip_forward`. Only read when span > 1.
        self.lazy_span_mix = True
        self.compact = bool(compact)
        self.capacity_frac = None if capacity_frac is None else float(capacity_frac)
        self.enabled = True
        self.calls = 0
        self.collect_loss = True
        self.routed_layers: List[int] = []
        self._choice_cache: Dict[tuple, torch.Tensor] = {}
        #: Kept-token INDICES per (layer, B, T), when they are known without a host sync (fixed
        #: capacity). ``_compact_block_forward`` prefers these over ``keep.nonzero()``.
        self._idx_cache: Dict[tuple, torch.Tensor] = {}
        self._span_decision: Dict[int, tuple] = {}
        self._span_input: Dict[int, torch.Tensor] = {}
        self._span_last: Dict[int, int] = {}
        self._all_routers: Optional[nn.ModuleDict] = (
            None  # pre-span router set (see set_block_skip_span)
        )
        self._exp_keep: List[torch.Tensor] = []
        self._exp_layers: List[int] = []
        self._hard_kept: Dict[int, float] = {}
        self._n_tokens: Dict[int, int] = {}
        self._depth_count: Optional[torch.Tensor] = None
        self._seen: set = set()
        self.last_per_layer_keep: Dict[int, float] = {}
        self.last_depth_hist: List[float] = []
        self.last_exp_per_layer: Dict[int, float] = {}
        self.last_exp_mean: Optional[float] = None
        self.cum_kept = 0.0
        self.cum_tokens = 0

    def _reset(self) -> None:
        self._exp_keep, self._exp_layers, self._hard_kept, self._n_tokens = [], [], {}, {}
        self._depth_count = None
        self._seen = set()

    def begin_forward(self, *, collect_loss: bool = True) -> None:
        if self._exp_keep:  # lagged expectations for budget_attach
            self.last_exp_per_layer = {
                int(li): float(e.detach()) for li, e in zip(self._exp_layers, self._exp_keep)
            }
            self.last_exp_mean = sum(self.last_exp_per_layer.values()) / len(
                self.last_exp_per_layer
            )
        if self._n_tokens:
            self.last_per_layer_keep = {
                li: self._hard_kept[li] / max(1, self._n_tokens[li]) for li in self._n_tokens
            }
            if self._depth_count is not None:
                n_r = max(1, len(self.routed_layers))
                hist = torch.bincount(self._depth_count, minlength=n_r + 1).float()
                self.last_depth_hist = (hist / hist.sum().clamp(min=1)).tolist()
        self._reset()
        self._choice_cache = {}
        self._idx_cache = {}
        self._span_decision = {}
        self._span_input = {}
        self.collect_loss = bool(collect_loss)
        self.calls += 1

    def set_calls(self, calls: int) -> None:
        self.calls = int(calls)

    def current_target(self) -> float:
        if self.target_anneal_calls <= 0:
            return self.target
        return 1.0 + (self.target - 1.0) * min(1.0, self.calls / self.target_anneal_calls)

    def accumulate(
        self,
        *,
        exp_keep: torch.Tensor,
        keep: torch.Tensor,
        layer_idx: int,
        span_layers: Optional[List[int]] = None,
    ) -> None:
        """
        Record one routing decision.

        ``span_layers`` (span skipping, ``span > 1``) is the full list of block indices this single
        decision governs. The decision is one tensor, so the ``.item()`` host sync and the depth
        bincount are paid ONCE for the whole span; the per-layer bookkeeping is then replicated on
        the host, which is exact because every block in the span has the same ``keep``.
        """
        layers = [layer_idx] if span_layers is None else list(span_layers)
        if self.collect_loss:
            # One expectation per BLOCK, so a span of K contributes K identical terms and the
            # budget mean stays a mean over routed blocks (not over decisions).
            for li in layers:
                self._exp_keep.append(exp_keep)
                self._exp_layers.append(li)
        kept = float(keep.sum().item())
        n = int(keep.numel())
        for li in layers:
            self._hard_kept[li] = kept
            self._n_tokens[li] = n
        self.cum_kept += kept * len(layers)
        self.cum_tokens += n * len(layers)
        flat = keep.reshape(-1).to(torch.int64)
        if len(layers) > 1:
            flat = flat * len(layers)
        self._depth_count = (
            flat
            if self._depth_count is None or self._depth_count.numel() != n
            else self._depth_count + flat
        )

    def regularization_loss(self) -> Optional[torch.Tensor]:
        if not self._exp_keep or self.budget_weight <= 0:
            return None
        gap = torch.stack(self._exp_keep).mean() - self.current_target()
        dev = gap.abs() if self.two_sided else torch.clamp(gap, min=0.0)
        return self.budget_weight * dev

    def mean_keep(self, *, last_forward: bool = True) -> float:
        d = (
            self.last_per_layer_keep
            if last_forward
            else {li: self._hard_kept[li] / max(1, self._n_tokens[li]) for li in self._n_tokens}
        )
        return sum(d.values()) / len(d) if d else 1.0

    def per_layer_keep(self, *, last_forward: bool = True) -> Dict[int, float]:
        if last_forward:
            return dict(self.last_per_layer_keep)
        return {li: self._hard_kept[li] / max(1, self._n_tokens[li]) for li in self._n_tokens}

    def cumulative_metrics(self) -> Dict[str, float]:
        return {
            "block_skip/cum_keep": self.cum_kept / max(1, self.cum_tokens),
            "block_skip/cum_tokens": float(self.cum_tokens),
        }


class BlockSkipRouter(nn.Module):
    """Per-token run/skip logit, initialised to "run everything"."""

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


def block_skip_decide(block: nn.Module, h: torch.Tensor):
    """
    Evaluate the (root-located) skip router on the block input outside the checkpoint region.

    Under **span skipping** (``span > 1``, see :func:`set_block_skip_span`) only the span's FIRST
    block carries a router: it decides once, and the remaining ``span - 1`` blocks read that
    decision back out of the holder. Those blocks therefore run NO router GEMM, take no decision,
    and trigger no host sync -- which is the whole point, since a 4B step sits at the CPU/GPU
    crossover and the per-block prologue is CPU work.
    """
    from .router_fn import router_logits

    cfg = block._bskip  # type: ignore[attr-defined]
    holder: BlockSkipHolder = cfg["holder"]
    layer_idx: int = cfg["layer_idx"]
    head: int = cfg.get("span_head", layer_idx)
    if head != layer_idx:
        cached = holder._span_decision.get(head)
        if cached is None:
            raise RuntimeError(
                f"block {layer_idx} is inside span {head} but that span has not decided yet; "
                "spans must be routed in block order (Transformer._run_blocks does this)"
            )
        return cached
    B, T, _ = h.shape
    logits = router_logits(h, cfg["router"]).squeeze(-1)
    p = torch.sigmoid(logits)
    key = (layer_idx, B, T)
    cache = holder._choice_cache
    if block.training and key in cache:
        keep = cache[key]
    elif holder.capacity_frac is not None:
        # Fixed capacity (mixture-of-depths): the block runs the top ``k`` tokens by router logit
        # instead of every token whose logit clears 0. ``k`` is a python int, so the compacted
        # shape is STATIC and ``_compact_block_forward`` needs no ``nonzero`` -- i.e. no host sync
        # per block. That is worth having once the rest of the step is cut: at 4B/16k the best
        # stacked arm runs 12.6% GPU-idle purely on those 32 syncs
        # (debug/flexcompute_40x/router_matrix_4B_16384_v3.json).
        #
        # This is a deliberate change of the router's SEMANTICS, not an optimisation of the
        # threshold rule: the keep fraction is met exactly, per block, by construction rather than
        # by the budget hinge. Keep-all init no longer reproduces the base model either -- capacity
        # binds from step 0 -- so it is opt-in.
        k = min(T, max(1, int(round(holder.capacity_frac * T))))
        idx = torch.topk(logits.detach(), k, dim=-1).indices.sort(dim=-1).values  # (B, k)
        keep = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, idx, True)
        if block.training:
            cache[key] = keep
            holder._idx_cache[key] = idx
    else:
        keep = logits.detach() > 0
        if block.training:
            cache[key] = keep
    if layer_idx not in holder._seen:
        span_layers = cfg.get("span_layers")
        holder.accumulate(
            exp_keep=p.mean(), keep=keep, layer_idx=layer_idx, span_layers=span_layers
        )
        holder._seen.add(layer_idx)
        for li in span_layers or ():
            holder._seen.add(li)
    if cfg.get("span_layers"):
        holder._span_decision[layer_idx] = (p, keep)
        holder._span_input[layer_idx] = h
    return p, keep


def _compact_kwargs(kwargs: Dict[str, Any], idx: torch.Tensor, seq_len: int) -> Dict[str, Any]:
    """
    Rewrite a block's kwargs for the compacted (kept-token-only) sequence.

    * ``block_keep`` is dropped -- there is nothing left to mask, so attention takes the plain
      flash path instead of the FlexAttention kept-key path. This is where most of the saving is.
    * ``position_ids`` carries the kept tokens' ORIGINAL positions, so RoPE geometry is unchanged.
      In the packed (``cu_doc_lens``) case that means each token's position WITHIN ITS DOCUMENT,
      which is what per-document RoPE would have given it uncompacted; ``Attention`` drops
      ``cu_doc_lens`` from the RoPE call when explicit positions are present, and still masks with
      it in the backend.
    * ``cu_doc_lens`` is re-derived as the per-document count of KEPT tokens, with emptied
      documents removed so no zero-length segment reaches flash-varlen.
    * ``max_doc_len`` is left alone: it is a launch-configuration upper bound and the original
      value still bounds the compacted documents, so keeping it avoids a second host sync.
    * per-token side inputs (``kv_route_p`` / ``kv_route_keep`` / ``kv_grad_mask``) are gathered
      the same way, so KV routing and block skipping compose.
    """
    kw = dict(kwargs)
    kw.pop("block_keep", None)
    pos = kw.get("position_ids")
    cu = kw.get("cu_doc_lens")
    if pos is not None:
        kw["position_ids"] = pos.index_select(1, idx)
    elif cu is None:
        kw["position_ids"] = idx.to(torch.int32).view(1, -1)
    if cu is not None:
        n_docs = cu.numel() - 1
        if n_docs > 1:
            doc = torch.bucketize(idx, cu[1:-1].to(idx.dtype), right=True)
        else:
            doc = torch.zeros_like(idx)
        if pos is None:
            # per-document RoPE position the token would have had uncompacted
            kw["position_ids"] = (idx - cu[:-1].to(idx.dtype)[doc]).to(torch.int32).view(1, -1)
        counts = torch.bincount(doc, minlength=n_docs)
        counts = counts[counts > 0]  # drop documents the router emptied
        kw["cu_doc_lens"] = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(cu.dtype)
    for name in ("kv_route_p", "kv_route_keep", "kv_grad_mask"):
        t = kw.get(name)
        if t is not None:
            kw[name] = t.index_select(1, idx)
    del seq_len
    return kw


def _straight_through_mix(h: torch.Tensor, out: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Residual update with the straight-through router coefficient (value 1, gradient 1)."""
    coef = (1.0 + p - p.detach()).to(h.dtype)[:, :, None]
    return h + coef * (out - h)


def _compact_block_forward(
    block: nn.Module,
    h: torch.Tensor,
    kwargs: Dict[str, Any],
    p: torch.Tensor,
    keep: torch.Tensor,
) -> torch.Tensor:
    """
    Run ``block`` on the KEPT tokens only, then scatter the result back.

    Costs one host sync per routed block (the kept count sets the compacted shape); that is the
    price of turning the skip into real work removed rather than masked. See the module docstring
    for what compaction changes numerically (the GDN short convolution).
    """
    T = h.shape[1]
    cfg = block._bskip  # type: ignore[attr-defined]
    holder: BlockSkipHolder = cfg["holder"]
    cached_idx = holder._idx_cache.get((int(cfg["layer_idx"]), h.shape[0], T))
    if cached_idx is not None:
        idx = cached_idx[0]  # fixed capacity: shape known on the host, no sync
    else:
        idx = keep[0].nonzero(as_tuple=True)[0]  # host sync: the compacted length
    n_keep = int(idx.numel())
    if n_keep == 0:
        return h
    if n_keep == T:
        # Nothing skipped: no gather, no key mask, plain block -- the fast path a keep-all
        # (freshly initialised or annealing) router spends most of its early steps on.
        return _straight_through_mix(h, block(h, **kwargs), p)
    h_c = h.index_select(1, idx)
    out_c = block(h_c, **_compact_kwargs(kwargs, idx, T))
    mixed = _straight_through_mix(h_c, out_c, p.index_select(1, idx))
    return h.index_copy(1, idx, mixed.to(h.dtype))


def block_skip_forward(
    block: nn.Module,
    h: torch.Tensor,
    kwargs: Dict[str, Any],
    p: Optional[torch.Tensor] = None,
    keep: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Run ``block`` on ``h`` with per-token skipping.

    The decision is made on the block input (:func:`block_skip_decide`, outside the block's
    checkpoint region); inside the block only the key mask is applied (patched forward,
    :func:`install_block_skip`); the straight-through residual mixing happens HERE, outside the
    region. Mixing inside the region pinned every layer's saved tensors under FSDP2 + activation
    checkpointing (72 GB vs 28 GB without it, ablation 2026-09-05); outside it retains one
    ``out - h`` tensor per block (~0.33 GB at 65k), which is affordable.
    """
    cfg = block._bskip  # type: ignore[attr-defined]
    holder: BlockSkipHolder = cfg["holder"]
    if not holder.enabled:
        return block(h, **kwargs)
    if p is None or keep is None:
        p, keep = block_skip_decide(block, h)
    if holder.compact and h.shape[0] == 1 and h.shape[1] > 1:
        return _compact_block_forward(block, h, kwargs, p, keep)
    out = block(h, **kwargs, skip_keep=keep)
    head = cfg.get("span_head")
    if head is not None and holder.span > 1 and holder.lazy_span_mix:
        # SPAN skipping, lazy mix: the where / straight-through is deferred to the LAST block of
        # the span and applied once against the span's INPUT. EXACT, not an approximation: in
        # every block of the span a skipped token is excluded as a key (attention) and does not
        # write the recurrent state (GDN), so no KEPT token's output depends on what a skipped
        # token's hidden state did inside the span, and the skipped token's own value is replaced
        # by the span input at the end. Saves span-1 mixes, span-1 wheres and span-1 retained
        # ``out - h`` tensors -- and the mix is ~20 of the ~55 launches the prologue adds.
        if cfg["layer_idx"] != holder._span_last[head]:
            return out
        h = holder._span_input[head]
    # Straight-through on the residual UPDATE of kept tokens (value: out; grad to p_keep through
    # <dL/dy, out - h>); skipped tokens pass h through untouched. At decode (T == 1) this is
    # where(keep, out, h) numerically, i.e. the query-side skip.
    p_sel = torch.where(keep, p, 1.0 - p)
    coef = (1.0 + p_sel - p_sel.detach()).to(h.dtype)[:, :, None]
    mixed = h + coef * (out - h)
    return torch.where(keep[:, :, None], mixed, h)


def _skipping_block_forward(self: nn.Module, x: torch.Tensor, *args, skip_keep=None, **kwargs):
    """``TransformerBlock.forward`` that also removes skipped tokens from the attention key set."""
    orig = self._bskip_orig_forward  # type: ignore[attr-defined]
    if skip_keep is None:
        return orig(x, *args, **kwargs)
    mixer = getattr(self, "attention", None)
    kvm = getattr(mixer, "kv_cache_manager", None) or getattr(mixer, "state_cache", None)
    if kvm is not None and x.shape[1] == 1:
        # Decode step: the token's K/V is still appended to every layer's cache (a per-row skip
        # would need per-row cache lengths); the query-side skip is applied by the caller.
        return orig(x, *args, **kwargs)
    block_keep = kwargs.get("block_keep")
    keep_attn = skip_keep if block_keep is None else (block_keep & skip_keep)
    return orig(x, *args, **{**kwargs, "block_keep": keep_attn})


def install_block_skip(
    blocks: nn.ModuleDict,
    holder: BlockSkipHolder,
    *,
    start_layer: int = 0,
    owner: nn.Module,
    span: int = 1,
) -> List[int]:
    """
    Attach a run/skip router for every block at or after ``start_layer`` and register the routers
    as ``owner.bskip_routers`` (a :class:`~torch.nn.ModuleDict` keyed by layer index) on the MODEL
    ROOT, not inside the blocks: the router runs *before* its block's forward, and under FSDP2 a
    block-child parameter is still sharded at that point (``aten.addmm: got mixed torch.Tensor and
    DTensor``, Qwen3 flexs runs 2026-09-05). Root-level parameters are gathered at the start of
    every forward. State-dict keys: ``bskip_routers.<i>.w.{weight,bias}``.

    :param span: Blocks per routing decision (1 = the default per-block behaviour). See
        :func:`set_block_skip_span`.

    :returns: The routed layer indices.
    """
    routers = nn.ModuleDict()
    routed: List[int] = []
    for key, block in blocks.items():
        li = int(key)
        if li < start_layer:
            continue
        attn = getattr(block, "attention", None)
        ref = getattr(attn, "w_q", None)  # Attention and GatedDeltaNet both expose w_q
        if ref is None:  # a mixer without a per-token key/state mask cannot be skipped exactly
            continue
        routers[key] = BlockSkipRouter(
            ref.in_features, device=ref.weight.device, dtype=ref.weight.dtype
        )
        block._bskip = {"holder": holder, "layer_idx": li, "router": routers[key]}  # type: ignore[attr-defined]
        if not hasattr(block, "_bskip_orig_forward"):
            import types

            block._bskip_orig_forward = block.forward  # type: ignore[attr-defined]
            block.forward = types.MethodType(_skipping_block_forward, block)  # type: ignore[method-assign]
        routed.append(li)
    owner.bskip_routers = routers  # type: ignore[assignment]
    holder.routed_layers = routed
    if span > 1:
        _regroup_into_spans(blocks, holder, owner, routed, span)
    return routed


def _regroup_into_spans(
    blocks: nn.ModuleDict,
    holder: BlockSkipHolder,
    owner: nn.Module,
    routed: List[int],
    span: int,
) -> None:
    """Keep one router per group of ``span`` consecutive routed blocks and drop the rest."""
    # The full set is stashed on the HOLDER, which is not an ``nn.Module``: dropped routers must
    # leave the model's state dict and parameter list (that is the point of spanning), but they
    # have to stay reachable so ``set_block_skip_span(model, 1)`` can put them back.
    all_routers = holder._all_routers
    if all_routers is None:
        all_routers = owner.bskip_routers  # type: ignore[assignment]
        holder._all_routers = all_routers
    kept = nn.ModuleDict()
    for pos, li in enumerate(routed):
        head_pos = (pos // span) * span
        head = routed[head_pos]
        cfg = blocks[str(li)]._bskip  # type: ignore[index]
        cfg["span_head"] = head
        if li == head:
            kept[str(li)] = all_routers[str(li)]
            cfg["router"] = all_routers[str(li)]
            cfg["span_layers"] = routed[head_pos : head_pos + span]
            holder._span_last[head] = cfg["span_layers"][-1]
        else:
            cfg["router"] = None  # this block never evaluates a router
            cfg.pop("span_layers", None)
    owner.bskip_routers = kept  # type: ignore[assignment]
    holder.span = int(span)


def set_block_skip_span(model: Any, span: int) -> int:
    """
    Group an already-installed block-skip routing into **spans of ``span`` consecutive routed
    blocks that share ONE decision**.

    Rationale: the binding constraint on a 4B step is CPU-side work, not FLOPs (a 4B/8k step is
    559 ms wall against 546 ms GPU-busy). Each per-block decision costs a router GEMM on the block
    input, a comparison, a ``keep.sum().item()`` HOST SYNC inside
    :meth:`BlockSkipHolder.accumulate`, and a straight-through residual mix. A span decision pays
    that once per ``span`` blocks for the same FLOP reduction.

    Everything else is unchanged: hard decision in the forward with the straight-through
    ``1 + p_sel - p_sel.detach()`` coefficient (now applied on ``span`` residual updates, so the
    one router receives gradient from all of them), keep-all init so an untrained model reproduces
    its base exactly, and the same two-sided annealed budget on the mean keep probability -- the
    budget still averages over BLOCKS, not over decisions, so a target of 0.5 means "half the
    block-passes", identically to ``span=1``.

    **What it costs in expressiveness.** A token that skips the head of a span skips every block in
    it; it cannot re-enter mid-span. The reachable set of per-token depth profiles shrinks from
    ``2^n_routed`` to ``2^(n_routed/span)``, and the router must commit ``span`` blocks ahead on the
    span head's input alone. Whether that matters is an empirical question about the task; it is not
    a strictly weaker family in the useful direction, since the trained FFN router's own solution
    (`rungs_v12-*.json`: 22 of 36 layers >99% one rung) is close to constant within a layer already.

    :param model: A model with block skipping already enabled (``enable_block_skip``).
    :param span: Blocks per decision; ``1`` restores per-block routing.

    :returns: The number of routers remaining.

    :raises ValueError: If block skipping is not enabled on ``model``.
    """
    bsk = getattr(model, "_block_skip", None)
    if bsk is None:
        raise ValueError("set_block_skip_span: block skipping is not enabled on this model")
    span = max(1, int(span))
    holder: BlockSkipHolder = bsk["holder"]
    routed: List[int] = list(bsk["routed"])
    if span == 1:
        allr = holder._all_routers
        for li in routed:
            cfg = model.blocks[str(li)]._bskip
            cfg.pop("span_head", None)
            cfg.pop("span_layers", None)
            holder._span_last.pop(li, None)
            if cfg.get("router") is None:
                if allr is None or str(li) not in allr:
                    raise ValueError(f"cannot restore span=1: no router kept for block {li}")
                cfg["router"] = allr[str(li)]
        if allr is not None:
            model.bskip_routers = allr
        holder.span = 1
        return len(model.bskip_routers)
    _regroup_into_spans(model.blocks, holder, model, routed, span)
    log.info(
        "Block skipping regrouped into spans of %d: %d routers over %d routed blocks",
        span,
        len(model.bskip_routers),
        len(routed),
    )
    return len(model.bskip_routers)


def reset_block_skip_extras(router: nn.Module) -> None:
    """Re-run the deterministic router init (after a strict=False base load)."""
    router.reset_parameters()


def enable_from_config_block(model: Any, block: Dict[str, Any]) -> None:
    """Enable block skipping on a built model as the trainer's ``config.json`` ``block_skip`` block says."""
    model.enable_block_skip(start_layer=int(block.get("start_layer", 0)))
    span = int(block.get("span", 1))
    if span > 1:
        set_block_skip_span(model, span)
    print(
        f"[block-skip] routing enabled from config.json: routed blocks "
        f"{model._block_skip['routed']} span={span}",
        flush=True,
    )
