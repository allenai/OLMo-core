"""What binds a TRAINING step once EVERY wall-clock component is cut at once?

``bench_router_matrix.py`` sweeps one lever at a time and one stacked point. This script asks the
Amdahl question directly: drive every lever down together, and at each aggressiveness level
re-measure the step partition ON THE ROUTED MODEL, so the bottleneck can be watched moving.

Four phases, selected with ``--phases`` (default all), each self-contained in one process with its
own repeated dense control -- nothing is ever compared across jobs, because eager timings on this
box drift up to 2x within a process.

``matrix``
    The FFN arms and the joint aggressiveness grid: head gating on, ``f_null`` x block-skip keep,
    plus one KV arm to re-confirm the KV router is negative. Interleaved, dense control every
    round, optional profiler pass for GPU-busy and launch count.

``partition``
    Cumulative ablation ON A ROUTED MODEL at several arms. Components are removed one after
    another so each delta is a marginal cost in the presence of what remains and the deltas sum
    exactly to the step. The tail is broken out -- the block shell (per-block norms and residual
    adds), the final norm, the surviving gated head, the embedding fwd+bwd (whose backward writes
    a full ``vocab x d_model`` gradient) and ``zero_grad`` -- because at aggressive settings that
    tail IS the answer.

``floor``
    The floor configuration (every block an Identity) swept over token count and batch size, and
    fitted as ``a + b*tokens``. ``a`` is the fixed per-step cost that amortises with step size and
    that CUDA graphs can remove; ``b*tokens`` is real per-token work that does not. That split is
    what decides whether the ceiling implied by the floor share is a wall or a function of batch.
    Also measures whether the gated head's cost depends on the number of supervised rows.

``graph``
    Attempts a full-step CUDA graph capture of the aggressive arms, and reports the measured GPU
    idle fraction (from the profiler) as the model-free upper bound on what graphs could buy --
    graphs can at best recover the idle time. See ``graph_probe.py`` for the bisection that finds
    WHICH host sync blocks a capture; this phase only reports whether the whole step captures.

    python research/kv_route/flexcompute_40x/bench_joint_sweep.py --scale 4B --ctx 16384
"""

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from olmo_core.nn.attention import Attention
from olmo_core.nn.lm_head import LMLossImplementation, LMOutputWithLoss
from olmo_core.nn.transformer import TransformerConfig

from ceiling import buckets, routed_total  # noqa: E402  (same directory)

VOCAB = 248320
OUT = os.path.dirname(os.path.abspath(__file__))
SCALES = {
    "0.8B": TransformerConfig.qwen3_5_0_8B,
    "4B": TransformerConfig.qwen3_5_4B,
    "9B": TransformerConfig.qwen3_5_9B,
    "27B": TransformerConfig.qwen3_5_27B,
}

#: Shape of the working (non-null) part of the FFN rung distribution, over rungs 0..n-2. Copied
#: from ``bench_router_matrix.py`` so the two scripts price the same operating points.
WORK_SHAPE = [0.08, 0.08, 0.28, 0.12, 0.44]


class Arm:
    """One operating point: FFN mean cost, KV keep fraction, block-skip keep fraction."""

    def __init__(
        self,
        name: str,
        *,
        ffn: Optional[float] = None,  # f_null, or None to disable the FFN router
        kv: Optional[float] = None,  # keep fraction, or None to disable
        bskip: Optional[float] = None,  # keep fraction, or None to disable
        head_gate: bool = False,
        compact: bool = False,  # gathered block skip instead of masked
        static_ffn: bool = False,  # delete the routed FFNs outright (no router at all)
        capacity: bool = False,  # fixed-capacity (top-k) block skip: static shapes, no host sync
    ):
        self.name = name
        self.ffn, self.kv, self.bskip, self.head_gate = ffn, kv, bskip, head_gate
        self.compact = compact
        self.static_ffn = static_ffn
        self.capacity = capacity


#: The joint grid. Head gating is on everywhere (it is loss-neutral and runs opposite to the
#: routers), block skip is compacted + fixed-capacity (the gathered path is the only one that
#: converts, and capacity removes the per-block host sync that low keeps would otherwise pay).
JOINT_FNULL = [0.50, 0.90, 1.00]
JOINT_KEEP = [0.50, 0.25, 0.125, 0.0625]


def build_arms(spec: str) -> List[Arm]:
    arms: List[Arm] = [
        Arm("dense"),
        # --- item 1: the FFN axis, re-measured with EVERY layer routed ------------------------
        Arm("ffn-overhead", ffn=-1.0),
        Arm("ffn-fnull0.00", ffn=0.00),
        Arm("ffn-fnull0.50", ffn=0.50),
        Arm("ffn-fnull0.75", ffn=0.75),
        Arm("ffn-fnull0.90", ffn=0.90),
        Arm("ffn-fnull1.00", ffn=1.00),
        Arm("ffn-static", static_ffn=True),  # deployment-only lever, labelled separately
        Arm("headgate", head_gate=True),
        Arm("bskipK-0.25", bskip=0.25, compact=True, capacity=True),
        # The graph-capture probes: no FFN router, fixed-capacity block skip, no head gating --
        # i.e. the aggressive stack with the step's head-side host syncs removed.
        Arm("staticFFN+bskipK-0.25", static_ffn=True, bskip=0.25, compact=True, capacity=True),
        Arm("staticFFN+bskipK-0.0625", static_ffn=True, bskip=0.0625, compact=True, capacity=True),
        # the previously published stacked points, for continuity
        Arm(
            "staticFFN+bskipK-0.25+headgate",
            static_ffn=True,
            bskip=0.25,
            compact=True,
            capacity=True,
            head_gate=True,
        ),
        Arm(
            "staticFFN+bskipK-0.0625+headgate",
            static_ffn=True,
            bskip=0.0625,
            compact=True,
            capacity=True,
            head_gate=True,
        ),
    ]
    # --- item 2: the joint aggressiveness grid (all-learned / trainable) ----------------------
    for f in JOINT_FNULL:
        for k in JOINT_KEEP:
            arms.append(
                Arm(
                    f"J-f{f:.2f}-k{k:.4f}",
                    ffn=f,
                    bskip=k,
                    compact=True,
                    capacity=True,
                    head_gate=True,
                )
            )
    # one KV arm to re-confirm the measured-negative result, then it is dropped
    arms.append(
        Arm(
            "J-f1.00-k0.2500+kv0.25",
            ffn=1.00,
            kv=0.25,
            bskip=0.25,
            compact=True,
            capacity=True,
            head_gate=True,
        )
    )
    if spec == "all":
        return arms
    wanted = set(spec.split(","))
    return [a for a in arms if a.name in wanted]


def rung_fracs(f_null: float, n_rungs: int) -> List[float]:
    """Token fractions per rung for a given null share (``f_null < 0`` = all on the full rung)."""
    if f_null < 0:
        return [1.0] + [0.0] * (n_rungs - 1)
    work = WORK_SHAPE[: n_rungs - 1]
    work = [w / sum(work) * (1.0 - f_null) for w in work]
    return work + [f_null]


def forced_choice(n: int, fracs: List[float], device) -> torch.Tensor:
    """A deterministic per-token rung assignment with the given fractions (contiguous blocks)."""
    out = torch.empty(n, dtype=torch.long, device=device)
    start = 0
    for g, f in enumerate(fracs):
        end = n if g == len(fracs) - 1 else min(n, start + int(round(f * n)))
        out[start:end] = g
        start = end
    return out


def forced_keep(B: int, T: int, frac: float, device) -> torch.Tensor:
    """A (B, T) bool keep-mask with exactly ``round(frac*T)`` kept, spread evenly over positions."""
    k = int(round(frac * T))
    keep = torch.zeros(B, T, dtype=torch.bool, device=device)
    if k > 0:
        idx = torch.linspace(0, T - 1, k, device=device).round().long().unique()
        keep[:, idx] = True
    return keep


class Forcer:
    """Refills every holder's decision cache after ``begin_forward`` clears it."""

    def __init__(self, model, B: int, T: int, device):
        self.model, self.B, self.T, self.device = model, B, T, device
        self.ffn_fracs: Optional[List[float]] = None
        self.kv_keep: Optional[float] = None
        self.bskip_keep: Optional[float] = None
        for cfg, filler in (
            (model._nested_ffn_moe, self._fill_ffn),
            (model._kv_route, self._fill_kv),
            (model._block_skip, self._fill_bskip),
        ):
            holder = cfg["holder"]
            orig = holder.begin_forward

            def wrapped(*a, _orig=orig, _filler=filler, _h=holder, **kw):
                _orig(*a, **kw)
                _filler(_h)

            holder.begin_forward = wrapped

    def _fill_ffn(self, holder) -> None:
        if self.ffn_fracs is None or not holder.enabled:
            return
        n = self.B * self.T
        choice = forced_choice(n, self.ffn_fracs, self.device)
        for li in self.model._nested_ffn_moe.get("routed_layers", []):
            holder._choice_cache[(int(li), n)] = choice

    def _fill_kv(self, holder) -> None:
        if self.kv_keep is None or not holder.enabled:
            return
        keep = forced_keep(self.B, self.T, self.kv_keep, self.device)
        for li in self.model._kv_route["routed"]:
            holder._choice_cache[(int(li), self.B, self.T)] = keep

    def _fill_bskip(self, holder) -> None:
        if self.bskip_keep is None or not holder.enabled:
            return
        keep = forced_keep(self.B, self.T, self.bskip_keep, self.device)
        # Under fixed capacity the real router hands the compacted path its indices for free
        # (they ARE the top-k), so the forced arm must supply them too or it would measure a
        # ``nonzero`` sync the capacity router never pays.
        idx = keep[0].nonzero(as_tuple=True)[0][None, :] if holder.capacity_frac else None
        for li in self.model._block_skip["routed"]:
            holder._choice_cache[(int(li), self.B, self.T)] = keep
            if idx is not None:
                holder._idx_cache[(int(li), self.B, self.T)] = idx


def timed(step, warmup: int, iters: int) -> List[float]:
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts


def median(xs: List[float]) -> float:
    return sorted(xs)[len(xs) // 2]


def profile_step(step, iters: int = 3) -> Tuple[float, float, Dict[str, float]]:
    """MEASURED GPU-busy seconds per iteration, kernel launches per iteration, per-kernel time."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
    tot, cnt = 0.0, 0
    per_kernel: Dict[str, float] = defaultdict(float)
    for e in prof.key_averages():
        if e.self_device_time_total > 0:
            tot += e.self_device_time_total
            cnt += e.count
            per_kernel[e.key] += e.self_device_time_total
    return tot / iters / 1e6, cnt / iters, {k: v / iters / 1e6 for k, v in per_kernel.items()}


class Identity(nn.Module):
    """Stands in for a statically deleted submodule: same shape out, no parameters in the graph."""

    def forward(self, x, *args, **kwargs):
        return x


class NoHead(nn.Module):
    """A loss with no vocabulary projection, so the head's cost can be ablated like any other.

    ``h.float().sum()`` keeps the backward connected to every hidden state (gradient of ones) at
    the cost of one reduction, so what the delta isolates is the head's projection, its CE kernel
    and its ``vocab x d_model`` weight gradient -- not the rest of the backward.
    """

    def forward(self, h, **kwargs):
        loss = h.float().sum()
        return LMOutputWithLoss(logits=None, loss=loss, ce_loss=loss.detach(), z_loss=None)


class ConstEmbeddings(nn.Module):
    """Stands in for the token embedding: a leaf activation of the same shape.

    Ablating this removes BOTH the gather and, more importantly, the embedding backward's dense
    ``vocab x d_model`` gradient write -- a fixed per-step cost that does not shrink with tokens.
    The stand-in is a leaf that requires grad, so the backward still runs to the same depth.
    """

    def __init__(self, h: torch.Tensor):
        super().__init__()
        self.h = h

    def forward(self, x, *args, **kwargs):
        return self.h


def _inner(mod):
    """Unwrap an activation-checkpoint wrapper.

    Under ``--ac`` ``model.blocks`` holds ``CheckpointWrapper`` objects whose ``__getattr__``
    forwards reads to the inner block but whose ``__setattr__`` does not, so assigning
    ``wrapper.feed_forward`` registers a submodule nothing ever runs and the ablation silently
    measures nothing (this exact trap made 27B's static-FFN arm read 1.000x once).
    """
    return getattr(mod, "_checkpoint_wrapped_module", mod)


class Rig:
    """One model with all three routers installed, plus every knob the phases need."""

    def __init__(self, args):
        self.args = args
        self.dev = torch.device("cuda")
        self.T = args.ctx
        self.B = args.batch
        from olmo_core.nn.attention import AttentionBackendName

        cfg = SCALES[args.scale](
            vocab_size=VOCAB, attn_backend=AttentionBackendName(args.attn_backend)
        )
        cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear
        torch.manual_seed(0)
        model = cfg.build(init_device="cpu").to(self.dev)
        model.init_weights(max_seq_len=self.T, device=self.dev)
        model = model.to(torch.bfloat16)
        model.train()
        self.model = model

        model.enable_nested_ffn_moe(
            start_layer=args.ffn_start_layer,
            divisors=[float(x) for x in args.divisors.split(",")],
            include_null=True,
            target_cost=0.01,
            explore_prob=0.0,
            width_multiple=8,
        )
        model._nested_ffn_moe["routed_layers"] = [
            i for i in range(len(model.blocks)) if i >= args.ffn_start_layer
        ]
        model.enable_kv_route(start_layer=0, target=0.5, explore_prob=0.0)
        model.enable_block_skip(start_layer=0, target=0.5, compact=True)
        model.budget_attach = True
        if args.ac:
            from olmo_core.nn.transformer import TransformerActivationCheckpointingMode

            model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)

        self.widths = model._nested_ffn_moe["widths"]
        self.costs = model._nested_ffn_moe["costs"]
        self.n_rungs = len(self.widths)
        self.n_layers = len(model.blocks)
        self.ffn_routed_layers = list(model._nested_ffn_moe["routed_layers"])
        # The whole reason for this re-run: with --ffn-start-layer 12 a third of the FFN was never
        # routed, so every FFN and stacked number was understated. Printed, not assumed.
        print(
            f"# FFN routed layers ({len(self.ffn_routed_layers)}/{self.n_layers}): "
            f"{self.ffn_routed_layers}",
            flush=True,
        )
        print(f"# rungs widths={self.widths} costs={[round(c, 4) for c in self.costs]}", flush=True)
        print(
            f"# kv-routed attention layers: {model._kv_route['routed']}\n"
            f"# block-skip routed blocks: {len(model._block_skip['routed'])}",
            flush=True,
        )
        self.ffn_routed_frac = len(self.ffn_routed_layers) / self.n_layers

        self.bkt = buckets(model, self.T)
        self.dense_flops = self.bkt["total"]
        self.forcer = Forcer(model, self.B, self.T, self.dev)
        self.ffn_holder = model._nested_ffn_moe["holder"]
        self.kv_holder = model._kv_route["holder"]
        self.bs_holder = model._block_skip["holder"]
        # The block-shell ablation replaces whole ModuleDict entries, so the entries have to be
        # restorable -- otherwise the first visit to that stage deletes every block for the rest
        # of the process and every later measurement is of an empty model.
        self.real_blocks = {k: b for k, b in model.blocks.items()}
        self.bskip_cfgs = {k: b._bskip for k, b in model.blocks.items()}
        self.real_ffn = {k: _inner(b).feed_forward for k, b in model.blocks.items()}
        self.real_mix = {k: _inner(b).attention for k, b in model.blocks.items()}
        self.is_attn = {k: type(m) is Attention for k, m in self.real_mix.items()}
        self.real_head = model.lm_head
        # The final norm lives INSIDE the LM head (LMHead.norm), not on the model root.
        self.real_norm = getattr(self.real_head, "norm", None)
        self.real_emb = model.embeddings
        self.dead = Identity().to(self.dev)
        self.no_head = NoHead().to(self.dev)
        self._const_emb: Dict[Tuple[int, int], ConstEmbeddings] = {}

        self.set_tokens(self.B, self.T)
        self.use_zero_grad = True

    # -- data ----------------------------------------------------------------------------------
    def set_tokens(self, B: int, T: int, n_sup: Optional[int] = None) -> None:
        """Re-make the inputs (and the Forcer's shapes) for a new batch/length."""
        self.B, self.T = B, T
        self.forcer.B, self.forcer.T = B, T
        if n_sup is None:
            n_sup = max(1, int(T * self.args.loss_density))
        self.n_sup = n_sup
        self.labels = torch.full((B, T), -100, dtype=torch.long, device=self.dev)
        self.labels[:, -n_sup:] = torch.randint(0, VOCAB, (B, n_sup), device=self.dev)
        self.ids = torch.randint(0, VOCAB, (B, T), device=self.dev)
        self.doc_kwargs: Dict[str, Any] = {}
        if self.args.docs > 1:
            per = T // self.args.docs
            lens = [per] * (self.args.docs - 1) + [T - per * (self.args.docs - 1)]
            self.doc_kwargs = {
                "doc_lens": torch.tensor([lens] * B, dtype=torch.int32, device=self.dev),
                "max_doc_lens": [max(lens)] * B,
            }

    def const_emb(self) -> ConstEmbeddings:
        """The embedding stand-in for the current (batch, tokens), cached so the ablation does not
        allocate a fresh activation every time a stage is re-visited."""
        key = (self.B, self.T)
        if key not in self._const_emb:
            h = torch.zeros(
                self.B,
                self.T,
                self.real_emb.weight.shape[-1],
                device=self.dev,
                dtype=torch.bfloat16,
            ).requires_grad_(True)
            self._const_emb[key] = ConstEmbeddings(h)
        return self._const_emb[key]

    def step(self) -> None:
        if self.use_zero_grad:
            self.model.zero_grad(set_to_none=True)
        out = self.model(
            self.ids,
            labels=self.labels,
            loss_reduction="sum",
            return_logits=False,
            **self.doc_kwargs,
        )
        loss = out.loss
        if getattr(out, "auxiliary_losses", None):
            for v in out.auxiliary_losses.values():
                loss = loss + v
        loss.backward()

    # -- arms ----------------------------------------------------------------------------------
    def apply(self, arm: Arm, *, head_gate: Optional[bool] = None) -> None:
        m = self.model
        for k, b in self.real_blocks.items():
            m.blocks[k] = b
        for k, b in m.blocks.items():
            _inner(b).feed_forward = (
                self.dead
                if (arm.static_ffn and int(k) >= self.args.ffn_start_layer)
                else self.real_ffn[k]
            )
            _inner(b).attention = self.real_mix[k]
        m.lm_head = self.real_head
        if self.real_norm is not None:
            self.real_head.norm = self.real_norm
        m.embeddings = self.real_emb
        self.ffn_holder.enabled = arm.ffn is not None and not arm.static_ffn
        self.kv_holder.enabled = arm.kv is not None
        self.bs_holder.enabled = arm.bskip is not None
        self.bs_holder.compact = arm.compact
        self.bs_holder.capacity_frac = arm.bskip if arm.capacity else None
        for k, b in m.blocks.items():
            b._bskip = self.bskip_cfgs[k] if arm.bskip is not None else None
        self.forcer.ffn_fracs = rung_fracs(arm.ffn, self.n_rungs) if arm.ffn is not None else None
        self.forcer.kv_keep = arm.kv
        self.forcer.bskip_keep = arm.bskip
        self.real_head._sparse_loss_positions = arm.head_gate if head_gate is None else head_gate
        self.use_zero_grad = True

    def arm_flops(self, arm: Arm) -> float:
        """CALCULATED per-token training FLOPs at this arm's operating point."""
        ffn_cost = 1.0
        if arm.static_ffn:
            ffn_cost = 1 - self.ffn_routed_frac
        elif arm.ffn is not None:
            fr = rung_fracs(arm.ffn, self.n_rungs)
            routed_cost = sum(f * c for f, c in zip(fr, self.costs))
            ffn_cost = (1 - self.ffn_routed_frac) + self.ffn_routed_frac * routed_cost
        skip = arm.bskip if arm.bskip is not None else 1.0
        kv = arm.kv if arm.kv is not None else 1.0
        return routed_total(
            self.bkt,
            skip_keep=skip,
            ffn_cost=ffn_cost,
            kv_keep=kv * skip,
            head_frac=self.args.loss_density if arm.head_gate else 1.0,
        )


# -----------------------------------------------------------------------------------------------
# phase: matrix
# -----------------------------------------------------------------------------------------------
def phase_matrix(rig: Rig, args) -> Dict[str, Any]:
    arms = build_arms(args.arms)
    samples: Dict[str, List[float]] = {a.name: [] for a in arms}
    control: List[float] = []
    for rnd in range(args.rounds):
        rig.apply(Arm("dense"))
        try:
            control.append(median(timed(rig.step, 1, 3)))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            control.append(float("nan"))
        print(f"# round {rnd}: control median {control[-1] * 1e3:.1f} ms", flush=True)
        for arm in arms:
            rig.apply(arm)
            try:
                t0 = time.perf_counter()
                ts = timed(rig.step, 1 if rnd else 2, args.iters)
                samples[arm.name].extend(ts)
                print(
                    f"#   [{rnd}] {arm.name}: median {median(ts) * 1e3:.1f} ms "
                    f"({time.perf_counter() - t0:.0f}s)",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"#   {arm.name}: OOM", flush=True)
            except Exception as exc:
                torch.cuda.empty_cache()
                print(f"#   {arm.name}: ERROR {type(exc).__name__}: {exc}", flush=True)

    prof: Dict[str, Dict[str, float]] = {}
    if args.profile:
        for arm in arms:
            rig.apply(arm)
            try:
                gpu, launches, per_kernel = profile_step(rig.step)
                prof[arm.name] = {"gpu_busy_s": gpu, "launches": launches}
                if args.kernels:
                    top = sorted(per_kernel.items(), key=lambda kv: -kv[1])[:12]
                    print(f"#   kernels {arm.name} (total {gpu * 1e3:.1f} ms):", flush=True)
                    for name, t in top:
                        print(f"#     {t * 1e3:>8.1f} ms  {name[:72]}", flush=True)
            except Exception as exc:
                print(f"#   profile {arm.name}: {type(exc).__name__}: {exc}", flush=True)

    def stats(xs: List[float]):
        if not xs:
            return None
        xs = sorted(xs)
        p = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]  # noqa: E731
        return dict(p10=p(0.1), median=xs[len(xs) // 2], p90=p(0.9), n=len(xs))

    dense_stat = stats(samples.get("dense", []))
    rows = []
    for arm in arms:
        s = stats(samples[arm.name])
        f = rig.arm_flops(arm)
        row = dict(
            arm=arm.name,
            ffn_fnull=arm.ffn,
            kv_keep=arm.kv,
            bskip_keep=arm.bskip,
            compact=arm.compact,
            capacity=arm.capacity,
            static_ffn=arm.static_ffn,
            head_gate=arm.head_gate,
            trainable=not arm.static_ffn,
            flops_per_token=f,
            flop_reduction=rig.dense_flops / f if f > 0 else float("inf"),
            **(s or {}),
        )
        if s and dense_stat:
            row["wall_speedup"] = dense_stat["median"] / s["median"]
            row["conversion"] = row["wall_speedup"] / row["flop_reduction"]
        if arm.name in prof:
            row.update(prof[arm.name])
            if s:
                row["gpu_busy_frac"] = prof[arm.name]["gpu_busy_s"] / s["median"]
                row["idle_frac"] = 1.0 - row["gpu_busy_frac"]
        rows.append(row)

    print(
        f"\n{'arm':>28} {'FLOPx':>7} {'p10 ms':>8} {'med ms':>8} {'p90 ms':>8} "
        f"{'wallx':>7} {'conv':>6} {'busy%':>6} {'launch':>7}"
    )
    for r in rows:
        if r.get("median") is None:
            print(f"{r['arm']:>28} {r['flop_reduction']:>7.2f}   (no samples)")
            continue
        print(
            f"{r['arm']:>28} {r['flop_reduction']:>7.2f} {r['p10'] * 1e3:>8.1f} "
            f"{r['median'] * 1e3:>8.1f} {r['p90'] * 1e3:>8.1f} "
            f"{r.get('wall_speedup', float('nan')):>7.3f} "
            f"{r.get('conversion', float('nan')):>6.2f} "
            f"{100 * r.get('gpu_busy_frac', float('nan')):>6.1f} "
            f"{r.get('launches', float('nan')):>7.0f}"
        )
    cd = [c for c in control if c == c]
    if cd:
        print(
            f"\n# control drift across rounds: {max(cd) / min(cd):.3f}x "
            f"({' '.join(f'{c * 1e3:.0f}' for c in control)} ms)",
            flush=True,
        )
    return dict(rows=rows, control=control)


# -----------------------------------------------------------------------------------------------
# phase: partition (on the ROUTED model)
# -----------------------------------------------------------------------------------------------
#: Ablation order. Each entry removes one more component; the deltas are marginal costs in the
#: presence of what remains and sum exactly to ``full``.
PARTITION_STAGES = [
    "full",  # the arm, head gating OFF so the gating saving is visible
    "headgate",  # + gate the head to supervised rows
    "ffn",  # + delete the FFN (and its router)
    "attn",  # + delete the full-attention mixers
    "gdn",  # + delete the GatedDeltaNet mixers
    "bskip",  # + turn the block-skip / KV routing machinery off entirely
    "shell",  # + delete the BLOCK ITSELF (its two norms and two residual adds)
    "norm",  # + remove the final norm (LMHead.norm)
    "head",  # + replace the LM head with a projection-free loss
    "emb",  # + replace the token embedding with a leaf activation
    "zerograd",  # + stop calling zero_grad
]
PARTITION_KEYS = [
    ("head_waste", "full", "headgate"),
    ("ffn", "headgate", "ffn"),
    ("attn", "ffn", "attn"),
    ("gdn", "attn", "gdn"),
    ("route_overhead", "gdn", "bskip"),
    ("block_shell", "bskip", "shell"),
    ("final_norm", "shell", "norm"),
    ("head_resid", "norm", "head"),
    ("embed", "head", "emb"),
    ("zero_grad", "emb", "zerograd"),
]


def set_stage(rig: Rig, arm: Arm, k: int) -> None:
    """Configure the model with the first ``k`` ablations of ``PARTITION_STAGES`` applied.

    Indexable rather than a one-way chain, because a strictly sequential chain is not measurable
    on this box: the process drifts upward by up to 2x, so every later stage is biased slow and
    the small tail components (the floor's own parts) came out NEGATIVE. With stages addressable,
    the caller can visit them in alternating order across rounds, which cancels a monotone drift.
    """
    m = rig.model
    rig.apply(arm, head_gate=False)  # stage 0 = the arm itself, head gating off
    if k >= 1:
        rig.real_head._sparse_loss_positions = True
    if k >= 2:
        for key, b in m.blocks.items():
            _inner(b).feed_forward = rig.dead
        rig.ffn_holder.enabled = False
    if k >= 3:
        for key, b in m.blocks.items():
            if rig.is_attn[key]:
                _inner(b).attention = rig.dead
    if k >= 4:
        for key, b in m.blocks.items():
            if not rig.is_attn[key]:
                _inner(b).attention = rig.dead
    if k >= 5:
        rig.bs_holder.enabled = False
        rig.kv_holder.enabled = False
        for key, b in m.blocks.items():
            b._bskip = None
    if k >= 6:
        # Replacing the MIXER and the FFN leaves the block SHELL running: two RMSNorms and two
        # residual adds per block, over the full (B, T, d_model) stream, forward and backward.
        # Those are per-token GPU work that no router touches, and until this stage existed they
        # sat unattributed inside 'base' and made the floor look like unexplained overhead.
        for key in list(m.blocks.keys()):
            m.blocks[key] = rig.dead
    if k >= 7 and rig.real_norm is not None:
        rig.real_head.norm = rig.dead
    if k >= 8:
        m.lm_head = rig.no_head
    if k >= 9:
        m.embeddings = rig.const_emb()
    if k >= 10:
        rig.use_zero_grad = False


def partition_once(rig: Rig, arm: Arm, iters: int, rounds: int) -> Dict[str, Any]:
    """Cumulative ablation on the routed model at ``arm``'s operating point.

    Every stage is visited once per round, forward on even rounds and backward on odd ones, and
    the reported time per stage is the median over rounds. The spread of stage 0 across rounds is
    reported as ``full_drift`` and is the honest error bar on every delta below.
    """
    n = len(PARTITION_STAGES)
    samples: Dict[str, List[float]] = {s: [] for s in PARTITION_STAGES}
    for rnd in range(rounds):
        order = range(n) if rnd % 2 == 0 else range(n - 1, -1, -1)
        for k in order:
            set_stage(rig, arm, k)
            samples[PARTITION_STAGES[k]].append(median(timed(rig.step, 1, iters)))
    t = {s: median(v) for s, v in samples.items()}
    rig.apply(arm)  # full restore
    out: Dict[str, Any] = {f"t_{k}_s": v for k, v in t.items()}
    out["stage_samples"] = samples
    full_s = samples["full"]
    out["full_drift"] = max(full_s) / min(full_s) if min(full_s) > 0 else float("nan")
    full = t["full"]
    for name, a, b_ in PARTITION_KEYS:
        out[f"{name}_s"] = t[a] - t[b_]
        out[f"{name}_pct"] = 100 * (t[a] - t[b_]) / full
    out["base_s"] = t["zerograd"]
    out["base_pct"] = 100 * t["zerograd"] / full
    out["sum_pct"] = sum(out[f"{n_}_pct"] for n_, _, _ in PARTITION_KEYS) + out["base_pct"]
    # Everything the routers cannot touch, at THIS operating point. The reciprocal is the
    # remaining headroom above the arm, not above dense. ``block_shell`` belongs here and not with
    # the components: ablating a block's MIXER and FFN leaves its two norms and two residual adds
    # running over the whole stream, and no router removes them -- measured at 4B/16k it IS the
    # floor (8.4% of a dense step, with nothing measurable left underneath).
    out["floor_s"] = (
        sum(
            out[f"{n_}_s"]
            for n_ in (
                "route_overhead",
                "block_shell",
                "head_resid",
                "final_norm",
                "embed",
                "zero_grad",
            )
        )
        + out["base_s"]
    )
    out["floor_pct"] = 100 * out["floor_s"] / full
    out["cap_from_here"] = full / out["floor_s"] if out["floor_s"] > 0 else float("inf")
    return out


def phase_partition(rig: Rig, args) -> Dict[str, Any]:
    names = args.partition_arms.split(",")
    all_by_name = {a.name: a for a in build_arms("all")}
    rows = []
    dense_full: Optional[float] = None
    for i, name in enumerate(names):
        arm = all_by_name.get(name)
        if arm is None:
            print(f"# partition: unknown arm {name}", flush=True)
            continue
        try:
            r = partition_once(rig, arm, args.partition_iters, args.partition_rounds)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rig.apply(arm)
            print(f"# partition {name}: OOM", flush=True)
            continue
        except Exception as exc:
            torch.cuda.empty_cache()
            rig.apply(arm)
            print(f"# partition {name}: ERROR {type(exc).__name__}: {exc}", flush=True)
            continue
        r["arm"] = name
        if i == 0 and name == "dense":
            dense_full = r["t_full_s"]
        if dense_full:
            r["speedup_vs_dense"] = dense_full / r["t_full_s"]
        rows.append(r)
        print(
            f"# partition {name}: full {r['t_full_s'] * 1e3:.1f} ms "
            f"(drift {r['full_drift']:.3f}x)  "
            + "  ".join(f"{n_}={r[f'{n_}_pct']:.1f}%" for n_, _, _ in PARTITION_KEYS)
            + f"  base={r['base_pct']:.1f}%  sum={r['sum_pct']:.1f}%  "
            f"floor={r['floor_pct']:.1f}%  cap_from_here={r['cap_from_here']:.2f}x",
            flush=True,
        )
    return dict(rows=rows)


# -----------------------------------------------------------------------------------------------
# phase: floor scaling
# -----------------------------------------------------------------------------------------------
def phase_floor(rig: Rig, args) -> Dict[str, Any]:
    """The floor configuration swept over tokens and batch, plus the head's row-dependence.

    Every block is an Identity and the routers are off, so what is left is exactly the bucket the
    partition calls 'floor'. Fitting ``a + b*tokens`` splits the fixed per-step part (amortisable:
    it is what CUDA graphs remove and what a larger step dilutes) from the per-token part
    (irreducible at fixed batch).
    """
    m = rig.model
    B0, T0 = rig.B, rig.T
    all_by_name = {a.name: a for a in build_arms("all")}
    floor_arm = all_by_name.get(args.floor_arm)

    def to_floor() -> None:
        """Every block an Identity, every router off, head gated: the un-routable remainder."""
        rig.apply(Arm("dense"))
        for k, b in m.blocks.items():
            _inner(b).feed_forward = rig.dead
            _inner(b).attention = rig.dead
        rig.ffn_holder.enabled = False
        rig.kv_holder.enabled = False
        rig.bs_holder.enabled = False
        for k, b in m.blocks.items():
            b._bskip = None
        rig.real_head._sparse_loss_positions = True

    rows = []
    grid = [(1, t) for t in [int(x) for x in args.floor_tokens.split(",")]]
    grid += [
        (bb, args.floor_batch_tokens) for bb in [int(x) for x in args.floor_batches.split(",")]
    ]
    for B, T in grid:
        try:
            rig.set_tokens(B, T)
            # The floor SHARE is what decides the ceiling, so the same (B, T) also gets a dense
            # step and an aggressive-arm step. Without them the fitted floor curve says nothing
            # about whether the ceiling moves with step size.
            dense_s = arm_s = float("nan")
            try:
                rig.apply(Arm("dense"))
                dense_s = median(timed(rig.step, 2, args.partition_iters))
                if floor_arm is not None:
                    rig.apply(floor_arm)
                    arm_s = median(timed(rig.step, 2, args.partition_iters))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
            rig.set_tokens(B, T)
            # Sub-split of the floor, measured with the same alternating-direction rounds the
            # partition uses. Stage f0 = the whole floor; each later stage removes one more piece.
            # A strictly sequential chain reads NEGATIVE here -- the floor is small enough that
            # the process's own upward drift is larger than the components being measured.
            f_stages = ["f_all", "f_no_norm", "f_no_head", "f_no_emb", "f_no_zg"]

            def set_f(k: int) -> None:
                to_floor()
                rig.set_tokens(B, T)
                if k >= 1 and rig.real_norm is not None:
                    rig.real_head.norm = rig.dead
                if k >= 2:
                    m.lm_head = rig.no_head
                if k >= 3:
                    m.embeddings = rig.const_emb()
                if k >= 4:
                    rig.use_zero_grad = False

            fs: Dict[str, List[float]] = {s: [] for s in f_stages}
            for rnd in range(args.partition_rounds):
                order = range(5) if rnd % 2 == 0 else range(4, -1, -1)
                for k in order:
                    set_f(k)
                    fs[f_stages[k]].append(median(timed(rig.step, 1, args.partition_iters)))
            ft = {s: median(v) for s, v in fs.items()}
            full = ft["f_all"]
            set_f(4)
            gpu, launches, _ = profile_step(rig.step)
            rows.append(
                dict(
                    batch=B,
                    tokens=T,
                    total_tokens=B * T,
                    dense_s=dense_s,
                    arm_s=arm_s,
                    arm=args.floor_arm,
                    floor_s=full,
                    floor_drift=(
                        max(fs["f_all"]) / min(fs["f_all"]) if min(fs["f_all"]) > 0 else None
                    ),
                    floor_pct_of_dense=100 * full / dense_s if dense_s == dense_s else None,
                    floor_pct_of_arm=100 * full / arm_s if arm_s == arm_s else None,
                    cap_from_dense=dense_s / full if full > 0 else None,
                    norm_s=ft["f_all"] - ft["f_no_norm"],
                    head_s=ft["f_no_norm"] - ft["f_no_head"],
                    embed_s=ft["f_no_head"] - ft["f_no_emb"],
                    zero_grad_s=ft["f_no_emb"] - ft["f_no_zg"],
                    base_s=ft["f_no_zg"],
                    base_gpu_busy_s=gpu,
                    base_launches=launches,
                )
            )
            print(f"# floor B={B} T={T}: {json.dumps(rows[-1])}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"# floor B={B} T={T}: OOM", flush=True)
        finally:
            m.lm_head = rig.real_head
            if rig.real_norm is not None:
                rig.real_head.norm = rig.real_norm
            m.embeddings = rig.real_emb
            rig.use_zero_grad = True
            to_floor()

    # head row-dependence: below sparse_loss_unfused_max_rows LMHead drops the fused kernel, so
    # this measures whether any N-independent head cost survives at the model level.
    head_rows = []
    to_floor()
    rig.set_tokens(1, T0)
    ns = [int(x) for x in args.head_rows.split(",") if int(x) <= T0]
    with_head: Dict[int, List[float]] = {n: [] for n in ns}
    without_head: Dict[int, List[float]] = {n: [] for n in ns}
    try:
        for rnd in range(args.partition_rounds):
            seq = ns if rnd % 2 == 0 else list(reversed(ns))
            for n_sup in seq:
                rig.set_tokens(1, T0, n_sup=n_sup)
                m.lm_head = rig.real_head
                with_head[n_sup].append(median(timed(rig.step, 1, args.partition_iters)))
                m.lm_head = rig.no_head
                without_head[n_sup].append(median(timed(rig.step, 1, args.partition_iters)))
        m.lm_head = rig.real_head
        for n_sup in ns:
            t, t_nh = median(with_head[n_sup]), median(without_head[n_sup])
            head_rows.append(dict(n_sup=n_sup, floor_s=t, head_s=t - t_nh))
            print(f"# head rows n_sup={n_sup}: head {(t - t_nh) * 1e3:.2f} ms", flush=True)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        m.lm_head = rig.real_head

    fits = {}
    for key in ("floor_s", "head_s", "norm_s", "embed_s", "zero_grad_s", "base_s"):
        pts = [(r["total_tokens"], r[key]) for r in rows if r["batch"] == 1]
        if len(pts) >= 2:
            n = len(pts)
            sx = sum(p[0] for p in pts)
            sy = sum(p[1] for p in pts)
            sxx = sum(p[0] * p[0] for p in pts)
            sxy = sum(p[0] * p[1] for p in pts)
            den = n * sxx - sx * sx
            b = (n * sxy - sx * sy) / den if den else 0.0
            a = (sy - b * sx) / n
            fits[key] = dict(fixed_s=a, per_token_s=b)
    print(f"# floor fits (a + b*tokens): {json.dumps(fits)}", flush=True)
    rig.set_tokens(B0, T0)
    rig.apply(Arm("dense"))
    return dict(rows=rows, head_rows=head_rows, fits=fits)


# -----------------------------------------------------------------------------------------------
# phase: CUDA graph
# -----------------------------------------------------------------------------------------------
def phase_graph(rig: Rig, args) -> Dict[str, Any]:
    """Try to capture a whole training step in a CUDA graph, at dense and at aggressive arms.

    A capture fails on any host synchronisation or dynamic shape. That failure IS a result: it says
    which parts of the routing machinery would have to change before the launch-bound regime can be
    attacked with graphs (``graph_probe.py`` bisects WHICH one). The measured GPU-idle fraction is
    reported alongside as the model-free upper bound on what a successful capture could buy.
    """
    all_by_name = {a.name: a for a in build_arms("all")}
    rows = []
    for name in args.graph_arms.split(","):
        arm = all_by_name.get(name)
        if arm is None:
            continue
        rig.apply(arm)
        row: Dict[str, Any] = {"arm": name}
        try:
            eager = timed(rig.step, 2, args.iters)
            row["eager_median_s"] = median(eager)
            gpu, launches, _ = profile_step(rig.step)
            row["gpu_busy_s"] = gpu
            row["launches"] = launches
            row["idle_frac"] = 1.0 - gpu / row["eager_median_s"]
            row["graph_upper_bound"] = row["eager_median_s"] / gpu if gpu > 0 else float("inf")
            # How long the CPU needs to ENQUEUE one step with no sync. NOT a valid launch-bound
            # indicator at this launch count -- with thousands of launches the CUDA queue fills and
            # the CPU blocks inside cudaLaunchKernel, so this reads ~1.0 regardless. Recorded only
            # so nobody re-derives it; ``idle_frac`` is the trustworthy column.
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                rig.step()
            row["cpu_enqueue_s"] = (time.perf_counter() - t0) / 3
            torch.cuda.synchronize()
            row["cpu_frac"] = row["cpu_enqueue_s"] / row["eager_median_s"]
        except Exception as exc:
            row["eager_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue
        try:
            torch.cuda.synchronize()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    rig.step()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            rig.model.zero_grad(set_to_none=False)
            rig.use_zero_grad = False
            with torch.cuda.graph(g):
                rig.step()
            torch.cuda.synchronize()
            ts = []
            for _ in range(args.iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            row["graph_median_s"] = median(ts)
            row["graph_speedup"] = row["eager_median_s"] / row["graph_median_s"]
            del g
        except Exception as exc:
            row["graph_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        finally:
            rig.use_zero_grad = True
            torch.cuda.empty_cache()
        rows.append(row)
        print(f"# graph {name}: {json.dumps(row)}", flush=True)
    return dict(rows=rows)


# -----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="4B")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--arms", default="all")
    ap.add_argument("--phases", default="matrix,partition,floor,graph")
    # 0 = route every layer, matching the trainer default and the trained checkpoints. The
    # previous default of 12 left a third of the FFN un-routed and understated every FFN number.
    ap.add_argument("--ffn-start-layer", type=int, default=0)
    ap.add_argument("--divisors", default="1,2.4,4.9,10,32")
    ap.add_argument("--attn-backend", default="flash_2")
    ap.add_argument("--kernels", action="store_true")
    ap.add_argument("--loss-density", type=float, default=0.0064)
    ap.add_argument("--docs", type=int, default=0)
    ap.add_argument("--ac", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--partition-iters", type=int, default=4)
    # Stages are visited once per round, forward then backward, so a monotone within-process
    # drift cancels. 2 rounds is the minimum that cancels anything; 4 is the default.
    ap.add_argument("--partition-rounds", type=int, default=4)
    ap.add_argument(
        "--partition-arms",
        default="dense,J-f0.50-k0.5000,J-f1.00-k0.2500,J-f1.00-k0.0625,"
        "staticFFN+bskipK-0.25+headgate",
    )
    ap.add_argument("--floor-arm", default="J-f1.00-k0.2500")
    ap.add_argument("--floor-tokens", default="2048,4096,8192,16384")
    ap.add_argument("--floor-batches", default="1,2,4")
    ap.add_argument("--floor-batch-tokens", type=int, default=4096)
    ap.add_argument("--head-rows", default="1,16,128,1024,8192")
    ap.add_argument(
        "--graph-arms",
        default="staticFFN+bskipK-0.25,dense,J-f1.00-k0.2500,staticFFN+bskipK-0.25+headgate",
        help="capture attempts, in order. A failed capture can poison later ones, so put the "
        "most graph-friendly arm first: no FFN router (no bincount().tolist()), fixed-capacity "
        "block skip (no .item()), no head gating (no data-dependent gather shape).",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"# GPU {torch.cuda.get_device_name(0)} torch {torch.__version__}", flush=True)
    print(
        f"# scale={args.scale} ctx={args.ctx} batch={args.batch} ac={args.ac} "
        f"docs={args.docs} rounds={args.rounds} iters={args.iters}",
        flush=True,
    )
    rig = Rig(args)
    phases = args.phases.split(",")
    result: Dict[str, Any] = dict(
        scale=args.scale,
        ctx=args.ctx,
        batch=args.batch,
        ac=args.ac,
        docs=args.docs,
        loss_density=args.loss_density,
        ffn_start_layer=args.ffn_start_layer,
        ffn_routed_layers=rig.ffn_routed_layers,
        n_layers=rig.n_layers,
        widths=rig.widths,
        costs=rig.costs,
        dense_flops_per_token=rig.dense_flops,
        buckets=dict(rig.bkt),
    )
    for name, fn in (
        ("matrix", phase_matrix),
        ("partition", phase_partition),
        ("floor", phase_floor),
        ("graph", phase_graph),
    ):
        if name not in phases:
            continue
        print(f"\n===== phase {name} =====", flush=True)
        t0 = time.perf_counter()
        try:
            result[name] = fn(rig, args)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            result[name] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"# phase {name} took {time.perf_counter() - t0:.0f}s", flush=True)

    out = args.out or os.path.join(OUT, f"joint_sweep_{args.scale}_{args.ctx}.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"# wrote {out}", flush=True)


if __name__ == "__main__":
    main()
