"""
Generate GraphViz DOT-format flowcharts of transformer model architectures
from :class:`TransformerConfig` objects.

Usage::

    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.nn.transformer.visualize import to_dot, render

    config = TransformerConfig.olmo2_7B(vocab_size=100352)

    # Get DOT text (no dependencies required).
    dot_text = to_dot(config)

    # Render to SVG (requires the ``graphviz`` Python package).
    render(config, "olmo2_7b", format="svg")
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

from ..attention import AttentionConfig
from ..feed_forward import FeedForwardConfig
from ..layer_norm import LayerNormConfig
from ..moe import MoEConfig
from .config import TransformerBlockConfig, TransformerBlockType

if TYPE_CHECKING:
    from .config import TransformerConfig

__all__ = ["to_dot", "render"]

DetailLevel = Literal["overview", "block", "full"]

# -- Colors ------------------------------------------------------------------

_COLOR_ATTN = "#dae8fc"  # light blue
_COLOR_FFN = "#d5e8d4"  # light green
_COLOR_MOE = "#fff2cc"  # light orange
_COLOR_NORM = "#f5f5f5"  # light gray
_COLOR_RESIDUAL = "#e1d5e7"  # light purple
_COLOR_EMBED = "#e8e8e8"  # silver
_COLOR_IO = "#ffffff"  # white


# -- DOT builder -------------------------------------------------------------


class _DotBuilder:
    """Incremental builder for GraphViz DOT source text."""

    def __init__(self) -> None:
        self._lines: List[str] = []
        self._counter = 0
        self._indent = 1

    # -- primitives -----------------------------------------------------------

    def _id(self, prefix: str = "n") -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def _emit(self, line: str) -> None:
        self._lines.append("    " * self._indent + line)

    # -- public API -----------------------------------------------------------

    def node(
        self,
        label: str,
        *,
        shape: str = "box",
        style: str = "filled",
        fillcolor: str = _COLOR_NORM,
        html: bool = False,
        width: Optional[str] = None,
        height: Optional[str] = None,
    ) -> str:
        """Add a node and return its ID."""
        nid = self._id()
        lbl = f"<{label}>" if html else f'"{_dot_escape(label)}"'
        extras = ""
        if width:
            extras += f", width={width}"
        if height:
            extras += f", height={height}"
        self._emit(
            f'{nid} [label={lbl}, shape={shape}, style="{style}"'
            f', fillcolor="{fillcolor}"{extras}];'
        )
        return nid

    def edge(
        self,
        src: str,
        dst: str,
        *,
        label: str = "",
        style: str = "",
        color: str = "",
        constraint: str = "",
    ) -> None:
        """Add a directed edge."""
        attrs: List[str] = []
        if label:
            attrs.append(f'label="{_dot_escape(label)}"')
        if style:
            attrs.append(f'style="{style}"')
        if color:
            attrs.append(f'color="{color}"')
        if constraint:
            attrs.append(f"constraint={constraint}")
        attr_str = f" [{', '.join(attrs)}]" if attrs else ""
        self._emit(f"{src} -> {dst}{attr_str};")

    def begin_subgraph(self, name: str, *, label: str = "", style: str = "dashed") -> None:
        self._emit(f"subgraph cluster_{name} {{")
        self._indent += 1
        self._emit(f'label="{_dot_escape(label)}";')
        self._emit(f'style="{style}";')
        self._emit('fontname="Helvetica";')
        self._emit("fontsize=10;")

    def end_subgraph(self) -> None:
        self._indent -= 1
        self._emit("}")

    def to_dot(self, *, title: str = "") -> str:
        header = [
            "digraph Transformer {",
            "    rankdir=TB;",
            '    fontname="Helvetica";',
            '    node [fontname="Helvetica", fontsize=10];',
            "    edge [fontsize=9];",
        ]
        if title:
            header.append(f'    label="{_dot_escape(title)}";')
            header.append("    labelloc=t;")
            header.append("    fontsize=14;")
        return "\n".join(header + self._lines + ["}"])


# -- Helpers ------------------------------------------------------------------


def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _norm_label(norm: Optional[LayerNormConfig]) -> str:
    if norm is None:
        return "LayerNorm"
    return str(norm.name).replace("_", " ").title().replace(" ", "")


def _attn_summary(attn: AttentionConfig, d_model: int) -> str:
    """One-line summary for attention in block-level view."""
    n_heads = attn.n_heads
    n_kv = attn.n_kv_heads or n_heads
    hd = attn.head_dim or d_model // n_heads
    parts = [f"{n_heads}h"]
    if n_kv != n_heads:
        parts.append(f"kv={n_kv}")
    parts.append(f"d={hd}")
    if attn.rope is not None:
        parts.append("RoPE")
    if attn.qk_norm is not None:
        parts.append("QK-norm")
    if attn.sliding_window is not None:
        parts.append("SWA")
    if attn.gate is not None:
        parts.append("gated")
    return ", ".join(parts)


def _ffn_summary(ff: FeedForwardConfig) -> str:
    """One-line summary for FFN."""
    act = str(ff.activation)
    return f"h={ff.hidden_size}, {act}"


def _moe_summary(moe: MoEConfig) -> str:
    """One-line summary for MoE."""
    parts = [f"{moe.num_experts} experts", f"top-{moe.router.top_k}", f"h={moe.hidden_size}"]
    if moe.shared_mlp is not None:
        parts.append("shared MLP")
    return ", ".join(parts)


# -- Layer grouping -----------------------------------------------------------


@dataclass
class _BlockGroup:
    block_config: TransformerBlockConfig
    layer_indices: List[int] = field(default_factory=list)
    block_name: Optional[str] = None
    _cfg_key: str = ""

    @property
    def label(self) -> str:
        n = len(self.layer_indices)
        name = self.block_name or str(self.block_config.name)
        if n == 1:
            return f"{name} block (layer {self.layer_indices[0]})"
        first, last = self.layer_indices[0], self.layer_indices[-1]
        if self.layer_indices == list(range(first, last + 1)):
            return f"{name} block (layers {first}\u2013{last})"
        return f"{name} block (\u00d7{n})"


def _group_layers(config: "TransformerConfig") -> List[_BlockGroup]:
    """Group consecutive layers with identical block configs."""
    resolved = config.resolved_block_configs

    # Build a mapping from config dict repr -> canonical config for equality.
    groups: List[_BlockGroup] = []

    # If we have a block pattern, use the pattern names for grouping.
    if config.block_pattern is not None and isinstance(config.block, dict):
        from itertools import cycle, islice

        full_pattern = list(islice(cycle(config.block_pattern), config.n_layers))
        for i, (name, block_cfg) in enumerate(zip(full_pattern, resolved)):
            cfg_key = repr(block_cfg.as_config_dict())
            if groups and groups[-1]._cfg_key == cfg_key:
                groups[-1].layer_indices.append(i)
            else:
                g = _BlockGroup(
                    block_config=block_cfg,
                    layer_indices=[i],
                    block_name=name,
                    _cfg_key=cfg_key,
                )
                groups.append(g)
    else:
        for i, block_cfg in enumerate(resolved):
            cfg_key = repr(block_cfg.as_config_dict())
            if groups and groups[-1]._cfg_key == cfg_key:
                groups[-1].layer_indices.append(i)
            else:
                g = _BlockGroup(block_config=block_cfg, layer_indices=[i], _cfg_key=cfg_key)
                groups.append(g)

    return groups


# -- Attention detail ---------------------------------------------------------


def _add_attention_detail(
    b: _DotBuilder,
    attn: AttentionConfig,
    d_model: int,
    input_id: str,
    output_id: str,
) -> None:
    """Expand attention into Q/K/V projections, optional components, and SDPA."""
    n_heads = attn.n_heads
    n_kv = attn.n_kv_heads or n_heads
    hd = attn.head_dim or d_model // n_heads

    # Q, K, V projections (parallel fan-out).
    q_node = b.node(f"w_q\\n{d_model}\u2192{n_heads}\u00d7{hd}", fillcolor=_COLOR_ATTN)
    k_node = b.node(f"w_k\\n{d_model}\u2192{n_kv}\u00d7{hd}", fillcolor=_COLOR_ATTN)
    v_node = b.node(f"w_v\\n{d_model}\u2192{n_kv}\u00d7{hd}", fillcolor=_COLOR_ATTN)
    b.edge(input_id, q_node)
    b.edge(input_id, k_node)
    b.edge(input_id, v_node)

    q_out, k_out, v_out = q_node, k_node, v_node

    # Optional clip_qkv.
    if attn.clip_qkv is not None:
        clip = b.node(f"clip QKV\\n|x| \u2264 {attn.clip_qkv}", fillcolor=_COLOR_ATTN)
        b.edge(q_out, clip)
        b.edge(k_out, clip)
        b.edge(v_out, clip)
        q_out = k_out = v_out = clip

    # Optional QK norm.
    if attn.qk_norm is not None:
        norm_lbl = _norm_label(attn.qk_norm)
        if attn.use_head_qk_norm:
            norm_lbl += " (per-head)"
        qn = b.node(f"Q {norm_lbl}", fillcolor=_COLOR_NORM)
        kn = b.node(f"K {norm_lbl}", fillcolor=_COLOR_NORM)
        b.edge(q_out, qn)
        b.edge(k_out, kn)
        q_out = qn
        k_out = kn

    # Optional RoPE.
    if attn.rope is not None:
        rope_lbl = f"RoPE (\u03b8={attn.rope.theta:,})"
        rope_node = b.node(rope_lbl, fillcolor=_COLOR_ATTN)
        b.edge(q_out, rope_node)
        b.edge(k_out, rope_node)
        q_out = rope_node
        k_out = rope_node

    # SDPA.
    sdpa_lbl = "SDPA"
    if attn.sliding_window is not None:
        sdpa_lbl += " (SWA)"
    sdpa = b.node(sdpa_lbl, fillcolor=_COLOR_ATTN)
    b.edge(q_out, sdpa)
    if k_out != q_out:
        b.edge(k_out, sdpa)
    b.edge(v_out, sdpa)

    attn_out = sdpa

    # Optional gate.
    if attn.gate is not None:
        gate = b.node(f"Gate ({attn.gate.granularity})", fillcolor=_COLOR_ATTN)
        b.edge(attn_out, gate)
        attn_out = gate

    # Output projection.
    w_out = b.node(f"w_out\\n{n_heads}\u00d7{hd}\u2192{d_model}", fillcolor=_COLOR_ATTN)
    b.edge(attn_out, w_out)
    b.edge(w_out, output_id)


# -- FFN detail ---------------------------------------------------------------


def _add_ffn_detail(
    b: _DotBuilder,
    ff: FeedForwardConfig,
    d_model: int,
    input_id: str,
    output_id: str,
) -> None:
    """Expand FFN into w1/w3 gated structure."""
    act = str(ff.activation)
    w1 = b.node(f"w1\\n{d_model}\u2192{ff.hidden_size}", fillcolor=_COLOR_FFN)
    act_node = b.node(f"{act}", fillcolor=_COLOR_FFN)
    w3 = b.node(f"w3\\n{d_model}\u2192{ff.hidden_size}", fillcolor=_COLOR_FFN)
    mul = b.node("\u00d7", shape="circle", fillcolor=_COLOR_FFN, width="0.4", height="0.4")
    w2 = b.node(f"w2\\n{ff.hidden_size}\u2192{d_model}", fillcolor=_COLOR_FFN)

    b.edge(input_id, w1)
    b.edge(input_id, w3)
    b.edge(w1, act_node)
    b.edge(act_node, mul)
    b.edge(w3, mul)
    b.edge(mul, w2)
    b.edge(w2, output_id)


# -- MoE detail ---------------------------------------------------------------


def _add_moe_detail(
    b: _DotBuilder,
    moe: MoEConfig,
    d_model: int,
    input_id: str,
    output_id: str,
) -> None:
    """Expand MoE into router + experts + optional shared MLP."""
    router = b.node(
        f"Router\\ntop-{moe.router.top_k} of {moe.num_experts}",
        shape="diamond",
        fillcolor=_COLOR_MOE,
    )
    b.edge(input_id, router)

    experts = b.node(
        f"Experts ({moe.num_experts}\u00d7)\\n" f"h={moe.hidden_size}",
        fillcolor=_COLOR_MOE,
    )
    b.edge(router, experts, label="dispatch")

    if moe.shared_mlp is not None:
        shared = b.node(
            f"Shared MLP\\nh={moe.shared_mlp.hidden_size}",
            fillcolor=_COLOR_MOE,
        )
        b.edge(input_id, shared)
        combine = b.node("+", shape="circle", fillcolor=_COLOR_MOE, width="0.4", height="0.4")
        b.edge(experts, combine, label="combine")
        b.edge(shared, combine)
        b.edge(combine, output_id)
    else:
        b.edge(experts, output_id, label="combine")


# -- Block builders -----------------------------------------------------------


def _add_block_overview(
    b: _DotBuilder,
    block: TransformerBlockConfig,
    d_model: int,
    input_id: str,
    output_id: str,
) -> None:
    """Overview: single node for the whole block."""
    parts = [str(block.name)]
    if isinstance(block.sequence_mixer, AttentionConfig):
        parts.append(_attn_summary(block.sequence_mixer, d_model))
    if block.feed_forward is not None:
        parts.append(_ffn_summary(block.feed_forward))
    if block.feed_forward_moe is not None:
        parts.append(_moe_summary(block.feed_forward_moe))
    label = "\\n".join(parts)
    n = b.node(label, fillcolor=_COLOR_ATTN)
    b.edge(input_id, n)
    b.edge(n, output_id)


def _attn_node_or_detail(
    b: _DotBuilder,
    attn_cfg: AttentionConfig,
    d_model: int,
    input_id: str,
    output_id: str,
    detail: DetailLevel,
) -> None:
    """Either add a single Attention node or expand internals."""
    if detail == "full" and isinstance(attn_cfg, AttentionConfig):
        _add_attention_detail(b, attn_cfg, d_model, input_id, output_id)
    else:
        summary = _attn_summary(attn_cfg, d_model) if isinstance(attn_cfg, AttentionConfig) else ""
        label = f"Attention\\n{summary}" if summary else "Sequence Mixer"
        n = b.node(label, fillcolor=_COLOR_ATTN)
        b.edge(input_id, n)
        b.edge(n, output_id)


def _ffn_node_or_detail(
    b: _DotBuilder,
    ff_cfg: FeedForwardConfig,
    d_model: int,
    input_id: str,
    output_id: str,
    detail: DetailLevel,
) -> None:
    """Either add a single FFN node or expand internals."""
    if detail == "full":
        _add_ffn_detail(b, ff_cfg, d_model, input_id, output_id)
    else:
        label = f"FeedForward\\n{_ffn_summary(ff_cfg)}"
        n = b.node(label, fillcolor=_COLOR_FFN)
        b.edge(input_id, n)
        b.edge(n, output_id)


def _moe_node_or_detail(
    b: _DotBuilder,
    moe_cfg: MoEConfig,
    d_model: int,
    input_id: str,
    output_id: str,
    detail: DetailLevel,
) -> None:
    """Either add a single MoE node or expand internals."""
    if detail == "full":
        _add_moe_detail(b, moe_cfg, d_model, input_id, output_id)
    else:
        label = f"MoE\\n{_moe_summary(moe_cfg)}"
        n = b.node(label, fillcolor=_COLOR_MOE)
        b.edge(input_id, n)
        b.edge(n, output_id)


# -- Per-block-type builders --------------------------------------------------


# forward: h = residual(x, attn(norm(x))) ; residual(h, ffn(norm(h)))
def _add_block_default(
    b: _DotBuilder,
    block: TransformerBlockConfig,
    d_model: int,
    input_id: str,
    output_id: str,
    detail: DetailLevel,
) -> None:
    norm_lbl = _norm_label(block.layer_norm)

    # Attention path.
    attn_norm = b.node(f"{norm_lbl}", fillcolor=_COLOR_NORM)
    attn_out = b.node("", shape="point", width="0.01", height="0.01")
    b.edge(input_id, attn_norm)
    _attn_node_or_detail(b, block.sequence_mixer, d_model, attn_norm, attn_out, detail)
    res1 = b.node("+", shape="circle", fillcolor=_COLOR_RESIDUAL, width="0.4", height="0.4")
    b.edge(attn_out, res1)
    b.edge(input_id, res1)

    # FFN path.
    assert block.feed_forward is not None
    ffn_norm = b.node(f"{norm_lbl}", fillcolor=_COLOR_NORM)
    ffn_out = b.node("", shape="point", width="0.01", height="0.01")
    b.edge(res1, ffn_norm)
    _ffn_node_or_detail(b, block.feed_forward, d_model, ffn_norm, ffn_out, detail)
    res2 = b.node("+", shape="circle", fillcolor=_COLOR_RESIDUAL, width="0.4", height="0.4")
    b.edge(ffn_out, res2)
    b.edge(res1, res2)
    b.edge(res2, output_id)


# forward: same as default but norm output scaled by 1/sqrt(layer_id)
def _add_block_default_scaled(
    b: _DotBuilder,
    block: TransformerBlockConfig,
    d_model: int,
    input_id: str,
    output_id: str,
    detail: DetailLevel,
) -> None:
    norm_lbl = _norm_label(block.layer_norm)

    attn_norm = b.node(f"{norm_lbl}\\n\u00d7 1/\u221a(layer_id)", fillcolor=_COLOR_NORM)
    attn_out = b.node("", shape="point", width="0.01", height="0.01")
    b.edge(input_id, attn_norm)
    _attn_node_or_detail(b, block.sequence_mixer, d_model, attn_norm, attn_out, detail)
    res1 = b.node("+", shape="circle", fillcolor=_COLOR_RESIDUAL, width="0.4", height="0.4")
    b.edge(attn_out, res1)
    b.edge(input_id, res1)

    assert block.feed_forward is not None
    ffn_norm = b.node(f"{norm_lbl}\\n\u00d7 1/\u221a(layer_id)", fillcolor=_COLOR_NORM)
    ffn_out = b.node("", shape="point", width="0.01", height="0.01")
    b.edge(res1, ffn_norm)
    _ffn_node_or_detail(b, block.feed_forward, d_model, ffn_norm, ffn_out, detail)
    res2 = b.node("+", shape="circle", fillcolor=_COLOR_RESIDUAL, width="0.4", height="0.4")
    b.edge(ffn_out, res2)
    b.edge(res1, res2)
    b.edge(res2, output_id)


# forward: h = residual(x, norm(attn(x))) ; residual(h, norm(ffn(h)))
def _add_block_reordered_norm(
    b: _DotBuilder,
    block: TransformerBlockConfig,
    d_model: int,
    input_id: str,
    output_id: str,
    detail: DetailLevel,
) -> None:
    norm_lbl = _norm_label(block.layer_norm)

    attn_out = b.node("", shape="point", width="0.01", height="0.01")
    _attn_node_or_detail(b, block.sequence_mixer, d_model, input_id, attn_out, detail)
    attn_norm = b.node(f"{norm_lbl}", fillcolor=_COLOR_NORM)
    b.edge(attn_out, attn_norm)
    res1 = b.node("+", shape="circle", fillcolor=_COLOR_RESIDUAL, width="0.4", height="0.4")
    b.edge(attn_norm, res1)
    b.edge(input_id, res1)

    assert block.feed_forward is not None
    ffn_out = b.node("", shape="point", width="0.01", height="0.01")
    _ffn_node_or_detail(b, block.feed_forward, d_model, res1, ffn_out, detail)
    ffn_norm = b.node(f"{norm_lbl}", fillcolor=_COLOR_NORM)
    b.edge(ffn_out, ffn_norm)
    res2 = b.node("+", shape="circle", fillcolor=_COLOR_RESIDUAL, width="0.4", height="0.4")
    b.edge(ffn_norm, res2)
    b.edge(res1, res2)
    b.edge(res2, output_id)


# forward: h = residual(x, post_norm(attn(pre_norm(x)))) ; ...
def _add_block_peri_norm(
    b: _DotBuilder,
    block: TransformerBlockConfig,
    d_model: int,
    input_id: str,
    output_id: str,
    detail: DetailLevel,
) -> None:
    norm_lbl = _norm_label(block.layer_norm)

    pre_attn_norm = b.node(f"Pre-{norm_lbl}", fillcolor=_COLOR_NORM)
    attn_out = b.node("", shape="point", width="0.01", height="0.01")
    b.edge(input_id, pre_attn_norm)
    _attn_node_or_detail(b, block.sequence_mixer, d_model, pre_attn_norm, attn_out, detail)
    post_attn_norm = b.node(f"Post-{norm_lbl}", fillcolor=_COLOR_NORM)
    b.edge(attn_out, post_attn_norm)
    res1 = b.node("+", shape="circle", fillcolor=_COLOR_RESIDUAL, width="0.4", height="0.4")
    b.edge(post_attn_norm, res1)
    b.edge(input_id, res1)

    assert block.feed_forward is not None
    pre_ffn_norm = b.node(f"Pre-{norm_lbl}", fillcolor=_COLOR_NORM)
    ffn_out = b.node("", shape="point", width="0.01", height="0.01")
    b.edge(res1, pre_ffn_norm)
    _ffn_node_or_detail(b, block.feed_forward, d_model, pre_ffn_norm, ffn_out, detail)
    post_ffn_norm = b.node(f"Post-{norm_lbl}", fillcolor=_COLOR_NORM)
    b.edge(ffn_out, post_ffn_norm)
    res2 = b.node("+", shape="circle", fillcolor=_COLOR_RESIDUAL, width="0.4", height="0.4")
    b.edge(post_ffn_norm, res2)
    b.edge(res1, res2)
    b.edge(res2, output_id)


# forward: h = l2_norm(lerp(x, l2_norm(attn(x)), alpha)) ; ...
def _add_block_normalized(
    b: _DotBuilder,
    block: TransformerBlockConfig,
    d_model: int,
    input_id: str,
    output_id: str,
    detail: DetailLevel,
) -> None:
    attn_out = b.node("", shape="point", width="0.01", height="0.01")
    _attn_node_or_detail(b, block.sequence_mixer, d_model, input_id, attn_out, detail)
    l2_1 = b.node("L2 Normalize", fillcolor=_COLOR_NORM)
    b.edge(attn_out, l2_1)
    lerp1 = b.node("Lerp(\u03b1_attn)", fillcolor=_COLOR_RESIDUAL)
    b.edge(l2_1, lerp1)
    b.edge(input_id, lerp1)
    l2_2 = b.node("L2 Normalize", fillcolor=_COLOR_NORM)
    b.edge(lerp1, l2_2)

    assert block.feed_forward is not None
    ffn_out = b.node("", shape="point", width="0.01", height="0.01")
    _ffn_node_or_detail(b, block.feed_forward, d_model, l2_2, ffn_out, detail)
    l2_3 = b.node("L2 Normalize", fillcolor=_COLOR_NORM)
    b.edge(ffn_out, l2_3)
    lerp2 = b.node("Lerp(\u03b1_mlp)", fillcolor=_COLOR_RESIDUAL)
    b.edge(l2_3, lerp2)
    b.edge(l2_2, lerp2)
    l2_4 = b.node("L2 Normalize", fillcolor=_COLOR_NORM)
    b.edge(lerp2, l2_4)
    b.edge(l2_4, output_id)


# forward: h = x + attn(norm(x)) ; h + moe(norm(h))
def _add_block_moe(
    b: _DotBuilder,
    block: TransformerBlockConfig,
    d_model: int,
    input_id: str,
    output_id: str,
    detail: DetailLevel,
) -> None:
    norm_lbl = _norm_label(block.layer_norm)

    attn_norm = b.node(f"{norm_lbl}", fillcolor=_COLOR_NORM)
    attn_out = b.node("", shape="point", width="0.01", height="0.01")
    b.edge(input_id, attn_norm)
    _attn_node_or_detail(b, block.sequence_mixer, d_model, attn_norm, attn_out, detail)
    res1 = b.node("+", shape="circle", fillcolor=_COLOR_RESIDUAL, width="0.4", height="0.4")
    b.edge(attn_out, res1)
    b.edge(input_id, res1)

    assert block.feed_forward_moe is not None
    ffn_norm = b.node(f"{norm_lbl}", fillcolor=_COLOR_NORM)
    moe_out = b.node("", shape="point", width="0.01", height="0.01")
    b.edge(res1, ffn_norm)
    _moe_node_or_detail(b, block.feed_forward_moe, d_model, ffn_norm, moe_out, detail)
    res2 = b.node("+", shape="circle", fillcolor=_COLOR_RESIDUAL, width="0.4", height="0.4")
    b.edge(moe_out, res2)
    b.edge(res1, res2)
    b.edge(res2, output_id)


# forward: h = x + norm(attn(x)) ; h + norm(moe(h))
def _add_block_moe_reordered_norm(
    b: _DotBuilder,
    block: TransformerBlockConfig,
    d_model: int,
    input_id: str,
    output_id: str,
    detail: DetailLevel,
) -> None:
    norm_lbl = _norm_label(block.layer_norm)

    attn_out = b.node("", shape="point", width="0.01", height="0.01")
    _attn_node_or_detail(b, block.sequence_mixer, d_model, input_id, attn_out, detail)
    attn_norm = b.node(f"{norm_lbl}", fillcolor=_COLOR_NORM)
    b.edge(attn_out, attn_norm)
    res1 = b.node("+", shape="circle", fillcolor=_COLOR_RESIDUAL, width="0.4", height="0.4")
    b.edge(attn_norm, res1)
    b.edge(input_id, res1)

    assert block.feed_forward_moe is not None
    moe_out = b.node("", shape="point", width="0.01", height="0.01")
    _moe_node_or_detail(b, block.feed_forward_moe, d_model, res1, moe_out, detail)
    ffn_norm = b.node(f"{norm_lbl}", fillcolor=_COLOR_NORM)
    b.edge(moe_out, ffn_norm)
    res2 = b.node("+", shape="circle", fillcolor=_COLOR_RESIDUAL, width="0.4", height="0.4")
    b.edge(ffn_norm, res2)
    b.edge(res1, res2)
    b.edge(res2, output_id)


# forward (non-combined): sparse_forward(x) + dense_forward(x)
# dense: h = x + attn(norm(x)) ; h + ffn(norm(h))
# sparse: moe(moe_norm(x))
def _add_block_moe_hybrid(
    b: _DotBuilder,
    block: TransformerBlockConfig,
    d_model: int,
    input_id: str,
    output_id: str,
    detail: DetailLevel,
) -> None:
    norm_lbl = _norm_label(block.layer_norm)

    # -- Dense path --
    b.begin_subgraph(b._id("dense"), label="Dense path", style="dashed")
    attn_norm = b.node(f"{norm_lbl}", fillcolor=_COLOR_NORM)
    attn_out = b.node("", shape="point", width="0.01", height="0.01")
    b.edge(input_id, attn_norm)
    _attn_node_or_detail(b, block.sequence_mixer, d_model, attn_norm, attn_out, detail)
    res1 = b.node("+", shape="circle", fillcolor=_COLOR_RESIDUAL, width="0.4", height="0.4")
    b.edge(attn_out, res1)
    b.edge(input_id, res1)

    assert block.feed_forward is not None
    ffn_norm = b.node(f"{norm_lbl}", fillcolor=_COLOR_NORM)
    ffn_out = b.node("", shape="point", width="0.01", height="0.01")
    b.edge(res1, ffn_norm)
    _ffn_node_or_detail(b, block.feed_forward, d_model, ffn_norm, ffn_out, detail)
    res2 = b.node("+", shape="circle", fillcolor=_COLOR_RESIDUAL, width="0.4", height="0.4")
    b.edge(ffn_out, res2)
    b.edge(res1, res2)
    dense_out = res2
    b.end_subgraph()

    # -- Sparse path --
    b.begin_subgraph(b._id("sparse"), label="Sparse path", style="dashed")
    assert block.feed_forward_moe is not None
    moe_norm = b.node(f"MoE {norm_lbl}", fillcolor=_COLOR_NORM)
    moe_out = b.node("", shape="point", width="0.01", height="0.01")
    b.edge(input_id, moe_norm)
    _moe_node_or_detail(b, block.feed_forward_moe, d_model, moe_norm, moe_out, detail)
    sparse_out = moe_out
    b.end_subgraph()

    # Combine.
    combine = b.node("+", shape="circle", fillcolor=_COLOR_RESIDUAL, width="0.4", height="0.4")
    b.edge(dense_out, combine)
    b.edge(sparse_out, combine)
    b.edge(combine, output_id)


# forward (non-combined): sparse_forward(x) + dense_forward(x)
# dense: h = x + norm(attn(x)) ; h + norm(ffn(h))
# sparse: moe_norm(moe(x))
def _add_block_moe_hybrid_reordered_norm(
    b: _DotBuilder,
    block: TransformerBlockConfig,
    d_model: int,
    input_id: str,
    output_id: str,
    detail: DetailLevel,
) -> None:
    norm_lbl = _norm_label(block.layer_norm)

    # -- Dense path --
    b.begin_subgraph(b._id("dense"), label="Dense path", style="dashed")
    attn_out = b.node("", shape="point", width="0.01", height="0.01")
    _attn_node_or_detail(b, block.sequence_mixer, d_model, input_id, attn_out, detail)
    attn_norm = b.node(f"{norm_lbl}", fillcolor=_COLOR_NORM)
    b.edge(attn_out, attn_norm)
    res1 = b.node("+", shape="circle", fillcolor=_COLOR_RESIDUAL, width="0.4", height="0.4")
    b.edge(attn_norm, res1)
    b.edge(input_id, res1)

    assert block.feed_forward is not None
    ffn_out = b.node("", shape="point", width="0.01", height="0.01")
    _ffn_node_or_detail(b, block.feed_forward, d_model, res1, ffn_out, detail)
    ffn_norm = b.node(f"{norm_lbl}", fillcolor=_COLOR_NORM)
    b.edge(ffn_out, ffn_norm)
    res2 = b.node("+", shape="circle", fillcolor=_COLOR_RESIDUAL, width="0.4", height="0.4")
    b.edge(ffn_norm, res2)
    b.edge(res1, res2)
    dense_out = res2
    b.end_subgraph()

    # -- Sparse path --
    b.begin_subgraph(b._id("sparse"), label="Sparse path", style="dashed")
    assert block.feed_forward_moe is not None
    moe_out = b.node("", shape="point", width="0.01", height="0.01")
    _moe_node_or_detail(b, block.feed_forward_moe, d_model, input_id, moe_out, detail)
    moe_norm = b.node(f"MoE {norm_lbl}", fillcolor=_COLOR_NORM)
    b.edge(moe_out, moe_norm)
    sparse_out = moe_norm
    b.end_subgraph()

    # Combine.
    combine = b.node("+", shape="circle", fillcolor=_COLOR_RESIDUAL, width="0.4", height="0.4")
    b.edge(dense_out, combine)
    b.edge(sparse_out, combine)
    b.edge(combine, output_id)


# Dispatch table.
_BLOCK_BUILDERS: Dict[TransformerBlockType, type] = {  # values are callables, not types
    TransformerBlockType.default: _add_block_default,  # type: ignore[dict-item]
    TransformerBlockType.default_scaled: _add_block_default_scaled,  # type: ignore[dict-item]
    TransformerBlockType.reordered_norm: _add_block_reordered_norm,  # type: ignore[dict-item]
    TransformerBlockType.peri_norm: _add_block_peri_norm,  # type: ignore[dict-item]
    TransformerBlockType.normalized: _add_block_normalized,  # type: ignore[dict-item]
    TransformerBlockType.moe: _add_block_moe,  # type: ignore[dict-item]
    TransformerBlockType.moe_reordered_norm: _add_block_moe_reordered_norm,  # type: ignore[dict-item]
    TransformerBlockType.moe_hybrid: _add_block_moe_hybrid,  # type: ignore[dict-item]
    TransformerBlockType.moe_hybrid_reordered_norm: _add_block_moe_hybrid_reordered_norm,  # type: ignore[dict-item]
}


# -- Top-level orchestration --------------------------------------------------


def to_dot(
    config: "TransformerConfig",
    *,
    detail: DetailLevel = "full",
    title: Optional[str] = None,
) -> str:
    """
    Generate a Graphviz DOT-format string representing the architecture of the
    given :class:`TransformerConfig`.

    :param config: The transformer configuration to visualize.
    :param detail: Level of detail to include:

        - ``"overview"``: High-level view with major components as single nodes.
        - ``"block"``: Expand each distinct block type to show internal data flow.
        - ``"full"``: (Default) Also expand attention, FFN, and MoE internals.

    :param title: Optional title for the graph. Defaults to a summary of the config.

    :returns: A string in Graphviz DOT format.
    """
    b = _DotBuilder()

    if title is None:
        title = (
            f"d_model={config.d_model}, n_layers={config.n_layers}, "
            f"vocab_size={config.vocab_size:,}\n"
            f"{config.num_params:,} params"
        )

    # Input.
    inp = b.node("Input\\n(B, T)", shape="ellipse", fillcolor=_COLOR_IO, style="filled")

    # Embedding.
    embed = b.node(
        f"Embedding\\n{config.vocab_size:,} \u00d7 {config.d_model}",
        fillcolor=_COLOR_EMBED,
    )
    b.edge(inp, embed)
    prev = embed

    # Optional embed_scale.
    if config.embed_scale is not None:
        scale = b.node(f"\u00d7 {config.embed_scale}", fillcolor=_COLOR_NORM)
        b.edge(prev, scale)
        prev = scale

    # Optional embedding norm.
    if config.embedding_norm is not None:
        norm_lbl = _norm_label(config.embedding_norm)
        enorm = b.node(f"Embedding {norm_lbl}", fillcolor=_COLOR_NORM)
        b.edge(prev, enorm)
        prev = enorm

    # Blocks.
    groups = _group_layers(config)
    for gi, group in enumerate(groups):
        block_cfg = group.block_config
        sg_name = f"block_{gi}"

        if detail == "overview":
            block_in = b.node("", shape="point", width="0.01", height="0.01")
            block_out = b.node("", shape="point", width="0.01", height="0.01")
            b.edge(prev, block_in)
            b.begin_subgraph(sg_name, label=group.label)
            _add_block_overview(b, block_cfg, config.d_model, block_in, block_out)
            b.end_subgraph()
            prev = block_out
        else:
            block_in = b.node("", shape="point", width="0.01", height="0.01")
            block_out = b.node("", shape="point", width="0.01", height="0.01")
            b.edge(prev, block_in)
            b.begin_subgraph(sg_name, label=group.label)
            builder_fn = _BLOCK_BUILDERS.get(block_cfg.name)
            if builder_fn is not None:
                builder_fn(b, block_cfg, config.d_model, block_in, block_out, detail)
            else:
                # Fallback for unknown block types.
                _add_block_overview(b, block_cfg, config.d_model, block_in, block_out)
            b.end_subgraph()
            prev = block_out

    # LM Head.
    if detail == "overview":
        lm = b.node("LM Head", fillcolor=_COLOR_EMBED)
        b.edge(prev, lm)
        prev = lm
    else:
        b.begin_subgraph("lm_head", label="LM Head")
        if config.lm_head.layer_norm is not None:
            lm_norm = b.node(
                f"{_norm_label(config.lm_head.layer_norm)}",
                fillcolor=_COLOR_NORM,
            )
            b.edge(prev, lm_norm)
            prev = lm_norm
        lm_proj = b.node(
            f"Linear\\n{config.d_model} \u2192 {config.vocab_size:,}",
            fillcolor=_COLOR_EMBED,
        )
        b.edge(prev, lm_proj)
        prev = lm_proj
        b.end_subgraph()

    # Output.
    out = b.node("Output\\nLogits (B, T, V)", shape="ellipse", fillcolor=_COLOR_IO, style="filled")
    b.edge(prev, out)

    return b.to_dot(title=title)


def render(
    config: "TransformerConfig",
    output_path: str,
    *,
    detail: DetailLevel = "full",
    title: Optional[str] = None,
    format: str = "svg",
) -> str:
    """
    Render the architecture flowchart to a file.

    Requires the ``graphviz`` Python package to be installed.

    :param config: The transformer configuration to visualize.
    :param output_path: The output file path (without extension).
    :param detail: Level of detail (see :func:`to_dot`).
    :param title: Optional title for the graph.
    :param format: Output format (``"svg"``, ``"pdf"``, ``"png"``).

    :returns: The path to the rendered file.

    :raises ImportError: If the ``graphviz`` Python package is not installed.
    """
    try:
        import graphviz as gv
    except ImportError:
        raise ImportError(
            "The 'graphviz' Python package is required for rendering. "
            "Install it with: pip install graphviz"
        )

    dot_source = to_dot(config, detail=detail, title=title)
    src = gv.Source(dot_source)
    rendered_path = src.render(outfile=f"{output_path}.{format}", format=format, cleanup=True)
    return rendered_path
