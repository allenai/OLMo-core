"""Vendor the `mainline_ladder` transformers plugin with Scalable-Softmax support added.

The stock plugin has NO SSMax: `ssmax_scale` in a checkpoint loads as UNEXPECTED and is silently
ignored, so any harness driving it evaluates an SSMax-trained model with SSMax off -- it loads
clean and scores, which is the dangerous kind of wrong.

This copies the upstream plugin and applies three changes:

1. ``MainlineLadderConfig.scalable_softmax`` (default False, so non-SSMax models are untouched).
2. ``MainlineLadderAttention.ssmax_scale``, a per-head parameter, registered only when the flag is
   set -- which is what makes the checkpoint's weights actually load.
3. ``_apply_scalable_softmax`` on the query after q-norm, matching olmo-core's
   ``Attention._apply_scalable_softmax`` (branch prasann/ctc-sft-hybridish).

Two deliberate differences from the reference, both required here:

* The reference takes ``q`` as ``(B, T, H, D)``; this plugin has ``(B, H, T, D)``.
* The reference RAISES on KV caching. Grading decodes with a cache, so positions come from
  ``cache_position`` (absolute, cache-aware) instead of a local ``arange``. A local arange gives
  every decode step position 1 -- log(2) where the truth is log(prompt_len+step), scaling q by
  ~1/13 on SSMax layers. That is the bug this port exists to remove.

Position 0 yields log(1) = 0 exactly as upstream does; it is not clamped.

    python debug/hybridish_sft/apply_ssmax_to_plugin.py --src <plugin> --out <dir>
"""

import argparse
import os
import shutil

DOC_ANCHOR = """    use_head_qk_norm (`bool`, *optional*, defaults to `True`):
        Whether to use per-head QK normalization (norm of size head_dim per head) instead of
        full-dim norm (norm of size num_heads * head_dim).
"""
DOC_NEW = """    use_head_qk_norm (`bool`, *optional*, defaults to `True`):
        Whether to use per-head QK normalization (norm of size head_dim per head) instead of
        full-dim norm (norm of size num_heads * head_dim).
    scalable_softmax (`bool`, *optional*, defaults to `False`):
        Whether the full-attention layers use Scalable-Softmax: a learned per-head scale on the
        query, multiplied by the log of the number of tokens visible to that query. The scale is
        trained, so it cannot be recovered post-hoc -- a checkpoint carrying `ssmax_scale` must
        set this or those weights load as unexpected and are silently ignored.
"""

CFG_ANCHOR = """    use_attention_gate: bool = True
    use_head_qk_norm: bool = True
    head_dim: int = 128
"""
CFG_NEW = """    use_attention_gate: bool = True
    use_head_qk_norm: bool = True
    head_dim: int = 128
    # Scalable-Softmax: a learned per-head scale on q, multiplied by log(visible length). Trained,
    # so it cannot be recovered post-hoc -- a checkpoint carrying ssmax_scale MUST set this.
    scalable_softmax: bool = False
"""

INIT_ANCHOR = """        # Elementwise attention gate
        if config.use_attention_gate:"""
INIT_NEW = """        # Scalable-Softmax per-head scale. Registered only when the config asks for it, so models
        # without it keep their exact parameter set (and a stray ssmax_scale still reports missing).
        self.ssmax_scale = None
        if getattr(config, "scalable_softmax", False):
            self.ssmax_scale = nn.Parameter(torch.ones(config.num_attention_heads))

        # Elementwise attention gate
        if config.use_attention_gate:"""

METHOD = '''
    def _apply_scalable_softmax(self, q, cache_position, past_key_values):
        """Scale q by log(visible length) * per-head ssmax_scale.

        :param q: Query states, shape ``(batch, heads, seq, head_dim)``.
        :param cache_position: Absolute position of each query token, or ``None``.
        :param past_key_values: Cache, used only to recover the offset when ``cache_position`` is
            absent.

        :returns: The scaled query states.
        """
        if self.ssmax_scale is None:
            return q
        seq_len = q.shape[2]
        if cache_position is not None:
            positions = cache_position.to(device=q.device)
        else:
            # No cache_position: fall back to the cache's own length. Never a bare arange(seq_len)
            # -- during decoding that reports position 0 for every step.
            past = 0
            if past_key_values is not None:
                past = past_key_values.get_seq_length(self.layer_idx)
            positions = torch.arange(past, past + seq_len, device=q.device)
        # visible length is 1-based: position 0 sees 1 token, and log(1) = 0 as upstream.
        visible = (positions + 1).to(torch.float32)
        scale = visible.log().to(q.dtype).view(1, 1, seq_len, 1)
        scale = scale * self.ssmax_scale.to(device=q.device, dtype=q.dtype).view(1, -1, 1, 1)
        return q * scale
'''

FWD_ANCHOR = """        # NoPE: never apply rotary embeddings (position_embeddings is always None)

        if past_key_values is not None:"""
FWD_NEW = """        # NoPE: never apply rotary embeddings (position_embeddings is always None)

        # AFTER q_norm, deliberately: RMS q-norm renormalises each head, so a scale applied before
        # it divides straight back out and measures as a no-op.
        if self.ssmax_scale is not None:
            query_states = self._apply_scalable_softmax(
                query_states, kwargs.get("cache_position"), past_key_values
            )

        if past_key_values is not None:"""


def patch(src: str, out: str) -> None:
    """
    Copy the plugin from ``src`` to ``out`` and add SSMax support.

    :param src: Upstream plugin package root (the dir holding ``transformers_plugin/``).
    :param out: Destination; replaced if it exists.

    :raises SystemExit: If any anchor is missing, which means upstream moved and the patch is stale.
    """
    if os.path.exists(out):
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("__pycache__"))
    pkg = os.path.join(out, "transformers_plugin")

    cfg_p = os.path.join(pkg, "configuration_mainline_ladder.py")
    cfg = open(cfg_p).read()
    for name, anchor in (("config field", CFG_ANCHOR), ("config docstring", DOC_ANCHOR)):
        if anchor not in cfg:
            raise SystemExit(f"{name} anchor not found in {cfg_p}; upstream changed")
    cfg = cfg.replace(DOC_ANCHOR, DOC_NEW, 1).replace(CFG_ANCHOR, CFG_NEW, 1)
    open(cfg_p, "w").write(cfg)

    mod_p = os.path.join(pkg, "modeling_mainline_ladder.py")
    mod = open(mod_p).read()
    for name, anchor in (("attention __init__", INIT_ANCHOR), ("attention forward", FWD_ANCHOR)):
        if anchor not in mod:
            raise SystemExit(f"{name} anchor not found in {mod_p}; upstream changed")
    mod = mod.replace(INIT_ANCHOR, INIT_NEW, 1)
    mod = mod.replace(FWD_ANCHOR, FWD_NEW, 1)
    # method goes just before the attention class's forward
    fwd_def = "    def forward(\n        self,\n        hidden_states: torch.Tensor,\n        position_embeddings:"
    if fwd_def not in mod:
        raise SystemExit("attention forward signature not found; upstream changed")
    mod = mod.replace(fwd_def, METHOD + "\n" + fwd_def, 1)
    open(mod_p, "w").write(mod)

    import ast
    for f in (cfg_p, mod_p):
        ast.parse(open(f).read())
    print(f"[patched] {out}")
    print("  + MainlineLadderConfig.scalable_softmax")
    print("  + MainlineLadderAttention.ssmax_scale (per-head parameter)")
    print("  + _apply_scalable_softmax, cache-position aware (decoding works, not refused)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/scratch/users/prasann/hyb_sft/plugin_src")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    patch(a.src, a.out)
