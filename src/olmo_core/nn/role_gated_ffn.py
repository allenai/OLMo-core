"""
Role-gated FFN ("flexible-compute FFN") -- context tokens get small/zero MLP compute.

At long context, the FFN is the dominant FLOP term once attention is compressed (~2/3 of
non-attention FLOPs at 4B). This gate routes tokens by ROLE: context-document tokens (the
overwhelming majority of a corpus-reasoning prompt) skip the full FFN entirely -- their residual
stream passes through unchanged -- while free/query/answer tokens (and generated tokens at
decode time, which are always FREE) keep the full FFN. Because the gate is a deterministic
function of the token stream (the same ``<|doc_start|>``/``<|doc_end|>`` markers used by
document-chunked attention), training and inference apply identical routing with no learned
router and no load-balancing machinery -- an AdaMoE-style learned null-expert router is the
natural next rung if role-based routing preserves accuracy.

Implementation: the existing ``feed_forward`` modules are kept in place (state-dict keys
unchanged, so base checkpoints load untouched); their bound ``forward`` is shadowed with a gated
version that runs the original computation on the GATHERED subset of full-FFN tokens -- a real
dense-matmul saving, not a masked multiply.

Enabled via :meth:`~olmo_core.nn.transformer.model.Transformer.enable_role_gated_ffn`.

FLOP accounting: with a fraction ``f`` of tokens gated off from layer ``start_layer`` on, FFN
FLOPs shrink by ``~f * (n_layers - start_layer) / n_layers``; a 32k contradiction row is ~90%
context tokens, so gating from layer 4 of 36 cuts FFN FLOPs ~5x (and composes with soft-token
compaction at train time, where kept-doc tokens are still context tokens).
"""

import logging
from types import MethodType
from typing import List, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

__all__ = ["RoleGateHolder", "install_role_gated_ffn"]


class RoleGateHolder:
    """Per-forward gate state shared by every gated FFN in a model.

    ``full_mask`` is ``(B, T)`` bool -- ``True`` = run the full FFN for this token. ``None``
    disables gating (everything full); a decode step whose sequence shape no longer matches the
    stored mask also runs full (a generated token is FREE).
    """

    def __init__(self):
        self.full_mask: Optional[torch.Tensor] = None

    def set_from_chunk_ids(self, chunk_ids: torch.Tensor) -> None:
        """Context-doc tokens (``chunk_id >= 0``) are gated OFF; FREE/PAD keep the full FFN.
        (PAD tokens carry no loss and are never attended, so leaving them 'full' only costs the
        few pad positions and keeps the mask logic trivial.)"""
        self.full_mask = chunk_ids < 0

    def clear(self) -> None:
        self.full_mask = None


def _gated_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    holder: RoleGateHolder = self._role_gate_holder  # type: ignore[attr-defined]
    mask = holder.full_mask
    if x.dim() != 3:
        return self._role_gate_orig_forward(x)  # type: ignore[attr-defined]
    B, T = x.shape[:2]
    if mask is None or mask.shape != (B, T) or bool(mask.all()):
        return self._role_gate_orig_forward(x)  # type: ignore[attr-defined]
    idx = mask.reshape(-1).nonzero(as_tuple=True)[0]
    flat = x.reshape(B * T, -1)
    full = self._role_gate_orig_forward(flat[idx].unsqueeze(0)).squeeze(0)  # type: ignore[attr-defined]
    out = torch.zeros_like(flat)
    out = out.index_copy(0, idx, full.to(out.dtype))
    return out.reshape(B, T, -1)


def install_role_gated_ffn(
    blocks: nn.ModuleDict, holder: RoleGateHolder, *, start_layer: int
) -> List[str]:
    """
    Shadow ``feed_forward.forward`` with the role-gated version on every block whose index is
    ``>= start_layer``. Module tree and state-dict keys are untouched.

    :param blocks: The model's block dict (keys are layer indices as strings).
    :param holder: Shared gate state (set per forward by the model).
    :param start_layer: First layer to gate (earlier layers keep the full FFN everywhere --
        early layers build token identity that attention later redistributes).

    :returns: The block keys gated.
    """
    gated = []
    for key, block in blocks.items():
        if int(key) < start_layer:
            continue
        ff = getattr(block, "feed_forward", None)
        if ff is None or hasattr(ff, "_role_gate_orig_forward"):
            continue
        ff._role_gate_holder = holder
        ff._role_gate_orig_forward = ff.forward
        ff.forward = MethodType(_gated_forward, ff)
        gated.append(key)
    return gated
