"""Debug helper for locating the parameter behind a non-finite gradient norm."""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.distributed as dist

from olmo_core.utils import rank_matches_filter

log = logging.getLogger(__name__)


def _format_entry(e: Dict[str, Any]) -> str:
    return (
        f"norm={e['local_norm']:.6g} max_abs={e['max_abs']:.6g} "
        f"norm_finite={e['norm_finite']} max_abs_finite={e['max_abs_finite']} "
        f"pg={e['param_group']} placements={e['placements']} dtype={e['dtype']} "
        f"shape={e['shape']} name={e['name']}"
    )


def debug_nan_inf_grad_norm(
    total_grad_norm_local: torch.Tensor,
    *,
    step: int,
    component_norms: Callable[[], Mapping[str, torch.Tensor]],
    iter_local_grads: Callable[[], Iterable[Tuple[str, str, str, torch.Tensor]]],
    ranks_filter: str = "all",
    max_log_entries: int = 20,
    dump_dir: Optional[str] = None,
    rank: Optional[int] = None,
) -> None:
    """
    When ``total_grad_norm_local`` is non-finite, log the per-parameter local grad norms and the
    per-component local norms to help locate the offending parameter, and -- if ``dump_dir`` is set
    -- ``torch.save`` the same data there. A no-op when the norm is finite, so the
    ``component_norms``/``iter_local_grads`` callables are only invoked once we've decided to report.

    :param total_grad_norm_local: The (already node-local) total grad norm.
    :param step: Training step to label the report with.
    :param component_norms: Returns a mapping of component name -> local norm tensor.
    :param iter_local_grads: Returns an iterable of ``(name, param_group, placements, local_grad)``.
    :param ranks_filter: Which ranks report (see :func:`olmo_core.utils.rank_matches_filter`).
    :param max_log_entries: How many entries to include in the "top local norms" context list.
    :param dump_dir: If set, also ``torch.save`` the report under this directory.
    :param rank: Current rank; defaults to the distributed rank (or 0).
    """
    if rank is None:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if not rank_matches_filter(ranks_filter, rank):
        return
    if bool(torch.isfinite(total_grad_norm_local).all().item()):
        return

    max_log_entries = max(max_log_entries, 0)
    component_values = {
        name: float(norm.detach().float().item()) for name, norm in component_norms().items()
    }

    bad_entries: List[Dict[str, Any]] = []
    top_entries: List[Tuple[float, Dict[str, Any]]] = []
    for name, param_group, placements, local_grad in iter_local_grads():
        local_grad_float = local_grad.float()
        local_norm = float(torch.linalg.vector_norm(local_grad_float, ord=2).item())
        max_abs = float(torch.linalg.vector_norm(local_grad_float, ord=float("inf")).item())
        norm_finite = math.isfinite(local_norm)
        max_abs_finite = math.isfinite(max_abs)
        entry: Dict[str, Any] = {
            "name": name,
            "param_group": param_group,
            "placements": placements,
            "dtype": str(local_grad.dtype),
            "shape": tuple(local_grad.shape),
            "local_norm": local_norm,
            "max_abs": max_abs,
            "norm_finite": norm_finite,
            "max_abs_finite": max_abs_finite,
        }
        if not norm_finite or not max_abs_finite:
            bad_entries.append(entry)
        top_entries.append((local_norm, entry))

    top_entries.sort(
        reverse=True, key=lambda item: item[0] if math.isfinite(item[0]) else float("inf")
    )
    top_for_dump = [e for _, e in top_entries]
    top_for_log = top_for_dump[:max_log_entries]

    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        torch.save(
            {
                "kind": "optim_nonfinite_grad_norm",
                "rank": rank,
                "step": step,
                "total_grad_norm": total_grad_norm_local.float().cpu(),
                "components": component_values,
                "bad_entries": bad_entries,
                "top_entries": top_for_dump,
            },
            os.path.join(dump_dir, f"rank{rank:03d}_step{step:06d}_optim_nonfinite_grad_norm.pt"),
        )

    # NOTE: `max_log_entries` caps BOTH lists, including the non-finite ones (bad_entries[:max_log_entries]) -- so with
    # more than `max_log_entries` non-finite params the culprit list is truncated. Kept intentionally.
    bad_lines = "\n".join(f"  BAD {_format_entry(e)}" for e in bad_entries[:max_log_entries])
    top_lines = "\n".join(f"  {i + 1:02d}. {_format_entry(e)}" for i, e in enumerate(top_for_log))
    log.error(
        "Non-finite grad norm diagnostic on rank %s step %s: total=%s components=%s "
        "bad_entries=%s/%s\n%s%s%s",
        rank,
        step,
        total_grad_norm_local.float().cpu(),
        component_values,
        len(bad_entries),
        len(top_for_dump),
        bad_lines,
        "\nTop local grad norms:\n" if top_lines else "",
        top_lines,
    )
