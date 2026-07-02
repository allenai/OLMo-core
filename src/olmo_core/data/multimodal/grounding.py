"""Point/grounding formatting for Molmo2 pointing & counting data.

Dependency-free port of the single-image path of ``mm_olmo``'s
``GroundingPreprocessor`` (``olmo/models/molmo2/grounding_formatter.py``, the
``html-v2`` format used by stage-1) plus the pointing / counting answer assembly
(``get_point_string`` in ``olmo/data/data_formatter.py``).

A set of points for one image is rendered as::

    <points coords="1 1 XXX YYY 2 XXX YYY ...">label</points>

where the leading ``1`` is the (single) image index, then ``PTID X Y`` per point,
with X/Y scaled to 0-1000 (3-digit, zero-padded) and points sorted by (x, y).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "format_points_tag",
    "pointing_answer",
    "POINTING_PROMPTS",
    "POINT_COUNT_PROMPTS",
]


def _scale_point(x: float, y: float) -> Tuple[int, int]:
    """Clamp to [0,1] then scale to 0-1000 integers (matches ``_scale_point``)."""
    x = min(max(float(x), 0.0), 1.0)
    y = min(max(float(y), 0.0), 1.0)
    return round(1000 * x), round(1000 * y)


def format_points_tag(points_norm: Sequence[Sequence[float]], label: str) -> str:
    """Render normalized (0-1) points for a single image as an html-v2 ``<points>`` tag.

    :param points_norm: ``(N, 2)`` array-like of already-normalized ``(x, y)`` in [0, 1].
    :param label: The object label / text inside the tag.

    :returns: ``<points coords="1 1 XXX YYY ...">label</points>``; empty string if no points.
    """
    pts = [_scale_point(x, y) for x, y in points_norm]
    if not pts:
        return ""
    # Sort by (x, y) after rounding (matches build_single_image_coordinates).
    pts.sort()
    body = " ".join(f"{i} {x:03d} {y:03d}" for i, (x, y) in enumerate(pts, start=1))
    coord_str = f"1 {body}"  # leading "1" is the single image index
    return f'<points coords="{coord_str}">{label}</points>'


def pointing_answer(
    points_norm: Sequence[Sequence[float]],
    label: str,
    style: str,
    count: Optional[int] = None,
) -> str:
    """Assemble the assistant answer text (matches ``get_point_string``).

    :param points_norm: normalized ``(N, 2)`` points.
    :param label: object label.
    :param style: ``"pointing"``/``"point"``/``"cosyn_point"`` (just the points tag) or
        ``"point_count"`` (``Counting the <points…> shows a total of N.``).
    :param count: number of points; defaults to ``len(points_norm)``.
    """
    n = len(points_norm) if count is None else count
    if n == 0:
        return "There are none."
    tag = format_points_tag(points_norm, label)
    if style in ("point_count", "point_then_count"):
        return f"Counting the {tag} shows a total of {n}."
    if style in ("count_then_point", "count_point"):
        return f"There are {n} {tag}."
    if style == "count":
        return str(n)
    # "pointing" / "point" / "cosyn_point" / None
    return tag


def normalize_points(
    xy: np.ndarray, point_scale: Optional[float], image_size: Optional[Tuple[int, int]]
) -> np.ndarray:
    """Normalize raw ``(N, 2)`` points to [0, 1] (matches ``normalize_coordinates``).

    :param xy: raw ``(N, 2)`` points.
    :param point_scale: if set, divide by this scalar (e.g. 100 for 0-100 percent coords).
    :param image_size: ``(w, h)`` to divide by when ``point_scale`` is None (pixel coords).
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.size == 0:
        return xy.reshape(0, 2)
    if point_scale is not None:
        return xy / float(point_scale)
    assert image_size is not None, "image_size required when point_scale is None (pixel coords)"
    w, h = image_size
    out = xy.copy()
    out[:, 0] /= float(w)
    out[:, 1] /= float(h)
    return out


# Prompt pools (verbatim subset of mm_olmo GENERAL_PROMPTS_V1; one is sampled per branch,
# `{label}` filled with the object label). data_formatter.py:257 / :302.
POINTING_PROMPTS: Tuple[str, ...] = (
    "Point to {label}\nPlease say 'There are none.' if it is not in the image.",
    'Point to all occurrences of "{label}"',
    "Point to any {label} in the image.",
    "Point: Where are the {label}",
    "Show me where the {label} are",
    "Can you show me where the {label} are?",
    "Where are the {label}?",
    "Generate a list of points showing where the {label} are.",
    'Find the "{label}".',
    "Locate all {label}.",
    "Locate a {label}.",
)
POINT_COUNT_PROMPTS: Tuple[str, ...] = (
    "How many {label} are there?",
    "How many {label}?",
    'How many "{label}" are there in the image?',
    "Tell me how many {label} there are and point to them.",
    "Count the {label}.",
    "How many {label} do you see?",
    "count {label}",
)
