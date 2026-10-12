"""
Where gold sits after the shuffle, and what number to call it.

An off-by-one here is not a crash, it is a score: a model that answers correctly is graded wrong,
uniformly, and the result reads as a modelling finding. The pre-migration tree had ~24 hand-written
copies of this arithmetic in three families plus five one-offs, and mixed both bases inside the
suite -- ``grouping_labeled`` emits 0-based clusters while the structurally identical ``cycle`` and
``groups4`` emit 1-based ones, safe only because the grader routed them to different scorers.

So: **``base`` is a required keyword argument with no default.** A default is what lets a new
generator silently inherit the wrong convention. Each task's declared
:attr:`~ctc.format.registry.TaskSpec.gold_index_base` is the authority and :func:`check_indices` is
how a generator proves it agreed.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple, TypeVar

__all__ = ["shuffle_with_remap", "remap", "remap_groups", "check_indices"]

T = TypeVar("T")


def shuffle_with_remap(items: Sequence[T], *, rng, base: int) -> Tuple[List[T], Dict[int, int]]:
    """
    Shuffle items into display order and return the construction-position -> display-index map.

    For tasks whose gold is a *structure over positions* (pairs, cycles, groups): build the items in
    a convenient order, shuffle here, then push the gold structure through :func:`remap_groups`.

    :param items: Items in construction order.
    :param rng: Seeded :class:`random.Random`. Consumes exactly one ``shuffle`` over a list of
        integer positions -- the pre-migration idiom, kept identical so golden fixtures hold.
    :param base: ``0`` or ``1``. Baked into the returned map, so callers never add one themselves.

    :returns: ``(shuffled_items, old_to_new)``.

    :raises ValueError: If ``base`` is not 0 or 1.
    """
    if base not in (0, 1):
        raise ValueError(f"gold_index_base must be 0 or 1, got {base!r}")
    order = list(range(len(items)))
    rng.shuffle(order)
    return [items[old] for old in order], {old: new + base for new, old in enumerate(order)}


def remap(indices: Iterable[int], old_to_new: Dict[int, int]) -> List[int]:
    """
    :param indices: Construction-order positions.
    :param old_to_new: From :func:`shuffle_with_remap`; carries the base.

    :returns: The new indices, ascending.

    :raises KeyError: If an index was not in the shuffled list -- a real bug, so not silently
        dropped.
    """
    return sorted(old_to_new[i] for i in indices)


def remap_groups(groups: Iterable[Sequence[int]], old_to_new: Dict[int, int]) -> List[List[int]]:
    """
    Push groups (pairs, cycles, clusters) through a shuffle.

    Sorted at both levels so two generators that found the same answer emit the same bytes.

    :param groups: Groups of construction-order positions.
    :param old_to_new: From :func:`shuffle_with_remap`.

    :returns: The remapped groups, each sorted, the outer list sorted.
    """
    return sorted(remap(group, old_to_new) for group in groups)


def check_indices(indices: Iterable, n_items: int, *, base: int, field: str = "gold") -> None:
    """
    Assert emitted indices are in range for the declared base. Accepts flat lists or groups.

    Cheap, and it catches the failure this module exists for: base-0 indices read as base-1 give a
    plausible file that scores near zero.

    :param indices: Flat indices or groups of indices.
    :param n_items: Documents in the example.
    :param base: The task's declared ``gold_index_base``.
    :param field: Field name, for the message.

    :raises ValueError: If ``base`` is invalid or an index falls outside
        ``[base, n_items - 1 + base]``.
    """
    if base not in (0, 1):
        raise ValueError(f"gold_index_base must be 0 or 1, got {base!r}")
    flat: List[int] = []
    for entry in indices:
        flat.extend(entry) if isinstance(entry, (list, tuple)) else flat.append(entry)
    lo, hi = base, n_items - 1 + base
    bad = sorted({i for i in flat if not lo <= i <= hi})
    if bad:
        raise ValueError(
            f"{field} indices {bad[:8]} fall outside [{lo}, {hi}] for {n_items} items at "
            f"base={base}: either the generator used the other base, or gold refers to a "
            f"document that was dropped."
        )
