"""
Primitives shared by the synthetic generators.

Small, but worth having: :func:`build_expression` existed twice in the pre-migration tree as
independent rewrites (``generate_mathmatch_data.py`` and ``generate_groups4_data.py``) that produced
the same output and consumed the RNG identically -- groups4 drew ``randint(4, 5)`` terms while
mathmatch drew ``randint(3, 4)`` operators and added one, which is the same draw written twice.
Two copies of a function whose RNG behaviour must not drift is exactly the shape of thing that
drifts.
"""

from __future__ import annotations

import random

__all__ = ["build_expression", "auto_value_range"]


def auto_value_range(num_docs: int, tolerance: int, *, per_window: int = 1) -> int:
    """
    A value range wide enough to hold ``num_docs`` distinct answers at the required spacing.

    The numeric tasks need every non-gold value more than ``tolerance`` from its neighbours, so N
    values need a span of at least ``N * (tolerance + 1) / per_window``. Fixed defaults cannot
    satisfy that at ladder scale, and the pre-migration defaults did not: mathmatch's ``[-50, 50]``
    holds 101 integers, which at ``tolerance=2`` fits about 34 mutually-distant values -- so the
    build command recorded for it in ``BUILD_MATRIX.md`` was infeasible at **every** CTC-suite rung
    (48 through 900 documents), and groups4's ``[-500, 500]`` at the top three. Neither had been run.

    The 4x factor is slack for rejection sampling: the generators draw from a shuffled pool and
    keep what fits, so a span equal to the theoretical minimum would almost never pack successfully.

    :param num_docs: Corpus size N.
    :param tolerance: Closeness threshold X.
    :param per_window: How many values may share one X-wide window. 1 for mathmatch, where every
        non-gold value is isolated; ``group_size - 1`` for groups4, whose distractors may cluster.

    :returns: The half-range; callers use ``[-r, r]``.
    """
    needed = 4 * num_docs * (tolerance + 1) // max(1, per_window)
    return max(50, needed // 2)


def build_expression(
    target: int,
    rng: random.Random,
    *,
    n_ops_min: int = 3,
    n_ops_max: int = 4,
    term_min: int = 1,
    term_max: int = 99,
) -> str:
    """
    A short signed sum evaluating exactly to ``target``, e.g. ``"47 + 12 - 8 + 3"``.

    All but the last term are free; the last is whatever makes the sum land on the target, so the
    value is exact rather than approximate.

    :param target: The value the expression must evaluate to.
    :param rng: Seeded RNG.
    :param n_ops_min: Fewest operators.
    :param n_ops_max: Most operators.
    :param term_min: Smallest free term.
    :param term_max: Largest free term.

    :returns: The rendered expression.
    """
    n_terms = rng.randint(n_ops_min, n_ops_max) + 1
    terms = [rng.randint(term_min, term_max) for _ in range(n_terms - 1)]
    signs = [rng.choice([1, -1]) for _ in range(n_terms - 1)]
    last = target - sum(s * t for s, t in zip(signs, terms))
    if last == 0:  # avoid a bare +0 term
        terms[0] += 1
        last = target - sum(s * t for s, t in zip(signs, terms))
    terms.append(abs(last))
    signs.append(1 if last >= 0 else -1)

    parts = [str(signs[0] * terms[0])]
    for s, t in zip(signs[1:], terms[1:]):
        parts += ["+" if s == 1 else "-", str(t)]
    return " ".join(parts)
