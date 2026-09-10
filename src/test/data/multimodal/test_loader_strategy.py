"""Tests for the single loader-strategy field on :class:`MixtureDataLoader`.

``prefetch_workers`` (threads) and ``dl_num_workers`` (processes) are mutually exclusive
prefetch mechanisms. The legality rules used to live as three separate checks inside the
constructor — two raises and one silent zeroing — which is what turned a non-zero
``DL_NUM_WORKERS`` default into an unconditional startup error for
``--pack_sequences=false``. They now resolve in one place.
"""

from __future__ import annotations

import pytest

from olmo_core.data.multimodal.mixture_data_loader import (
    MixtureLoaderStrategy,
    resolve_loader_strategy,
)
from olmo_core.exceptions import OLMoConfigurationError


@pytest.mark.parametrize(
    ("pack", "pack_max_crops", "prefetch_workers", "dl_num_workers", "expected"),
    [
        (True, 25, 0, 0, MixtureLoaderStrategy.synchronous),
        (False, None, 0, 0, MixtureLoaderStrategy.synchronous),
        (True, 25, 4, 0, MixtureLoaderStrategy.threads),
        (False, None, 4, 0, MixtureLoaderStrategy.threads),
        (True, 25, 0, 2, MixtureLoaderStrategy.processes),
        # Both requested: processes win, since they also do the packing.
        (True, 25, 4, 2, MixtureLoaderStrategy.processes),
    ],
)
def test_resolve_loader_strategy(pack, pack_max_crops, prefetch_workers, dl_num_workers, expected):
    strategy, threads, procs = resolve_loader_strategy(
        pack=pack,
        pack_max_crops=pack_max_crops,
        prefetch_workers=prefetch_workers,
        dl_num_workers=dl_num_workers,
    )
    assert strategy is expected
    # Exactly one mechanism is left active.
    assert not (threads > 0 and procs > 0)
    if expected is MixtureLoaderStrategy.synchronous:
        assert threads == 0 and procs == 0
    elif expected is MixtureLoaderStrategy.threads:
        assert threads > 0 and procs == 0
    else:
        assert procs > 0 and threads == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"pack": False, "pack_max_crops": None, "prefetch_workers": 0, "dl_num_workers": 2},
            "pack=True",
        ),
        (
            {"pack": True, "pack_max_crops": None, "prefetch_workers": 0, "dl_num_workers": 2},
            "pack_max_crops",
        ),
        ({"pack": True, "pack_max_crops": 25, "prefetch_workers": -1, "dl_num_workers": 0}, ">= 0"),
        ({"pack": True, "pack_max_crops": 25, "prefetch_workers": 0, "dl_num_workers": -1}, ">= 0"),
    ],
)
def test_resolve_loader_strategy_rejects_illegal_combinations(kwargs, match):
    with pytest.raises(OLMoConfigurationError, match=match):
        resolve_loader_strategy(**kwargs)


def test_error_message_names_the_escape_hatch():
    """The `--pack_sequences=false` trap: the message must say how to get out of it."""
    with pytest.raises(OLMoConfigurationError) as exc:
        resolve_loader_strategy(
            pack=False, pack_max_crops=None, prefetch_workers=0, dl_num_workers=2
        )
    assert "--dl_num_workers=0" in str(exc.value)
